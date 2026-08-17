# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone integration tests for the DeepSeek V3.2 MX MLP kernel.

Following shapes are covered:
  * (1024/8192, 7168, 512)  -- dense MLP layers: full weights fit -> hoisted, token-sharded.
  * (128/256/512, 7168, 2048) -- shared experts: weights streamed (tiled) + I-sharded.
"""

import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.utils.common_types import (
    ActFnType,
    ComputationMode,
    MLPGateUpWeightLayout,
    NormType,
    QuantizationType,
)
from nkilib_src.nkilib.experimental.deepseekv32_mlp import mlp_deepseek_mx
from nkilib_src.nkilib.experimental.deepseekv32_mlp.mlp_deepseek_mx_torch import (
    mlp_deepseek_mx_torch_ref,
)

from test.integration.nkilib.core.mlp.test_mlp_common import build_fused_norm_mlp
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Params the standalone kernel actually consumes (its slim signature).
_KERNEL_KEYS = (
    "hidden_tensor",
    "gate_proj_weights_tensor",
    "up_proj_weights_tensor",
    "down_proj_weights_tensor",
    "gate_w_scale",
    "up_w_scale",
    "down_w_scale",
    "activation_fn",
    "output_dtype",
)


def _deepseek_output_descriptor(kernel_input):
    """Output shape [B, S, H] bf16. H comes from the down weight's [128, I/512, H, 4] layout."""
    batch, seqlen = kernel_input["hidden_tensor"].shape[0], kernel_input["hidden_tensor"].shape[1]
    hidden = kernel_input["down_proj_weights_tensor"].shape[2]
    return {"out": np.zeros((batch, seqlen, hidden), dtype=nl.bfloat16)}


def _reduce_gate_up_scale_to_compact128(native, H, I):
    """Reduce a native block-32 gate/up scale [16, H/512, I/512, 4, 128] to compact block-128
    [H/128, I/128] by sampling one representative per 128x128 block.

    Sampling matches the kernel/ref expand: native[kk, ht, it, a, b] carries compact-K-block
    ``ht*4 + kk//4`` and compact-N-block ``it*4 + b//32``. So compact[K128, I128] samples
    kk=(K128%4)*4, ht=K128//4, it=I128//4, a=0, b=(I128%4)*32.
    """
    native = native.cpu().numpy() if hasattr(native, "cpu") else np.asarray(native)
    n_H128, n_I128 = H // 128, I // 128
    compact = np.empty((n_H128, n_I128), dtype=np.uint8)
    for K128 in range(n_H128):
        for I128 in range(n_I128):
            compact[K128, I128] = native[(K128 % 4) * 4, K128 // 4, I128 // 4, 0, (I128 % 4) * 32]
    return compact


def _reduce_down_scale_to_compact128(native, H, n_I512):
    """Reduce a native block-32 down scale [16, I/512, H] to compact block-128 [I/128, H/128].

    native[kk, it, h] carries compact-I-block ``it*4 + kk//4`` and compact-H-block ``h//128``.
    So compact[I128, H128] samples kk=(I128%4)*4, it=I128//4, h=H128*128.
    """
    native = native.cpu().numpy() if hasattr(native, "cpu") else np.asarray(native)
    n_I128, n_H128 = n_I512 * 4, H // 128
    compact = np.empty((n_I128, n_H128), dtype=np.uint8)
    for I128 in range(n_I128):
        for H128 in range(n_H128):
            compact[I128, H128] = native[(I128 % 4) * 4, I128 // 4, H128 * 128]
    return compact


class TestDeepSeekMlpMx:
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(
        "seqlen, hidden, intermediate, with_routed, scale_mode",
        [
            # ---- Native block-32 scales (compact_scales=False) ----
            # MLP + TP over I: Hoisted weights (plain MLP, no routed-expert add).
            (8192, 7168, 512, False, "native"),
            # MLP + SP: streamed (tiled) weights + intermediate-sharding (plain MLP).
            (128, 7168, 18432, False, "native"),
            # Shared Experts + SP: streamed (tiled) weights + intermediate-sharding.
            (128, 7168, 2048, False, "native"),
            pytest.param(128, 7168, 2048, True, "native", marks=pytest.mark.fast),
            (256, 7168, 2048, False, "native"),
            (256, 7168, 2048, True, "native"),
            (512, 7168, 2048, False, "native"),
            (512, 7168, 2048, True, "native"),
            # ---- Compact block-128 scales (compact_scales=True): expand+swizzle in-kernel. ----
            (128, 7168, 18432, False, "compact"),
            pytest.param(128, 7168, 2048, True, "compact", marks=pytest.mark.fast),
            (512, 7168, 2048, False, "compact"),
        ],
    )
    def test_deepseek_mlp_mx_block_scale_input(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        seqlen,
        hidden,
        intermediate,
        with_routed,
        scale_mode,
    ):
        compiler_args = CompilerArgs(logical_nc_config=2, platform_target=platform_target)
        compact_scales = scale_mode == "compact"

        # Reuse the shared MX helpers to build weights/scales + the packed block-scale hidden input.
        full_input = build_fused_norm_mlp(
            batch=1,
            seqlen=seqlen,
            hidden=hidden,
            intermediate=intermediate,
            dtype=nl.bfloat16,
            quantization_type=QuantizationType.MX,
            quant_dtype=nl.float8_e4m3fn,
            is_input_quantized=True,
            norm_type=NormType.NO_NORM,
            gate_up_w_layout=MLPGateUpWeightLayout.H_X4_INNERMOST,
            mode=ComputationMode.PREFILL,
            use_mx_block_scale_input=True,
        )

        # Keep only what the slim kernel / reference consume.
        kernel_input = {k: full_input[k] for k in _KERNEL_KEYS}
        kernel_input["output_dtype"] = nl.bfloat16
        kernel_input["activation_fn"] = ActFnType.SiLU
        # Compact block-128 path: reduce the native block-32 scales to one representative per
        # 128x128 block. Both kernel and golden re-expand the SAME compact scale, so the test is
        # self-consistent (both dequant with identical scales) and exercises the in-kernel
        # expand+swizzle. Down uses no column swizzle; gate/up carry the (128,4)->(4,128) swizzle.
        kernel_input["compact_scales"] = compact_scales
        if compact_scales:
            n_I512 = intermediate // 512
            kernel_input["gate_w_scale"] = _reduce_gate_up_scale_to_compact128(
                kernel_input["gate_w_scale"], hidden, intermediate
            )
            kernel_input["up_w_scale"] = _reduce_gate_up_scale_to_compact128(
                kernel_input["up_w_scale"], hidden, intermediate
            )
            kernel_input["down_w_scale"] = _reduce_down_scale_to_compact128(
                kernel_input["down_w_scale"], hidden, n_I512
            )
        # Shared-experts mode: pass a routed-expert output [B, S, H] that the kernel sums into its
        # own result; both kernel and golden receive the same tensor. None -> plain MLP path.
        # Provide it as a bf16 NUMPY array (like every other kernel input) so it becomes an
        # ap-able device tensor; torch_ref_wrapper converts numpy->torch for the golden.
        if with_routed:
            routed = np.random.default_rng(42).standard_normal((1, seqlen, hidden))
            kernel_input["routed_expert_output"] = dt.static_cast(routed, nl.bfloat16)
        else:
            kernel_input["routed_expert_output"] = None

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=mlp_deepseek_mx,
            torch_ref=torch_ref_wrapper(mlp_deepseek_mx_torch_ref),
            kernel_input_generator=lambda _: kernel_input,
            output_tensor_descriptor=_deepseek_output_descriptor,
            check_unused_params=True,
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            rtol=5e-2,
        )
