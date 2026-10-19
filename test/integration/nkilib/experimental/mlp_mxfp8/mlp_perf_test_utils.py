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

"""Shared inputs, references, and quantization variants for dense MXFP8 MLP tests."""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypedDict, cast

import ml_dtypes
import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.core.utils.kernel_helpers import _NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_config import MatmulMxfp8KernelConfig
from nkilib_src.nkilib.experimental.mlp_mxfp8.mlp_bwd_mxfp8 import mlp_backward_mxfp8_nki
from nkilib_src.nkilib.experimental.mlp_mxfp8.mlp_fwd_mxfp8 import mlp_forward_mxfp8_nki
from nkilib_src.nkilib.experimental.moe_mxfp8.bwd.config import MXFP8MOEBwdConfig
from nkilib_src.nkilib.experimental.moe_mxfp8.config_helpers import ExecutionMode
from nkilib_src.nkilib.experimental.moe_mxfp8.fwd.config import MXFP8MOEFwdConfig
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.common_dataclasses import (
    QuantScheme,
    SwizzleMode,
    TensorDescriptor,
    TensorOrientation,
)

from test.integration.nkilib.experimental.moe_mxfp8.mxfp8_moe_bwd_test_utils import prequantize_moe_weights
from test.integration.nkilib.experimental.moe_mxfp8.mxfp8_moe_fwd_test_utils import prequantize_moe_weights_fwd
from test.integration.nkilib.experimental.moe_mxfp8.mxfp8_moe_validated_common import make_moe_validated_comparator
from test.utils import common_dataclasses
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

MlpTestVariant = Literal[
    "wrapx_bf16_inputs_dma_transpose_enabled",
    "1x32_prequantized_weights_dma_transpose_enabled",
    "1x32_prequantized_weights_dma_transpose_disabled",
]


class _BackwardPhasePresets(TypedDict, total=False):
    phase1_config: MatmulMxfp8KernelConfig
    phase2_config: MatmulMxfp8KernelConfig
    phase3_config: MatmulMxfp8KernelConfig
    phase4_config: MatmulMxfp8KernelConfig


@dataclass(frozen=True)
class MlpShape:
    model: str
    tensor_parallel_degree: int
    hidden: int
    intermediate: int
    tokens: int = 4096


def _inputs(tokens, hidden, intermediate):
    np.random.seed(42)
    x = np.random.randn(tokens, hidden).astype(np.float32)
    gate_up = (np.random.randn(hidden, 2, intermediate) / np.sqrt(hidden)).astype(np.float32)
    down = (np.random.randn(intermediate, hidden) / np.sqrt(intermediate)).astype(np.float32)
    output_grad = np.random.randn(tokens, hidden).astype(np.float32)
    return (
        x.astype(ml_dtypes.bfloat16),
        gate_up.astype(ml_dtypes.bfloat16),
        down.astype(ml_dtypes.bfloat16),
        output_grad.astype(ml_dtypes.bfloat16),
    )


def _project(hidden_states, gate_up_proj_weight):
    x = hidden_states.float()
    gate_up = gate_up_proj_weight.float()
    gate = (x @ gate_up[:, 0, :]).to(torch.bfloat16).float()
    up = (x @ gate_up[:, 1, :]).to(torch.bfloat16).float()
    return gate, up


def _checkpoint(hidden_states, gate_up_proj_weight, block_size):
    gate, up = _project(hidden_states, gate_up_proj_weight)
    I = gate.shape[1]
    N = gate.shape[0] // block_size
    return torch.stack((gate.reshape(N, block_size, I), up.reshape(N, block_size, I)), dim=1)


def _fwd_ref(
    hidden_states,
    gate_up_proj_weight,
    down_proj_weight,
    config=None,
    gate_up_weight_scales=None,
    down_weight_scales=None,
):
    del config, gate_up_weight_scales, down_weight_scales
    gate, up = _project(hidden_states, gate_up_proj_weight)
    intermediate = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16).float()
    output = (intermediate @ down_proj_weight.float()).to(torch.bfloat16).float()
    checkpoint = _checkpoint(hidden_states, gate_up_proj_weight, hidden_states.shape[0] // 2)
    return {
        "output_hidden_states": output,
        "gate_up_proj_act_checkpoint_T": checkpoint,
    }


def _bwd_ref(
    hidden_states,
    gate_up_proj_weight,
    down_proj_weight,
    output_hidden_states_grad,
    gate_up_proj_act_checkpoint_T,
    config=None,
    gate_up_weight_scales=None,
    down_weight_scales=None,
):
    del config, gate_up_weight_scales, down_weight_scales
    x = hidden_states.float()
    gate_up = gate_up_proj_weight.float()
    down = down_proj_weight.float()
    dy = output_hidden_states_grad.float()
    checkpoint = gate_up_proj_act_checkpoint_T.float()
    gate = checkpoint[:, 0].reshape(x.shape[0], -1)
    up = checkpoint[:, 1].reshape(x.shape[0], -1)
    sigmoid = torch.sigmoid(gate)
    gate_act = gate * sigmoid
    silu_dx = sigmoid + gate * sigmoid * (1 - sigmoid)
    intermediate = (gate_act * up).to(torch.bfloat16).float()
    d_intermediate = (dy @ down.T).to(torch.bfloat16).float()
    d_gate = (d_intermediate * silu_dx * up).to(torch.bfloat16).float()
    d_up = (d_intermediate * gate_act).to(torch.bfloat16).float()
    hidden_grad = d_gate @ gate_up[:, 0, :].T + d_up @ gate_up[:, 1, :].T
    gate_up_grad = torch.stack((x.T @ d_gate, x.T @ d_up), dim=1)
    down_grad = intermediate.T @ dy
    return {
        "hidden_states_grad": hidden_grad,
        "gate_up_proj_weight_grad": gate_up_grad,
        "down_proj_weight_grad": down_grad,
    }


def _require_dma_mode(variant: MlpTestVariant) -> None:
    no_dma = variant == "1x32_prequantized_weights_dma_transpose_disabled"
    if _NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE != no_dma:
        setting = "true" if no_dma else "unset"
        pytest.skip(f"{variant} requires NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE={setting} before import")


def _forward_config(variant: MlpTestVariant) -> MXFP8MOEFwdConfig:
    wrapx = variant == "wrapx_bf16_inputs_dma_transpose_enabled"
    no_dma = variant == "1x32_prequantized_weights_dma_transpose_disabled"

    def weight_td():
        return TensorDescriptor(
            orientation=TensorOrientation.F_BY_K if wrapx else TensorOrientation.K_BY_F,
            quant_scheme=QuantScheme.WRAPX if wrapx else QuantScheme._1x32,
            swizzle_mode=SwizzleMode.DGT if wrapx else SwizzleMode.PE,
            disable_dma_transpose=no_dma,
        )

    return MXFP8MOEFwdConfig(
        execution_mode=ExecutionMode.MLP,
        dma_transpose_enabled=not no_dma,
        fast_dma_transpose=not no_dma,
        use_1x32_pe_swizzle=not wrapx,
        gate_up_weight_td=weight_td(),
        down_weight_td=weight_td(),
    )


def _backward_config(variant: MlpTestVariant) -> MXFP8MOEBwdConfig:
    wrapx = variant == "wrapx_bf16_inputs_dma_transpose_enabled"
    no_dma = variant == "1x32_prequantized_weights_dma_transpose_disabled"
    presets: _BackwardPhasePresets = {}
    if wrapx:
        # BF16 inputs have no weight-scale descriptor from which to infer WrapX.
        for name in ("phase1_config", "phase2_config", "phase3_config", "phase4_config"):
            presets[name] = MatmulMxfp8KernelConfig(
                M=0,
                K=0,
                N=0,
                quant_scheme=QuantScheme.WRAPX,
                spill_reload=True,
                enable_scale_packing=True,
            )
    config = MXFP8MOEBwdConfig(
        **presets,
        execution_mode=ExecutionMode.MLP,
        dma_transpose_enabled=not no_dma,
        fast_dma_transpose=not no_dma,
        pe_transpose_only=no_dma,
    )
    for name in (
        "output_grad_swizzle_mode",
        "down_weight_swizzle_mode",
        "d_gate_up_swizzle_mode",
        "gate_up_weight_swizzle_mode",
    ):
        setattr(config, name, SwizzleMode.DGT if wrapx else SwizzleMode.PE)
    for name in (
        "d_gate_up_t_swizzle_mode",
        "hidden_states_t_swizzle_mode",
        "output_grad_t_swizzle_mode",
        "scaled_intermediate_t_swizzle_mode",
    ):
        setattr(config, name, SwizzleMode.PE)
    return config


def _run(
    test_manager: Orchestrator,
    platform_target,
    shape: MlpShape,
    variant: MlpTestVariant,
    backward: bool,
) -> None:
    _require_dma_mode(variant)
    wrapx = variant == "wrapx_bf16_inputs_dma_transpose_enabled"
    no_dma = variant == "1x32_prequantized_weights_dma_transpose_disabled"
    assert shape.tokens > 0 and shape.tokens % 256 == 0
    block_size = shape.tokens // 2
    x, gate_up, down, dy = _inputs(shape.tokens, shape.hidden, shape.intermediate)
    config = _backward_config(variant) if backward else _forward_config(variant)
    inputs = {"hidden_states": x, "config": config}
    if backward:
        checkpoint = _checkpoint(
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(gate_up.astype(np.float32)),
            block_size,
        )
        inputs.update(
            output_hidden_states_grad=dy,
            gate_up_proj_act_checkpoint_T=checkpoint.numpy().astype(ml_dtypes.bfloat16),
        )
    if wrapx:
        inputs.update(
            gate_up_proj_weight=gate_up if backward else np.transpose(gate_up, (2, 1, 0)).copy(),
            down_proj_weight=down if backward else down.T.copy(),
        )
    else:
        if backward:
            packed_gate, gate_scales, packed_down, down_scales = prequantize_moe_weights(
                gate_up[None], down[None], quant_scheme=QuantScheme._1x32
            )
        else:
            packed_gate, gate_scales, packed_down, down_scales = prequantize_moe_weights_fwd(
                gate_up[None], down[None], use_1x32=True
            )
        inputs.update(
            gate_up_proj_weight=packed_gate[0],
            down_proj_weight=packed_down[0],
            gate_up_weight_scales=gate_scales[0],
            down_weight_scales=down_scales[0],
        )

    reference = cast(Callable[..., dict[str, torch.Tensor]], _bwd_ref if backward else _fwd_ref)

    @functools.wraps(reference)
    def ref(**kwargs):
        kwargs["gate_up_proj_weight"] = torch.from_numpy(gate_up.astype(np.float32))
        kwargs["down_proj_weight"] = torch.from_numpy(down.astype(np.float32))
        return reference(**kwargs)

    if backward:
        shapes = {
            "hidden_states_grad": ((shape.tokens, shape.hidden), nl.bfloat16),
            "gate_up_proj_weight_grad": ((shape.hidden, 2, shape.intermediate), nl.bfloat16),
            "down_proj_weight_grad": ((shape.intermediate, shape.hidden), nl.bfloat16),
        }
    else:
        shapes = {
            "output_hidden_states": ((shape.tokens, shape.hidden), nl.bfloat16),
            "gate_up_proj_act_checkpoint_T": (
                (2, 2, block_size, shape.intermediate),
                nl.bfloat16,
            ),
        }
    direction = "backward" if backward else "forward"
    test_manager.collector.set_kernel_params(
        {
            "model": shape.model,
            "tensor_parallel_degree": shape.tensor_parallel_degree,
            "kernel": "mlp_mxfp8",
            "direction": direction,
            "test_variant": variant,
            "tokens": shape.tokens,
            "hidden": shape.hidden,
            "intermediate": shape.intermediate,
            "block_size": block_size,
            "num_blocks": 2,
            "checkpoint_block_size": block_size,
            "quant_scheme": "wrapX" if wrapx else "1x32",
            "weight_input": "bf16" if wrapx else "prequantized",
            "disable_dma_transpose": no_dma,
            "operand_loader": "dma_transpose" if wrapx else "pe",
        }
    )
    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=mlp_backward_mxfp8_nki if backward else mlp_forward_mxfp8_nki,
        torch_ref=torch_ref_wrapper(ref),
        kernel_input_generator=lambda _: inputs,
        output_tensor_descriptor=lambda _: {
            name: np.zeros(output_shape, dtype=dtype) for name, (output_shape, dtype) in shapes.items()
        },
    )
    framework.run_test(
        test_config=None,
        compiler_args=common_dataclasses.CompilerArgs(logical_nc_config=2, platform_target=platform_target),
        custom_comparator=make_moe_validated_comparator(shapes),
    )


def run_forward(test_manager: Orchestrator, platform_target, shape: MlpShape, variant: MlpTestVariant) -> None:
    _run(test_manager, platform_target, shape, variant, backward=False)


def run_backward(test_manager: Orchestrator, platform_target, shape: MlpShape, variant: MlpTestVariant) -> None:
    _run(test_manager, platform_target, shape, variant, backward=True)
