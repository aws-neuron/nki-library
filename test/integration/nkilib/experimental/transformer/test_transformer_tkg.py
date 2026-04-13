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
"""Integration tests for transformer_tkg kernel."""

import math
import os

os.environ["NKI_FRONTEND"] = "beta2"

from test.utils.common_dataclasses import (
    CompilerArgs,
    InferenceArgs,
    KernelArgs,
    LazyGoldenGenerator,
    ValidationArgs,
)
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.ranged_test_harness import (
    DimensionRangeConfig,
    RangeManualGeneratorStrategy,
    RangeTestCase,
    RangeTestConfig,
    TensorConfig,
    TensorRangeConfig,
    range_test_config,
)
from test.utils.test_orchestrator import Orchestrator
from typing import List, Optional, final

import ml_dtypes
import neuron_dtypes as dt
import nki
import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.experimental.transformer.transformer_tkg import transformer_tkg
from nkilib_src.nkilib.experimental.transformer.transformer_tkg_torch import llama3_transformer_fwd_tkg_torch

# Dimension names
TP_DIM = "tp"
BATCH_DIM = "batch"
S_TKG_DIM = "S_tkg"
S_CTX_DIM = "S_ctx"
Q_HEADS_DIM = "q_heads"
D_HEAD_DIM = "d_head"
H_DIM = "H"
I_DIM = "I"
LNC_DIM = "lnc"
REL_TOL_DIM = "rel_tol"


def generate_llama3_transformer_tkg_combinations(
    tp_values: list[int] = [8, 16, 32],
    global_batch_values: list[int] = [8, 16, 32, 64, 128],
    S_tkg: int = 1,
    S_ctx_values: list[int] = [1024, 10240, 36864],
    q_heads: int = 64,
    d_head: int = 128,
    H: int = 8192,
    I: int = 28672,
    lnc: int = 2,
    rel_diff_tolerance: int = 2.5,
    enable_separation_pass_values: list[bool] = [False],
) -> list[list]:
    """Generate test combinations for transformer TKG kernel.

    Uses Llama 70B model dimensions.

    Returns list of [tp, batch, S_tkg, S_ctx, q_heads, d_head, H, I, lnc, rel_tol]
    """
    combinations = []

    for tp in tp_values:
        for global_batch in global_batch_values:
            dp_degree = 64 // tp
            batch = global_batch // dp_degree

            for S_ctx in S_ctx_values:
                for enable_separation_pass in enable_separation_pass_values:
                    combination = [
                        tp,
                        batch,
                        S_tkg,
                        S_ctx,
                        q_heads,
                        d_head,
                        H,
                        I,
                        lnc,
                        rel_diff_tolerance,
                    ]

                    # FIXME: This combination fails accuracy check, need fix NKILIB-797
                    if combination == [32, 64, 1, 1024, 64, 128, 8192, 28672, 2, 2.5]:
                        continue
                    if combination == [16, 32, 1, 1024, 64, 128, 8192, 28672, 2, 2.5]:
                        continue
                    if combination == [32, 32, 1, 1024, 64, 128, 8192, 28672, 2, 2.5]:
                        continue

                    combinations.append(combination)

    return combinations


# Tensor names
DIMS_TENSOR = "dims"

# ── Variance-preserving tensor generators ──────────────────────────────────
# bf16 has ~0.4% worst-case quantization error. With N(0,1) weights, QKV
# projections have std ≈ √H, making attention scores O(H). When two positions
# score nearly identically, 0.4% noise can flip the softmax argmax, picking
# wrong V rows. Using 1/√fan_in scaling keeps scores O(1) where softmax is
# stable. See commit 50ec39f for full analysis.
_rng = np.random.default_rng(42)


def _uniform_activation(shape, dtype):
    """Uniform[-1, 1]"""
    return np.ascontiguousarray(dt.static_cast(_rng.uniform(-1.0, 1.0, shape).astype(np.float32), dtype))


def _gaussian(shape, dtype, std):
    """N(0, std)"""
    return np.ascontiguousarray(dt.static_cast(_rng.normal(0.0, std, shape).astype(np.float32), dtype))


def _fan_in_projection(shape, dtype, fan_in):
    """N(0, 1/√fan_in). Keeps matmul output variance ≈ input variance."""
    return _gaussian(shape, dtype, std=1.0 / np.sqrt(fan_in))


def _near_unity(shape, dtype):
    """Uniform[0.5, 1.5]. RMSNorm gammas are ~1.0 in trained models."""
    return np.ascontiguousarray(dt.static_cast(_rng.uniform(0.5, 1.5, shape).astype(np.float32), dtype))


def transformer_tkg_tp4_4layer_wrapper(
    X: nl.ndarray,
    W_qkv_0: nl.ndarray,
    W_qkv_1: nl.ndarray,
    W_qkv_2: nl.ndarray,
    W_qkv_3: nl.ndarray,
    W_out_0: nl.ndarray,
    W_out_1: nl.ndarray,
    W_out_2: nl.ndarray,
    W_out_3: nl.ndarray,
    W_gate_0: nl.ndarray,
    W_gate_1: nl.ndarray,
    W_gate_2: nl.ndarray,
    W_gate_3: nl.ndarray,
    W_up_0: nl.ndarray,
    W_up_1: nl.ndarray,
    W_up_2: nl.ndarray,
    W_up_3: nl.ndarray,
    W_down_0: nl.ndarray,
    W_down_1: nl.ndarray,
    W_down_2: nl.ndarray,
    W_down_3: nl.ndarray,
    W_gamma_qkv_0: nl.ndarray,
    W_gamma_qkv_1: nl.ndarray,
    W_gamma_qkv_2: nl.ndarray,
    W_gamma_qkv_3: nl.ndarray,
    W_gamma_mlp_0: nl.ndarray,
    W_gamma_mlp_1: nl.ndarray,
    W_gamma_mlp_2: nl.ndarray,
    W_gamma_mlp_3: nl.ndarray,
    K_cache_0: nl.ndarray,
    K_cache_1: nl.ndarray,
    K_cache_2: nl.ndarray,
    K_cache_3: nl.ndarray,
    V_cache_0: nl.ndarray,
    V_cache_1: nl.ndarray,
    V_cache_2: nl.ndarray,
    V_cache_3: nl.ndarray,
    RoPE_cos: nl.ndarray,
    RoPE_sin: nl.ndarray,
    mask_cache: nl.ndarray,
    mask_active: nl.ndarray,
    position_ids: Optional[nl.ndarray],
    eps: float,
    sbuf_residual_and_cc: bool,
    W_gate_scale_0: nl.ndarray,
    W_gate_scale_1: nl.ndarray,
    W_gate_scale_2: nl.ndarray,
    W_gate_scale_3: nl.ndarray,
    W_up_scale_0: nl.ndarray,
    W_up_scale_1: nl.ndarray,
    W_up_scale_2: nl.ndarray,
    W_up_scale_3: nl.ndarray,
    W_down_scale_0: nl.ndarray,
    W_down_scale_1: nl.ndarray,
    W_down_scale_2: nl.ndarray,
    W_down_scale_3: nl.ndarray,
    replica_groups: List[List[int]],
):
    """Wrapper for TP=4, 4-layer transformer (quantized)."""
    return transformer_tkg(
        X=X,
        W_qkvs=[W_qkv_0, W_qkv_1, W_qkv_2, W_qkv_3],
        W_outs=[W_out_0, W_out_1, W_out_2, W_out_3],
        W_gates=[W_gate_0, W_gate_1, W_gate_2, W_gate_3],
        W_ups=[W_up_0, W_up_1, W_up_2, W_up_3],
        W_downs=[W_down_0, W_down_1, W_down_2, W_down_3],
        W_gamma_qkvs=[W_gamma_qkv_0, W_gamma_qkv_1, W_gamma_qkv_2, W_gamma_qkv_3],
        W_gamma_mlps=[W_gamma_mlp_0, W_gamma_mlp_1, W_gamma_mlp_2, W_gamma_mlp_3],
        K_caches=[K_cache_0, K_cache_1, K_cache_2, K_cache_3],
        V_caches=[V_cache_0, V_cache_1, V_cache_2, V_cache_3],
        RoPE_cos=RoPE_cos,
        RoPE_sin=RoPE_sin,
        mask_cache=mask_cache,
        mask_active=None,
        position_ids=position_ids,
        num_layers=4,
        eps=eps,
        replica_groups=replica_groups,
        sbuf_residual_and_cc=sbuf_residual_and_cc,
        W_gate_scales=[W_gate_scale_0, W_gate_scale_1, W_gate_scale_2, W_gate_scale_3],
        W_up_scales=[W_up_scale_0, W_up_scale_1, W_up_scale_2, W_up_scale_3],
        W_down_scales=[W_down_scale_0, W_down_scale_1, W_down_scale_2, W_down_scale_3],
    )


@pytest_test_metadata(
    name="Transformer TKG",
    pytest_marks=["transformer_tkg", "transformer"],
)
@final
class TestTransformerTKG:
    """Integration tests for transformer_tkg kernel."""

    def prepare_test_parametrized(
        self,
        test_manager: Orchestrator,
        test_options: RangeTestCase,
    ):
        """Prepare and execute a parametrized test case for transformer_tkg kernel."""
        dims = test_options.tensors[DIMS_TENSOR]
        tp = dims[TP_DIM]
        B = dims[BATCH_DIM]
        S_tkg = dims[S_TKG_DIM]
        S_ctx = dims[S_CTX_DIM]
        num_q_heads = dims[Q_HEADS_DIM]
        d_head = dims[D_HEAD_DIM]
        H = dims[H_DIM]
        I = dims[I_DIM]
        lnc = dims[LNC_DIM]
        rel_tol = dims[REL_TOL_DIM]

        num_kv_heads = 1
        num_layers = 4
        eps = 1e-6
        abs_tol = 1e-2
        dtype = nl.bfloat16

        # Compute per-core dimensions (sharded across TP)
        q_heads_per_core = num_q_heads // tp
        fd_per_core = I // tp
        # Round up to the nearest 128 - MLP only supports I that are multiples of 128
        fd_per_core = math.ceil(fd_per_core / 128) * 128

        qkv_dim = d_head * (q_heads_per_core + 2 * num_kv_heads)

        # Generate per-layer tensors with variance-preserving distributions
        W_qkvs = [_fan_in_projection((H, qkv_dim), dtype, fan_in=H) for i in range(num_layers)]
        W_outs = [_gaussian((q_heads_per_core * d_head, H), dtype, std=0.5) for i in range(num_layers)]
        W_gates = [_fan_in_projection((H, fd_per_core), dtype, fan_in=H) for i in range(num_layers)]
        W_ups = [_fan_in_projection((H, fd_per_core), dtype, fan_in=H) for i in range(num_layers)]
        W_downs = [_fan_in_projection((fd_per_core, H), dtype, fan_in=fd_per_core) for i in range(num_layers)]
        W_gamma_qkvs = [_near_unity((1, H), dtype) for i in range(num_layers)]
        W_gamma_mlps = [_near_unity((1, H), dtype) for i in range(num_layers)]
        K_caches = [_uniform_activation((B, num_kv_heads, d_head, S_ctx), dtype) for i in range(num_layers)]
        V_caches = [_uniform_activation((B, num_kv_heads, S_ctx, d_head), dtype) for i in range(num_layers)]

        # Generate cache lengths: each batch has its own assumed context length
        assumed_actual_ctx_lens = np.arange(B) * 3 + (S_ctx // 4 * 3)
        assert assumed_actual_ctx_lens.max() < S_ctx  # Make sure not to go out of bound

        # Generate cascaded attention mask using the ported function
        # NOTE: while the kernel can take the single mask_cache where unify_for_cascaded is True,
        # the golden expect explicit mask_cache and mask_active

        mask_cache_shape = (S_ctx, B, q_heads_per_core, S_tkg)

        arr = np.ones(mask_cache_shape, dtype=np.bool_)
        mask_cache = arr.reshape(B, q_heads_per_core, S_tkg, S_ctx)
        mask_active = np.tril(np.ones((B, q_heads_per_core, S_tkg, S_tkg), dtype=np.bool_), k=0)
        mask_cache[:, :, :, S_ctx - S_tkg :] = mask_active
        mask_cache = mask_cache.transpose((3, 0, 1, 2))

        position_ids = assumed_actual_ctx_lens[:, np.newaxis] + np.arange(S_tkg)

        # Generate position_ids based on cache lengths

        # Generate MLP scales: first and last layers are non-quantized (no scales)
        # Use fixed seed RNG matching old frontend strategy: fill with single random scalar
        nonquantized_layers = {0, num_layers - 1}
        scale_rng = np.random.default_rng(0)
        W_gate_scales = []
        W_up_scales = []
        W_down_scales = []
        for layer in range(num_layers):
            if layer in nonquantized_layers:
                W_gate_scales.append(None)
                W_up_scales.append(None)
                W_down_scales.append(None)
            else:
                W_gate_scales.append(np.full((128, fd_per_core), scale_rng.random(), dtype=np.float32))
                W_up_scales.append(np.full((128, fd_per_core), scale_rng.random(), dtype=np.float32))
                W_down_scales.append(np.full((128, H), scale_rng.random(), dtype=np.float32))

        # Build kernel_input with individual keys (required by test framework)
        kernel_input = {"X": _uniform_activation((B, S_tkg, H), dtype)}
        for i in range(num_layers):
            kernel_input[f"W_qkv_{i}"] = W_qkvs[i]
        for i in range(num_layers):
            kernel_input[f"W_out_{i}"] = W_outs[i]
        for i in range(num_layers):
            kernel_input[f"W_gate_{i}"] = W_gates[i]
        for i in range(num_layers):
            kernel_input[f"W_up_{i}"] = W_ups[i]
        for i in range(num_layers):
            kernel_input[f"W_down_{i}"] = W_downs[i]
        for i in range(num_layers):
            kernel_input[f"W_gamma_qkv_{i}"] = W_gamma_qkvs[i]
        for i in range(num_layers):
            kernel_input[f"W_gamma_mlp_{i}"] = W_gamma_mlps[i]
        for i in range(num_layers):
            kernel_input[f"K_cache_{i}"] = K_caches[i]
        for i in range(num_layers):
            kernel_input[f"V_cache_{i}"] = V_caches[i]
        kernel_input["RoPE_cos"] = _uniform_activation((d_head // 2, B, S_tkg), dtype)
        kernel_input["RoPE_sin"] = _uniform_activation((d_head // 2, B, S_tkg), dtype)

        kernel_input["mask_cache"] = mask_cache
        kernel_input["mask_active"] = mask_active

        kernel_input["position_ids"] = position_ids.astype(np.uint32)
        kernel_input["eps"] = eps
        kernel_input["sbuf_residual_and_cc"] = False

        collective_ranks = tp
        # Shared-fleet instances (trn2.3xlarge) have 4 NeuronCores; demote to compile-only
        # when the test needs more ranks than available.
        # Set TEST_TRANSFORMER_TKG_FORCE_INFER=1 to override and run on hardware anyway.
        _MAX_SHARED_FLEET_RANKS = 4
        if not os.environ.get("TEST_TRANSFORMER_TKG_FORCE_INFER") and tp > _MAX_SHARED_FLEET_RANKS:
            import warnings

            warnings.warn(
                f"Demoting to use replica group of size 4: tp={tp} exceeds shared-fleet limit "
                f"of {_MAX_SHARED_FLEET_RANKS} NeuronCores. "
                f"Set TEST_TRANSFORMER_TKG_FORCE_INFER=1 to override.",
                stacklevel=2,
            )

            collective_ranks = 4
        kernel_input["replica_groups"] = (tuple(range(collective_ranks)),)

        for i in range(num_layers):
            kernel_input[f"W_gate_scale_{i}"] = W_gate_scales[i]
            kernel_input[f"W_up_scale_{i}"] = W_up_scales[i]
            kernel_input[f"W_down_scale_{i}"] = W_down_scales[i]

        def to_torch(arr):
            if arr is None:
                return None
            if hasattr(arr, 'dtype') and arr.dtype == ml_dtypes.bfloat16:
                return torch.tensor(arr.astype(np.float32)).to(torch.bfloat16)
            return torch.from_numpy(arr)

        def create_lazy_golden():
            golden_output = llama3_transformer_fwd_tkg_torch(
                X=to_torch(kernel_input["X"]),
                W_qkvs=[to_torch(w) for w in W_qkvs],
                W_outs=[to_torch(w) for w in W_outs],
                W_gates=[to_torch(w) for w in W_gates],
                W_gate_scales=[to_torch(s) for s in W_gate_scales],
                W_ups=[to_torch(w) for w in W_ups],
                W_up_scales=[to_torch(s) for s in W_up_scales],
                W_downs=[to_torch(w) for w in W_downs],
                W_down_scales=[to_torch(s) for s in W_down_scales],
                W_gamma_qkvs=[to_torch(w) for w in W_gamma_qkvs],
                W_gamma_mlps=[to_torch(w) for w in W_gamma_mlps],
                RoPE_cos=to_torch(kernel_input["RoPE_cos"]),
                RoPE_sin=to_torch(kernel_input["RoPE_sin"]),
                mask_cache=to_torch(mask_cache),
                mask_active=to_torch(mask_active),
                position_ids=to_torch(position_ids),
                K_caches=[to_torch(k) for k in K_caches],
                V_caches=[to_torch(v) for v in V_caches],
                num_layers=num_layers,
                replica_groups=kernel_input["replica_groups"],
                eps=eps,
                mlp_down_proj_layout_enabled=False,
            )
            golden_np = golden_output.to(torch.float32).numpy().astype(ml_dtypes.bfloat16)
            return {"layer_output": golden_np}

        output_tensors = {"layer_output": np.zeros((B, S_tkg, H), dtype=ml_dtypes.bfloat16)}

        test_manager.execute(
            KernelArgs(
                kernel_func=nki.jit(transformer_tkg_tp4_4layer_wrapper),
                compiler_input=CompilerArgs(logical_nc_config=lnc),
                inference_args=InferenceArgs(collective_ranks=collective_ranks),
                kernel_input=kernel_input,
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(
                        lazy_golden_generator=create_lazy_golden,
                        output_ndarray=output_tensors,
                    ),
                    relative_accuracy=rel_tol / 100.0,
                    absolute_accuracy=abs_tol,
                ),
            )
        )

    @staticmethod
    def manual_test_config(test_grid) -> RangeTestConfig:
        """Generate manual test configuration from test grid."""
        test_cases = []
        for test_case in test_grid:
            test_cases.append(
                {
                    DIMS_TENSOR: {
                        TP_DIM: test_case[0],
                        BATCH_DIM: test_case[1],
                        S_TKG_DIM: test_case[2],
                        S_CTX_DIM: test_case[3],
                        Q_HEADS_DIM: test_case[4],
                        D_HEAD_DIM: test_case[5],
                        H_DIM: test_case[6],
                        I_DIM: test_case[7],
                        LNC_DIM: test_case[8],
                        REL_TOL_DIM: test_case[9],
                    },
                }
            )
        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=TensorRangeConfig(
                tensor_configs={
                    DIMS_TENSOR: TensorConfig(
                        [
                            DimensionRangeConfig(name=TP_DIM),
                            DimensionRangeConfig(name=BATCH_DIM),
                            DimensionRangeConfig(name=S_TKG_DIM),
                            DimensionRangeConfig(name=S_CTX_DIM),
                            DimensionRangeConfig(name=Q_HEADS_DIM),
                            DimensionRangeConfig(name=D_HEAD_DIM),
                            DimensionRangeConfig(name=H_DIM),
                            DimensionRangeConfig(name=I_DIM),
                            DimensionRangeConfig(name=LNC_DIM),
                            DimensionRangeConfig(name=REL_TOL_DIM),
                        ]
                    ),
                },
                monotonic_step_size=1,
                custom_generators=[RangeManualGeneratorStrategy(test_cases=test_cases)],
            ),
        )

    # Test vectors generated using generate_llama3_transformer_tkg_combinations()
    # Uses Llama 70B model dimensions (H=8192, I=28672, q_heads=64)
    # Format: [tp, batch, S_tkg, S_ctx, q_heads, d_head, H, I, lnc, rel_tol]
    TEST_GRID = generate_llama3_transformer_tkg_combinations()

    @pytest.mark.fast
    @range_test_config(manual_test_config(TEST_GRID))
    def test_transformer_tkg(self, test_manager: Orchestrator, range_test_options: RangeTestCase):
        """Test transformer_tkg with various configurations."""
        self.prepare_test_parametrized(
            test_manager=test_manager,
            test_options=range_test_options,
        )
