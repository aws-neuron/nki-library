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

"""
Integration tests for MXFP4 MLP projection kernels (gate_up and down projections).

Tests the gate_up_projection_mx_tp_shard_H and down_projection_mx_shard_H sub-kernels
with LNC2 sharding on the H (hidden) dimension.
"""

from test.integration.nkilib.utils.tensor_generators import generate_stabilized_mx_data
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper
from typing import final

import neuron_dtypes as dt
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.mlp.mlp_tkg.down_projection_mx_shard_H import down_projection_mx_shard_H
from nkilib_src.nkilib.core.mlp.mlp_tkg.gate_up_projection_mx_shard_H import gate_up_projection_mx_tp_shard_H
from nkilib_src.nkilib.core.mlp.mlp_tkg.mlp_proj_mxfp4_torch import (
    down_proj_mxfp4_torch_ref,
    gate_up_proj_mxfp4_torch_ref,
)
from nkilib_src.nkilib.core.mlp.mlp_tkg.projection_mx_constants import ProjConfig, _pmax, _q_height, _q_width
from nkilib_src.nkilib.core.utils.kernel_helpers import div_ceil
from nkilib_src.nkilib.core.utils.tensor_view import TensorView

# =============================================================================
# Input Builders
# =============================================================================


def build_gate_up_proj_mxfp4_input(BxS: int, H: int, I: int) -> dict:
    """Build input tensors for gate/up projection MXFP4 test."""
    np.random.seed(42)

    n_H512_tile = H // 512
    n_I512_tile = div_ceil(I, 512)

    _, hidden_qtz, hidden_scale = generate_stabilized_mx_data(nl.float8_e5m2_x4, (128, H // 128 * BxS))
    _, weight_qtz, weight_scale = generate_stabilized_mx_data(nl.float4_e2m1fn_x4, (128, H // 128 * I))

    hidden_qtz = hidden_qtz.reshape(128, -1, BxS)
    hidden_scale = hidden_scale.reshape(16, -1, BxS)
    weight_qtz = weight_qtz.reshape(128, -1, I)
    weight_scale = weight_scale.reshape(16, -1, I)

    bias_par_dim = 128 if I >= 512 else I // 4
    bias = np.random.rand(bias_par_dim, n_I512_tile, 4).astype(np.float32)
    if I % 512 != 0 and I > 512:
        last_I512_tile = I // 512
        num_padded_rows = (512 - (I % 512)) // 4
        bias[128 - num_padded_rows :, last_I512_tile, :] = 0
    bias = dt.static_cast(bias, nl.bfloat16)

    return {
        "hidden_qtz": hidden_qtz,
        "hidden_scale": hidden_scale,
        "weight_qtz": weight_qtz,
        "weight_scale": weight_scale,
        "bias": bias,
        "H": H,
        "I": I,
        "BxS": BxS,
    }


def build_down_proj_mxfp4_input(BxS: int, H: int, I: int, use_stream_shuffle_broadcast: bool = True) -> dict:
    """Build input tensors for down projection MXFP4 test."""
    np.random.seed(42)

    n_I512_tiles = div_ceil(I, 512)

    inter, _, _ = generate_stabilized_mx_data(nl.float8_e5m2_x4, (I // 4, BxS * 4))
    inter = inter.reshape(I // 4, BxS, 4).transpose(2, 0, 1).reshape(I, BxS)
    inter = inter.reshape(I // 4, BxS, 4)
    inter = dt.static_cast(inter, dtype=nl.bfloat16)

    inter_kernel = np.full((128, n_I512_tiles, BxS, _q_width), 6767)
    inter_kernel = dt.static_cast(inter_kernel, dtype=nl.bfloat16)
    for i_tile in range(n_I512_tiles):
        rows_filled = 128
        if (i_tile == n_I512_tiles - 1) and (I % 512 != 0):
            rows_filled = (I % 512) // 4
        cur_tile = inter[i_tile * 128 : i_tile * 128 + rows_filled, ...]
        inter_kernel[:rows_filled, i_tile, :, :] = cur_tile

    _, weight_qtz, weight_scale = generate_stabilized_mx_data(nl.float4_e2m1fn_x4, (I // 4, H * 4))

    p_size = _pmax if I >= 512 else I // _q_width
    tmp_qtz = np.zeros((p_size, n_I512_tiles, H), dtype=weight_qtz.dtype)
    tmp_scale = np.zeros((p_size // _q_height, n_I512_tiles, H), dtype=np.uint8)

    for i_I512_tile in range(n_I512_tiles):
        n_rows_qtz = min(128, I // 4 - i_I512_tile * 128)
        n_rows_scale = min(16, I // 32 - i_I512_tile * 16)
        tmp_qtz[:n_rows_qtz, i_I512_tile, :] = weight_qtz[i_I512_tile * 128 : i_I512_tile * 128 + n_rows_qtz, :]
        tmp_scale[:n_rows_scale, i_I512_tile, :] = weight_scale[i_I512_tile * 16 : i_I512_tile * 16 + n_rows_scale, :]

    weight_qtz, weight_scale = tmp_qtz, tmp_scale

    bias = np.random.rand(1, H).astype(np.float32)
    bias = bias.reshape(H // 128, 128).T.reshape(1, H)
    bias = dt.static_cast(bias, nl.bfloat16)

    return {
        "inter": inter_kernel,
        "weight_qtz": weight_qtz,
        "weight_scale": weight_scale,
        "bias": bias,
        "H": H,
        "I": I,
        "BxS": BxS,
        "use_stream_shuffle_broadcast": use_stream_shuffle_broadcast,
    }


# =============================================================================
# Kernel Wrapper Functions
# =============================================================================


def gate_up_proj_mxfp4_wrapper(
    hidden_qtz,
    hidden_scale,
    weight_qtz,
    weight_scale,
    bias,
    H: int,
    I: int,
    BxS: int,
    hidden_unpack_fn=None,
    weight_unpack_fn=None,
):
    """Wrapper for gate/up projection sub-kernel."""
    bias_par_dim = 128 if I >= 512 else I // 4

    n_prgs, prg_id = nl.num_programs(0), nl.program_id(0)
    cfg = ProjConfig(H, I, BxS, n_prgs, prg_id)
    n_H512_tile_sharded = H // 512 // n_prgs

    # Load shuffled & quantized hidden
    hidden_sb = nl.ndarray((128, n_H512_tile_sharded, BxS), dtype=nl.float8_e5m2_x4, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=hidden_sb, src=hidden_qtz[:, prg_id * n_H512_tile_sharded : (prg_id + 1) * n_H512_tile_sharded, :]
    )

    # Load shuffled bias
    bias_sb = nl.ndarray((128, div_ceil(I, 512), 4), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=bias_sb, value=0.0)
    nisa.dma_copy(dst=bias_sb[:bias_par_dim, :, :], src=bias)

    # Load hidden scales, 4 partitions per sb quadrant
    hidden_scale_sb = nl.ndarray(hidden_sb.shape, dtype=nl.uint8, buffer=nl.sbuf)
    n_quadrants_needed = 128 // 32
    for i_quad in nl.affine_range(n_quadrants_needed):
        nisa.dma_copy(
            dst=hidden_scale_sb[i_quad * 32 : i_quad * 32 + 4, :, :],
            src=hidden_scale[
                i_quad * 4 : i_quad * 4 + 4, prg_id * n_H512_tile_sharded : (prg_id + 1) * n_H512_tile_sharded, :
            ],
        )

    # Call sub-kernel
    proj_out_sb = gate_up_projection_mx_tp_shard_H(
        hidden_qtz_sb=TensorView(hidden_sb),
        hidden_scale_sb=TensorView(hidden_scale_sb),
        weight_qtz=TensorView(weight_qtz),
        weight_scale=TensorView(weight_scale),
        bias_sb=TensorView(bias_sb),
        cfg=cfg,
    )

    # Store output
    out = nl.ndarray(proj_out_sb.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)
    if prg_id == 0:
        nisa.dma_copy(dst=out, src=proj_out_sb)
        if I % 512 != 0:
            last_I512_tile = I // 512
            num_padded_rows = (512 - (I % 512)) // 4
            zeros = nl.ndarray([num_padded_rows, 1, BxS, 4], dtype=out.dtype, buffer=nl.sbuf)
            nisa.memset(dst=zeros, value=0.0)
            nisa.dma_copy(dst=out[128 - num_padded_rows :, last_I512_tile, :, :], src=zeros)

    return out


def down_proj_mxfp4_wrapper(
    inter,
    weight_qtz,
    weight_scale,
    bias,
    H: int,
    I: int,
    BxS: int,
    use_stream_shuffle_broadcast: bool = True,
    weight_unpack_fn=None,
):
    """Wrapper for down projection sub-kernel."""
    n_prgs, prg_id = nl.num_programs(0), nl.program_id(0)
    cfg = ProjConfig(H, I, BxS, n_prgs, prg_id, use_stream_shuffle_broadcast=use_stream_shuffle_broadcast)
    H_sharded = H // n_prgs

    # Load intermediate tensor
    inter_sb = nl.ndarray((128, div_ceil(I, 512), BxS, 4), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=inter_sb, src=inter)

    # Load bias tensor with shape (1, H_sharded) for Path 1 matmul broadcast
    bias_sb = nl.ndarray((1, H_sharded), dtype=bias.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=bias_sb, src=bias[:, prg_id * H_sharded : (prg_id + 1) * H_sharded])

    # Call sub-kernel
    proj_out_sb = down_projection_mx_shard_H(
        inter_sb=inter_sb,
        weight=weight_qtz,
        weight_scale=weight_scale,
        bias_sb=bias_sb,
        cfg=cfg,
    )

    # Store output - kernel returns [BxS_tile_sz, n_BxS_tile, H] tiled layout
    # Copy to [BxS, H] flat layout using slicing
    n_BxS_tiles = div_ceil(BxS, 128)
    out = nl.ndarray([BxS, H], dtype=nl.bfloat16, buffer=nl.shared_hbm)

    if prg_id == 0:
        for i_b_tile in nl.affine_range(n_BxS_tiles):
            num_p = min(128, BxS - i_b_tile * 128)
            nisa.dma_copy(
                dst=out[i_b_tile * 128 : i_b_tile * 128 + num_p, :],
                src=proj_out_sb[:num_p, i_b_tile, :],
            )

    return out


# Wrap with nki.jit()
gate_up_proj_mxfp4_kernel = nki.jit()(gate_up_proj_mxfp4_wrapper)
down_proj_mxfp4_kernel = nki.jit()(down_proj_mxfp4_wrapper)


# =============================================================================
# Test Vectors
# =============================================================================

MXFP4_PROJ_LNC2_TEST_VECTORS = [
    [4, 3072, 192],
    [2048, 3072, 192],
    [4, 3072, 384],
    [256, 3072, 384],
    [2048, 3072, 384],
    [4, 3072, 768],
    [4, 3072, 1536],
]

GATE_UP_PARAM_NAMES = "BxS, H, I"
GATE_UP_TEST_PARAMS = [tuple(v) for v in MXFP4_PROJ_LNC2_TEST_VECTORS]

DOWN_PARAM_NAMES = "BxS, H, I, use_stream_shuffle_broadcast"
DOWN_TEST_PARAMS = [(BxS, H, I, use_ssb) for BxS, H, I in MXFP4_PROJ_LNC2_TEST_VECTORS for use_ssb in [True, False]]


# =============================================================================
# Test Class
# =============================================================================


@pytest_test_metadata(
    name="MLP Projection MXFP4",
    pytest_marks=["mlp", "mxfp4", "projection"],
)
@final
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
class TestMlpProjMxfp4Kernel:
    """Test suite for MXFP4 MLP projection kernels."""

    @pytest.mark.fast
    @pytest.mark.parametrize(GATE_UP_PARAM_NAMES, GATE_UP_TEST_PARAMS)
    def test_mxfp4_gate_up_proj_unit(
        self,
        test_manager: Orchestrator,
        BxS: int,
        H: int,
        I: int,
        platform_target: Platforms,
    ):
        def input_generator(test_config, input_tensor_def=None):
            return build_gate_up_proj_mxfp4_input(BxS, H, I)

        def output_tensors(kernel_input):
            n_I512_tile = div_ceil(I, 512)
            return {"out": np.zeros((128, n_I512_tile, BxS, 4), dtype=nl.bfloat16)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=gate_up_proj_mxfp4_kernel,
            torch_ref=torch_ref_wrapper(gate_up_proj_mxfp4_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=2e-2,
            atol=1e-5,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(DOWN_PARAM_NAMES, DOWN_TEST_PARAMS)
    def test_mxfp4_down_proj_unit(
        self,
        test_manager: Orchestrator,
        BxS: int,
        H: int,
        I: int,
        use_stream_shuffle_broadcast: bool,
        platform_target: Platforms,
    ):
        def input_generator(test_config, input_tensor_def=None):
            return build_down_proj_mxfp4_input(BxS, H, I, use_stream_shuffle_broadcast)

        def output_tensors(kernel_input):
            return {"out": np.zeros((BxS, H), dtype=nl.bfloat16)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=down_proj_mxfp4_kernel,
            torch_ref=torch_ref_wrapper(down_proj_mxfp4_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=2e-2,
            atol=1e-5,
        )
