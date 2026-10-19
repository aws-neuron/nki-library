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

from typing import final

import neuron_dtypes as dt
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.mlp.mlp_tkg.down_projection_mx_shard_H import down_projection_mx_shard_H
from nkilib_src.nkilib.core.mlp.mlp_tkg.gate_up_projection_mx_shard_H import gate_up_projection_mx_tp_shard_H
from nkilib_src.nkilib.core.mlp.mlp_tkg.mlp_proj_mx_torch import (
    down_proj_mx_torch_ref,
    gate_up_proj_mx_torch_ref,
)
from nkilib_src.nkilib.core.mlp.mlp_tkg.projection_mx_constants import ProjConfig, _pmax, _q_height, _q_width
from nkilib_src.nkilib.core.utils.kernel_helpers import div_ceil

from test.integration.nkilib.utils.tensor_generators import generate_stabilized_mx_data
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# =============================================================================
# Input Builders
# =============================================================================


def build_gate_up_proj_mxfp4_input(BxS: int, H: int, I: int) -> dict:
    """Build input tensors for gate/up projection MXFP4 test."""
    np.random.seed(42)

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
        hidden_qtz_sb=hidden_sb,
        hidden_scale_sb=hidden_scale_sb,
        weight_qtz=weight_qtz,
        weight_scale=weight_scale,
        bias_sb=bias_sb,
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

    # Load bias tensor with shape (1, H_sharded) for Path 1 matmul broadcast.
    # bias=None exercises the no-bias branch inside the sub-kernel.
    if bias is not None:
        bias_sb = nl.ndarray((1, H_sharded), dtype=bias.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=bias_sb, src=bias[:, prg_id * H_sharded : (prg_id + 1) * H_sharded])
    else:
        bias_sb = None

    # Call sub-kernel
    proj_out_sb = down_projection_mx_shard_H(
        inter_sb=inter_sb,
        weight=weight_qtz,
        weight_scale=weight_scale,
        bias_sb=bias_sb,  # ty: ignore[invalid-argument-type]
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


# =============================================================================
# Coverage Wrappers (branch-coverage-focused variants of the sub-kernels)
#
# These exercise sub-kernel branches not hit by the primary wrappers above.
# - "validated" wrappers keep the EXACT param set of the torch reference so
#   they can reuse it; kernel-internal config (out_sb passed, out_p_offset) is
#   hardcoded inside rather than exposed as a param.
# - "trace" wrappers are run with trace_only=True (no golden). Their early-return
#   debug paths and dtype-reinterpret paths produce non-projection outputs that
#   cannot be validated against the projection reference, so we only trace them.
#   Branch coverage is collected at trace time regardless of validation.
# =============================================================================


def _down_stage_inter_and_bias(inter, bias, H, I, BxS, prg_id, n_prgs):
    """Shared input staging for down-projection coverage wrappers."""
    inter_sb = nl.ndarray((128, div_ceil(I, 512), BxS, 4), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=inter_sb, src=inter)

    H_sharded = H // n_prgs
    if bias is not None:
        bias_sb = nl.ndarray((1, H_sharded), dtype=bias.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=bias_sb, src=bias[:, prg_id * H_sharded : (prg_id + 1) * H_sharded])
    else:
        bias_sb = None
    return inter_sb, bias_sb


def down_proj_mxfp4_out_sb_wrapper(
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
    """Down-projection variant that passes a pre-allocated out_sb into the sub-kernel."""
    n_prgs, prg_id = nl.num_programs(0), nl.program_id(0)
    cfg = ProjConfig(H, I, BxS, n_prgs, prg_id, use_stream_shuffle_broadcast=use_stream_shuffle_broadcast)
    inter_sb, bias_sb = _down_stage_inter_and_bias(inter, bias, H, I, BxS, prg_id, n_prgs)

    n_BxS_tiles = div_ceil(BxS, 128)
    # Pre-allocate out_sb and pass it: exercises the "out_sb is provided" branch.
    out_sb = nl.ndarray((128, n_BxS_tiles, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    proj_out_sb = down_projection_mx_shard_H(
        inter_sb=inter_sb,
        weight=weight_qtz,
        weight_scale=weight_scale,
        bias_sb=bias_sb,
        cfg=cfg,
        out_sb=out_sb,
    )

    out = nl.ndarray([BxS, H], dtype=nl.bfloat16, buffer=nl.shared_hbm)
    if prg_id == 0:
        for i_b_tile in nl.affine_range(n_BxS_tiles):
            num_p = min(128, BxS - i_b_tile * 128)
            nisa.dma_copy(dst=out[i_b_tile * 128 : i_b_tile * 128 + num_p, :], src=proj_out_sb[:num_p, i_b_tile, :])
    return out


def down_proj_mxfp4_trace_wrapper(
    dummy_out,
    inter,
    weight_qtz,
    weight_scale,
    bias,
    H: int,
    I: int,
    BxS: int,
    use_stream_shuffle_broadcast: bool = True,
    dbg: bool = False,
):
    """Trace-only down-projection variant for debug early-returns and uint32 weights.

    - dbg=True sets both dbg_hidden and dbg_weight so the sub-kernel takes its two
      debug early-return branches (in the prep helper and in the CTE entry).
    - dbg=False with a uint32-labeled weight exercises the torch_xla reinterpret path.
    """
    n_prgs, prg_id = nl.num_programs(0), nl.program_id(0)
    cfg = ProjConfig(
        H,
        I,
        BxS,
        n_prgs,
        prg_id,
        use_stream_shuffle_broadcast=use_stream_shuffle_broadcast,
        dbg_hidden=dbg,
        dbg_weight=dbg,
    )
    inter_sb, bias_sb = _down_stage_inter_and_bias(inter, bias, H, I, BxS, prg_id, n_prgs)
    _ = down_projection_mx_shard_H(
        inter_sb=inter_sb,
        weight=weight_qtz,
        weight_scale=weight_scale,
        bias_sb=bias_sb,
        cfg=cfg,
    )
    return dummy_out


def _gate_up_stage_hidden_and_scale(hidden_qtz, hidden_scale, H, BxS, prg_id, n_H512_tile_sharded):
    """Shared hidden/scale staging for gate/up coverage wrappers (mirrors primary wrapper)."""
    hidden_sb = nl.ndarray((128, n_H512_tile_sharded, BxS), dtype=nl.float8_e5m2_x4, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=hidden_sb, src=hidden_qtz[:, prg_id * n_H512_tile_sharded : (prg_id + 1) * n_H512_tile_sharded, :]
    )
    hidden_scale_sb = nl.ndarray(hidden_sb.shape, dtype=nl.uint8, buffer=nl.sbuf)
    n_quadrants_needed = 128 // 32
    for i_quad in nl.affine_range(n_quadrants_needed):
        nisa.dma_copy(
            dst=hidden_scale_sb[i_quad * 32 : i_quad * 32 + 4, :, :],
            src=hidden_scale[
                i_quad * 4 : i_quad * 4 + 4, prg_id * n_H512_tile_sharded : (prg_id + 1) * n_H512_tile_sharded, :
            ],
        )
    return hidden_sb, hidden_scale_sb


def gate_up_proj_mxfp4_hbm_trace_wrapper(
    dummy_out,
    hidden_qtz,
    hidden_scale,
    weight_qtz,
    weight_scale,
    bias,
    H: int,
    I: int,
    BxS: int,
    dbg_hidden: bool = False,
    dbg_weight: bool = False,
):
    """Trace-only gate/up variant, weight+scale in HBM. Covers dbg early-returns and uint32 HBM weight."""
    bias_par_dim = 128 if I >= 512 else I // 4
    n_prgs, prg_id = nl.num_programs(0), nl.program_id(0)
    cfg = ProjConfig(H, I, BxS, n_prgs, prg_id, dbg_hidden=dbg_hidden, dbg_weight=dbg_weight)
    n_H512_tile_sharded = H // 512 // n_prgs

    hidden_sb, hidden_scale_sb = _gate_up_stage_hidden_and_scale(
        hidden_qtz, hidden_scale, H, BxS, prg_id, n_H512_tile_sharded
    )
    bias_sb = nl.ndarray((128, div_ceil(I, 512), 4), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=bias_sb, value=0.0)
    nisa.dma_copy(dst=bias_sb[:bias_par_dim, :, :], src=bias)

    _ = gate_up_projection_mx_tp_shard_H(
        hidden_qtz_sb=hidden_sb,
        hidden_scale_sb=hidden_scale_sb,
        weight_qtz=weight_qtz,
        weight_scale=weight_scale,
        bias_sb=bias_sb,
        cfg=cfg,
    )
    return dummy_out


def gate_up_proj_mxfp4_wsbuf_trace_wrapper(
    dummy_out,
    hidden_qtz,
    hidden_scale,
    weight_qtz,
    weight_scale,
    bias,
    H: int,
    I: int,
    BxS: int,
):
    """Trace-only gate/up variant that stages weight AND scale into SBUF before the sub-kernel.

    Exercises the SBUF-weight and SBUF-scale branches. A uint32-labeled weight additionally
    hits the reinterpret path; a float4 weight hits its complement.
    """
    bias_par_dim = 128 if I >= 512 else I // 4
    n_prgs, prg_id = nl.num_programs(0), nl.program_id(0)
    cfg = ProjConfig(H, I, BxS, n_prgs, prg_id)
    n_H512_tile_sharded = H // 512 // n_prgs

    hidden_sb, hidden_scale_sb = _gate_up_stage_hidden_and_scale(
        hidden_qtz, hidden_scale, H, BxS, prg_id, n_H512_tile_sharded
    )
    bias_sb = nl.ndarray((128, div_ceil(I, 512), 4), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=bias_sb, value=0.0)
    nisa.dma_copy(dst=bias_sb[:bias_par_dim, :, :], src=bias)

    # Stage weight into SBUF (shard on H512 tiles).
    weight_sb = nl.ndarray((128, n_H512_tile_sharded, I), dtype=weight_qtz.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=weight_sb, src=weight_qtz[:, prg_id * n_H512_tile_sharded : (prg_id + 1) * n_H512_tile_sharded, :]
    )
    # Stage a scale tensor into SBUF (values irrelevant for trace-only).
    weight_scale_sb = nl.ndarray((128, n_H512_tile_sharded, I), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.memset(dst=weight_scale_sb, value=0)

    _ = gate_up_projection_mx_tp_shard_H(
        hidden_qtz_sb=hidden_sb,
        hidden_scale_sb=hidden_scale_sb,
        weight_qtz=weight_sb,
        weight_scale=weight_scale_sb,
        bias_sb=bias_sb,
        cfg=cfg,
    )
    return dummy_out


def gate_up_proj_mxfp4_biasshared_trace_wrapper(
    dummy_out,
    hidden_qtz,
    hidden_scale,
    weight_qtz,
    weight_scale,
    bias,
    H: int,
    I: int,
    BxS: int,
):
    """Trace-only gate/up variant with a 4D gate+up-shared bias (bias_t_shared_between_gate_up=True)."""
    bias_par_dim = 128 if I >= 512 else I // 4
    n_prgs, prg_id = nl.num_programs(0), nl.program_id(0)
    cfg = ProjConfig(H, I, BxS, n_prgs, prg_id, bias_t_shared_between_gate_up=True, bias_t_shared_base_offset=0)
    n_H512_tile_sharded = H // 512 // n_prgs

    hidden_sb, hidden_scale_sb = _gate_up_stage_hidden_and_scale(
        hidden_qtz, hidden_scale, H, BxS, prg_id, n_H512_tile_sharded
    )
    # 4D shared bias: [_pmax, 2 (gate/up), n_I512_tile, _q_width].
    bias_sb = nl.ndarray((128, 2, div_ceil(I, 512), 4), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=bias_sb, value=0.0)
    nisa.dma_copy(dst=bias_sb[:bias_par_dim, :, :, :], src=bias)

    _ = gate_up_projection_mx_tp_shard_H(
        hidden_qtz_sb=hidden_sb,
        hidden_scale_sb=hidden_scale_sb,
        weight_qtz=weight_qtz,
        weight_scale=weight_scale,
        bias_sb=bias_sb,
        cfg=cfg,
    )
    return dummy_out


# Wrap with nki.jit()
gate_up_proj_mxfp4_kernel = nki.jit()(gate_up_proj_mxfp4_wrapper)
down_proj_mxfp4_kernel = nki.jit()(down_proj_mxfp4_wrapper)
down_proj_mxfp4_out_sb_kernel = nki.jit()(down_proj_mxfp4_out_sb_wrapper)
down_proj_mxfp4_trace_kernel = nki.jit()(down_proj_mxfp4_trace_wrapper)
gate_up_proj_mxfp4_hbm_trace_kernel = nki.jit()(gate_up_proj_mxfp4_hbm_trace_wrapper)
gate_up_proj_mxfp4_wsbuf_trace_kernel = nki.jit()(gate_up_proj_mxfp4_wsbuf_trace_wrapper)
gate_up_proj_mxfp4_biasshared_trace_kernel = nki.jit()(gate_up_proj_mxfp4_biasshared_trace_wrapper)


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
_GATE_UP_ABBREVS = {"BxS": "bs"}

_GATE_UP_FAST_KEYS = frozenset({(4, 3072, 192), (4, 3072, 1536)})
GATE_UP_TEST_PARAMS = [
    pytest.param(*v, marks=pytest.mark.fast) if tuple(v) in _GATE_UP_FAST_KEYS else pytest.param(*v)
    for v in MXFP4_PROJ_LNC2_TEST_VECTORS
]

DOWN_PARAM_NAMES = "BxS, H, I, use_stream_shuffle_broadcast"
_DOWN_ABBREVS = {"BxS": "bs", "use_stream_shuffle_broadcast": "ssb"}

_DOWN_FAST_KEYS = frozenset({(256, 3072, 384, False), (4, 3072, 768, True)})
DOWN_TEST_PARAMS = [
    pytest.param(*p, marks=pytest.mark.fast) if p in _DOWN_FAST_KEYS else pytest.param(*p)
    for BxS, H, I in MXFP4_PROJ_LNC2_TEST_VECTORS
    for use_ssb in [True, False]
    for p in [(BxS, H, I, use_ssb)]
]


# =============================================================================
# Test Class
# =============================================================================


@pytest_test_metadata(name="MLP Projection MXFP4")
@pytest_marks(["mlp", "mxfp4", "projection"])
@final
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
class TestMlpProjMxfp4Kernel:
    """Test suite for MXFP4 MLP projection kernels."""

    @pytest_parametrize(GATE_UP_PARAM_NAMES, GATE_UP_TEST_PARAMS, abbrevs=_GATE_UP_ABBREVS)
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
            torch_ref=torch_ref_wrapper(gate_up_proj_mx_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=2e-2,
            atol=1e-5,
        )

    @pytest_parametrize(DOWN_PARAM_NAMES, DOWN_TEST_PARAMS, abbrevs=_DOWN_ABBREVS)
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
            torch_ref=torch_ref_wrapper(down_proj_mx_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=2e-2,
            atol=1e-5,
        )

    # =========================================================================
    # Coverage tests (close branch-coverage gaps in the sub-kernels)
    # =========================================================================

    @pytest.mark.fast
    def test_mxfp4_down_proj_nobias(self, test_manager: Orchestrator, platform_target: Platforms):
        """No-bias path: bias_sb=None exercises the sub-kernel's no-bias branches."""
        BxS, H, I = 4, 3072, 384

        def input_generator(test_config, input_tensor_def=None):
            kernel_input = build_down_proj_mxfp4_input(BxS, H, I)
            kernel_input["bias"] = None
            return kernel_input

        def output_tensors(kernel_input):
            return {"out": np.zeros((BxS, H), dtype=nl.bfloat16)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=down_proj_mxfp4_kernel,
            torch_ref=torch_ref_wrapper(down_proj_mx_torch_ref),
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
    def test_mxfp4_down_proj_lnc1(self, test_manager: Orchestrator, platform_target: Platforms):
        """LNC1 (n_prgs==1): skips the LNC sendrecv branch."""
        BxS, H, I = 4, 3072, 384

        def input_generator(test_config, input_tensor_def=None):
            return build_down_proj_mxfp4_input(BxS, H, I)

        def output_tensors(kernel_input):
            return {"out": np.zeros((BxS, H), dtype=nl.bfloat16)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=down_proj_mxfp4_kernel,
            torch_ref=torch_ref_wrapper(down_proj_mx_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=2e-2,
            atol=1e-5,
        )

    @pytest.mark.fast
    def test_mxfp4_down_proj_out_sb(self, test_manager: Orchestrator, platform_target: Platforms):
        """Pre-allocated out_sb passed into the sub-kernel."""
        BxS, H, I = 4, 3072, 384

        def input_generator(test_config, input_tensor_def=None):
            return build_down_proj_mxfp4_input(BxS, H, I)

        def output_tensors(kernel_input):
            return {"out": np.zeros((BxS, H), dtype=nl.bfloat16)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=down_proj_mxfp4_out_sb_kernel,
            torch_ref=torch_ref_wrapper(down_proj_mx_torch_ref),
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
    def test_mxfp4_down_proj_dbg_trace(self, test_manager: Orchestrator, platform_target: Platforms):
        """Trace-only: dbg_hidden + dbg_weight early-return branches."""
        BxS, H, I = 4, 3072, 384

        def input_generator(test_config, input_tensor_def=None):
            kernel_input = build_down_proj_mxfp4_input(BxS, H, I)
            kernel_input["dummy_out.must_alias_input"] = np.zeros((1,), dtype=np.float32)
            kernel_input["dbg"] = True
            return kernel_input

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=down_proj_mxfp4_trace_kernel,
            kernel_input_generator=input_generator,
            trace_only=True,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
        )

    @pytest.mark.fast
    def test_mxfp4_down_proj_uint32_weight_trace(self, test_manager: Orchestrator, platform_target: Platforms):
        """Trace-only: uint32-labeled (torch_xla container) weight reinterpret path."""
        BxS, H, I = 4, 3072, 768

        def input_generator(test_config, input_tensor_def=None):
            kernel_input = build_down_proj_mxfp4_input(BxS, H, I)
            # Relabel the (already-shaped) packed weight as its uint32 container.
            w = kernel_input["weight_qtz"]
            kernel_input["weight_qtz"] = np.zeros(w.shape, dtype=np.uint32)
            kernel_input["dummy_out.must_alias_input"] = np.zeros((1,), dtype=np.float32)
            return kernel_input

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=down_proj_mxfp4_trace_kernel,
            kernel_input_generator=input_generator,
            trace_only=True,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "dbg_hidden, dbg_weight, uint32_weight",
        [(True, False, False), (False, True, False), (False, False, True)],
        ids=["dbg_hidden", "dbg_weight", "uint32"],
    )
    def test_mxfp4_gate_up_proj_hbm_trace(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        dbg_hidden: bool,
        dbg_weight: bool,
        uint32_weight: bool,
    ):
        """Trace-only gate/up (HBM weight+scale): dbg early-returns and uint32 HBM weight."""
        BxS, H, I = 4, 3072, 384

        def input_generator(test_config, input_tensor_def=None):
            kernel_input = build_gate_up_proj_mxfp4_input(BxS, H, I)
            if uint32_weight:
                w = kernel_input["weight_qtz"]
                kernel_input["weight_qtz"] = np.zeros(w.shape, dtype=np.uint32)
            kernel_input["dummy_out.must_alias_input"] = np.zeros((1,), dtype=np.float32)
            kernel_input["dbg_hidden"] = dbg_hidden
            kernel_input["dbg_weight"] = dbg_weight
            return kernel_input

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=gate_up_proj_mxfp4_hbm_trace_kernel,
            kernel_input_generator=input_generator,
            trace_only=True,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("uint32_weight", [True, False], ids=["uint32", "float4"])
    def test_mxfp4_gate_up_proj_wsbuf_trace(
        self, test_manager: Orchestrator, platform_target: Platforms, uint32_weight: bool
    ):
        """Trace-only gate/up with weight+scale staged into SBUF (uint32 and float4 weight)."""
        BxS, H, I = 4, 3072, 384

        def input_generator(test_config, input_tensor_def=None):
            kernel_input = build_gate_up_proj_mxfp4_input(BxS, H, I)
            if uint32_weight:
                w = kernel_input["weight_qtz"]
                kernel_input["weight_qtz"] = np.zeros(w.shape, dtype=np.uint32)
            kernel_input["dummy_out.must_alias_input"] = np.zeros((1,), dtype=np.float32)
            return kernel_input

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=gate_up_proj_mxfp4_wsbuf_trace_kernel,
            kernel_input_generator=input_generator,
            trace_only=True,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
        )

    @pytest.mark.fast
    def test_mxfp4_gate_up_proj_biasshared_trace(self, test_manager: Orchestrator, platform_target: Platforms):
        """Trace-only gate/up with a 4D gate+up-shared bias (bias_t_shared_between_gate_up)."""
        np.random.seed(42)
        BxS, H, I = 4, 3072, 384

        def input_generator(test_config, input_tensor_def=None):
            kernel_input = build_gate_up_proj_mxfp4_input(BxS, H, I)
            bias_par_dim = 128 if I >= 512 else I // 4
            n_I512_tile = div_ceil(I, 512)
            # 4D shared bias: [bias_par_dim, 2 (gate/up), n_I512_tile, 4].
            bias = np.random.rand(bias_par_dim, 2, n_I512_tile, 4).astype(np.float32)
            kernel_input["bias"] = dt.static_cast(bias, nl.bfloat16)
            kernel_input["dummy_out.must_alias_input"] = np.zeros((1,), dtype=np.float32)
            return kernel_input

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=gate_up_proj_mxfp4_biasshared_trace_kernel,
            kernel_input_generator=input_generator,
            trace_only=True,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
        )
