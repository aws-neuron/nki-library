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

"""Integration tests for matmul_mxfp8_generic_api exercising API-specific data paths
beyond what the matmul_mxfp8_generic_kernel tests cover.

Tested data paths:
  HBM K-loop: API handles full K loop from HBM (single M/N block) -> HBM & SBUF
  HBM K-loop with spill: Same as above with spill_reload=True
  SBUF preloaded: Caller pre-loads SBUF, API does single-block matmul -> HBM & SBUF
  SBUF empty TD fill: API loads from HBM and populates caller-provided empty TDs
  LHS M-sharded: LNC2 shards on M dimension via lhs_m_offset
"""

import os

import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_config import (
    MatmulMxfp8KernelConfig,
    auto_generate_default,
    validate_shapes,
)
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_constants import (
    PRECISION_BFLOAT16,
    PRECISION_FP32,
    PRECISION_MXFP8,
    PRECISION_MXFP8_X4,
)
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_generic_api import (
    generic_matmul_mxfp8_api,
)
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_torch import matmul_mxfp8_torch_ref
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils import (
    load_block,
    quantize_mxfp8_block,
)
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.common_dataclasses import (
    TensorDescriptor,
)
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.common_utils import (
    create_and_set_active_sbm,
    get_active_sbm,
    with_active_sbm,
)
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.quantize_mxfp8_utils import (
    get_fp8_dtype_x4,
)

from test.integration.nkilib.experimental.matmul_mxfp8 import config_helper, constants
from test.integration.nkilib.experimental.matmul_mxfp8.test_matmul_mxfp8_generic_kernel import (
    _mxfp8_comparator,
    build_matmul_inputs,
    get_output_dtype,
)
from test.utils import common_dataclasses
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.unit_test_framework import UnitTestFramework

# ---------------------------------------------------------------------------
# Common: build TDs + shapes from kernel args (same as matmul_mxfp8 entry)
# ---------------------------------------------------------------------------


class _SbufOutputTd(nl.NKIObject):
    """Lightweight output TD for 3D SBUF tensors (avoids TensorDescriptor shape computation)."""

    def __init__(self, data):
        self.data = data


def _setup(
    lhs,
    rhs,
    lhs_scales,
    rhs_scales,
    lhs_is_swizzled,
    rhs_is_swizzled,
    run_with_lnc2,
    TILES_IN_BLOCK_M,
    TILES_IN_BLOCK_N,
    TILES_IN_BLOCK_K,
    TILES_IN_LOAD_M,
    TILES_IN_LOAD_N,
    lhs_matmul_tile_shape_logical,
    rhs_matmul_tile_shape_logical,
    use_scale_packing,
    block_loop_order=None,
    spill_reload=False,
    lhs_load_with_PE_swizzle=False,
    rhs_load_with_PE_swizzle=False,
    lhs_is_f_by_k=None,
    rhs_is_f_by_k=None,
):
    lhs_td = TensorDescriptor(
        data=lhs,
        scales=lhs_scales,
        is_swizzled=lhs_is_swizzled,
        load_with_PE_swizzle=lhs_load_with_PE_swizzle if not lhs_is_swizzled else False,
        is_f_by_k=lhs_is_f_by_k,
    )
    rhs_td = TensorDescriptor(
        data=rhs,
        scales=rhs_scales,
        is_swizzled=rhs_is_swizzled,
        is_col_parallel_sharded=run_with_lnc2,
        load_with_PE_swizzle=rhs_load_with_PE_swizzle if not rhs_is_swizzled else False,
        is_f_by_k=rhs_is_f_by_k,
    )

    K_logical, M_logical = lhs_td.sharded_logical_shape
    _, N_logical = rhs_td.sharded_logical_shape
    lhs_precision = (
        PRECISION_BFLOAT16 if not lhs_td.is_quantized else (PRECISION_MXFP8_X4 if lhs_td.is_x4 else PRECISION_MXFP8)
    )
    rhs_precision = (
        PRECISION_BFLOAT16 if not rhs_td.is_quantized else (PRECISION_MXFP8_X4 if rhs_td.is_x4 else PRECISION_MXFP8)
    )

    config = MatmulMxfp8KernelConfig(
        M=M_logical,
        K=K_logical,
        N=N_logical,
        tile_m=lhs_matmul_tile_shape_logical[1] if lhs_matmul_tile_shape_logical else None,
        tile_k=lhs_matmul_tile_shape_logical[0] if lhs_matmul_tile_shape_logical else None,
        tile_n=rhs_matmul_tile_shape_logical[1] if rhs_matmul_tile_shape_logical else None,
        TILES_IN_BLOCK_M=TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N=TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K=TILES_IN_BLOCK_K,
        TILES_IN_LOAD_M=TILES_IN_LOAD_M,
        TILES_IN_LOAD_N=TILES_IN_LOAD_N,
        enable_scale_packing=use_scale_packing,
        run_with_lnc2=run_with_lnc2,
        lnc_2_shard_rhs=True,
        lhs_is_swizzled=lhs_is_swizzled,
        rhs_is_swizzled=rhs_is_swizzled,
    )
    auto_generate_default(config, lhs_precision, rhs_precision, PRECISION_FP32)
    validate_shapes(config, lhs_td, rhs_td)

    shapes = {
        'bd': config.bd,
        'BLOCKS_IN_M': config.BLOCKS_IN_M,
        'BLOCKS_IN_N': config.BLOCKS_IN_N,
        'BLOCKS_IN_K': config.BLOCKS_IN_K,
        'lhs_matmul_tile_shape_physical': config.lhs_matmul_tile_shape_physical,
        'rhs_matmul_tile_shape_physical': config.rhs_matmul_tile_shape_physical,
        'lhs_load_tile_shape': config.lhs_load_tile_shape,
        'rhs_load_tile_shape': config.rhs_load_tile_shape,
        'lhs_quantize_tile_shape': config.lhs_quantize_tile_shape,
        'rhs_quantize_tile_shape': config.rhs_quantize_tile_shape,
    }
    return lhs_td, rhs_td, shapes, config.TILES_IN_LOAD_M, config.TILES_IN_LOAD_N


def _alloc_output_and_lnc(lhs_td, rhs_td, shapes, run_with_lnc2, output_dtype):
    """Allocate HBM output and compute LNC2 sharding offsets."""
    output_hbm = nl.ndarray(
        (lhs_td.logical_shape[1], rhs_td.logical_shape[1]),
        dtype=output_dtype,
        buffer=nl.shared_hbm,
    )
    N_LOGICAL_SHARDED = rhs_td.sharded_logical_shape[1]
    N_PHYSICAL_SHARDED = rhs_td.sharded_physical_shape[1]

    if run_with_lnc2:
        LNC_ID = nl.program_id(axis=0)
        out_col = LNC_ID * N_LOGICAL_SHARDED
        output_sharded = output_hbm[:, out_col : out_col + N_LOGICAL_SHARDED]
        rhs_n_offset = LNC_ID * N_PHYSICAL_SHARDED
        BLOCKS_IN_N_sharded = (shapes['BLOCKS_IN_N'] + 1) // 2
    else:
        output_sharded = output_hbm
        rhs_n_offset = 0
        BLOCKS_IN_N_sharded = shapes['BLOCKS_IN_N']

    return output_hbm, output_sharded, rhs_n_offset, BLOCKS_IN_N_sharded


def _store_sbuf_to_hbm(output_sbuf, output_hbm, bd, LHS_MATMUL_TILE_M):
    """DMA one (m=0, n=0) block from 3D SBUF accumulator to HBM."""
    for tile_m in range(bd.TILES_IN_BLOCK_M):
        out_m = tile_m * LHS_MATMUL_TILE_M
        step_p = bd.TILES_IN_BLOCK_M * bd.BLOCK_N_LOGICAL
        offset = tile_m * bd.BLOCK_N_LOGICAL
        nisa.dma_copy(
            dst=output_hbm[out_m : out_m + LHS_MATMUL_TILE_M, 0 : bd.BLOCK_N_LOGICAL],
            src=output_sbuf.ap(
                pattern=[[step_p, LHS_MATMUL_TILE_M], [1, bd.BLOCK_N_LOGICAL]],
                offset=offset,
            ),
        )


# ---------------------------------------------------------------------------
# HBM K-loop: Load from HBM, API handles K loop (single M/N block) -> HBM/SBUF
#   spill_reload and output_to_sbuf are passed as kernel parameters.
# ---------------------------------------------------------------------------


@with_active_sbm
def _kernel_hbm_k_loop(
    lhs,
    rhs,
    output_dtype,
    run_with_lnc2,
    float8_dtype,
    use_scale_packing,
    tile_loop_order,
    lhs_scales,
    rhs_scales,
    TILES_IN_BLOCK_M=None,
    TILES_IN_BLOCK_N=None,
    TILES_IN_BLOCK_K=None,
    TILES_IN_LOAD_M=None,
    TILES_IN_LOAD_N=None,
    lhs_matmul_tile_shape_logical=None,
    rhs_matmul_tile_shape_logical=None,
    lhs_is_swizzled=True,
    rhs_is_swizzled=True,
    block_loop_order=None,
    spill_reload=False,
    output_to_sbuf=False,
    lhs_load_with_PE_swizzle=False,
    rhs_load_with_PE_swizzle=False,
    lhs_is_f_by_k=None,
    rhs_is_f_by_k=None,
    enable_psum_copy_in=None,
    quant_scheme="wrapX",
):
    create_and_set_active_sbm()
    sbm = get_active_sbm()
    sbm.open_scope("TEST API")

    lhs_td, rhs_td, shapes, TILES_IN_LOAD_M, TILES_IN_LOAD_N = _setup(
        lhs=lhs,
        rhs=rhs,
        lhs_scales=lhs_scales,
        rhs_scales=rhs_scales,
        lhs_is_swizzled=lhs_is_swizzled,
        rhs_is_swizzled=rhs_is_swizzled,
        run_with_lnc2=run_with_lnc2,
        use_scale_packing=use_scale_packing,
        TILES_IN_BLOCK_M=TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N=TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K=TILES_IN_BLOCK_K,
        TILES_IN_LOAD_M=TILES_IN_LOAD_M,
        TILES_IN_LOAD_N=TILES_IN_LOAD_N,
        lhs_matmul_tile_shape_logical=lhs_matmul_tile_shape_logical,
        rhs_matmul_tile_shape_logical=rhs_matmul_tile_shape_logical,
        lhs_load_with_PE_swizzle=lhs_load_with_PE_swizzle,
        rhs_load_with_PE_swizzle=rhs_load_with_PE_swizzle,
        lhs_is_f_by_k=lhs_is_f_by_k,
        rhs_is_f_by_k=rhs_is_f_by_k,
    )
    bd = shapes['bd']
    LHS_MATMUL_TILE_M = shapes['lhs_matmul_tile_shape_physical'][1]
    output_hbm, output_sharded, rhs_n_offset, BLOCKS_IN_N_sharded = _alloc_output_and_lnc(
        lhs_td,
        rhs_td,
        shapes,
        run_with_lnc2,
        output_dtype,
    )

    lhsq_td, rhsq_td = None, None
    if spill_reload:
        fp8_x4_dtype = get_fp8_dtype_x4(float8_dtype)
        data_buffer = nl.private_hbm if run_with_lnc2 else nl.hbm
        TILE_K = shapes['lhs_load_tile_shape'][0]
        BLOCK_K_SIZE = TILE_K * bd.TILES_IN_BLOCK_K
        if not lhs_td.is_quantized:
            lhsq_td = TensorDescriptor(
                data=nl.ndarray(
                    (shapes['BLOCKS_IN_K'] * BLOCK_K_SIZE, shapes['BLOCKS_IN_M'] * bd.BLOCK_M_LOGICAL),
                    dtype=fp8_x4_dtype,
                    buffer=data_buffer,
                ),
                scales=nl.ndarray(
                    (shapes['BLOCKS_IN_K'] * BLOCK_K_SIZE, shapes['BLOCKS_IN_M'] * bd.BLOCK_M_LOGICAL),
                    dtype=nl.float8_e8m0fnu,
                    buffer=data_buffer,
                ),
                is_swizzled=True,
                is_x4=True,
            )
        if not rhs_td.is_quantized:
            rhsq_td = TensorDescriptor(
                data=nl.ndarray(
                    (shapes['BLOCKS_IN_K'] * BLOCK_K_SIZE, BLOCKS_IN_N_sharded * bd.BLOCK_N_LOGICAL),
                    dtype=fp8_x4_dtype,
                    buffer=data_buffer,
                ),
                scales=nl.ndarray(
                    (shapes['BLOCKS_IN_K'] * BLOCK_K_SIZE, BLOCKS_IN_N_sharded * bd.BLOCK_N_LOGICAL),
                    dtype=nl.float8_e8m0fnu,
                    buffer=data_buffer,
                ),
                is_swizzled=True,
                is_x4=True,
            )

    if output_to_sbuf:
        output_sbuf = sbm.alloc_stack(
            shape=(LHS_MATMUL_TILE_M, bd.TILES_IN_BLOCK_M, bd.BLOCK_N_LOGICAL),
            dtype=output_dtype,
            buffer=nl.sbuf,
        )
        out_td = _SbufOutputTd(output_sbuf)
    else:
        out_td = TensorDescriptor(data=output_sharded)

    generic_matmul_mxfp8_api(
        lhs_hbm_td=lhs_td,
        rhs_hbm_td=rhs_td,
        bd=bd,
        output_td=out_td,
        output_dtype=output_dtype,
        TILES_IN_LOAD_M=TILES_IN_LOAD_M,
        TILES_IN_LOAD_N=TILES_IN_LOAD_N,
        block_idx_m=(0, 1),
        block_idx_n=(0, 1),
        block_idx_k=(0, shapes['BLOCKS_IN_K']),
        lhs_matmul_tile_shape_physical=shapes['lhs_matmul_tile_shape_physical'],
        rhs_matmul_tile_shape_physical=shapes['rhs_matmul_tile_shape_physical'],
        lhs_load_tile_shape=shapes['lhs_load_tile_shape'],
        rhs_load_tile_shape=shapes['rhs_load_tile_shape'],
        lhs_quantize_tile_shape=shapes['lhs_quantize_tile_shape'],
        rhs_quantize_tile_shape=shapes['rhs_quantize_tile_shape'],
        tile_loop_order=tile_loop_order,
        float8_dtype=float8_dtype,
        use_scale_packing=use_scale_packing,
        spill_reload=spill_reload,
        lhsq_td=lhsq_td,
        rhsq_td=rhsq_td,
        rhs_n_offset=rhs_n_offset,
    )

    if output_to_sbuf:
        _store_sbuf_to_hbm(output_sbuf, output_sharded, bd, LHS_MATMUL_TILE_M)

    sbm.close_scope()

    return output_hbm


# ---------------------------------------------------------------------------
# SBUF preloaded: Caller pre-loads SBUF, API does single-block matmul -> HBM & SBUF
#   Caller loads + quantizes into SBUF, passes via lhs_sbuf_td/rhs_sbuf_td.
#   output_to_sbuf is passed as a kernel parameter.
# ---------------------------------------------------------------------------


@with_active_sbm
def _kernel_sbuf_preloaded(
    lhs,
    rhs,
    output_dtype,
    run_with_lnc2,
    float8_dtype,
    use_scale_packing,
    tile_loop_order,
    lhs_scales,
    rhs_scales,
    TILES_IN_BLOCK_M=None,
    TILES_IN_BLOCK_N=None,
    TILES_IN_BLOCK_K=None,
    TILES_IN_LOAD_M=None,
    TILES_IN_LOAD_N=None,
    lhs_matmul_tile_shape_logical=None,
    rhs_matmul_tile_shape_logical=None,
    lhs_is_swizzled=True,
    rhs_is_swizzled=True,
    block_loop_order=None,
    spill_reload=False,
    output_to_sbuf=False,
    lhs_load_with_PE_swizzle=False,
    rhs_load_with_PE_swizzle=False,
    lhs_is_f_by_k=None,
    rhs_is_f_by_k=None,
    enable_psum_copy_in=None,
    quant_scheme="wrapX",
):
    create_and_set_active_sbm()
    sbm = get_active_sbm()
    sbm.open_scope("TEST API")

    lhs_td, rhs_td, shapes, TILES_IN_LOAD_M, TILES_IN_LOAD_N = _setup(
        lhs=lhs,
        rhs=rhs,
        lhs_scales=lhs_scales,
        rhs_scales=rhs_scales,
        lhs_is_swizzled=lhs_is_swizzled,
        rhs_is_swizzled=rhs_is_swizzled,
        run_with_lnc2=run_with_lnc2,
        use_scale_packing=use_scale_packing,
        TILES_IN_BLOCK_M=TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N=TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K=TILES_IN_BLOCK_K,
        TILES_IN_LOAD_M=TILES_IN_LOAD_M,
        TILES_IN_LOAD_N=TILES_IN_LOAD_N,
        lhs_matmul_tile_shape_logical=lhs_matmul_tile_shape_logical,
        rhs_matmul_tile_shape_logical=rhs_matmul_tile_shape_logical,
    )
    bd = shapes['bd']
    LHS_MATMUL_TILE_M = shapes['lhs_matmul_tile_shape_physical'][1]
    output_hbm, output_sharded, rhs_n_offset, _ = _alloc_output_and_lnc(
        lhs_td,
        rhs_td,
        shapes,
        run_with_lnc2,
        output_dtype,
    )

    # Load block (m=0, k=0, n=0) from HBM to SBUF
    (
        lhs_loaded,
        rhs_loaded,
        lhs_data_loaded,
        lhs_scales_loaded,
        rhs_data_loaded,
        rhs_scales_loaded,
    ) = load_block.load_lhs_and_rhs(
        lhs_td=lhs_td,
        rhs_td=rhs_td,
        TILES_IN_LOAD_M=TILES_IN_LOAD_M,
        TILES_IN_LOAD_N=TILES_IN_LOAD_N,
        lhs_load_tile_shape=shapes['lhs_load_tile_shape'],
        rhs_load_tile_shape=shapes['rhs_load_tile_shape'],
        block_idx=(0, 0, 0),
        bd=bd,
        rhs_n_offset=rhs_n_offset,
    )

    # Quantize if BF16 input
    if lhs_loaded is not None:
        lhs_data_loaded, lhs_scales_loaded = quantize_mxfp8_block.quantize_mxfp8_block(
            lhs_loaded,
            shapes['lhs_quantize_tile_shape'],
            True,
            float8_dtype,
        )
    if rhs_loaded is not None:
        rhs_data_loaded, rhs_scales_loaded = quantize_mxfp8_block.quantize_mxfp8_block(
            rhs_loaded,
            shapes['rhs_quantize_tile_shape'],
            True,
            float8_dtype,
        )

    lhs_sbuf_td = TensorDescriptor(
        data=lhs_data_loaded,
        scales=lhs_scales_loaded,
        is_swizzled=True,
        is_x4=True,
    )
    rhs_sbuf_td = TensorDescriptor(
        data=rhs_data_loaded,
        scales=rhs_scales_loaded,
        is_swizzled=True,
        is_x4=True,
    )

    if output_to_sbuf:
        output_sbuf = sbm.alloc_stack(
            shape=(LHS_MATMUL_TILE_M, bd.TILES_IN_BLOCK_M, bd.BLOCK_N_LOGICAL),
            dtype=output_dtype,
            buffer=nl.sbuf,
        )
        out_td = _SbufOutputTd(output_sbuf)
    else:
        out_td = TensorDescriptor(data=output_sharded)

    generic_matmul_mxfp8_api(
        lhs_hbm_td=lhs_td,
        rhs_hbm_td=rhs_td,
        bd=bd,
        output_td=out_td,
        output_dtype=output_dtype,
        TILES_IN_LOAD_M=TILES_IN_LOAD_M,
        TILES_IN_LOAD_N=TILES_IN_LOAD_N,
        block_idx_m=(0, 1),
        block_idx_n=(0, 1),
        block_idx_k=(0, 1),
        lhs_matmul_tile_shape_physical=shapes['lhs_matmul_tile_shape_physical'],
        rhs_matmul_tile_shape_physical=shapes['rhs_matmul_tile_shape_physical'],
        lhs_load_tile_shape=shapes['lhs_load_tile_shape'],
        rhs_load_tile_shape=shapes['rhs_load_tile_shape'],
        lhs_quantize_tile_shape=shapes['lhs_quantize_tile_shape'],
        rhs_quantize_tile_shape=shapes['rhs_quantize_tile_shape'],
        tile_loop_order=tile_loop_order,
        float8_dtype=float8_dtype,
        use_scale_packing=use_scale_packing,
        lhs_sbuf_td=lhs_sbuf_td,
        rhs_sbuf_td=rhs_sbuf_td,
        rhs_n_offset=rhs_n_offset,
    )

    if output_to_sbuf:
        _store_sbuf_to_hbm(output_sbuf, output_sharded, bd, LHS_MATMUL_TILE_M)

    sbm.close_scope()

    return output_hbm


# ---------------------------------------------------------------------------
# SBUF empty TD fill: Pass empty TDs, API loads from HBM and fills them.
#   Verifies that the API correctly populates empty sbuf_td with loaded data.
# ---------------------------------------------------------------------------


@with_active_sbm
def _kernel_sbuf_empty_td_fill(
    lhs,
    rhs,
    output_dtype,
    run_with_lnc2,
    float8_dtype,
    use_scale_packing,
    tile_loop_order,
    lhs_scales,
    rhs_scales,
    TILES_IN_BLOCK_M=None,
    TILES_IN_BLOCK_N=None,
    TILES_IN_BLOCK_K=None,
    TILES_IN_LOAD_M=None,
    TILES_IN_LOAD_N=None,
    lhs_matmul_tile_shape_logical=None,
    rhs_matmul_tile_shape_logical=None,
    lhs_is_swizzled=True,
    rhs_is_swizzled=True,
    block_loop_order=None,
    spill_reload=False,
    output_to_sbuf=False,
    lhs_load_with_PE_swizzle=False,
    rhs_load_with_PE_swizzle=False,
    lhs_is_f_by_k=None,
    rhs_is_f_by_k=None,
    enable_psum_copy_in=None,
    quant_scheme="wrapX",
):
    """Pass empty TDs, API loads and fills them. Then use the filled TDs
    in a second API call (which skips loading). If the second call produces
    correct matmul output, the TDs were filled correctly."""
    create_and_set_active_sbm()
    sbm = get_active_sbm()
    sbm.open_scope("TEST API")

    lhs_td, rhs_td, shapes, TILES_IN_LOAD_M, TILES_IN_LOAD_N = _setup(
        lhs=lhs,
        rhs=rhs,
        lhs_scales=lhs_scales,
        rhs_scales=rhs_scales,
        lhs_is_swizzled=lhs_is_swizzled,
        rhs_is_swizzled=rhs_is_swizzled,
        run_with_lnc2=run_with_lnc2,
        use_scale_packing=use_scale_packing,
        TILES_IN_BLOCK_M=TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N=TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K=TILES_IN_BLOCK_K,
        TILES_IN_LOAD_M=TILES_IN_LOAD_M,
        TILES_IN_LOAD_N=TILES_IN_LOAD_N,
        lhs_matmul_tile_shape_logical=lhs_matmul_tile_shape_logical,
        rhs_matmul_tile_shape_logical=rhs_matmul_tile_shape_logical,
    )
    bd = shapes['bd']
    output_hbm, output_sharded, rhs_n_offset, _ = _alloc_output_and_lnc(
        lhs_td,
        rhs_td,
        shapes,
        run_with_lnc2,
        output_dtype,
    )

    # First call: pass empty TDs — API loads from HBM and fills them
    lhs_sbuf_td = TensorDescriptor()
    rhs_sbuf_td = TensorDescriptor()

    # Use a throwaway output for the first call
    throwaway_hbm = nl.ndarray(
        (lhs_td.logical_shape[1], rhs_td.logical_shape[1]),
        dtype=output_dtype,
        buffer=nl.shared_hbm,
    )
    generic_matmul_mxfp8_api(
        lhs_hbm_td=lhs_td,
        rhs_hbm_td=rhs_td,
        bd=bd,
        output_td=TensorDescriptor(data=throwaway_hbm),
        output_dtype=output_dtype,
        TILES_IN_LOAD_M=TILES_IN_LOAD_M,
        TILES_IN_LOAD_N=TILES_IN_LOAD_N,
        block_idx_m=(0, 1),
        block_idx_n=(0, 1),
        block_idx_k=(0, 1),
        lhs_matmul_tile_shape_physical=shapes['lhs_matmul_tile_shape_physical'],
        rhs_matmul_tile_shape_physical=shapes['rhs_matmul_tile_shape_physical'],
        lhs_load_tile_shape=shapes['lhs_load_tile_shape'],
        rhs_load_tile_shape=shapes['rhs_load_tile_shape'],
        lhs_quantize_tile_shape=shapes['lhs_quantize_tile_shape'],
        rhs_quantize_tile_shape=shapes['rhs_quantize_tile_shape'],
        tile_loop_order=tile_loop_order,
        float8_dtype=float8_dtype,
        use_scale_packing=use_scale_packing,
        lhs_sbuf_td=lhs_sbuf_td,
        rhs_sbuf_td=rhs_sbuf_td,
        rhs_n_offset=rhs_n_offset,
    )

    # Second call: pass the now-filled TDs with POISONED HBM data.
    # We create zero-filled HBM tensors with the same shape. If the API
    # incorrectly loads from HBM instead of using the cached SBUF TDs,
    # the output will be all zeros (wrong). If it correctly skips loading
    # and uses the filled TDs, the output matches the original golden.
    poison_lhs = nl.ndarray(lhs_td.data.shape, dtype=lhs_td.data.dtype, buffer=nl.hbm)
    poison_rhs = nl.ndarray(rhs_td.data.shape, dtype=rhs_td.data.dtype, buffer=nl.hbm)
    poison_lhs_td = TensorDescriptor(data=poison_lhs, scales=lhs_scales, is_swizzled=lhs_is_swizzled)
    poison_rhs_td = TensorDescriptor(data=poison_rhs, scales=rhs_scales, is_swizzled=rhs_is_swizzled)

    generic_matmul_mxfp8_api(
        lhs_hbm_td=poison_lhs_td,
        rhs_hbm_td=poison_rhs_td,
        bd=bd,
        output_td=TensorDescriptor(data=output_sharded),
        output_dtype=output_dtype,
        TILES_IN_LOAD_M=TILES_IN_LOAD_M,
        TILES_IN_LOAD_N=TILES_IN_LOAD_N,
        block_idx_m=(0, 1),
        block_idx_n=(0, 1),
        block_idx_k=(0, 1),
        lhs_matmul_tile_shape_physical=shapes['lhs_matmul_tile_shape_physical'],
        rhs_matmul_tile_shape_physical=shapes['rhs_matmul_tile_shape_physical'],
        lhs_load_tile_shape=shapes['lhs_load_tile_shape'],
        rhs_load_tile_shape=shapes['rhs_load_tile_shape'],
        lhs_quantize_tile_shape=shapes['lhs_quantize_tile_shape'],
        rhs_quantize_tile_shape=shapes['rhs_quantize_tile_shape'],
        tile_loop_order=tile_loop_order,
        float8_dtype=float8_dtype,
        use_scale_packing=use_scale_packing,
        lhs_sbuf_td=lhs_sbuf_td,
        rhs_sbuf_td=rhs_sbuf_td,
        rhs_n_offset=rhs_n_offset,
    )

    sbm.close_scope()

    return output_hbm


# ---------------------------------------------------------------------------
# LHS M-sharded: LNC2 shards on M (LHS) instead of N (RHS).
#   Each core processes half the M rows, full N columns.
# ---------------------------------------------------------------------------


@with_active_sbm
def _kernel_lhs_m_sharded(
    lhs,
    rhs,
    output_dtype,
    run_with_lnc2,
    float8_dtype,
    use_scale_packing,
    tile_loop_order,
    lhs_scales,
    rhs_scales,
    TILES_IN_BLOCK_M=None,
    TILES_IN_BLOCK_N=None,
    TILES_IN_BLOCK_K=None,
    TILES_IN_LOAD_M=None,
    TILES_IN_LOAD_N=None,
    lhs_matmul_tile_shape_logical=None,
    rhs_matmul_tile_shape_logical=None,
    lhs_is_swizzled=True,
    rhs_is_swizzled=True,
    block_loop_order=None,
    spill_reload=False,
    output_to_sbuf=False,
    lhs_load_with_PE_swizzle=False,
    rhs_load_with_PE_swizzle=False,
    lhs_is_f_by_k=None,
    rhs_is_f_by_k=None,
    enable_psum_copy_in=None,
    quant_scheme="wrapX",
):
    """M-sharded LNC2: LHS is col_parallel_sharded (halves M), RHS is full."""
    create_and_set_active_sbm()
    sbm = get_active_sbm()
    sbm.open_scope("TEST API")

    # M-sharding: LHS gets is_col_parallel_sharded, RHS does not
    lhs_td = TensorDescriptor(data=lhs, scales=lhs_scales, is_swizzled=lhs_is_swizzled, is_col_parallel_sharded=True)
    rhs_td = TensorDescriptor(data=rhs, scales=rhs_scales, is_swizzled=rhs_is_swizzled, is_col_parallel_sharded=False)

    K_logical, M_logical = lhs_td.sharded_logical_shape
    _, N_logical = rhs_td.sharded_logical_shape
    lhs_precision = (
        PRECISION_BFLOAT16 if not lhs_td.is_quantized else (PRECISION_MXFP8_X4 if lhs_td.is_x4 else PRECISION_MXFP8)
    )
    rhs_precision = (
        PRECISION_BFLOAT16 if not rhs_td.is_quantized else (PRECISION_MXFP8_X4 if rhs_td.is_x4 else PRECISION_MXFP8)
    )

    config = MatmulMxfp8KernelConfig(
        M=M_logical,
        K=K_logical,
        N=N_logical,
        tile_m=lhs_matmul_tile_shape_logical[1] if lhs_matmul_tile_shape_logical else None,
        tile_k=lhs_matmul_tile_shape_logical[0] if lhs_matmul_tile_shape_logical else None,
        tile_n=rhs_matmul_tile_shape_logical[1] if rhs_matmul_tile_shape_logical else None,
        TILES_IN_BLOCK_M=TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N=TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K=TILES_IN_BLOCK_K,
        TILES_IN_LOAD_M=TILES_IN_LOAD_M,
        TILES_IN_LOAD_N=TILES_IN_LOAD_N,
        enable_scale_packing=use_scale_packing,
        run_with_lnc2=True,
        lnc_2_shard_rhs=False,
        lhs_is_swizzled=lhs_is_swizzled,
        rhs_is_swizzled=rhs_is_swizzled,
    )
    auto_generate_default(config, lhs_precision, rhs_precision, PRECISION_FP32)
    validate_shapes(config, lhs_td, rhs_td)

    shapes = {
        'bd': config.bd,
        'BLOCKS_IN_M': config.BLOCKS_IN_M,
        'BLOCKS_IN_N': config.BLOCKS_IN_N,
        'BLOCKS_IN_K': config.BLOCKS_IN_K,
        'lhs_matmul_tile_shape_physical': config.lhs_matmul_tile_shape_physical,
        'rhs_matmul_tile_shape_physical': config.rhs_matmul_tile_shape_physical,
        'lhs_load_tile_shape': config.lhs_load_tile_shape,
        'rhs_load_tile_shape': config.rhs_load_tile_shape,
        'lhs_quantize_tile_shape': config.lhs_quantize_tile_shape,
        'rhs_quantize_tile_shape': config.rhs_quantize_tile_shape,
    }
    bd = shapes['bd']
    TILES_IN_LOAD_M = config.TILES_IN_LOAD_M
    TILES_IN_LOAD_N = config.TILES_IN_LOAD_N

    # Output: shared HBM, full [M, N]
    M_LOGICAL = lhs_td.logical_shape[1]
    N_LOGICAL = rhs_td.logical_shape[1]
    output_hbm = nl.ndarray((M_LOGICAL, N_LOGICAL), dtype=output_dtype, buffer=nl.shared_hbm)

    # Per-core M sharding
    M_LOGICAL_SHARDED = lhs_td.sharded_logical_shape[1]
    M_PHYSICAL_SHARDED = lhs_td.sharded_physical_shape[1]
    LNC_ID = nl.program_id(axis=0)
    out_row = LNC_ID * M_LOGICAL_SHARDED
    output_sharded = output_hbm[out_row : out_row + M_LOGICAL_SHARDED, :]
    lhs_m_offset = LNC_ID * M_PHYSICAL_SHARDED
    BLOCKS_IN_M_sharded = (shapes['BLOCKS_IN_M'] + 1) // 2

    generic_matmul_mxfp8_api(
        lhs_hbm_td=lhs_td,
        rhs_hbm_td=rhs_td,
        bd=bd,
        output_td=TensorDescriptor(data=output_sharded),
        output_dtype=output_dtype,
        TILES_IN_LOAD_M=TILES_IN_LOAD_M,
        TILES_IN_LOAD_N=TILES_IN_LOAD_N,
        block_idx_m=(0, BLOCKS_IN_M_sharded),
        block_idx_n=(0, shapes['BLOCKS_IN_N']),
        block_idx_k=(0, shapes['BLOCKS_IN_K']),
        lhs_matmul_tile_shape_physical=shapes['lhs_matmul_tile_shape_physical'],
        rhs_matmul_tile_shape_physical=shapes['rhs_matmul_tile_shape_physical'],
        lhs_load_tile_shape=shapes['lhs_load_tile_shape'],
        rhs_load_tile_shape=shapes['rhs_load_tile_shape'],
        lhs_quantize_tile_shape=shapes['lhs_quantize_tile_shape'],
        rhs_quantize_tile_shape=shapes['rhs_quantize_tile_shape'],
        tile_loop_order=tile_loop_order,
        float8_dtype=float8_dtype,
        use_scale_packing=use_scale_packing,
        lhs_m_offset=lhs_m_offset,
        rhs_n_offset=0,
    )

    sbm.close_scope()

    return output_hbm


# ---------------------------------------------------------------------------
# Test grids
# ---------------------------------------------------------------------------

# Single M/N block, multiple K blocks.
# Explicit tile configs to guarantee BLOCKS_IN_M=1, BLOCKS_IN_N=1, BLOCKS_IN_K>1.
# BLOCK_M = TILES_IN_BLOCK_M * tile_m = 8*128 = 1024 >= M
# BLOCK_N = TILES_IN_BLOCK_N * tile_n = 2*512 = 1024 >= N
# BLOCK_K = TILES_IN_BLOCK_K * tile_k = 2*512 = 1024 < K=2048 -> BLOCKS_IN_K=2
GRID_HBM_K_LOOP = [
    config_helper.TestConfig(
        M=1024,
        K=2048,
        N=1024,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_M=8,
        TILES_IN_BLOCK_N=2,
        TILES_IN_BLOCK_K=2,
        TILES_IN_LOAD_M=8,
        TILES_IN_LOAD_N=2,
        run_with_lnc2=False,
        output_dtype=constants.MatrixPrecision.FP32,
        float8_dtype="float8_e4m3fn",
        dists=["normal", "normal"],
        params=[{"mean": 0, "std": 1}, {"mean": 0, "std": 1}],
        description="single M/N block, 2 K blocks",
        seed=52,
    ),
]

# Single block in all dimensions.
# BLOCK_K = 2*512 = 1024 >= K
GRID_SBUF_PRELOADED = [
    config_helper.TestConfig(
        M=1024,
        K=1024,
        N=1024,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_M=8,
        TILES_IN_BLOCK_N=2,
        TILES_IN_BLOCK_K=2,
        TILES_IN_LOAD_M=8,
        TILES_IN_LOAD_N=2,
        run_with_lnc2=False,
        float8_dtype="float8_e4m3fn",
        output_dtype=constants.MatrixPrecision.FP32,
        dists=["normal", "normal"],
        params=[{"mean": 0, "std": 1}, {"mean": 0, "std": 1}],
        description="single block all dims",
        seed=52,
    ),
]

# M-sharded LNC2: M=1024 with BLOCK_M=4*128=512 -> BLOCKS_IN_M=2, one per core.
# Full N and K dimensions per core.
GRID_LHS_M_SHARDED = [
    config_helper.TestConfig(
        M=1024,
        K=1024,
        N=1024,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=2,
        TILES_IN_BLOCK_K=2,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=2,
        run_with_lnc2=True,
        output_dtype=constants.MatrixPrecision.FP32,
        float8_dtype="float8_e4m3fn",
        dists=["normal", "normal"],
        params=[{"mean": 0, "std": 1}, {"mean": 0, "std": 1}],
        description="M-sharded LNC2, 2 M blocks",
        seed=52,
    ),
]


# PE swizzle mixed: LHS uses PE swizzle, RHS uses DGT (both unswizzled)
GRID_PE_SWIZZLE_MIXED = [
    config_helper.TestConfig(
        M=512,
        K=1024,
        N=512,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_BLOCK_K=2,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        run_with_lnc2=False,
        output_dtype=constants.MatrixPrecision.FP32,
        float8_dtype="float8_e4m3fn",
        dists=["normal", "normal"],
        params=[{"mean": 0, "std": 1}, {"mean": 0, "std": 1}],
        description="LHS PE swizzle, RHS DGT (both unswizzled)",
        seed=52,
    ),
]

# K-by-F: LHS in [K, M] layout (PE swizzle auto-forced)
GRID_K_BY_F_API = [
    config_helper.TestConfig(
        M=512,
        K=1024,
        N=512,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_BLOCK_K=2,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        run_with_lnc2=False,
        output_dtype=constants.MatrixPrecision.FP32,
        float8_dtype="float8_e4m3fn",
        dists=["normal", "normal"],
        params=[{"mean": 0, "std": 1}, {"mean": 0, "std": 1}],
        description="K-by-F: both sides via API",
        seed=52,
        lhs_is_f_by_k=False,
        rhs_is_f_by_k=False,
    ),
    # K-by-F with LNC2 (RHS col-parallel sharded on N). N=1024 -> 2 N-blocks, shardable.
    config_helper.TestConfig(
        M=512,
        K=1024,
        N=1024,
        tile_m=128,
        tile_n=512,
        tile_k=512,
        TILES_IN_BLOCK_M=4,
        TILES_IN_BLOCK_N=1,
        TILES_IN_BLOCK_K=2,
        TILES_IN_LOAD_M=4,
        TILES_IN_LOAD_N=1,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        run_with_lnc2=True,
        output_dtype=constants.MatrixPrecision.FP32,
        float8_dtype="float8_e4m3fn",
        dists=["normal", "normal"],
        params=[{"mean": 0, "std": 1}, {"mean": 0, "std": 1}],
        description="K-by-F: both sides via API, LNC2",
        seed=52,
        lhs_is_f_by_k=False,
        rhs_is_f_by_k=False,
    ),
]

# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


def _api_torch_ref(
    lhs,
    rhs,
    output_dtype,
    run_with_lnc2,
    float8_dtype,
    use_scale_packing,
    tile_loop_order,
    lhs_scales,
    rhs_scales,
    TILES_IN_BLOCK_M=None,
    TILES_IN_BLOCK_N=None,
    TILES_IN_BLOCK_K=None,
    TILES_IN_LOAD_M=None,
    TILES_IN_LOAD_N=None,
    lhs_matmul_tile_shape_logical=None,
    rhs_matmul_tile_shape_logical=None,
    lhs_is_swizzled=True,
    rhs_is_swizzled=True,
    block_loop_order=None,
    spill_reload=False,
    output_to_sbuf=False,
    lhs_load_with_PE_swizzle=False,
    rhs_load_with_PE_swizzle=False,
    lhs_is_f_by_k=None,
    rhs_is_f_by_k=None,
    enable_psum_copy_in=None,
    quant_scheme="wrapX",
):
    """Torch ref for API test kernels — delegates to matmul_mxfp8_torch_ref."""
    return matmul_mxfp8_torch_ref(
        lhs=lhs,
        rhs=rhs,
        TILES_IN_BLOCK_M=TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N=TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K=TILES_IN_BLOCK_K,
        TILES_IN_LOAD_M=TILES_IN_LOAD_M,
        TILES_IN_LOAD_N=TILES_IN_LOAD_N,
        lhs_matmul_tile_shape_logical=lhs_matmul_tile_shape_logical,
        rhs_matmul_tile_shape_logical=rhs_matmul_tile_shape_logical,
        block_loop_order=block_loop_order,
        tile_loop_order=tile_loop_order,
        float8_dtype=float8_dtype,
        output_dtype=output_dtype,
        run_with_lnc2=run_with_lnc2,
        lhs_scales=lhs_scales,
        rhs_scales=rhs_scales,
        use_scale_packing=use_scale_packing,
        spill_reload=spill_reload,
        lhs_is_swizzled=lhs_is_swizzled,
        rhs_is_swizzled=rhs_is_swizzled,
        lhs_is_f_by_k=lhs_is_f_by_k,
        rhs_is_f_by_k=rhs_is_f_by_k,
    )


@pytest_test_metadata(name="Generic Matmul MXFP8 API Data Paths")
@pytest_marks(["generic_matmul_mxfp8_api", "mx", "mxfp8"])
@pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN1, common_dataclasses.Platforms.TRN2])
class TestGenericMatmulMxfp8Api:
    """Tests for matmul_mxfp8_generic_api exercising HBM K-loop, SBUF preloaded,
    empty TD fill, and LHS M-sharded data paths."""

    def _run(self, test_manager, platform_target, conf, kernel_func, extra_kernel_args=None):
        if os.environ.get('TEST_COLLECTION_BENCHMARK') == '1':
            pytest.skip("Benchmark mode - skipping test execution")
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")

        test_manager.collector.set_kernel_params(conf.to_metrics_dict())
        output_dtype = get_output_dtype(conf)
        kernel_input = build_matmul_inputs(conf)
        if extra_kernel_args:
            kernel_input.update(extra_kernel_args)

        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=2 if conf.run_with_lnc2 else 1,
            platform_target=platform_target,
        )

        custom_validation_args = _mxfp8_comparator(conf, output_dtype)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_func,
            torch_ref=_api_torch_ref,
            kernel_input_generator=lambda _: kernel_input,
            output_tensor_descriptor=lambda _: {"out": np.zeros((conf.M, conf.N), dtype=output_dtype)},
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            custom_comparator=custom_validation_args,
        )

    # --- HBM K-loop: Load from HBM, API handles K loop (single M/N block) -> HBM/SBUF
    @pytest.mark.fast
    @pytest.mark.parametrize("conf", GRID_HBM_K_LOOP)
    @pytest.mark.parametrize("spill_reload", [False, True], ids=["no_spill", "spill"])
    @pytest.mark.parametrize("output_to_sbuf", [False, True], ids=["to_hbm", "to_sbuf"])
    def test_hbm_loop_k(self, test_manager, conf, spill_reload, output_to_sbuf, platform_target):
        self._run(
            test_manager,
            platform_target,
            conf,
            _kernel_hbm_k_loop,
            extra_kernel_args={"output_to_sbuf": output_to_sbuf, "spill_reload": spill_reload},
        )

    # --- SBUF preloaded: Caller pre-loads SBUF, API does single-block matmul -> HBM & SBUF
    @pytest.mark.fast
    @pytest.mark.parametrize("conf", GRID_SBUF_PRELOADED)
    @pytest.mark.parametrize("output_to_sbuf", [False, True], ids=["to_hbm", "to_sbuf"])
    def test_sbuf_single_block(self, test_manager, conf, output_to_sbuf, platform_target):
        self._run(
            test_manager,
            platform_target,
            conf,
            _kernel_sbuf_preloaded,
            extra_kernel_args={"output_to_sbuf": output_to_sbuf},
        )

    # --- Empty TD fill: Pass empty TDs, API loads and fills them -> HBM ---
    @pytest.mark.fast
    @pytest.mark.parametrize("conf", GRID_SBUF_PRELOADED)
    def test_empty_td_fill(self, test_manager, conf, platform_target):
        self._run(test_manager, platform_target, conf, _kernel_sbuf_empty_td_fill)

    # --- LHS M-sharded: LNC2 shards on M dimension via lhs_m_offset ---
    @pytest.mark.fast
    @pytest.mark.parametrize("conf", GRID_LHS_M_SHARDED)
    def test_lhs_m_sharded(self, test_manager, conf, platform_target):
        self._run(test_manager, platform_target, conf, _kernel_lhs_m_sharded)

    # --- PE swizzle mixed: LHS PE swizzle, RHS DGT (both unswizzled) ---
    @pytest.mark.fast
    @pytest.mark.parametrize("conf", GRID_PE_SWIZZLE_MIXED)
    def test_pe_swizzle_mixed(self, test_manager, conf, platform_target):
        self._run(
            test_manager,
            platform_target,
            conf,
            _kernel_hbm_k_loop,
            extra_kernel_args={"lhs_load_with_PE_swizzle": True, "rhs_load_with_PE_swizzle": False},
        )

    # --- K-by-F: Input in [K, F] layout, PE swizzle auto-forced ---
    @pytest.mark.fast
    @pytest.mark.parametrize("conf", GRID_K_BY_F_API)
    def test_k_by_f(self, test_manager, conf, platform_target):
        self._run(
            test_manager,
            platform_target,
            conf,
            _kernel_hbm_k_loop,
            extra_kernel_args={"lhs_is_f_by_k": False, "rhs_is_f_by_k": False},
        )
