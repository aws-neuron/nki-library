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

"""Qwen3-235B-A22B backward perf through the FUSED backward kernel.

test_matmul_mxfp8_model_configs.py sweeps the backward shapes as two independent
matmuls; this drives matmul_mxfp8_backward directly, so numbers include the fused
kernel's own scheduling of dX = dY @ W and dW = dY^T @ X and its reuse of dY across
both phases.
The BF16 comparison uses a measured fused baseline for the same reason (one graph
sharing dY, not two isolated matmuls); it lives in bf16_backward_baseline_cache.json.

All five of the forward suite's arms are reachable:
  - BF16 inputs, prequant weight [wrapX], prequantized: covered.
  - PE-swizzle 1x32: covered -- the 1x32 PE-swizzle loader now transposes K-by-F
    unswizzled operands as well as F-by-K, so every backward phase quantizes its
    BF16 operands on the fly with the 1x32 group layout.
  - Prequant weight [1x32]: covered -- same single 1x32 quant_scheme, but W is stored
    pre-quantized (dX loads dY on the fly against a pre-quantized W); dW loads dY and X
    on the fly, since W is not a dW operand. Mirrors the prequant-weight-wrapx arm.
  - Prequantized: dY is necessarily BF16 (the two phases contract it over different
    axes so it can be swizzled only one way; matches production, where dY is fresh).

Plus a "prequantized inputs" arm (BOTH W and X pre-quantized, 1x32) with a DMA-transpose
ablation: test_bwd_tp4_prequantized_1x32 (dY transposed via a DMA transpose) and
test_bwd_tp4_prequantized_1x32_no_dma_transpose (disable_dma_transpose=True -> dY via
dma_copy + nc_transpose). Both W (dX RHS) and X (dW RHS) load pre-quantized 1x32, so dY is
the only on-the-fly operand and the sole transpose in dW; the ablation isolates its cost.

TP4 shapes: O-proj is 4096x2048x4096 (K = Q*h/TP = 2048), fused QKV is 4096x4096x2304.
Both projections' backward are reachable: the K-by-F PE-swizzle loader now masks
partial sub-tiles down to MIN_F_FOR_QUANTIZATION, so the QKV dW transpose of N=2304
(2304 % 8 == 0) compiles rather than hitting the old F % 512 constraint.

1x32 operand orientation (perf-table columns G/H vs I/J):
  W and X are passed pre-transposed (F-by-K) so the 1x32 loader reads them with a plain
  dma_copy and dY is the only transposed operand. The two 1x32 group pairs are:
    - test_bwd_tp4_pe_swizzle_1x32 / _prequant_weight_1x32          -> G/H (dY via DMA transpose)
    - test_bwd_tp4_pe_swizzle_1x32_no_dma_transpose / _..._no_dma_transpose -> I/J
      (disable_dma_transpose=True: dY via dma_copy + nc_transpose, no DMA transpose)

Retuning the cache for these arms (both routes are cached separately):
  The autotune key appends "_nodma" for the no-DMA route, so DMA and no-DMA each keep
  their own tuned tiling. To re-tune, sweep test_matmul_mxfp8_autotune_sweep.py with
  AUTOTUNE_SHAPES set to the per-phase shapes and AUTOTUNE_METHODS=pe_1x32 (dX),
  dw_pe_1x32 (dW DMA), dw_pe_1x32_no_dma (dW no-DMA), then fold the winners in with
  autotune_update_cache.py <output-dir>. Non-power-of-two K (e.g. QKV dX K=2304) also
  sweeps exact-divisor tile_k (256/384), which the fixed tile_k=512 cannot tile evenly.
"""

import neuron_dtypes as dtype
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.matmul_mxfp8 import matmul_mxfp8_generic_backward_kernel

from test.integration.nkilib.experimental.matmul_mxfp8 import (
    config_helper,
    constants,
    model_config_reader,
)
from test.integration.nkilib.experimental.matmul_mxfp8 import utils as matmul_utils
from test.integration.nkilib.experimental.matmul_mxfp8.backward_config_helper import (
    make_backward_configs,
)
from test.integration.nkilib.experimental.matmul_mxfp8.test_input_microbenchmarks import (
    _make_lhs_1x32_pe_swizzle_rhs_1x32_prequantized,
    _make_pe_swizzle_1x32,
)
from test.integration.nkilib.experimental.matmul_mxfp8.test_matmul_mxfp8_generic_backward_kernel import (
    _bwd_comparator,
    matmul_mxfp8_backward_torch_ref,
)
from test.integration.nkilib.experimental.matmul_mxfp8.test_qwen235b_perf import (
    _make_bf16_inputs_fast_dgt,
    _make_bf16_inputs_pe_wrapx,
)
from test.integration.nkilib.experimental.quantize_mxfp8.test_quantize_mxfp8_utils import (
    Q_TILE_K,
    generate_golden_packed_scales,
)
from test.utils import common_dataclasses
from test.utils.mx_utils import quantize_mx_golden
from test.utils.pytest_test_metadata import pytest_marks
from test.utils.unit_test_framework import UnitTestFramework

# B1-only: every reachable arm loads dY via fast_dma_transpose, a B1-only instruction
# with no arch guard -- on A0 it faults the device and can take the node down, so these
# suites MUST never be collected on A0. TRN3 / TRN3_PDS are the B1 platforms.
_TRN3_B1_PLATFORMS = {common_dataclasses.Platforms.TRN3, common_dataclasses.Platforms.TRN3_PDS}

# ============================================================================
# Shape selection
# ============================================================================

# Only the projections the generic matmul kernel owns in production. MoE expert
# MLP and core attention are served by the MoE and flash attention kernels.
_MATMUL_KERNEL_PROJECTIONS = ("FuseQKV", "OProj")


def _forward_projection_shapes(configs):
    """Extract the FORWARD (M, K, N, description) projection shapes.

    Backward is derived from the forward shape in BackwardTestConfig, so only FWD-*
    entries are taken; the generator's BWD-* entries would double-count.
    """
    return [
        (c.M, c.K, c.N, c.description)
        for c in configs
        if "FWD-" in c.description and any(p in c.description for p in _MATMUL_KERNEL_PROJECTIONS)
    ]


_TP_SHAPES = {
    4: _forward_projection_shapes(model_config_reader.qwen3_235b_tp4),
}


# ============================================================================
# Per-phase config factories for the three reachable input use cases
# ============================================================================


def _make_bf16_inputs(shapes):
    """Both operands unswizzled BF16; delegates to the forward fast-DGT generator so
    both suites resolve tiling the same way (fast DGT speeds up phase 1's dY load; B1 only).
    """
    configs = []
    for M, K, N, desc in shapes:
        configs += _make_bf16_inputs_fast_dgt([(M, K, N, desc)])
    return configs


def _make_prequant_weight_wrapx_dx(shapes):
    """dX phase: dY unswizzled BF16 (F-by-K, fast DGT), W pre-quantized wrapX MXFP8.

    Matches the forward prequant-weight-wrapx arm (fast DGT speeds up dY; W via dma_copy).
    """
    configs = []
    for M, K, N, desc in shapes:
        if K < 512:  # prequant tiling needs a full K tile
            continue
        conf = config_helper.TestConfig(
            M=M,
            K=K,
            N=N,
            description=f"{desc} [prequant-weight-wrapx]",
            run_with_lnc2=None,
            enable_scale_packing=True,
            output_dtype=constants.MatrixPrecision.BFLOAT16,
            lhs_dtype=constants.MatrixPrecision.BFLOAT16,
            rhs_dtype=constants.MatrixPrecision.MXFP8,
            lhs_is_swizzled=False,
            rhs_is_swizzled=True,
            fast_dma_transpose=True,
            spill_reload=False,
        )
        configs += conf.autoGenerateDefault()
    return configs


def _make_prequant_weight_wrapx_dw(shapes):
    """dW phase: both operands (dY, X) unswizzled BF16.

    W is not a dW operand, so this phase is tiled for BF16 x BF16 (not the dX phase's
    mixed-dtype tiling); fast_dma_transpose speeds up both DGT loads.
    """
    configs = []
    for M, K, N, desc in shapes:
        if K < 512:  # prequant-arm dW keeps the same K-tile floor as its dX phase
            continue
        conf = config_helper.TestConfig(
            M=M,
            K=K,
            N=N,
            description=f"{desc} [prequant-weight-wrapx]",
            run_with_lnc2=None,
            enable_scale_packing=True,
            output_dtype=constants.MatrixPrecision.BFLOAT16,
            lhs_dtype=constants.MatrixPrecision.BFLOAT16,
            rhs_dtype=constants.MatrixPrecision.BFLOAT16,
            lhs_is_swizzled=False,
            rhs_is_swizzled=False,
            fast_dma_transpose=True,
            spill_reload=False,
        )
        configs += conf.autoGenerateDefault()
    return configs


def _make_prequantized(shapes):
    """Both stored operands (W for dX, X for dW) pre-quantized MXFP8; dY stays BF16.

    dY cannot be pre-quantized: the two phases contract it over different axes (N in
    dX, M in dW), so one buffer can't be swizzled for both. This matches production,
    where dY is a freshly computed gradient. Only W and X are stored tensors a
    framework can hold pre-quantized, and both are covered.
    """
    configs = []
    for M, K, N, desc in shapes:
        if K < 512:  # prequant tiling needs a full K tile
            continue
        conf = config_helper.TestConfig(
            M=M,
            K=K,
            N=N,
            description=f"{desc} [prequantized-weights-and-acts]",
            run_with_lnc2=None,
            enable_scale_packing=True,
            output_dtype=constants.MatrixPrecision.BFLOAT16,
            lhs_dtype=constants.MatrixPrecision.BFLOAT16,
            rhs_dtype=constants.MatrixPrecision.MXFP8,
            lhs_is_swizzled=False,
            rhs_is_swizzled=True,
            # dY is the unswizzled BF16 operand; fast_dma_transpose accelerates its
            # DGT load (the pre-quantized W/X operand loads via dma_copy).
            fast_dma_transpose=True,
            spill_reload=False,
        )
        configs += conf.autoGenerateDefault()
    return configs


def _make_pe_wrapx_dx(shapes):
    """dX phase (PE-swizzle wrapX): dY unswizzled BF16 transposed via PE-swizzle (wrapX),
    W pre-quantized wrapX MXFP8. No fast_dma_transpose -- the transpose is a PE op, so with
    NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE=true no DMA transpose is used anywhere.
    """
    configs = []
    for M, K, N, desc in shapes:
        if K < 512:  # prequant tiling needs a full K tile
            continue
        conf = config_helper.TestConfig(
            M=M,
            K=K,
            N=N,
            description=f"{desc} [pe-wrapx]",
            run_with_lnc2=None,
            enable_scale_packing=True,
            output_dtype=constants.MatrixPrecision.BFLOAT16,
            lhs_dtype=constants.MatrixPrecision.BFLOAT16,
            rhs_dtype=constants.MatrixPrecision.MXFP8,
            lhs_is_swizzled=False,
            rhs_is_swizzled=True,
            load_with_PE_swizzle=True,
            quant_scheme="wrapX",
            spill_reload=False,
        )
        configs += conf.autoGenerateDefault()
    return configs


def _make_pe_wrapx_dw(shapes):
    """dW phase (PE-swizzle wrapX): both operands (dY, X) unswizzled BF16 via PE-swizzle
    transpose (wrapX). W is not a dW operand, so this phase is BF16 x BF16.
    """
    configs = []
    for M, K, N, desc in shapes:
        if K < 512:
            continue
        conf = config_helper.TestConfig(
            M=M,
            K=K,
            N=N,
            description=f"{desc} [pe-wrapx]",
            run_with_lnc2=None,
            enable_scale_packing=True,
            output_dtype=constants.MatrixPrecision.BFLOAT16,
            lhs_dtype=constants.MatrixPrecision.BFLOAT16,
            rhs_dtype=constants.MatrixPrecision.BFLOAT16,
            lhs_is_swizzled=False,
            rhs_is_swizzled=False,
            load_with_PE_swizzle=True,
            quant_scheme="wrapX",
            spill_reload=False,
        )
        configs += conf.autoGenerateDefault()
    return configs


def _make_pe_swizzle_1x32_no_dma_transpose(shapes):
    """Like _make_pe_swizzle_1x32 but disable_dma_transpose=True (column I).

    Both operands unswizzled BF16, 1x32 on the fly; the flag applies to both phases.
    """
    configs = []
    for M, K, N, desc in shapes:
        conf = config_helper.TestConfig(
            M=M,
            K=K,
            N=N,
            description=f"{desc} [pe-swizzle-1x32-no-dma-transpose]",
            run_with_lnc2=None,
            enable_scale_packing=False,
            output_dtype=constants.MatrixPrecision.BFLOAT16,
            lhs_dtype=constants.MatrixPrecision.BFLOAT16,
            rhs_dtype=constants.MatrixPrecision.BFLOAT16,
            lhs_is_swizzled=False,
            rhs_is_swizzled=False,
            load_with_PE_swizzle=True,
            quant_scheme="1x32",
            disable_dma_transpose=True,
            spill_reload=False,
        )
        configs += conf.autoGenerateDefault()
    return configs


def _make_lhs_1x32_pe_swizzle_rhs_1x32_prequantized_no_dma_transpose(shapes):
    """Like _make_lhs_1x32_pe_swizzle_rhs_1x32_prequantized but disable_dma_transpose=True (col J).

    Flag comes off the dX config and applies to both phases; it's a no-op in dX (dY F-by-K, W
    prequant) and bites in dW (dY, X K-by-F 1x32).
    """
    configs = []
    for M, K, N, desc in shapes:
        if K < 512:
            continue
        conf = config_helper.TestConfig(
            M=M,
            K=K,
            N=N,
            description=f"{desc} [lhs-pe-1x32-rhs-prequant-1x32-no-dma-transpose]",
            run_with_lnc2=None,
            enable_scale_packing=True,
            output_dtype=constants.MatrixPrecision.BFLOAT16,
            lhs_dtype=constants.MatrixPrecision.BFLOAT16,
            rhs_dtype=constants.MatrixPrecision.MXFP8,
            lhs_is_swizzled=False,
            rhs_is_swizzled=True,
            load_with_PE_swizzle=True,
            quant_scheme="1x32",
            disable_dma_transpose=True,
            spill_reload=False,
        )
        configs += conf.autoGenerateDefault()
    return configs


# ============================================================================
# Grids: 4 reachable input use cases (TP4)
# ============================================================================


def _grid(tp, factory, label, dw_factory=None):
    return make_backward_configs(_TP_SHAPES[tp], factory, f"TP{tp} {label}", weight_grad_config_factory=dw_factory)


# _grid over _TP_SHAPES[4] now yields BOTH projections (QKV + O-proj): the loader's
# partial-F-tile masking removed the old F%512 filter that dropped QKV.
GRID_TP4_BF16_INPUTS = _grid(4, _make_bf16_inputs, "bf16-inputs")
GRID_TP4_PREQUANT_WEIGHT_WRAPX = _grid(
    4, _make_prequant_weight_wrapx_dx, "prequant-weight-wrapx", _make_prequant_weight_wrapx_dw
)
GRID_TP4_PREQUANTIZED = _grid(4, _make_prequantized, "prequantized")
# Prequantized inputs, 1x32: BOTH W and X pre-quantized (1x32, swizzled). One factory serves
# both phases -- dX's RHS is W, dW's RHS is X, both pre-quantized 1x32 -- so dY (BF16, 1x32 on
# the fly) is the only operand loaded on the fly in either phase. This is the "input
# activations also prequantized" row. Ablation over the dY transpose:
#   DMA transpose  (default):            GRID_TP4_PREQUANTIZED_1X32
#   NO DMA transpose (disable_dma_transpose=True, dY via dma_copy + nc_transpose):
#                                        GRID_TP4_PREQUANTIZED_1X32_NO_DMA_TRANSPOSE
GRID_TP4_PREQUANTIZED_1X32 = _grid(4, _make_lhs_1x32_pe_swizzle_rhs_1x32_prequantized, "prequantized-1x32")
GRID_TP4_PREQUANTIZED_1X32_NO_DMA_TRANSPOSE = _grid(
    4,
    _make_lhs_1x32_pe_swizzle_rhs_1x32_prequantized_no_dma_transpose,
    "prequantized-1x32-no-dma-transpose",
)
# Both phases load every operand unswizzled BF16 and quantize on the fly with the 1x32
# group layout, so one factory serves dX and dW (reused from the forward suite for
# identical tiling). No fast_dma_transpose: the 1x32 PE-swizzle load handles the
# transpose itself.
GRID_TP4_PE_SWIZZLE_1X32 = _grid(4, _make_pe_swizzle_1x32, "pe-swizzle-1x32")
# "1x32 NO DMA TRANSPOSE" (column I): pe-swizzle-1x32 with disable_dma_transpose=True.
GRID_TP4_PE_SWIZZLE_1X32_NO_DMA_TRANSPOSE = _grid(
    4, _make_pe_swizzle_1x32_no_dma_transpose, "pe-swizzle-1x32-no-dma-transpose"
)
# "1x32 NO DMA TRANSPOSE" prequant-weight (column J): flag no-ops in dX, bites in dW.
GRID_TP4_PREQUANT_WEIGHT_1X32_NO_DMA_TRANSPOSE = _grid(
    4,
    _make_lhs_1x32_pe_swizzle_rhs_1x32_prequantized_no_dma_transpose,
    "prequant-weight-1x32-no-dma-transpose",
    _make_pe_swizzle_1x32_no_dma_transpose,
)
# 1x32 quant scheme with the weight (W) pre-quantized, mirroring prequant-weight-wrapx:
# dX loads dY on the fly (1x32) against a pre-quantized W (pe_1x32 x prequant); W is not a
# dW operand, so dW loads dY and X on the fly (both 1x32, same factory as pe-swizzle-1x32).
GRID_TP4_PREQUANT_WEIGHT_1X32 = _grid(
    4, _make_lhs_1x32_pe_swizzle_rhs_1x32_prequantized, "prequant-weight-1x32", _make_pe_swizzle_1x32
)
# PE-swizzle wrapX: dX loads dY via PE-swizzle (wrapX) against pre-quantized W; dW loads
# dY and X via PE-swizzle (wrapX). No fast_dma_transpose, so with
# NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE=true no DMA transpose is used in either phase.
GRID_TP4_PE_WRAPX = _grid(4, _make_pe_wrapx_dx, "pe-wrapx", _make_pe_wrapx_dw)
# BF16 Inputs, WrapX PE Swizzle: both operands BF16 via PE-swizzle (wrapX). One factory
# serves dX and dW (both phases are both-BF16 on-the-fly quant, like pe-swizzle-1x32).
GRID_TP4_BF16_INPUTS_PE_WRAPX = _grid(4, _make_bf16_inputs_pe_wrapx, "bf16-inputs-pe-wrapx")


# ============================================================================
# Input construction
# ============================================================================


def _swizzle_and_quantize(tensor_k_by_f, conf, K, F):
    """Swizzle a K-by-F BF16 tensor into wrapX layout, then quantize to MXFP8.

    Swizzle FIRST, then quantize on the swizzled buffer (mirrors build_matmul_inputs):
    the interleave decides which K values share a scale group, so quantizing the
    unswizzled tensor would group wrong and surface as an accuracy failure. Quantizes
    to the _x4 dtype and views back to non-x4, as the MXFP8 operand dtype expects.
    K, F are the logical contraction/free dims for the packed-scale layout.
    """
    x4_name = conf.float8_dtype if conf.float8_dtype.endswith("_x4") else conf.float8_dtype + "_x4"
    non_x4_name = x4_name.replace("_x4", "")

    swizzled = matmul_utils.swizzle_tensor(tensor_k_by_f)
    data, scales_compact = quantize_mx_golden(swizzled, getattr(nl, x4_name))
    if conf.enable_scale_packing:
        scales = generate_golden_packed_scales(scales_compact, K, F, Q_TILE_K)
    else:
        scales = matmul_utils.resize_scales_compact_to_oversized_2d(scales_compact)
    return data.view(getattr(dtype, non_x4_name)), scales


def build_backward_inputs(bwd_conf):
    """Build the three operands matmul_mxfp8_backward needs (dY=[M,N], W=[N,K], X=[M,K]).

    Which are pre-quantized is read off the phase configs (not a separate flag) so the
    tensors can't disagree with the tiling: dY iff BOTH phases say so, W from dX's RHS,
    X from dW's RHS.
    """
    import nki.language as nl

    M, K, N = bwd_conf.M, bwd_conf.K, bwd_conf.N
    dx_conf, dw_conf = bwd_conf.input_grad_config, bwd_conf.weight_grad_config
    MXFP8 = (constants.MatrixPrecision.MXFP8, constants.MatrixPrecision.MXFP8_X4)

    dy_prequant = dx_conf.lhs_dtype in MXFP8 and dw_conf.lhs_dtype in MXFP8
    w_prequant = dx_conf.rhs_dtype in MXFP8
    x_prequant = dw_conf.rhs_dtype in MXFP8

    # 1x32: pass the unswizzled BF16 rhs operands (W, X) pre-transposed as F-by-K so they
    # skip the loader's K-by-F round-trip transpose, leaving dY as the sole transpose.
    use_f_by_k = dx_conf.quant_scheme == "1x32"

    rng = np.random.default_rng(dx_conf.seed)
    dy_f32 = rng.standard_normal((M, N), dtype=np.float32)
    w_f32 = rng.standard_normal((N, K), dtype=np.float32)
    x_f32 = rng.standard_normal((M, K), dtype=np.float32)

    kernel_input = {
        "input_grad_config": dx_conf.kernel_config,
        "weight_grad_config": dw_conf.kernel_config,
        "output_dtype": nl.bfloat16,
        "run_with_lnc2": dx_conf.run_with_lnc2,
        "float8_dtype": dx_conf.float8_dtype,
        "use_scale_packing": dx_conf.enable_scale_packing,
        "spill_reload": dx_conf.spill_reload,
        "quant_scheme": dx_conf.quant_scheme,
        "fast_dma_transpose": getattr(dx_conf, "fast_dma_transpose", False),
        "disable_dma_transpose": getattr(dx_conf, "disable_dma_transpose", False),
        "output_grad_scales": None,
        "weight_scales": None,
        "input_scales": None,
        "weights_is_f_by_k": False,
        "input_is_f_by_k": False,
    }

    # dY: pre-quantized only in phase 1's K=N orientation (the phases disagree on dY's
    # K axis, so it can be swizzled only one way); matches the kernel's default F-by-K LHS.
    if dy_prequant:
        # dY is [M, N]; phase 1 contracts over N, so transpose to put N on axis 0.
        data, scales = _swizzle_and_quantize(dy_f32.astype(nl.bfloat16).T, dx_conf, N, M)
        kernel_input["output_grad"] = data
        kernel_input["output_grad_scales"] = scales
        kernel_input["output_grad_is_swizzled"] = True
    else:
        kernel_input["output_grad"] = dy_f32.astype(nl.bfloat16)
        kernel_input["output_grad_is_swizzled"] = False

    # W: [N, K], contracted over N -> K axis is N, so swizzle the [N, K] buffer as-is.
    if w_prequant:
        data, scales = _swizzle_and_quantize(w_f32.astype(nl.bfloat16), dx_conf, N, K)
        kernel_input["weights"] = data
        kernel_input["weight_scales"] = scales
        kernel_input["weights_is_swizzled"] = True
    elif use_f_by_k:
        # F-by-K: pass Wᵀ = [K, N] so the loader takes the no-transpose dma_copy path.
        kernel_input["weights"] = w_f32.astype(nl.bfloat16).T
        kernel_input["weights_is_swizzled"] = False
        kernel_input["weights_is_f_by_k"] = True
    else:
        kernel_input["weights"] = w_f32.astype(nl.bfloat16)
        kernel_input["weights_is_swizzled"] = False

    # X: [M, K], contracted over M -> K axis is M, so swizzle the [M, K] buffer as-is.
    if x_prequant:
        data, scales = _swizzle_and_quantize(x_f32.astype(nl.bfloat16), dw_conf, M, K)
        kernel_input["input_activation"] = data
        kernel_input["input_scales"] = scales
        kernel_input["input_is_swizzled"] = True
    elif use_f_by_k:
        # F-by-K: pass Xᵀ = [K, M] so the loader takes the no-transpose dma_copy path.
        kernel_input["input_activation"] = x_f32.astype(nl.bfloat16).T
        kernel_input["input_is_swizzled"] = False
        kernel_input["input_is_f_by_k"] = True
    else:
        kernel_input["input_activation"] = x_f32.astype(nl.bfloat16)
        kernel_input["input_is_swizzled"] = False

    return kernel_input


# ============================================================================
# Test class
# ============================================================================


# LOCAL-ONLY: no @pytest_test_metadata, so the pipeline does not discover it (see
# test/README.internal.md, "Pipeline Test Discovery"). Every reachable arm loads dY via
# fast_dma_transpose, not yet supported in the pipeline; run by hand with -k.
# TODO: when fast_dma_transpose is pipeline-supported, add @pytest_test_metadata(
# name="Qwen3-235B Matmul Backward Perf Benchmarks", tags=["model"]) + model/optimal marks.
@pytest_marks(["generic_matmul_mxfp8_perf", "mx", "mxfp8"])
@pytest.mark.platforms(
    exclude=[
        common_dataclasses.Platforms.TRN1,
        common_dataclasses.Platforms.TRN2,
        common_dataclasses.Platforms.TRN3_A0,
        common_dataclasses.Platforms.TRN3_PDS_A0,
    ]
)
class TestQwen235BMatmulBackwardPerf:
    """Qwen3-235B QKV/O projection backward perf through the fused backward kernel."""

    def _run_test(self, test_manager, conf, platform_target):
        # B1 only: fast_dma_transpose is a B1-only instruction with no arch guard; on A0
        # it faults the device (can take the node down), so never run these on A0. The
        # platform mark already deselects A0 at collection; this is a backstop.
        if platform_target not in _TRN3_B1_PLATFORMS:
            pytest.skip("MXFP8 matmul backward perf requires TRN3 B1.")

        import nki.language as nl

        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=2 if conf.input_grad_config.run_with_lnc2 else 1,
            platform_target=platform_target,
        )

        test_manager.collector.set_kernel_params(conf.to_metrics_dict())
        kernel_input = build_backward_inputs(conf)

        def output_tensors(_):
            return {
                "input_grad": np.zeros((conf.M, conf.K), dtype=nl.bfloat16),
                "weight_grad": np.zeros((conf.N, conf.K), dtype=nl.bfloat16),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=matmul_mxfp8_generic_backward_kernel.matmul_mxfp8_backward,
            torch_ref=matmul_mxfp8_backward_torch_ref,
            kernel_input_generator=lambda _: kernel_input,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            custom_comparator=_bwd_comparator,
        )

    # --- TP4: each arm parametrizes over both projections (QKV + O-proj). ---

    @pytest.mark.parametrize("conf", GRID_TP4_BF16_INPUTS)
    def test_bwd_tp4_bf16_inputs_fp8_matmul(self, test_manager, conf, platform_target):
        self._run_test(test_manager, conf, platform_target)

    @pytest.mark.parametrize("conf", GRID_TP4_PREQUANT_WEIGHT_WRAPX)
    def test_bwd_tp4_prequant_weight_wrapx(self, test_manager, conf, platform_target):
        self._run_test(test_manager, conf, platform_target)

    @pytest.mark.parametrize("conf", GRID_TP4_PREQUANTIZED)
    def test_bwd_tp4_prequantized(self, test_manager, conf, platform_target):
        self._run_test(test_manager, conf, platform_target)

    # Prequantized inputs (W and X both pre-quantized, 1x32) -- DMA-transpose ablation.
    @pytest.mark.parametrize("conf", GRID_TP4_PREQUANTIZED_1X32)
    def test_bwd_tp4_prequantized_1x32(self, test_manager, conf, platform_target):
        self._run_test(test_manager, conf, platform_target)

    @pytest.mark.parametrize("conf", GRID_TP4_PREQUANTIZED_1X32_NO_DMA_TRANSPOSE)
    def test_bwd_tp4_prequantized_1x32_no_dma_transpose(self, test_manager, conf, platform_target):
        self._run_test(test_manager, conf, platform_target)

    @pytest.mark.parametrize("conf", GRID_TP4_PE_SWIZZLE_1X32)
    def test_bwd_tp4_pe_swizzle_1x32(self, test_manager, conf, platform_target):
        self._run_test(test_manager, conf, platform_target)

    @pytest.mark.parametrize("conf", GRID_TP4_PREQUANT_WEIGHT_1X32)
    def test_bwd_tp4_prequant_weight_1x32(self, test_manager, conf, platform_target):
        self._run_test(test_manager, conf, platform_target)

    @pytest.mark.parametrize("conf", GRID_TP4_PE_SWIZZLE_1X32_NO_DMA_TRANSPOSE)
    def test_bwd_tp4_pe_swizzle_1x32_no_dma_transpose(self, test_manager, conf, platform_target):
        self._run_test(test_manager, conf, platform_target)

    @pytest.mark.parametrize("conf", GRID_TP4_PREQUANT_WEIGHT_1X32_NO_DMA_TRANSPOSE)
    def test_bwd_tp4_prequant_weight_1x32_no_dma_transpose(self, test_manager, conf, platform_target):
        self._run_test(test_manager, conf, platform_target)

    @pytest.mark.parametrize("conf", GRID_TP4_PE_WRAPX)
    def test_bwd_tp4_pe_wrapx(self, test_manager, conf, platform_target):
        self._run_test(test_manager, conf, platform_target)

    @pytest.mark.parametrize("conf", GRID_TP4_BF16_INPUTS_PE_WRAPX)
    def test_bwd_tp4_bf16_inputs_pe_wrapx(self, test_manager, conf, platform_target):
        self._run_test(test_manager, conf, platform_target)
