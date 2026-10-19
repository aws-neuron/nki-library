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

"""Backward pass for MXFP8 matrix multiplication (linear layer).

Given the forward pass: Y = X @ W^T (where X is [M, K] and W is [N, K]),
the backward pass computes:
    dX = dY @ W        (input gradient, shape [M, K])
    dW = dY^T @ X      (weight gradient, shape [N, K])

This kernel uses the operand-orientation feature to avoid explicit transpose operations:
    - For dX: lhs=dY[M,N] (F-by-K, F=M, K_contraction=N), rhs=W[N,K] (K-by-F, K_contraction=N, F=K)
    - For dW: lhs=dY[M,N] (K-by-F, K_contraction=M, F=N), rhs=X[M,K] (K-by-F, K_contraction=M, F=K)
"""

import nki.language as nl

from ..mxfp_utils.mxfp8_utils.common_dataclasses import TensorOrientation
from .matmul_mxfp8_config import MatmulMxfp8KernelConfig
from .matmul_mxfp8_generic_kernel import matmul_mxfp8


def matmul_mxfp8_backward(
    output_grad,
    weights,
    input_activation,
    # Per-phase matmul configs (auto-resolved if None)
    input_grad_config: MatmulMxfp8KernelConfig = None,
    weight_grad_config: MatmulMxfp8KernelConfig = None,
    # Shared parameters
    tile_loop_order: str = "mnk",
    float8_dtype: str = "float8_e4m3fn",
    output_dtype=nl.bfloat16,
    run_with_lnc2: bool = True,
    lnc_2_shard_rhs: bool = True,
    output_grad_scales=None,
    weight_scales=None,
    input_scales=None,
    use_scale_packing: bool = False,
    spill_reload: bool = False,
    output_grad_is_swizzled: bool = False,
    weights_is_swizzled: bool = False,
    input_is_swizzled: bool = False,
    fast_dma_transpose: bool = False,
    quant_scheme: str = "wrapX",
    disable_dma_transpose: bool = False,
    weights_is_f_by_k: bool = False,
    input_is_f_by_k: bool = False,
) -> tuple:
    """
    Backward pass for matrix multiplication with MXFP8 quantization.

    Computes both input gradients (dX) and weight gradients (dW) for a linear layer.

    Forward pass convention:
        Y = X @ W^T, where X is [M, K], W is [N, K], Y is [M, N]

    Backward pass (two separate matmuls with different dimensions):
        dX = dY @ W     (shape [M, K]):  M_logical=M, K_contraction=N, N_logical=K
        dW = dY^T @ X   (shape [N, K]):  M_logical=N, K_contraction=M, N_logical=K

    Args:
        output_grad: Output gradient (dY), shape [M, N] in BF16.
        weights: Weight matrix (W), shape [N, K]. BF16 (quantized on the fly), or a
            pre-quantized MXFP8 buffer when weights_is_swizzled=True (+ weight_scales).
        input_activation: Input activation (X), shape [M, K]. BF16 (quantized on the fly),
            or a pre-quantized MXFP8 buffer when input_is_swizzled=True (+ input_scales).
            W and X are independent: either, both, or neither may be pre-quantized.
        input_grad_config: MatmulMxfp8KernelConfig for the dX phase (auto-resolved if None).
        weight_grad_config: MatmulMxfp8KernelConfig for the dW phase (auto-resolved if None).
        tile_loop_order (str): Tile processing order within blocks, default 'mnk'.
        float8_dtype (str): FP8 dtype for quantization, default "float8_e5m2".
        output_dtype: Output data type, default nl.bfloat16.
        run_with_lnc2 (bool): Fallback LNC2 enable used only when a phase config is not
            supplied; otherwise each phase honors its config's resolved value.
        lnc_2_shard_rhs (bool): Fallback shard axis (True=N/RHS) used only when a phase
            config is not supplied; otherwise each phase honors the axis auto_generate_default
            resolved and tuned its tiling/cache key for (dX and dW may shard different axes).
        output_grad_scales: Optional pre-computed scales for output gradient.
        weight_scales: Optional pre-computed scales for weights.
        input_scales: Optional pre-computed scales for input activation.
        use_scale_packing (bool): Assert packed scales for pre-quantized inputs.
        spill_reload (bool): Spill quantized blocks to HBM for reuse.
        output_grad_is_swizzled (bool): Whether output gradient is pre-swizzled.
        weights_is_swizzled (bool): Whether weights are pre-swizzled.
        input_is_swizzled (bool): Whether input activation is pre-swizzled.
        fast_dma_transpose (bool): Use the direct 4D DMA gather-transpose access pattern
            for unswizzled BF16 operands. Only affects the F-by-K operand (dY in phase 1);
            the K-by-F operands load through the PE transpose, which ignores it.
        quant_scheme (str): MX quantization group layout for on-the-fly quantization,
            "wrapX" or "1x32", applied to every operand. The 1x32 PE-swizzle loader
            handles F-by-K and K-by-F unswizzled operands alike, so both phases work.
        disable_dma_transpose (bool): When True, the 1x32 loader transposes K-by-F operands
            with dma_copy + nc_transpose instead of a DMA transpose (both phases). Default False.
        weights_is_f_by_k (bool): W passed pre-transposed as [K, N] (F-by-K) so its load skips
            the K-by-F round-trip transpose. Default False ([N, K], K-by-F).
        input_is_f_by_k (bool): X passed pre-transposed as [K, M] (F-by-K); leaves dY as the
            sole transposed operand in dW. Default False ([M, K], K-by-F).

    Returns:
        tuple: (input_grad, weight_grad) where:
            - input_grad: Shape [M, K], gradient with respect to input
            - weight_grad: Shape [N, K], gradient with respect to weights

    Pseudocode:
        # Phase 1: Input gradient (dX = dY @ W)
        input_grad = matmul_mxfp8(dY, W, lhs_orientation=None, rhs_orientation=TensorOrientation.K_BY_F)

        # Phase 2: Weight gradient (dW = dY^T @ X)
        weight_grad = matmul_mxfp8(
            dY, X, lhs_orientation=TensorOrientation.K_BY_F, rhs_orientation=TensorOrientation.K_BY_F
        )
    """

    # Extract per-phase tiling from configs
    dx_tiles_m, dx_tiles_n, dx_tiles_k, dx_load_m, dx_load_n, dx_lhs_tile, dx_rhs_tile = _unpack_config(
        input_grad_config
    )
    dw_tiles_m, dw_tiles_n, dw_tiles_k, dw_load_m, dw_load_n, dw_lhs_tile, dw_rhs_tile = _unpack_config(
        weight_grad_config
    )

    # Honor each phase's config-resolved shard axis (fall back to the kernel args if a
    # phase config was not supplied).
    dx_run_lnc2 = input_grad_config.run_with_lnc2 if input_grad_config is not None else run_with_lnc2
    dx_shard_rhs = input_grad_config.lnc_2_shard_rhs if input_grad_config is not None else lnc_2_shard_rhs
    dw_run_lnc2 = weight_grad_config.run_with_lnc2 if weight_grad_config is not None else run_with_lnc2
    dw_shard_rhs = weight_grad_config.lnc_2_shard_rhs if weight_grad_config is not None else lnc_2_shard_rhs

    # =========================================================================
    # Phase 1: Input gradient  dX = dY @ W
    # =========================================================================
    input_grad = matmul_mxfp8(
        lhs=output_grad,
        rhs=weights,
        TILES_IN_BLOCK_M=dx_tiles_m,
        TILES_IN_BLOCK_N=dx_tiles_n,
        TILES_IN_BLOCK_K=dx_tiles_k,
        TILES_IN_LOAD_M=dx_load_m,
        TILES_IN_LOAD_N=dx_load_n,
        lhs_matmul_tile_shape_logical=dx_lhs_tile,
        rhs_matmul_tile_shape_logical=dx_rhs_tile,
        lhs_scales=output_grad_scales,
        rhs_scales=weight_scales,
        # dY is [M, N] and phase 1 contracts over N, so it is F-by-K (None auto-resolves).
        lhs_orientation=None,
        # W is [N, K] contracted over N, so it is K-by-F. Derive from is_swizzled: an
        # unswizzled operand must declare K_BY_F so TensorDescriptor sets load_with_PE_swizzle;
        # a swizzled operand is already K-by-F but matmul_mxfp8 rejects an explicit K_BY_F on a
        # pre-swizzled input, so it must pass None (auto-resolves to K_BY_F).
        # W is F-by-K when passed pre-transposed ([K, N]): skips the loader's K-by-F
        # round-trip transpose so only dY transposes. Else K-by-F ([N, K], the default).
        rhs_orientation=(
            TensorOrientation.F_BY_K
            if weights_is_f_by_k
            else (None if weights_is_swizzled else TensorOrientation.K_BY_F)
        ),
        lhs_is_swizzled=output_grad_is_swizzled,
        rhs_is_swizzled=weights_is_swizzled,
        tile_loop_order=tile_loop_order,
        float8_dtype=float8_dtype,
        output_dtype=output_dtype,
        run_with_lnc2=dx_run_lnc2,
        lnc_2_shard_rhs=dx_shard_rhs,
        use_scale_packing=use_scale_packing,
        spill_reload=spill_reload,
        fast_dma_transpose=fast_dma_transpose,
        quant_scheme=quant_scheme,
        # When W is F-by-K neither dX operand is K-by-F, so the flag is a load-path no-op:
        # gate it off so dX keys as the DMA entry (its no-DMA cache key would never be hit).
        disable_dma_transpose=disable_dma_transpose and not weights_is_f_by_k,
    )

    # =========================================================================
    # Phase 2: Weight gradient  dW = dY^T @ X
    # =========================================================================
    weight_grad = matmul_mxfp8(
        lhs=output_grad,
        rhs=input_activation,
        TILES_IN_BLOCK_M=dw_tiles_m,
        TILES_IN_BLOCK_N=dw_tiles_n,
        TILES_IN_BLOCK_K=dw_tiles_k,
        TILES_IN_LOAD_M=dw_load_m,
        TILES_IN_LOAD_N=dw_load_n,
        lhs_matmul_tile_shape_logical=dw_lhs_tile,
        rhs_matmul_tile_shape_logical=dw_rhs_tile,
        lhs_scales=output_grad_scales,
        rhs_scales=input_scales,
        # Both operands contract over M here, so both are K-by-F. Derive from is_swizzled
        # (see phase 1): K_BY_F declares the layout for an unswizzled operand, None
        # (auto-resolves to K_BY_F) satisfies matmul_mxfp8's assert for a swizzled one.
        lhs_orientation=None if output_grad_is_swizzled else TensorOrientation.K_BY_F,
        # X is F-by-K when passed pre-transposed ([K, M]): avoids its transpose so dY
        # (K-by-F lhs) is the sole transpose in dW. Else K-by-F ([M, K], the default).
        rhs_orientation=(
            TensorOrientation.F_BY_K if input_is_f_by_k else (None if input_is_swizzled else TensorOrientation.K_BY_F)
        ),
        lhs_is_swizzled=output_grad_is_swizzled,
        rhs_is_swizzled=input_is_swizzled,
        tile_loop_order=tile_loop_order,
        float8_dtype=float8_dtype,
        output_dtype=output_dtype,
        run_with_lnc2=dw_run_lnc2,
        lnc_2_shard_rhs=dw_shard_rhs,
        use_scale_packing=use_scale_packing,
        spill_reload=spill_reload,
        fast_dma_transpose=fast_dma_transpose,
        quant_scheme=quant_scheme,
        disable_dma_transpose=disable_dma_transpose,
    )

    return input_grad, weight_grad


def _unpack_config(config: MatmulMxfp8KernelConfig) -> tuple:
    """Extract tiling parameters from config. Returns all None if config is None."""
    if config is None:
        return None, None, None, None, None, None, None
    lhs_tile = (config.tile_k, config.tile_m) if config.tile_k and config.tile_m else None
    rhs_tile = (config.tile_k, config.tile_n) if config.tile_k and config.tile_n else None
    return (
        config.TILES_IN_BLOCK_M,
        config.TILES_IN_BLOCK_N,
        config.TILES_IN_BLOCK_K,
        config.TILES_IN_LOAD_M,
        config.TILES_IN_LOAD_N,
        lhs_tile,
        rhs_tile,
    )
