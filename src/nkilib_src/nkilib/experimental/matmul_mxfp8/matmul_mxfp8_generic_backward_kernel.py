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

This kernel uses the `is_f_by_k` feature to avoid explicit transpose operations:
    - For dX: lhs=dY[M,N] (F-by-K, F=M, K_contraction=N), rhs=W[N,K] (K-by-F, K_contraction=N, F=K)
    - For dW: lhs=dY[M,N] (K-by-F, K_contraction=M, F=N), rhs=X[M,K] (K-by-F, K_contraction=M, F=K)
"""

import nki.language as nl

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
    float8_dtype: str = "float8_e5m2",
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
        weights: Weight matrix (W), shape [N, K] in BF16.
        input_activation: Input activation (X), shape [M, K] in BF16.
        input_grad_config: MatmulMxfp8KernelConfig for the dX phase (auto-resolved if None).
        weight_grad_config: MatmulMxfp8KernelConfig for the dW phase (auto-resolved if None).
        tile_loop_order (str): Tile processing order within blocks, default 'mnk'.
        float8_dtype (str): FP8 dtype for quantization, default "float8_e5m2".
        output_dtype: Output data type, default nl.bfloat16.
        run_with_lnc2 (bool): Enable LNC2 parallelization, default True.
        lnc_2_shard_rhs (bool): Shard on N dimension (RHS), default True.
        output_grad_scales: Optional pre-computed scales for output gradient.
        weight_scales: Optional pre-computed scales for weights.
        input_scales: Optional pre-computed scales for input activation.
        use_scale_packing (bool): Assert packed scales for pre-quantized inputs.
        spill_reload (bool): Spill quantized blocks to HBM for reuse.
        output_grad_is_swizzled (bool): Whether output gradient is pre-swizzled.
        weights_is_swizzled (bool): Whether weights are pre-swizzled.
        input_is_swizzled (bool): Whether input activation is pre-swizzled.

    Returns:
        tuple: (input_grad, weight_grad) where:
            - input_grad: Shape [M, K], gradient with respect to input
            - weight_grad: Shape [N, K], gradient with respect to weights

    Pseudocode:
        # Phase 1: Input gradient (dX = dY @ W)
        input_grad = matmul_mxfp8(dY, W, lhs_is_f_by_k=True, rhs_is_f_by_k=False)

        # Phase 2: Weight gradient (dW = dY^T @ X)
        weight_grad = matmul_mxfp8(dY, X, lhs_is_f_by_k=False, rhs_is_f_by_k=False)
    """

    # Extract per-phase tiling from configs
    dx_tiles_m, dx_tiles_n, dx_tiles_k, dx_load_m, dx_load_n, dx_lhs_tile, dx_rhs_tile = _unpack_config(
        input_grad_config
    )
    dw_tiles_m, dw_tiles_n, dw_tiles_k, dw_load_m, dw_load_n, dw_lhs_tile, dw_rhs_tile = _unpack_config(
        weight_grad_config
    )

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
        lhs_is_f_by_k=True,
        rhs_is_f_by_k=False,
        lhs_is_swizzled=output_grad_is_swizzled,
        rhs_is_swizzled=weights_is_swizzled,
        tile_loop_order=tile_loop_order,
        float8_dtype=float8_dtype,
        output_dtype=output_dtype,
        run_with_lnc2=run_with_lnc2,
        lnc_2_shard_rhs=lnc_2_shard_rhs,
        use_scale_packing=use_scale_packing,
        spill_reload=spill_reload,
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
        lhs_is_f_by_k=False,
        rhs_is_f_by_k=False,
        lhs_is_swizzled=output_grad_is_swizzled,
        rhs_is_swizzled=input_is_swizzled,
        tile_loop_order=tile_loop_order,
        float8_dtype=float8_dtype,
        output_dtype=output_dtype,
        run_with_lnc2=run_with_lnc2,
        lnc_2_shard_rhs=lnc_2_shard_rhs,
        use_scale_packing=use_scale_packing,
        spill_reload=spill_reload,
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
