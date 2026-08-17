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

"""Rotational top-k kernel finding the k largest elements along a dimension.

Uses multi-stage rotation and reduction optimized for NeuronCore architecture.
"""

from enum import Enum
from typing import Optional, Tuple

import nki
import nki.isa as nisa
import nki.language as nl

from ..max.cascaded_max_utils import predicated_folded_load, unfolded_store
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import get_verified_program_sharding_info
from ..utils.rotation_matrix import build_rotation_matrix
from .rotational_topk_utils import (
    HW_PARAMS,
    RotationalTopkConfig,
    TopkConfig,
    build_stage_offsets,
    create_rotational_topk_config,
    create_topk_config,
    insert,
    naive_scanning_topk,
    reshape_with_dma,
    rotate,
    sort,
    topk_core,
    validate_config,
    validate_topk_input,
)


class SupportedTopkMethods(Enum):
    """Enumeration of supported top-k algorithm methods."""

    SCANNING = 0
    CASCADED = 1
    ROTATIONAL = 2


@nki.jit
def rotational_topk(inp: nl.NkiTensor, config: RotationalTopkConfig) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Find the k largest elements along the last dimension using rotational algorithm.

    This kernel implements a multi-stage rotational reduction algorithm that efficiently
    finds top-k elements by rotating local maxima across stages and accumulating results.
    The algorithm is optimized for NeuronCore architecture with support for LNC sharding.

    Dimensions:
        B: Batch size
        S: Sequence length
        V: Vocabulary size (dimension to reduce over)
        k: Number of top elements to retrieve

    Args:
        inp (nl.NkiTensor): [B, S, V] or [BxS, V], Input tensor in HBM
        config (RotationalTopkConfig): Configuration object containing algorithm parameters

    Returns:
        Tuple[nl.NkiTensor, nl.NkiTensor]: A tuple containing:
            - topk_values: [B, S, k], Top-k values with original shape preserved
            - topk_indices: [B, S, k], Global indices of top-k elements

    Notes:
        - Falls back to scanning approach if only 1 stage fits in memory
        - Supports optional sorting of output via config.sorted flag
        - Uses LNC sharding for parallel execution across multiple cores
        - Handles padding when k is not divisible by 8
        - Optimizes tile size based on vocab_size, k, and sort requirements
        - HW constraints:
            * vocab_size/n_stages must be <= 2^14 (DVE instruction limit)
        - Supports tiling over BxS dimension when BxS > 128
        - Tested range: vocab_size up to 151,936, k up to 2,048, batch up to 1,024

    Pseudocode:
        # Validate inputs
        validate_topk_input(inp)
        validate_config(config.topk_config)

        # Handle single-stage case
        if n_stages == 1:
            return naive_scanning_topk(inp, config.topk_config)

        # Multi-stage rotational algorithm per tile
        value, global_index = _topk_rotated_core(inp, config, n_programs, program_id)

        # Optional sorting (per tile)
        if sorted:
            flat_value = reshape_with_dma(value, n_stages)
            flat_index = reshape_with_dma(global_index, n_stages)
            sorted_val, sorted_idx = sort(flat_value, flat_index)
            dma_copy(sorted_val[:true_k], sorted_idx[:true_k] to HBM)
        else:
            unfolded_store(value, global_index to HBM)

        return topk_values, topk_indices
    """
    validate_topk_input(inp, n_fold=config.n_stages, local_top_k_per_stage=config.local_top_k_per_stage)
    validate_config(config.topk_config)

    # Query runtime shard info (replaces the old config.update_shard_info() call).
    # prg_id and n_prgs must come from the NKI runtime context inside the kernel,
    # not from config construction time outside the kernel.
    shard_info = get_verified_program_sharding_info("topk", (0, 1), 2)
    kernel_assert(shard_info[1] == config.n_prgs or config.BxS == 1, "n_prgs mismatch")
    if config.BxS > 1:
        n_prgs = shard_info[1]
        prg_id = shard_info[2]
    else:
        kernel_assert(config.n_prgs == 1, f"n_prgs mismatch, BxS {config.BxS}, n_programs {config.n_prgs}")
        n_prgs = config.n_prgs
        prg_id = config.prg_id

    BxS = config.BxS
    true_k = config.orig_k
    sorted_flag = config.sorted
    index_dtype = config.topk_config.index_dtype
    output_shape = (BxS, true_k)

    # Trivial case: k == vocab_size, return input as-is with sequential indices.
    if true_k == config.vocab_size:
        kernel_assert(
            not sorted_flag,
            f"sorted=True is not supported when k == vocab_size ({true_k}). Use k < vocab_size for sorted output.",
        )
        P_MAX = nl.tile_size.pmax
        topk_indices = nl.ndarray(output_shape, dtype=index_dtype, buffer=nl.shared_hbm)

        tile_rows = min(BxS, P_MAX)
        idx_sb = nl.ndarray((tile_rows, true_k), dtype=index_dtype, buffer=nl.sbuf)
        nisa.iota(idx_sb, [[1, true_k]], offset=0)

        n_full_tiles = BxS // tile_rows
        remainder = BxS % tile_rows
        for tile_idx in nl.affine_range(n_full_tiles):
            nisa.dma_copy(dst=topk_indices[nl.ds(tile_idx * tile_rows, tile_rows), :], src=idx_sb)
        if remainder > 0:
            nisa.dma_copy(
                dst=topk_indices[nl.ds(n_full_tiles * tile_rows, remainder), :], src=idx_sb[nl.ds(0, remainder), :]
            )

        return inp, topk_indices

    # Handle single-stage case (falls back to scanning)
    if config.n_stages == 1:
        # Create a runtime-corrected TopkConfig with the actual prg_id/n_prgs
        # (the original update_shard_info() did this by reconstructing topk_config)
        runtime_topk_config = TopkConfig(
            inp_shape=config.topk_config.inp_shape,
            k=config.orig_k,
            sorted=config.sorted,
            inp_dtype=config.inp_dtype,
            index_dtype=config.index_dtype,
            BxS=config.topk_config.BxS,
            vocab_size=config.topk_config.vocab_size,
            out_shape=config.topk_config.out_shape,
            n_prgs=n_prgs,
            prg_id=prg_id,
            per_lnc_BxS=config.topk_config.per_lnc_BxS,
            _pmax=config.topk_config._pmax,
        )
        topk_values, topk_indices = naive_scanning_topk(inp=inp, topk_config=runtime_topk_config)
        return topk_values, topk_indices

    topk_values = nl.ndarray(output_shape, dtype=inp.dtype, buffer=nl.shared_hbm)
    topk_indices = nl.ndarray(output_shape, dtype=index_dtype, buffer=nl.shared_hbm)

    tile_size = config.tile_size
    n_bxs_tiles = config.n_bxs_tiles
    lnc_batch_start = prg_id * config.per_lnc_BxS

    # Hoist tile-invariant constants out of the loop
    n_stages = config.n_stages
    stage_free_size = config.stage_free_size
    total_partition_dim = n_stages * tile_size
    concatenated_stage_free_dim = stage_free_size + (n_stages * config.local_top_k_per_stage)

    indices = nl.ndarray((total_partition_dim, concatenated_stage_free_dim), dtype=nl.float32)
    nisa.iota(
        dst=indices[:, nl.ds(0, stage_free_size)],
        pattern=[[1, stage_free_size]],
        offset=0,
    )

    stage_offsets = build_stage_offsets(n_stages, tile_size, stage_free_size)

    rotation = build_rotation_matrix(n_stages, tile_size, dtype=inp.dtype)
    rotation_f32 = nl.ndarray((total_partition_dim, total_partition_dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rotation_f32, src=rotation, engine=nisa.vector_engine)

    nisa.tensor_scalar(
        dst=indices[:, nl.ds(0, stage_free_size)],
        data=indices[:, nl.ds(0, stage_free_size)],
        op0=nl.add,
        operand0=stage_offsets,
        engine=nisa.vector_engine,
    )

    for tile_idx in nl.sequential_range(n_bxs_tiles):
        tile_batch_start = lnc_batch_start + tile_idx * tile_size
        tile_batch_end = min(tile_batch_start + tile_size, min(lnc_batch_start + config.per_lnc_BxS, BxS))

        value, global_index = _topk_rotated_core(
            inp=inp,
            config=config,
            batch_start=tile_batch_start,
            batch_end=tile_batch_end,
            rotation=rotation,
            rotation_f32=rotation_f32,
            indices=indices,
        )

        tile_bxs = tile_batch_end - tile_batch_start
        hbm_slice = nl.ds(tile_batch_start, tile_bxs)
        sbuf_slice = nl.ds(0, tile_bxs)

        if sorted_flag:
            flat_value = reshape_with_dma(value, config.n_stages, dtype=inp.dtype)
            flat_index = reshape_with_dma(global_index, config.n_stages, dtype=index_dtype)

            trimmed_val, trimmed_idx = sort(flat_value, flat_index, true_k)
            nisa.dma_copy(dst=topk_indices[hbm_slice, :true_k], src=trimmed_idx[sbuf_slice, :true_k])
            nisa.dma_copy(dst=topk_values[hbm_slice, :true_k], src=trimmed_val[sbuf_slice, :true_k])
        else:
            global_index_int = nl.ndarray(global_index.shape, dtype=index_dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=global_index_int, src=global_index)

            unfolded_store(
                global_index_int[:, :],
                topk_indices,
                fold_factor=config.n_stages,
                batch_start=tile_batch_start,
                batch_end=tile_batch_end,
            )
            unfolded_store(
                value[:, :],
                topk_values,
                fold_factor=config.n_stages,
                batch_start=tile_batch_start,
                batch_end=tile_batch_end,
            )

    return topk_values, topk_indices


def _topk_rotated_core(
    inp: nl.NkiTensor,
    config: RotationalTopkConfig,
    batch_start: int,
    batch_end: int,
    rotation: nl.ndarray,
    rotation_f32: nl.ndarray,
    indices: nl.ndarray,
) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Core rotational top-k algorithm implementation.

    Performs multi-stage rotation and reduction to find top-k elements efficiently
    by rotating local maxima across stages and accumulating results.
    Uses on-chip index generation via nisa.iota + nisa.tensor_scalar,
    fast_folded_load with targeted padding, and skips rotation on the last stage.

    Args:
        inp (nl.NkiTensor): [BxS, V], Input tensor in HBM
        config (RotationalTopkConfig): Configuration with algorithm parameters
        batch_start (int): Start index for batch tile
        batch_end (int): End index for batch tile

    Returns:
        Tuple[nl.NkiTensor, nl.NkiTensor]: A tuple containing:
            - value: [total_partition_dim, local_top_k_per_stage], Top-k values
            - global_index: [total_partition_dim, local_top_k_per_stage], Global indices

    Pseudocode:
        # Initialize buffers with on-chip index generation
        values = folded_load(inp, n_stages)
        indices = iota(0..stage_free_size) + stage_offsets
        rotation_matrix = load_circulant_permutation(n_stages, BxS)

        # Iterative rotation and top-k
        for stage_idx in range(n_stages):
            offset = stage_free_size + (local_top_k * stage_idx)
            local_vals, local_idx = topk_core(values[:, :offset], k=local_top_k)
            global_idx = gather(indices, local_idx)
            if stage_idx < n_stages - 1:
                rotated_vals = matmul(rotation_matrix, local_vals)
                rotated_idx = matmul(rotation_matrix, global_idx)
                values[:, offset:offset+local_top_k] = rotated_vals
                indices[:, offset:offset+local_top_k] = rotated_idx

        return local_vals, global_idx
    """
    n_stages = config.n_stages
    local_top_k_per_stage = config.local_top_k_per_stage
    stage_free_size = config.stage_free_size
    BxS_size = config.tile_size

    total_partition_dim = n_stages * BxS_size
    concatenated_stage_free_dim = stage_free_size + (n_stages * local_top_k_per_stage)

    values = nl.ndarray((total_partition_dim, concatenated_stage_free_dim), dtype=inp.dtype)

    predicated_folded_load(
        data_hbm=inp,
        fold_factor=n_stages,
        data_sb=values,
        batch_start=batch_start,
        batch_end=batch_end,
    )

    # predicated_folded_load only writes this shard's valid BxS elements, which
    # occupy partitions [0:valid_pdim] (n_stages consecutive partitions per
    # element). Restrict every downstream op to those partitions so none reads
    # the padding partitions [valid_pdim:total_partition_dim] that the load
    # leaves uninitialized. The rotation matrix is block-diagonal with
    # one n_stages-wide block per element, so its [0:valid_pdim, 0:valid_pdim]
    # sub-block is exactly the rotation for the valid partitions.
    valid_pdim = (batch_end - batch_start) * n_stages
    values = values[0:valid_pdim, :]
    indices = indices[0:valid_pdim, :]
    rotation = rotation[0:valid_pdim, 0:valid_pdim]
    rotation_f32 = rotation_f32[0:valid_pdim, 0:valid_pdim]

    for stage_idx in nl.static_range(n_stages):
        offset = stage_free_size + (local_top_k_per_stage * stage_idx)

        value, local_index = topk_core(data=values[:, :offset], k=local_top_k_per_stage)

        global_index = nl.ndarray(local_index.shape, dtype=indices.dtype, buffer=nl.sbuf)
        # Tile the gather into chunks no wider than the nc_n_gather ISA group size. A
        # single gather wider than this splits into multiple internal ISA groups, and
        # that multi-group form corrupts the tail elements of the last BxS tile on
        # hardware while the simulator (which executes the gather atomically) stays
        # correct (NKILIB-1592).
        gather_group_size = HW_PARAMS.gather_group_size
        gather_width = local_index.shape[1]
        n_gather_tiles = (gather_width + gather_group_size - 1) // gather_group_size
        for gather_tile in nl.static_range(n_gather_tiles):
            chunk = min(gather_group_size, gather_width - gather_tile * gather_group_size)
            chunk_slice = nl.ds(gather_tile * gather_group_size, chunk)
            nisa.nc_n_gather(
                dst=global_index[:, chunk_slice],
                data=indices[:, :offset],
                indices=local_index[:, chunk_slice],
            )

        if stage_idx < n_stages - 1:
            rotated_index = nl.ndarray(global_index.shape, dtype=nl.float32, buffer=nl.psum)
            rotated = nl.ndarray(value.shape, dtype=nl.float32, buffer=nl.psum)

            rotate(dst=rotated_index, tensor=global_index, rotation_matrix=rotation_f32)
            rotate(dst=rotated, tensor=value, rotation_matrix=rotation)

            insert(tensor=values, values=rotated, offset=offset)
            insert(tensor=indices, values=rotated_index, offset=offset)

    return value, global_index


SUPPORTED_TOPK_METHOD_MAPPING = {
    SupportedTopkMethods.SCANNING: naive_scanning_topk,
    SupportedTopkMethods.ROTATIONAL: rotational_topk,
}


def _kernel(fn):
    """
    Decorator to create kernel wrapper with grid support.

    Creates a wrapper class that enables grid-based kernel launching syntax
    using bracket notation (e.g., kernel[grid](...)).

    Args:
        fn: Function to wrap with grid support

    Returns:
        Wrapper: Wrapper instance with __getitem__ support for grid launching
    """

    class Wrapper:
        __name__: str = "topk_kernel"

        def __getitem__(self, grid):
            def launcher(*args, **kw):
                return fn(*args, **kw, lnc=grid)

            return launcher

    return Wrapper()


@_kernel
def topk(
    inp: nl.NkiTensor,
    k: int,
    sorted_flag: bool = True,
    method: SupportedTopkMethods = SupportedTopkMethods.ROTATIONAL,
    lnc: Optional[int] = None,
) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Find the k largest elements along the last dimension of input tensor.

    This is the main entry point for top-k operations, supporting multiple algorithm
    implementations (scanning, rotational) with automatic method selection and
    configuration.

    Dimensions:
        B: Batch size
        S: Sequence length
        V: Vocabulary size (dimension to reduce over)
        k: Number of top elements to retrieve

    Args:
        inp (nl.NkiTensor): [B, S, V], Input tensor in HBM
        k (int): Number of top elements to retrieve
        sorted_flag (bool): Whether to sort the output (default: True)
        method (SupportedTopkMethods): Algorithm to use (default: ROTATIONAL)
        lnc (Optional[int]): Number of logical cores to use (default: None, auto-detect)

    Returns:
        Tuple[nl.NkiTensor, nl.NkiTensor]: A tuple containing:
            - topk_values: [B, S, k], Top-k values
            - topk_indices: [B, S, k], Indices of top-k elements

    Notes:
        - Automatically fuses batch and sequence dimensions for processing
        - Validates configuration before execution
        - Supports LNC sharding for parallel execution
        - Tested range: vocab_size up to 151,936, k up to 2,048, batch up to 1,024
        - Performance: ~4% mean difference vs historical benchmarks

    Pseudocode:
        # Validate method
        if method not in SupportedTopkMethods:
            raise ValueError("Unsupported method")

        # Initialize and validate configuration
        topk_config = TopkConfig(inp.shape, inp.dtype, k, sorted_flag, lnc)
        validate_config(topk_config)

        # Reshape input to 2D
        inp_2d = inp.reshape([BxS, vocab_size])

        # Create method-specific configuration
        config = RotationalTopkConfig(inp_2d.shape, topk_config)

        # Invoke selected method
        topk_values, topk_indices = selected_method(inp_2d, config)

        # Reshape outputs to original shape
        return topk_values.reshape(original_shape), topk_indices.reshape(original_shape)
    """
    if method not in SupportedTopkMethods:  # pragma: no cover - arg validation, not a functional branch
        raise ValueError(f"Unsupported method '{method}'. Supported methods are: {list(SupportedTopkMethods)}")

    topk_config = create_topk_config(
        inp_shape=inp.shape,
        inp_dtype=getattr(nl, str(inp.dtype).split(".")[-1]),
        k=k,
        sorted=sorted_flag,
        num_programs=lnc or 2,
    )
    kernel_assert(topk_config.is_valid(), f"top k config {topk_config.__dict__} is not valid")
    kernel_assert(
        topk_config.n_prgs == lnc or topk_config.BxS == 1,
        f"num programs mismatch user {lnc}, derived {topk_config.n_prgs}",
    )
    inp = inp.reshape((topk_config.BxS, topk_config.vocab_size))
    selected_topk_method = SUPPORTED_TOPK_METHOD_MAPPING[method]

    config = create_rotational_topk_config(inp_shape=inp.shape, topk_config=topk_config)
    config.log_strategy()
    grid = config.n_prgs
    topk_values, topk_indices = selected_topk_method[grid](inp=inp, config=config)

    topk_values = topk_values.reshape(topk_config.out_shape)
    topk_indices = topk_indices.reshape(topk_config.out_shape)

    return topk_values, topk_indices
