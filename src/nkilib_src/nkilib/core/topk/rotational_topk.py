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

"""This kernel implements rotational top-k algorithm to find the k largest elements along a dimension using multi-stage rotation and reduction optimized for NeuronCore architecture."""

from enum import Enum
from typing import Optional, Tuple

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

from ..max.cascaded_max_utils import predicated_folded_load, unfolded_store
from ..utils.kernel_assert import kernel_assert
from .rotational_topk_utils import (
    RotationalConstants,
    RotationalTopkConfig,
    TopkConfig,
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
def rotational_topk(inp: nl.ndarray, config: RotationalTopkConfig) -> Tuple[nl.ndarray, nl.ndarray]:
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
        inp (nl.ndarray): [B, S, V] or [BxS, V], Input tensor in HBM
        config (RotationalTopkConfig): Configuration object containing algorithm parameters

    Returns:
        Tuple[nl.ndarray, nl.ndarray]: A tuple containing:
            - topk_values: [B, S, k], Top-k values with original shape preserved
            - topk_indices: [B, S, k], Global indices of top-k elements

    Notes:
        - Falls back to scanning approach if only 1 stage fits in memory
        - Supports optional sorting of output via config.sorted flag
        - Uses LNC sharding for parallel execution across multiple cores
        - Handles padding when k is not divisible by 8
        - Optimizes tile size based on vocab_size, k, and sort requirements
        - HW constraint: vocab_size/n_stages must be <= 2^14
        - TODO: Tile over BxS dimension when BxS > 128
        - TODO: Relax BxS <= 128 restriction
        - TODO: Specify intended usage range (e.g., vocabulary size, k value)

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
    config.update_shard_info()

    validate_topk_input(inp, n_fold=config.n_stages, local_top_k_per_stage=config.local_top_k_per_stage)
    validate_config(config.topk_config)

    BxS = config.BxS
    true_k = config.orig_k
    sorted_flag = config.sorted
    index_dtype = config.topk_config.index_dtype
    output_shape = (BxS, true_k)

    # Handle single-stage case (falls back to scanning)
    if config.n_stages == 1:
        topk_values, topk_indices = naive_scanning_topk(inp=inp, topk_config=config.topk_config)
        return topk_values, topk_indices

    topk_values = nl.ndarray(output_shape, dtype=inp.dtype, buffer=nl.shared_hbm)
    topk_indices = nl.ndarray(output_shape, dtype=index_dtype, buffer=nl.shared_hbm)

    value, global_index = _topk_rotated_core(inp=inp, config=config, n_programs=config.n_prgs, program_id=config.prg_id)

    # Calculate actual BxS handled by this LNC
    max_per_lnc_bxs = min(config.per_lnc_BxS, BxS - config.prg_id * config.per_lnc_BxS)
    hbm_slice = nl.ds(config.prg_id * config.per_lnc_BxS, max_per_lnc_bxs)
    sbuf_slice = nl.ds(0, max_per_lnc_bxs)

    if sorted_flag:
        flat_value = reshape_with_dma(value, config.n_stages, dtype=inp.dtype)
        flat_index = reshape_with_dma(global_index, config.n_stages, dtype=index_dtype)

        sorted_val, sorted_global_idx = sort(flat_value, indices=flat_index)
        nisa.dma_copy(dst=topk_indices[hbm_slice, :true_k], src=sorted_global_idx[sbuf_slice, :true_k])
        nisa.dma_copy(dst=topk_values[hbm_slice, :true_k], src=sorted_val[sbuf_slice, :true_k])
    else:
        global_index_int = nl.ndarray(global_index.shape, dtype=index_dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=global_index_int, src=global_index)

        unfolded_store(
            global_index_int[:, :],
            topk_indices,
            fold_factor=config.n_stages,
            program_id=config.prg_id,
            n_programs=config.n_prgs,
        )
        unfolded_store(
            value[:, :],
            topk_values,
            fold_factor=config.n_stages,
            program_id=config.prg_id,
            n_programs=config.n_prgs,
        )

    return topk_values, topk_indices


def _topk_rotated_core(
    inp: nl.ndarray,
    config: RotationalTopkConfig,
    n_programs: int,
    program_id: int,
) -> Tuple[nl.ndarray, nl.ndarray]:
    """
    Core rotational top-k algorithm implementation.

    Performs multi-stage rotation and reduction to find top-k elements efficiently
    by rotating local maxima across stages and accumulating results.

    Args:
        inp (nl.ndarray): [BxS, V], Input tensor in HBM
        config (RotationalTopkConfig): Configuration with algorithm parameters
        n_programs (int): Total number of parallel programs
        program_id (int): Current program ID

    Returns:
        Tuple[nl.ndarray, nl.ndarray]: A tuple containing:
            - value: [total_partition_dim, local_top_k_per_stage], Top-k values
            - global_index: [total_partition_dim, local_top_k_per_stage], Global indices

    Notes:
        - This is a hidden helper function for internal use
        - Assumes input is already validated and properly shaped

    Pseudocode:
        # Initialize buffers
        values = folded_load(inp, n_stages)  # [n_stages * BxS, stage_free_size]
        indices = iota(0..padded_vocab_size)  # Global index tracking
        rotation_matrix = load_circulant_permutation(n_stages, BxS)

        # Iterative rotation and top-k
        for stage_idx in range(n_stages):
            offset = stage_free_size + (local_top_k * stage_idx)

            # Find local top-k from current values
            local_vals, local_idx = topk_core(values[:, :offset], k=local_top_k)

            # Gather global indices corresponding to local top-k
            global_idx = gather(indices, local_idx)

            # Rotate values and indices across partitions
            rotated_vals = matmul(rotation_matrix, local_vals)
            rotated_idx = matmul(rotation_matrix, global_idx)

            # Append rotated results for next stage
            values[:, offset:offset+local_top_k] = rotated_vals
            indices[:, offset:offset+local_top_k] = rotated_idx

        return local_vals, global_idx
    """
    n_stages = config.n_stages
    local_top_k_per_stage = config.local_top_k_per_stage
    stage_free_size = config.stage_free_size
    BxS_size = config.per_lnc_BxS
    padded_vocab_size = config.padded_vocab_size

    total_partition_dim = n_stages * BxS_size
    concatenated_stage_free_dim = stage_free_size + (n_stages * local_top_k_per_stage)

    rotation_matrix_file = config._shared_const_cache[f"{n_stages}_{BxS_size}"]
    rotate_hbm = builtin.lang.shared_constant(rotation_matrix_file)

    values = nl.ndarray(
        (
            total_partition_dim,
            concatenated_stage_free_dim,
        ),
        dtype=inp.dtype,
    )
    indices = nl.ndarray(
        (
            total_partition_dim,
            concatenated_stage_free_dim,
        ),
        dtype=nl.float32,
    )
    nisa.memset(dst=indices, value=0)
    nisa.dma_copy(
        dst=indices[:, :stage_free_size],
        src=builtin.lang.shared_constant(
            config._shared_const_cache[f"{padded_vocab_size}_{n_stages}_{stage_free_size}_{BxS_size}"]
        ),
    )

    partition_slice = nl.ds(0, total_partition_dim)
    free_slice = nl.ds(0, stage_free_size)

    predicated_folded_load(
        data_hbm=inp, fold_factor=n_stages, n_programs=n_programs, program_id=program_id, data_sb=values
    )
    kernel_assert(
        values.shape == (total_partition_dim, concatenated_stage_free_dim),
        f"shape mismatch expected {(total_partition_dim, stage_free_size)} but got {values.shape}",
    )
    kernel_assert(
        indices.shape == (total_partition_dim, concatenated_stage_free_dim),
        f"shape mismatch expected {(total_partition_dim, stage_free_size)} but got {indices.shape}",
    )
    free_slice = nl.ds(0, total_partition_dim)
    rotation = nl.ndarray(rotate_hbm.shape, dtype=inp.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=rotation, src=rotate_hbm[partition_slice, free_slice])

    # Create float32 rotation matrix for indices when input is not float32
    rotation_f32 = nl.ndarray(rotate_hbm.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rotation_f32, src=rotation, dtype=nl.float32)

    local_index = nl.ndarray(
        shape=(total_partition_dim, local_top_k_per_stage), dtype=config.index_dtype, buffer=nl.sbuf
    )
    nisa.memset(dst=local_index, value=0)

    for stage_idx in nl.static_range(n_stages):
        offset = stage_free_size + (local_top_k_per_stage * stage_idx)

        value, local_index = topk_core(data=values[:, :offset], k=local_top_k_per_stage)

        global_index = nl.ndarray(local_index.shape, dtype=indices.dtype, buffer=nl.sbuf)
        nisa.nc_n_gather(dst=global_index, data=indices, indices=local_index)

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
    inp: nl.ndarray,
    k: int,
    sorted_flag: bool = True,
    method: SupportedTopkMethods = SupportedTopkMethods.ROTATIONAL,
    lnc: Optional[int] = None,
) -> Tuple[nl.ndarray, nl.ndarray]:
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
        inp (nl.ndarray): [B, S, V], Input tensor in HBM
        k (int): Number of top elements to retrieve
        sorted_flag (bool): Whether to sort the output (default: True)
        method (SupportedTopkMethods): Algorithm to use (default: ROTATIONAL)
        lnc (Optional[int]): Number of logical cores to use (default: None, auto-detect)

    Returns:
        Tuple[nl.ndarray, nl.ndarray]: A tuple containing:
            - topk_values: [B, S, k], Top-k values
            - topk_indices: [B, S, k], Indices of top-k elements

    Notes:
        - Automatically fuses batch and sequence dimensions for processing
        - Validates configuration before execution
        - Supports LNC sharding for parallel execution
        - TODO: Specify intended usage range (e.g., vocabulary size, k value)

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
    if method not in SupportedTopkMethods:
        raise ValueError(f"Unsupported method '{method}'. Supported methods are: {list(SupportedTopkMethods)}")

    topk_config = TopkConfig(
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

    config = RotationalTopkConfig(inp_shape=inp.shape, topk_config=topk_config)
    prepare_rotational_constants(config)
    grid = config.n_prgs
    topk_values, topk_indices = selected_topk_method[grid](inp=inp, config=config)

    topk_values = topk_values.reshape(topk_config.out_shape)
    topk_indices = topk_indices.reshape(topk_config.out_shape)
    cleanup_rotational_constants()

    return topk_values, topk_indices


def prepare_rotational_constants(config: RotationalTopkConfig) -> None:
    """Prepare rotational constants for topk kernel execution.

    Args:
        config: RotationalTopkConfig with kernel parameters
    """
    # Use float32 for constants - indices need precision for large values
    const_dtype = np.float32
    RotationalConstants._get_permutation_matrix(config.n_stages, config.per_lnc_BxS, const_dtype)
    RotationalConstants._get_global_indices(
        config.n_stages,
        config.stage_free_size,
        config.per_lnc_BxS,
        config.padded_vocab_size,
        const_dtype,
    )
    config._shared_const_cache = RotationalConstants._shared_const_cache


def cleanup_rotational_constants() -> None:
    """Cleanup rotational constants after topk kernel execution."""
    RotationalConstants.cleanup()
