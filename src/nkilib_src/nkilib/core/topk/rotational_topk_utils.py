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

"""Utility functions and configuration classes for top-k operations including rotational algorithm support and scanning implementations."""

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from tempfile import NamedTemporaryFile
from typing import List, Optional, Tuple

import nki.isa as nisa
import nki.language as nl
import numpy as np
from scipy.linalg import circulant

from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from ..utils.logging import get_logger

logger = get_logger("topk")

FLOAT32_MIN = np.finfo(np.float32).min.item()
BFLOAT16_MIN = -9948.0


def _get_dtype_min(dtype):
    """Get minimum value for padding based on dtype."""
    if dtype == nl.bfloat16:
        return BFLOAT16_MIN
    return FLOAT32_MIN


@dataclass
class TopkHardwareParams(nl.NKIObject):
    """
    Hardware parameters for top-k operations.

    Encapsulates hardware-specific constants that may vary across Trainium generations.

    Attributes:
        dve_max_alus (int): Maximum ALUs available in DVE engine
        topk_per_stage (int): Number of top-k elements found per DVE pass
        index_dtype: Data type for indices
        num_sbuf_quadrants (int): Number of SBUF quadrants
        fixed_dve_inst_overhead (int): Fixed DVE instruction overhead in cycles
        max_free_dim (int): Maximum free dimension size (2^14 for DVE instructions)
    """

    dve_max_alus: int = 8
    topk_per_stage: int = field(init=False)
    index_dtype: type = nl.uint32
    num_sbuf_quadrants: int = 4
    fixed_dve_inst_overhead: int = 144
    max_free_dim: int = 2**14

    def __post_init__(self):
        self.topk_per_stage = self.dve_max_alus


HW_PARAMS = TopkHardwareParams()


def reduce(op: str = 'mul', input_list: Optional[List] = None, initial_value=None):
    """
    Apply reduction operation over a list of values.

    Args:
        op (str): Operation to apply ('mul', 'add', 'max', 'min')
        input_list (Optional[List]): List of values to reduce
        initial_value: Starting value for reduction

    Returns:
        Reduced value after applying operation
    """
    supported_ops = ['mul', 'add', 'max', 'min']
    kernel_assert(initial_value is not None, "initial_value must be set")
    kernel_assert(input_list is not None, "input_list must be set")
    kernel_assert(op in supported_ops, f"only ops in {supported_ops} are supported, got {op}")
    for element in input_list:
        if op == 'mul':
            initial_value = initial_value * element
        elif op == 'add':
            initial_value = initial_value + element
        elif op == 'min':
            initial_value = min(initial_value, element)
        elif op == 'max':
            initial_value = max(initial_value, element)
    return initial_value


@dataclass
class TopkConfig(nl.NKIObject):
    """
    Configuration class for top-k algorithm.

    Generic configuration for top-k algorithms that accept inputs [B, S, V] and
    perform top-k reduction along vocabulary dimension to produce [B, S, k].

    Args:
        inp_shape (Tuple): Shape of input tensor (2D or 3D)
        inp_dtype (np.dtype): Data type of input tensor
        k (int): Number of top elements to retrieve
        sorted (bool): Whether to sort the output (default: True)
        num_programs (int): Number of logical cores (default: 2)

    Attributes:
        inp_shape (Tuple): Input tensor shape
        k (int): Number of top elements
        sorted (bool): Sort flag
        inp_dtype (np.dtype): Input data type
        index_dtype (np.dtype): Index data type (nl.uint32)
        BxS (int): Combined batch and sequence dimensions
        vocab_size (int): Vocabulary dimension size
        out_shape (list): Output shape [B, S, k]
        n_prgs (int): Number of logical cores
        prg_id (int): Program ID for SPMD grid
        per_lnc_BxS (int): Batch size per logical core
    """

    def __init__(
        self, inp_shape: Tuple, inp_dtype: np.dtype, k: int, sorted: bool = True, num_programs: int = 2
    ) -> None:
        if nl.tile_size.pmax > 0:
            self._pmax = nl.tile_size.pmax
        else:
            self._pmax = 128
        self.inp_shape = inp_shape
        self.k = k
        self.sorted = sorted
        self.inp_dtype = inp_dtype
        self.index_dtype = HW_PARAMS.index_dtype

        self.BxS = reduce('mul', self.inp_shape[:-1], 1)
        self.vocab_size = inp_shape[-1]
        self.out_shape = []
        for dim_size in inp_shape[:-1]:
            self.out_shape.append(dim_size)
        self.out_shape.append(self.k)

        # Handle LNC sharding configuration.
        # Eventually hope to remove LNC info from config once NKI support allows
        # for helper functions to be LNC agnostic.
        shard_info = get_verified_program_sharding_info("topk", (0, 1), 2)
        self.n_prgs = shard_info[1] or num_programs
        self.prg_id = shard_info[2] or 0

        if self.BxS == 1:
            logger.info(f"Setting num_programs to 1 since {self.BxS}, user specified num_programs {self.n_prgs}")
            self.prg_id = 0
            self.n_prgs = 1

        self.per_lnc_BxS = (self.BxS + self.n_prgs - 1) // self.n_prgs
        kernel_assert(self.inp_shape_valid(), "topk expects input to be at least 2D")
        kernel_assert(self.vocab_size_valid(), f"topk expects vocab_size ({self.vocab_size}) >= k ({self.k})")

    def inp_shape_valid(self) -> bool:
        return len(self.inp_shape) >= 2

    def vocab_size_valid(self) -> bool:
        return self.vocab_size >= self.k

    def is_valid(self) -> bool:
        return self.inp_shape_valid() and self.vocab_size_valid()

    def cost_estimate(self) -> int:
        """
        Estimate DVE clock cycles for scanning approach.

        A scanning approach scans the entire vocab size k/8 times. Each scan requires
        2 passes to find the top 8 and replace them with -inf.

        Returns:
            int: Estimated number of DVE clock cycles required

        Notes:
            - Provides static analysis of instruction count
            - Actual performance may vary based on memory access patterns
            - DVE instructions account for fixed DVE instruction overhead
        """
        return div_ceil(self.k, HW_PARAMS.dve_max_alus) * 2 * (self.vocab_size + HW_PARAMS.fixed_dve_inst_overhead)


class RotationalConstants:
    """Helper class for managing rotational algorithm constants and shared cache."""

    _shared_const_cache = {}

    def _get_permutation_matrix(block_size, num_blocks, inp_dtype):
        """
        Generate permutation matrix for rotational algorithm.

        Args:
            block_size (int): Size of each block
            num_blocks (int): Number of blocks
            inp_dtype: Data type for matrix

        Returns:
            None: Matrix is saved to temporary file and cached

        Notes:
            - Creates circulant block-diagonal matrix using Kronecker product
            - Saves result to temporary file for shared constant access
        """
        shift = 1

        base_perm = np.zeros(block_size)
        base_perm[shift % block_size] = 1
        P_block = circulant(base_perm)

        I_blocks = np.eye(num_blocks)
        B = np.kron(I_blocks, P_block)
        out = B.astype(inp_dtype)
        cache_key = map(str, (block_size, num_blocks))
        with NamedTemporaryFile(suffix='.npy', delete=False) as f:
            np.save(f, out)
        RotationalConstants._shared_const_cache['_'.join(cache_key)] = f.name

    def _get_global_indices(n_stages, stage_free_size, per_lnc_BxS, padded_vocab_size, inp_dtype):
        """
        Generate global index array for rotational algorithm.

        Args:
            n_stages (int): Number of stages
            stage_free_size (int): Free dimension size per stage
            per_lnc_BxS (int): Batch size per logical core
            padded_vocab_size (int): Padded vocabulary size
            inp_dtype: Data type for indices

        Returns:
            None: Index array is saved to temporary file and cached

        Notes:
            - Creates tiled index array for tracking global positions
            - Saves result to temporary file for shared constant access
        """
        BxS_size = per_lnc_BxS
        out = np.tile(
            np.arange(padded_vocab_size).astype(inp_dtype).reshape((n_stages, stage_free_size)),
            (BxS_size, 1),
        )
        cache_key = '_'.join(map(str, (padded_vocab_size, n_stages, stage_free_size, BxS_size)))
        with NamedTemporaryFile(suffix='.npy', delete=False) as f:
            np.save(f, out)
            RotationalConstants._shared_const_cache[cache_key] = f.name

    def cleanup():
        """
        Clean up temporary files created for shared constants.

        Returns:
            None: Removes temporary files from filesystem
        """
        for file_path in RotationalConstants._shared_const_cache.values():
            if os.path.exists(file_path):
                os.remove(file_path)


@dataclass
class RotationalTopkConfig(nl.NKIObject):
    """
    Configuration class for rotational top-k algorithm.

    Configures rotational top-k algorithm that accepts inputs [BxS, V] and performs
    top-k reduction along vocabulary dimension. May use padded k for efficiency.
    Supports tiling over BxS dimension for BxS > 128.

    Args:
        inp_shape (Tuple): Shape of input tensor (must be 2D)
        topk_config (TopkConfig): Base top-k configuration

    Attributes:
        inp_shape (Tuple): Input tensor shape
        BxS (int): Combined batch and sequence dimensions
        vocab_size (int): Vocabulary dimension size
        orig_k (int): Original number of top elements requested
        padded_k (int): Padded number of top elements (for efficiency)
        n_prgs (int): Number of logical cores
        prg_id (int): Program ID for SPMD grid
        per_lnc_BxS (int): Batch size per logical core
        inp_dtype (np.dtype): Input data type
        index_dtype (np.dtype): Index data type
        local_top_k_per_stage (int): Local top-k per stage
        n_stages (int): Number of rotational stages
        stage_free_size (int): Free dimension size per stage
        padded_vocab_size (int): Padded vocabulary size
        sorted (bool): Whether to sort output (always True if padded_k != orig_k)
        tile_size (int): Optimal tile size for BxS dimension
        n_bxs_tiles (int): Number of tiles over BxS dimension
    """

    _shared_const_cache: dict = None

    def __init__(self, inp_shape: Tuple, topk_config: TopkConfig) -> None:
        if nl.tile_size.pmax > 0:
            self._pmax = nl.tile_size.pmax
        else:
            self._pmax = 128
        self.inp_shape = inp_shape
        kernel_assert(self.inp_shape_valid(), f"rotated topk expects input to be 2D, actual was {len(inp_shape)}")
        self.BxS = inp_shape[0]
        self.vocab_size = inp_shape[1]
        kernel_assert(
            self.BxS == topk_config.BxS, f"TopkConfig BxS dim {topk_config} does not match input BxS dim {self.BxS}"
        )

        self.topk_config = topk_config

        self.orig_k = topk_config.k
        self.n_prgs = topk_config.n_prgs
        self.prg_id = topk_config.prg_id
        self.per_lnc_BxS = topk_config.per_lnc_BxS
        self.inp_dtype = topk_config.inp_dtype
        self.index_dtype = topk_config.index_dtype

        # Find optimal tile size if BxS > PMAX
        if self.per_lnc_BxS > self._pmax:
            self.tile_size = self._find_optimal_tile_size()
            self.n_bxs_tiles = div_ceil(self.per_lnc_BxS, self.tile_size)
        else:
            self.tile_size = self.per_lnc_BxS
            self.n_bxs_tiles = 1

        const_info = self._calculate_rotational_constants(self.tile_size)
        self.local_top_k_per_stage = const_info[0]
        self.padded_k = const_info[1]
        self.n_stages = const_info[2]
        self.stage_free_size = const_info[3]
        self.padded_vocab_size = const_info[4]
        if self.orig_k != self.padded_k:
            self.sorted = True
        else:
            self.sorted = topk_config.sorted

    def _calculate_rotational_constants(self, BxS_tile: int) -> Tuple[int, int, int, int, int]:
        """
        Calculate rotational algorithm constants for given tile size.

        Args:
            BxS_tile: Tile size for BxS dimension

        Returns:
            Tuple[int, int, int, int, int]: (local_top_k_per_stage, padded_k, n_stages, chunk_size, padded_vocab_size)
        """
        P_MAX = self._pmax
        MAX_FREE_DIM = 2**14

        max_n_stages = math.floor(P_MAX // BxS_tile)
        ideal_n_stages = div_ceil(min(self.orig_k, self.vocab_size), HW_PARAMS.topk_per_stage)

        # Enforce HW constraint: vocab_size / n_stages <= 2^14
        min_n_stages_for_hw = div_ceil(self.vocab_size, MAX_FREE_DIM)

        n_stages = min(max_n_stages, ideal_n_stages)
        n_stages = max(n_stages, min_n_stages_for_hw)

        # Verify we can satisfy HW constraint
        kernel_assert(
            n_stages <= max_n_stages,
            f"Cannot satisfy HW constraint: need {min_n_stages_for_hw} stages but only {max_n_stages} fit with BxS_tile={BxS_tile}",
        )

        local_top_k_per_stage = get_ceil_aligned_size(div_ceil(self.orig_k, n_stages), HW_PARAMS.topk_per_stage)
        padded_k = local_top_k_per_stage * n_stages

        chunk_size = div_ceil(self.vocab_size, n_stages)
        padded_vocab_size = chunk_size * n_stages

        # Verify HW constraints
        kernel_assert(
            chunk_size <= HW_PARAMS.max_free_dim,
            f"HW constraint violated: stage_free_size={chunk_size} > {HW_PARAMS.max_free_dim}",
        )
        concatenated_free = chunk_size + n_stages * local_top_k_per_stage
        kernel_assert(
            concatenated_free <= HW_PARAMS.max_free_dim,
            f"HW constraint violated: concatenated_free_dim={concatenated_free} > {HW_PARAMS.max_free_dim}",
        )

        return local_top_k_per_stage, padded_k, n_stages, chunk_size, padded_vocab_size

    def _estimate_dve_cost(self, BxS_tile: int, n_stages: int) -> float:
        """
        Estimate DVE clock cycles for given tile size and n_stages.

        Args:
            BxS_tile: Batch size per tile
            n_stages: Number of rotational stages

        Returns:
            Estimated DVE clock cycles (inf if HW constraint violated)
        """
        MAX_FREE_DIM = 2**14
        stage_free_size = div_ceil(self.vocab_size, n_stages)

        # HW constraint: stage_free_size must be <= 2^14
        if stage_free_size > MAX_FREE_DIM:
            return float('inf')

        k_per_stage = get_ceil_aligned_size(div_ceil(self.orig_k, n_stages), HW_PARAMS.topk_per_stage)

        # HW constraint: concatenated free dim must fit
        if stage_free_size + n_stages * k_per_stage > MAX_FREE_DIM:
            return float('inf')

        padded_k = k_per_stage * n_stages

        # Per-stage cost (repeated n_stages times)
        per_stage_cost = div_ceil(k_per_stage, 8) * 2 * (stage_free_size + HW_PARAMS.fixed_dve_inst_overhead)
        unsorted_cost = n_stages * per_stage_cost

        # Final sort cost (if needed)
        needs_sort = self.topk_config.sorted or padded_k != self.orig_k
        if needs_sort:
            base_sort_cost = (
                div_ceil(padded_k, HW_PARAMS.dve_max_alus) * 2 * (padded_k + HW_PARAMS.fixed_dve_inst_overhead)
            )
            # Sort underutilization penalty: sort needs full 128 channels
            sort_efficiency = BxS_tile / self._pmax
            sorted_cost = base_sort_cost / sort_efficiency
        else:
            sorted_cost = 0

        return unsorted_cost + sorted_cost

    def _find_optimal_tile_size(self) -> int:
        """
        Find optimal tile size that minimizes total cost while respecting HW constraints.

        Strategy:
        - For large V, smaller tiles allow more n_stages, reducing stage_free_size
        - CRITICAL HW constraints:
            1. vocab_size/n_stages <= 2^14 (max8/match_replace8 limit)
            2. Sort needs full 128 channels for efficiency
        - Minimize: n_tiles × cost_per_tile

        Cost model:
            unsorted_cost ≈ (k×V)/(4×n_stages) + 36×k
            sorted_cost ≈ [k²/4 + 36×k] × (128/BxS_tile)  # Underutilization penalty

        Tradeoff:
            - Smaller BxS_tile → more n_stages → lower unsorted_cost
            - Smaller BxS_tile → worse sort efficiency → higher sorted_cost

        Returns:
            Optimal tile size
        """
        P_MAX = self._pmax

        # Calculate minimum n_stages required by HW constraint
        min_n_stages_for_hw = div_ceil(self.vocab_size, HW_PARAMS.max_free_dim)

        # Try different tile sizes from 1 to PMAX
        candidates = range(1, self._pmax + 1)

        best_tile_size = None
        best_cost = float('inf')

        for tile_size in candidates:
            # Calculate n_stages for this tile size
            max_n_stages = math.floor(P_MAX // tile_size)
            ideal_n_stages = div_ceil(min(self.orig_k, self.vocab_size), HW_PARAMS.topk_per_stage)
            n_stages = min(max_n_stages, ideal_n_stages)
            n_stages = max(n_stages, min_n_stages_for_hw)  # Enforce all constraints

            # Check if this configuration is feasible
            if n_stages > max_n_stages:
                continue

            # Skip if only 1 stage (falls back to scanning anyway)
            if n_stages <= 1:
                continue

            # Calculate cost per tile
            cost_per_tile = self._estimate_dve_cost(tile_size, n_stages)

            # Skip if HW constraint violated
            if cost_per_tile == float('inf'):
                continue

            # Calculate number of tiles needed
            n_tiles = div_ceil(self.per_lnc_BxS, tile_size)

            # Total cost
            total_cost = n_tiles * cost_per_tile

            logger.debug(
                f"Tile size {tile_size}: n_stages={n_stages}, n_tiles={n_tiles} cost_per_tile={int(cost_per_tile)}, total={int(total_cost)}"
            )

            if total_cost < best_cost:
                best_cost = total_cost
                best_tile_size = tile_size

        # If no valid tile size found, use smallest possible that satisfies HW constraint
        if best_tile_size is None:
            best_tile_size = max(16, div_ceil(P_MAX, min_n_stages_for_hw))
            logger.warn(
                f"No optimal tile size found, using {best_tile_size} to satisfy HW constraint, fallback to naive_scaning"
            )

        logger.info(
            f"Optimal tile size: {best_tile_size} (BxS={self.per_lnc_BxS}, V={self.vocab_size}, k={self.orig_k})"
        )
        return best_tile_size

    def update_shard_info(self):
        """
        Update sharding information from program context.

        Returns:
            None: Updates instance attributes n_prgs and prg_id
        """
        shard_info = get_verified_program_sharding_info("topk", (0, 1), 2)
        kernel_assert(shard_info[1] == self.n_prgs or self.BxS == 1, "n_prgs mismatch")
        if self.BxS > 1:
            self.n_prgs = shard_info[1]
            self.prg_id = shard_info[2]
        else:
            kernel_assert(self.n_prgs == 1, f"n_prgs mismatch, BxS {self.BxS}, n_programs {self.n_prgs}")

        self.topk_config = TopkConfig(
            inp_shape=self.topk_config.inp_shape, inp_dtype=self.inp_dtype, k=self.orig_k, sorted=self.sorted
        )

    def inp_shape_valid(self) -> bool:
        return len(self.inp_shape) == 2

    def vocab_size_valid(self) -> bool:
        return self.vocab_size >= self.orig_k

    def BxS_dim_valid(self) -> bool:
        return self.per_lnc_BxS <= self._pmax

    def is_valid(self) -> bool:
        return self.inp_shape_valid() and self.vocab_size_valid()

    def assert_valid(self) -> None:
        kernel_assert(self.inp_shape_valid(), f"topk expects input to be at least 2D, got ({self.inp_shape})")
        kernel_assert(self.vocab_size_valid(), "topk expects vocab_size ({self.vocab_size}) >= k ({self.orig_k}),")

    def log_strategy(self) -> None:
        """Log the topk execution strategy before kernel launch."""
        trivial = self.orig_k == self.vocab_size
        scanning = self.n_stages == 1 and not trivial
        method = "trivial" if trivial else "scanning" if scanning else "rotational"
        tiled = self.n_bxs_tiles > 1

        lines = [
            "+" + "=" * 50 + "+",
            "|          TopK Execution Strategy                 |",
            "+" + "-" * 50 + "+",
            f"| Method:       {method}",
            f"| Input:        BxS={self.BxS}, vocab={self.vocab_size}, k={self.orig_k}",
            f"| Sharding:     n_prgs={self.n_prgs}, per_lnc_BxS={self.per_lnc_BxS}",
            "+" + "-" * 50 + "+",
            f"| Tiling:       tile_size={self.tile_size}, n_tiles={self.n_bxs_tiles}, tiled={tiled}",
            f"| Stages:       n_stages={self.n_stages}, stage_free={self.stage_free_size}",
            f"| TopK:         local_k/stage={self.local_top_k_per_stage}, padded_k={self.padded_k}",
            f"| Output:       sorted={self.sorted}",
            "+" + "=" * 50 + "+",
        ]
        msg = "\n".join(lines)
        print(msg, file=sys.stderr)
        logging.getLogger("topk").info(msg)
        logger.info(msg)


def rotate(dst: nl.ndarray, tensor: nl.ndarray, rotation_matrix: nl.ndarray) -> nl.ndarray:
    """
    Apply rotation matrix to tensor.

    Args:
        dst (nl.ndarray): Destination tensor for result
        tensor (nl.ndarray): Input tensor to rotate
        rotation_matrix (nl.ndarray): Rotation matrix

    Returns:
        nl.ndarray: Rotated tensor (dst)
    """
    f_max = nl.tile_size.gemm_moving_fmax
    free_size = tensor.shape[1]
    n_tiles = div_ceil(free_size, f_max)
    for i in nl.affine_range(n_tiles):
        tile_slice = nl.ds(i * f_max, min(f_max, free_size - i * f_max))
        nisa.nc_matmul(dst[:, tile_slice], rotation_matrix, tensor[:, tile_slice])


def insert(tensor: nl.ndarray, values: nl.ndarray, offset: int = 0) -> None:
    """
    Insert values into tensor at specified offset (in-place).

    Args:
        tensor (nl.ndarray): [m, n], 2D SBUF array
        values (nl.ndarray): [m, f], 2D SBUF array where f <= n
        offset (int): Offset position for insertion (default: 0)
    """
    num_values_to_insert = values.shape[1]
    nisa.tensor_copy(dst=tensor[:, nl.ds(offset, num_values_to_insert)], src=values, engine=nisa.scalar_engine)


def validate_topk_input(inp: nl.ndarray, n_fold: int = 1, local_top_k_per_stage: int = 0) -> None:
    """
    Validate top-k input tensor shape and constraints.

    Args:
        inp (nl.ndarray): Input tensor to validate
        n_fold (int): Number of folds/stages for processing (default: 1)
        local_top_k_per_stage (int): Local top-k per stage (default: 0)
    """
    kernel_assert(len(inp.shape) == 2, f"input has {len(inp.shape)} dims, topk expects input to be 2D")
    kernel_assert(
        div_ceil(inp.shape[1], n_fold) + local_top_k_per_stage <= 2**14,
        f"topk tensor cannot have dim 1 > 16k * n_fold, got {inp.shape[1]} with n_fold={n_fold}",
    )


def validate_config(topk_config: TopkConfig) -> None:
    """
    Validate top-k configuration parameters.

    Args:
        topk_config (TopkConfig): Configuration to validate
    """
    P_MAX = nl.tile_size.pmax
    kernel_assert(
        topk_config.vocab_size_valid(),
        f"topk expects vocab_size >= k, got vocab_size={topk_config.vocab_size}, k={topk_config.k}",
    )


def naive_scanning_topk(inp: nl.ndarray, topk_config: TopkConfig) -> Tuple[nl.ndarray, nl.ndarray]:
    """
    Top-K kernel using scanning approach with DVE instructions.

    Implements top-k reduction using DVE max8 and nc_match_replace8 instructions,
    sharded across multiple NeuronCores. Supports tiling over BxS dimension when
    per_lnc_BxS exceeds PMAX (128).

    Args:
        inp (nl.ndarray): [BxS, V], Input tensor in HBM
        topk_config (TopkConfig): Configuration with algorithm parameters

    Returns:
        Tuple[nl.ndarray, nl.ndarray]: A tuple containing:
            - topk_values: [BxS, k], Top-k values
            - topk_indices: [BxS, k], Indices of top-k elements
    """
    validate_topk_input(inp)
    validate_config(topk_config)

    per_lnc_BxS = topk_config.per_lnc_BxS
    k = topk_config.k
    vocab_size = topk_config.vocab_size
    BxS = topk_config.BxS
    P_MAX = topk_config._pmax
    n_programs, program_id = topk_config.n_prgs, topk_config.prg_id

    topk_values = nl.ndarray((BxS, k), dtype=inp.dtype, buffer=nl.shared_hbm)
    topk_indices = nl.ndarray((BxS, k), dtype=HW_PARAMS.index_dtype, buffer=nl.shared_hbm)

    tile_size = min(per_lnc_BxS, P_MAX)
    n_tiles = div_ceil(per_lnc_BxS, tile_size)
    lnc_batch_start = program_id * per_lnc_BxS

    for tile_idx in nl.sequential_range(n_tiles):
        tile_batch_start = lnc_batch_start + tile_idx * tile_size
        tile_batch_end = min(tile_batch_start + tile_size, min(lnc_batch_start + per_lnc_BxS, BxS))
        tile_bxs = tile_batch_end - tile_batch_start

        inp_sbuf = nl.ndarray((tile_size, vocab_size), dtype=inp.dtype, buffer=nl.sbuf)
        if tile_bxs < tile_size:
            nisa.memset(inp_sbuf, value=_get_dtype_min(inp.dtype))

        hbm_slice = nl.ds(tile_batch_start, tile_bxs)
        sbuf_slice = nl.ds(0, tile_bxs)
        nisa.dma_copy(dst=inp_sbuf[sbuf_slice, :], src=inp[hbm_slice, :])

        sbuf_topk_values, sbuf_topk_indices = topk_core(data=inp_sbuf, k=k)
        nisa.dma_copy(dst=topk_values[hbm_slice, :], src=sbuf_topk_values[sbuf_slice, :])
        nisa.dma_copy(dst=topk_indices[hbm_slice, :], src=sbuf_topk_indices[sbuf_slice, :])

    return topk_values, topk_indices


def topk_core(data: nl.ndarray, k: int) -> Tuple[nl.ndarray, nl.ndarray]:
    """
    Core top-k implementation using DVE instructions.

    Performs top-k using repeated max8 and nc_match_replace8 combinations.
    Expects all inputs in SBUF. Modifies data in-place.

    Args:
        data (nl.ndarray): [BxS, V], Input data in SBUF (modified in-place)
        k (int): Number of top elements to find

    Returns:
        Tuple[nl.ndarray, nl.ndarray]: A tuple containing:
            - out_vals: [BxS, k], Top-k values in SBUF
            - out_inds: [BxS, k], Indices of top-k elements in SBUF
    """
    BxS, vocab_size = data.shape
    n_fold = div_ceil(k, HW_PARAMS.topk_per_stage)

    out_vals = nl.ndarray((BxS, k), dtype=data.dtype, buffer=nl.sbuf)
    out_inds = nl.ndarray((BxS, k), dtype=HW_PARAMS.index_dtype, buffer=nl.sbuf)

    for fold_idx in nl.static_range(n_fold):
        if (k % HW_PARAMS.topk_per_stage != 0) and (fold_idx == n_fold - 1):
            val_buf = nl.ndarray((BxS, HW_PARAMS.topk_per_stage), dtype=out_vals.dtype, buffer=nl.sbuf)
            ind_buf = nl.ndarray((BxS, HW_PARAMS.topk_per_stage), dtype=out_inds.dtype, buffer=nl.sbuf)

            nisa.max8(dst=val_buf[...], src=data[:, :vocab_size])
            nisa.nc_find_index8(dst=ind_buf[...], data=data[:, :vocab_size], vals=val_buf)

            elts_remain = k % HW_PARAMS.topk_per_stage
            nisa.tensor_copy(
                dst=out_vals[:, k - elts_remain :], src=val_buf[:, :elts_remain], engine=nisa.scalar_engine
            )
            nisa.tensor_copy(
                dst=out_inds[:, k - elts_remain :], src=ind_buf[:, :elts_remain], engine=nisa.scalar_engine
            )

        else:
            i_free_dim = nl.ds(fold_idx * HW_PARAMS.topk_per_stage, HW_PARAMS.topk_per_stage)

            nisa.max8(dst=out_vals[0:BxS, i_free_dim], src=data[:, :vocab_size])

            nisa.nc_match_replace8(
                dst=data[:, :vocab_size],
                dst_idx=out_inds[0:BxS, i_free_dim],
                data=data[:, :vocab_size],
                vals=out_vals[0:BxS, i_free_dim],
                imm=float('-inf'),
            )

    return out_vals, out_inds


def sort(data_sbuf: nl.ndarray, indices: Optional[nl.ndarray] = None) -> Tuple[nl.ndarray, nl.ndarray]:
    """
    Sort data using top-k algorithm.

    Performs sorting by repeatedly finding top-8 elements and masking them.

    Args:
        data_sbuf (nl.ndarray): [m, n], Unsorted data in SBUF
        indices (Optional[nl.ndarray]): [m, n], Global indices corresponding to elements (default: None)

    Returns:
        Tuple[nl.ndarray, nl.ndarray]: A tuple containing:
            - sorted_values: [m, n], Sorted values in SBUF
            - sorted_indices: [m, n], Global or local indices corresponding to sorted elements
    """
    m, n = data_sbuf.shape
    num_pass = div_ceil(n, HW_PARAMS.dve_max_alus)
    kernel_assert(n % HW_PARAMS.dve_max_alus == 0, f"n {n} must be divisible by DVE_MAX_ALUS {HW_PARAMS.dve_max_alus}")

    topk_val_buf = nl.ndarray((nl.par_dim(m), n), dtype=data_sbuf.dtype, buffer=nl.sbuf)
    topk_idx_buf = nl.ndarray((nl.par_dim(m), n), dtype=nl.uint32, buffer=nl.sbuf)
    global_topk_idx_buf = nl.ndarray((nl.par_dim(m), n), dtype=nl.uint32, buffer=nl.sbuf)

    ix, iy = nl.ds(0, m), nl.ds(0, HW_PARAMS.dve_max_alus)

    ix_data, iy_data = nl.ds(0, m), nl.ds(0, n)

    for pass_num in nl.sequential_range(num_pass):
        cur_slice = nl.ds(pass_num * HW_PARAMS.dve_max_alus, HW_PARAMS.dve_max_alus)
        nisa.max8(dst=topk_val_buf[:, cur_slice], src=data_sbuf)
        if nisa.get_nc_version() <= nisa.nc_version.gen2:
            nisa.nc_find_index8(dst=topk_idx_buf[:, cur_slice], data=data_sbuf[...], vals=topk_val_buf[:, cur_slice])
            nisa.nc_match_replace8(
                dst=data_sbuf[...], data=data_sbuf[...], vals=topk_val_buf[:, cur_slice], imm=float('-inf')
            )
        else:
            nisa.nc_match_replace8(
                dst=data_sbuf,
                data=data_sbuf,
                vals=topk_val_buf[:, cur_slice],
                imm=float('-inf'),
                dst_idx=topk_idx_buf[:, cur_slice],
            )
        if indices is not None:
            nisa.nc_n_gather(
                dst=global_topk_idx_buf[ix, cur_slice],
                data=indices[ix_data, iy_data],
                indices=topk_idx_buf[ix, cur_slice],
            )
    if indices is not None:
        return topk_val_buf, global_topk_idx_buf
    return topk_val_buf, topk_idx_buf


def reshape_with_dma(src, fold_factor, dtype):
    """
    Reshape tensor using DMA operations.

    Reshapes from stages layout [s*b, n/s] to original layout [b, n] using HBM as intermediate.

    Args:
        src (nl.ndarray): Source tensor in SBUF
        fold_factor (int): Folding factor
        dtype: Target data type

    Returns:
        nl.ndarray: Reshaped tensor in SBUF
    """
    m, n = src.shape
    data_hbm = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.private_hbm)
    nisa.dma_copy(src=src, dst=data_hbm)
    data_hbm = data_hbm.reshape((m // fold_factor, n * fold_factor))
    out_sbuf = nl.ndarray(data_hbm.shape, dtype=dtype, buffer=nl.sbuf)
    nisa.dma_copy(src=data_hbm, dst=out_sbuf)
    return out_sbuf


def get_ceil_aligned_size(size: int, alignment: int) -> int:
    """
    Calculate ceiling-aligned size.

    Args:
        size (int): Original size
        alignment (int): Alignment requirement

    Returns:
        int: Smallest multiple of alignment >= size
    """
    return ((size + alignment - 1) // alignment) * alignment
