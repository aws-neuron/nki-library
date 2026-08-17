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

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import nki.isa as nisa
import nki.language as nl
import numpy as np

from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil
from ..utils.logging import get_logger

logger = get_logger("topk")

FLOAT32_MIN = np.finfo(np.float32).min.item()
BFLOAT16_MIN = -9948.0


def _get_dtype_min(dtype):
    """Get minimum value for padding based on dtype."""
    if dtype == nl.bfloat16:
        return BFLOAT16_MIN
    return FLOAT32_MIN


@dataclass(frozen=True, eq=True)
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
        sort_fixed_overhead (int): Fixed overhead cycles for sort operation
        gather_latency_per_pass (int): Cycles per nc_n_gather pass
        rotation_latency_per_tile (int): Cycles per rotation matrix multiply tile
        rotation_tile_size (int): Free dimension tile size for rotation cost calculation
        insert_latency_per_element (int): Cycles per element for insert (tensor_copy) operation
        sync_latency_per_stage (int): Synchronization cycles per rotational stage
        tile_overhead (int): Fixed overhead cycles per BxS tile iteration
        gather_group_size (int): Max elements per partition that nc_n_gather handles in a
            single ISA group. A gather wider than this splits into ceil(width/group_size)
            internal groups; the multi-group form corrupts results on hardware
            (NKILIB-1592), so wide gathers must be tiled to this width.
    """

    dve_max_alus: int = 8
    topk_per_stage: int = 8
    index_dtype: type = nl.uint32
    num_sbuf_quadrants: int = 4
    fixed_dve_inst_overhead: int = 144
    max_free_dim: int = 2**14
    sort_fixed_overhead: int = 8000
    gather_latency_per_pass: int = 600
    rotation_latency_per_tile: int = 400
    rotation_tile_size: int = 512
    insert_latency_per_element: int = 10
    sync_latency_per_stage: int = 500
    tile_overhead: int = 5000
    gather_group_size: int = 512


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
    kernel_assert(initial_value != None, "initial_value must be set")
    kernel_assert(input_list != None, "input_list must be set")
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
        # This else only absorbs the phantom false-arc coverage flags off the last
        # elif; it is unreachable given the supported_ops assert above.
        else:  # pragma: no cover - unreachable: op validated against supported_ops above
            kernel_assert(False, f"reduce: no handler for op '{op}' despite passing supported_ops check")
    return initial_value


@dataclass(frozen=True, eq=True)
class TopkConfig(nl.NKIObject):
    """
    Configuration class for top-k algorithm.

    Generic configuration for top-k algorithms that accept inputs [B, S, V] and
    perform top-k reduction along vocabulary dimension to produce [B, S, k].

    Attributes:
        inp_shape (Tuple): Input tensor shape
        k (int): Number of top elements
        sorted (bool): Sort flag
        inp_dtype (np.dtype): Input data type
        index_dtype (np.dtype): Index data type (nl.uint32)
        BxS (int): Combined batch and sequence dimensions
        vocab_size (int): Vocabulary dimension size
        out_shape (tuple): Output shape (B, S, k)
        n_prgs (int): Number of logical cores
        prg_id (int): Program ID for SPMD grid
        per_lnc_BxS (int): Batch size per logical core
        _pmax (int): Maximum partition size
    """

    inp_shape: Tuple
    k: int
    sorted: bool
    inp_dtype: np.dtype
    index_dtype: np.dtype
    BxS: int
    vocab_size: int
    out_shape: tuple
    n_prgs: int
    prg_id: int
    per_lnc_BxS: int
    _pmax: int

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


def create_topk_config(
    inp_shape: Tuple, inp_dtype: np.dtype, k: int, sorted: bool = True, num_programs: int = 2
) -> TopkConfig:
    """
    Factory function to create a TopkConfig with all derived values pre-calculated.

    Args:
        inp_shape (Tuple): Shape of input tensor (2D or 3D)
        inp_dtype (np.dtype): Data type of input tensor
        k (int): Number of top elements to retrieve
        sorted (bool): Whether to sort the output (default: True)
        num_programs (int): Number of logical cores (default: 2)

    Returns:
        TopkConfig: Frozen, hashable configuration instance
    """
    pmax = 128
    index_dtype = HW_PARAMS.index_dtype

    BxS = reduce('mul', inp_shape[:-1], 1)
    vocab_size = inp_shape[-1]
    out_shape = tuple(list(inp_shape[:-1]) + [k])

    # Use num_programs directly — actual shard info is queried inside the kernel
    n_prgs = num_programs
    prg_id = 0

    if BxS == 1:
        logger.info(f"Setting num_programs to 1 since {BxS}, user specified num_programs {n_prgs}")
        prg_id = 0
        n_prgs = 1

    per_lnc_BxS = (BxS + n_prgs - 1) // n_prgs

    config = TopkConfig(
        inp_shape=inp_shape,
        k=k,
        sorted=sorted,
        inp_dtype=inp_dtype,
        index_dtype=index_dtype,
        BxS=BxS,
        vocab_size=vocab_size,
        out_shape=out_shape,
        n_prgs=n_prgs,
        prg_id=prg_id,
        per_lnc_BxS=per_lnc_BxS,
        _pmax=pmax,
    )
    kernel_assert(config.inp_shape_valid(), "topk expects input to be at least 2D")
    kernel_assert(config.vocab_size_valid(), f"topk expects vocab_size ({vocab_size}) >= k ({k})")
    return config


@dataclass(frozen=True, eq=True)
class RotationalTopkConfig(nl.NKIObject):
    """
    Configuration class for rotational top-k algorithm.

    Configures rotational top-k algorithm that accepts inputs [BxS, V] and performs
    top-k reduction along vocabulary dimension. May use padded k for efficiency.
    Supports tiling over BxS dimension for BxS > 128.

    This is a frozen (immutable) dataclass to ensure hashability when passed
    as a parameter to @nki.jit kernels. Use create_rotational_topk_config()
    factory function to construct instances.

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
        topk_config (TopkConfig): Base top-k configuration
        _pmax (int): Maximum partition size
        _shared_const_cache (dict): Cache of shared constant file paths
    """

    inp_shape: Tuple
    BxS: int
    vocab_size: int
    orig_k: int
    padded_k: int
    n_prgs: int
    prg_id: int
    per_lnc_BxS: int
    inp_dtype: np.dtype
    index_dtype: np.dtype
    local_top_k_per_stage: int
    n_stages: int
    stage_free_size: int
    padded_vocab_size: int
    sorted: bool
    tile_size: int
    n_bxs_tiles: int
    topk_config: TopkConfig
    _pmax: int
    _shared_const_cache: dict

    def __hash__(self):
        # Custom hash that excludes the unhashable _shared_const_cache dict
        return hash(
            (
                self.inp_shape,
                self.BxS,
                self.vocab_size,
                self.orig_k,
                self.padded_k,
                self.n_prgs,
                self.prg_id,
                self.per_lnc_BxS,
                self.inp_dtype,
                self.index_dtype,
                self.local_top_k_per_stage,
                self.n_stages,
                self.stage_free_size,
                self.padded_vocab_size,
                self.sorted,
                self.tile_size,
                self.n_bxs_tiles,
                self.topk_config,
                self._pmax,
            )
        )

    def __eq__(self, other):
        if not isinstance(other, RotationalTopkConfig):
            return NotImplemented
        return (
            self.inp_shape == other.inp_shape
            and self.BxS == other.BxS
            and self.vocab_size == other.vocab_size
            and self.orig_k == other.orig_k
            and self.padded_k == other.padded_k
            and self.n_prgs == other.n_prgs
            and self.prg_id == other.prg_id
            and self.per_lnc_BxS == other.per_lnc_BxS
            and self.inp_dtype == other.inp_dtype
            and self.index_dtype == other.index_dtype
            and self.local_top_k_per_stage == other.local_top_k_per_stage
            and self.n_stages == other.n_stages
            and self.stage_free_size == other.stage_free_size
            and self.padded_vocab_size == other.padded_vocab_size
            and self.sorted == other.sorted
            and self.tile_size == other.tile_size
            and self.n_bxs_tiles == other.n_bxs_tiles
            and self.topk_config == other.topk_config
            and self._pmax == other._pmax
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
        print(msg)
        logger.info(msg)


_SMALL_K_THRESHOLD = 64


def _calculate_rotational_constants_baseline(
    orig_k: int, vocab_size: int, pmax: int, BxS_tile: int
) -> Tuple[int, int, int, int, int]:
    """Calculate rotational constants using the simple ideal-stages heuristic.

    Used as fallback for small K (<= _SMALL_K_THRESHOLD) where the detailed
    cost model over-estimates rotation/insert benefits.
    """
    MAX_FREE_DIM = 2**14

    max_n_stages = math.floor(pmax // BxS_tile)
    ideal_n_stages = div_ceil(min(orig_k, vocab_size), HW_PARAMS.topk_per_stage)
    min_n_stages_for_hw = div_ceil(vocab_size, MAX_FREE_DIM)

    n_stages = min(max_n_stages, ideal_n_stages)
    n_stages = max(n_stages, min_n_stages_for_hw)

    kernel_assert(
        n_stages <= max_n_stages,
        f"Cannot satisfy HW constraint: need {min_n_stages_for_hw} stages but only {max_n_stages} fit with BxS_tile={BxS_tile}",
    )

    local_top_k_per_stage = get_ceil_aligned_size(div_ceil(orig_k, n_stages), HW_PARAMS.topk_per_stage)
    padded_k = local_top_k_per_stage * n_stages

    chunk_size = div_ceil(vocab_size, n_stages)
    padded_vocab_size = chunk_size * n_stages

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


def _estimate_rotational_cost(orig_k: int, vocab_size: int, ns: int, sorted: bool = True) -> float:
    """Estimate total DVE cycle cost for a given n_stages configuration.

    Returns float('inf') if the configuration violates HW constraints.
    """
    MAX_FREE_DIM = 2**14
    lk = get_ceil_aligned_size(div_ceil(orig_k, ns), HW_PARAMS.topk_per_stage)
    pk = lk * ns
    sf = div_ceil(vocab_size, ns)
    cf = sf + ns * lk
    if sf > MAX_FREE_DIM or cf > MAX_FREE_DIM:
        return float("inf")

    topk_cost = 0
    for stage_idx in range(ns):
        topk_cost += div_ceil(lk, 8) * 2 * (sf + lk * stage_idx + HW_PARAMS.fixed_dve_inst_overhead)

    sort_latency = 0
    if sorted:
        sort_passes = div_ceil(pk, 8)
        sort_latency = sort_passes * 2 * (pk + HW_PARAMS.fixed_dve_inst_overhead) + HW_PARAMS.sort_fixed_overhead

    gather_latency = ns * div_ceil(lk, 8) * HW_PARAMS.gather_latency_per_pass
    rotation_latency = (ns - 1) * 2 * div_ceil(lk, HW_PARAMS.rotation_tile_size) * HW_PARAMS.rotation_latency_per_tile
    insert_latency = (ns - 1) * lk * HW_PARAMS.insert_latency_per_element
    sync_latency = ns * HW_PARAMS.sync_latency_per_stage
    return topk_cost + sort_latency + gather_latency + rotation_latency + insert_latency + sync_latency


def _calculate_rotational_constants(
    orig_k: int, vocab_size: int, pmax: int, BxS_tile: int
) -> Tuple[int, int, int, int, int]:
    """Calculate rotational constants using detailed component-level DVE cost model.

    Accounts for topk, sort, gather, rotation, insert, and sync costs separately
    to select the optimal n_stages. Falls back to the baseline heuristic for
    small K (<= 64) where the detailed model regresses.

    Args:
        orig_k: Original number of top elements requested
        vocab_size: Vocabulary dimension size
        pmax: Maximum partition size
        BxS_tile: Tile size for BxS dimension

    Returns:
        Tuple[int, int, int, int, int]: (local_top_k_per_stage, padded_k, n_stages, chunk_size, padded_vocab_size)
    """
    if orig_k <= _SMALL_K_THRESHOLD:
        return _calculate_rotational_constants_baseline(orig_k, vocab_size, pmax, BxS_tile)

    MAX_FREE_DIM = 2**14
    max_n_stages = pmax // BxS_tile
    min_n_stages_for_hw = div_ceil(vocab_size, MAX_FREE_DIM)
    best_ns = None
    best_cost = float("inf")

    for ns in range(max(2, min_n_stages_for_hw), max_n_stages + 1):
        cost = _estimate_rotational_cost(orig_k, vocab_size, ns)
        if cost < best_cost:
            best_cost = cost
            best_ns = ns

    if best_ns == None:
        return _calculate_rotational_constants_baseline(orig_k, vocab_size, pmax, BxS_tile)

    ns = best_ns
    lk = get_ceil_aligned_size(div_ceil(orig_k, ns), HW_PARAMS.topk_per_stage)
    pk = lk * ns
    sf = div_ceil(vocab_size, ns)
    kernel_assert(sf <= HW_PARAMS.max_free_dim, f"HW constraint: sf={sf}")
    cf = sf + ns * lk
    kernel_assert(cf <= HW_PARAMS.max_free_dim, f"HW constraint: cf={cf}")
    return lk, pk, ns, sf, sf * ns


def _find_optimal_tile_size(orig_k: int, vocab_size: int, per_lnc_BxS: int, pmax: int, topk_sorted: bool) -> int:
    """Find optimal tile size using detailed component-level DVE cost model.

    Falls back to the baseline heuristic for small K (<= 64).

    Args:
        orig_k: Original number of top elements requested
        vocab_size: Vocabulary dimension size
        per_lnc_BxS: Batch size per logical core
        pmax: Maximum partition size
        topk_sorted: Whether the topk config requests sorted output

    Returns:
        Optimal tile size
    """
    if orig_k <= _SMALL_K_THRESHOLD:
        return _find_optimal_tile_size_baseline(orig_k, vocab_size, per_lnc_BxS, pmax, topk_sorted)

    MAX_FREE_DIM = 2**14
    best_tile_size = None
    best_cost = float("inf")

    for tile_size in range(1, pmax + 1):
        n_tiles = div_ceil(per_lnc_BxS, tile_size)
        max_n_stages = pmax // tile_size
        best_ns_cost = float("inf")
        min_ns_hw = div_ceil(vocab_size, MAX_FREE_DIM)

        for ns in range(max(2, min_ns_hw), max_n_stages + 1):
            cost = _estimate_rotational_cost(orig_k, vocab_size, ns)
            if cost < best_ns_cost:
                best_ns_cost = cost

        if best_ns_cost == float("inf"):
            continue

        total_cost = n_tiles * (best_ns_cost + HW_PARAMS.tile_overhead)
        if total_cost < best_cost:
            best_cost = total_cost
            best_tile_size = tile_size

    if best_tile_size == None:
        best_tile_size = max(16, div_ceil(pmax, div_ceil(vocab_size, MAX_FREE_DIM)))
    return best_tile_size


def _find_optimal_tile_size_baseline(
    orig_k: int, vocab_size: int, per_lnc_BxS: int, pmax: int, topk_sorted: bool
) -> int:
    """Find optimal tile size using simple per-stage DVE cost estimate.

    Used as fallback for small K where the detailed cost model regresses.
    """
    min_n_stages_for_hw = div_ceil(vocab_size, HW_PARAMS.max_free_dim)
    best_tile_size = None
    best_cost = float("inf")

    for tile_size in range(1, pmax + 1):
        max_n_stages = math.floor(pmax // tile_size)
        ideal_n_stages = div_ceil(min(orig_k, vocab_size), HW_PARAMS.topk_per_stage)
        n_stages = min(max_n_stages, ideal_n_stages)
        n_stages = max(n_stages, min_n_stages_for_hw)

        if n_stages > max_n_stages:
            continue
        if n_stages <= 1:
            continue

        stage_free_size = div_ceil(vocab_size, n_stages)
        if (
            stage_free_size > HW_PARAMS.max_free_dim
        ):  # pragma: no cover - unreachable: n_stages>=ceil(vocab/max_free) forces stage_free=ceil(vocab/n_stages)<=max_free
            continue
        k_per_stage = get_ceil_aligned_size(div_ceil(orig_k, n_stages), HW_PARAMS.topk_per_stage)
        if stage_free_size + n_stages * k_per_stage > HW_PARAMS.max_free_dim:
            continue

        padded_k = k_per_stage * n_stages
        per_stage_cost = div_ceil(k_per_stage, 8) * 2 * (stage_free_size + HW_PARAMS.fixed_dve_inst_overhead)
        unsorted_cost = n_stages * per_stage_cost
        needs_sort = topk_sorted or padded_k != orig_k
        if needs_sort:
            base_sort_cost = (
                div_ceil(padded_k, HW_PARAMS.dve_max_alus) * 2 * (padded_k + HW_PARAMS.fixed_dve_inst_overhead)
            )
            sort_efficiency = tile_size / pmax
            sorted_cost = base_sort_cost / sort_efficiency
        else:
            sorted_cost = 0
        cost_per_tile = unsorted_cost + sorted_cost

        n_tiles = div_ceil(per_lnc_BxS, tile_size)
        total_cost = n_tiles * cost_per_tile

        if total_cost < best_cost:
            best_cost = total_cost
            best_tile_size = tile_size

    if best_tile_size == None:
        best_tile_size = max(16, div_ceil(pmax, min_n_stages_for_hw))

    return best_tile_size


def _exceeds_concatenated_free_dim(orig_k: int, vocab_size: int, tile_size: int, pmax: int) -> bool:
    """Check if tile_size would cause the concatenated SBUF free dimension to exceed the HW limit.

    The rotational algorithm concatenates vocab chunks and per-stage top-k results along the
    free dimension. When vocab is large and K is large, a single stage may exceed 16384 elements.
    """
    max_n_stages = pmax // tile_size
    min_n_stages = div_ceil(vocab_size, HW_PARAMS.max_free_dim)
    n_stages = max(max_n_stages, min_n_stages)
    if n_stages > max_n_stages:
        return True
    chunk = div_ceil(vocab_size, n_stages)
    local_k = get_ceil_aligned_size(div_ceil(orig_k, n_stages), HW_PARAMS.topk_per_stage)
    return chunk + n_stages * local_k > HW_PARAMS.max_free_dim


def create_rotational_topk_config(
    inp_shape: Tuple, topk_config: TopkConfig, shared_const_cache: Optional[dict] = None
) -> RotationalTopkConfig:
    """
    Factory function to create a RotationalTopkConfig with all derived values pre-calculated.

    All computation and validation is performed here before constructing the frozen
    (immutable, hashable) dataclass instance.

    Args:
        inp_shape (Tuple): Shape of input tensor (must be 2D)
        topk_config (TopkConfig): Base top-k configuration
        shared_const_cache (Optional[dict]): Pre-computed shared constant cache.
            If None, an empty dict is used.

    Returns:
        RotationalTopkConfig: Frozen, hashable configuration instance
    """
    pmax = 128

    kernel_assert(len(inp_shape) == 2, f"rotated topk expects input to be 2D, actual was {len(inp_shape)}")

    BxS = inp_shape[0]
    vocab_size = inp_shape[1]
    kernel_assert(BxS == topk_config.BxS, f"TopkConfig BxS dim {topk_config} does not match input BxS dim {BxS}")

    orig_k = topk_config.k
    n_prgs = topk_config.n_prgs
    prg_id = topk_config.prg_id
    per_lnc_BxS = topk_config.per_lnc_BxS
    inp_dtype = topk_config.inp_dtype
    index_dtype = topk_config.index_dtype

    # Find optimal tile size if BxS > PMAX or default tile violates HW constraints
    if per_lnc_BxS > pmax or _exceeds_concatenated_free_dim(orig_k, vocab_size, per_lnc_BxS, pmax):
        tile_size = _find_optimal_tile_size(orig_k, vocab_size, per_lnc_BxS, pmax, topk_config.sorted)
        n_bxs_tiles = div_ceil(per_lnc_BxS, tile_size)
    else:
        tile_size = per_lnc_BxS
        n_bxs_tiles = 1

    const_info = _calculate_rotational_constants(orig_k, vocab_size, pmax, tile_size)
    local_top_k_per_stage = const_info[0]
    padded_k = const_info[1]
    n_stages = const_info[2]
    stage_free_size = const_info[3]
    padded_vocab_size = const_info[4]

    if orig_k != padded_k:
        sorted_flag = True
    else:
        sorted_flag = topk_config.sorted

    if shared_const_cache == None:
        shared_const_cache = {}

    return RotationalTopkConfig(
        inp_shape=inp_shape,
        BxS=BxS,
        vocab_size=vocab_size,
        orig_k=orig_k,
        padded_k=padded_k,
        n_prgs=n_prgs,
        prg_id=prg_id,
        per_lnc_BxS=per_lnc_BxS,
        inp_dtype=inp_dtype,
        index_dtype=index_dtype,
        local_top_k_per_stage=local_top_k_per_stage,
        n_stages=n_stages,
        stage_free_size=stage_free_size,
        padded_vocab_size=padded_vocab_size,
        sorted=sorted_flag,
        tile_size=tile_size,
        n_bxs_tiles=n_bxs_tiles,
        topk_config=topk_config,
        _pmax=pmax,
        _shared_const_cache=shared_const_cache,
    )


def build_stage_offsets(n_stages: int, bxs_size: int, stage_free_size: int):
    """Generate per-partition interleaved stage offsets on-device.

    Builds offsets[p] = (p % n_stages) * stage_free_size. Used to convert
    stage-local indices into global vocabulary indices by adding the
    per-stage base offset to each partition's local index range.

    Args:
        n_stages: Number of rotational stages.
        bxs_size: Batch-sequence tile size (partitions per stage).
        stage_free_size: Free dimension size per stage.

    Returns:
        [total_partition_dim, 1] SBUF tensor (float32) of stage offsets.

    Example (n_stages=3, bxs_size=2, stage_free_size=4096)::

        offsets = [[    0],   # partition 0: stage 0
                   [ 4096],   # partition 1: stage 1
                   [ 8192],   # partition 2: stage 2
                   [    0],   # partition 3: stage 0
                   [ 4096],   # partition 4: stage 1
                   [ 8192]]   # partition 5: stage 2
    """
    total_partition_dim = n_stages * bxs_size
    padded = ((total_partition_dim + 31) // 32) * 32
    lhs = nl.ndarray((1, padded), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(lhs[:, 0:total_partition_dim], pattern=[[0, bxs_size], [stage_free_size, n_stages]])
    offsets = nl.ndarray((padded, 1), dtype=nl.float32, buffer=nl.sbuf)
    # Transpose in <=32-wide chunks, clamping the last chunk to total_partition_dim
    # so it never reads the alignment padding that iota does not fill.
    n_chunks = (total_partition_dim + 31) // 32
    for i in nl.affine_range(n_chunks):
        chunk = min(32, total_partition_dim - i * 32)
        chunk_slice = nl.ds(i * 32, chunk)
        nisa.nc_transpose(dst=offsets[chunk_slice, 0:1], data=lhs[0:1, chunk_slice], engine=nisa.vector_engine)
    return offsets[0:total_partition_dim, 0:1]


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
    for tile_idx in nl.affine_range(n_tiles):
        tile_slice = nl.ds(tile_idx * f_max, min(f_max, free_size - tile_idx * f_max))
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


def sort(data_sbuf, indices, true_k):
    """
    Sort data by extracting top true_k elements using repeated max8 passes and masking them..

    Only runs ceil(true_k/8) DVE passes rather than sorting the entire buffer,

    Args:
        data_sbuf (nl.ndarray): [m, n], Unsorted data in SBUF
        indices (Optional[nl.ndarray]): [m, n], Global indices corresponding to elements (default: None)

    Returns:
        Tuple[nl.ndarray, nl.ndarray]: A tuple containing:
            - sorted_values: [m, n], Sorted values in SBUF
            - sorted_indices: [m, n], Global or local indices corresponding to sorted elements
    """
    m, pk = data_sbuf.shape
    num_pass = div_ceil(true_k, HW_PARAMS.dve_max_alus)
    padded_k = num_pass * HW_PARAMS.dve_max_alus

    topk_val_buf = nl.ndarray((m, padded_k), dtype=data_sbuf.dtype, buffer=nl.sbuf)
    topk_idx_buf = nl.ndarray((m, padded_k), dtype=nl.uint32, buffer=nl.sbuf)
    global_topk_idx_buf = nl.ndarray((m, padded_k), dtype=nl.uint32, buffer=nl.sbuf)

    ix_data, iy_data = nl.ds(0, m), nl.ds(0, pk)

    for pass_num in nl.sequential_range(num_pass):
        cur_slice = nl.ds(pass_num * HW_PARAMS.dve_max_alus, HW_PARAMS.dve_max_alus)
        nisa.max8(dst=topk_val_buf[:, cur_slice], src=data_sbuf)
        if (
            nisa.get_nc_version() <= nisa.nc_version.gen2
        ):  # pragma: no cover - gen2/TRN1 legacy path; coverage targets trn2(gen3) and trn3_a0(gen4)
            nisa.nc_find_index8(dst=topk_idx_buf[:, cur_slice], data=data_sbuf[...], vals=topk_val_buf[:, cur_slice])
            nisa.nc_match_replace8(
                dst=data_sbuf[...], data=data_sbuf[...], vals=topk_val_buf[:, cur_slice], imm=float("-inf")
            )
        else:
            nisa.nc_match_replace8(
                dst=data_sbuf,
                data=data_sbuf,
                vals=topk_val_buf[:, cur_slice],
                imm=float("-inf"),
                dst_idx=topk_idx_buf[:, cur_slice],
            )
        nisa.nc_n_gather(
            dst=global_topk_idx_buf[nl.ds(0, m), cur_slice],
            data=indices[ix_data, iy_data],
            indices=topk_idx_buf[nl.ds(0, m), cur_slice],
        )

    return topk_val_buf[:, :true_k], global_topk_idx_buf[:, :true_k]


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
