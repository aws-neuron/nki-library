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
"""Schedule-contract tests for packed token-generation attention."""

import math
from dataclasses import replace
from typing import final

import neuron_dtypes as dt
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.core.attention.attention_tkg import AttnTKGConfig, TileConstants, _compute_tile_params
from nkilib_src.nkilib.core.attention.attention_tkg_seq_packed import attention_tkg_seq_packed
from nkilib_src.nkilib.core.attention.attention_tkg_seq_packed_torch import (
    build_packed_accumulator_route_tables,
)
from nkilib_src.nkilib.core.attention.attention_tkg_torch import attention_tkg_torch_ref
from nkilib_src.nkilib.core.attention.attention_tkg_utils import (
    resize_cache_block_len_for_attention_tkg_kernel,
)
from nkilib_src.nkilib.core.utils.allocator import SbufManager, create_auto_alloc_manager
from nkilib_src.nkilib.core.utils.common_types import DtypeMode
from nkilib_src.nkilib.core.utils.kernel_helpers import resolve_fp8_e4m3_dtype
from nkilib_src.nkilib.core.utils.logging import Logger

from test.integration.nkilib.core.attention.test_attention_tkg_utils import (
    build_active_attention_mask,
    gen_deterministic_active_block_table,
)
from test.integration.nkilib.utils.tensor_generators import np_random_sample, np_random_sample_fp8
from test.integration.nkilib.utils.test_kernel_common import resolve_dtype_mode_for_torch_ref
from test.utils.common_dataclasses import TKG_INFERENCE_ARGS, CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

P_MAX = 128
FA_TILE_SIZE = 8192
DEFAULT_BLOCK_LEN = 16
INVALID = -1


def _fp8_kv_dtype(platform_target):
    """Pick the E4M3 variant the platform uses: OCP e4m3fn on TRN3, e4m3 elsewhere.

    Mirrors DtypeMode.AUTO, which the kernel resolves at trace time from
    nisa.get_nc_version(); the test must pre-resolve it because the golden is CPU-side.
    """
    return resolve_fp8_e4m3_dtype(resolve_dtype_mode_for_torch_ref(DtypeMode.AUTO, platform_target))


# GPT-OSS 20B and 120B share the same attention geometry: TP8 gives
# 8 local Q heads, one local KV head, and d_head=64. The B=8 request
# bucket is split contiguously across LNC2.
GPT_OSS_20B_TP8_UTIL50_LENGTHS = (
    31744,
    31744,
    16384,
    16384,
    8192,
    8192,
    8192,
    8192,
)
GPT_OSS_20B_TP8_UTIL50_SCHEDULE = (
    ((0, 0), (1, 0), (2, 0), (3, 0)),
    ((0, 1), (1, 1), (2, 1), (3, 1)),
    ((0, 2), (1, 2), (-1, 0), (-1, 0)),
    ((0, 3), (1, 3), (-1, 0), (-1, 0)),
    ((4, 0), (5, 0), (6, 0), (7, 0)),
    ((-1, 0), (-1, 0), (-1, 0), (-1, 0)),
    ((-1, 0), (-1, 0), (-1, 0), (-1, 0)),
    ((-1, 0), (-1, 0), (-1, 0), (-1, 0)),
)
# The 120B model differs in layer and expert counts, which are outside this
# single attention kernel; its TP8 attention workload is otherwise identical.
GPT_OSS_120B_TP8_UTIL50_LENGTHS = GPT_OSS_20B_TP8_UTIL50_LENGTHS
GPT_OSS_120B_TP8_UTIL50_SCHEDULE = GPT_OSS_20B_TP8_UTIL50_SCHEDULE

# Fixed-shape regression anchors for the original batch-8, 128K optimization
# configurations. Lengths include the single active token; the schedule contains
# only prior-cache FA tiles.
GPT_OSS_BATCH8_128K_FOCUS_SPECS = (
    {
        "id": "gptoss_b8_c128k_68tiles_s4_r16",
        "lengths": (126976, 49152, 40960, 32768, 98304, 65536, 24576, 114688),
        "slots": 4,
        "rows_per_nc": 16,
        # Preserve one-tile-per-sequence row scheduling, then pad each NC to the
        # original 64-slot shape.
        "compact_rows": False,
        "expected_real_tiles": 68,
    },
    {
        "id": "gptoss_b8_c128k_44tiles_s4_r8",
        "lengths": (126976, 32768, 32768, 32768, 32768, 32768, 32768, 32768),
        "slots": 4,
        "rows_per_nc": 8,
        # Densely pack remaining work, allowing several tiles from one sequence
        # to occupy the same row.
        "compact_rows": True,
        "expected_real_tiles": 44,
    },
)


def _case(case_id, local_tile_counts, slots, distribution, baseline_id=None):
    """Build one mirrored LNC2 workload from per-NC FA-tile counts."""
    local_lengths = tuple(tile_count * FA_TILE_SIZE for tile_count in local_tile_counts)
    return {
        "id": case_id,
        "local_lengths": local_lengths,
        "lengths": local_lengths * 2,
        "slots": slots,
        "distribution": distribution,
        "baseline_id": baseline_id or case_id,
    }


# Each local length vector is mirrored across the two NeuronCores. Cases that
# differ only in slot width deliberately share an unpacked baseline_id.
COMPARISON_CASES = (
    # Utilization crossover at 32K context and four slots.
    _case("util50_bimodal_b16_c32k_s4", (4, 4, 2, 2, 1, 1, 1, 1), 4, "bimodal"),
    _case("util62_bimodal_b16_c32k_s4", (4, 4, 4, 4, 1, 1, 1, 1), 4, "bimodal"),
    _case("util75_bimodal_b16_c32k_s4", (4, 4, 4, 4, 2, 2, 2, 2), 4, "bimodal"),
    _case("util87_bimodal_b16_c32k_s4", (4, 4, 4, 4, 3, 3, 3, 3), 4, "bimodal"),
    _case("util100_uniform_b16_c32k_s4", (4, 4, 4, 4, 4, 4, 4, 4), 4, "uniform"),
    # Context sensitivity: the same crossover at 16K, 64K and 128K.
    _case("util75_bimodal_b16_c16k_s4", (2, 2, 2, 2, 1, 1, 1, 1), 4, "bimodal"),
    _case("util100_uniform_b16_c16k_s4", (2, 2, 2, 2, 2, 2, 2, 2), 4, "uniform"),
    _case("util62_bimodal_b16_c64k_s4", (8, 8, 8, 8, 2, 2, 2, 2), 4, "bimodal"),
    _case("util75_bimodal_b16_c64k_s4", (8, 8, 8, 8, 4, 4, 4, 4), 4, "bimodal"),
    _case("util87_bimodal_b16_c64k_s4", (8, 8, 8, 8, 6, 6, 6, 6), 4, "bimodal"),
    _case("util100_uniform_b16_c64k_s4", (8, 8, 8, 8, 8, 8, 8, 8), 4, "uniform"),
    # 128K is the largest context the kernel admits: s_prior stays a multiple of
    # p_max and _MAX_S_PRIOR_ACCURATE_ROPE is 2**17, so 16 FA tiles per full sequence.
    _case("util50_bimodal_b16_c128k_s4", (16, 16, 8, 8, 4, 4, 4, 4), 4, "bimodal"),
    _case("util62_bimodal_b16_c128k_s4", (16, 16, 16, 16, 4, 4, 4, 4), 4, "bimodal"),
    _case("util75_bimodal_b16_c128k_s4", (16, 16, 16, 16, 8, 8, 8, 8), 4, "bimodal"),
    _case("util87_bimodal_b16_c128k_s4", (16, 16, 16, 16, 12, 12, 12, 12), 4, "bimodal"),
    _case("util90_bimodal_b16_c128k_s4", (16, 16, 16, 16, 13, 13, 13, 13), 4, "bimodal"),
    _case("util100_uniform_b16_c128k_s4", (16,) * 8, 4, "uniform"),
    # Slot width at the acceptance utilization. All three of s2/s4/s8 do the SAME 232
    # slot-tiles here, so they isolate the merge workspace (staged_records grows as
    # num_groups x num_sources) from tile elimination.
    _case("util90_bimodal_b16_c128k_s2", (16, 16, 16, 16, 13, 13, 13, 13), 2, "bimodal", "util90_bimodal_b16_c128k_s4"),
    _case("util90_bimodal_b16_c128k_s8", (16, 16, 16, 16, 13, 13, 13, 13), 8, "bimodal", "util90_bimodal_b16_c128k_s4"),
    _case(
        "util90_bimodal_b16_c128k_s16", (16, 16, 16, 16, 13, 13, 13, 13), 16, "bimodal", "util90_bimodal_b16_c128k_s4"
    ),
    # Slot width: two and eight slots over the 32K length vectors above, so
    # each shares that vector's unpacked baseline.
    _case("util62_bimodal_b16_c32k_s2", (4, 4, 4, 4, 1, 1, 1, 1), 2, "bimodal", "util62_bimodal_b16_c32k_s4"),
    _case("util87_bimodal_b16_c32k_s2", (4, 4, 4, 4, 3, 3, 3, 3), 2, "bimodal", "util87_bimodal_b16_c32k_s4"),
    _case("util100_uniform_b16_c32k_s2", (4, 4, 4, 4, 4, 4, 4, 4), 2, "uniform", "util100_uniform_b16_c32k_s4"),
    _case("util62_bimodal_b16_c32k_s8", (4, 4, 4, 4, 1, 1, 1, 1), 8, "bimodal", "util62_bimodal_b16_c32k_s4"),
    _case("util87_bimodal_b16_c32k_s8", (4, 4, 4, 4, 3, 3, 3, 3), 8, "bimodal", "util87_bimodal_b16_c32k_s4"),
    _case("util100_uniform_b16_c32k_s8", (4, 4, 4, 4, 4, 4, 4, 4), 8, "uniform", "util100_uniform_b16_c32k_s4"),
    # Focused eight-slot alternatives for four-slot geometries that remain
    # above the sweep's gap-to-expected target.
    _case("util75_bimodal_b16_c16k_s8", (2, 2, 2, 2, 1, 1, 1, 1), 8, "bimodal", "util75_bimodal_b16_c16k_s4"),
    _case("util75_bimodal_b16_c32k_s8", (4, 4, 4, 4, 2, 2, 2, 2), 8, "bimodal", "util75_bimodal_b16_c32k_s4"),
    _case("util87_bimodal_b16_c128k_s8", (16, 16, 16, 16, 12, 12, 12, 12), 8, "bimodal", "util87_bimodal_b16_c128k_s4"),
    # Batch scaling at 32K context and four slots.
    _case("util62_bimodal_b32_c32k_s4", (4,) * 8 + (1,) * 8, 4, "bimodal"),
    _case("util87_bimodal_b32_c32k_s4", (4,) * 8 + (3,) * 8, 4, "bimodal"),
    _case("util100_uniform_b32_c32k_s4", (4,) * 16, 4, "uniform"),
    _case("util62_bimodal_b32_c32k_s8", (4,) * 8 + (1,) * 8, 8, "bimodal", "util62_bimodal_b32_c32k_s4"),
    _case("util87_bimodal_b32_c32k_s8", (4,) * 8 + (3,) * 8, 8, "bimodal", "util87_bimodal_b32_c32k_s4"),
    _case("util100_uniform_b32_c32k_s8", (4,) * 16, 8, "uniform", "util100_uniform_b32_c32k_s4"),
    _case("util62_bimodal_b48_c32k_s4", (4,) * 12 + (1,) * 12, 4, "bimodal"),
    _case("util87_bimodal_b48_c32k_s4", (4,) * 12 + (3,) * 12, 4, "bimodal"),
    _case("util100_uniform_b48_c32k_s4", (4,) * 24, 4, "uniform"),
    _case("util62_bimodal_b48_c32k_s8", (4,) * 12 + (1,) * 12, 8, "bimodal", "util62_bimodal_b48_c32k_s4"),
    _case("util87_bimodal_b48_c32k_s8", (4,) * 12 + (3,) * 12, 8, "bimodal", "util87_bimodal_b48_c32k_s4"),
    _case("util100_uniform_b48_c32k_s8", (4,) * 24, 8, "uniform", "util100_uniform_b48_c32k_s4"),
    _case("util100_uniform_b48_c32k_s16", (4,) * 24, 16, "uniform", "util100_uniform_b48_c32k_s4"),
    # Low occupancy, where packing has the most idle cache to reclaim. A sequence
    # always costs at least one FA tile, so the lowest reachable utilization is
    # (max_tiles + n_local - 1) / (max_tiles * n_local): 34% at 32K, 23% at 64K and
    # 18% at 128K for these eight local sequences. 25% is therefore unreachable at
    # 32K, and util34 below is that context's floor.
    _case("util34_longtail_b16_c32k_s4", (4, 1, 1, 1, 1, 1, 1, 1), 4, "long-tail"),
    _case("util34_longtail_b16_c32k_s8", (4, 1, 1, 1, 1, 1, 1, 1), 8, "long-tail", "util34_longtail_b16_c32k_s4"),
    _case("util25_longtail_b16_c64k_s4", (8, 2, 1, 1, 1, 1, 1, 1), 4, "long-tail"),
    _case("util50_bimodal_b16_c64k_s4", (8, 8, 4, 4, 2, 2, 2, 2), 4, "bimodal"),
    _case("util25_longtail_b16_c128k_s4", (16, 4, 2, 2, 2, 2, 2, 2), 4, "long-tail"),
    _case("util25_longtail_b16_c128k_s8", (16, 4, 2, 2, 2, 2, 2, 2), 8, "long-tail", "util25_longtail_b16_c128k_s4"),
    # Distribution sensitivity at fixed 75% utilization; util75_bimodal_b16_c32k_s4
    # is the reference these are compared against.
    _case("util75_staircase_b16_c32k_s4", (4, 4, 4, 3, 3, 2, 2, 2), 4, "staircase"),
    _case("util75_longtail_b16_c32k_s4", (4, 4, 4, 4, 4, 2, 1, 1), 4, "long-tail"),
    # Final-row padding; util75_bimodal_b16_c32k_s4 is the zero-padding reference.
    _case("util65_paddedfinalrow_b16_c32k_s4", (4, 4, 4, 4, 2, 1, 1, 1), 4, "padded-final-row"),
    _case("util71_paddedfinalrow_b16_c32k_s4", (4, 4, 4, 4, 3, 2, 1, 1), 4, "padded-final-row"),
)

EXISTING_PACKED_CASES = {
    "util62_bimodal_b16_c32k_s4": "b16_mixed32k8k_lnc2_d64_s4_profile",
    "util100_uniform_b16_c16k_s4": "b16_uniform16k_lnc2_d64_s4_profile",
}

EXISTING_UNPACKED_CASES = {
    "util62_bimodal_b16_c32k_s4": "test_attn_tkg_unpacked_mixed32k8k_lnc2_d64_baseline",
    "util100_uniform_b16_c16k_s4": "test_attn_tkg_unpacked_uniform16k_lnc2_d64_baseline",
}

# Packed kernels require num_slots <= batch / lnc, so these widths are legal only when one NC
# owns the whole batch. They stay in the spec pools for the LNC1 ladders and leave every
# mirrored LNC2 sweep; _assert_lnc2_cases_route below rejects a wide case missing from here.
LNC1_ONLY_CASE_IDS = frozenset({"util90_bimodal_b16_c128k_s16"})

PACKED_SWEEP_CASES = tuple(case for case in COMPARISON_CASES if case["id"] not in EXISTING_PACKED_CASES)
assert LNC1_ONLY_CASE_IDS <= {case["id"] for case in PACKED_SWEEP_CASES}, "stale id in LNC1_ONLY_CASE_IDS"


def _build_unpacked_cases(comparisons, excluded_baseline_ids=()):
    """Return one unpacked baseline for each distinct length vector.

    Slot-width variants perform the same unpacked work and therefore share a
    baseline. ``excluded_baseline_ids`` names workloads already covered by
    separately defined baseline tests.
    """
    cases = []
    seen = set()
    for comparison in comparisons:
        key = comparison["lengths"]
        if key in seen:
            continue
        seen.add(key)
        baseline_id = comparison["baseline_id"]
        if baseline_id in excluded_baseline_ids:
            continue
        cases.append(
            {
                "id": baseline_id,
                "lengths": comparison["lengths"],
                "distribution": comparison["distribution"],
            }
        )
    return tuple(cases)


UNPACKED_SWEEP_CASES = _build_unpacked_cases(COMPARISON_CASES, EXISTING_UNPACKED_CASES)

# FP8-packed auto-allocation sweep: all 28 four-slot cases plus
# targeted S2, S8, and S16 alternatives used to compare scheduling geometry.
_FP8_PACKED_AUTO_ALLOC_SLOT_VARIANTS = {
    "util87_bimodal_b16_c32k_s2",
    "util100_uniform_b16_c32k_s2",
    "util87_bimodal_b16_c32k_s8",
    "util100_uniform_b16_c32k_s8",
    "util75_bimodal_b16_c16k_s8",
    "util75_bimodal_b16_c32k_s8",
    "util87_bimodal_b16_c128k_s8",
    "util90_bimodal_b16_c128k_s8",
    "util90_bimodal_b16_c128k_s2",
    "util90_bimodal_b16_c128k_s16",
    "util62_bimodal_b32_c32k_s8",
    "util87_bimodal_b32_c32k_s8",
    "util100_uniform_b32_c32k_s8",
    "util34_longtail_b16_c32k_s8",
    "util25_longtail_b16_c128k_s8",
    "util62_bimodal_b48_c32k_s8",
    "util87_bimodal_b48_c32k_s8",
    "util100_uniform_b48_c32k_s8",
    "util100_uniform_b48_c32k_s16",
}
FP8_PACKED_AUTO_ALLOC_SWEEP_SPECS = tuple(
    case
    for case in COMPARISON_CASES
    if (case["slots"] == 4 and case["id"] not in EXISTING_PACKED_CASES)
    or case["id"] in _FP8_PACKED_AUTO_ALLOC_SLOT_VARIANTS
)
assert len(FP8_PACKED_AUTO_ALLOC_SWEEP_SPECS) == 48
assert sum(case["slots"] == 4 for case in FP8_PACKED_AUTO_ALLOC_SWEEP_SPECS) == 29
assert sum(case["slots"] == 2 for case in FP8_PACKED_AUTO_ALLOC_SWEEP_SPECS) == 3
assert sum(case["slots"] == 8 for case in FP8_PACKED_AUTO_ALLOC_SWEEP_SPECS) == 14
assert sum(case["slots"] == 16 for case in FP8_PACKED_AUTO_ALLOC_SWEEP_SPECS) == 2


# A slot-width variant reuses its length vector's unpacked baseline. The two
# asymmetric batch-8 focus workloads are appended because COMPARISON_CASES does
# not otherwise contain those vectors.
FP8_PACKED_AUTO_ALLOC_UNPACKED_CASES = _build_unpacked_cases(FP8_PACKED_AUTO_ALLOC_SWEEP_SPECS) + tuple(
    {
        "id": spec["id"],
        "lengths": spec["lengths"],
        "distribution": "optimization-focus",
        "s_prior": 131072,
    }
    for spec in GPT_OSS_BATCH8_128K_FOCUS_SPECS
)
assert len(FP8_PACKED_AUTO_ALLOC_UNPACKED_CASES) == 31


def _fp8_sampler(value_range=None):
    """FP8 sampler, optionally over a narrower range than the shared default.

    The shared generator draws +/-240, which is right for exercising FP8 range and
    precision and wrong for a sink case: unscaled FP8 scores span thousands, so the
    softmax is too peaked for any sink value to blend.
    """
    if value_range is None:
        return np_random_sample_fp8()

    def generator(shape, dtype, name=None):
        rand = np.random.random_sample(shape) * (2.0 * value_range) - value_range
        return dt.static_cast(rand, dtype)

    return generator


def _build_ti_table(schedule, d_head, s_active_qh):
    """Build snake-layout TI rows; invalid slots use sequence zero as a safe source."""
    num_slots = len(schedule[0])
    num_elem = num_slots * s_active_qh
    cols = math.ceil(num_elem / 16)
    table = np.zeros((len(schedule), d_head, cols), dtype=np.uint16)
    for row_idx, row in enumerate(schedule):
        for slot, (seq_id, _) in enumerate(row):
            safe_seq_id = max(seq_id, 0)
            for position in range(s_active_qh):
                logical_idx = slot * s_active_qh + position
                table[row_idx, logical_idx % 16, logical_idx // 16] = safe_seq_id * s_active_qh + position
        for partition_start in range(16, d_head, 16):
            table[row_idx, partition_start : partition_start + 16] = table[row_idx, 0:16]
    return table


def _build_seq_id_table(schedule):
    """Build the canonical absolute sequence ID for every schedule slot."""
    return np.asarray([[seq_id for seq_id, _ in row] for row in schedule], dtype=np.int32)


def _build_packed_accumulator_ti_tables(
    schedule,
    d_head,
    s_active_qh,
    rows_per_nc,
    bs_per_nc,
):
    """Derive grouped-accumulator TI routes from the semantic schedule.

    The shared Torch helper groups duplicate sequence IDs within each row,
    converts absolute IDs to NC-local persistent-state offsets, and assigns
    unused groups to a dummy state record. The returned arrays are
    the partial-scatter and group-state routes consumed by the kernel wrapper;
    they do not add new attention semantics beyond ``schedule``.
    """
    seq_id_table = torch.tensor(
        [[seq_id for seq_id, _ in row] for row in schedule],
        dtype=torch.int32,
    )
    routes = build_packed_accumulator_route_tables(
        seq_id_table,
        d_head=d_head,
        s_active_qh=s_active_qh,
        rows_per_nc=rows_per_nc,
        bs_per_nc=bs_per_nc,
    )
    return tuple(table.numpy() for table in routes)


def _build_packed_blocks(per_sequence_table, schedule, num_slots, block_len):
    """Pack selected cache blocks, right-aligning partial terminal tiles."""
    blocks_per_tile = FA_TILE_SIZE // block_len
    packed = np.full((len(schedule), num_slots * blocks_per_tile), INVALID, dtype=np.int32)
    for row_idx, row in enumerate(schedule):
        for slot, (seq_id, tile_idx) in enumerate(row):
            if seq_id < 0:
                continue
            source_start = tile_idx * blocks_per_tile
            source_end = min(source_start + blocks_per_tile, per_sequence_table.shape[1])
            source = per_sequence_table[seq_id, source_start:source_end]
            valid_source = source[source != INVALID]
            destination_start = slot * blocks_per_tile + blocks_per_tile - valid_source.size
            packed[row_idx, destination_start : destination_start + valid_source.size] = valid_source
    return packed


def _build_packed_mask(lengths, schedule, q_head, s_active, num_slots, block_len):
    """Build masks for packed block-KV layout with partial terminal tiles right-aligned.

    The layout exactly matches block-KV QK layout:
    ``[row, P, fold * block_len, slot, q_head, s_active]``. Right alignment keeps
    each sequence's terminal range at the physical tile tail. Active tokens are
    never exposed here: the kernel folds them into the running
    statistics at finalization.
    """
    n_sprior_tiles = FA_TILE_SIZE // P_MAX
    num_folds = FA_TILE_SIZE // (P_MAX * block_len)
    masks = np.zeros(
        (len(schedule), P_MAX, n_sprior_tiles, num_slots, q_head, s_active),
        dtype=np.uint8,
    )
    for row_idx, row in enumerate(schedule):
        for slot, (seq_id, tile_idx) in enumerate(row):
            if seq_id < 0:
                continue
            cache_len = lengths[seq_id] - s_active
            tile_start = tile_idx * FA_TILE_SIZE
            is_active_slot = tile_start <= cache_len < tile_start + FA_TILE_SIZE
            tile_token_count = max(0, min(lengths[seq_id], tile_start + FA_TILE_SIZE) - tile_start)
            terminal_offset = FA_TILE_SIZE - tile_token_count if is_active_slot else 0
            for fold in range(num_folds):
                for partition in range(P_MAX):
                    for block_offset in range(block_len):
                        free_idx = fold * block_len + block_offset
                        physical_idx = fold * P_MAX * block_len + partition * block_len + block_offset
                        logical_idx = physical_idx - terminal_offset
                        if logical_idx < 0 or logical_idx >= FA_TILE_SIZE:
                            continue
                        token_idx = tile_start + logical_idx
                        if token_idx < cache_len:
                            masks[row_idx, partition, free_idx, slot, :, :] = 1
                        # Active-token positions stay masked: the kernel folds
                        # active K/V into the running statistics at finalization
                        # rather than stitching it into a cache tile, so these
                        # tile positions hold stale cache data.
    return masks.reshape((len(schedule), P_MAX, n_sprior_tiles * num_slots * q_head * s_active))


@nki.jit
def attn_tkg_packed_wrapper(
    q,
    k_active,
    v_active,
    k_prior,
    v_prior,
    mask,
    rope_pos_ids,
    sink,
    active_blocks_table,
    active_blocks_table_orig,
    q_index_table,
    seq_id_table,
    accumulator_partial_route_table,
    accumulator_group_state_route_table,
    attn_out_shape,
    attn_out_sb_shape,
    cfg: AttnTKGConfig,
    dtype,
    use_auto_alloc,
):
    """Stage test inputs in SBUF, run packed attention, and return HBM output."""
    _compute_tile_params(
        cfg,
        TileConstants.get_tile_constants(),
        q,
        k_prior,
        v_prior,
        k_active,
        v_active,
        active_blocks_table,
    )
    if use_auto_alloc:
        sbm = create_auto_alloc_manager(Logger("SBM"))
    else:
        sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size - 16 * 1024, Logger("SBM"))
    sbm.open_scope()

    d_tile_size = min(cfg.d_head, P_MAX)
    n_d_tiles = math.ceil(cfg.d_head / P_MAX)
    out = sbm.alloc_stack(attn_out_sb_shape, dtype=dtype, buffer=nl.sbuf)

    q_free = cfg.bs * cfg.q_head * cfg.s_active
    q_input = sbm.alloc_stack((d_tile_size, n_d_tiles * q_free), dtype=q.dtype, buffer=nl.sbuf)
    k_free = cfg.bs * cfg.s_active
    k_active_input = sbm.alloc_stack((d_tile_size, n_d_tiles * k_free), dtype=k_active.dtype, buffer=nl.sbuf)
    for d_tile in range(n_d_tiles):
        d_start = d_tile * d_tile_size
        nisa.dma_copy(
            q_input[:, d_tile * q_free : (d_tile + 1) * q_free],
            q[d_start : d_start + d_tile_size, :],
        )
        nisa.dma_copy(
            k_active_input[:, d_tile * k_free : (d_tile + 1) * k_free],
            k_active[d_start : d_start + d_tile_size, :],
        )

    attn_out, _ = attention_tkg_seq_packed(
        q_input,
        k_active_input,
        v_active,
        k_prior,
        v_prior,
        mask,
        out,
        cfg,
        sbm,
        rope_pos_ids=rope_pos_ids,
        sink=sink,
        active_blocks_table=active_blocks_table,
        q_index_table=q_index_table,
        seq_id_table=seq_id_table,
        accumulator_partial_route_table=accumulator_partial_route_table,
        accumulator_group_state_route_table=accumulator_group_state_route_table,
    )

    output = nl.ndarray(attn_out_shape, dtype=attn_out.dtype, buffer=nl.shared_hbm)
    for d_tile in range(n_d_tiles):
        d_start = d_tile * d_tile_size
        src_offset = d_tile * q_free
        nisa.dma_copy(
            dst=output[d_start : d_start + d_tile_size, :],
            src=attn_out[:, src_offset : src_offset + q_free],
        )
    sbm.close_scope()
    return output


DESIGN_DOC_SCHEDULE = [
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(3, 0), (3, 1), (3, 2), (3, 3)],
    [(3, 4), (3, 5), (3, 6), (3, 7)],
    [(4, 0), (5, 0), (5, 1), (5, 2)],
    [(5, 3), (6, 0), (6, 1), (7, 0)],
    [(7, 1), (INVALID, 0), (INVALID, 0), (INVALID, 0)],
]
DESIGN_DOC_LENGTHS = [4096, 8192, 16384, 65536, 8192, 32768, 16384, 10240]

GPT_B8_LENGTHS = [4096, 8192, 16384, 32768, 4096, 12288, 24576, 32768]
GPT_B8_UNPADDED_SCHEDULE = [
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(3, 0), (3, 1), (3, 2), (3, 3)],
    [(4, 0), (5, 0), (5, 1), (6, 0)],
    [(6, 1), (6, 2), (7, 0), (7, 1)],
    [(7, 2), (7, 3), (INVALID, 0), (INVALID, 0)],
]
GPT_B8_PADDED_SCHEDULE = [
    *GPT_B8_UNPADDED_SCHEDULE[:2],
    [(INVALID, 0)] * 4,
    [(INVALID, 0)] * 4,
    *GPT_B8_UNPADDED_SCHEDULE[2:],
    [(INVALID, 0)] * 4,
]


CASES = [
    {
        "name": "gpt_oss_20b_tp8_util50_b8_h8_d64_sprior32k_block64",
        "lengths": list(GPT_OSS_20B_TP8_UTIL50_LENGTHS),
        # Flattened [NC0 rows 0..3][NC1 rows 0..3]. The model's B=8
        # request bucket is batch-sharded across LNC2.
        "schedule": GPT_OSS_20B_TP8_UTIL50_SCHEDULE,
        "q_head": 8,
        "s_active": 1,
        "d_head": 64,
        "block_len": 64,
        "curr_sprior": 32768,
        "rows_per_nc": 4,
        "sink_values": [20.0, -20.0, 2.0, 8.0, 4.0, -4.0, 1.0, 0.0],
        "lnc": 2,
    },
    {
        "name": "gpt_oss_120b_tp8_util50_b8_h8_d64_sprior32k_block64",
        "lengths": list(GPT_OSS_120B_TP8_UTIL50_LENGTHS),
        # GPT-OSS 120B has the same TP8 attention geometry as 20B; its extra
        # layers and experts do not change this rank-local raw-attention case.
        "schedule": GPT_OSS_120B_TP8_UTIL50_SCHEDULE,
        "q_head": 8,
        "s_active": 1,
        "d_head": 64,
        "block_len": 64,
        "curr_sprior": 32768,
        "rows_per_nc": 4,
        "sink_values": [20.0, -20.0, 2.0, 8.0, 4.0, -4.0, 1.0, 0.0],
        "lnc": 2,
    },
    {
        "name": "gpt_b8_h8_d64_sprior32k_block64",
        "lengths": GPT_B8_LENGTHS,
        # Flattened [NC0 rows 0..3][NC1 rows 0..3]. Invalid slots select
        # the no-op partial, so the fixed padding rows leave output unchanged.
        "schedule": GPT_B8_PADDED_SCHEDULE,
        "unpadded_schedule": GPT_B8_UNPADDED_SCHEDULE,
        "q_head": 8,
        "s_active": 1,
        "d_head": 64,
        "block_len": 64,
        "curr_sprior": 32768,
        "rows_per_nc": 4,
        "sink_values": [20.0, -20.0, 2.0, 8.0, 4.0, -4.0, 1.0, 0.0],
        "lnc": 2,
    },
    {
        "name": "b8_s4_six_iterations",
        "lengths": DESIGN_DOC_LENGTHS,
        "schedule": DESIGN_DOC_SCHEDULE,
        "lnc": 1,
    },
    {
        "name": "b16_mixed32k8k_lnc2_d64_s4_profile",
        "lengths": [32768] * 4 + [8192] * 4 + [32768] * 4 + [8192] * 4,
        # Flattened [NC0 rows][NC1 rows]. LNC2 batch-shards this BQS=128
        # case. Each NC packs four 32K sequences and four 8K sequences.
        "schedule": [
            [(seq_id, tile_idx) for seq_id in range(seq_start, seq_start + 4)]
            for nc_start in (0, 8)
            for seq_start, tile_idx in (
                (nc_start, 0),
                (nc_start + 4, 0),
                (nc_start, 1),
                (nc_start, 2),
                (nc_start, 3),
            )
        ],
        "q_head": 8,
        "s_active": 1,
        "d_head": 64,
        "lnc": 2,
    },
    {
        "name": "b16_uniform16k_lnc2_d64_s4_profile",
        "lengths": [16384] * 16,
        # Flattened [NC0 rows][NC1 rows] for LNC2 batch sharding.
        "schedule": [
            [(seq_id, tile_idx) for seq_id in range(seq_start, seq_start + 4)]
            for nc_start in (0, 8)
            for tile_idx in range(2)
            for seq_start in range(nc_start, nc_start + 8, 4)
        ],
        "q_head": 8,
        "s_active": 1,
        "d_head": 64,
        "lnc": 2,
    },
    {
        "name": "permute_two_tiles",
        "lengths": [16384] * 4,
        "schedule": [
            [(2, 0), (0, 0), (3, 0), (1, 0)],
            [(1, 1), (3, 1), (0, 1), (2, 1)],
        ],
        # Keep one representative case on the serial path so the optional-route
        # fallback remains covered while the rest of the suite exercises the
        # runtime-routed accumulator across supported geometries.
        "use_grouped_accumulator_routes": False,
        "lnc": 1,
    },
    {
        "name": "two_multi_tile_groups",
        "lengths": [8192, 16384, 16384, 8192],
        # Two sequences carry a second tile; the freed slots are padded. Slot
        # work must be unique, so the tails cannot be replicated across slots.
        "schedule": [
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            [(1, 1), (2, 1), (INVALID, 0), (INVALID, 0)],
        ],
        "lnc": 1,
    },
    {
        "name": "b8_mixed_lengths",
        "lengths": [8192, 16384, 8192, 16384, 8192, 16384, 8192, 16384],
        "schedule": [
            [(seq_id, 0) for seq_id in range(8)],
            [(1, 1), (3, 1), (5, 1), (7, 1)] + [(INVALID, 0)] * 4,
        ],
        "lnc": 1,
    },
    {
        "name": "pairwise_different_tiles",
        "lengths": [16384] * 4,
        "schedule": [
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(2, 0), (2, 1), (3, 0), (3, 1)],
        ],
        "lnc": 1,
    },
    {
        "name": "four_tiles_same_sequence_per_iteration",
        "lengths": [32768] * 4,
        "schedule": [[(seq_id, tile_id) for tile_id in range(4)] for seq_id in range(4)],
        "lnc": 1,
    },
    {
        "name": "d64_permuted",
        "lengths": [16384] * 4,
        "schedule": [
            [(2, 0), (0, 0), (3, 0), (1, 0)],
            [(1, 1), (3, 1), (0, 1), (2, 1)],
        ],
        "d_head": 64,
        "distinct_active_kv": True,
        "lnc": 1,
    },
    {
        "name": "d64_mixed_lengths",
        "lengths": [8192, 16384, 16384, 8192],
        "schedule": [
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            [(1, 1), (2, 1), (INVALID, 0), (INVALID, 0)],
        ],
        "d_head": 64,
        "distinct_active_kv": True,
        "prior_weight_sensitive": True,
        "lnc": 1,
    },
    {
        "name": "full_row_wide_transpose",
        "lengths": [8192] * 8,
        "schedule": [
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            [(4, 0), (5, 0), (6, 0), (7, 0)],
        ],
        "q_head": 8,
        "lnc": 1,
    },
    {
        "name": "lnc2_batch_nc_local_rows",
        "lengths": [16384] * 8,
        # Flattened [NC0 rows][NC1 rows]. Each NC schedules both tiles for
        # its four persistent sequences. The kernel derives NC-local
        # accumulator destinations from the absolute IDs.
        "schedule": [
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            [(0, 1), (1, 1), (2, 1), (3, 1)],
            [(4, 0), (5, 0), (6, 0), (7, 0)],
            [(4, 1), (5, 1), (6, 1), (7, 1)],
        ],
        "lnc": 2,
    },
    {
        "name": "lnc2_batch_two_slots_per_nc",
        "lengths": [8192] * 8,
        # Batch sharding is forced by B*QH*S_active > P_MAX. Four global
        # slots are explicitly two slots on each NC.
        "schedule": [
            [(0, 0), (1, 0)],
            [(2, 0), (3, 0)],
            [(4, 0), (5, 0)],
            [(6, 0), (7, 0)],
        ],
        "q_head": 4,
        "distinct_active_kv": True,
        "lnc": 2,
    },
    {
        "name": "lnc2_multi_tile_padded_routing",
        "lengths": [8192, 16384, 8192, 8192, 8192, 16384, 8192, 8192],
        # Both NCs accumulate a second row holding one sequence's tail tile plus
        # padded slots. NC1 uses absolute IDs and must derive local destinations.
        "schedule": [
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            [(1, 1), (INVALID, 0), (INVALID, 0), (INVALID, 0)],
            [(4, 0), (5, 0), (6, 0), (7, 0)],
            [(5, 1), (INVALID, 0), (INVALID, 0), (INVALID, 0)],
        ],
        "d_head": 64,
        "q_head": 4,
        "s_active": 1,
        "distinct_active_kv": True,
        "prior_weight_sensitive": True,
        "lnc": 2,
    },
    {
        "name": "blocklen128_resize2_b4_d64_sprior16k",
        # block_len 128 exceeds fa_tile_size // p_max = 64, so the kernel halves it to
        # 64 and each caller block expands to 2 physical blocks. Covers the resized
        # packed block-table path.
        "lengths": [16384] * 4,
        "schedule": [
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            [(0, 1), (1, 1), (2, 1), (3, 1)],
        ],
        "q_head": 4,
        "s_active": 1,
        "d_head": 64,
        "block_len": 128,
        "expect_resize_factor": 2,
        "lnc": 1,
    },
    {
        "name": "blocklen256_resize4_b4_d64_sprior16k_multitile",
        # Factor 4, with a padded final row and a sequence contributing two tiles, so
        # the expansion is exercised alongside padded routing.
        "lengths": [16384, 16384, 8192, 8192],
        "schedule": [
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            [(0, 1), (1, 1), (INVALID, 0), (INVALID, 0)],
        ],
        "q_head": 4,
        "s_active": 1,
        "d_head": 64,
        "block_len": 256,
        "expect_resize_factor": 4,
        "distinct_active_kv": True,
        "prior_weight_sensitive": True,
        "lnc": 1,
    },
    {
        # Mirrors the unpacked kernel's resize configs, which trigger the other resize
        # path: too few blocks per batch rather than an oversized block_len. s_prior is
        # not a multiple of the FA tile either, so the partial terminal tile is exercised
        # at the same time. bs=8/lnc=2 -> 4 sequences per NC, factor 8.
        "name": "blocklen16_resize8_b8_lnc2_sprior16640",
        "lengths": [16640] * 8,
        "schedule": [
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            [(0, 1), (1, 1), (2, 1), (3, 1)],
            [(0, 2), (1, 2), (2, 2), (3, 2)],
            [(4, 0), (5, 0), (6, 0), (7, 0)],
            [(4, 1), (5, 1), (6, 1), (7, 1)],
            [(4, 2), (5, 2), (6, 2), (7, 2)],
        ],
        "q_head": 8,
        "s_active": 5,
        "d_head": 128,
        "block_len": 16,
        "curr_sprior": 16640,
        "expect_resize_factor": 8,
        "rows_per_nc": 3,
        "lnc": 2,
    },
    {
        # Factor 16 reduces the cache block length to 1, the most aggressive expansion
        # the kernel will ever perform: every caller block becomes 16 physical blocks.
        "name": "blocklen16_resize16_b4_lnc2_sprior16640",
        "lengths": [16640] * 4,
        "schedule": [
            [(0, 0), (1, 0)],
            [(0, 1), (1, 1)],
            [(0, 2), (1, 2)],
            [(2, 0), (3, 0)],
            [(2, 1), (3, 1)],
            [(2, 2), (3, 2)],
        ],
        "q_head": 1,
        "s_active": 5,
        "d_head": 128,
        "block_len": 16,
        "curr_sprior": 16640,
        "expect_resize_factor": 16,
        "rows_per_nc": 3,
        "lnc": 2,
    },
    {
        # Factor 2 reached via the block-count trigger rather than the FA-divisibility
        # trigger that blocklen128_resize2 covers.
        "name": "blocklen16_resize2_b4_lnc2_sprior10240",
        "lengths": [10240] * 4,
        "schedule": [
            [(0, 0), (1, 0)],
            [(0, 1), (1, 1)],
            [(2, 0), (3, 0)],
            [(2, 1), (3, 1)],
        ],
        "q_head": 1,
        "s_active": 5,
        "d_head": 128,
        "block_len": 16,
        "curr_sprior": 10240,
        "expect_resize_factor": 2,
        "rows_per_nc": 2,
        "lnc": 2,
    },
    {
        "name": "sink_late_sequence_and_multi_tile",
        "lengths": [16384, 16384, 8192, 8192],
        # Sequence zero occupies two row-zero slots while the other persistent
        # sequences first appear later. Sink must be folded once for all four.
        "schedule": [
            [(0, 0), (0, 1)],
            [(1, 0), (1, 1)],
            [(2, 0), (3, 0)],
        ],
        "q_head": 4,
        "s_active": 1,
        "d_head": 64,
        "sink_values": [20.0, -20.0, 2.0, 8.0],
        "lnc": 1,
    },
    {
        "name": "sink_late_sequence_and_padded_slots",
        "lengths": [8192, 16384, 8192, 8192],
        # Sequence one first appears after row zero, contributing one unique
        # tile per row with the remaining slots padded.
        "schedule": [
            [(0, 0), (2, 0), (3, 0), (INVALID, 0)],
            [(1, 0), (INVALID, 0), (INVALID, 0), (INVALID, 0)],
            [(1, 1), (INVALID, 0), (INVALID, 0), (INVALID, 0)],
        ],
        "q_head": 4,
        "s_active": 1,
        "d_head": 64,
        "sink_values": [20.0, -20.0, 2.0, 8.0],
        "lnc": 1,
    },
    {
        "name": "sink_lnc2_late_sequences",
        "lengths": [16384, 16384, 8192, 8192, 16384, 16384, 8192, 8192],
        # Flattened [NC0 rows][NC1 rows]. Each NC's first row contains two
        # tiles for one sequence; its other persistent sequences appear later.
        "schedule": [
            [(0, 0), (0, 1)],
            [(1, 0), (1, 1)],
            [(2, 0), (3, 0)],
            [(4, 0), (4, 1)],
            [(5, 0), (5, 1)],
            [(6, 0), (7, 0)],
        ],
        "q_head": 4,
        "s_active": 1,
        "d_head": 64,
        "sink_values": [20.0, -20.0, 2.0, 8.0],
        "lnc": 2,
    },
    # FP8 E4M3 KV cache. The packed path needs fuse_rope=False, a non-fp32 q and
    # qk_in_sb=True (attention_tkg_seq_packed.py FP8 preconditions); the shared cfg
    # above already satisfies all three.
    {
        "name": "fp8_kv_permute_two_tiles",
        "lengths": [16384] * 4,
        "schedule": [
            [(2, 0), (0, 0), (3, 0), (1, 0)],
            [(1, 1), (3, 1), (0, 1), (2, 1)],
        ],
        "fp8_kv": True,
        "lnc": 1,
    },
    {
        "name": "fp8_kv_d64_mixed_lengths",
        "lengths": [8192, 16384, 16384, 8192],
        "schedule": [
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            [(1, 1), (2, 1), (INVALID, 0), (INVALID, 0)],
        ],
        "d_head": 64,
        # Distinct per-sequence active K/V isolates the active-token fold: a
        # cross-sequence leak or a dropped active contribution changes the output
        # even where the prior tiles agree.
        "distinct_active_kv": True,
        "prior_weight_sensitive": True,
        "fp8_kv": True,
        "lnc": 1,
    },
    {
        "name": "fp8_kv_lnc2_batch_nc_local_rows",
        "lengths": [16384] * 8,
        "schedule": [
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            [(0, 1), (1, 1), (2, 1), (3, 1)],
            [(4, 0), (5, 0), (6, 0), (7, 0)],
            [(4, 1), (5, 1), (6, 1), (7, 1)],
        ],
        "fp8_kv": True,
        "lnc": 2,
    },
    {
        "name": "fp8_kv_sink_late_sequence_and_multi_tile",
        "lengths": [16384, 16384, 8192, 8192],
        "schedule": [
            [(0, 0), (0, 1)],
            [(1, 0), (1, 1)],
            [(2, 0), (3, 0)],
        ],
        "q_head": 4,
        "s_active": 1,
        "d_head": 64,
        "sink_values": [20.0, -20.0, 2.0, 8.0],
        # Narrow the FP8 KV draw for this case. Under FP8 the query is NOT scaled by
        # 1/sqrt(d_head): that scaling lives in the fuse_rope path, which FP8 forbids.
        # With the shared +/-240 sampler the scores then span ~+/-2400 and softmax over
        # thousands of keys is so peaked that NO fixed sink blends -- every value below
        # the key maximum is inert, anything above it takes weight 1, and the
        # sink-sensitivity guard compares reference against reference so it passes on
        # the collapse. E4M3 represents small values fine, so a +/-1 draw puts scores
        # near O(1) where the sink genuinely competes, which is what exercises the fold.
        "fp8_kv_range": 1.0,
        "fp8_kv": True,
        "lnc": 1,
    },
    {
        "name": "fp8_packed_grouped_duplicates",
        "lengths": [16384] * 8,
        # Each row has two sequence groups and two unused group IDs. This
        # exercises direct group-state routing and its dummy destination.
        "schedule": [
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(2, 0), (2, 1), (3, 0), (3, 1)],
            [(4, 0), (4, 1), (5, 0), (5, 1)],
            [(6, 0), (6, 1), (7, 0), (7, 1)],
        ],
        "q_head": 8,
        "s_active": 1,
        "d_head": 64,
        "fp8_kv": True,
        "fp8_packed": True,
        "lnc": 2,
    },
    {
        "name": "fp8_packed_grouped_padding",
        "lengths": [8192, 16384, 8192, 8192] * 2,
        # The second row on each NC has one real tail tile and three padded
        # slots. Unused groups must update only the dummy state record.
        "schedule": [
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            [(1, 1), (INVALID, 0), (INVALID, 0), (INVALID, 0)],
            [(4, 0), (5, 0), (6, 0), (7, 0)],
            [(5, 1), (INVALID, 0), (INVALID, 0), (INVALID, 0)],
        ],
        "q_head": 8,
        "s_active": 1,
        "d_head": 64,
        "distinct_active_kv": True,
        "prior_weight_sensitive": True,
        "fp8_kv": True,
        "fp8_packed": True,
        "lnc": 2,
    },
    {
        "name": "fp8_packed_grouped_sink",
        "lengths": [16384, 16384, 8192, 8192] * 2,
        # Combine duplicate sequence groups, singleton groups, padding, and a
        # competitive sink so finalization observes the fully merged state.
        "schedule": [
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(2, 0), (3, 0), (INVALID, 0), (INVALID, 0)],
            [(4, 0), (4, 1), (5, 0), (5, 1)],
            [(6, 0), (7, 0), (INVALID, 0), (INVALID, 0)],
        ],
        "q_head": 8,
        "s_active": 1,
        "d_head": 64,
        "sink_values": [20.0, -20.0, 2.0, 8.0, 4.0, -4.0, 1.0, 0.0],
        "fp8_kv_range": 1.0,
        "fp8_kv": True,
        "fp8_packed": True,
        "lnc": 2,
    },
]


@pytest_test_metadata(name="Attention TKG Sequence Packed")
@pytest_marks(["attention", "tkg"])
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
@final
class TestAttentionTkgSeqPackedKernel:
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "use_auto_alloc",
        [False, True],
        ids=["manualalloc", "autoalloc"],
    )
    @pytest.mark.parametrize("case", CASES, ids=[str(case["name"]) for case in CASES])
    def test_attn_tkg_seq_packed_schedule_contract(self, test_manager, platform_target, case, use_auto_alloc):
        """Validate packed schedules across allocators and accumulator lowerings."""
        if "trn3" not in str(platform_target).lower():
            pytest.skip("Tensor-indirection packed attention requires NeuronCore-v4 (Trn3).")
        np.random.seed(42)
        lengths = case["lengths"]
        schedule = case["schedule"]
        bs = len(lengths)
        num_slots = len(schedule[0])
        q_head = case.get("q_head", 1)
        d_head = case.get("d_head", 128)
        s_active = case.get("s_active", 5)
        s_active_qh = q_head * s_active
        s_prior = case.get("curr_sprior", max(lengths))
        block_len = case.get("block_len", DEFAULT_BLOCK_LEN)
        lnc = case["lnc"]
        fp8_kv = case.get("fp8_kv", False)
        fp8_packed = case.get("fp8_packed", False)
        assert not fp8_packed or fp8_kv, "fp8_packed needs an FP8 cache: the kernel asserts all four KV tensors."

        assert all(len(row) == num_slots for row in schedule)
        assert len(schedule) % lnc == 0
        if "rows_per_nc" in case:
            assert len(schedule) == lnc * case["rows_per_nc"]
        if "unpadded_schedule" in case:
            padded_work = [work for row in schedule for work in row if work[0] >= 0]
            unpadded_work = [work for row in case["unpadded_schedule"] for work in row if work[0] >= 0]
            assert padded_work == unpadded_work
        # Packed slot work must be unique: each slot merges its own partial exactly
        # once, so a repeated (sequence, tile) would be accumulated twice.
        valid_work = [work for row in schedule for work in row if work[0] >= 0]
        assert len(valid_work) == len(set(valid_work))

        cfg = AttnTKGConfig(
            bs=bs,
            q_head=q_head,
            d_head=d_head,
            s_active=s_active,
            curr_sprior=s_prior,
            full_sprior=s_prior,
            block_len=block_len,
            tp_k_prior=True,
            strided_mm1=False,
            use_pos_id=False,
            enable_fa_s_prior_tiling=True,
            qk_in_sb=True,
            k_out_in_sb=True,
            out_in_sb=True,
            fp8_packed=fp8_packed,
        )
        # The kernel may reduce the cache block length (see
        # resize_cache_block_len_for_attention_tkg_kernel). The packed block table stays
        # in the caller's block units -- the kernel expands each entry into resize_factor
        # physical blocks -- but the packed mask is laid out over the kernel's QK columns,
        # which are in reduced units, so it must be built with the reduced length.
        resized_block_len, resize_factor = resize_cache_block_len_for_attention_tkg_kernel(
            num_blocks_per_batch=s_prior // block_len,
            block_len=block_len,
            lnc=lnc,
            p_max=P_MAX,
            bs=bs,
            q_head=q_head,
            s_active=s_active,
            full_sprior=s_prior,
            enable_fa_s_prior_tiling=True,
        )
        assert resized_block_len * resize_factor == block_len
        assert case.get("expect_resize_factor", 1) == resize_factor
        compiler_args = CompilerArgs(
            platform_target=platform_target,
            logical_nc_config=lnc,
        )

        q_shape = (d_head, bs * s_active_qh)
        k_active_shape = (d_head, bs * s_active)
        num_cache_blocks = bs * s_prior // block_len
        cache_shape = (num_cache_blocks, block_len, d_head)
        # fp8_packed carries K as parity-split pairs: logical row 2*i+p lands at
        # [block, i, d_head, p], so the cache is 4-D and half as deep. V is unchanged.
        k_cache_shape = (num_cache_blocks, block_len // 2, d_head, 2) if fp8_packed else cache_shape
        n_d_tiles = math.ceil(d_head / P_MAX)
        d_tile_size = min(d_head, P_MAX)
        attn_out_shape = (d_head, bs * s_active_qh)
        attn_out_sb_shape = (d_tile_size, n_d_tiles * bs * s_active_qh)
        pos_id = np.asarray(lengths, dtype=np.int32)[:, None] - s_active

        q_index_table = _build_ti_table(schedule, d_head, s_active_qh)
        seq_id_table = _build_seq_id_table(schedule)
        accumulator_partial_route_table = None
        accumulator_group_state_route_table = None
        # Routes are a hardware lowering of seq_id_table. Most cases exercise
        # the grouped path; one case omits both tables to retain coverage of the
        # ordered fallback.
        if case.get("use_grouped_accumulator_routes", True):
            (
                accumulator_partial_route_table,
                accumulator_group_state_route_table,
            ) = _build_packed_accumulator_ti_tables(
                schedule,
                d_head,
                s_active_qh,
                rows_per_nc=len(schedule) // lnc,
                bs_per_nc=bs // lnc,
            )
        packed_mask = _build_packed_mask(lengths, schedule, q_head, s_active, num_slots, resized_block_len)

        def input_generator(_):
            random_gen = np_random_sample()
            # FP8 KV is only engaged when all four KV tensors are E4M3; a mixed set
            # is rejected by the kernel's `not any_fp8 or all_fp8` assert.
            kv_random_gen = _fp8_sampler(case.get("fp8_kv_range")) if fp8_kv else random_gen
            kv_dtype = _fp8_kv_dtype(platform_target) if fp8_kv else "bfloat16"
            if case.get("distinct_active_kv", False):
                q = np.ones(q_shape, dtype=np.float32).astype(nl.bfloat16)
                k_active_f32 = np.empty(k_active_shape, dtype=np.float32)
                v_active_f32 = np.empty((bs, 1, s_active, d_head), dtype=np.float32)
                for seq_id in range(bs):
                    active_sign = 1.0 if seq_id % 2 == 0 else -1.0
                    k_active_f32[:, seq_id * s_active : (seq_id + 1) * s_active] = 4.0 * active_sign
                    v_active_f32[seq_id, :, :, :] = float(seq_id + 1)
                k_active = k_active_f32.astype(kv_dtype)
                v_active = v_active_f32.astype(kv_dtype)
                if case.get("prior_weight_sensitive", False):
                    k_prior = kv_random_gen(shape=k_cache_shape, dtype=kv_dtype, name="k_prior").astype(kv_dtype)
                    v_prior = kv_random_gen(shape=cache_shape, dtype=kv_dtype, name="v_prior").astype(kv_dtype)
                else:
                    k_prior = np.zeros(k_cache_shape, dtype=np.float32).astype(kv_dtype)
                    v_prior = np.zeros(cache_shape, dtype=np.float32).astype(kv_dtype)
            else:
                q = random_gen(shape=q_shape, dtype="bfloat16", name="q")
                k_active = kv_random_gen(shape=k_active_shape, dtype=kv_dtype, name="k_active").astype(kv_dtype)
                v_active = kv_random_gen(shape=(bs, 1, s_active, d_head), dtype=kv_dtype, name="v_active").astype(
                    kv_dtype
                )
                k_prior = kv_random_gen(shape=k_cache_shape, dtype=kv_dtype, name="k_prior").astype(kv_dtype)
                v_prior = kv_random_gen(shape=cache_shape, dtype=kv_dtype, name="v_prior").astype(kv_dtype)
            rope_pos_ids = (
                np.broadcast_to(pos_id, (bs, s_active)).astype(np.float32)
                + np.arange(s_active, dtype=np.float32)[None, :]
            )
            sink_values = case.get("sink_values")
            sink = np.asarray(sink_values, dtype=np.float32).reshape(q_head, 1) if sink_values is not None else None
            per_sequence_table = gen_deterministic_active_block_table(
                bs,
                s_prior,
                s_active,
                pos_id,
                block_len,
                num_cache_blocks,
            ).astype(np.int32)
            active_blocks_table = _build_packed_blocks(per_sequence_table, schedule, num_slots, block_len)
            return {
                "q": q,
                "k_active": k_active,
                "v_active": v_active,
                "k_prior": k_prior,
                "v_prior": v_prior,
                "mask": packed_mask,
                "rope_pos_ids": rope_pos_ids,
                "sink": sink,
                "active_blocks_table": active_blocks_table,
                "active_blocks_table_orig": per_sequence_table,
                "q_index_table": q_index_table,
                "seq_id_table": seq_id_table,
                "accumulator_partial_route_table": accumulator_partial_route_table,
                "accumulator_group_state_route_table": accumulator_group_state_route_table,
                "attn_out_shape": attn_out_shape,
                "attn_out_sb_shape": attn_out_sb_shape,
                "cfg": cfg,
                "dtype": "bfloat16",
                "use_auto_alloc": use_auto_alloc,
            }

        def torch_ref(
            q,
            k_active,
            v_active,
            k_prior,
            v_prior,
            mask,
            rope_pos_ids,
            sink,
            active_blocks_table,
            active_blocks_table_orig,
            q_index_table,
            seq_id_table,
            accumulator_partial_route_table,
            accumulator_group_state_route_table,
            attn_out_shape,
            attn_out_sb_shape,
            cfg,
            dtype,
            use_auto_alloc,
        ):
            import neuron_dtypes as dt

            # Allocation mode and accumulator routes affect only the kernel
            # lowering. The reference evaluates the same attention semantics
            # from canonical per-sequence inputs and intentionally ignores them.
            active_mask = build_active_attention_mask(
                batch=bs, num_heads=q_head, s_active=s_active, transposed=True
            ).to(torch.uint8)
            out = torch.zeros(attn_out_shape, dtype=torch.float32)
            # The packed kernel consumes its prepacked mask directly. The generic
            # reference independently builds the equivalent per-sequence mask.
            ref_cfg = replace(cfg, use_pos_id=True)
            attn_out, _ = attention_tkg_torch_ref[lnc](
                q=q,
                k_active=k_active,
                v_active=v_active,
                k_prior=k_prior,
                v_prior=v_prior,
                mask=active_mask,
                out=out,
                cfg=ref_cfg,
                sbm=None,
                rope_pos_ids=rope_pos_ids,
                sink=sink,
                active_blocks_table=active_blocks_table_orig,
            )
            if sink is not None:
                no_sink_out, _ = attention_tkg_torch_ref[lnc](
                    q=q,
                    k_active=k_active,
                    v_active=v_active,
                    k_prior=k_prior,
                    v_prior=v_prior,
                    mask=active_mask,
                    out=torch.zeros_like(out),
                    cfg=ref_cfg,
                    sbm=None,
                    rope_pos_ids=rope_pos_ids,
                    sink=None,
                    active_blocks_table=active_blocks_table_orig,
                )
                sink_delta = torch.max(torch.abs(attn_out.float() - no_sink_out.float()))
                assert sink_delta > 1e-5
            return {"golden_out": dt.static_cast(attn_out.numpy(), dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=attn_tkg_packed_wrapper,
            torch_ref=torch_ref_wrapper(torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=lambda _: {"golden_out": np.zeros(attn_out_shape, "bfloat16")},
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            inference_args=TKG_INFERENCE_ARGS,
            rtol=3e-2 if fp8_kv else 1e-2,
            atol=3e-2 if fp8_kv else 1e-5,
        )


def _pack_local_sweep_schedule(local_lengths, sequence_offset, num_slots):
    """Pack one NC's tiles deterministically with at most one tile per sequence per row."""
    remaining = [math.ceil(length / FA_TILE_SIZE) for length in local_lengths]
    next_tile = [0] * len(local_lengths)
    schedule = []

    while any(remaining):
        selected = sorted(
            (local_id for local_id, count in enumerate(remaining) if count),
            key=lambda local_id: (-remaining[local_id], local_id),
        )[:num_slots]
        row = []
        for local_id in selected:
            row.append((sequence_offset + local_id, next_tile[local_id]))
            next_tile[local_id] += 1
            remaining[local_id] -= 1
        row.extend([(INVALID, 0)] * (num_slots - len(row)))
        schedule.append(row)

    expected = {
        (sequence_offset + local_id, tile_id)
        for local_id, length in enumerate(local_lengths)
        for tile_id in range(math.ceil(length / FA_TILE_SIZE))
    }
    actual = {work for row in schedule for work in row if work[0] >= 0}
    assert actual == expected
    assert all(
        len({seq_id for seq_id, _ in row if seq_id >= 0}) == sum(seq_id >= 0 for seq_id, _ in row) for row in schedule
    )
    return schedule


def _build_batch8_128k_focus_case(spec):
    """Build one fixed-row LNC2 schedule for a batch-8, 128K regression case.

    ``spec["lengths"]`` contains total sequence lengths, including one active
    token. Packed cache work covers only prior tokens, so the builder subtracts
    that token before counting FA tiles. Batch ownership is contiguous: the
    first four sequences belong to NC0 and the remaining four belong to NC1.

    ``compact_rows=False`` uses the ordinary sweep scheduler, which places at
    most one tile from a sequence in each row. ``compact_rows=True`` densely
    chunks all tile work and may place multiple tiles from one sequence in a
    row. In both modes, each NC is padded to ``rows_per_nc`` so the generated
    metadata has the exact fixed shape represented by the specification.
    """
    # S_tkg is one: the final token is active K/V, not part of the prior cache.
    cache_lengths = tuple(length - 1 for length in spec["lengths"])
    num_slots = spec["slots"]
    rows_per_nc = spec["rows_per_nc"]
    lnc = 2
    local_batch = len(cache_lengths) // lnc
    schedule = []

    # Packed rows are contiguous by NeuronCore, matching the kernel contract.
    for nc_id in range(lnc):
        sequence_offset = nc_id * local_batch
        local_lengths = cache_lengths[sequence_offset : sequence_offset + local_batch]
        if spec["compact_rows"]:
            tile_counts = tuple(math.ceil(length / FA_TILE_SIZE) for length in local_lengths)
            # Tile-major order interleaves requests while they have work. Once
            # shorter requests finish, consecutive tiles from a long request
            # fill the remaining row slots and exercise duplicate sequence IDs.
            work = [
                (sequence_offset + local_id, tile_id)
                for tile_id in range(max(tile_counts))
                for local_id, tile_count in enumerate(tile_counts)
                if tile_id < tile_count
            ]
            local_schedule = [work[start : start + num_slots] for start in range(0, len(work), num_slots)]
            for row in local_schedule:
                row.extend([(INVALID, 0)] * (num_slots - len(row)))
        else:
            # The sweep scheduler guarantees at most one tile per sequence in a row.
            local_schedule = _pack_local_sweep_schedule(local_lengths, sequence_offset, num_slots)

        # Both NCs must expose the same fixed row count. All-invalid rows model
        # unused packed capacity without contributing to attention.
        assert len(local_schedule) <= rows_per_nc
        local_schedule.extend([[(INVALID, 0)] * num_slots for _ in range(rows_per_nc - len(local_schedule))])
        schedule.extend(local_schedule)

    # Keep these tests tied to their named performance geometries.
    real_tiles = sum(math.ceil(length / FA_TILE_SIZE) for length in cache_lengths)
    assert real_tiles == spec["expected_real_tiles"]
    assert len(schedule) == lnc * rows_per_nc
    assert sum(seq_id >= 0 for row in schedule for seq_id, _ in row) == real_tiles

    return {
        "name": spec["id"],
        "lengths": list(spec["lengths"]),
        "schedule": schedule,
        "q_head": 8,
        "s_active": 1,
        "d_head": 64,
        "block_len": 64,
        "curr_sprior": 131072,
        "rows_per_nc": rows_per_nc,
        "lnc": lnc,
    }


def _build_packed_sweep_case(spec):
    """Build the mirrored LNC2 schedule for one comparison-matrix workload."""
    local_lengths = spec["local_lengths"]
    local_bs = len(local_lengths)
    nc0_schedule = _pack_local_sweep_schedule(local_lengths, 0, spec["slots"])
    nc1_schedule = _pack_local_sweep_schedule(local_lengths, local_bs, spec["slots"])
    assert len(nc0_schedule) == len(nc1_schedule)
    return {
        "name": spec["id"],
        "lengths": list(spec["lengths"]),
        "schedule": nc0_schedule + nc1_schedule,
        "q_head": 8,
        "s_active": 1,
        "d_head": 64,
        # Match the production paged-cache geometry. Smaller blocks fragment the
        # KV gather into descriptor-bound DMA and understate both baselines.
        "block_len": 64,
        "lnc": 2,
    }


def _assert_lnc2_cases_route(cases):
    """Fail collection on an LNC2 case that the packed route builder rejects.

    The kernel and the builder reject such a case only at run time, and builds run only fast
    tests, so it would otherwise first fail in the pipeline.
    """
    for case in cases:
        # The block-level sweep runs these at LNC2 whatever the case records.
        assert case["lnc"] == 2, f"{case['name']} has lnc={case['lnc']}; these lists run at LNC2"
        try:
            _build_packed_accumulator_ti_tables(
                case["schedule"],
                case["d_head"],
                case["q_head"] * case["s_active"],
                rows_per_nc=len(case["schedule"]) // 2,
                bs_per_nc=len(case["lengths"]) // 2,
            )
        except ValueError as error:
            raise AssertionError(
                f"{case['name']}: {error} (a width legal only at LNC1 belongs in LNC1_ONLY_CASE_IDS)"
            ) from error


# The LNC1-only cases are built in the same mirrored layout; the LNC1 ladders read only
# their lengths and width and repack them onto one NC.
_ALL_GENERATED_PACKED_SWEEP_CASES = tuple(_build_packed_sweep_case(spec) for spec in PACKED_SWEEP_CASES)
GENERATED_PACKED_SWEEP_CASES = tuple(
    case for case in _ALL_GENERATED_PACKED_SWEEP_CASES if case["name"] not in LNC1_ONLY_CASE_IDS
)
# Reuse the generic schedules selected above, then append the asymmetric
# fixed-row regression cases built from their dedicated specifications.
_GENERATED_PACKED_SWEEP_CASES_BY_NAME = {case["name"]: case for case in _ALL_GENERATED_PACKED_SWEEP_CASES}
_ALL_GENERATED_FP8_PACKED_AUTO_ALLOC_SEQ_PACKED_CASES = tuple(
    dict(_GENERATED_PACKED_SWEEP_CASES_BY_NAME[spec["id"]], fp8_kv=True, fp8_packed=True)
    for spec in FP8_PACKED_AUTO_ALLOC_SWEEP_SPECS
) + tuple(
    dict(_build_batch8_128k_focus_case(spec), fp8_kv=True, fp8_packed=True) for spec in GPT_OSS_BATCH8_128K_FOCUS_SPECS
)
assert len(_ALL_GENERATED_FP8_PACKED_AUTO_ALLOC_SEQ_PACKED_CASES) == (
    len(FP8_PACKED_AUTO_ALLOC_SWEEP_SPECS) + len(GPT_OSS_BATCH8_128K_FOCUS_SPECS)
)
GENERATED_FP8_PACKED_AUTO_ALLOC_SEQ_PACKED_CASES = tuple(
    case for case in _ALL_GENERATED_FP8_PACKED_AUTO_ALLOC_SEQ_PACKED_CASES if case["name"] not in LNC1_ONLY_CASE_IDS
)
GENERATED_FP8_PACKED_AUTO_ALLOC_LNC1_ONLY_CASES = tuple(
    case for case in _ALL_GENERATED_FP8_PACKED_AUTO_ALLOC_SEQ_PACKED_CASES if case["name"] in LNC1_ONLY_CASE_IDS
)
_assert_lnc2_cases_route(GENERATED_PACKED_SWEEP_CASES)
_assert_lnc2_cases_route(GENERATED_FP8_PACKED_AUTO_ALLOC_SEQ_PACKED_CASES)


@pytest_marks(["attention", "tkg"])
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
@final
class TestAttentionTkgSeqPackedSweep:
    @pytest.mark.parametrize(
        "use_auto_alloc",
        [False, True],
        ids=["manualalloc", "autoalloc"],
    )
    @pytest.mark.parametrize(
        "case",
        GENERATED_PACKED_SWEEP_CASES,
        ids=[case["name"] for case in GENERATED_PACKED_SWEEP_CASES],
    )
    def test_attn_tkg_seq_packed_sweep(self, test_manager, platform_target, case, use_auto_alloc):
        TestAttentionTkgSeqPackedKernel().test_attn_tkg_seq_packed_schedule_contract(
            test_manager, platform_target, case, use_auto_alloc=use_auto_alloc
        )


@pytest_marks(["attention", "tkg"])
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
@final
class TestAttentionTkgSeqPackedFp8PackedAutoAllocSweep:
    """FP8-packed, auto-allocation sequence-packing coverage sweep."""

    @pytest.mark.parametrize(
        "case",
        GENERATED_FP8_PACKED_AUTO_ALLOC_SEQ_PACKED_CASES,
        ids=[case["name"] for case in GENERATED_FP8_PACKED_AUTO_ALLOC_SEQ_PACKED_CASES],
    )
    def test_attn_tkg_seq_packed_fp8_packed_auto_alloc_sweep(
        self,
        test_manager,
        platform_target,
        case,
    ):
        TestAttentionTkgSeqPackedKernel().test_attn_tkg_seq_packed_schedule_contract(
            test_manager,
            platform_target,
            case,
            use_auto_alloc=True,
        )


# Three arms, so each pair varies one thing. See the class docstring.
_FP8_PACKED_AB_NAMES = (
    "util100_uniform_b16_c32k_s2",
    "util87_bimodal_b48_c32k_s4",
    "util100_uniform_b16_c32k_s8",
    "util62_bimodal_b16_c32k_s8",
)
_FP8_PACKED_AB_ARMS = (
    ("bf16", {}),
    ("fp8", {"fp8_kv": True}),
    ("fp8packed", {"fp8_kv": True, "fp8_packed": True}),
)
_FP8_PACKED_AB_BASES = [case for case in GENERATED_PACKED_SWEEP_CASES if case["name"] in _FP8_PACKED_AB_NAMES]
# Fail loudly rather than silently shrinking the matrix if a swept case is renamed.
assert len(_FP8_PACKED_AB_BASES) == len(_FP8_PACKED_AB_NAMES), (
    f"expected {len(_FP8_PACKED_AB_NAMES)} base cases, matched {sorted(case['name'] for case in _FP8_PACKED_AB_BASES)}"
)
# Two arms cannot isolate a variable; the third arm is the whole point.
assert len(_FP8_PACKED_AB_ARMS) == 3, f"expected 3 arms, got {[a for a, _ in _FP8_PACKED_AB_ARMS]}"
_FP8_PACKED_AB_PARAMS = [
    dict(case, name=f"{case['name']}_{arm}", **overrides)
    for case in _FP8_PACKED_AB_BASES
    for arm, overrides in _FP8_PACKED_AB_ARMS
]


@pytest_marks(["attention", "tkg"])
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
@final
class TestAttentionTkgSeqPackedFp8PackedAB:
    """Separate the KV dtype from the K DMA transpose on the packed path.

    ``fp8_packed`` is the K-only DMA-transpose layout, two FP8 rows parity-split into one
    bf16-width row. A BF16-versus-FP8 pair cannot attribute anything, because FP8 with
    ``fp8_packed`` off also gives up ``use_dma_transpose``, which drops
    ``qk_row_tile_factor`` from 2 to 1 and so changes the PSUM tile shape. The third arm
    restores the transpose under FP8, so fp8packed-versus-fp8 varies only the transpose and
    row tiling, and fp8packed-versus-bf16 varies only the dtype.
    """

    @pytest.mark.parametrize(
        "use_auto_alloc",
        [False, True],
        ids=["manualalloc", "autoalloc"],
    )
    # In the gate at 10.6 CPU-s median, 77.4 max against its --cpu-timeout 240; the
    # sibling sweep class is unmarked but nothing automated runs non-fast tests, and
    # compile-only alone fails these on a wrong K cache shape.
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "case",
        _FP8_PACKED_AB_PARAMS,
        ids=[case["name"] for case in _FP8_PACKED_AB_PARAMS],
    )
    def test_attn_tkg_seq_packed_fp8_packed_ab(self, test_manager, platform_target, case, use_auto_alloc):
        TestAttentionTkgSeqPackedKernel().test_attn_tkg_seq_packed_schedule_contract(
            test_manager,
            platform_target,
            case,
            use_auto_alloc=use_auto_alloc,
        )
