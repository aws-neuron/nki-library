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
"""Unit tests for the trace-time predicates behind attention_cte sequence-packing tile pruning.

These are pure-Python checks: they never prune a tile that holds an in-bounds (q, k) pair, and
they decline to prune under striped CP when the layout differs per rank. They are necessary but
not sufficient; output correctness is covered by the integration tests.
"""

import numpy as np
import pytest
from nkilib_src.nkilib.core.attention.attention_cte import (
    _Q_GRP_SZ,
    AttnConfig,
    AttnTileParams,
    _has_any_compute_bounds,
    _has_any_compute_bounds_section,
    _striped_prune_is_safe,
)


def _ac(spans, seqlen):
    """AttnConfig with only the fields the pruning predicates read."""
    return AttnConfig(
        seqlen_q=seqlen,
        segment_spans=spans,
        kvp_group_size=0,
        cp_strided_q_slicing=False,
        cp_striped_input=False,
        is_prefix_caching=False,
    )


def _atp(section_len, seqlen):
    """AttnTileParams with only the fields the section predicate reads."""
    return AttnTileParams(section_len=section_len, seqlen_k_active_updated=seqlen)


def _random_layout(seqlen, lo, hi, seed):
    rng = np.random.default_rng(seed)
    cu = [0]
    while cu[-1] < seqlen:
        cu.append(min(seqlen, cu[-1] + int(rng.integers(lo, hi + 1))))
    return list(zip(cu[:-1], cu[1:], strict=True))


def _row_bounds(spans, seqlen):
    lo = np.zeros(seqlen, dtype=np.int64)
    hi = np.zeros(seqlen, dtype=np.int64)
    for s, e in spans:
        lo[s:e] = s
        hi[s:e] = e
    return lo, hi


@pytest.mark.parametrize(
    "seqlen, spans",
    [
        (4096, [(i, i + 1024) for i in range(0, 4096, 1024)]),
        (4096, [(0, 1500), (1500, 2800), (2800, 3596), (3596, 4096)]),
        (8192, _random_layout(8192, 64, 3000, seed=0)),
        (16384, _random_layout(16384, 256, 4096, seed=1)),
    ],
    ids=["uniform", "unaligned", "random_small", "random_large"],
)
def test_tile_predicate_never_prunes_live_tile(seqlen, spans):
    ac = _ac(spans, seqlen)
    lo, hi = _row_bounds(spans, seqlen)
    num_pruned = 0
    for tile_w in (512, 128):
        for grp in range((seqlen + _Q_GRP_SZ - 1) // _Q_GRP_SZ):
            qs, qe = grp * _Q_GRP_SZ, min((grp + 1) * _Q_GRP_SZ, seqlen)
            for ks in range(0, seqlen, tile_w):
                ke = min(ks + tile_w, seqlen)
                live = bool(((lo[qs:qe] < ke) & (ks < hi[qs:qe])).any())
                keep = _has_any_compute_bounds(grp, ks, ke - ks, ac)
                assert keep or not live, f"pruned a live tile: tile_w={tile_w} grp={grp} k_start={ks}"
                num_pruned += int(not keep)
    assert num_pruned > 0, "predicate pruned nothing; the descriptor would be inert"


@pytest.mark.parametrize("seqlen, section_len", [(16384, 8192), (24576, 8192)])
def test_section_predicate_never_skips_live_section(seqlen, section_len):
    spans = _random_layout(seqlen, 256, 6000, seed=2)
    ac = _ac(spans, seqlen)
    atp = _atp(section_len, seqlen)
    lo, hi = _row_bounds(spans, seqlen)
    for grp in range(seqlen // _Q_GRP_SZ):
        qs, qe = grp * _Q_GRP_SZ, (grp + 1) * _Q_GRP_SZ
        for off in range(0, seqlen, section_len):
            live = bool(((lo[qs:qe] < off + section_len) & (off < hi[qs:qe])).any())
            assert _has_any_compute_bounds_section(grp, ac, atp, off) == live
    # A section past the end of active K has no work (callers use this for "last live section").
    assert _has_any_compute_bounds_section(0, ac, atp, seqlen) is False


def test_predicates_inert_without_descriptor():
    ac = _ac(None, 4096)
    atp = _atp(8192, 4096)
    assert _has_any_compute_bounds(0, 3584, 512, ac) is True
    assert _has_any_compute_bounds_section(0, ac, atp, 0) is True


def test_striped_prune_requires_divisible_boundaries():
    ac = _ac([(0, 1024), (1024, 2048)], 256)
    ac.cp_striped_input = True
    ac.global_cp_deg = 8
    assert _striped_prune_is_safe(ac) is True
    ac.segment_spans = [(0, 1004), (1004, 2048)]  # 1004 % 8 != 0: local layout differs per rank
    assert _striped_prune_is_safe(ac) is False
    ac = _ac([(0, 1024), (1024, 2048)], 256)
    ac.cp_striped_input = True  # global_cp_deg left unset
    assert _striped_prune_is_safe(ac) is False
