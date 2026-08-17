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

"""Host-side unit tests for ``rotational_topk_utils``.

These target functional branches in the pure-Python configuration and DVE
cost-model helpers of ``rotational_topk_utils.py`` — dtype-min selection, the
``reduce`` op table, cost estimation, optimal-tile-size search, the
concatenated-free-dim check, and config construction / equality. They have no
device component, so they are validated by direct function calls rather than
through the kernel execution framework, and pass identically in simulation and
on hardware.

Device (kernel-execution) tests for the top-k kernel live in ``test_topk.py``.
See ``neuron_test_output/coverage_comparison/coverage_report_core_topk.md`` for
the full branch accounting.
"""

from typing import final

import nki.language as nl
import pytest
from nkilib_src.nkilib.core.topk.rotational_topk_utils import (
    BFLOAT16_MIN,
    FLOAT32_MIN,
    _estimate_rotational_cost,
    _exceeds_concatenated_free_dim,
    _find_optimal_tile_size,
    _find_optimal_tile_size_baseline,
    _get_dtype_min,
    create_rotational_topk_config,
    create_topk_config,
    reduce,
)

from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata

# Maximum DVE free dimension (2^14). Used to construct vocab sizes that force
# specific cost-model arcs.
_MAX_FREE_DIM = 2**14


@pytest_test_metadata(name="Rotational TopK Utils")
@pytest_marks(["topk", "rotational", "coverage"])
@final
class TestTopKUtils:
    """Pure-Python unit tests for top-k configuration and cost-model helpers.

    These exercise host-side decision logic (dtype-min selection, the reduce
    op table, DVE cost estimation, optimal tile-size search, and config
    construction / equality). They contain no device component and pass
    identically in simulation and on hardware.
    """

    @pytest.mark.fast
    def test_get_dtype_min(self):
        """bfloat16 selects the bf16 pad value; other dtypes use the fp32 min."""
        assert _get_dtype_min(nl.bfloat16) == BFLOAT16_MIN
        assert _get_dtype_min(nl.float32) == FLOAT32_MIN

    @pytest.mark.fast
    def test_reduce_all_ops(self):
        """reduce dispatches each supported op correctly over a multi-element list."""
        assert reduce("mul", [2, 3, 4], 1) == 24
        assert reduce("add", [2, 3, 4], 0) == 9
        assert reduce("min", [3, 1, 2], 9) == 1
        assert reduce("max", [1, 5, 2], 0) == 5

    @pytest.mark.fast
    def test_rotational_config_eq_and_cache(self):
        """__eq__ handles wrong types and equality; cache defaults to {} when None."""
        tc = create_topk_config(inp_shape=(4, 3168), inp_dtype=nl.float32, k=256, sorted=True, num_programs=2)
        # shared_const_cache=None -> internal {} default (create_rotational_topk_config L672 True arc)
        cfg_none = create_rotational_topk_config(inp_shape=(4, 3168), topk_config=tc)
        # shared_const_cache provided -> used directly (L672 False arc)
        cfg_dict = create_rotational_topk_config(inp_shape=(4, 3168), topk_config=tc, shared_const_cache={"a": "b"})
        assert cfg_none._shared_const_cache == {}
        assert cfg_dict._shared_const_cache == {"a": "b"}
        # __eq__ non-RotationalTopkConfig -> NotImplemented; identical instance -> equal
        assert cfg_none.__eq__(123) is NotImplemented
        assert cfg_none == cfg_none

    @pytest.mark.fast
    def test_estimate_rotational_cost(self):
        """Cost estimate returns inf for infeasible n_stages; finite (sorted/unsorted) otherwise."""
        # vocab/n_stages = 20000 > 2^14 -> infeasible (L436 True arc)
        assert _estimate_rotational_cost(orig_k=256, vocab_size=20000, ns=1) == float("inf")
        sorted_cost = _estimate_rotational_cost(orig_k=256, vocab_size=4000, ns=16, sorted=True)
        unsorted_cost = _estimate_rotational_cost(orig_k=256, vocab_size=4000, ns=16, sorted=False)
        assert sorted_cost != float("inf")
        assert unsorted_cost != float("inf")
        # The sort pass adds latency (L444 True arc), so sorted must exceed unsorted (L444 False arc).
        assert sorted_cost > unsorted_cost

    @pytest.mark.fast
    def test_find_optimal_tile_size_fallback(self):
        """When no n_stages is feasible for any tile size, fall back to a derived tile size.

        vocab = 2^14 * 129 forces min_n_stages_for_hw = 129 > pmax (128), so the
        inner search never finds a valid configuration and the fallback path
        (best_tile_size is None -> L542 True arc) runs.
        """
        tile = _find_optimal_tile_size(
            orig_k=128, vocab_size=_MAX_FREE_DIM * 129, per_lnc_BxS=200, pmax=128, topk_sorted=True
        )
        assert tile >= 16

    @pytest.mark.fast
    def test_find_optimal_tile_size_baseline(self):
        """Baseline (K<=64) tile-size search: concat-exceeds skip arc and unsorted cost arc.

        - vocab=32768 -> n_stages=2, stage_free=16384, +2*8 = 16400 > 16384, so the
          concatenated free-dim exceeds the HW limit for small tiles and the
          `continue` skip (L573 True arc) fires.
        - sorted=False with padded_k == orig_k makes needs_sort False (L580 False
          arc), exercising the no-sort cost branch.
        """
        assert (
            _find_optimal_tile_size_baseline(orig_k=1, vocab_size=32768, per_lnc_BxS=200, pmax=128, topk_sorted=True)
            >= 1
        )
        assert (
            _find_optimal_tile_size_baseline(orig_k=16, vocab_size=20000, per_lnc_BxS=200, pmax=128, topk_sorted=False)
            >= 1
        )

    @pytest.mark.fast
    def test_exceeds_concatenated_free_dim(self):
        """Returns True when no stage count fits the free-dim limit; False otherwise."""
        # vocab > pmax * max_free_dim (128 * 2^14 = 2,097,152) forces
        # min_n_stages > max_n_stages, taking the early `return True` arc (L612 True).
        assert _exceeds_concatenated_free_dim(orig_k=8, vocab_size=2_097_153, tile_size=1, pmax=128) is True
        # Modest vocab fits comfortably -> False (L612 False arc).
        assert _exceeds_concatenated_free_dim(orig_k=256, vocab_size=3168, tile_size=2, pmax=128) is False
