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
"""Integration tests for tutorials in test/docs/neurotile/examples/05_indirect/.

Note: 06_coalesced_remainder.py kernels write only to interior tiles; OOB
regions are intentionally garbage. Their tutorial __main__ blocks use
slice-only assertions which UnitTestFramework's whole-tensor comparison
cannot replicate. Those five tests are skipped here pending a custom
comparator.
"""

import ml_dtypes
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.neurotile.examples._05_indirect import (
    _01_gather_scatter as gather_mod,
)
from nkilib_src.nkilib.experimental.neurotile.examples._05_indirect import (
    _01_gather_scatter_torch as gather_refs,
)
from nkilib_src.nkilib.experimental.neurotile.examples._05_indirect import (
    _02_dynamic_select as dynamic_mod,
)
from nkilib_src.nkilib.experimental.neurotile.examples._05_indirect import (
    _02_dynamic_select_torch as dynamic_refs,
)
from nkilib_src.nkilib.experimental.neurotile.examples._05_indirect import (
    _03_kv_cache as kv_mod,
)
from nkilib_src.nkilib.experimental.neurotile.examples._05_indirect import (
    _03_kv_cache_torch as kv_refs,
)
from nkilib_src.nkilib.experimental.neurotile.examples._05_indirect import (
    _04_subtile_indexing as subtile_mod,
)
from nkilib_src.nkilib.experimental.neurotile.examples._05_indirect import (
    _04_subtile_indexing_torch as subtile_refs,
)
from nkilib_src.nkilib.experimental.neurotile.examples._05_indirect import (
    _05_remainder_handling as remainder_mod,
)
from nkilib_src.nkilib.experimental.neurotile.examples._05_indirect import (
    _05_remainder_handling_torch as remainder_refs,
)
from nkilib_src.nkilib.experimental.neurotile.examples._05_indirect import (
    _06_coalesced_remainder as coalesced_mod,
)
from nkilib_src.nkilib.experimental.neurotile.examples._05_indirect import (
    _06_coalesced_remainder_torch as coalesced_refs,
)

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _run(test_manager, platform_target, kernel, ref, inputs, outputs, rtol=1e-2, atol=1e-2):
    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=kernel,
        torch_ref=torch_ref_wrapper(ref),
        kernel_input_generator=inputs,
        output_tensor_descriptor=outputs,
    )
    framework.run_test(
        test_config=None,
        compiler_args=CompilerArgs(platform_target=platform_target),
        rtol=rtol,
        atol=atol,
    )


def _bf16(*shape, seed=42):
    np.random.seed(seed)
    return np.random.randn(*shape).astype(ml_dtypes.bfloat16)


@pytest_marks(["neurotile"])
class TestNeurotileGatherScatter:
    """Tutorials in 01_gather_scatter.py."""

    @pytest.mark.fast
    def test_gather(self, test_manager: Orchestrator, platform_target: Platforms):
        # data: [128, 256], indices: [64, 1] -> out: [64, 256]
        N, D, K = 128, 256, 64

        def _inputs(_):
            np.random.seed(42)
            data = np.random.randn(N, D).astype(ml_dtypes.bfloat16)
            indices = np.random.permutation(N)[:K].astype(np.int32).reshape(K, 1)
            return {"data": data, "indices": indices}

        _run(
            test_manager,
            platform_target,
            gather_mod.gather_kernel,
            gather_refs.gather_torch_ref,
            _inputs,
            lambda _: {"out": np.zeros((K, D), dtype=ml_dtypes.bfloat16)},
        )

    @pytest.mark.fast
    def test_scatter(self, test_manager, platform_target):
        N, D, K = 128, 256, 64

        def _inputs(_):
            np.random.seed(43)
            source = np.random.randn(K, D).astype(ml_dtypes.bfloat16)
            indices = np.random.permutation(N)[:K].astype(np.int32).reshape(K, 1)
            return {"source": source, "indices": indices, "out_rows": N}

        _run(
            test_manager,
            platform_target,
            gather_mod.scatter_kernel,
            gather_refs.scatter_torch_ref,
            _inputs,
            lambda _: {"out": np.zeros((N, D), dtype=ml_dtypes.bfloat16)},
        )

    @pytest.mark.fast
    def test_scalar_gather_dim0(self, test_manager, platform_target):
        N, D = 128, 64
        _run(
            test_manager,
            platform_target,
            gather_mod.scalar_gather_dim0,
            gather_refs.scalar_gather_dim0_torch_ref,
            lambda _: {"data": _bf16(N, D, seed=46), "row_idx_tensor": np.array([[42]], dtype=np.int32)},
            lambda _: {"out": np.zeros((1, D), dtype=ml_dtypes.bfloat16)},
        )

    @pytest.mark.fast
    def test_scalar_gather_dim1(self, test_manager, platform_target):
        N, D = 128, 64
        _run(
            test_manager,
            platform_target,
            gather_mod.scalar_gather_dim1,
            gather_refs.scalar_gather_dim1_torch_ref,
            lambda _: {"data": _bf16(N, D, seed=47), "col_idx_tensor": np.array([[17]], dtype=np.int32)},
            lambda _: {"out": np.zeros((N, 1), dtype=ml_dtypes.bfloat16)},
        )


@pytest_marks(["neurotile"])
class TestNeurotileDynamicSelect:
    """Tutorial in 02_dynamic_select.py."""

    @pytest.mark.fast
    def test_expert_select(self, test_manager, platform_target):
        E, P, F = 8, 128, 64
        _run(
            test_manager,
            platform_target,
            dynamic_mod.expert_select,
            dynamic_refs.expert_select_torch_ref,
            lambda _: {"weights": _bf16(E, P, F, seed=50), "expert_id_tensor": np.array([[5]], dtype=np.int32)},
            lambda _: {"out": np.zeros((P, F), dtype=ml_dtypes.bfloat16)},
        )


@pytest_marks(["neurotile"])
class TestNeurotileKVCache:
    """Tutorials in 03_kv_cache.py."""

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "kernel,ref,seed,batch_id,seq",
        [
            (kv_mod.kv_cache_load, kv_refs.kv_cache_load_torch_ref, 60, 2, 42),
            (kv_mod.kv_cache_load_raw_sbuf_index, kv_refs.kv_cache_load_raw_sbuf_index_torch_ref, 62, 3, 77),
        ],
    )
    def test_single_position(self, test_manager, platform_target, kernel, ref, seed, batch_id, seq):
        B, S, D = 4, 128, 64
        _run(
            test_manager,
            platform_target,
            kernel,
            ref,
            lambda _: {
                "kv_cache": _bf16(B, S, D, seed=seed),
                "batch_id": batch_id,
                "seq_offset_tensor": np.array([[seq]], dtype=np.int32),
            },
            lambda _: {"out": np.zeros((1, D), dtype=ml_dtypes.bfloat16)},
        )

    @pytest.mark.fast
    def test_multi_position(self, test_manager, platform_target):
        B, S, D = 8, 256, 64
        _run(
            test_manager,
            platform_target,
            kv_mod.kv_cache_multi_pos,
            kv_refs.kv_cache_multi_pos_torch_ref,
            lambda _: {
                "kv_cache": _bf16(B, S, D, seed=61),
                "batch_indices": np.array([[1], [5]], dtype=np.int32),
                "seq_positions": np.array([[100], [150]], dtype=np.int32),
            },
            lambda _: {"out": np.zeros((2, D), dtype=ml_dtypes.bfloat16)},
        )


@pytest_marks(["neurotile"])
class TestNeurotileSubtileIndexing:
    """Tutorials in 04_subtile_indexing.py."""

    @pytest.mark.fast
    def test_static_view_select(self, test_manager, platform_target):
        D0, P, F = 4, 128, 64
        _run(
            test_manager,
            platform_target,
            subtile_mod.static_view_select,
            subtile_refs.static_view_select_torch_ref,
            lambda _: {"data": _bf16(D0, P, F, seed=200), "select_idx": 2},
            lambda _: {"out": np.zeros((P, F), dtype=ml_dtypes.bfloat16)},
        )

    @pytest.mark.fast
    def test_chained_view_select(self, test_manager, platform_target):
        D0, P, F = 4, 128, 64
        _run(
            test_manager,
            platform_target,
            subtile_mod.chained_view_select,
            subtile_refs.chained_view_select_torch_ref,
            lambda _: {"data": _bf16(D0, P, F, seed=201), "slab_idx": 1, "row_idx": 42},
            lambda _: {"out": np.zeros((1, F), dtype=ml_dtypes.bfloat16)},
        )

    @pytest.mark.fast
    def test_loop_view_select(self, test_manager, platform_target):
        E, P, F = 4, 128, 64
        _run(
            test_manager,
            platform_target,
            subtile_mod.loop_view_select,
            subtile_refs.loop_view_select_torch_ref,
            lambda _: {"data": _bf16(E, P, F, seed=202)},
            lambda _: {"out": np.zeros((E, P, F), dtype=ml_dtypes.bfloat16)},
        )

    @pytest.mark.fast
    def test_tile_row_extract(self, test_manager, platform_target):
        P, F = 128, 64
        _run(
            test_manager,
            platform_target,
            subtile_mod.tile_row_extract,
            subtile_refs.tile_row_extract_torch_ref,
            lambda _: {"data": _bf16(P, F, seed=203), "row_idx": 5},
            lambda _: {"out": np.zeros((1, F), dtype=ml_dtypes.bfloat16)},
        )

    @pytest.mark.fast
    def test_tile_element_extract(self, test_manager, platform_target):
        P, F = 128, 64
        _run(
            test_manager,
            platform_target,
            subtile_mod.tile_element_extract,
            subtile_refs.tile_element_extract_torch_ref,
            lambda _: {"data": _bf16(P, F, seed=205), "row_idx": 42, "col_idx": 17},
            lambda _: {"out": np.zeros((1, 1), dtype=ml_dtypes.bfloat16)},
        )

    @pytest.mark.fast
    def test_tile_subblock_extract(self, test_manager, platform_target):
        P, F = 128, 64
        _run(
            test_manager,
            platform_target,
            subtile_mod.tile_subblock_extract,
            subtile_refs.tile_subblock_extract_torch_ref,
            lambda _: {
                "data": _bf16(P, F, seed=207),
                "row_start": 8,
                "row_count": 16,
                "col_start": 4,
                "col_count": 32,
            },
            lambda _: {"out": np.zeros((16, 32), dtype=ml_dtypes.bfloat16)},
        )

    @pytest.mark.fast
    def test_dynamic_select_row_extract(self, test_manager, platform_target):
        E, P, F = 8, 128, 64
        _run(
            test_manager,
            platform_target,
            subtile_mod.dynamic_select_row_extract,
            subtile_refs.dynamic_select_row_extract_torch_ref,
            lambda _: {
                "weights": _bf16(E, P, F, seed=204),
                "expert_id_tensor": np.array([[5]], dtype=np.int32),
                "row_idx": 42,
            },
            lambda _: {"out": np.zeros((1, F), dtype=ml_dtypes.bfloat16)},
        )


@pytest_marks(["neurotile"])
class TestNeurotileRemainderHandling:
    """Tutorials in 05_remainder_handling.py."""

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "kernel,ref,seed,K",
        [
            (remainder_mod.indirect_gather_oob, remainder_refs.indirect_gather_oob_torch_ref, 42, 16),
            (remainder_mod.is_remainder_guard, remainder_refs.is_remainder_guard_torch_ref, 45, 16),
        ],
    )
    def test_indirect_gather_arange(self, test_manager, platform_target, kernel, ref, seed, K):
        N, D = 128, 64

        def _inputs(_):
            np.random.seed(seed)
            data = np.random.randn(N, D).astype(ml_dtypes.bfloat16)
            indices = np.arange(K, dtype=np.int32).reshape(K, 1)
            return {"data": data, "indices": indices}

        _run(
            test_manager,
            platform_target,
            kernel,
            ref,
            _inputs,
            lambda _: {"out": np.zeros((K, D), dtype=ml_dtypes.bfloat16)},
        )

    @pytest.mark.fast
    def test_indirect_gather_oob_value(self, test_manager, platform_target):
        N, D, K = 128, 64, 16

        def _inputs(_):
            np.random.seed(43)
            data = np.random.randn(N, D).astype(ml_dtypes.bfloat16)
            indices = np.arange(0, K * 5, 5, dtype=np.int32).reshape(K, 1)
            return {"data": data, "indices": indices}

        _run(
            test_manager,
            platform_target,
            remainder_mod.indirect_gather_oob_value,
            remainder_refs.indirect_gather_oob_value_torch_ref,
            _inputs,
            lambda _: {"out": np.zeros((K, D), dtype=ml_dtypes.bfloat16)},
        )

    @pytest.mark.fast
    def test_indirect_scatter_oob(self, test_manager, platform_target):
        N, D, K = 128, 64, 8

        def _inputs(_):
            np.random.seed(44)
            source = np.random.randn(K, D).astype(ml_dtypes.bfloat16)
            indices = np.array([3, 0, 7, 2, 100, 50, 33, 120], dtype=np.int32).reshape(K, 1)
            return {"source": source, "indices": indices, "out_rows": N}

        _run(
            test_manager,
            platform_target,
            remainder_mod.indirect_scatter_oob,
            remainder_refs.indirect_scatter_oob_torch_ref,
            _inputs,
            lambda _: {"out": np.zeros((N, D), dtype=ml_dtypes.bfloat16)},
        )

    @pytest.mark.fast
    def test_multi_tile_gather_oob(self, test_manager, platform_target):
        N, D, K = 128, 64, 64

        def _inputs(_):
            np.random.seed(46)
            data = np.random.randn(N, D).astype(ml_dtypes.bfloat16)
            indices = np.random.permutation(N)[:K].astype(np.int32).reshape(K, 1)
            return {"data": data, "indices": indices}

        _run(
            test_manager,
            platform_target,
            remainder_mod.multi_tile_gather_oob,
            remainder_refs.multi_tile_gather_oob_torch_ref,
            _inputs,
            lambda _: {"out": np.zeros((K, D), dtype=ml_dtypes.bfloat16)},
        )


def _coalesced_inputs(shape, seed):
    """bf16 inputs derived from fp32 randn — matches the tutorial's _make_src pattern."""

    def _gen(_):
        np.random.seed(seed)
        src_np = np.random.randn(*shape).astype(np.float32)
        return {"src": src_np.astype(ml_dtypes.bfloat16)}

    return _gen


def _coalesced_outputs(kernel_input):
    return {"out": np.zeros_like(kernel_input["src"])}


@pytest_marks(["neurotile"])
class TestNeurotileCoalescedRemainder:
    """Tutorials in 06_coalesced_remainder.py.

    Each kernel writes to a specific interior slice; OOB regions stay at the
    HBM allocation's zero-init value. The torch refs zero-pad accordingly so
    UnitTestFramework's whole-tensor comparison matches.
    """

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "kernel,ref,shape,seed",
        [
            (coalesced_mod.row_hoist_f_remainder, coalesced_refs.row_hoist_f_remainder_torch_ref, (300, 500), 42),
            (coalesced_mod.col_hoist_p_remainder, coalesced_refs.col_hoist_p_remainder_torch_ref, (300, 512), 43),
            (coalesced_mod.range_slice_remainder, coalesced_refs.range_slice_remainder_torch_ref, (300, 500), 44),
            (coalesced_mod.store_oob_mode, coalesced_refs.store_oob_mode_torch_ref, (128, 500), 46),
            (coalesced_mod.multi_range_interior, coalesced_refs.multi_range_interior_torch_ref, (300, 500), 51),
        ],
    )
    def test_coalesced_remainder(self, test_manager, platform_target, kernel, ref, shape, seed):
        _run(
            test_manager,
            platform_target,
            kernel,
            ref,
            _coalesced_inputs(shape, seed),
            _coalesced_outputs,
            rtol=0,
            atol=0,
        )
