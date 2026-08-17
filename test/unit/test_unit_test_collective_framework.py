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

"""Unit tests for CollectiveUnitTestFramework and SimDistRunner multi-pass logic."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
from nki.collectives import ReplicaGroup
from nkilib_src.nkilib.experimental.collectives.distributed_adapter import get_pg, get_rank

from test.utils.common_dataclasses import (
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    InferenceArgs,
    PerRankLazyGoldenGenerator,
    PerRankLazyInputGenerator,
    Platforms,
)
from test.utils.unit_test_collective_framework import (
    CollectiveUnitTestFramework,
    SimDistRunner,
    _MPSimProcessGroup,
    _run_torch_refs_parallel,
)


class TestCollectiveUnitTestFramework:
    """Tests for CollectiveUnitTestFramework class."""

    def test_init_validates_signature(self):
        """Should validate kernel_entry <-> torch_ref signature on init."""

        def kernel(a, b):
            pass

        def torch_ref(a, c):
            pass

        with pytest.raises(ValueError, match="Missing in torch_ref"):
            CollectiveUnitTestFramework(
                test_manager=MagicMock(),
                kernel_entry=kernel,
                torch_ref=torch_ref,
                per_rank_input_generator=lambda rank_id: {},
                collective_ranks=2,
            )

    def test_init_check_unused_params(self):
        """Should check unused params when enabled."""

        def kernel(a, unused):
            return a

        def torch_ref(a, unused):
            return a

        with pytest.raises(ValueError, match="unused.*may be unused"):
            CollectiveUnitTestFramework(
                test_manager=MagicMock(),
                kernel_entry=kernel,
                torch_ref=torch_ref,
                per_rank_input_generator=lambda rank_id: {},
                collective_ranks=2,
                check_unused_params=True,
            )

    def test_run_test_validates_input_keys(self):
        """Should validate rank-0 input keys against kernel_entry."""

        def kernel(a):
            return a

        def torch_ref(a):
            pass

        framework = CollectiveUnitTestFramework(
            test_manager=MagicMock(),
            kernel_entry=kernel,
            torch_ref=torch_ref,
            per_rank_input_generator=lambda rank_id: {"a": 1, "extra": 2},
            collective_ranks=2,
        )

        with pytest.raises(ValueError, match="extra.*don't match"):
            framework.run_test(test_config=None, compiler_args=MagicMock(), output_keys=["out"])

    def test_run_test_validates_cross_rank_consistency(self):
        """Should check cross-rank shape consistency for collective_ranks >= 2."""

        def kernel(a):
            return a

        def torch_ref(a):
            pass

        def input_gen(rank_id):
            shape = (2, 3) if rank_id == 0 else (4, 3)
            return {"a": np.zeros(shape)}

        framework = CollectiveUnitTestFramework(
            test_manager=MagicMock(),
            kernel_entry=kernel,
            torch_ref=torch_ref,
            per_rank_input_generator=input_gen,
            collective_ranks=2,
        )

        with pytest.raises(ValueError, match="shape mismatch"):
            framework.run_test(test_config=None, compiler_args=MagicMock(), output_keys=["out"])

    def test_run_test_skips_cross_rank_check_single_rank(self):
        """Should skip cross-rank check when collective_ranks < 2."""

        def kernel(a):
            return a

        def torch_ref(a):
            return {"out": a}

        call_count = 0

        def input_gen(rank_id):
            nonlocal call_count
            call_count += 1
            return {"a": np.array([1.0])}

        mock_manager = MagicMock()
        framework = CollectiveUnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref,
            per_rank_input_generator=input_gen,
            collective_ranks=1,
        )
        framework.run_test(test_config=None, compiler_args=MagicMock(), output_keys=["out"])
        # With single rank, input_gen is called for validation + golden generation, but NOT for cross-rank check
        # (cross-rank check requires collective_ranks >= 2)
        mock_manager.execute.assert_called_once()

    def test_run_test_executes_with_correct_args(self):
        """Should pass correct KernelArgs to test_manager.execute."""

        def kernel(a):
            return a

        def torch_ref(a):
            return {"out": a}

        def input_gen(rank_id):
            return {"a": np.array([rank_id], dtype=np.float32)}

        mock_manager = MagicMock()
        compiler_args = CompilerArgs(logical_nc_config=2, platform_target=Platforms.TRN2)

        framework = CollectiveUnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref,
            per_rank_input_generator=input_gen,
            collective_ranks=4,
        )
        framework.run_test(test_config=None, compiler_args=compiler_args, output_keys=["out"], rtol=0.01, atol=0.02)

        mock_manager.execute.assert_called_once()
        kernel_args = mock_manager.execute.call_args[0][0]
        assert kernel_args.kernel_func is kernel
        assert kernel_args.compiler_input is compiler_args
        assert isinstance(kernel_args.kernel_input, PerRankLazyInputGenerator)
        assert kernel_args.inference_args.collective_ranks == 4
        assert isinstance(kernel_args.validation_args.golden_output, PerRankLazyGoldenGenerator)
        assert kernel_args.validation_args.relative_accuracy == 0.01
        assert kernel_args.validation_args.absolute_accuracy == 0.02

    def test_run_test_sets_base_input_on_generator(self):
        """PerRankLazyInputGenerator passed to execute should have .base_input set."""

        def kernel(a):
            return a

        def torch_ref(a):
            return {"out": a}

        rank0 = {"a": np.array([0.0])}

        mock_manager = MagicMock()
        framework = CollectiveUnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref,
            per_rank_input_generator=lambda rank_id: {"a": np.array([float(rank_id)])},
            collective_ranks=2,
        )
        framework.run_test(test_config=None, compiler_args=MagicMock(), output_keys=["out"])

        kernel_args = mock_manager.execute.call_args[0][0]
        assert hasattr(kernel_args.kernel_input, "base_input")
        np.testing.assert_array_equal(kernel_args.kernel_input.base_input["a"], rank0["a"])

    def test_run_test_filters_must_alias_input(self):
        """Should filter .must_alias_input keys through to kernel input."""

        def kernel(a, output):
            return a, output

        def torch_ref(a, output):
            return {"out": a}

        def input_gen(rank_id):
            return {"a": np.array([1.0]), "output.must_alias_input": np.array([0.0])}

        mock_manager = MagicMock()
        framework = CollectiveUnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref,
            per_rank_input_generator=input_gen,
            collective_ranks=2,
        )
        framework.run_test(test_config=None, compiler_args=MagicMock(), output_keys=["out"])
        mock_manager.execute.assert_called_once()

        # Verify the filtered input generator works correctly
        kernel_args = mock_manager.execute.call_args[0][0]
        filtered = kernel_args.kernel_input.for_rank(0)
        assert "a" in filtered
        assert "output.must_alias_input" in filtered

    def test_run_test_golden_via_torch_ref(self):
        """Default golden should be generated via torch_ref per rank."""

        def kernel(a):
            return a

        def torch_ref(a):
            return {"out": a * 2}

        def input_gen(rank_id):
            return {"a": np.array([float(rank_id + 1)])}

        mock_manager = MagicMock()
        framework = CollectiveUnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref,
            per_rank_input_generator=input_gen,
            collective_ranks=2,
        )
        framework.run_test(test_config=None, compiler_args=MagicMock(), output_keys=["out"])

        kernel_args = mock_manager.execute.call_args[0][0]
        golden_gen = kernel_args.validation_args.golden_output
        # Rank 0: a=1.0 -> out=2.0
        np.testing.assert_array_equal(golden_gen.for_rank(0)["out"], [2.0])
        # Rank 1: a=2.0 -> out=4.0
        np.testing.assert_array_equal(golden_gen.for_rank(1)["out"], [4.0])

    def test_run_test_custom_inference_args(self):
        """inference_args should override default collective_ranks."""

        def kernel(a):
            return a

        def torch_ref(a):
            return {"out": a}

        custom_inference = InferenceArgs(collective_ranks=8)
        mock_manager = MagicMock()
        framework = CollectiveUnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref,
            per_rank_input_generator=lambda rank_id: {"a": np.array([1.0])},
            collective_ranks=2,
        )
        framework.run_test(
            test_config=None, compiler_args=MagicMock(), output_keys=["out"], inference_args=custom_inference
        )

        kernel_args = mock_manager.execute.call_args[0][0]
        assert kernel_args.inference_args is custom_inference

    def test_run_test_default_inference_args(self):
        """Without inference_args, should use collective_ranks from init."""

        def kernel(a):
            return a

        def torch_ref(a):
            return {"out": a}

        mock_manager = MagicMock()
        framework = CollectiveUnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref,
            per_rank_input_generator=lambda rank_id: {"a": np.array([1.0])},
            collective_ranks=4,
        )
        framework.run_test(test_config=None, compiler_args=MagicMock(), output_keys=["out"])

        kernel_args = mock_manager.execute.call_args[0][0]
        assert kernel_args.inference_args.collective_ranks == 4

    def test_run_test_collector_with_metadata(self):
        """Should call collector when metadata is provided."""

        def kernel(a):
            return a

        def torch_ref(a):
            return {"out": a}

        mock_collector = MagicMock()
        mock_manager = MagicMock()
        metadata_list = [{"test_settings": {"k": 1}}]

        framework = CollectiveUnitTestFramework(
            test_manager=mock_manager,
            collector=mock_collector,
            kernel_entry=kernel,
            torch_ref=torch_ref,
            per_rank_input_generator=lambda rank_id: {"a": np.array([1.0])},
            collective_ranks=2,
        )

        with patch("test.utils.unit_test_framework.load_model_configs", return_value=metadata_list):
            framework.run_test(
                test_config=None,
                compiler_args=MagicMock(),
                output_keys=["out"],
                metadata={"config_name": "test_cfg", "key": {"k": 1}},
            )
        mock_collector.match_and_add_metadata_dimensions.assert_called_once()

    def test_run_test_custom_comparator(self):
        """custom_comparator should receive torch_ref golden and rank_id."""

        def kernel(a):
            return a

        def torch_ref(a):
            return {"out": a * 3}

        received = {}

        def comparator(rank_id, golden_dict):
            received[rank_id] = golden_dict["out"].copy()
            return {
                "out": CustomValidatorWithOutputTensorData(
                    validator=type("V", (CustomValidator,), {"validate": lambda self, x: True}),
                    output_ndarray=golden_dict["out"],
                )
            }

        mock_manager = MagicMock()
        framework = CollectiveUnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref,
            per_rank_input_generator=lambda rank_id: {"a": np.array([float(rank_id + 1)])},
            collective_ranks=2,
        )
        framework.run_test(
            test_config=None, compiler_args=MagicMock(), output_keys=["out"], custom_comparator=comparator
        )

        kernel_args = mock_manager.execute.call_args[0][0]
        golden_gen = kernel_args.validation_args.golden_output
        result_r0 = golden_gen.for_rank(0)
        result_r1 = golden_gen.for_rank(1)
        # torch_ref(a=1.0) -> out=3.0, torch_ref(a=2.0) -> out=6.0
        np.testing.assert_array_equal(received[0], [3.0])
        np.testing.assert_array_equal(received[1], [6.0])
        assert isinstance(result_r0["out"], CustomValidatorWithOutputTensorData)
        assert isinstance(result_r1["out"], CustomValidatorWithOutputTensorData)


class TestRunTorchRefsParallel:
    """Tests for _run_torch_refs_parallel."""

    def test_all_reduce_via_fake_pg(self):
        """Torch refs using dist.all_reduce should produce correct results via SimProcessGroup."""

        def torch_ref(x, rg):
            t = torch.from_numpy(x.copy())
            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=get_pg(rg))
            return {"out": t.numpy()}

        rg = ReplicaGroup([[0, 1]])
        inputs = {
            0: {"x": np.array([1.0, 2.0], dtype=np.float32), "rg": rg},
            1: {"x": np.array([3.0, 4.0], dtype=np.float32), "rg": rg},
        }
        results = _run_torch_refs_parallel(torch_ref, inputs, num_ranks=2)
        np.testing.assert_array_equal(results[0]["out"], [4.0, 6.0])
        np.testing.assert_array_equal(results[1]["out"], [4.0, 6.0])

    def test_all_gather_via_fake_pg(self):
        """Torch refs using dist.all_gather should concatenate across ranks."""

        def torch_ref(x, rg):
            t = torch.from_numpy(x.copy())
            gathered = [torch.zeros_like(t) for _ in range(2)]
            dist.all_gather(gathered, t, group=get_pg(rg))
            return {"out": torch.cat(gathered, dim=0).numpy()}

        rg = ReplicaGroup([[0, 1]])
        inputs = {
            0: {"x": np.array([1.0], dtype=np.float32), "rg": rg},
            1: {"x": np.array([2.0], dtype=np.float32), "rg": rg},
        }
        results = _run_torch_refs_parallel(torch_ref, inputs, num_ranks=2)
        np.testing.assert_array_equal(results[0]["out"], [1.0, 2.0])
        np.testing.assert_array_equal(results[1]["out"], [1.0, 2.0])

    def test_rank_id_via_get_rank(self):
        """Torch refs using get_rank() should get correct per-rank values."""

        def torch_ref(data):
            return {"out": data[get_rank()]}

        data = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        inputs = {0: {"data": data}, 1: {"data": data}}
        results = _run_torch_refs_parallel(torch_ref, inputs, num_ranks=2)
        np.testing.assert_array_equal(results[0]["out"], [10.0, 20.0])
        np.testing.assert_array_equal(results[1]["out"], [30.0, 40.0])

    def test_replica_groups_respected(self):
        """Sub-groups should get separate SimProcessGroup instances."""

        def torch_ref(x, replica_group):
            t = torch.from_numpy(x.copy())
            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=get_pg(replica_group))
            return {"out": t.numpy()}

        rg = ReplicaGroup([[0, 1], [2, 3]])
        inputs = {r: {"x": np.array([float(r + 1)], dtype=np.float32), "replica_group": rg} for r in range(4)}
        results = _run_torch_refs_parallel(torch_ref, inputs, num_ranks=4)
        # Group [0,1]: 1+2=3, Group [2,3]: 3+4=7
        np.testing.assert_array_equal(results[0]["out"], [3.0])
        np.testing.assert_array_equal(results[1]["out"], [3.0])
        np.testing.assert_array_equal(results[2]["out"], [7.0])
        np.testing.assert_array_equal(results[3]["out"], [7.0])

    def test_error_propagation(self):
        """Errors in torch_ref threads should be raised."""

        def torch_ref(x):
            raise ValueError("test error")

        inputs = {0: {"x": np.array([1.0])}}
        with pytest.raises(RuntimeError, match="test error"):
            _run_torch_refs_parallel(torch_ref, inputs, num_ranks=1)

    def test_collective_framework_uses_parallel_golden(self):
        """CollectiveUnitTestFramework should use parallel golden generation."""

        rg = ReplicaGroup([[0, 1]])

        def kernel(x, replica_group):
            return x, replica_group

        def torch_ref(x, replica_group):
            t = torch.from_numpy(x.copy())
            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=get_pg(replica_group))
            return {"out": t.numpy()}

        def input_gen(rank_id):
            return {"x": np.array([float(rank_id + 1)], dtype=np.float32), "replica_group": rg}

        mock_manager = MagicMock()
        framework = CollectiveUnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref,
            per_rank_input_generator=input_gen,
            collective_ranks=2,
        )
        framework.run_test(test_config=None, compiler_args=MagicMock(), output_keys=["out"])

        kernel_args = mock_manager.execute.call_args[0][0]
        golden_gen = kernel_args.validation_args.golden_output
        # all_reduce sum: rank0=1+2=3, rank1=1+2=3
        np.testing.assert_array_equal(golden_gen.for_rank(0)["out"], [3.0])
        np.testing.assert_array_equal(golden_gen.for_rank(1)["out"], [3.0])

    def test_golden_gen_is_lazy(self):
        """Golden generation should not run until for_rank() is called (compile-only safe)."""
        rg = ReplicaGroup([[0, 1]])

        def kernel(x, replica_group):
            return x, replica_group

        def torch_ref(x, replica_group):
            return {"out": x.copy()}

        def input_gen(rank_id):
            return {"x": np.array([float(rank_id)], dtype=np.float32), "replica_group": rg}

        mock_manager = MagicMock()
        framework = CollectiveUnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref,
            per_rank_input_generator=input_gen,
            collective_ranks=2,
        )

        # Patch _run_torch_refs_parallel to track if it's called
        with patch("test.utils.unit_test_collective_framework._run_torch_refs_parallel") as mock_run:
            mock_run.return_value = {0: {"out": np.array([0.0])}, 1: {"out": np.array([1.0])}}
            framework.run_test(test_config=None, compiler_args=MagicMock(), output_keys=["out"])

            # Should NOT have been called during run_test (lazy)
            mock_run.assert_not_called()

            # Only when golden is requested
            kernel_args = mock_manager.execute.call_args[0][0]
            golden_gen = kernel_args.validation_args.golden_output
            golden_gen.for_rank(0)
            mock_run.assert_called_once()

    def test_compile_only_skips_golden_gen(self):
        """In compile-only mode with output_keys, golden gen is completely skipped."""
        from test.utils.common_dataclasses import TraceMode

        rg = ReplicaGroup([[0, 1]])

        def kernel(x, replica_group):
            return x, replica_group

        def torch_ref(x, replica_group):
            return {"out": x.copy()}

        def input_gen(rank_id):
            return {"x": np.array([float(rank_id)], dtype=np.float32), "replica_group": rg}

        mock_manager = MagicMock()
        mock_manager.trace_mode = TraceMode.CompileOnly
        framework = CollectiveUnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref,
            per_rank_input_generator=input_gen,
            collective_ranks=2,
        )

        with patch("test.utils.unit_test_collective_framework._run_torch_refs_parallel") as mock_run:
            framework.run_test(test_config=None, compiler_args=MagicMock(), output_keys=["out"])

            # Golden gen should never be called in compile-only mode
            mock_run.assert_not_called()

            # validation_args should be None in compile-only mode (no golden gen needed)
            kernel_args = mock_manager.execute.call_args[0][0]
            assert kernel_args.validation_args is None

    def test_multiple_replica_groups(self):
        """Kernels with multiple ReplicaGroups should get separate SimProcessGroups per group."""

        def torch_ref(x, rg_reduce, rg_gather):
            t = torch.from_numpy(x.copy())
            # all_reduce within rg_reduce sub-group
            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=get_pg(rg_reduce))
            # all_gather across rg_gather sub-group
            gathered = [torch.zeros_like(t) for _ in range(2)]
            dist.all_gather(gathered, t, group=get_pg(rg_gather))
            return {"out": torch.cat(gathered, dim=0).numpy()}

        # 4 ranks: reduce within [0,1] and [2,3], then gather across [0,2] and [1,3]
        rg_reduce = ReplicaGroup([[0, 1], [2, 3]])
        rg_gather = ReplicaGroup([[0, 2], [1, 3]])
        inputs = {
            r: {"x": np.array([float(r + 1)], dtype=np.float32), "rg_reduce": rg_reduce, "rg_gather": rg_gather}
            for r in range(4)
        }
        results = _run_torch_refs_parallel(torch_ref, inputs, num_ranks=4)
        # reduce: [0,1]->1+2=3, [2,3]->3+4=7
        # gather [0,2]: rank0 has 3, rank2 has 7 -> [3,7]
        # gather [1,3]: rank1 has 3, rank3 has 7 -> [3,7]
        for r in range(4):
            np.testing.assert_array_equal(results[r]["out"], [3.0, 7.0])


class TestSimDistRunnerMultiPass:
    """Test multi-pass collective discovery and sequential final pass."""

    def test_single_collective_two_ranks(self):
        """Two ranks do one all_reduce, results are correct."""
        rg = ReplicaGroup([[0, 1]])

        def torch_ref(x):
            pg = get_pg(rg)
            out = x.clone()
            dist.all_reduce(out, group=pg)
            return {"out": out}

        inputs = {0: {"x": torch.tensor([1.0, 2.0])}, 1: {"x": torch.tensor([3.0, 4.0])}}
        runner = SimDistRunner(num_ranks=2, replica_groups=[rg])
        results = runner.run(torch_ref, inputs)
        assert torch.allclose(results[0]["out"], torch.tensor([4.0, 6.0]))
        assert torch.allclose(results[1]["out"], torch.tensor([4.0, 6.0]))

    def test_two_collectives(self):
        """Torch ref with two sequential collectives — both discovered and replayed."""
        rg = ReplicaGroup([[0, 1]])

        def torch_ref(x):
            pg = get_pg(rg)
            # First collective: all_reduce
            out = x.clone()
            dist.all_reduce(out, group=pg)
            # Second collective: all_reduce again
            dist.all_reduce(out, group=pg)
            return {"out": out}

        inputs = {0: {"x": torch.tensor([1.0])}, 1: {"x": torch.tensor([2.0])}}
        runner = SimDistRunner(num_ranks=2, replica_groups=[rg])
        results = runner.run(torch_ref, inputs)
        # First all_reduce: [1]+[2]=3, [2]+[1]=3 → both have [3]
        # Second all_reduce: [3]+[3]=6 → both have [6]
        assert torch.allclose(results[0]["out"], torch.tensor([6.0]))
        assert torch.allclose(results[1]["out"], torch.tensor([6.0]))

    def test_no_collective(self):
        """Torch ref with no collectives — probe discovers immediately, runs final pass."""
        rg = ReplicaGroup([[0, 1]])

        def torch_ref(x):
            return {"out": x * 2}

        inputs = {0: {"x": torch.tensor([1.0])}, 1: {"x": torch.tensor([3.0])}}
        runner = SimDistRunner(num_ranks=2, replica_groups=[rg])
        results = runner.run(torch_ref, inputs)
        assert torch.allclose(results[0]["out"], torch.tensor([2.0]))
        assert torch.allclose(results[1]["out"], torch.tensor([6.0]))

    def test_four_ranks_allgather(self):
        """Four ranks do all_gather — verifies multi-rank correctness."""
        rg = ReplicaGroup([[0, 1, 2, 3]])

        def torch_ref(x):
            pg = get_pg(rg)
            gathered = [torch.zeros_like(x) for _ in range(4)]
            dist.all_gather(gathered, x, group=pg)
            return {"out": torch.cat(gathered)}

        inputs = {r: {"x": torch.tensor([float(r)])} for r in range(4)}
        runner = SimDistRunner(num_ranks=4, replica_groups=[rg])
        results = runner.run(torch_ref, inputs)
        expected = torch.tensor([0.0, 1.0, 2.0, 3.0])
        for r in range(4):
            assert torch.allclose(results[r]["out"], expected)

    def test_computation_between_collectives(self):
        """Heavy computation between collectives — verifies replay correctness."""
        rg = ReplicaGroup([[0, 1]])

        def torch_ref(x):
            pg = get_pg(rg)
            # First collective
            out = x.clone()
            dist.all_reduce(out, group=pg)
            # Computation between collectives
            out = out * 10 + get_rank()
            # Second collective
            dist.all_reduce(out, group=pg)
            return {"out": out}

        inputs = {0: {"x": torch.tensor([1.0])}, 1: {"x": torch.tensor([2.0])}}
        runner = SimDistRunner(num_ranks=2, replica_groups=[rg])
        results = runner.run(torch_ref, inputs)
        # After first all_reduce: both have [3]
        # rank 0: 3*10+0=30, rank 1: 3*10+1=31
        # After second all_reduce: both have [30+31=61]
        assert torch.allclose(results[0]["out"], torch.tensor([61.0]))
        assert torch.allclose(results[1]["out"], torch.tensor([61.0]))

    def test_multi_group_collectives(self):
        """Two replica groups — each group operates independently."""
        rg1 = ReplicaGroup([[0, 1]])
        rg2 = ReplicaGroup([[2, 3]])

        def torch_ref(x):
            rank = get_rank()
            if rank < 2:
                pg = get_pg(rg1)
            else:
                pg = get_pg(rg2)
            out = x.clone()
            dist.all_reduce(out, group=pg)
            return {"out": out}

        inputs = {
            0: {"x": torch.tensor([1.0])},
            1: {"x": torch.tensor([2.0])},
            2: {"x": torch.tensor([10.0])},
            3: {"x": torch.tensor([20.0])},
        }
        runner = SimDistRunner(num_ranks=4, replica_groups=[rg1, rg2])
        results = runner.run(torch_ref, inputs)
        assert torch.allclose(results[0]["out"], torch.tensor([3.0]))
        assert torch.allclose(results[1]["out"], torch.tensor([3.0]))
        assert torch.allclose(results[2]["out"], torch.tensor([30.0]))
        assert torch.allclose(results[3]["out"], torch.tensor([30.0]))

    def test_skip_redundant_writes(self):
        """Files written in earlier passes are not overwritten in later passes."""
        import os
        import tempfile

        rg = ReplicaGroup([[0, 1]])

        def torch_ref(x):
            pg = get_pg(rg)
            out = x.clone()
            dist.all_reduce(out, group=pg)
            dist.all_reduce(out, group=pg)
            return {"out": out}

        inputs = {0: {"x": torch.tensor([1.0])}, 1: {"x": torch.tensor([2.0])}}
        runner = SimDistRunner(num_ranks=2, replica_groups=[rg])

        # Run and verify correctness
        results = runner.run(torch_ref, inputs)
        assert torch.allclose(results[0]["out"], torch.tensor([6.0]))
        assert torch.allclose(results[1]["out"], torch.tensor([6.0]))

        # Verify the optimization works by running again with a spy on os.rename
        # (os.rename is only called for actual writes, not skipped ones)
        rename_log = tempfile.mktemp(suffix=".log")
        original_rename = os.rename

        def spy_rename(src, dst):
            if dst.endswith(".pkl"):
                with open(rename_log, "a") as f:
                    f.write(dst + "\n")
            return original_rename(src, dst)

        # Run a second time — but SimDistRunner uses fresh tempdir each time,
        # so we can't test cross-run skipping. Instead verify that within a single
        # run, the final pass doesn't re-write files from discovery passes.
        # We verify this by checking the _put_and_sync logic directly:
        coll_dir = tempfile.mkdtemp(prefix="test_skip_")
        pg = _MPSimProcessGroup(0, 2, coll_dir, "grp", stop_after_call=None, global_counter=[0])

        # Simulate: write collective 0 data (as if from pass 0)
        import pickle

        path = pg._file_path(0, 0)
        with open(path, "wb") as f:
            pickle.dump(torch.tensor([99.0]), f)

        # Now call _put_and_sync — it should skip writing since file exists
        orig_mtime = os.path.getmtime(path)
        import time

        time.sleep(0.01)
        # Reset counter to 0 so it targets call_id 0
        pg._global_counter[0] = 0
        pg._put_and_sync(torch.tensor([1.0]))

        # File should NOT have been modified
        new_mtime = os.path.getmtime(path)
        assert orig_mtime == new_mtime, "File was overwritten when it should have been skipped"

        # Verify the original data is preserved (not overwritten with new data)
        with open(path, "rb") as f:
            saved = pickle.load(f)
        assert torch.allclose(saved, torch.tensor([99.0])), "Original data was corrupted"

        import shutil

        shutil.rmtree(coll_dir)

    def test_skip_write_correctness_four_ranks(self):
        """Skip-write optimization produces correct results with 4 ranks and multiple collectives."""
        rg = ReplicaGroup([[0, 1, 2, 3]])

        def torch_ref(x):
            pg = get_pg(rg)
            # Collective 0: all_reduce
            out = x.clone()
            dist.all_reduce(out, group=pg)
            # Collective 1: all_gather
            gathered = [torch.zeros_like(out) for _ in range(4)]
            dist.all_gather(gathered, out, group=pg)
            return {"out": torch.cat(gathered)}

        inputs = {r: {"x": torch.tensor([float(r)])} for r in range(4)}
        runner = SimDistRunner(num_ranks=4, replica_groups=[rg])
        results = runner.run(torch_ref, inputs)

        # all_reduce: 0+1+2+3=6, all ranks have [6]
        # all_gather: [6, 6, 6, 6]
        expected = torch.tensor([6.0, 6.0, 6.0, 6.0])
        for r in range(4):
            assert torch.allclose(results[r]["out"], expected)
