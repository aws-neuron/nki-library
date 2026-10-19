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

"""Which all-to-all hop decomposition actually runs at 128 ranks, and does it permute correctly.

A single all_to_all over all 128 ranks aborts in the Neuron runtime
(``encd_mesh_memcopy`` descriptor-count assertion); a rank sweep puts the limit between 32 and
64 participants. These cases ask, on hardware, which multi-hop decomposition reaches the same
end state -- full vocab per rank -- while keeping every hop under that limit.

The input encodes its own provenance so the test checks the PERMUTATION rather than assuming
it: column 0 of every row carries the source rank and column 1 carries the source batch row,
both integers under 256 and therefore exact in bfloat16. One run recovers the whole mapping,
which is what lets a decomposition be accepted only if it also preserves
``column index == global vocab id``.
"""

import os
from typing import Any, final

import ml_dtypes
import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.experimental.gpt_oss_tail.tail_chunked_megakernel import compose_hops
from nkilib_src.nkilib.experimental.gpt_oss_tail.tail_chunked_megakernel_torch import rank_row_block
from typing_extensions import override

from test.utils.common_dataclasses import (
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    InferenceArgs,
    Platforms,
)
from test.utils.metrics_collector import MetricsCollector
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_collective_framework import CollectiveUnitTestFramework
from test.utils.unit_test_framework import torch_ref_wrapper

from .collective_probe_kernels import allgather_probe, collective_probe

# Probe width = the production per-rank vocab shard (V/VTP = 201088/128). Using a token width
# like 4 made the per-peer chunk 8 bytes and the compiler's collective verifier rejected it
# outright; the real kernel moves 1571 bf16 = 3142 B per peer and compiles, so the probe has to
# match the production transfer size to be measuring the same thing.
_W = 1571


def _probe_input(gbs: int, rank: int) -> npt.NDArray[Any]:
    x = np.zeros((gbs, _W), dtype=np.float32)
    x[:, 0] = rank
    x[:, 1] = np.arange(gbs)
    return x.astype(ml_dtypes.bfloat16)


def _expected(vtp: int, gbs: int, rank: int, hops: tuple) -> npt.NDArray[Any]:
    """The assembled ``[gbs/vtp, vtp*_W]`` result this rank must hold after the hops."""
    r0, nr = rank_row_block(rank, vtp, gbs, hops)
    out = np.zeros((nr, vtp * _W), dtype=np.float32)
    for p in range(vtp):
        out[:, p * _W + 0] = p  # peer p's block -> columns [p*_W, ...): p IS the vocab shard
        out[:, p * _W + 1] = np.arange(r0, r0 + nr)
    return out


class TestComposeHopsArithmetic:
    """Hardware-free gates on the hop-mesh construction the probes exercise on device."""

    @pytest.mark.fast
    @pytest.mark.parametrize("vtp,hops", [(128, (8, 16)), (32, (4, 8)), (16, (16,))])
    def test_compose_hops_meshes_partition_ranks_in_order(self, vtp: int, hops: tuple):
        """Every hop's groups must partition the ranks, ascending within each group.

        Ascending order is the no-index-correction property (column index == global vocab
        id); partitioning exactly once is what makes the hops compose to a permutation.
        """
        meshes = compose_hops(vtp, hops)
        assert len(meshes) == len(hops)
        stride = 1
        for h, mesh in zip(hops, meshes, strict=True):
            covered = sorted(r for g in mesh for r in g)
            assert covered == list(range(vtp)), f"hop groups do not partition ranks: {mesh}"
            for g in mesh:
                assert len(g) == h
                assert list(g) == sorted(g), f"group not in ascending rank order: {g}"
                assert all(g[i + 1] - g[i] == stride for i in range(h - 1)), f"group stride != {stride}: {g}"
            stride *= h

    @pytest.mark.fast
    def test_compose_hops_rejects_sizes_that_do_not_multiply_to_vtp(self):
        with pytest.raises(ValueError):
            compose_hops(128, (8, 8))


@pytest_test_metadata(name="GPT-OSS Tail Collective Probe")
@pytest_marks(["gpt_oss_tail", "collectives"])
@final
class TestCollectiveProbe:
    """Hardware ascertainment only: no norm, no GEMM, no top-k, no sampler."""

    def _run(self, test_manager, platform_target, vtp: int, gbs: int, hops: tuple):
        meshes = compose_hops(vtp, hops)
        rows_per_rank = gbs // vtp

        def per_rank_input_generator(rank_id: int):
            return {"x": _probe_input(gbs, rank_id), "hop_meshes": meshes}

        def _ref(x, hop_meshes=None):
            del hop_meshes
            # Recover this rank from the provenance it was handed: column 0 is its own rank.
            arr = np.asarray(x.detach().to("cpu").float().numpy() if hasattr(x, "detach") else x, dtype=np.float32)
            return {"out": _expected(vtp, gbs, int(round(float(arr[0, 0]))), hops)}

        def comparator(rank_id: int, golden: dict):
            want = np.asarray(golden["out"], dtype=np.float32).reshape(rows_per_rank, vtp * _W)

            class OutValidator(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    got = np.asarray(
                        np.frombuffer(inference_output, dtype=ml_dtypes.bfloat16).reshape(rows_per_rank, vtp * _W),
                        dtype=np.float32,
                    )
                    if np.array_equal(got, want):
                        return True
                    bad_rank = np.flatnonzero(got[0, 0::_W] != want[0, 0::_W])
                    bad_row = np.flatnonzero(got[:, 1] != want[:, 1])
                    self._print_with_log(
                        f"FAIL probe rank {rank_id} hops={hops}: "
                        f"{bad_rank.size} vocab-shard columns land from the wrong peer "
                        f"(first {bad_rank[:6].tolist()}), {bad_row.size} rows come from the wrong "
                        f"batch offset. Wrong peer columns mean column index no longer equals the "
                        f"global vocab id; wrong rows mean rank_row_block disagrees with the hops."
                    )
                    return False

            return {
                "out": CustomValidatorWithOutputTensorData(
                    OutValidator, np.zeros((rows_per_rank, vtp * _W), dtype=ml_dtypes.bfloat16)
                )
            }

        framework = CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=collective_probe,
            torch_ref=torch_ref_wrapper(_ref),
            per_rank_input_generator=per_rank_input_generator,
            collective_ranks=vtp,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=1, platform_target=platform_target),
            output_keys=["out"],
            rtol=0,
            atol=0,
            inference_args=InferenceArgs(collective_ranks=vtp),
            custom_comparator=comparator,
        )

    @pytest.mark.skip_simulation  # class-B nccl collectives have no CPU-sim backend
    @pytest.mark.skipif(
        os.environ.get("NKILIB_RESERVED_TRN3") != "1",
        reason=(
            "128-rank ascertainment sweep: needs 128 cores AND a host that allows more than one "
            "replica-group mesh per NEFF. The shared fleet's trn3_a0 has the cores but refuses "
            "multi-mesh, so every multi-hop case fails there for host reasons and the single-hop "
            "(128,) case is the mixed device layout all_to_all cannot express -- i.e. most of this "
            "sweep is EXPECTED to fail off a reserved trn3, which is the measurement, not a "
            "regression. Set NKILIB_RESERVED_TRN3=1 to run it where the answer is meaningful."
        ),
    )
    @pytest.mark.parametrize(
        "hops",
        [
            # Ascertainment sweep: several of these are EXPECTED to be rejected, and which ones
            # is the finding. A group must be device-local or one-core-per-device, so (128,) and
            # any hop of 16 CONTIGUOUS ranks are refused; (8,16), (8,4,4) and (8,8,2) are legal.
            # Needs 128 cores, so it only runs on a reserved trn3.
            (128,),  # the direct form: 16 devices x 8 cores, refused. Kept as the control.
            (8, 16),  # contiguous-then-strided, the shape the tail kernel would use
            (16, 8),
            (4, 32),
            (32, 4),
            (2, 64),  # contains a 64-way hop: expected to abort, bounds the limit from above
            (8, 4, 4),  # three hops, all small
        ],
    )
    def test_hop_decompositions_at_128_ranks(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        hops: tuple,
    ):
        """Run each decomposition at VTP=128 and check both that it executes and that it permutes.

        A decomposition is only usable if it does BOTH: survives the runtime's participant
        limit, and still leaves column index equal to global vocab id.
        """
        self._run(test_manager, platform_target, 128, 128, hops)

    @pytest.mark.skip_simulation
    @pytest.mark.parametrize(
        "vtp,hops",
        [
            (32, (32,)),  # control: ONE mesh, known-good at 32 ranks on trn3_a0
            (16, (16,)),  # control at 16
            # The two-mesh cases are HOST DEPENDENT and are the measurement, not a regression:
            # the shared fleet's trn3_a0 refuses more than one replica-group mesh per NEFF at any
            # rank count ("Failed to build and load collectives resources"), while a reserved
            # 16-device trn3 host runs three. xfail non-strict so they report the host's answer
            # without turning the suite red, and so they XPASS visibly on a host that allows it.
            # TODO: drop the xfail if the shared fleet gains multi-mesh support.
            pytest.param(
                32,
                (4, 8),
                marks=pytest.mark.xfail(reason="two meshes per NEFF: unsupported on trn3_a0", strict=False),
            ),
            pytest.param(
                16,
                (4, 4),
                marks=pytest.mark.xfail(reason="two meshes per NEFF: unsupported on trn3_a0", strict=False),
            ),
        ],
    )
    def test_one_mesh_vs_two_meshes(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        vtp: int,
        hops: tuple,
    ):
        """Separate "too many ranks" from "more than one replica-group mesh in a NEFF".

        At 128 ranks the single-hop form dies in the runtime's mesh descriptor assertion while
        every multi-hop form instead fails to BUILD AND LOAD collectives resources -- two
        different failures, so the cause is not simply participant count. Holding the rank
        count at a value where one mesh is known to work and adding a second mesh isolates
        which of the two properties the host actually objects to.
        """
        self._run(test_manager, platform_target, vtp, vtp, hops)


# No @pytest_test_metadata here: the repo allows exactly ONE per file and TestCollectiveProbe
# above carries it.
@pytest_marks(["gpt_oss_tail", "collectives"])
@final
class TestAllGatherProbe:
    """Is the >=64-participant limit specific to all_to_all, or does any mesh that size fail?"""

    @pytest.mark.skip_simulation
    @pytest.mark.parametrize("vtp", [32, 128])
    def test_all_gather_participant_limit(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        vtp: int,
    ):
        """all_gather over ALL ranks: one mesh, vtp participants.

        32 is the control (all_to_all passes there); 128 is the question. Validation checks the
        gathered layout -- peer p's rows must land at ``[p*gbs, (p+1)*gbs)`` -- so a pass means
        the collective both ran and placed data where the caller would need it.
        """
        gbs = vtp
        groups = (tuple(range(vtp)),)

        def per_rank_input_generator(rank_id: int):
            return {"x": _probe_input(gbs, rank_id), "groups": groups}

        def _ref(x, groups=None):
            del groups
            out = np.zeros((vtp * gbs, _W), dtype=np.float32)
            for p in range(vtp):
                out[p * gbs : (p + 1) * gbs, 0] = p
                out[p * gbs : (p + 1) * gbs, 1] = np.arange(gbs)
            return {"out": out}

        def comparator(rank_id: int, golden: dict):
            want = np.asarray(golden["out"], dtype=np.float32).reshape(vtp * gbs, _W)

            class OutValidator(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    got = np.asarray(
                        np.frombuffer(inference_output, dtype=ml_dtypes.bfloat16).reshape(vtp * gbs, _W),
                        dtype=np.float32,
                    )
                    if np.array_equal(got[:, :2], want[:, :2]):
                        return True
                    self._print_with_log(
                        f"FAIL all_gather rank {rank_id} vtp={vtp}: gathered layout wrong; "
                        f"peer block 1 col0 got {got[gbs, 0]} want {want[gbs, 0]}"
                    )
                    return False

            return {
                "out": CustomValidatorWithOutputTensorData(
                    OutValidator, np.zeros((vtp * gbs, _W), dtype=ml_dtypes.bfloat16)
                )
            }

        framework = CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=allgather_probe,
            torch_ref=torch_ref_wrapper(_ref),
            per_rank_input_generator=per_rank_input_generator,
            collective_ranks=vtp,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=1, platform_target=platform_target),
            output_keys=["out"],
            rtol=0,
            atol=0,
            inference_args=InferenceArgs(collective_ranks=vtp),
            custom_comparator=comparator,
        )
