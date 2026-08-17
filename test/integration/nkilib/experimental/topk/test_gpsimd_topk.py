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

"""Integration tests for the GpSIMD nisa.topk kernel (gpsimd_topk).

Mirrors the rotational_topk validator: values are compared sorted (internal order
forgiving), a strict descending-order check runs when sorted output is requested,
and indices are validated indirectly by gathering the input at the returned indices.
"""

from typing import Any, final

import ml_dtypes
import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.experimental.topk.gpsimd_topk import create_gpsimd_topk_config, gpsimd_topk
from nkilib_src.nkilib.experimental.topk.gpsimd_topk_torch import gpsimd_topk_torch_ref
from typing_extensions import override

from test.utils.common_dataclasses import (
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    Platforms,
)
from test.utils.comparators import maxAllClose
from test.utils.metrics_collector import MetricsCollector
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _get_np_dtype(dtype):
    """Convert NKI dtype to numpy dtype."""
    if dtype == nl.bfloat16:
        return ml_dtypes.bfloat16
    return dtype


@pytest_test_metadata(name="GpSIMD TopK")
@pytest_marks(["topk", "gpsimd"])
@final
class TestGpsimdTopKKernel:
    @staticmethod
    def generate_inputs(batch: int, seqlen: int, vocab: int, k: int, dtype, sorted: bool = True):
        """Generate input tensors for the topk test."""
        np.random.seed(seed=42)
        np_dtype = _get_np_dtype(dtype)
        BxS = batch * seqlen
        inp = np.random.randn(BxS, vocab).astype(np_dtype)
        return {"inp": inp, "k": k, "sorted": sorted, "batch": batch, "seqlen": seqlen, "vocab": vocab, "dtype": dtype}

    @staticmethod
    def output_tensors(kernel_input):
        """Define output tensor shapes for validation."""
        inp = kernel_input["inp"]
        config = kernel_input["config"]
        k = config.k
        BxS, _ = inp.shape
        np_dtype = _get_np_dtype(config.inp_dtype)
        return {
            "topk_values": np.zeros((BxS, k), dtype=np_dtype),
            "topk_indices": np.zeros((BxS, k), dtype=np.uint32),
        }

    def run_topk_test(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        lnc_degree: int,
        batch: int,
        seqlen: int,
        vocab: int,
        k: int,
        dtype,
        sorted: bool = True,
    ):
        """Run a single GpSIMD topk test case via UnitTestFramework."""

        def input_generator(test_config, input_tensor_def=None):
            inputs = self.generate_inputs(batch, seqlen, vocab, k, dtype, sorted)
            inp_3d = inputs["inp"].reshape((batch, seqlen, vocab))
            config = create_gpsimd_topk_config(
                inp_shape=inp_3d.shape,
                inp_dtype=dtype,
                k=k,
                sorted=sorted,
                num_programs=lnc_degree,
            )
            inp_reshaped = inp_3d.reshape((config.BxS, config.vocab_size))
            return {"inp": inp_reshaped, "config": config}

        inputs = input_generator(test_config=None)

        def topk_comparator(golden_dict, output_tensors):
            input_tensor = inputs["inp"]
            golden_values = golden_dict["topk_values"]

            class ValuesValidator(CustomValidator):
                @override
                def validate(self, actual_raw_output: npt.NDArray[Any]):
                    BxS, _ = input_tensor.shape
                    k_local = inputs["config"].k
                    output = np.frombuffer(actual_raw_output, dtype=input_tensor.dtype).reshape(BxS, k_local)
                    self._print_with_log("Results for topk_values:")
                    values_correct = maxAllClose(
                        np.sort(output, axis=-1),
                        np.sort(golden_values, axis=-1),
                        verbose=1,
                        logfile=self.logfile,
                    )
                    if not values_correct:
                        return False
                    vocab_size = inputs["config"].vocab_size
                    if sorted and k_local > 1 and k_local < vocab_size:
                        output_f32 = output.astype(np.float32)
                        diffs = np.diff(output_f32, axis=-1)
                        # diffs > 0 means value increased (violates descending). Ties allowed.
                        non_descending = int(np.sum(diffs > 0))
                        if non_descending > 0:
                            worst_row = int(np.argmax(np.sum(diffs > 0, axis=-1)))
                            self._print_with_log(
                                f"FAIL: output not in descending order. "
                                f"{non_descending} violations. "
                                f"Worst row {worst_row}: {output_f32[worst_row, :8]}..."
                            )
                            return False
                    return True

            class IndicesValidator(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]):
                    BxS, _ = input_tensor.shape
                    k_local = inputs["config"].k
                    output = np.frombuffer(inference_output, dtype=np.uint32).reshape(BxS, k_local)
                    val = np.take_along_axis(input_tensor, output.astype(np.uint64), axis=-1)
                    self._print_with_log("Results for topk_indices:")
                    return maxAllClose(
                        np.sort(golden_values, axis=-1),
                        np.sort(val, axis=-1),
                        verbose=1,
                        logfile=self.logfile,
                    )

            return {
                "topk_values": CustomValidatorWithOutputTensorData(
                    validator=ValuesValidator,
                    output_ndarray=output_tensors["topk_values"],
                ),
                "topk_indices": CustomValidatorWithOutputTensorData(
                    validator=IndicesValidator,
                    output_ndarray=output_tensors["topk_indices"],
                ),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=gpsimd_topk,
            torch_ref=torch_ref_wrapper(gpsimd_topk_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=self.output_tensors,
        )

        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                logical_nc_config=lnc_degree,
                platform_target=platform_target,
            ),
            rtol=1e-3,
            atol=1e-5,
            custom_comparator=topk_comparator,
        )

    # fmt: off
    topk_unit_params = "lnc_degree, batch, seqlen, vocab_size, K, dtype"
    _ABBREVS = {"lnc_degree": "lnc", "batch": "b", "seqlen": "s", "vocab_size": "v", "K": "K", "dtype": "dt"}

    # A/B head-to-head shapes vs the rotational baseline (all bf16, lnc=2).
    topk_ab_perms = [
        # BxS=1,  v=3168,  k=256
        pytest.param(2, 1, 1, 3168, 256, nl.bfloat16, marks=pytest.mark.fast),
        # BxS=32, v=3168,  k=256
        [2, 32, 1, 3168, 256, nl.bfloat16],
        # BxS=40, v=4058,  k=256  (b=8, s=5 -> BxS=40)
        [2, 8, 5, 4058, 256, nl.bfloat16],
        # BxS=1,  v=16000, k=256
        [2, 1, 1, 16000, 256, nl.bfloat16],
    ]

    # bf16 edge cases.
    topk_edge_perms = [
        pytest.param(2, 1, 1, 8, 8, nl.bfloat16, marks=pytest.mark.fast),  # n == 8 minimum, k == n
        [2, 1, 1, 64, 1, nl.bfloat16],         # k == 1
        pytest.param(2, 1, 1, 3168, 7, nl.bfloat16, marks=pytest.mark.fast),  # k non-multiple-of-16 remainder
        [2, 1, 1, 3168, 99, nl.bfloat16],      # k remainder, multi-column
        [2, 4, 1, 3168, 256, nl.bfloat16],     # partial tile (4 rows < 8 groups)
        pytest.param(2, 17, 1, 3168, 256, nl.bfloat16, marks=pytest.mark.fast),  # multi-tile + partial last tile
    ]

    # GPT-OSS-120b MXFP4 decode shapes (TP8xDP8xEP64, k=256, n_prgs=2). Each GBS
    # makes two rotational_topk calls per step: a local call over the per-rank vocab
    # shard (3142 = 201088/64) and a global call over the all-gathered top-k width
    # (16384 = 256*64). n_rows = GBS/8. nisa.topk is bf16-only, so these run in bf16
    # (the model upcasts to fp32 only for the DVE rotational kernel's accuracy).
    topk_gptoss_perms = [
        # GBS=128 -> n_rows=16
        pytest.param(2, 16, 1, 3142, 256, nl.bfloat16, marks=pytest.mark.fast),   # local
        [2, 16, 1, 16384, 256, nl.bfloat16],                                      # global
        # GBS=512 -> n_rows=64
        [2, 64, 1, 3142, 256, nl.bfloat16],                                       # local
        [2, 64, 1, 16384, 256, nl.bfloat16],                                      # global
        # GBS=1024 -> n_rows=128
        [2, 128, 1, 3142, 256, nl.bfloat16],                                      # local
        [2, 128, 1, 16384, 256, nl.bfloat16],                                     # global
    ]

    # Regression guard for the small-tile (partial-tile, par_dim <= 32) cross-engine
    # index/value races on the two on-chip sort paths. k values here select the sort
    # paths the k in {1, 7, 8, 99, 256} suite above never exercises. Real sampling
    # passes arbitrary top_k (16, 32, 50 are common), and each of these lands on a path
    # the existing coverage skips:
    #   - k=16 -> k_pad=16, HALF=8  : split-sort + bitonic merge with the SMALLEST
    #             half width (one max8 pass, HALF == 8). Not covered (k=8 takes the
    #             full-width path since k_pad=16 > n_pass*8=8; k=64+ have HALF>=32).
    #   - k=32 -> k_pad=32, HALF=16 : split path, log2(k_pad)=5 merge stages (the only
    #             tested split shape with an ODD stage count). The merge's stage-0
    #             cross-engine read of the reverse-gathered valley high half raced the
    #             gather at n_srows==1, corrupting VALUES (not just indices) on trn3.
    #   - k=50 -> k_pad=64 > k, k % 16 == 2 : full-width path with SCATTERED unwritten
    #             de-snake columns (_invalid_desnake_col_runs), non-power-of-two-k so
    #             it misses the split path. The index gather -> snake->vocab remap
    #             cross-engine read raced the last gather pass at n_srows==1. k=99
    #             covers scattered padding too, but at a different k_cols; k=50 is the
    #             common sampling value and its own case.
    # All at lnc=2 with small BxS (per_lnc_BxS not a multiple of 8 -> the safe DMA
    # path), mirroring decode sampling (batch ~1 per sequence). A second row (BxS=2)
    # keeps the shape a normal 2D case. The BxS=16 gptoss rows are the large-batch
    # controls (fast_dma_safe path), which were always correct.
    topk_kbug_perms = [
        pytest.param(2, 2, 1, 1024, 16, nl.bfloat16, marks=pytest.mark.fast),   # split, HALF=8
        pytest.param(2, 2, 1, 1024, 32, nl.bfloat16, marks=pytest.mark.fast),   # split, odd stages, value+index path
        pytest.param(2, 2, 1, 1024, 50, nl.bfloat16, marks=pytest.mark.fast),   # full-width, scattered padding
        [2, 16, 1, 3142, 16, nl.bfloat16],    # gptoss local vocab, k=16 (decode-realistic)
        [2, 16, 1, 3142, 32, nl.bfloat16],    # gptoss local vocab, k=32
        [2, 16, 1, 3142, 50, nl.bfloat16],    # gptoss local vocab, k=50
    ]

    # Dense k-sweep across the full small-tile path-selection surface: every k below
    # hits a distinct sort-path regime (SPLIT vs FULL-width; small/large HALF; k%16==0
    # vs scattered padding), all on the fixed shape (BxS=2, v=1024, lnc=2) that isolates
    # k on the safe (partial-tile) DMA path. This is the broad guard that any future
    # change keeps every path correct across k, not just the representative kbug values.
    topk_ksweep_perms = [
        # FULL-width, no padding (k%16==0 or k_cols==1 tail).
        [2, 2, 1, 1024, 8, nl.bfloat16],    # FULL HALF8-region, k_cols=1
        [2, 2, 1, 1024, 48, nl.bfloat16],   # FULL, k_pad==k, no pad
        [2, 2, 1, 1024, 80, nl.bfloat16],   # FULL, k_pad==k
        [2, 2, 1, 1024, 96, nl.bfloat16],   # FULL, k_pad==k
        # SPLIT path, smallest HALF (8): k=10,12,16.
        [2, 2, 1, 1024, 10, nl.bfloat16],
        [2, 2, 1, 1024, 12, nl.bfloat16],
        # SPLIT, HALF=16: k=30,32.
        [2, 2, 1, 1024, 30, nl.bfloat16],
        # SPLIT, HALF>=32: k=60,64,128.
        [2, 2, 1, 1024, 60, nl.bfloat16],
        [2, 2, 1, 1024, 64, nl.bfloat16],
        [2, 2, 1, 1024, 128, nl.bfloat16],
        # FULL-width, SCATTERED padding (k%16!=0, non-pow2 k_pad): k=20,24,40,100,150,200.
        [2, 2, 1, 1024, 20, nl.bfloat16],
        [2, 2, 1, 1024, 24, nl.bfloat16],
        [2, 2, 1, 1024, 40, nl.bfloat16],
        [2, 2, 1, 1024, 100, nl.bfloat16],
        [2, 2, 1, 1024, 150, nl.bfloat16],
        [2, 2, 1, 1024, 200, nl.bfloat16],
    ]

    # Full-width-path index guard across the partial-tile boundary. The full-width sort
    # failures (k=20,24,150) manifested only at small BxS (par_dim <= 32, the safe DMA
    # path) and were always correct at BxS=16 (full 128-partition tiles, fast path) --
    # the SAME partial-tile cross-engine race as the split path, not a distinct
    # padding-mask defect. Pair each k across both regimes so a regression on either the
    # small-tile fix or the large-batch path is caught. v=1024, lnc=2. BxS=2 -> par_dim=16
    # (safe path, fix active); BxS=16 -> per_lnc=8 -> par_dim=128 (fast path).
    topk_fullpath_batch_perms = [
        [2, 2, 1, 1024, 20, nl.bfloat16],    # BxS=2 (par_dim=16, safe path)
        [2, 16, 1, 1024, 20, nl.bfloat16],   # BxS=16 (par_dim=128, fast path)
        [2, 2, 1, 1024, 24, nl.bfloat16],
        [2, 16, 1, 1024, 24, nl.bfloat16],
        [2, 2, 1, 1024, 150, nl.bfloat16],
        [2, 16, 1, 1024, 150, nl.bfloat16],
    ]

    topk_batchsweep_perms = [
        [2, 1, 1, 1024, 32, nl.bfloat16],    # BxS=1 -> n_prgs=1, per_lnc=1, par_dim=16 (TRUE decode shape)
        [2, 2, 1, 1024, 32, nl.bfloat16],    # per_lnc=1, par_dim=16
        [2, 4, 1, 1024, 32, nl.bfloat16],    # per_lnc=2, par_dim=32 (partial tile)
        [2, 8, 1, 1024, 32, nl.bfloat16],    # per_lnc=4, par_dim=64 (partial tile)
        [2, 9, 1, 1024, 32, nl.bfloat16],    # per_lnc=5 (odd, partial tile)
        [2, 16, 1, 1024, 32, nl.bfloat16],   # per_lnc=8 (full tile, fast_dma_safe=True)
        [2, 128, 1, 1024, 32, nl.bfloat16],  # per_lnc=64 (many full tiles)
    ]

    # sorted=False fast path (skips the descending sort, compacts the K results in
    # arbitrary order). Covers both the k % 16 == 0 single-run compaction (k=256) and
    # the k % 16 != 0 scattered-column compaction (v=3168, k=99 and k=7).
    topk_unsorted_perms = [
        # k % 16 == 0: valid de-snaked columns are already a contiguous prefix
        # (single full-width compaction run).
        pytest.param(2, 16, 1, 3142, 256, nl.bfloat16, marks=pytest.mark.fast),   # gptoss-style local
        [2, 1, 1, 3168, 256, nl.bfloat16],                                        # BxS=1
        # k % 16 != 0: valid columns are scattered, multi-run compaction.
        pytest.param(2, 1, 1, 3168, 99, nl.bfloat16, marks=pytest.mark.fast),     # multi-column compaction
        [2, 1, 1, 3168, 7, nl.bfloat16],                                          # k <= 16, single-run compaction
    ]
    # fmt: on

    @pytest_parametrize(topk_unit_params, topk_ab_perms, abbrevs=_ABBREVS)
    def test_gpsimd_topk_ab(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        lnc_degree,
        batch,
        seqlen,
        vocab_size,
        K,
        dtype,
    ):
        self.run_topk_test(
            test_manager=test_manager,
            platform_target=platform_target,
            lnc_degree=lnc_degree,
            batch=batch,
            seqlen=seqlen,
            vocab=vocab_size,
            k=K,
            dtype=dtype,
        )

    @pytest_parametrize(topk_unit_params, topk_edge_perms, abbrevs=_ABBREVS)
    def test_gpsimd_topk_edge(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        lnc_degree,
        batch,
        seqlen,
        vocab_size,
        K,
        dtype,
    ):
        self.run_topk_test(
            test_manager=test_manager,
            platform_target=platform_target,
            lnc_degree=lnc_degree,
            batch=batch,
            seqlen=seqlen,
            vocab=vocab_size,
            k=K,
            dtype=dtype,
        )

    @pytest_parametrize(topk_unit_params, topk_kbug_perms, abbrevs=_ABBREVS)
    def test_gpsimd_topk_kbug(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        lnc_degree,
        batch,
        seqlen,
        vocab_size,
        K,
        dtype,
    ):
        """Regression: k values (16, 32, 50) that exercise the split-sort/bitonic-merge
        and scattered-padding paths the k in {1,7,8,99,256} suite skips. torch.topk with
        these k is bit-exact on the value SET (validated sorted) and the descending order
        must hold; indices are validated via gather. Guards the small-tile (par_dim <= 32)
        cross-engine sync fixes on the split-merge and full-width-remap index paths."""
        self.run_topk_test(
            test_manager=test_manager,
            platform_target=platform_target,
            lnc_degree=lnc_degree,
            batch=batch,
            seqlen=seqlen,
            vocab=vocab_size,
            k=K,
            dtype=dtype,
        )

    @pytest_parametrize(topk_unit_params, topk_ksweep_perms, abbrevs=_ABBREVS)
    def test_gpsimd_topk_ksweep(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        lnc_degree,
        batch,
        seqlen,
        vocab_size,
        K,
        dtype,
    ):
        """Dense k-sweep mapping the failure surface (which k / path / validator break)."""
        self.run_topk_test(
            test_manager=test_manager,
            platform_target=platform_target,
            lnc_degree=lnc_degree,
            batch=batch,
            seqlen=seqlen,
            vocab=vocab_size,
            k=K,
            dtype=dtype,
        )

    @pytest_parametrize(topk_unit_params, topk_fullpath_batch_perms, abbrevs=_ABBREVS)
    def test_gpsimd_topk_fullpath_batch(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        lnc_degree,
        batch,
        seqlen,
        vocab_size,
        K,
        dtype,
    ):
        """Cross-check: do the full-width-path k failures also clear at BxS=16 (full tile)?"""
        self.run_topk_test(
            test_manager=test_manager,
            platform_target=platform_target,
            lnc_degree=lnc_degree,
            batch=batch,
            seqlen=seqlen,
            vocab=vocab_size,
            k=K,
            dtype=dtype,
        )

    @pytest_parametrize(topk_unit_params, topk_batchsweep_perms, abbrevs=_ABBREVS)
    def test_gpsimd_topk_batchsweep(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        lnc_degree,
        batch,
        seqlen,
        vocab_size,
        K,
        dtype,
    ):
        """Batch-dependence probe on k=32: is the bug pure-k or batch-dependent?"""
        self.run_topk_test(
            test_manager=test_manager,
            platform_target=platform_target,
            lnc_degree=lnc_degree,
            batch=batch,
            seqlen=seqlen,
            vocab=vocab_size,
            k=K,
            dtype=dtype,
        )

    @pytest_parametrize(topk_unit_params, topk_gptoss_perms, abbrevs=_ABBREVS)
    def test_gpsimd_topk_gptoss(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        lnc_degree,
        batch,
        seqlen,
        vocab_size,
        K,
        dtype,
    ):
        """GPT-OSS-120b MXFP4 decode top-k shapes (local + global calls per GBS)."""
        self.run_topk_test(
            test_manager=test_manager,
            platform_target=platform_target,
            lnc_degree=lnc_degree,
            batch=batch,
            seqlen=seqlen,
            vocab=vocab_size,
            k=K,
            dtype=dtype,
        )

    @pytest_parametrize(topk_unit_params, topk_unsorted_perms, abbrevs=_ABBREVS)
    def test_gpsimd_topk_sorted_flag(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        lnc_degree,
        batch,
        seqlen,
        vocab_size,
        K,
        dtype,
    ):
        """Unsorted fast path (config.sorted == False): the K results are validated as
        a value SET (sorted-compare) and via index gather; the descending-order check
        is skipped by the validator when sorted is False."""
        self.run_topk_test(
            test_manager=test_manager,
            platform_target=platform_target,
            lnc_degree=lnc_degree,
            batch=batch,
            seqlen=seqlen,
            vocab=vocab_size,
            k=K,
            dtype=dtype,
            sorted=False,
        )

    @pytest_parametrize(topk_unit_params, topk_gptoss_perms, abbrevs=_ABBREVS)
    def test_gpsimd_topk_gptoss_unsorted(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        lnc_degree,
        batch,
        seqlen,
        vocab_size,
        K,
        dtype,
    ):
        """GPT-OSS-120b decode shapes on the unsorted fast path (config.sorted == False).
        Mirrors test_gpsimd_topk_gptoss for a head-to-head sorted-vs-unsorted comparison
        across all 6 shapes. All gptoss shapes are k=256 (k % 16 == 0), so this exercises
        the single-run compaction path; the descending-order check is skipped by the
        validator when sorted is False (value SET + index gather are still validated)."""
        self.run_topk_test(
            test_manager=test_manager,
            platform_target=platform_target,
            lnc_degree=lnc_degree,
            batch=batch,
            seqlen=seqlen,
            vocab=vocab_size,
            k=K,
            dtype=dtype,
            sorted=False,
        )


# MODEL TESTING ENTRY POINT
# The "model" mark routes these configs into the pipeline's Model Regression stage,
# which compiles AND runs on hardware (the compile-only Development stage cannot catch
# the small-tile cross-engine races / padding bug this suite guards, since they compile
# clean and only diverge on overlapped-execution hardware). Kept to a tight set of the
# confirmed small-tile regression configs (one per fixed defect) plus a large-batch
# control, so the recurring hardware cost stays small; the dense sweeps above remain
# compile-only. Validated on trn2 and trn3_a0 (full suite, abs diff = 0). Delegates to
# TestGpsimdTopKKernel.run_topk_test (a second @pytest_test_metadata per file is rejected,
# so this class carries only the marks).
@pytest_marks(["topk", "gpsimd", "model"])
@final
class TestGpsimdTopKModel:
    """Model regression tests for gpsimd_topk (on-hardware, small-tile guard)."""

    topk_model_params = "lnc_degree, batch, seqlen, vocab_size, K, dtype"
    # (lnc, BxS, seqlen, vocab, k): split-path race (k=16/32), full-width index race
    # (k=50), split-path padding (k=60) -- all at the small partial-tile BxS=2 that
    # triggers the bugs -- plus a large-batch GPT-OSS control (fast_dma_safe path).
    _MODEL_PARAMS = [
        (2, 2, 1, 1024, 16, nl.bfloat16),
        (2, 2, 1, 1024, 32, nl.bfloat16),
        (2, 2, 1, 1024, 50, nl.bfloat16),
        (2, 2, 1, 1024, 60, nl.bfloat16),
        (2, 16, 1, 3142, 256, nl.bfloat16),
    ]
    _MODEL_IDS = [f"lnc-{p[0]}_b-{p[1]}_s-{p[2]}_v-{p[3]}_K-{p[4]}_dt-bfloat16" for p in _MODEL_PARAMS]

    @pytest.mark.parametrize(topk_model_params, _MODEL_PARAMS, ids=_MODEL_IDS)
    def test_model(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        lnc_degree,
        batch,
        seqlen,
        vocab_size,
        K,
        dtype,
    ):
        """Small-tile correctness guard, run on hardware in Model Regression."""
        TestGpsimdTopKKernel().run_topk_test(
            test_manager=test_manager,
            platform_target=platform_target,
            lnc_degree=lnc_degree,
            batch=batch,
            seqlen=seqlen,
            vocab=vocab_size,
            k=K,
            dtype=dtype,
        )
