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

On top of that, an elementwise value<->index pairing check asserts
input[i, indices[i, j]] == values[i, j]. The sorted set comparisons above are blind
to the two output tensors being permuted differently from each other, which is
exactly the failure mode a sampler consuming (value, index) at slot j would hit.
"""

from typing import Any, final

import ml_dtypes
import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.experimental.topk.gpsimd_topk import (
    BFLOAT16_MIN,
    create_gpsimd_topk_config,
    gpsimd_topk,
)
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
    def generate_inputs(batch: int, seqlen: int, vocab: int, k: int, dtype, sorted: bool = True, data: str = "randn"):
        """Generate input tensors for the topk test.

        ``data`` selects the input distribution. The constant ones (``floor``, ``ninf``,
        ``pad_isolate``) draw nothing; the ones that overlay a mask onto ordinary randn draw
        after the same fixed seed, so their unmasked rows hold the data the default
        ``randn`` case produces at that shape.
        """
        np.random.seed(seed=42)
        np_dtype = _get_np_dtype(dtype)
        BxS = batch * seqlen
        if data == "uninit":
            # Arbitrary 16-bit patterns read as bfloat16: ~0.4% NaN plus both infinities
            # and saturated finites -- what an UNINITIALIZED buffer holds. The decode-bench
            # KV connector deliberately leaves the KV cache unfilled, so decode benchmarking
            # reaches this kernel with exactly this. Non-orderable values break the
            # max8 -> nc_match_replace8 -> nc_n_gather chain: an unmatchable value leaves its
            # reported position undefined, and that position is used as a gather index.
            raw = np.random.randint(0, 1 << 16, size=(BxS, vocab), dtype=np.uint16)
            inp = raw.view(ml_dtypes.bfloat16).astype(np_dtype)
        elif data == "floor":
            # Row filled with the kernel's OWN padding sentinel, plus two real winners.
            # The real data therefore TIES the padding value exactly. nc_match_replace8
            # reports the FIRST occurrence of a matched value, so on a tie the real slots
            # (lower snake positions) win and a padding slot need never be selected --
            # which is why this distribution does NOT expose the padded-index cause.
            inp = np.full((BxS, vocab), BFLOAT16_MIN, dtype=np.float32)
            inp[:, 0] = 1.0
            inp[:, vocab // 2] = 2.0
            inp = inp.astype(np_dtype)
        elif data == "ninf":
            # Genuine -inf everywhere. BFLOAT16_MIN is FINITE, so every padding slot is
            # STRICTLY GREATER than every real element: all full_n - vocab padding slots
            # win the top-k deterministically and remap to indices in [vocab, full_n).
            inp = np.full((BxS, vocab), -np.inf, dtype=np.float32).astype(np_dtype)
        elif data == "pad_isolate":
            # Isolates the padded-index cause from the unmatchable-value cause. Use a vocab
            # with EXACTLY ONE padding slot (vocab % 16 == 15) and give the row k-1 DISTINCT
            # finite values, the remainder -inf. The top-k is then exactly those k-1 distinct
            # finite values plus the single padding sentinel: the selected set is fully
            # distinct and orderable, so no value is unmatchable and the NaN/duplicate-extreme
            # cause cannot fire. Any returned index >= vocab is therefore attributable to the
            # padding slot alone.
            inp = np.full((BxS, vocab), -np.inf, dtype=np.float32)
            n_fin = k - 1
            vals = np.linspace(1.0, float(n_fin), n_fin, dtype=np.float32)
            inp[:, :n_fin] = vals
            inp = inp.astype(np_dtype)
        elif data == "neg_inf_masked":
            # A CONSTRAINED-DECODING row: row 0 is -inf (a disallowed token) everywhere
            # except the first k-1 entries. Only k-1 finite values exist, so at least one
            # output slot MUST be filled from a -inf element and the k-largest set is
            # nevertheless well defined and fully orderable (-inf is a legitimate,
            # comparable logit -- this is not garbage input).
            #
            # -inf is exactly the marker nc_match_replace8 writes over a consumed slot
            # (imm=float("-inf")), so a real -inf in the DATA is indistinguishable from a
            # slot an earlier pass already claimed: match_replace8 then reports the
            # position of the WRONG slot and the paired index is gathered from there.
            # The index stays IN RANGE, so only the elementwise value<->index pairing
            # check sees it; a sampler consuming (probability, token_id) at slot j emits
            # the wrong token silently. Remaining rows are ordinary randn so the shape's
            # normal path is exercised alongside.
            inp = np.random.randn(BxS, vocab).astype(np.float32)
            inp[0, :] = -np.inf
            inp[0, : k - 1] = np.arange(k - 1, dtype=np.float32)
            inp = inp.astype(np_dtype)
        elif data == "nan_masked":
            # Same structure as neg_inf_masked but with NaN instead of -inf, to answer
            # whether the pairing is ALSO broken for NaN. NaN never compares equal to
            # itself, so nc_match_replace8 can never match a NaN that max8 selected and
            # its reported position is undefined by the ISA contract -- a different
            # mechanism from the -inf/imm collision. There is no well-defined "k largest"
            # with NaN present, so only the PAIRING is checked here (NaN==NaN counted as
            # paired), never the value set against a golden.
            inp = np.random.randn(BxS, vocab).astype(np.float32)
            inp[0, :] = np.nan
            inp[0, : k - 1] = np.arange(k - 1, dtype=np.float32)
            inp = inp.astype(np_dtype)
        elif data == "dup_finite":
            # Control for neg_inf_masked: row 0 holds ONE repeated FINITE value, so the
            # selected set is maximally duplicated but contains no -inf and no NaN. If
            # duplication alone broke the max8 -> match_replace8 -> gather chain this
            # would fail too; it passes, which isolates the defect to the -inf/imm
            # collision rather than to duplicate values in general.
            inp = np.random.randn(BxS, vocab).astype(np.float32)
            inp[0, :] = 1.5
            inp = inp.astype(np_dtype)
        else:
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
        data: str = "randn",
        index_bounds_only: bool = False,
    ):
        """Run a single GpSIMD topk test case via UnitTestFramework.

        ``index_bounds_only`` drops the golden/value-SET and descending-order
        comparisons -- with NaN present there is no well-defined "the k largest", so
        there is nothing to compare a value set against. It KEEPS the index-range check
        and the value<->index pairing check (in its NaN-tolerant form, see
        ``check_pairing_nan_tolerant``): whatever the kernel chose to return, the index
        it reports at slot j must point at the value it reports at slot j, and that is
        what a sampler consumes.
        """

        def input_generator(test_config, input_tensor_def=None):
            inputs = self.generate_inputs(batch, seqlen, vocab, k, dtype, sorted, data)
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
            # Decoded kernel outputs shared across the two validators, each of which
            # only receives its own output tensor. The value<->index pairing check
            # needs both, so it runs from whichever validator fills this last.
            # topk_comparator runs once per test (the golden is memoized), so this
            # starts empty for every test case.
            kernel_out: dict[str, npt.NDArray[Any]] = {}

            def report_out_of_range(validator: CustomValidator, indices: npt.NDArray[Any]) -> bool:
                """Report indices outside [0, vocab_size); True if any were found.

                Guards every gather of the input at kernel-returned indices. Without
                it an out-of-range index raises a bare IndexError from
                np.take_along_axis, which fails the test but says nothing about which
                slot went wrong. The kernel's snake->vocab remap can exceed the last
                valid column whenever 16 * k_cols > vocab_size (true for v=8, 3142
                and 4058 in this suite), so a padding-mask regression lands here.
                """
                vocab_size = inputs["config"].vocab_size
                bad = indices >= vocab_size
                n_bad = int(np.count_nonzero(bad))
                if n_bad == 0:
                    return False
                row = int(np.argmax(np.count_nonzero(bad, axis=-1)))
                cols = np.flatnonzero(bad[row])[:8]
                # full_n - 1 is the largest index a PADDING slot can remap to; anything
                # above it can only have come from a garbage gather position. Reporting
                # max_index against that bound is what separates the two causes.
                n_cols_ = -(-vocab_size // 16)
                full_n_ = 16 * n_cols_
                validator._print_with_log(
                    f"FAIL: {n_bad} returned indices are out of range for vocab_size={vocab_size}. "
                    f"max_index={int(indices.max())} (padding can reach full_n-1={full_n_ - 1}). "
                    f"Worst row {row}, first bad slots j={cols.tolist()}: "
                    f"indices={indices[row, cols].tolist()}"
                )
                return True

            def check_pairing_nan_tolerant(
                validator: CustomValidator,
                values: npt.NDArray[Any],
                indices: npt.NDArray[Any],
            ) -> bool:
                """Pairing check that stays meaningful when the input contains NaN.

                With NaN present there is no well-defined "the k largest", so the value
                SET cannot be compared against a golden. The PAIRING is still a real
                contract though: whatever the kernel selected, the value it reports at
                slot j must be the input element at the index it reports at slot j. That
                is what a sampler consumes, and it is checkable without any notion of
                ordering.

                Two things stop this from being a plain bitwise comparison:
                  - NaN != NaN, so `==` cannot be used on the NaN slots.
                  - the value path is bf16 -> float32 (de-snake HBM + sort buffers) ->
                    bf16, and that round trip CANONICALISES NaN payloads (e.g. a
                    signalling 0x7F81 comes back as the quiet 0x7FC0). Requiring bitwise
                    equality would therefore fail on correct hardware output; the kernel
                    does not promise to preserve payload bits, only the value.

                So the assertion is per-class: the gathered input element and the reported
                value must agree as NaN / +Inf / -Inf / finite, and on the finite slots
                they must be bitwise equal (exactly, no tolerance -- the kernel only ever
                MOVES a finite bf16 value, it never computes on it). This catches a
                values<->indices desync on every non-NaN slot, and on the NaN slots it
                still catches a desync that pairs a NaN value with a finite element.
                """
                gathered = np.take_along_axis(input_tensor, indices.astype(np.uint64), axis=-1)
                g32 = gathered.astype(np.float32)
                v32 = values.astype(np.float32)

                def classify(a):
                    # 0 finite, 1 +Inf, 2 -Inf, 3 NaN
                    out = np.zeros(a.shape, dtype=np.uint8)
                    out[np.isposinf(a)] = 1
                    out[np.isneginf(a)] = 2
                    out[np.isnan(a)] = 3
                    return out

                g_cls, v_cls = classify(g32), classify(v32)
                class_mismatch = g_cls != v_cls
                finite_both = (g_cls == 0) & (v_cls == 0)
                # Bit-exact on the finite slots (same bf16 bits, so same float32 bits).
                finite_mismatch = finite_both & (g32.view(np.uint32) != v32.view(np.uint32))
                # A NaN input slot is EXPECTED to come back as the kernel's NaN-fold
                # floor (BFLOAT16_MIN), not as NaN: the kernel folds NaN out of the snake
                # before nisa.topk ranks it, precisely so the emitted (value, position)
                # pair is faithful. That substitution is the fix, not a desync, so do not
                # flag it. A REAL desync -- a value paired with an unrelated index -- is
                # still caught, because then the returned value is some other row element
                # rather than the floor.
                nan_folded = (g_cls == 3) & (v32 == np.float32(BFLOAT16_MIN))
                mismatch = (class_mismatch | finite_mismatch) & ~nan_folded

                n_nan = int(np.count_nonzero(v_cls == 3))
                n_checked = int(mismatch.size)
                n_mismatch = int(np.count_nonzero(mismatch))
                if n_mismatch == 0:
                    validator._print_with_log(
                        f"INFO: NaN-tolerant value<->index pairing passed ({n_checked} slots; "
                        f"{n_nan} slots returned NaN, {n_checked - n_nan} non-NaN checked bit-exactly)"
                    )
                    return True
                row = int(np.argmax(np.count_nonzero(mismatch, axis=-1)))
                cols = np.flatnonzero(mismatch[row])[:8]
                validator._print_with_log(
                    f"FAIL: value<->index desync under NaN. {n_mismatch} of {n_checked} slots "
                    f"where input[i, indices[i, j]] does not match values[i, j] "
                    f"(class or exact bits). Worst row {row}, first bad slots j={cols.tolist()}: "
                    f"values={v32[row, cols].tolist()} vs "
                    f"input[indices]={g32[row, cols].tolist()} "
                    f"(indices={indices[row, cols].tolist()})"
                )
                return False

            def check_value_index_pairing(validator: CustomValidator) -> bool:
                """Check input[i, indices[i, j]] == values[i, j] elementwise.

                The set-level checks in both validators sort each side before
                comparing, so they pass even when the values and indices tensors are
                permuted differently from each other -- i.e. values[j] not describing
                indices[j]. Top-k sampling consumes the pair at slot j (probability
                from values, token id from indices), so a desync emits the wrong token
                while each tensor still looks individually correct.

                Elementwise, and therefore tie-safe: it constrains each returned index
                against the value the kernel itself reported for that slot, never
                against torch's arbitrary choice among duplicate values.
                """
                if "topk_values" not in kernel_out or "topk_indices" not in kernel_out:
                    # The other validator has not run yet; it will run this check.
                    validator._print_with_log("INFO: value<->index pairing check deferred to the other topk validator")
                    return True

                values = kernel_out["topk_values"]
                indices = kernel_out["topk_indices"]
                if report_out_of_range(validator, indices):
                    return False
                if index_bounds_only:
                    return check_pairing_nan_tolerant(validator, values, indices)
                gathered = np.take_along_axis(input_tensor, indices.astype(np.uint64), axis=-1)
                mismatch = gathered.astype(np.float32) != values.astype(np.float32)
                n_mismatch = int(np.count_nonzero(mismatch))
                if n_mismatch == 0:
                    validator._print_with_log(
                        f"INFO: value<->index pairing check passed "
                        f"({values.shape[0]}x{values.shape[1]} slots, all paired)"
                    )
                    return True

                row = int(np.argmax(np.count_nonzero(mismatch, axis=-1)))
                cols = np.flatnonzero(mismatch[row])[:8]
                validator._print_with_log(
                    f"FAIL: value<->index desync. {n_mismatch} slots where "
                    f"input[i, indices[i, j]] != values[i, j]. "
                    f"Worst row {row}, first bad slots j={cols.tolist()}: "
                    f"values={values[row, cols].astype(np.float32).tolist()} vs "
                    f"input[indices]={gathered[row, cols].astype(np.float32).tolist()} "
                    f"(indices={indices[row, cols].tolist()})"
                )
                return False

            class ValuesValidator(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    BxS, _ = input_tensor.shape
                    k_local = inputs["config"].k
                    output = np.frombuffer(inference_output, dtype=input_tensor.dtype).reshape(BxS, k_local)
                    kernel_out["topk_values"] = output
                    if index_bounds_only:
                        # No golden value SET to compare against under NaN, but the
                        # value<->index pairing is still checkable. Whichever of the two
                        # validators runs SECOND has both tensors and runs the check.
                        self._print_with_log("topk_values: golden set comparison skipped (unorderable input)")
                        return check_value_index_pairing(self)
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
                    kernel_out["topk_indices"] = output
                    self._print_with_log("Results for topk_indices:")
                    if report_out_of_range(self, output):
                        return False
                    if index_bounds_only:
                        vocab_size = inputs["config"].vocab_size
                        self._print_with_log(
                            f"topk_indices: all {output.size} indices within "
                            f"[0, {vocab_size}); max = {int(output.max())}"
                        )
                        # The value SET is not checkable under NaN, but the value<->index
                        # PAIRING still is (see check_pairing_nan_tolerant). Run it via
                        # check_value_index_pairing so it waits for ValuesValidator to
                        # publish topk_values; whichever validator runs second does it.
                        return check_value_index_pairing(self)
                    val = np.take_along_axis(input_tensor, output.astype(np.uint64), axis=-1)
                    indices_correct = maxAllClose(
                        np.sort(golden_values, axis=-1),
                        np.sort(val, axis=-1),
                        verbose=1,
                        logfile=self.logfile,
                    )
                    # Run the pairing check even when the index SET is wrong: it
                    # distinguishes a values<->indices permutation desync from a
                    # genuinely wrong index, which is the first thing to know when
                    # debugging. Both must pass for the output to be correct.
                    paired = check_value_index_pairing(self)
                    return indices_correct and paired

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

    # ---- Spec-decoding shapes, as the KERNEL actually sees them ----------------
    # The reported failing points were quoted as global B*S = gbs*(spec_len+1), e.g.
    # bs16 -> 512. That is NOT a kernel shape: the sampler's batch_sharded_topk does
    # all_to_all(split_dim=0, concat_dim=-1) FIRST, which divides rows by dp_degree
    # and multiplies vocab by it. With TP=8 / DP=8 and vocab 201088:
    #   stage 1 (pre-all-to-all):  rows = bs*(spec_len+1),  vocab = 201088/8 = 25136
    #   stage 2 (post-all-to-all): rows = that/8,           vocab = 201088
    # Stage 2's vocab fails this kernel's own `8 <= n < 65536` assert, so stage 2 CANNOT
    # run gpsimd_topk (it falls back to rotational/torch). Only stage 1 does. So these --
    # vocab 25136, rows bs*(spec_len+1) -- are the real shapes behind the reports.
    # 25136 is a multiple of 16, so no snake padding exists here: any out-of-range index
    # isolates the unmatchable-value cause.
    topk_specdecode_stage1_perms = [
        # vocab from the failing run's own `[topk] kernel=` lines: 16384 for the local
        # stage and 3142 (= 201088/64, TP8 x DP8 shards) for the final stage. Rows are
        # the per-rank top-k row counts logged for each batch size.
        [2, 32, 1, 16384, 256, nl.bfloat16],    # bs4  local  (reported PASS post-patch)
        [2, 128, 1, 16384, 256, nl.bfloat16],   # bs4  final  (the shape that faulted 60x)
        [2, 128, 1, 3142, 256, nl.bfloat16],    # bs4  final, sharded vocab
        [2, 256, 1, 16384, 256, nl.bfloat16],   # bs8  (reported PASS post-patch)
        [2, 512, 1, 16384, 256, nl.bfloat16],   # bs16 rows
        [2, 1024, 1, 16384, 256, nl.bfloat16],  # bs32 rows
    ]

    # ---- Padded-vocab index range (the "padded" cause) -------------------------
    # When 16 does not divide vocab the snake carries full_n - vocab padding slots that
    # phase 1 memsets to BFLOAT16_MIN. Those slots sit at partition q_full = vocab //
    # n_cols, columns [vocab % n_cols, n_cols), and the snake->vocab remap sends them to
    # EXACTLY the indices [vocab, full_n) -- so a padding slot that wins the top-k
    # returns an index >= vocab, up to full_n - 1.
    #
    # Whether it wins is decided entirely by how the real data compares to the FINITE
    # sentinel, which is why the distribution matters more than the shape:
    #   data="floor" -> real data == BFLOAT16_MIN, a TIE, first occurrence wins -> real
    #                   slots win, padding is not selected, no out-of-range index.
    #   data="ninf"  -> real data is -inf < BFLOAT16_MIN strictly -> every padding slot
    #                   outranks all real data and is selected deterministically.
    # Each perm is carried at BOTH distributions so the pair localises the cause rather
    # than just reporting a failure. vocab is chosen so vocab % 16 (hence the padding
    # count full_n - vocab) varies: 3142 -> 10 pads, 1022 -> 2, 4058 -> 6, 2050 -> 14,
    # 1000 -> 8, 3151 -> 1. v=16384 is the 16 | vocab control with NO padding slots,
    # which must stay in range under either distribution.
    topk_padded_index_range_perms = [
        # (lnc, BxS, seqlen, vocab, k, dtype, data)
        # The two shapes the CR's evidence table quotes, at both distributions.
        [2, 128, 1, 3142, 256, nl.bfloat16, "floor"],
        [2, 128, 1, 3142, 256, nl.bfloat16, "ninf"],
        [2, 2, 1, 1022, 32, nl.bfloat16, "floor"],
        [2, 2, 1, 1022, 32, nl.bfloat16, "ninf"],
        # Other vocab % 16 residues: 10, 14, 8 and the single-padding-slot case.
        [2, 2, 1, 4058, 256, nl.bfloat16, "ninf"],
        [2, 4, 1, 2050, 64, nl.bfloat16, "ninf"],
        [2, 2, 1, 1000, 32, nl.bfloat16, "ninf"],
        [2, 2, 1, 3151, 256, nl.bfloat16, "ninf"],
        # 16 | vocab control: no padding exists, so this must pass either way.
        [2, 2, 1, 16384, 256, nl.bfloat16, "ninf"],
        # Padding-only attribution: k-1 distinct finite values + one padding slot, so the
        # selected set is orderable and the unmatchable-value cause cannot contribute.
        [2, 2, 1, 3151, 256, nl.bfloat16, "pad_isolate"],
        [2, 2, 1, 1023, 32, nl.bfloat16, "pad_isolate"],
    ]

    # ---- Real -inf in the DATA: value<->index pairing (the imm-collision cause) -----
    # Constrained decoding and speculative decoding mask disallowed tokens to -inf, so
    # -inf is a LEGITIMATE logit that reaches this kernel. It is also the marker
    # nc_match_replace8 writes over a consumed slot (imm=float("-inf")) in both phase-2
    # sort paths, so a real -inf in the data is indistinguishable from a slot an earlier
    # pass already claimed. match_replace8 reports the first occurrence of the matched
    # value -- a stale marker at a lower position -- and that position gathers the paired
    # index. The index stays IN RANGE, so an index-range check cannot see it; only the
    # elementwise value<->index pairing check can.
    #
    # The perms split the surface into the two independent mechanisms:
    #   16 | vocab  (1024, 16384): NO snake padding exists, so a failure here is
    #       attributable to the phase-2 imm collision alone.
    #   16 !| vocab (3142, 1022):  the phase-1 snake padding sentinel BFLOAT16_MIN is
    #       FINITE and therefore strictly outranks real -inf, so nisa.topk itself pulls a
    #       padding slot into the top-k before phase 2 ever runs. That is a SECOND,
    #       independent defect; these perms carry both.
    # BxS is varied so both phase-2 sort paths are covered: BxS<=128 -> per_lnc<=64 ->
    # SPLIT path; BxS=256 -> full-width path.
    topk_neg_inf_perms = [
        # (lnc, BxS, seqlen, vocab, k, dtype)
        [2, 16, 1, 1024, 32, nl.bfloat16],     # 16 | vocab, split path  (no-padding control)
        [2, 16, 1, 16384, 256, nl.bfloat16],   # 16 | vocab, split path  (no-padding control)
        [2, 16, 1, 3142, 256, nl.bfloat16],    # vocab % 16 = 6, split path
        [2, 128, 1, 3142, 256, nl.bfloat16],   # vocab % 16 = 6, split path, per_lnc=64
        [2, 2, 1, 1022, 32, nl.bfloat16],      # vocab % 16 = 14, split path, par_dim=16
    ]

    # NaN pairing probe: is the value<->index pairing ALSO broken for NaN input? The
    # garbage/uninit suites deliberately skip the pairing check (with NaN there is no
    # well-defined k-largest), so this is currently UNKNOWN rather than known-clean.
    # These perms run pairing_only: no golden/value comparison (meaningless with NaN),
    # just input[i, indices[i, j]] == values[i, j] with NaN==NaN treated as paired.
    topk_nan_pairing_perms = [
        [2, 16, 1, 1024, 32, nl.bfloat16],
        [2, 16, 1, 16384, 256, nl.bfloat16],
        [2, 16, 1, 3142, 256, nl.bfloat16],
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

    @pytest_parametrize(topk_unit_params, topk_specdecode_stage1_perms, abbrevs=_ABBREVS)
    def test_gpsimd_topk_specdecode_stage1_index_range(
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
        """Indices stay in [0, vocab) at the real spec-decoding stage-1 shapes.

        vocab 25136 is the per-TP-rank vocab shard for GPT-OSS-120b (201088/8), which is
        what the local top-k of ``batch_sharded_topk`` actually reduces over; rows are
        ``bs*(spec_len+1)``. See the note on ``topk_specdecode_stage1_perms`` for why the
        globally-quoted B*S is not the kernel's shape.

        HARDWARE ONLY -- the simulator returns in-range indices even with the fix
        removed, so a green simulation run proves nothing here.
        """
        self.run_topk_test(
            test_manager=test_manager,
            platform_target=platform_target,
            lnc_degree=lnc_degree,
            batch=batch,
            seqlen=seqlen,
            vocab=vocab_size,
            k=K,
            dtype=dtype,
            data="uninit",
            index_bounds_only=True,
        )

    @pytest_parametrize(
        topk_unit_params + ", data",
        topk_padded_index_range_perms,
        abbrevs={**_ABBREVS, "data": "d"},
    )
    def test_gpsimd_topk_padded_index_range(
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
        data,
    ):
        """Indices stay in [0, vocab) when 16 does not divide vocab (padded snake).

        See ``topk_padded_index_range_perms`` for why the padding slots remap to exactly
        [vocab, full_n) and why the distribution decides whether one wins the top-k.

        The values here are orderable (a floor plus finite winners, or a uniform floor),
        so the FULL validator set runs -- value/golden comparison, descending order, and
        the value<->index pairing check. Nothing is skipped: a padding slot that wins is
        visible both as an out-of-range index and as a value the golden does not contain.

        HARDWARE ONLY -- the simulator does not reproduce these defects.
        """
        # A uniformly -inf row makes the GOLDEN top-k all -inf, and the shared comparator
        # reduces over abs(golden)[isfinite(...)] to pick its scale -- an empty array,
        # which raises "zero-size array to reduction operation maximum". That is a
        # limitation of the value-SET comparator on an all-non-finite golden, not a
        # property of the kernel, and it would mask the result being measured here. For
        # those perms assert the index contract directly (range + value<->index pairing,
        # both of which remain well defined); the "floor" and "pad_isolate" perms keep a
        # finite golden and therefore run the full value/order/pairing validator set.
        bounds_only = data == "ninf"
        self.run_topk_test(
            test_manager=test_manager,
            platform_target=platform_target,
            lnc_degree=lnc_degree,
            batch=batch,
            seqlen=seqlen,
            vocab=vocab_size,
            k=K,
            dtype=dtype,
            data=data,
            index_bounds_only=bounds_only,
        )

    @pytest_parametrize(topk_unit_params, topk_neg_inf_perms, abbrevs=_ABBREVS)
    def test_gpsimd_topk_neg_inf_masked(
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
        """Real -inf in the DATA must not desync the value<->index pairing.

        Constrained decoding and speculative decoding mask disallowed tokens to -inf, so
        -inf is a legitimate logit here -- the input is orderable and the correct answer
        is well defined. See ``topk_neg_inf_perms`` for the mechanism: -inf is also the
        marker nc_match_replace8 writes over a consumed slot, so a real -inf in the data
        is indistinguishable from an already-consumed slot and the paired index is
        gathered from the wrong position. The index stays IN RANGE, so only the
        elementwise pairing check sees it.

        The full validator set runs (values vs golden, descending order, pairing): the
        selected values are orderable, so a mis-paired index is a real defect and not an
        artifact of an ambiguous answer.

        HARDWARE ONLY -- the simulator returns correct, in-range, correctly-paired
        results with the defect present, so a green simulation run proves nothing.
        """
        self.run_topk_test(
            test_manager=test_manager,
            platform_target=platform_target,
            lnc_degree=lnc_degree,
            batch=batch,
            seqlen=seqlen,
            vocab=vocab_size,
            k=K,
            dtype=dtype,
            data="neg_inf_masked",
        )

    @pytest_parametrize(topk_unit_params, topk_neg_inf_perms, abbrevs=_ABBREVS)
    def test_gpsimd_topk_dup_finite(
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
        """Diagnostic control for test_gpsimd_topk_neg_inf_masked: duplicates alone are fine.

        Row 0 is ONE repeated finite value (1.5), so the selected set is maximally
        duplicated but holds no -inf and no NaN. nc_match_replace8 has a defined
        duplicate-vals contract (it processes vals in reverse order so each duplicate gets
        a distinct dst_idx column), so this PASSES -- which is what pins the -inf failure
        on the imm/data value-domain collision rather than on duplication in general.
        This test must keep passing alongside any fix.
        """
        self.run_topk_test(
            test_manager=test_manager,
            platform_target=platform_target,
            lnc_degree=lnc_degree,
            batch=batch,
            seqlen=seqlen,
            vocab=vocab_size,
            k=K,
            dtype=dtype,
            data="dup_finite",
        )

    @pytest_parametrize(topk_unit_params, topk_nan_pairing_perms, abbrevs=_ABBREVS)
    def test_gpsimd_topk_nan_pairing(
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
        """Is the value<->index pairing ALSO broken for NaN input?

        The garbage/uninit suites skip the pairing check entirely, so this is currently
        unknown rather than known-clean. NaN never compares equal to itself, so
        nc_match_replace8 can never match a NaN that max8 selected and its reported
        position is undefined by the ISA contract -- a mechanism distinct from the
        -inf/imm collision. index_bounds_only drops the meaningless golden value-SET
        comparison but keeps the range check and the NaN-tolerant pairing check.

        HARDWARE ONLY -- the simulator does not reproduce this.
        """
        self.run_topk_test(
            test_manager=test_manager,
            platform_target=platform_target,
            lnc_degree=lnc_degree,
            batch=batch,
            seqlen=seqlen,
            vocab=vocab_size,
            k=K,
            dtype=dtype,
            data="nan_masked",
            index_bounds_only=True,
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
