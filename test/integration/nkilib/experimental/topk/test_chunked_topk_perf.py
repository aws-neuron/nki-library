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

"""Head-to-head profiling: chunked_topk FUSED vs the SPLIT two-launch baseline.

The question these cases answer is what fusing the two top-k stages into one kernel buys
over running them as two separate kernel launches, which is what a caller would otherwise
have to do (the vocab exceeds nisa.topk's ceiling, so SOMETHING has to be two-stage).

    FUSED  : chunked_topk on [BxS, 201088]                        -- one NEFF, one launch
    SPLIT  : gpsimd_topk on [BxS*P, 201088/P]   (stage 1)         -- launch 1
             gpsimd_topk on [max(BxS,8), P*256] (reduce)          -- launch 2

Compare ``InferenceTime`` of FUSED against the SUM of the two SPLIT launches. The split
pair is deliberately given the EXACT shapes the fused kernel runs internally, so the
difference is fusion overhead only -- not a shape mismatch:

    BxS=1 (gbs=128 per rank, P=8):  stage1 [8, 25136]  reduce [8, 2048]
    BxS=8 (gbs=1024,        P=4):  stage1 [32, 50272]  reduce [8, 1024]

The split baseline UNDERSTATES its own cost: it omits the index fix-up between the stages
and, more importantly, a real caller would also pay a second framework dispatch and an
extra HBM round trip for the candidate tensors, which a single-launch profile folds in.

Stage 1 always runs unsorted (see chunked_topk's STAGE 1 IS NEVER SORTED section), and the
split stage-1 baseline runs sorted=False to match. The REDUCE leg runs sorted=False too,
for the mirror-image reason: gpsimd_topk's sorted=True is served by its SOFTWARE sort,
while the fused reduce's sorted output is the hardware sort -- so a sorted split leg would
contaminate the comparison with a sort-algorithm swap in the other direction. The fused
kernel is profiled at BOTH sort settings: hold the UNSORTED fused arm against the split
sum (fusion machinery only), and the sorted arm against the unsorted arm (the hardware
sort's own cost, in the production configuration).

Validation here is deliberately light (value multiset only) -- test_chunked_topk.py is the
correctness suite. These cases exist to produce comparable profiles.
"""

from typing import Any, final

import ml_dtypes
import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.experimental.topk.chunked_topk import (
    _HW_SORT_AVAILABLE,
    chunked_topk,
    create_chunked_topk_config,
    pick_chunks,
)
from nkilib_src.nkilib.experimental.topk.chunked_topk_torch import chunked_topk_torch_ref
from nkilib_src.nkilib.experimental.topk.gpsimd_topk import (
    create_gpsimd_topk_config,
    gpsimd_topk,
)
from nkilib_src.nkilib.experimental.topk.gpsimd_topk_torch import gpsimd_topk_torch_ref
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
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

_GPTOSS_VOCAB = 201088
_K = 256
# nl dtypes type as plain strings, and the config factories annotate inp_dtype as
# np.dtype (the convention across the topk kernels). Parametrised callers pass opaque
# variables, so only LITERAL call sites trip the type checker in the release build;
# route them through this Any-typed constant.
_BF16: Any = nl.bfloat16
# Warmups inside the NEFF execution loop; only the LAST run is profiled, so one-time
# setup lands in a discarded run instead of the measured one.
_NUM_RUNS = 5


def _np_dtype(dtype):
    return ml_dtypes.bfloat16 if dtype == nl.bfloat16 else dtype


def _randn(rows: int, cols: int, dtype) -> npt.NDArray[Any]:
    np.random.seed(42)
    return (np.random.randn(rows, cols) * 0.1).astype(_np_dtype(dtype))


def _value_multiset_comparator(rows: int, k: int, dtype):
    """Values-only gate: the SET must match, internal order is forgiven.

    bf16 ties make the index choice undetermined at this width, and the correctness suite
    already carries the pairing and distinctness gates. Here the only job is to confirm the
    run really computed a top-k before its timing is believed.
    """

    def comparator(golden_dict, output_tensors):
        golden = np.asarray(golden_dict["topk_values"], dtype=np.float32)

        class ValuesValidator(CustomValidator):
            @override
            def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                got = np.asarray(
                    np.frombuffer(inference_output, dtype=_np_dtype(dtype)).reshape(rows, k), dtype=np.float32
                )
                ok = np.allclose(np.sort(got, axis=-1), np.sort(golden, axis=-1), rtol=1e-3, atol=1e-5)
                if not ok:
                    self._print_with_log(
                        f"FAIL values: sorted set differs. got_top6={np.sort(got[0])[-6:].tolist()} "
                        f"golden_top6={np.sort(golden[0])[-6:].tolist()}"
                    )
                return ok

        class PassThroughValidator(CustomValidator):
            @override
            def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                # Indices are gated by test_chunked_topk.py; not re-checked here.
                return True

        return {
            "topk_values": CustomValidatorWithOutputTensorData(
                validator=ValuesValidator, output_ndarray=output_tensors["topk_values"]
            ),
            "topk_indices": CustomValidatorWithOutputTensorData(
                validator=PassThroughValidator, output_ndarray=output_tensors["topk_indices"]
            ),
        }

    return comparator


def _out_desc(kernel_input):
    inp = kernel_input["inp"]
    config = kernel_input["config"]
    rows, _ = inp.shape
    return {
        "topk_values": np.zeros((rows, config.k), dtype=_np_dtype(config.inp_dtype)),
        "topk_indices": np.zeros((rows, config.k), dtype=np.uint32),
    }


@pytest_test_metadata(name="Chunked TopK Perf")
@pytest_marks(["topk", "gpsimd", "chunked", "perf"])
@final
@pytest.mark.skipif(not _HW_SORT_AVAILABLE, reason="hardware sort unavailable; chunked_topk refuses sorted=True")
class TestChunkedTopkPerf:
    """One profile per case. Fused vs split at the two production row counts."""

    def _run(self, test_manager, platform_target, kernel, torch_ref, inp, config, rows):
        def input_generator(test_config, input_tensor_def=None):
            return {"inp": inp, "config": config}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref_wrapper(torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_out_desc,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=1, platform_target=platform_target),
            rtol=1e-3,
            atol=1e-5,
            inference_args=InferenceArgs(num_runs=_NUM_RUNS),
            custom_comparator=_value_multiset_comparator(rows, config.k, nl.bfloat16),
        )

    # ---- FUSED: both stages inside one kernel, one launch ------------------------------
    @pytest.mark.parametrize("bxs", [1, 8])
    @pytest.mark.parametrize("out_sorted", [True, False], ids=["sorted", "unsorted"])
    def test_perf_fused(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        bxs: int,
        out_sorted: bool,
    ):
        """chunked_topk over the full 201,088 vocab. P auto-picked (8 at bxs=1, 4 at bxs=8).

        Both sort settings are profiled, for different comparisons: ``sorted`` is the
        production config; ``unsorted`` is the arm to hold against the split baseline,
        because the split reduce leg runs unsorted (gpsimd_topk's sorted=True is served by
        its SOFTWARE sort, while the fused reduce uses the hardware sort -- comparing across
        that swap would credit "fusion" with the hw-vs-sw sort difference). unsorted-fused
        vs split isolates the fusion machinery; sorted-fused vs unsorted-fused isolates the
        hardware sort's own cost.
        """
        config = create_chunked_topk_config(inp_shape=(bxs, _GPTOSS_VOCAB), inp_dtype=_BF16, k=_K, sorted=out_sorted)
        self._run(
            test_manager,
            platform_target,
            chunked_topk,
            chunked_topk_torch_ref,
            _randn(bxs, _GPTOSS_VOCAB, nl.bfloat16),
            config,
            bxs,
        )

    # ---- SPLIT baseline, launch 1: the per-chunk top-k ---------------------------------
    @pytest.mark.parametrize("bxs", [1, 8])
    def test_perf_split_stage1(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        bxs: int,
    ):
        """gpsimd_topk on exactly the shape the fused kernel's stage 1 sees.

        sorted=False, matching the fused kernel's stage 1 (which is UNCONDITIONALLY
        unsorted -- its ordering is unobservable). A sorted baseline pays a descending
        sort the fused path never executes (14-21.6% of stage-1 time), which inflates
        every fused-vs-split comparison by that much.
        """
        p = pick_chunks(bxs, _GPTOSS_VOCAB, _K)
        rows, width = bxs * p, _GPTOSS_VOCAB // p
        config = create_gpsimd_topk_config(inp_shape=(rows, width), inp_dtype=_BF16, k=_K, sorted=False, num_programs=1)
        self._run(
            test_manager,
            platform_target,
            gpsimd_topk,
            gpsimd_topk_torch_ref,
            _randn(rows, width, nl.bfloat16),
            config,
            rows,
        )

    # ---- SPLIT baseline, launch 2: the reduce ------------------------------------------
    @pytest.mark.parametrize("bxs", [1, 8])
    def test_perf_split_reduce(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        bxs: int,
    ):
        """gpsimd_topk on exactly the shape the fused kernel's reduce sees.

        Rows are ``max(bxs, 8)`` because the fused kernel pads the reduce up to one whole
        128-partition tile; the baseline must pay that too or the comparison is unfair.

        sorted=False, same reasoning as the stage-1 leg: gpsimd_topk's sorted=True is its
        SOFTWARE sort, which the fused reduce (hardware sort) never executes, so a sorted
        baseline contaminates the fusion comparison with a sort-algorithm swap. Hold this
        against the UNSORTED fused arm; the sorted fused arm carries the hw-sort cost
        separately.
        """
        p = pick_chunks(bxs, _GPTOSS_VOCAB, _K)
        rows, width = max(bxs, 8), p * _K
        config = create_gpsimd_topk_config(inp_shape=(rows, width), inp_dtype=_BF16, k=_K, sorted=False, num_programs=1)
        self._run(
            test_manager,
            platform_target,
            gpsimd_topk,
            gpsimd_topk_torch_ref,
            _randn(rows, width, nl.bfloat16),
            config,
            rows,
        )
