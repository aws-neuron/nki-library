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
"""Integration tests for sequence-packing tile pruning in attention_cte (``segment_cu_seqlens``).

``segment_cu_seqlens`` is a trace-time copy of the packed layout that bound_min/bound_max describe
on the device. With it, attention_cte skips K tiles that no row of a Q group can attend to and
(Q group, flash section) pairs with no work at all. The skipped work only contributes zeros, so
results match the unpruned call up to accumulation-order rounding.

The configurations are chosen to exercise what pruning interacts with:
  * >= 4 documents and >= 16 Q groups, so pruned tiles land in pipeline buffers that a
    neighbouring Q group also uses;
  * document boundaries that are not multiples of 128 / 512, so some tiles are only partly live;
  * seqlen > 8192, so a Q group's first live flash section is not section 0;
  * a single-document control, where nothing is prunable.

Every case is validated against the torch reference.
"""

from typing import final

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.attention.attention_cte import attention_cte
from nkilib_src.nkilib.core.attention.attention_cte_torch import attention_cte_torch_ref

from test.integration.nkilib.utils.tensor_generators import np_random_sample
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _cu_seqlens(seqlen, doc_lens):
    """Cumulative document boundaries. ``doc_lens`` is an int (uniform) or an explicit list."""
    if isinstance(doc_lens, int):
        assert seqlen % doc_lens == 0
        return tuple(range(0, seqlen + 1, doc_lens))
    cu = [0]
    for n in doc_lens:
        cu.append(cu[-1] + n)
    assert cu[-1] == seqlen, f"document lengths sum to {cu[-1]}, expected {seqlen}"
    return tuple(cu)


def _bounds(cu, bs, seqlen):
    lo = np.zeros((bs, seqlen, 1), dtype=np.float32)
    hi = np.zeros((bs, seqlen, 1), dtype=np.float32)
    for s, e in zip(cu[:-1], cu[1:], strict=True):
        lo[:, s:e, 0] = s
        hi[:, s:e, 0] = e
    return lo, hi


def _output_tensors(kernel_input):
    q = kernel_input["q"]
    return {"out": np.ndarray(shape=q.shape, dtype=q.dtype)}


# fmt: off
seqpack_prune_params = "lnc_degree, batch, seqlen, d, dtype, causal_mask, doc_lens"
seqpack_prune_perms = [
    # Single flash section (seqlen <= 8192)
    pytest.param(2, 1, 2048, 128, nl.bfloat16, False, [700, 500, 600, 248], marks=pytest.mark.fast),
    pytest.param(2, 1, 4096, 128, nl.bfloat16, False, 1024, marks=pytest.mark.fast),
    pytest.param(2, 1, 4096, 128, nl.bfloat16, True,  1024, marks=pytest.mark.fast),
    [1, 1, 4096, 128, nl.bfloat16, False, 512],
    [2, 1, 4096, 128, nl.bfloat16, True,  [1500, 1300, 796, 500]],
    [2, 1, 4096, 128, np.float32,  False, [300, 900, 129, 1700, 1067]],
    [2, 2, 2048, 96,  nl.bfloat16, False, 512],
    [2, 1, 8192, 128, nl.bfloat16, False, 1024],
    [2, 1, 4096, 128, nl.bfloat16, False, 4096],  # control: one document, nothing prunable
    # Multiple flash sections (seqlen > 8192)
    [2, 1, 16384, 128, nl.bfloat16, False, 1024],
    [2, 1, 16384, 128, nl.bfloat16, True,  1024],
    [2, 1, 16384, 128, nl.bfloat16, False, [3000, 6000, 1500, 2884, 3000]],
    [2, 1, 24576, 128, nl.bfloat16, False, 2048],
]
# fmt: on


@pytest_test_metadata(name="Attention CTE sequence-packing tile pruning", tags=["model"])
@pytest_marks(["attention", "cte"])
@final
class TestAttentionCteSeqpackPrune:
    @pytest_parametrize(seqpack_prune_params, seqpack_prune_perms)
    def test_attention_cte_seqpack_prune(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        lnc_degree,
        batch,
        seqlen,
        d,
        dtype,
        causal_mask,
        doc_lens,
    ):
        np.random.seed(42)
        cu = _cu_seqlens(seqlen, doc_lens)

        def input_generator(test_config):
            random_gen = np_random_sample()
            bound_min, bound_max = _bounds(cu, batch, seqlen)
            return {
                "q": random_gen(shape=(batch, seqlen, d), dtype=dtype),
                "k": random_gen(shape=(batch, d, seqlen), dtype=dtype),
                "v": random_gen(shape=(batch, seqlen, d), dtype=dtype),
                "scale": 1.0,
                "causal_mask": causal_mask,
                "tp_q": True,
                "tp_k": False,
                "tp_out": False,
                "bound_min": bound_min,
                "bound_max": bound_max,
                "segment_cu_seqlens": cu,
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=attention_cte,
            torch_ref=torch_ref_wrapper(attention_cte_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc_degree, platform_target=platform_target),
            rtol=2e-2,
            atol=1e-5,
        )
