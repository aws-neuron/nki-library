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

"""Rotational top-k baseline at the GPT-OSS-120b MXFP4 decode shapes.

This file exists only to measure the existing DVE ``rotational_topk`` kernel at
the exact shapes the gpt-oss-120b decode path drives, so the GpSIMD
``nisa.topk`` kernel (``test_gpsimd_topk.py::test_gpsimd_topk_gptoss``) has a
head-to-head reference. It reuses ``TestTopKKernel.run_topk_test`` from
``test_topk.py`` unchanged (no new validator logic).

Two dtypes are measured per shape:

* ``float32`` — what the model actually runs today (the distributed ``topk()``
  upcasts the bf16 sampling logits to fp32 for the DVE kernel's accuracy).
* ``bfloat16`` — apples-to-apples vs the bf16-only GpSIMD kernel.

The 6 shapes are the local + global ``rotational_topk`` calls per decode step at
global batch sizes 128 / 512 / 1024 (n_rows = GBS / 8); k=256, n_prgs=2,
vocab 3142 (local, = 201088 / 64) and 16384 (global, = 256 * 64).
"""

from typing import final

import nki.language as nl
import pytest

# Alias with a leading underscore so pytest does NOT re-collect TestTopKKernel in this
# module. A bare `import TestTopKKernel` makes pytest collect the entire test_topk_unit
# suite a second time under this module; those duplicate nodes derive the same local
# output directory (derive_pytest_test_id strips the module path), so under pytest-xdist
# they race on one directory and clobber each other's file.neff, producing an intermittent
# "NEFF file not found" compile failure. We only need the class to reuse run_topk_test.
from test.integration.nkilib.core.topk.test_topk import TestTopKKernel as _TestTopKKernel
from test.utils.common_dataclasses import Platforms
from test.utils.metrics_collector import MetricsCollector
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator


@pytest_test_metadata(name="Rotational TopK GPT-OSS Baseline")
@pytest_marks(["topk", "rotational", "gptoss"])
@final
class TestRotationalTopKGptOssBaseline:
    """Rotational baseline measurements at GPT-OSS decode shapes (fp32 + bf16)."""

    _params = "lnc_degree, batch, seqlen, vocab_size, K, dtype"
    _ABBREVS = {"lnc_degree": "lnc", "batch": "b", "seqlen": "s", "vocab_size": "v", "K": "K", "dtype": "dt"}

    # fmt: off
    # 6 GPT-OSS shapes x 2 dtypes. n_rows = GBS/8 (16/64/128); local vocab 3142,
    # global vocab 16384; k=256, n_prgs(lnc)=2.
    _perms = [
        # --- float32: what the model runs today ---
        # GBS=128 -> n_rows=16
        [2, 16, 1, 3142, 256, nl.float32],     # local
        [2, 16, 1, 16384, 256, nl.float32],    # global
        # GBS=512 -> n_rows=64
        [2, 64, 1, 3142, 256, nl.float32],     # local
        [2, 64, 1, 16384, 256, nl.float32],    # global
        # GBS=1024 -> n_rows=128
        [2, 128, 1, 3142, 256, nl.float32],    # local
        [2, 128, 1, 16384, 256, nl.float32],   # global

        # --- bfloat16: apples-to-apples vs the bf16-only GpSIMD kernel ---
        pytest.param(2, 16, 1, 3142, 256, nl.bfloat16, marks=pytest.mark.fast),  # local
        [2, 16, 1, 16384, 256, nl.bfloat16],   # global
        [2, 64, 1, 3142, 256, nl.bfloat16],    # local
        [2, 64, 1, 16384, 256, nl.bfloat16],   # global
        [2, 128, 1, 3142, 256, nl.bfloat16],   # local
        [2, 128, 1, 16384, 256, nl.bfloat16],  # global
    ]
    # fmt: on

    @pytest_parametrize(_params, _perms, abbrevs=_ABBREVS)
    def test_rotational_gptoss(
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
        """Rotational top-k at a GPT-OSS decode shape (baseline for the GpSIMD A/B)."""
        _TestTopKKernel().run_topk_test(
            test_manager=test_manager,
            platform_target=platform_target,
            lnc_degree=lnc_degree,
            batch=batch,
            seqlen=seqlen,
            vocab=vocab_size,
            k=K,
            dtype=dtype,
        )
