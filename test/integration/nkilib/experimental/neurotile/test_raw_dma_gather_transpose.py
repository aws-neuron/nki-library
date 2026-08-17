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

"""Raw-NKI probe for nisa.dma_transpose indirect (gather) transpose.

Confirms the documented dma_gather_transpose semantics independently of
neurotile, to isolate whether neurotile's indirect .load(transpose=True) bug is
a missing axes=/index-routing in its plumbing vs a raw-NKI limitation. Mirrors
the nki.isa.dma_transpose docstring example (1D indices) exactly:
    flat = indices.T.flatten()[:src.shape[0]]; dst = src[flat, :].T
"""

import ml_dtypes
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import torch

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Gather-transpose contract (nki.isa.dma_transpose, indirect src):
#   flat = indices.T.flatten()[:src.shape[0]]; gathered = src[flat]; dst = gathered.T
# Constraints: src.shape[0] % 16 == 0; indices.shape[0] in [16,128] & % 16 == 0;
#   indices.shape[0]*indices.shape[1] >= src.shape[0]; src.shape[-1] <= 128; 2-byte.
# Square-only example in the docs leaves the AP-pattern orientation ambiguous;
# this sweeps candidate (pattern, dst) forms to pin the correct one on HARDWARE.


# src=(N, D); gather all N rows (src.shape[0]==N==indices count); dst=(D, N).
# pat_DN ([[D, N], [1, D]]) is the hardware-correct AP orientation. Shapes are
# kernel-body literals -- the NKI tracer can't close over free Python vars.
_N, _D = 16, 64


@nki.jit
def _gxt_pat_DN(src_hbm, idx_hbm):
    """AP pattern [[D, N], [1, D]] (rows-gathered count outer)."""
    out = nl.ndarray((_D, _N), dtype=src_hbm.dtype, buffer=nl.shared_hbm)
    idx_sb = nl.load(idx_hbm)
    dst = nl.ndarray((_D, _N), dtype=src_hbm.dtype, buffer=nl.sbuf)
    nisa.memset(dst=dst, value=0)
    ap = src_hbm.ap(pattern=[[_D, _N], [1, _D]], vector_offset=idx_sb, indirect_dim=0)
    nisa.dma_transpose(dst=dst, src=ap, axes=(1, 0))
    nisa.dma_copy(dst=out, src=dst)
    return out


def _gxt_inputs(_):
    np.random.seed(42)
    src = np.random.randn(_N, _D).astype(ml_dtypes.bfloat16)
    idx = np.random.permutation(_N).astype(np.uint32).reshape(_N, 1)
    return {"src_hbm": src, "idx_hbm": idx}


def _gxt_output(_):
    return {"out": np.zeros((_D, _N), dtype=ml_dtypes.bfloat16)}


def _gxt_ref(src_hbm: torch.Tensor, idx_hbm: torch.Tensor) -> torch.Tensor:
    flat = idx_hbm.t().flatten()[:_N]
    return src_hbm[flat.long(), :].t().contiguous()


# Note: oob_mode.skip on the 2-D gather-transpose whole-skips on trn2 (any OOB index
# zeroes the entire gather, not per-row), verified by device dump for both -1 sentinels
# and positive-OOB. Per-row OOB skip is tied to the N-D form -- see gaps doc Appendix D.


@pytest_marks(["neurotile"])
class TestRawDmaGatherTranspose:
    """Raw nisa.dma_transpose gather-transpose, per the nki.isa docstring.
    pat_DN is the hardware-correct AP orientation (found via sweep). Confirms
    neurotile's gather-transpose matches the raw NKI capability on hardware."""

    def test_gather_transpose_1d_indices(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_gxt_pat_DN,
            torch_ref=torch_ref_wrapper(_gxt_ref),
            kernel_input_generator=_gxt_inputs,
            output_tensor_descriptor=_gxt_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )
