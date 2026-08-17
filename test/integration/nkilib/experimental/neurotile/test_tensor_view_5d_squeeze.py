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

"""Real-DMA validation of 5-D squeeze / flatten+reshape transform chains on a
raw NkiTensor (the source feeds nisa.dma_copy directly -- no nt.tensor_view).

The reported failure was that a 5-D transform chain used as a DMA source raised
"dma_copy requires src and dst to have the same number of elements". These
kernels exercise that exact call path through the real compiler, so the AP
element-count check actually runs.

Kernel observable behavior is the loaded tile copied back to HBM, so the torch
reference is just the gathered sub-region.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
import torch

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Filter dims (shrunk C_in/C_out vs the issue's 1024 so the test stays cheap;
# the squeeze/stride-threading path is identical regardless of extent).
KD, KH, KW, CIN, COUT = 3, 3, 3, 256, 256
K_D_IDX, K_H_IDX, K_W_IDX = 1, 2, 0
CIN0, CIN1 = 0, 128  # C_in tile -> P = 128
COUT0, COUT1 = 0, 256  # C_out tile -> F = 256

# Output-store dims: D*H*W=504, W=12, D*H=42 so the correct partition stride
# (504) is distinct from the contiguous-of-logical value (12*12=144).
B, C_OUT, D, H, W = 2, 128, 6, 7, 12
BATCH_IDX = 1
COUT_TILE = 128
DH0, DH1 = 0, 12
W0, W1 = 0, 12


@nki.jit
def _filter_load_kernel(filters):
    """Triple slice+squeeze on a 5-D filter, then DMA the [C_in, C_out] tile."""
    dst = nl.ndarray((CIN1 - CIN0, COUT1 - COUT0), dtype=filters.dtype, buffer=nl.shared_hbm)

    filter_view = (
        filters.slice(0, K_D_IDX, K_D_IDX + 1)
        .squeeze_dim(0)  # [KH, KW, CIN, COUT]
        .slice(0, K_H_IDX, K_H_IDX + 1)
        .squeeze_dim(0)  # [KW, CIN, COUT]
        .slice(0, K_W_IDX, K_W_IDX + 1)
        .squeeze_dim(0)  # [CIN, COUT]
        .slice(0, CIN0, CIN1)
        .slice(1, COUT0, COUT1)
    )
    sbuf_buf = nl.ndarray((CIN1 - CIN0, COUT1 - COUT0), dtype=filters.dtype, buffer=nl.sbuf)
    # The exact call from the bug report:
    nisa.dma_copy(dst=sbuf_buf, src=filter_view)
    nisa.dma_copy(dst=dst, src=sbuf_buf)
    return dst


@nki.jit
def _output_store_kernel(y_in):
    """flatten+reshape+slice on a 5-D tensor; load the [Cout, dh, w] tile."""
    dst = nl.ndarray((COUT_TILE, DH1 - DH0, W1 - W0), dtype=y_in.dtype, buffer=nl.shared_hbm)

    y_view = (
        y_in.slice(0, BATCH_IDX, BATCH_IDX + 1)
        .squeeze_dim(0)  # [C_out, D, H, W]
        .slice(0, 0, COUT_TILE)  # [128, D, H, W]
        .flatten_dims(1, 3)  # [128, D*H*W]
        .reshape_dim(1, (D * H, W))  # [128, 42, 12]
        .slice(1, DH0, DH1)
        .slice(2, W0, W1)
    )
    sbuf_buf = nl.ndarray((COUT_TILE, DH1 - DH0, W1 - W0), dtype=y_in.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sbuf_buf, src=y_view)
    nisa.dma_copy(dst=dst, src=sbuf_buf)
    return dst


def _filter_inputs(_):
    np.random.seed(42)
    import ml_dtypes

    return {"filters": np.random.randn(KD, KH, KW, CIN, COUT).astype(ml_dtypes.bfloat16)}


def _filter_output(kernel_input):
    return {"out": np.zeros((CIN1 - CIN0, COUT1 - COUT0), dtype=kernel_input["filters"].dtype)}


def _filter_ref(filters: torch.Tensor) -> torch.Tensor:
    return filters[K_D_IDX, K_H_IDX, K_W_IDX, CIN0:CIN1, COUT0:COUT1].clone()


def _output_inputs(_):
    np.random.seed(42)
    import ml_dtypes

    return {"y_in": np.random.randn(B, C_OUT, D, H, W).astype(ml_dtypes.bfloat16)}


def _output_output(kernel_input):
    return {"out": np.zeros((COUT_TILE, DH1 - DH0, W1 - W0), dtype=kernel_input["y_in"].dtype)}


def _output_ref(y_in: torch.Tensor) -> torch.Tensor:
    sub = y_in[BATCH_IDX, 0:COUT_TILE]  # [C_out, D, H, W]
    flat = sub.reshape(COUT_TILE, D * H * W)  # [128, 504]
    re = flat.reshape(COUT_TILE, D * H, W)  # [128, 42, 12]
    return re[:, DH0:DH1, W0:W1].clone()


@pytest_marks(["neurotile"])
class TestTensorView5DSqueeze:
    """nt.tensor_view 5-D chains emit a single valid DMA (no element-count error)."""

    @pytest.mark.fast
    def test_filter_triple_squeeze_load(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_filter_load_kernel,
            torch_ref=torch_ref_wrapper(_filter_ref),
            kernel_input_generator=_filter_inputs,
            output_tensor_descriptor=_filter_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.fast
    def test_output_flatten_reshape_slice_store(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_output_store_kernel,
            torch_ref=torch_ref_wrapper(_output_ref),
            kernel_input_generator=_output_inputs,
            output_tensor_descriptor=_output_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )
