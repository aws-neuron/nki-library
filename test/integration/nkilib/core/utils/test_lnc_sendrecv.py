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

"""Simulator-mode branch-coverage tests for core/utils/lnc_sendrecv.

``lnc_sendrecv`` picks ``gpsimd_dma`` for small SBUF-to-SBUF exchanges and
falls back to regular DMA otherwise. Both paths need a real LNC-2 sendrecv, so
the tests drive a tiny two-core @nki.jit kernel that swaps each core's tile
with the other core's and run it via the simulator at ``logical_nc_config=2``.

Branch arcs exercised:
  - L41 (``for dim_size in src.shape[1:]``): both arcs -- a 2-D ``src`` runs the
    loop body and then exits, so every test covers both.
  - L44 (``allow_gpsimd and free_elems <= _GPSIMD_HW_LIMIT``): the gpsimd test
    takes the true arc (L45); the fallback test forces ``allow_gpsimd=False`` to
    take the false arc (L54).
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.core.utils.lnc_sendrecv import lnc_sendrecv

from test.utils.common_dataclasses import CompilerArgs
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Small enough to keep the gpsimd path (free_elems <= 256) and the sim fast.
_PARTITIONS = 4
_FREE = 2


@nki.jit
def kernel_lnc_sendrecv(input_tensor, allow_gpsimd):
    """Two cores exchange their tile via lnc_sendrecv; output is the swap."""
    out = nl.ndarray(input_tensor.shape, dtype=input_tensor.dtype, buffer=nl.shared_hbm)
    my_tile = nl.ndarray(input_tensor.shape[1:], dtype=input_tensor.dtype, buffer=nl.sbuf)
    recv_tile = nl.ndarray(input_tensor.shape[1:], dtype=input_tensor.dtype, buffer=nl.sbuf)

    i = nl.program_id(0)
    other = 1 - i

    nisa.dma_copy(my_tile, input_tensor[i])
    lnc_sendrecv(
        src=my_tile,
        dst=recv_tile,
        send_to_rank=other,
        recv_from_rank=other,
        allow_gpsimd=allow_gpsimd,
    )
    nisa.dma_copy(out[i], recv_tile)
    return out


def _sendrecv_torch_ref(input_tensor, allow_gpsimd):
    """Each core receives the other core's tile -> the two halves are swapped."""
    return {"out": torch.flip(input_tensor, dims=[0])}


def _input_generator(allow_gpsimd):
    def _gen(_test_config):
        np.random.seed(42)
        data = np.random.randn(2, _PARTITIONS, _FREE).astype(np.float32)
        return {"input_tensor": data, "allow_gpsimd": allow_gpsimd}

    return _gen


def _output_tensors(_kernel_input):
    return {"out": np.zeros((2, _PARTITIONS, _FREE), dtype=np.float32)}


@pytest_test_metadata(name="LncSendrecv")
@pytest_marks(["lnc_sendrecv"])
class TestLncSendrecv:
    """lnc_sendrecv gpsimd-vs-fallback path selection under LNC=2."""

    def _run(self, test_manager, platform_target, allow_gpsimd):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_lnc_sendrecv,
            torch_ref=torch_ref_wrapper(_sendrecv_torch_ref),
            kernel_input_generator=_input_generator(allow_gpsimd),
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=2),
        )

    @pytest.mark.fast
    def test_gpsimd_path(self, test_manager, platform_target):
        """Small free dim + allow_gpsimd=True -> gpsimd_dma sendrecv (L44 true, L45)."""
        self._run(test_manager, platform_target, allow_gpsimd=True)

    @pytest.mark.fast
    def test_regular_dma_fallback(self, test_manager, platform_target):
        """allow_gpsimd=False -> regular DMA sendrecv (L44 false, L54)."""
        self._run(test_manager, platform_target, allow_gpsimd=False)
