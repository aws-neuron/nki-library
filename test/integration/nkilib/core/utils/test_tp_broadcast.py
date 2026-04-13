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


from test.integration.nkilib.utils.tensor_generators import np_random_sample
from test.utils.common_dataclasses import CompilerArgs
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper
from typing import Tuple

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.utils.tensor_view import TensorView
from nkilib_src.nkilib.core.utils.tp_broadcast import tp_broadcast


@nki.jit
def tp_broadcast_hbm(src_hbm, src_offset, broadcast_count, src_select_index):
    dst_shape = (broadcast_count, src_hbm.shape[0])
    src_sbuf = nl.ndarray(src_hbm.shape, src_hbm.dtype, nl.sbuf)
    dst_hbm = nl.ndarray(dst_shape, src_hbm.dtype, nl.shared_hbm)
    dst_sbuf = nl.ndarray(dst_shape, src_hbm.dtype, nl.sbuf)

    nisa.dma_copy(src_sbuf, src_hbm)

    # check that some modification to the tensor to make it 2d works
    if src_select_index != None:
        src_in = TensorView(src_sbuf).select(1, src_select_index)
    else:
        src_in = src_sbuf

    tp_broadcast(src_in, dst_sbuf, src_offset)
    nisa.dma_copy(dst_hbm, dst_sbuf)
    return dst_hbm


def tp_broadcast_torch(src_hbm, src_offset, broadcast_count, src_select_index):
    if src_select_index != None:
        src_hbm = src_hbm.select(1, src_select_index)
    return src_hbm.reshape(src_hbm.shape[0], -1)[:, src_offset].unsqueeze(0).repeat(broadcast_count, 1)


_ABBREVS = {"src_shape": "shape", "src_dim_to_broadcast": "dim", "broadcast_count": "bc", "src_select_index": "sel"}


@pytest.mark.fast
@pytest_parametrize(
    "src_shape,src_dim_to_broadcast,broadcast_count,src_select_index",
    [
        ((5, 7), 0, 1, None),
        ((3, 8, 11), 1, 2, 4),
        ((8, 6), 5, 7, None),
        ((13, 3, 13), 9, 12, 2),
    ],
    abbrevs=_ABBREVS,
)
def test_tp_broadcast(
    test_manager: Orchestrator,
    src_shape: Tuple[int, int],
    src_dim_to_broadcast: int,
    broadcast_count: int,
    src_select_index: int | None,
):
    assert len(src_shape) == (3 if src_select_index else 2)

    def input_generator(test_config):
        random_gen = np_random_sample()

        src_hbm = random_gen(shape=src_shape, dtype=nl.float32, name='src_hbm')

        return {
            "src_hbm": src_hbm,
            "src_offset": src_dim_to_broadcast,
            "broadcast_count": broadcast_count,
            "src_select_index": src_select_index,
        }

    def output_tensors(kernel_input):
        return {"out": np.zeros((broadcast_count, src_shape[0]), dtype=np.float32)}

    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=tp_broadcast_hbm,
        torch_ref=torch_ref_wrapper(tp_broadcast_torch),
        kernel_input_generator=input_generator,
        output_tensor_descriptor=output_tensors,
    )
    framework.run_test(test_config=None, compiler_args=CompilerArgs(logical_nc_config=1), rtol=0, atol=0)
