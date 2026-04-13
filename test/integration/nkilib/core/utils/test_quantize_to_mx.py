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

"""Integration tests for quantize_to_mx.

Validates quantize_to_mx as torch ref against nisa.quantize_mx kernel
via UnitTestFramework (end-to-end on Trainium).
"""

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper
from typing import final

import neuron_dtypes as dt
import neuronxcc.nki.typing as nt
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.utils import mx_torch_common

# Hardware kernel: thin wrapper around nisa.quantize_mx


def quantize_mx_kernel(
    src_hbm: nl.ndarray,
    out_data_hbm: nt.mutable_tensor,
    out_scale_hbm: nt.mutable_tensor,
) -> None:
    """Quantize input to MX x4 format via nisa.quantize_mx.

    Output dtype is inferred from out_data_hbm.dtype.

    Dimensions:
        P: Partition dimension (must be divisible by 32, max 128)
        F: Free dimension (must be divisible by 4)

    Args:
        src_hbm (nl.ndarray): [P, F], fp16/bf16 input in HBM.
        out_data_hbm (nt.mutable_tensor): [P, F//4], packed x4 output in HBM.
        out_scale_hbm (nt.mutable_tensor): [P, F//4], uint8 scale output in HBM.

    Returns:
        None (outputs written in-place via nisa.dma_copy).

    Notes:
        - P must be divisible by 32 for nisa.quantize_mx, max 128.
        - Mathematically, MX quantization produces one scale per 8x4 block,
          so only P//8 scale rows are needed. However, nisa.quantize_mx stores
          scale in SBUF hardware quadrant layout [P, F//4]:

          Partition    Scale content       Description
          ──────────────────────────────────────────────
           0           scale[0, :]         block 0 (rows 0-7)
           1           scale[1, :]         block 1 (rows 8-15)
           2           scale[2, :]         block 2 (rows 16-23)
           3           scale[3, :]         block 3 (rows 24-31)
           4-31        0 (padding)         unused
          ──────────────────────────────────────────────
           32          scale[4, :]         next quadrant starts
           33          scale[5, :]
           34          scale[6, :]
           35          scale[7, :]
           36-63       0 (padding)         unused
          ──────────────────────────────────────────────

          Each 32-partition quadrant has 4 valid scale rows at the start
          (one per 8-row block), remaining 28 rows are zero padding.
          This matches the physical SBUF quadrant structure.

    Pseudocode:
        src_sb = dma_copy(src_hbm)
        dst_data, dst_scale = nisa.quantize_mx(src_sb)
        dma_copy(out_data_hbm, dst_data)
        dma_copy(out_scale_hbm, dst_scale)
    """
    P, F = src_hbm.shape
    out_x4_dtype = out_data_hbm.dtype

    src_sb = nl.ndarray((P, F), dtype=src_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=src_sb, src=src_hbm)

    dst_data_sb = nl.ndarray((P, F // 4), dtype=out_x4_dtype, buffer=nl.sbuf)
    dst_scale_sb = nl.ndarray((P, F // 4), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.quantize_mx(src=src_sb, dst=dst_data_sb, dst_scale=dst_scale_sb)

    nisa.dma_copy(dst=out_data_hbm, src=dst_data_sb)
    nisa.dma_copy(dst=out_scale_hbm, src=dst_scale_sb)

    # Explicitly return the mutable output tensors so the compiler generates
    # proper output_names in the KLIR/penguin IR.
    return out_data_hbm, out_scale_hbm


# Hardware test: nisa.quantize_mx vs quantize_to_mx

HW_SHAPES = [
    (128, 512),  # existing — full partition, large free dim
    (32, 512),  # minimum partition size (1 quadrant)
    (64, 256),  # 2 quadrants
    (32, 4),  # minimum both dimensions
    (128, 2048),  # large free dimension
]
HW_OUT_DTYPES = [nl.float8_e4m3fn_x4, nl.float8_e5m2_x4]  # float4_e2m1fn_x4 not supported by nisa.quantize_mx
HW_INPUT_DTYPES = [nl.bfloat16, nl.float16]

HW_PARAM_NAMES = "P, F, out_x4_dtype, input_dtype"
_ABBREVS = {"out_x4_dtype": "odt", "input_dtype": "idt"}
HW_TEST_PARAMS = [(P, F, out_dt, in_dt) for P, F in HW_SHAPES for out_dt in HW_OUT_DTYPES for in_dt in HW_INPUT_DTYPES]


@pytest_test_metadata(
    name="Quantize To MX Hardware",
    pytest_marks=["utils", "mx", "quantize"],
)
@final
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
class TestQuantizeToMxHardware:
    """Validate quantize_to_mx as torch ref against nisa.quantize_mx on Trainium."""

    @pytest.mark.fast
    @pytest_parametrize(HW_PARAM_NAMES, HW_TEST_PARAMS, abbrevs=_ABBREVS)
    def test_quantize_mx_hw(
        self,
        test_manager: Orchestrator,
        P: int,
        F: int,
        out_x4_dtype,
        input_dtype,
        platform_target: Platforms,
    ):
        kernel_func = quantize_mx_kernel
        torch_ref_func = mx_torch_common.quantize_mx_golden

        def input_generator(test_config, input_tensor_def=None):
            np.random.seed(42)
            data = dt.static_cast(np.random.randn(P, F).astype(np.float32), input_dtype)
            return {
                "src_hbm": data,
                "out_data_hbm.must_alias_input": np.zeros((P, F // 4), dtype=out_x4_dtype),
                "out_scale_hbm.must_alias_input": np.zeros((P, F // 4), dtype=np.uint8),
            }

        def output_tensors(kernel_input):
            return {
                "out_data_hbm": kernel_input["out_data_hbm.must_alias_input"].copy(),
                "out_scale_hbm": kernel_input["out_scale_hbm.must_alias_input"].copy(),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_func,
            torch_ref=torch_ref_wrapper(torch_ref_func),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=2e-2,
            atol=1e-5,
        )
