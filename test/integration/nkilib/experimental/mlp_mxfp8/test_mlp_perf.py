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

"""Qwen3-32B TP4 and Llama-70B TP1 MXFP8 MLP forward/backward cases.

Both models use T=4096. Forward, backward, and the checkpoint use two
2048-token blocks. Forward assigns one block per core. Cases cover WrapX with BF16
inputs and DMA transpose enabled, plus 1x32 with prequantized weights and
DMA transpose enabled or disabled. For cases with DMA transpose enabled,
leave NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE unset. For disabled cases, set it
to true in a fresh process.
"""

from typing import final

import pytest

from test.integration.nkilib.experimental.mlp_mxfp8.mlp_perf_test_utils import MlpShape, run_backward, run_forward
from test.utils import common_dataclasses
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator

QWEN32B_TP4_SHAPE = MlpShape(
    model="qwen3_32b",
    tensor_parallel_degree=4,
    hidden=5120,
    intermediate=6400,
    tokens=4096,
)

LLAMA70B_TP1_SHAPE = MlpShape(
    model="llama3_70b",
    tensor_parallel_degree=1,
    hidden=8192,
    intermediate=28672,
    tokens=4096,
)


@pytest_test_metadata(name="Qwen3-32B TP4 and Llama-70B TP1 MXFP8 MLP")
@pytest_marks(["mlp_mxfp8", "mxfp8", "model", "tuning"])
@pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN1, common_dataclasses.Platforms.TRN2])
@final
class TestQwen32BTp4MlpPerf:
    def test_fwd_wrapx_dma_bf16_inputs(self, test_manager: Orchestrator, platform_target):
        run_forward(test_manager, platform_target, QWEN32B_TP4_SHAPE, "wrapx_bf16_inputs_dma_transpose_enabled")

    def test_bwd_wrapx_dma_bf16_inputs(self, test_manager: Orchestrator, platform_target):
        run_backward(test_manager, platform_target, QWEN32B_TP4_SHAPE, "wrapx_bf16_inputs_dma_transpose_enabled")

    def test_fwd_1x32_dma_prequant_weight(self, test_manager: Orchestrator, platform_target):
        run_forward(test_manager, platform_target, QWEN32B_TP4_SHAPE, "1x32_prequantized_weights_dma_transpose_enabled")

    def test_bwd_1x32_dma_prequant_weight(self, test_manager: Orchestrator, platform_target):
        run_backward(
            test_manager, platform_target, QWEN32B_TP4_SHAPE, "1x32_prequantized_weights_dma_transpose_enabled"
        )

    def test_fwd_1x32_no_dma_transpose_prequant_weight(self, test_manager: Orchestrator, platform_target):
        run_forward(
            test_manager, platform_target, QWEN32B_TP4_SHAPE, "1x32_prequantized_weights_dma_transpose_disabled"
        )

    def test_bwd_1x32_no_dma_transpose_prequant_weight(self, test_manager: Orchestrator, platform_target):
        run_backward(
            test_manager, platform_target, QWEN32B_TP4_SHAPE, "1x32_prequantized_weights_dma_transpose_disabled"
        )


@pytest_marks(["mlp_mxfp8", "mxfp8", "model", "tuning"])
@pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN1, common_dataclasses.Platforms.TRN2])
@final
class TestLlama70BTp1MlpPerf:
    def test_fwd_wrapx_dma_bf16_inputs(self, test_manager: Orchestrator, platform_target):
        run_forward(test_manager, platform_target, LLAMA70B_TP1_SHAPE, "wrapx_bf16_inputs_dma_transpose_enabled")

    def test_bwd_wrapx_dma_bf16_inputs(self, test_manager: Orchestrator, platform_target):
        run_backward(test_manager, platform_target, LLAMA70B_TP1_SHAPE, "wrapx_bf16_inputs_dma_transpose_enabled")

    def test_fwd_1x32_dma_prequant_weight(self, test_manager: Orchestrator, platform_target):
        run_forward(
            test_manager, platform_target, LLAMA70B_TP1_SHAPE, "1x32_prequantized_weights_dma_transpose_enabled"
        )

    def test_bwd_1x32_dma_prequant_weight(self, test_manager: Orchestrator, platform_target):
        run_backward(
            test_manager, platform_target, LLAMA70B_TP1_SHAPE, "1x32_prequantized_weights_dma_transpose_enabled"
        )

    def test_fwd_1x32_no_dma_transpose_prequant_weight(self, test_manager: Orchestrator, platform_target):
        run_forward(
            test_manager, platform_target, LLAMA70B_TP1_SHAPE, "1x32_prequantized_weights_dma_transpose_disabled"
        )

    def test_bwd_1x32_no_dma_transpose_prequant_weight(self, test_manager: Orchestrator, platform_target):
        run_backward(
            test_manager, platform_target, LLAMA70B_TP1_SHAPE, "1x32_prequantized_weights_dma_transpose_disabled"
        )
