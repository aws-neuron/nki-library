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

"""Standalone matmuls matching dense MXFP8 MLP operands at T=block_size=4096.

Cases cover WrapX with BF16 inputs and DMA transpose enabled, plus 1x32
with prequantized weights and DMA transpose enabled or disabled. Weight
gradient matmuls use BF16 operands in every case.

Shapes are the active GEMM dimensions on one MLP core: forward uses the full
token block, while backward shards the feature dimensions across two cores.
The standalone matmul chooses its own sharding, tiles, blocks, and load sizes.
For cases with DMA transpose enabled, leave NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE
unset. For disabled cases, set it to true in a fresh process. Only input
layouts, quantization scheme, and DMA transpose mode are fixed.
"""

from dataclasses import dataclass
from typing import final

import numpy as np
import pytest
from nkilib_src.nkilib.core.utils.kernel_helpers import _NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE
from nkilib_src.nkilib.experimental.matmul_mxfp8 import matmul_mxfp8_generic_kernel
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_torch import matmul_mxfp8_torch_ref
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.common_dataclasses import TensorOrientation

from test.integration.nkilib.experimental.matmul_mxfp8 import config_helper, constants
from test.integration.nkilib.experimental.matmul_mxfp8.test_matmul_mxfp8_generic_kernel import (
    _mxfp8_comparator,
    build_matmul_inputs,
    get_output_dtype,
)
from test.utils import common_dataclasses
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.unit_test_framework import UnitTestFramework


@dataclass(frozen=True)
class MlpMatmulShape:
    phase: str
    M: int
    K: int
    N: int
    weight_gradient: bool = False


def _mlp_shapes(hidden, intermediate):
    forward = (
        MlpMatmulShape("gate", 4096, hidden, intermediate),
        MlpMatmulShape("up", 4096, hidden, intermediate),
        MlpMatmulShape("down", 4096, intermediate, hidden),
    )
    backward = (
        MlpMatmulShape("p1_d_intermediate", 4096, hidden, intermediate // 2),
        MlpMatmulShape("p2_d_hidden", 4096, 2 * intermediate, hidden // 2),
        MlpMatmulShape("p3_d_gate_up_weight", hidden // 2, 4096, 2 * intermediate, weight_gradient=True),
        MlpMatmulShape("p4_d_down_weight", intermediate, 4096, hidden // 2, weight_gradient=True),
    )
    return forward, backward


QWEN32B_TP4_FORWARD, QWEN32B_TP4_BACKWARD = _mlp_shapes(hidden=5120, intermediate=6400)
LLAMA70B_TP1_FORWARD, LLAMA70B_TP1_BACKWARD = _mlp_shapes(hidden=8192, intermediate=28672)


def _input_config(shape, variant):
    wrapx = variant == "wrapx_bf16_inputs_dma_transpose_enabled"
    prequant_weight = not wrapx and not shape.weight_gradient
    orientation = TensorOrientation.K_BY_F if shape.weight_gradient else TensorOrientation.F_BY_K
    return config_helper.TestConfig(
        M=shape.M,
        K=shape.K,
        N=shape.N,
        lhs_dtype=constants.MatrixPrecision.BFLOAT16,
        rhs_dtype=constants.MatrixPrecision.MXFP8_X4 if prequant_weight else constants.MatrixPrecision.BFLOAT16,
        lhs_is_swizzled=False,
        rhs_is_swizzled=prequant_weight,
        lhs_orientation=orientation,
        rhs_orientation=TensorOrientation.K_BY_F if prequant_weight else orientation,
        enable_scale_packing=True if prequant_weight else None,
        quant_scheme="wrapX" if wrapx else "1x32",
        disable_dma_transpose=variant == "1x32_prequantized_weights_dma_transpose_disabled",
        description=f"MLP-{shape.phase}-{variant}",
    )


def _kernel_inputs(conf):
    # Packed weight scales are an input-format requirement. Raw BF16 operands
    # leave scale packing to the matmul's defaults.
    prequant_weight = conf.rhs_dtype == constants.MatrixPrecision.MXFP8_X4
    generated = build_matmul_inputs(conf)
    inputs = {
        name: generated[name]
        for name in (
            "lhs",
            "rhs",
            "lhs_scales",
            "rhs_scales",
            "lhs_is_swizzled",
            "rhs_is_swizzled",
            "float8_dtype",
            "quant_scheme",
        )
    }
    inputs.update(
        lhs_orientation=conf.lhs_orientation,
        rhs_orientation=conf.rhs_orientation,
        disable_dma_transpose=conf.disable_dma_transpose,
    )
    if prequant_weight:
        inputs["use_scale_packing"] = True
    return inputs


def _run_matmul(test_manager, platform_target, model, shape, variant):
    no_dma = variant == "1x32_prequantized_weights_dma_transpose_disabled"
    if _NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE != no_dma:
        setting = "true" if no_dma else "unset"
        pytest.skip(f"{variant} requires NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE={setting} before import")
    conf = _input_config(shape, variant)
    output_dtype = get_output_dtype(conf)
    test_manager.collector.set_kernel_params(
        {
            "model": model,
            "mlp_phase": shape.phase,
            "test_variant": variant,
            "M": shape.M,
            "K": shape.K,
            "N": shape.N,
            "lhs_orientation": conf.lhs_orientation.value,
            "rhs_orientation": conf.rhs_orientation.value,
            "lhs_dtype": conf.lhs_dtype,
            "rhs_dtype": conf.rhs_dtype,
            "quant_scheme": conf.quant_scheme,
            "disable_dma_transpose": conf.disable_dma_transpose,
        }
    )
    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=matmul_mxfp8_generic_kernel.matmul_mxfp8,
        torch_ref=matmul_mxfp8_torch_ref,
        kernel_input_generator=lambda _: _kernel_inputs(conf),
        output_tensor_descriptor=lambda _: {"out": np.zeros((shape.M, shape.N), dtype=output_dtype)},
    )
    framework.run_test(
        test_config=None,
        compiler_args=common_dataclasses.CompilerArgs(logical_nc_config=2, platform_target=platform_target),
        custom_comparator=_mxfp8_comparator(conf, output_dtype),
    )


def _phase_id(shape):
    return f"{shape.phase}-{shape.M}x{shape.K}x{shape.N}"


@pytest_test_metadata(name="Qwen3-32B TP4 and Llama-70B TP1 MLP phase matmuls")
@pytest_marks(["generic_matmul_mxfp8_perf", "mx", "mxfp8", "model", "tuning"])
@pytest.mark.platforms(
    exclude=[
        common_dataclasses.Platforms.TRN1,
        common_dataclasses.Platforms.TRN2,
        common_dataclasses.Platforms.TRN3_A0,
        common_dataclasses.Platforms.TRN3_PDS_A0,
    ]
)
@final
class TestQwen32BTp4MlpMatmulPerf:
    @pytest.mark.parametrize("shape", QWEN32B_TP4_FORWARD, ids=_phase_id)
    def test_fwd_wrapx_dma_bf16_inputs(self, test_manager, platform_target, shape):
        _run_matmul(test_manager, platform_target, "qwen3_32b", shape, "wrapx_bf16_inputs_dma_transpose_enabled")

    @pytest.mark.parametrize("shape", QWEN32B_TP4_BACKWARD, ids=_phase_id)
    def test_bwd_wrapx_dma_bf16_inputs(self, test_manager, platform_target, shape):
        _run_matmul(test_manager, platform_target, "qwen3_32b", shape, "wrapx_bf16_inputs_dma_transpose_enabled")

    @pytest.mark.parametrize("shape", QWEN32B_TP4_FORWARD, ids=_phase_id)
    def test_fwd_1x32_dma_prequant_weight(self, test_manager, platform_target, shape):
        _run_matmul(
            test_manager, platform_target, "qwen3_32b", shape, "1x32_prequantized_weights_dma_transpose_enabled"
        )

    @pytest.mark.parametrize("shape", QWEN32B_TP4_BACKWARD, ids=_phase_id)
    def test_bwd_1x32_dma_prequant_weight(self, test_manager, platform_target, shape):
        _run_matmul(
            test_manager, platform_target, "qwen3_32b", shape, "1x32_prequantized_weights_dma_transpose_enabled"
        )

    @pytest.mark.parametrize("shape", QWEN32B_TP4_FORWARD, ids=_phase_id)
    def test_fwd_1x32_no_dma_transpose_prequant_weight(self, test_manager, platform_target, shape):
        _run_matmul(
            test_manager, platform_target, "qwen3_32b", shape, "1x32_prequantized_weights_dma_transpose_disabled"
        )

    @pytest.mark.parametrize("shape", QWEN32B_TP4_BACKWARD, ids=_phase_id)
    def test_bwd_1x32_no_dma_transpose_prequant_weight(self, test_manager, platform_target, shape):
        _run_matmul(
            test_manager, platform_target, "qwen3_32b", shape, "1x32_prequantized_weights_dma_transpose_disabled"
        )


@pytest_marks(["generic_matmul_mxfp8_perf", "mx", "mxfp8", "model", "tuning"])
@pytest.mark.platforms(
    exclude=[
        common_dataclasses.Platforms.TRN1,
        common_dataclasses.Platforms.TRN2,
        common_dataclasses.Platforms.TRN3_A0,
        common_dataclasses.Platforms.TRN3_PDS_A0,
    ]
)
@final
class TestLlama70BTp1MlpMatmulPerf:
    @pytest.mark.parametrize("shape", LLAMA70B_TP1_FORWARD, ids=_phase_id)
    def test_fwd_wrapx_dma_bf16_inputs(self, test_manager, platform_target, shape):
        _run_matmul(test_manager, platform_target, "llama3_70b", shape, "wrapx_bf16_inputs_dma_transpose_enabled")

    @pytest.mark.parametrize("shape", LLAMA70B_TP1_BACKWARD, ids=_phase_id)
    def test_bwd_wrapx_dma_bf16_inputs(self, test_manager, platform_target, shape):
        _run_matmul(test_manager, platform_target, "llama3_70b", shape, "wrapx_bf16_inputs_dma_transpose_enabled")

    @pytest.mark.parametrize("shape", LLAMA70B_TP1_FORWARD, ids=_phase_id)
    def test_fwd_1x32_dma_prequant_weight(self, test_manager, platform_target, shape):
        _run_matmul(
            test_manager, platform_target, "llama3_70b", shape, "1x32_prequantized_weights_dma_transpose_enabled"
        )

    @pytest.mark.parametrize("shape", LLAMA70B_TP1_BACKWARD, ids=_phase_id)
    def test_bwd_1x32_dma_prequant_weight(self, test_manager, platform_target, shape):
        _run_matmul(
            test_manager, platform_target, "llama3_70b", shape, "1x32_prequantized_weights_dma_transpose_enabled"
        )

    @pytest.mark.parametrize("shape", LLAMA70B_TP1_FORWARD, ids=_phase_id)
    def test_fwd_1x32_no_dma_transpose_prequant_weight(self, test_manager, platform_target, shape):
        _run_matmul(
            test_manager, platform_target, "llama3_70b", shape, "1x32_prequantized_weights_dma_transpose_disabled"
        )

    @pytest.mark.parametrize("shape", LLAMA70B_TP1_BACKWARD, ids=_phase_id)
    def test_bwd_1x32_no_dma_transpose_prequant_weight(self, test_manager, platform_target, shape):
        _run_matmul(
            test_manager, platform_target, "llama3_70b", shape, "1x32_prequantized_weights_dma_transpose_disabled"
        )
