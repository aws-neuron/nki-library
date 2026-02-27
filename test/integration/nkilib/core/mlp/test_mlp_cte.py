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

# LEGACY SWEEP TEST FRAMEWORK - Uses @range_test_config / RangeTestHarness
# New tests should use @pytest.mark.coverage_parametrize instead
import enum
import math

try:
    from test.integration.nkilib.core.mlp.test_mlp_cte_model_config import (
        mlp_cte_model_configs,
    )
except ImportError:
    mlp_cte_model_configs = []

from functools import lru_cache
from test.integration.nkilib.core.mlp.test_mlp_common import (
    build_fused_norm_mlp,
    gaussian_tensor_generator,
    golden_mlp,
    golden_quant_mlp,
    modify_for_row_quant,
    modify_fp8_static_scale,
)
from test.utils.common_dataclasses import (
    MODEL_TEST_TYPE,
    CompilerArgs,
    KernelArgs,
    LazyGoldenGenerator,
    Platforms,
    ValidationArgs,
)
from test.utils.metadata_loader import load_model_configs
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.ranged_test_harness import (
    DimensionRangeConfig,
    RangeManualGeneratorStrategy,
    RangeTestCase,
    RangeTestConfig,
    TensorConfig,
    TensorRangeConfig,
    assert_negative_test_case,
    range_test_config,
)
from test.utils.test_orchestrator import Orchestrator
from typing import Any, final

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.mlp.mlp import mlp
from nkilib_src.nkilib.core.utils.common_types import ActFnType, NormType, QuantizationType
from typing_extensions import override

MLP_CTE_CONFIG = "mlp_cte_config"
VNC_DEGREE_DIM_NAME = "vnc_degree"
BATCH_DIM_NAME = "batch"
SEQUENCE_LEN_DIM_NAME = "seqlen"
HIDDEN_DIM_NAME = "hidden"
INTERMEDIATE_DIM_NAME = "intermediate"
NORM_TYPE_DIM_NAME = "norm_type"
QUANT_TYPE_DIM_NAME = "quant_type"
FUSED_ADD_DIM_NAME = "fused_add"
STORE_ADD_DIM_NAME = "store_add"
SKIP_GATE_DIM_NAME = "skip_gate"
ACT_FN_TYPE_DIM_NAME = "act_fn_type"
GATE_BIAS_DIM_NAME = "gate_bias"
UP_BIAS_DIM_NAME = "up_bias"
DOWN_BIAS_DIM_NAME = "down_bias"
NORM_BIAS_DIM_NAME = "norm_bias"
RTOL_DIM_NAME = "rtol"
RANGE_CONFIG_DIM_NAME = "range_config"


def create_mlp_cte_test_config(test_vectors, test_type: str = "manual"):
    """
    Create RangeTestConfig from MLP CTE test vectors.

    Args:
        test_vectors: List of test parameter tuples (16 fields, vnc_degree and tpbSgCyclesSum ignored)
        test_type: Type identifier for the test config ("manual" or "model")

    Returns:
        RangeTestConfig for the test vectors
    """
    test_cases = []
    for test_params in test_vectors:
        (
            vnc_degree,
            batch,
            seqlen,
            hidden,
            intermediate,
            tpbSgCyclesSum,
            rtol,
            norm_type,
            quant_type,
            fused_add,
            store_add,
            skip_gate,
            act_fn_type,
            gate_bias,
            up_bias,
            down_bias,
            norm_bias,
        ) = test_params

        test_case = {
            MLP_CTE_CONFIG: {
                BATCH_DIM_NAME: batch,
                SEQUENCE_LEN_DIM_NAME: seqlen,
                HIDDEN_DIM_NAME: hidden,
                INTERMEDIATE_DIM_NAME: intermediate,
                NORM_TYPE_DIM_NAME: norm_type.value if hasattr(norm_type, "value") else norm_type,
                QUANT_TYPE_DIM_NAME: quant_type.value if hasattr(quant_type, "value") else quant_type,
                FUSED_ADD_DIM_NAME: int(fused_add),
                STORE_ADD_DIM_NAME: int(store_add),
                SKIP_GATE_DIM_NAME: int(skip_gate),
                ACT_FN_TYPE_DIM_NAME: act_fn_type.value if hasattr(act_fn_type, "value") else act_fn_type,
                GATE_BIAS_DIM_NAME: int(gate_bias),
                UP_BIAS_DIM_NAME: int(up_bias),
                DOWN_BIAS_DIM_NAME: int(down_bias),
                NORM_BIAS_DIM_NAME: int(norm_bias),
                RTOL_DIM_NAME: float(rtol),
            }
        }
        test_cases.append(test_case)

    return RangeTestConfig(
        additional_params={},
        global_tensor_configs=TensorRangeConfig(
            tensor_configs={},
            monotonic_step_size=1,
            custom_generators=[
                RangeManualGeneratorStrategy(test_cases=test_cases, test_type=test_type),
            ],
        ),
    )


# ----------------------------------------------------
# Configuration-based testing to avoid combinatorial explosion of parameters
# ----------------------------------------------------

BASIC_MLP_CONFIG = {
    "norm_type": NormType.NO_NORM,
    "fused_add": False,
    "store_add": False,
    "skip_gate": False,
    "act_fn_type": ActFnType.SiLU,
    "gate_bias": False,
    "up_bias": False,
    "down_bias": False,
    "norm_bias": False,
}

FULL_FEATURES_CONFIG = {
    "norm_type": NormType.RMS_NORM,
    "fused_add": True,
    "store_add": True,
    "skip_gate": False,
    "act_fn_type": ActFnType.SiLU,
    "gate_bias": True,
    "up_bias": True,
    "down_bias": True,
    "norm_bias": False,
}

LAYER_NORM_CONFIG = {
    "norm_type": NormType.LAYER_NORM,
    "fused_add": False,
    "store_add": False,
    "skip_gate": False,
    "act_fn_type": ActFnType.SiLU,
    "gate_bias": False,
    "up_bias": False,
    "down_bias": False,
    "norm_bias": True,
}
# Enumerate test configs so it can be sweeped by test infra
MLP_RANGE_TEST_CONFIGS = {0: BASIC_MLP_CONFIG, 1: FULL_FEATURES_CONFIG, 2: LAYER_NORM_CONFIG}

TENSOR_GEN_WEIGHT_LOWER = 0.0
TENSOR_GEN_WEIGHT_UPPER = 40.0


# ----------------------------------------------------
# MLP CTE size classification for CloudWatch metrics
# ----------------------------------------------------
class MLPCTEClassification(enum.Enum):
    SMALL = 1
    MEDIUM = 2
    LARGE = 3

    @staticmethod
    def classify(batch: int, seqlen: int, hidden: int):
        total_elements = batch * seqlen * hidden

        if total_elements <= 1024 * 1024:
            return MLPCTEClassification.SMALL
        elif total_elements <= 16 * 1024 * 1024:
            return MLPCTEClassification.MEDIUM
        else:
            return MLPCTEClassification.LARGE

    @override
    def __str__(self):
        return self.name


# fmt: off
# Parameters: vnc_degree, batch, seqlen, hidden, intermediate, tpbSgCyclesSum, rtol, norm_type, quantization_type,
#             fused_add, store_add, skip_gate, act_fn_type, gate_bias, up_bias, down_bias, norm_bias
MLP_CTE_UNIT_TEST_CASES_GATE_BIAS_FALSE = [
    [2, 1, 128, 8192, 896, 238002211, 2e-2, NormType.LAYER_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 512, 1024, 448, 80453957, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 512, 1024, 448, 82809370, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 1024, 1024, 448, 92239439, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 1024, 1024, 448, 88019279, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 1024, 1024, 448, 92000000, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 1024, 1024, 448, 90017692, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 768, 1024, 896, 140496530, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 768, 1024, 896, 107670831, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 8192, 8192, 448, 4296859369, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 8192, 8192, 448, 2520729395, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 1024, 16384, 832, 756458068, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 1024, 16384, 832, 895261601, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 1024, 16384, 416, 882105371, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 1024, 16384, 416, 1176268151, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 1024, 8192, 448, 278000000, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 8192, 8192, 448, 4780395634, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 8192, 8192, 448, 3784829419, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 1024, 16384, 416, 1000000000, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 1024, 16384, 416, 1246253552, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 8192, 8192, 448, 2737231223, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 8192, 8192, 448, 2127094342, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 1024, 16384, 416, 595047246, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 1024, 16384, 416, 755884001, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [2, 2, 8192, 8192, 448, 5298071805, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [2, 4, 8192, 8192, 448, 8376700327, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [2, 2, 1024, 16384, 416, 1111000000, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [2, 4, 1024, 16384, 416, 2557272337, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 128, 8192, 896, 210306921, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 128, 8192, 896, 258002211, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 128, 8192, 896, 210306921, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 128, 8192, 896, 258002211, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 512, 1024, 448, 80453957, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 512, 1024, 448, 82809370, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 128, 8192, 896, 210306921, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 128, 8192, 896, 210306921, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 2048, 8448, 1408, 1437844003, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 2048, 8448, 1408, 1603737495, 2e-2, NormType.LAYER_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 4096, 8192, 448, 941321030, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 4096, 8192, 448, 1176746495, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 4096, 8192, 448, 1297751306, 2e-2, NormType.LAYER_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 128, 7168, 364, 1.52e8, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 128, 7168, 364, 1.21e8, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 256, 7168, 364, 1.59e8, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 256, 7168, 364, 1.41e8, 2e-2, NormType.RMS_NORM_SKIP_GAMMA, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 256, 7168, 1536, 281286893, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 256, 7168, 1536, 258002211, 2e-2, NormType.RMS_NORM_SKIP_GAMMA, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 256, 7168, 1536, 258002211, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 578, 1408, 352, 66365979, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, True, ActFnType.SiLU, False, False, False, False],
    [1, 1, 578, 1408, 352, 66340813, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, True, ActFnType.GELU, False, False, False, False],
    [2, 1, 578, 1408, 352, 65561487, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, True, ActFnType.SiLU, False, False, False, False],
    [2, 1, 578, 1408, 352, 66839480, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, True, ActFnType.GELU, False, False, False, False],
    [1, 1, 578, 1408, 352, 71063222, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, True, ActFnType.SiLU, False, True, True, False],
    [1, 1, 578, 1408, 352, 71242055, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, True, ActFnType.GELU, False, True, True, False],
    [2, 1, 578, 1408, 352, 69290891, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, True, ActFnType.SiLU, False, True, True, False],
    [2, 1, 578, 1408, 352, 71074390, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, True, ActFnType.GELU, False, True, True, False],
    [1, 1, 578, 1408, 352, 81643622, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, True, True, False],
    [2, 1, 578, 1408, 352, 81231123, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, True, True, False],
    [1, 1, 578, 1408, 352, 84836617, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.GELU, False, False, False, False],
    [2, 1, 578, 1408, 352, 81332790, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.GELU, False, False, False, False],
    [1, 1, 578, 1408, 352, 93398179, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, True, ActFnType.SiLU, False, True, True, False],
    [2, 1, 578, 1408, 352, 83390954, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, True, ActFnType.SiLU, False, True, True, False],
    [1, 1, 578, 1408, 352, 97224098, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.GELU, False, True, True, False],
    [2, 1, 578, 1408, 352, 88505279, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.GELU, False, True, True, False],
    [1, 1, 578, 1408, 352, 90984235, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, True, ActFnType.GELU_Tanh_Approx, False, True, True, False],
    [2, 1, 578, 1408, 352, 83078600, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, True, ActFnType.GELU_Tanh_Approx, False, True, True, False],
    [1, 1, 578, 1408, 352, 91208523, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, True, ActFnType.GELU, False, True, False, False],
    [2, 1, 578, 1408, 352, 75500049, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, True, ActFnType.GELU, False, True, False, False],
    [1, 1, 578, 1408, 352, 93929554, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, True, ActFnType.GELU, False, False, True, False],
    [2, 1, 578, 1408, 352, 79235293, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, True, ActFnType.GELU, False, False, True, False],
    [1, 14, 578, 1408, 352, 564325284, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, True, ActFnType.GELU, False, True, False, False],
    [2, 14, 578, 1408, 352, 351045201, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, True, ActFnType.GELU, False, True, False, False],
    [1, 1, 578, 1408, 352, 105467001, 2e-2, NormType.LAYER_NORM, QuantizationType.NONE, False, False, True, ActFnType.GELU, False, True, True, True],
    [1, 1, 578, 1408, 352, 109600578, 2e-2, NormType.LAYER_NORM, QuantizationType.NONE, True, True, True, ActFnType.GELU, False, True, True, True],
    [2, 1, 578, 1408, 352, 98436096, 2e-2, NormType.LAYER_NORM, QuantizationType.NONE, False, False, True, ActFnType.GELU, False, True, True, True],
    [2, 1, 578, 1408, 352, 105223169, 2e-2, NormType.LAYER_NORM, QuantizationType.NONE, True, True, True, ActFnType.GELU, False, True, True, True],
    [2, 1, 36864, 8192, 512, 8367215426, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.GELU, False, False, False, False],
    [2, 1, 512, 8192, 3584, None, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
]

MLP_CTE_UNIT_TEST_CASES_GATE_BIAS_TRUE = [
    # Commented out due to known failure - FIXME: migrating to new test framework, addressed then
    # [2, 128, 5, 3072, 112, None, NormType.NO_NORM, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 10240, 3072, 112, 638_002_211, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 10240, 3072, 2160, 6_638_002_211, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 128, 8192, 896, 238002211, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 512, 1024, 448, 80453957, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 512, 1024, 448, 82809370, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 1024, 1024, 448, 92239439, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 1024, 1024, 448, 88019279, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 1024, 1024, 448, 92000000, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 1024, 1024, 448, 90017692, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 768, 1024, 896, 140496530, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 768, 1024, 896, 107670831, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 8192, 8192, 448, 4296859369, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 8192, 8192, 448, 2520729395, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 1024, 16384, 832, 756458068, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 1024, 16384, 832, 895261601, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 1024, 16384, 416, 882105371, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 1024, 16384, 416, 1176268151, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 1024, 8192, 448, 278000000, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 8192, 8192, 448, 4780395634, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 8192, 8192, 448, 3784829419, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 1024, 16384, 416, 1140095218, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 1024, 16384, 416, 1246253552, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 8192, 8192, 448, 2737231223, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 8192, 8192, 448, 2127094342, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 1024, 16384, 416, 595047246, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 1024, 16384, 416, 755884001, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [2, 2, 8192, 8192, 448, 5298071805, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [2, 4, 8192, 8192, 448, 8376700327, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [2, 2, 1024, 16384, 416, 1118514002, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [2, 4, 1024, 16384, 416, 2557272337, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 128, 8192, 896, 210306921, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 128, 8192, 896, 258002211, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 128, 8192, 896, 210306921, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 128, 8192, 896, 258002211, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 512, 1024, 448, 80453957, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 512, 1024, 448, 82809370, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 128, 8192, 896, 210306921, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 128, 8192, 896, 210306921, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 2048, 8448, 1408, 1437844003, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 2048, 8448, 1408, 1603737495, 2e-2, NormType.LAYER_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 4096, 8192, 448, 941321030, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 4096, 8192, 448, 1176746495, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 4096, 8192, 448, 1297751306, 2e-2, NormType.LAYER_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 128, 7168, 364, 1.52e8, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 128, 7168, 364, 1.21e8, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 256, 7168, 364, 1.59e8, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 256, 7168, 364, 1.41e8, 2e-2, NormType.RMS_NORM_SKIP_GAMMA, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 578, 1408, 352, 81643622, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, True, True, False],
    [2, 1, 578, 1408, 352, 81231123, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, True, True, False],
    [1, 1, 578, 1408, 352, 84836617, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.GELU, True, False, False, False],
    [2, 1, 578, 1408, 352, 81332790, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.GELU, True, False, False, False],
    [1, 1, 578, 1408, 352, 99168579, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.GELU, True, True, True, False],
    [2, 1, 578, 1408, 352, 88505279, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.GELU, True, True, True, False],
]

ceilalign = lambda n, a: math.ceil(n / a) * a

MLP_CTE_UNIT_TEST_CASES_ROW_QUANT = [
    [2, 1, 1024, 16384, ceilalign(896, 128), 5.42e8, 4e-2, NormType.NO_NORM, QuantizationType.ROW, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 512, 16384, ceilalign(896, 128), 5.44e8, 4e-2, NormType.NO_NORM, QuantizationType.ROW, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 256, 16384, ceilalign(896, 128), 8.05e8, 4e-2, NormType.NO_NORM, QuantizationType.ROW, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 4, 1024, 16384, ceilalign(896, 128), 1.96e9, 4e-2, NormType.NO_NORM, QuantizationType.ROW, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 2, 1024, 16384, ceilalign(896, 128), 1.01e9, 4e-2, NormType.NO_NORM, QuantizationType.ROW, False, False, False, ActFnType.SiLU, False, False, False, False],
]

MLP_CTE_UNIT_TEST_CASES_STATIC_QUANT = [
    # Llama 3.3 70B
    [2, 1, 10240, 8192, ceilalign(7168, 256), None, 3e-2, NormType.NO_NORM, QuantizationType.STATIC, False, False, False, ActFnType.SiLU, False, False, False, False],  # TP4
    [2, 1, 10240, 8192, ceilalign(3584, 128), None, 3e-2, NormType.NO_NORM, QuantizationType.STATIC, False, False, False, ActFnType.SiLU, False, False, False, False],  # TP8
    [2, 1, 10240, 8192, ceilalign(1792, 128), None, 3e-2, NormType.NO_NORM, QuantizationType.STATIC, False, False, False, ActFnType.SiLU, False, False, False, False],  # TP16
    # Qwen 3 32B
    [2, 1, 10240, 5120, ceilalign(6400, 256), None, 3e-2, NormType.NO_NORM, QuantizationType.STATIC, False, False, False, ActFnType.SiLU, False, False, False, False],  # TP4
    [2, 1, 10240, 5120, ceilalign(3200, 128), None, 3e-2, NormType.NO_NORM, QuantizationType.STATIC, False, False, False, ActFnType.SiLU, False, False, False, False],  # TP8
    [2, 1, 10240, 5120, ceilalign(1600, 128), None, 3e-2, NormType.NO_NORM, QuantizationType.STATIC, False, False, False, ActFnType.SiLU, False, False, False, False],  # TP16
]

MLP_CTE_UNIT_TEST_CASES_STATIC_MX_QUANT = [
    # Llama 3.3 70B
    [2, 1, 10240, 8192, ceilalign(7168, 1024), None, 3e-2, NormType.NO_NORM, QuantizationType.STATIC_MX, False, False, False, ActFnType.SiLU, False, False, False, False],  # TP4
    [2, 1, 10240, 8192, ceilalign(3584, 512), None, 3e-2, NormType.NO_NORM, QuantizationType.STATIC_MX, False, False, False, ActFnType.SiLU, False, False, False, False],  # TP8
    [2, 1, 10240, 8192, ceilalign(1792, 512), None, 3e-2, NormType.NO_NORM, QuantizationType.STATIC_MX, False, False, False, ActFnType.SiLU, False, False, False, False],  # TP16
    # Qwen 3 32B
    [2, 1, 10240, 5120, ceilalign(6400, 1024), None, 3e-2, NormType.NO_NORM, QuantizationType.STATIC_MX, False, False, False, ActFnType.SiLU, False, False, False, False],  # TP4
    [2, 1, 10240, 5120, ceilalign(3200, 512), None, 3e-2, NormType.NO_NORM, QuantizationType.STATIC_MX, False, False, False, ActFnType.SiLU, False, False, False, False],  # TP8
    [2, 1, 10240, 5120, ceilalign(1600, 512), None, 3e-2, NormType.NO_NORM, QuantizationType.STATIC_MX, False, False, False, ActFnType.SiLU, False, False, False, False],  # TP16
]
# fmt: on


@lru_cache(maxsize=1)
def _get_mlp_cte_metadata():
    return load_model_configs("test_mlp_cte")


@pytest_test_metadata(
    name="MLP CTE",
    pytest_marks=["mlp", "cte"],
)
@final
class TestMlpCteKernel:
    def run_range_mlp_cte_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        test_options: RangeTestCase,
        lnc_degree,
        dtype,
        norm_type,
        quant_type,
        fused_add,
        store_add,
        skip_gate,
        act_fn_type,
        gate_bias,
        up_bias,
        down_bias,
        norm_bias,
        collector: IMetricsCollector,
    ):
        mlp_config = test_options.tensors[MLP_CTE_CONFIG]

        batch = mlp_config[BATCH_DIM_NAME]
        seqlen = mlp_config[SEQUENCE_LEN_DIM_NAME]
        hidden = mlp_config[HIDDEN_DIM_NAME]
        intermediate = mlp_config[INTERMEDIATE_DIM_NAME]

        # Model configs should never be marked as negative test cases
        is_model_config = test_options.test_type == MODEL_TEST_TYPE
        is_negative_test_case = False if is_model_config else test_options.is_negative_test_case

        # Kernel constraint: bias is not supported for short seqlen
        # From mlp.py: SHORT_SEQLEN_THRESHOLD = 256
        SHORT_SEQLEN_THRESHOLD = 256
        TKG_BS_SEQLEN_THRESHOLD = 96

        shard_on_i = (
            (batch * seqlen <= SHORT_SEQLEN_THRESHOLD)
            and (intermediate % 256 == 0)
            and not (gate_bias or up_bias or norm_bias)
            and lnc_degree
            and lnc_degree > 1
        )

        if lnc_degree and lnc_degree > 1 and batch * seqlen > TKG_BS_SEQLEN_THRESHOLD and not shard_on_i:
            # Check if seqlen is divisible by num_shards
            if batch * seqlen % lnc_degree != 0 and not is_model_config:
                is_negative_test_case = True

        # Known OOM cases: seqlen=6272 with intermediate=2048 and fused_add causes OOM
        # Memory usage is borderline and results are flaky in parallel execution
        if seqlen == 6272 and intermediate == 2048 and fused_add:
            pytest.skip("OOM tests")

        # ----------------------------------------------------
        # CloudWatch metrics for kernel test outcome
        # ----------------------------------------------------
        test_size_classification = MLPCTEClassification.classify(batch, seqlen, hidden)

        # Safety check: model configs must never be marked as negative test cases
        assert not (
            is_model_config and is_negative_test_case
        ), "Model configs must never be marked as negative test cases"

        with assert_negative_test_case(is_negative_test_case):
            self.run_mlp_cte_test(
                test_manager,
                compiler_args,
                collector,
                batch,
                seqlen,
                hidden,
                intermediate,
                act_fn_type,
                dtype,
                quant_type,
                fused_add,
                gate_bias,
                lnc_degree,
                norm_bias,
                norm_type,
                skip_gate,
                store_add,
                down_bias,
                up_bias,
                rtol=mlp_config.get(RTOL_DIM_NAME, 2e-2),
            )

    def run_mlp_cte_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        collector: IMetricsCollector,
        batch: int,
        seqlen: int,
        hidden: int,
        intermediate: int,
        act_fn_type,
        dtype,
        quant_type,
        fused_add,
        gate_bias,
        lnc_degree,
        norm_bias,
        norm_type,
        skip_gate,
        store_add,
        down_bias,
        up_bias,
        rtol,
    ):
        if quant_type == QuantizationType.ROW:
            tensor_generator = gaussian_tensor_generator(
                mean=0.0,
                std=10.0,
                modifier_fn=modify_for_row_quant,
                lnc=lnc_degree,
            )
        elif quant_type in [QuantizationType.STATIC, QuantizationType.STATIC_MX]:
            tensor_generator = gaussian_tensor_generator(
                mean=0.0,
                std=5.0,
                modifier_fn=modify_fp8_static_scale,
                lnc=lnc_degree,
            )
        else:
            tensor_generator = gaussian_tensor_generator()
        kernel_input = build_fused_norm_mlp(
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            intermediate=intermediate,
            dtype=dtype,
            quantization_type=quant_type,
            quant_dtype=nl.float8_e4m3,
            is_input_quantized=(quant_type != QuantizationType.NONE),
            fused_add=fused_add,
            norm_type=norm_type,
            store_add=store_add,
            lnc_degree=lnc_degree if lnc_degree > 1 else None,
            skip_gate=skip_gate,
            act_fn_type=act_fn_type,
            gate_bias=gate_bias,
            up_bias=up_bias,
            down_bias=down_bias,
            norm_bias=norm_bias,
            tensor_generator=tensor_generator,
        )

        # Determine if store_add output will be produced
        store_add_valid = fused_add and store_add

        # Create placeholder tensors with correct shapes and dtype for compiler tracing
        output_tensors = {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}
        if store_add_valid:
            output_tensors["add_out"] = np.zeros((batch, seqlen, hidden), dtype=dtype)

        # Create lazy golden generator closure that captures all necessary variables
        def create_lazy_golden():
            if quant_type == QuantizationType.NONE:
                return golden_mlp(
                    inp_np=kernel_input,
                    norm_type=norm_type,
                    fused_add=fused_add,
                    store_add=store_add,
                    dtype=dtype,
                    skip_gate=skip_gate,
                    act_fn_type=act_fn_type,
                    lnc=lnc_degree if lnc_degree > 1 else None,
                )
            else:
                return golden_quant_mlp(
                    inp_np=kernel_input,
                    norm_type=norm_type,
                    fused_add=fused_add,
                    store_add=store_add,
                    dtype=dtype,
                    quant_dtype=nl.float8_e4m3,
                    quantization_type=quant_type,
                    skip_gate=skip_gate,
                    act_fn_type=act_fn_type,
                    lnc=lnc_degree if lnc_degree > 1 else None,
                )

        test_manager.execute(
            KernelArgs(
                kernel_func=mlp,
                compiler_input=compiler_args,
                kernel_input=kernel_input,
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(
                        lazy_golden_generator=create_lazy_golden,
                        output_ndarray=output_tensors,
                    ),
                    relative_accuracy=rtol,
                    absolute_accuracy=1e-5,
                ),
            )
        )

    @staticmethod
    def mlp_cte_unit_config() -> RangeTestConfig:
        """Create unit test config from manual MLP CTE parameter sets."""
        all_unit_params = (
            MLP_CTE_UNIT_TEST_CASES_GATE_BIAS_FALSE
            + MLP_CTE_UNIT_TEST_CASES_GATE_BIAS_TRUE
            + MLP_CTE_UNIT_TEST_CASES_ROW_QUANT
            + MLP_CTE_UNIT_TEST_CASES_STATIC_QUANT
        )
        # Create manual config with test_type="manual"
        manual_config = create_mlp_cte_test_config(all_unit_params, test_type="manual")
        # Create model config with test_type=MODEL_TEST_TYPE
        model_config = create_mlp_cte_test_config(mlp_cte_model_configs, test_type=MODEL_TEST_TYPE)
        # Combine both configs by merging generators
        manual_config.global_tensor_configs.custom_generators.extend(
            model_config.global_tensor_configs.custom_generators
        )
        return manual_config

    @staticmethod
    def mlp_cte_sweep_config() -> RangeTestConfig:
        # Sweep seqlen and hidden dimensions with fixed batch
        MAX_SEQLEN = 8192
        MAX_HIDDEN = 15360
        MAX_INTERMEDIATE = 2048

        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=TensorRangeConfig(
                tensor_configs={
                    MLP_CTE_CONFIG: TensorConfig(
                        [
                            DimensionRangeConfig(min=1, max=1, name=BATCH_DIM_NAME, max_is_binding=False),
                            DimensionRangeConfig(min=128, max=MAX_SEQLEN, name=SEQUENCE_LEN_DIM_NAME),
                            DimensionRangeConfig(
                                min=128,
                                max=MAX_HIDDEN,
                                multiple_of=256,
                                name=HIDDEN_DIM_NAME,
                            ),
                            DimensionRangeConfig(
                                min=512,
                                max=MAX_INTERMEDIATE,
                                name=INTERMEDIATE_DIM_NAME,
                                max_is_binding=False,
                                min_is_binding=False,
                            ),
                        ]
                    ),
                },
                monotonic_step_percent=75,
            ),
        )

    @staticmethod
    def mlp_cte_batch_sweep_config() -> RangeTestConfig:
        # Sweep batch dimension with fixed seqlen and hidden
        MAX_BATCH = 4

        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=TensorRangeConfig(
                tensor_configs={
                    MLP_CTE_CONFIG: TensorConfig(
                        [
                            DimensionRangeConfig(min=1, max=MAX_BATCH, name=BATCH_DIM_NAME, max_is_binding=False),
                            DimensionRangeConfig(min=784, max=784, name=SEQUENCE_LEN_DIM_NAME),
                            DimensionRangeConfig(min=1280, max=1280, name=HIDDEN_DIM_NAME),
                            DimensionRangeConfig(
                                min=512,
                                max=512,
                                name=INTERMEDIATE_DIM_NAME,
                                max_is_binding=False,
                                min_is_binding=False,
                            ),
                        ]
                    ),
                },
                monotonic_step_percent=50,
            ),
        )

    # Test Entry Point
    @pytest.mark.fast
    @range_test_config(mlp_cte_unit_config())
    def test_mlp_cte_unit(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: IMetricsCollector,
        request,
        platform_target: Platforms,
    ):
        config = range_test_options.tensors[MLP_CTE_CONFIG]

        # Add metadata dimensions for model configs
        if range_test_options.test_type == MODEL_TEST_TYPE:
            mlp_cte_metadata_list = _get_mlp_cte_metadata()
            # Pass entire config dict - all params used for metadata matching
            collector.match_and_add_metadata_dimensions(config, mlp_cte_metadata_list)

        skip_gate_bias = bool(config[SKIP_GATE_DIM_NAME])
        if skip_gate_bias:
            pytest.skip("Skipping skip gate tests for now")

        lnc_count = config.get(VNC_DEGREE_DIM_NAME, None)
        compiler_args = CompilerArgs(logical_nc_config=lnc_count, platform_target=platform_target)
        self.run_range_mlp_cte_test(
            test_manager=test_manager,
            dtype=nl.bfloat16,
            test_options=range_test_options,
            lnc_degree=compiler_args.logical_nc_config,
            compiler_args=compiler_args,
            norm_type=NormType(config[NORM_TYPE_DIM_NAME]),
            quant_type=QuantizationType(config[QUANT_TYPE_DIM_NAME]),
            fused_add=bool(config[FUSED_ADD_DIM_NAME]),
            store_add=bool(config[STORE_ADD_DIM_NAME]),
            skip_gate=skip_gate_bias,
            act_fn_type=ActFnType(config[ACT_FN_TYPE_DIM_NAME]),
            gate_bias=bool(config[GATE_BIAS_DIM_NAME]),
            up_bias=bool(config[UP_BIAS_DIM_NAME]),
            down_bias=bool(config[DOWN_BIAS_DIM_NAME]),
            norm_bias=bool(config[NORM_BIAS_DIM_NAME]),
            collector=collector,
        )

    @range_test_config(mlp_cte_sweep_config())
    @pytest.mark.parametrize(
        "config_name,config",
        [
            ("basic", BASIC_MLP_CONFIG),
            ("full_features", FULL_FEATURES_CONFIG),
            ("layer_norm", LAYER_NORM_CONFIG),
        ],
    )
    def test_mlp_cte_sweep(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        config_name: str,
        config: dict[str, Any],
        collector: IMetricsCollector,
        platform_target: Platforms,
    ):
        compiler_args = CompilerArgs(platform_target=platform_target)
        self.run_range_mlp_cte_test(
            test_manager=test_manager,
            dtype=nl.bfloat16,
            test_options=range_test_options,
            lnc_degree=compiler_args.logical_nc_config,
            compiler_args=compiler_args,
            norm_type=config["norm_type"],
            quant_type=QuantizationType.NONE,
            fused_add=config["fused_add"],
            store_add=config["store_add"],
            skip_gate=config["skip_gate"],
            act_fn_type=config["act_fn_type"],
            gate_bias=config["gate_bias"],
            up_bias=config["up_bias"],
            down_bias=config["down_bias"],
            norm_bias=config["norm_bias"],
            collector=collector,
        )

    @range_test_config(mlp_cte_batch_sweep_config())
    @pytest.mark.parametrize(
        "config_name,config",
        [
            ("basic", BASIC_MLP_CONFIG),
            ("full_features", FULL_FEATURES_CONFIG),
            ("layer_norm", LAYER_NORM_CONFIG),
        ],
    )
    def test_mlp_cte_batch_sweep(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        config_name: str,
        config: dict[str, Any],
        collector: IMetricsCollector,
        platform_target: Platforms,
    ):
        compiler_args = CompilerArgs(platform_target=platform_target)
        self.run_range_mlp_cte_test(
            test_manager=test_manager,
            dtype=nl.bfloat16,
            test_options=range_test_options,
            lnc_degree=compiler_args.logical_nc_config,
            compiler_args=compiler_args,
            norm_type=config["norm_type"],
            quant_type=QuantizationType.NONE,
            fused_add=config["fused_add"],
            store_add=config["store_add"],
            skip_gate=config["skip_gate"],
            act_fn_type=config["act_fn_type"],
            gate_bias=config["gate_bias"],
            up_bias=config["up_bias"],
            down_bias=config["down_bias"],
            norm_bias=config["norm_bias"],
            collector=collector,
        )
