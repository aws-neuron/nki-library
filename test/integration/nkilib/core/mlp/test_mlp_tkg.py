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
import enum
import math

try:
    from test.integration.nkilib.core.mlp.test_mlp_tkg_model_config import (
        mlp_tkg_model_configs,
    )
except ImportError:
    mlp_tkg_model_configs = []

from test.integration.nkilib.core.mlp.test_mlp_common import (
    build_fused_norm_mlp,
    golden_mlp,
    golden_quant_mlp,
    modify_down_proj_lhs_rhs_swap_unit_stride_layout,
    random_lhs_and_random_bound_weight_tensor_generator,
)
from test.integration.nkilib.utils.tensor_generators import (
    gaussian_tensor_generator,
)
from test.utils.common_dataclasses import (
    MODEL_TEST_TYPE,
    TKG_INFERENCE_ARGS,
    CompilerArgs,
    KernelArgs,
    LazyGoldenGenerator,
    ValidationArgs,
)
from test.utils.metadata_loader import load_model_configs
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.ranged_test_harness import (
    DimensionRangeConfig,
    RangeManualGeneratorStrategy,
    RangeMonotonicGeneratorStrategy,
    RangeTestCase,
    RangeTestConfig,
    TensorConfig,
    TensorRangeConfig,
    assert_negative_test_case,
    range_test_config,
)
from test.utils.test_orchestrator import Orchestrator
from typing import final

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.mlp.mlp import mlp
from nkilib_src.nkilib.core.mlp.mlp_parameters import TKG_BS_SEQLEN_THRESHOLD
from nkilib_src.nkilib.core.utils.common_types import ActFnType, NormType, QuantizationType
from typing_extensions import override

MLP_TKG_CONFIG = "mlp_tkg_config"
VNC_DEGREE_DIM_NAME = "vnc"
BATCH_DIM_NAME = "b"
SEQUENCE_LEN_DIM_NAME = "s"
HIDDEN_DIM_NAME = "h"
INTERMEDIATE_DIM_NAME = "i"
DTYPE_DIM_NAME = "dt"
QUANTIZATION_DTYPE_DIM_NAME = "q_dt"
QUANTIZATION_TYPE_DIM_NAME = "q_t"
NORM_TYPE_DIM_NAME = "norm_type"
FUSED_ADD_DIM_NAME = "fa"
STORE_ADD_DIM_NAME = "sa"
SKIP_GATE_DIM_NAME = "skip_gate"
ACT_FN_TYPE_DIM_NAME = "act_fn_type"
GATE_BIAS_DIM_NAME = "gb"
UP_BIAS_DIM_NAME = "ub"
DOWN_BIAS_DIM_NAME = "db"
NORM_BIAS_DIM_NAME = "nb"
USE_TKG_GATE_UP_PROJ_COLUMN_TILING_DIM_NAME = "gate_col"
USE_TKG_DOWN_PROJ_COLUMN_TILING_DIM_NAME = "down_col"
USE_TKG_DOWN_PROJ_OPTIMIZED_LAYOUT_DIM_NAME = "down_opt"
DUMMY_STEP_SIZE = 1  # Placeholder value that will be overridden by custom generators and/or DimensionRangeConfigs

# ----------------------------------------------------
# Configuration-based testing to avoid combinatorial explosion
# ----------------------------------------------------

COLUMN_TILING_BASIC_CONFIG = {
    "dtype": nl.bfloat16,
    "norm_type": NormType.NO_NORM,
    "fused_add": False,
    "store_add": False,
    "act_fn_type": ActFnType.SiLU,
    "gate_bias": False,
    "up_bias": False,
    "down_bias": False,
    "norm_bias": False,
    "quant_dtype": None,
    "use_tkg_gate_up_proj_column_tiling": True,
    "use_tkg_down_proj_column_tiling": True,
    "use_tkg_down_proj_optimized_layout": False,
}

COLUMN_TILING_FULL_FEATURE_RMSNORM_CONFIG = {
    "dtype": nl.bfloat16,
    "norm_type": NormType.RMS_NORM,
    "fused_add": True,
    "store_add": True,
    "act_fn_type": ActFnType.SiLU,
    "gate_bias": True,
    "up_bias": True,
    "down_bias": True,
    "norm_bias": False,
    "quant_dtype": None,
    "use_tkg_gate_up_proj_column_tiling": True,
    "use_tkg_down_proj_column_tiling": True,
    "use_tkg_down_proj_optimized_layout": False,
}

COLUMN_TILING_FULL_FEATURE_LAYERNORM_CONFIG = {
    "dtype": nl.bfloat16,
    "norm_type": NormType.LAYER_NORM,
    "fused_add": True,
    "store_add": True,
    "act_fn_type": ActFnType.SiLU,
    "gate_bias": True,
    "up_bias": True,
    "down_bias": True,
    "norm_bias": True,
    "quant_dtype": None,
    "use_tkg_gate_up_proj_column_tiling": True,
    "use_tkg_down_proj_column_tiling": True,
    "use_tkg_down_proj_optimized_layout": False,
}

NON_COLUMN_TILING_BASIC_CONFIG = {
    "dtype": nl.bfloat16,
    "norm_type": NormType.NO_NORM,
    "fused_add": False,
    "store_add": False,
    "act_fn_type": ActFnType.SiLU,
    "gate_bias": False,
    "up_bias": False,
    "down_bias": False,
    "norm_bias": False,
    "quant_dtype": None,
    "use_tkg_gate_up_proj_column_tiling": False,
    "use_tkg_down_proj_column_tiling": False,
    "use_tkg_down_proj_optimized_layout": False,
}

NON_COLUMN_TILING_FULL_FEATURE_CONFIG = {
    "dtype": nl.bfloat16,
    "norm_type": NormType.RMS_NORM,
    "fused_add": True,
    "store_add": True,
    "act_fn_type": ActFnType.SiLU,
    "gate_bias": True,
    "up_bias": True,
    "down_bias": True,
    "norm_bias": False,
    "quant_dtype": None,
    "use_tkg_gate_up_proj_column_tiling": False,
    "use_tkg_down_proj_column_tiling": False,
    "use_tkg_down_proj_optimized_layout": True,
}


# Classification for test size
class MLPTKGClassification(enum.Enum):
    SMALL = 1
    MEDIUM = 2
    LARGE = 3

    @staticmethod
    def classify(batch: int, seqlen: int, hidden: int, intermediate: int):
        total_elements = batch * seqlen * hidden

        if total_elements <= 1024 * 1024:
            return MLPTKGClassification.SMALL
        elif total_elements <= 16 * 1024 * 1024:
            return MLPTKGClassification.MEDIUM
        else:
            return MLPTKGClassification.LARGE

    @override
    def __str__(self):
        return self.name


# fmt: off
# Parameters: vnc_degree, batch, seqlen, hidden, intermediate, dtype, quant_dtype, quant_type,
#             tpbSgCyclesSum, norm_type, fused_add, store_add, act_fn_type, skip_gate_proj,
#             gate_bias, up_bias, down_bias, norm_bias, use_tkg_gate_up_proj_column_tiling,
#             use_tkg_down_proj_column_tiling, use_tkg_down_proj_optimized_layout
nki_tkg_fused_norm_mlp_kernel_spmd_vnc2_params = [
    [2, 2, 4, 8448, 1408, nl.bfloat16, None, QuantizationType.NONE, 146806437, NormType.LAYER_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 4, 2, 4096, 1024, nl.bfloat16, None, QuantizationType.NONE, 83028204, NormType.LAYER_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 1, 1, 8448, 1408, nl.bfloat16, None, QuantizationType.NONE, 135269789, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 1, 1, 8192, 448, nl.bfloat16, None, QuantizationType.NONE, 62157403, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 4, 5, 8192, 896, nl.bfloat16, None, QuantizationType.NONE, 122126476, NormType.RMS_NORM, True, True, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 1, 5, 16384, 832, nl.bfloat16, None, QuantizationType.NONE, 153948092, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 4, 7, 16384, 832, nl.bfloat16, None, QuantizationType.NONE, 157358921, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    # GPT-OSS draft
    [2, 64, 1, 3072, 135, nl.bfloat16, None, QuantizationType.NONE, 50378255, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 4, 1, 3072, 2160, nl.bfloat16, None, QuantizationType.NONE, 102256507, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    # Llama high batch with 2x array tiling
    [2, 8, 5, 8192, 896, nl.bfloat16, None, QuantizationType.NONE, 115811486, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 8, 5, 16384, 896, nl.bfloat16, None, QuantizationType.NONE, 172327230, NormType.RMS_NORM, True, True, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    # Llama3 1B fused speculation
    [2, 1, 1, 2048, 512, nl.bfloat16, None, QuantizationType.NONE, 42469100, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    # Llama3 8B fused speculation
    [2, 1, 1, 4096, 896, nl.bfloat16, None, QuantizationType.NONE, 66799895, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    # Text
    [2, 1, 1, 7168, 364, nl.bfloat16, None, QuantizationType.NONE, 50260755, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    #  Llama3 2T fused speculation
    [2, 1, 5, 32768, 896, nl.bfloat16, None, QuantizationType.NONE, 292073710, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 1, 5, 32768, 896, nl.bfloat16, None, QuantizationType.NONE, 300053697, NormType.RMS_NORM, True, True, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    #  Llama3 470B
    [2, 1, 5, 20480, 832, nl.bfloat16, None, QuantizationType.NONE, 179771386, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 2, 7, 20480, 832, nl.bfloat16, None, QuantizationType.NONE, 197313858, NormType.RMS_NORM, True, True, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    # Llama 4 sharedExpert
    [2, 4, 1, 5120, 128, nl.bfloat16, None, QuantizationType.NONE, 34008280, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 8, 7, 16384, 896, nl.bfloat16, None, QuantizationType.NONE, 190164703, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, True, True, False],
    [2, 8, 5, 8192, 512, nl.bfloat16, None, QuantizationType.NONE, 93846520, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, True, True, False],
    # Functional Test I > 4096
    [2, 4, 1, 8192, 5120, nl.bfloat16, None, QuantizationType.NONE, 435362653, NormType.NO_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, True, True, False],
    # Store Add, sb_input feature in rmsnorm/layernorm
    [2, 4, 1, 8192, 512, nl.bfloat16, None, QuantizationType.NONE, 75694882, NormType.RMS_NORM, True, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 4, 8, 8192, 512, nl.bfloat16, None, QuantizationType.NONE, 109073163, NormType.RMS_NORM, True, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 4, 1, 8192, 512, nl.bfloat16, None, QuantizationType.NONE, 82688204, NormType.LAYER_NORM, True, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 4, 8, 8192, 512, nl.bfloat16, None, QuantizationType.NONE, 104209837, NormType.LAYER_NORM, True, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
]

nki_tkg_fused_norm_mlp_kernel_spmd_vnc1_params = [
    [1, 1, 1, 8192, 448, nl.bfloat16, None, QuantizationType.NONE, 76487380, NormType.RMS_NORM, True, True, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [1, 1, 1, 7168, 896, nl.bfloat16, None, QuantizationType.NONE, 112156491, NormType.RMS_NORM, True, True, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [1, 1, 1, 16384, 416, nl.bfloat16, None, QuantizationType.NONE, 118365648, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [1, 1, 1, 16384, 832, nl.bfloat16, None, QuantizationType.NONE, 224507149, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    # Speculative tests, seqlen > 1
    [1, 3, 8, 8192, 832, nl.bfloat16, None, QuantizationType.NONE, 124893971, NormType.LAYER_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [1, 4, 2, 16384, 832, nl.bfloat16, None, QuantizationType.NONE, 229137142, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    # Bias test.
    [1, 4, 2, 16384, 832, nl.bfloat16, None, QuantizationType.NONE, 242479621, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, True, True, False],
    # Bias test with larger BxS
  	[1, 32, 2, 8192, 512, nl.bfloat16, None, QuantizationType.NONE, None, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, True, True, False],
]

nki_tkg_fused_norm_mlp_kernel_spmd_vnc2_swap_perms = [
    # LLaMA4 sharedExpert
    [2, 1, 1, 4096, 128, nl.bfloat16, None, QuantizationType.NONE, 25566627, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, True, False],
    [2, 4, 1, 5120, 256, nl.bfloat16, None, QuantizationType.NONE, 45419096, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False],
    [2, 8, 1, 5120, 128, nl.bfloat16, None, QuantizationType.NONE, 41267436, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False],
    [2, 16, 1, 5120, 128, nl.bfloat16, None, QuantizationType.NONE, 45513262, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False],
    [2, 32, 1, 5120, 128, nl.bfloat16, None, QuantizationType.NONE, 47358259, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False],
    # LLaMA3 TP64
    [2, 4, 1, 8192, 512, nl.bfloat16, None, QuantizationType.NONE, 82889870, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False],
    # LLaMA3 TP32
    [2, 4, 5, 8192, 1024, nl.bfloat16, None, QuantizationType.NONE, 131043962, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False],
    # Functional Test I > 1024
    [2, 4, 5, 8192, 1560, nl.bfloat16, None, QuantizationType.NONE, 182945547, NormType.NO_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, False, False],
    # Functional test
    [2, 4, 1, 16384, 832, nl.bfloat16, None, QuantizationType.NONE, 153945593, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 4, 1, 16384, 832, nl.bfloat16, None, QuantizationType.NONE, 186434709, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, False, False],
    [2, 4, 1, 16384, 832, nl.bfloat16, None, QuantizationType.NONE, 150452265, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, True, False],
    [2, 4, 1, 16384, 832, nl.bfloat16, None, QuantizationType.NONE, 189018038, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False],
    # Llama 3 TP32 all projections swapped, down_w layout optimized for unit stride loading
    [1, 4, 1, 8192, 1024, nl.bfloat16, None, QuantizationType.NONE, 166719739, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, True],
    [2, 4, 1, 8192, 1024, nl.bfloat16, None, QuantizationType.NONE, 117065650, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, True],
]

nki_tkg_fused_norm_mlp_kernel_spmd_skip_gate = [
    # Functional test
    [2, 4, 1, 16384, 832, nl.bfloat16, None, QuantizationType.NONE, 109052330, NormType.NO_NORM, False, False, ActFnType.SiLU, True, False, False, False, False, True, True, False],
    [2, 4, 1, 16384, 832, nl.bfloat16, None, QuantizationType.NONE, 146529771, NormType.NO_NORM, False, False, ActFnType.SiLU, True, False, False, False, False, True, False, False],
    [2, 4, 1, 16384, 832, nl.bfloat16, None, QuantizationType.NONE, 116152318, NormType.NO_NORM, False, False, ActFnType.SiLU, True, False, False, False, False, False, True, False],
    [2, 4, 1, 16384, 832, nl.bfloat16, None, QuantizationType.NONE, 142690611, NormType.NO_NORM, False, False, ActFnType.SiLU, True, False, False, False, False, False, False, False],
]

nki_tkg_fused_norm_mlp_row_quant_kernel_params = [
    [2, 1, 1, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 106305668, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [1, 1, 1, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 140940613, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [1, 1, 5, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 145970605, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 1, 5, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 108908163, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 1, 7, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 105090669, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 2, 7, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 109548162, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [1, 4, 7, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 158341419, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 8, 7, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 152673928, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    # BxS > 64
    [2, 14, 5, 8192, 128, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 64228233, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 16, 5, 8192, 128, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 62225736, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    # LLaMA3 70B
    [2, 4, 5, 8192, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 77140713, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 8, 5, 8192, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 87974029, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    # llama3 470B
    [2, 4, 7, 20480, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 126873135, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 2, 7, 20480, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 134724789, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 4, 7, 20480, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 145678105, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    #llama3-2T
    [2, 4, 7, 32768, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, None, NormType.NO_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, True, True, False],
    [2, 2, 7, 32768, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 188562206, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 4, 7, 32768, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 186326376, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    # Functional Test I > 4096
    [2, 1, 1, 16384, 4986, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, None, NormType.NO_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, True, True, False],
]

nki_tkg_fused_norm_mlp_row_quant_kernel_layout_swap_perms = [
    # LLaMA3 70B TP64-BS8
    [2, 8, 5, 8192, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, None, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False],
    # LLaMA3 70B TP32-DP2-BS4 (effective BS8)
    [2, 4, 5, 8192, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 112488158, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False],
    # With gate/up column tiling
    # LLaMA3 70B TP64-BS8
    [2, 8, 5, 8192, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 83822369, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, True, False],
    # LLaMA3 70B TP32-DP2-BS4 (effective BS8)
    [2, 4, 5, 8192, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 93608187, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, True, False],
    # Column tiling / swap option minimal tests
    [2, 4, 5, 1024, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 51984086, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 4, 5, 1024, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 51378253, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, False, False],
    [2, 4, 5, 1024, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 43303265, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, True, False],
    [2, 4, 5, 1024, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 48179925, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False],
    # Llama 3 TP32 all projections swapped, down_w layout optimized for unit stride loading
    [1, 4, 1, 8192, 1024, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 127450635, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, True],
    [2, 4, 1, 8192, 1024, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 87849029, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, True],
    # Column tiling / swap option minimal tests
    [2, 4, 5, 1024, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 51984086, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 4, 5, 1024, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 51378253, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, False, False],
    [2, 4, 5, 1024, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 43303265, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, True, False],
    [2, 4, 5, 1024, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 48179925, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False],
    # Bias test
    [2, 4, 1, 8192, 893, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 89110694, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, True, True, False],
    [2, 4, 1, 8192, 893, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 95652350, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, True, False, False],
    [2, 4, 1, 8192, 893, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 71973221, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, True, False],
    [2, 4, 1, 8192, 893, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 88949028, NormType.NO_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, False, False],
    # Functional Test I > 1024
    [2, 4, 1, 8192, 1560, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 131467295, NormType.NO_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, False, False],
]

nki_tkg_fused_norm_mlp_static_quant_kernel_params = [
    [2, 1, 1, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 105202336, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [1, 1, 1, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 151766430, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [1, 1, 5, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 166568907, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 1, 5, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 110590661, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 1, 7, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 112503157, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 2, 7, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 119785646, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [1, 4, 7, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 177883055, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 8, 7, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 147633102, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    # BxS > 64
    [2, 14, 5, 8192, 128, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 57351578, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 16, 5, 8192, 128, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 56858245, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    # LLaMA3 70B
    [2, 4, 5, 8192, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 79875708, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 8, 5, 8192, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 89981526, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    # llama3 470B
    [2, 4, 7, 20480, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 141169779, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 2, 7, 20480, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 137068120, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 4, 7, 20480, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 139171450, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    #llama3-2T
    [2, 4, 7, 32768, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, None, NormType.NO_NORM, False, False, ActFnType.SiLU, False, True, True, False, False, True, True, False],
    [2, 2, 7, 32768, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 201567185, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 4, 7, 32768, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 202325517, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    # Functional Test I > 4096
    [2, 4, 7, 8192, 5120, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, None, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, True, True, False],
]

nki_tkg_fused_norm_mlp_static_quant_kernel_layout_swap_perms = [
    # LLaMA3 70B TP64-BS8
    [2, 8, 5, 8192, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, None, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False],
    # LLaMA3 70B TP32-DP2-BS4 (effective BS8)
    [2, 4, 5, 8192, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 95099018, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False],
    # With gate/up column tiling
    # LLaMA3 70B TP64-BS8
    [2, 8, 5, 8192, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 88882361, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, True, False],
    # LLaMA3 70B TP32-DP2-BS4 (effective BS8)
    [2, 4, 5, 8192, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 87385697, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, True, False],
    # Column tiling / swap option minimal tests
    [2, 4, 5, 1024, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 46429095, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 4, 5, 1024, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 48652424, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, False, False],
    [2, 4, 5, 1024, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 42154934, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, True, False],
    [2, 4, 5, 1024, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 43614932, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False],
    # Llama 3 TP32 all projections swapped, down_w layout optimized for unit stride loading
    [1, 4, 1, 8192, 1024, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 124933138, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, True],
    [2, 4, 1, 8192, 1024, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 86152365, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, True],
    # Column tiling / swap option minimal tests
    [2, 4, 5, 1024, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 46429095, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False],
    [2, 4, 5, 1024, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 48652424, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, False, False],
    [2, 4, 5, 1024, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 42154934, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, True, False],
    [2, 4, 5, 1024, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 43614932, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False],
    # Bias test
    [2, 4, 1, 8192, 893, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 91222358, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, True, True, False],
    [2, 4, 1, 8192, 893, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 99567345, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, True, False, False],
    [2, 4, 1, 8192, 893, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 75601548, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, True, False],
    [2, 4, 1, 8192, 893, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 87649030, NormType.NO_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, False, False],
    # Functional Test I > 1024
    [2, 4, 1, 8192, 1560, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 132462293, NormType.NO_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, False, False],
]
# fmt: on


def create_mlp_tkg_test_config(test_vectors, test_type: str = "manual"):
    """
    Utility function to create complete RangeTestConfig from list of MLP TKG config vectors.

    Args:
        test_vectors: List of test config vectors
        test_type: Test type label for test names

    Returns:
        Complete RangeTestConfig ready to use with @range_test_config decorator
    """
    test_cases = []

    for test_params in test_vectors:
        (
            vnc_degree,
            batch,
            seqlen,
            hidden,
            intermediate,
            dtype,
            quant_dtype,
            quant_type,
            tpbSgCyclesSum,
            norm_type,
            fused_add,
            store_add,
            act_fn_type,
            skip_gate_proj,
            gate_bias,
            up_bias,
            down_bias,
            norm_bias,
            use_tkg_gate_up_proj_column_tiling,
            use_tkg_down_proj_column_tiling,
            use_tkg_down_proj_optimized_layout,
        ) = test_params

        test_case = {
            MLP_TKG_CONFIG: {
                VNC_DEGREE_DIM_NAME: vnc_degree,
                BATCH_DIM_NAME: batch,
                SEQUENCE_LEN_DIM_NAME: seqlen,
                HIDDEN_DIM_NAME: hidden,
                INTERMEDIATE_DIM_NAME: intermediate,
                DTYPE_DIM_NAME: dtype,
                QUANTIZATION_DTYPE_DIM_NAME: quant_dtype,
                QUANTIZATION_TYPE_DIM_NAME: quant_type,
                NORM_TYPE_DIM_NAME: norm_type.value,
                FUSED_ADD_DIM_NAME: int(fused_add),
                STORE_ADD_DIM_NAME: int(store_add),
                ACT_FN_TYPE_DIM_NAME: act_fn_type.value,
                SKIP_GATE_DIM_NAME: int(skip_gate_proj),
                GATE_BIAS_DIM_NAME: int(gate_bias),
                UP_BIAS_DIM_NAME: int(up_bias),
                DOWN_BIAS_DIM_NAME: int(down_bias),
                NORM_BIAS_DIM_NAME: int(norm_bias),
                USE_TKG_GATE_UP_PROJ_COLUMN_TILING_DIM_NAME: int(use_tkg_gate_up_proj_column_tiling),
                USE_TKG_DOWN_PROJ_COLUMN_TILING_DIM_NAME: int(use_tkg_down_proj_column_tiling),
                USE_TKG_DOWN_PROJ_OPTIMIZED_LAYOUT_DIM_NAME: int(use_tkg_down_proj_optimized_layout),
            }
        }
        test_cases.append(test_case)

    generators = [RangeManualGeneratorStrategy(test_cases=test_cases, test_type=test_type)]

    return RangeTestConfig(
        additional_params={},
        global_tensor_configs=TensorRangeConfig(
            tensor_configs={},
            monotonic_step_size=1,
            custom_generators=generators,
        ),
    )


# Load model metadata for matching test configs to model names
mlp_tkg_metadata_list = load_model_configs("test_mlp_tkg")


@pytest_test_metadata(
    name="MLP TKG",
    pytest_marks=["mlp", "tkg"],
    tags=["model"],
)
@final
class TestMlpTkgKernels:
    def run_range_mlp_tkg_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        test_options: RangeTestCase,
        lnc_degree,
        dtype,
        norm_type,
        fused_add,
        store_add,
        act_fn_type,
        gate_bias,
        up_bias,
        down_bias,
        norm_bias,
        quant_dtype,
        quantization_type: QuantizationType,
        skip_gate_proj,
        use_tkg_gate_up_proj_column_tiling,
        use_tkg_down_proj_column_tiling,
        use_tkg_down_proj_optimized_layout,
        collector: IMetricsCollector,
        gate_clamp_lower_limit=None,
        gate_clamp_upper_limit=None,
        up_clamp_lower_limit=None,
        up_clamp_upper_limit=None,
    ):
        # Model configs should never be marked as negative test cases
        is_model_config = test_options.test_type == MODEL_TEST_TYPE
        is_negative_test_case = False if is_model_config else test_options.is_negative_test_case

        mlp_config = test_options.tensors[MLP_TKG_CONFIG]

        batch = mlp_config[BATCH_DIM_NAME]
        seqlen = mlp_config[SEQUENCE_LEN_DIM_NAME]
        hidden = mlp_config[HIDDEN_DIM_NAME]
        intermediate = mlp_config[INTERMEDIATE_DIM_NAME]

        # BxS (batch * seqlen) must not exceed 128 partitions
        if batch * seqlen > 128 and not is_model_config:
            is_negative_test_case = True

        # H1 must be evenly divisible by num_shards for LNC2 sharding
        if lnc_degree is not None and lnc_degree > 1:  # LNC2 sharding
            H1 = hidden // 128
            if H1 % lnc_degree != 0 and not is_model_config:
                is_negative_test_case = True

        # Hidden for each core must be divisible by 128
        if hidden // lnc_degree % 128 != 0 and not is_model_config:
            is_negative_test_case = True

        # Implementation restriction when use_tkg_down_proj_column_tiling is False
        if not use_tkg_down_proj_column_tiling and not is_model_config:
            T = batch * seqlen
            H1_shard = hidden // 128 // lnc_degree
            perBankT = 512 // T  # 512 = psum_fmax
            num_required_down_psum_banks = math.ceil(H1_shard / perBankT)
            if num_required_down_psum_banks > 8:  # psum_bmax
                is_negative_test_case = True

        # ----------------------------------------------------
        # CloudWatch metrics for kernel test outcome
        # ----------------------------------------------------
        test_size_classification = MLPTKGClassification.classify(batch, seqlen, hidden, intermediate)

        # ----------------------------------------------------
        with assert_negative_test_case(is_negative_test_case):
            self.run_mlp_tkg_test(
                test_manager=test_manager,
                compiler_args=compiler_args,
                collector=collector,
                batch=batch,
                hidden=hidden,
                intermediate=intermediate,
                seqlen=seqlen,
                dtype=dtype,
                lnc_degree=lnc_degree,
                act_fn_type=act_fn_type,
                fused_add=fused_add,
                gate_bias=gate_bias,
                up_bias=up_bias,
                down_bias=down_bias,
                norm_bias=norm_bias,
                norm_type=norm_type,
                quant_dtype=quant_dtype,
                quantization_type=quantization_type,
                skip_gate_proj=skip_gate_proj,
                store_add=store_add,
                use_tkg_gate_up_proj_column_tiling=use_tkg_gate_up_proj_column_tiling,
                use_tkg_down_proj_column_tiling=use_tkg_down_proj_column_tiling,
                use_tkg_down_proj_optimized_layout=use_tkg_down_proj_optimized_layout,
                gate_clamp_upper_limit=gate_clamp_upper_limit,
                gate_clamp_lower_limit=gate_clamp_lower_limit,
                up_clamp_lower_limit=up_clamp_lower_limit,
                up_clamp_upper_limit=up_clamp_upper_limit,
            )

    def run_mlp_tkg_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        collector: IMetricsCollector,
        batch: int,
        hidden: int,
        intermediate: int,
        seqlen: int,
        dtype,
        lnc_degree,
        act_fn_type,
        fused_add,
        gate_bias,
        up_bias,
        down_bias,
        norm_bias,
        norm_type,
        quant_dtype,
        quantization_type: QuantizationType,
        skip_gate_proj,
        store_add,
        use_tkg_gate_up_proj_column_tiling: bool = True,
        use_tkg_down_proj_column_tiling: bool = True,
        use_tkg_down_proj_optimized_layout: bool = False,
        gate_clamp_upper_limit=None,
        gate_clamp_lower_limit=None,
        up_clamp_lower_limit=None,
        up_clamp_upper_limit=None,
    ):
        tensor_generator = gaussian_tensor_generator()
        if quant_dtype is not None:
            tensor_generator = random_lhs_and_random_bound_weight_tensor_generator(0, 241)
            if use_tkg_down_proj_optimized_layout:
                tensor_generator = random_lhs_and_random_bound_weight_tensor_generator(
                    0,
                    241,
                    modifier_fn=modify_down_proj_lhs_rhs_swap_unit_stride_layout,
                    lnc=lnc_degree,
                )
        else:
            if use_tkg_down_proj_optimized_layout:
                tensor_generator = gaussian_tensor_generator(
                    0,
                    241,
                    modifier_fn=modify_down_proj_lhs_rhs_swap_unit_stride_layout,
                    lnc=lnc_degree,
                )

        kernel_input = build_fused_norm_mlp(
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            intermediate=intermediate,
            dtype=dtype,
            quant_dtype=quant_dtype,
            quantization_type=quantization_type,
            fused_add=fused_add,
            norm_type=norm_type,
            store_add=store_add,
            lnc_degree=lnc_degree,
            act_fn_type=act_fn_type,
            gate_bias=gate_bias,
            up_bias=up_bias,
            down_bias=down_bias,
            norm_bias=norm_bias,
            skip_gate=skip_gate_proj,
            use_tkg_gate_up_proj_column_tiling=use_tkg_gate_up_proj_column_tiling,
            use_tkg_down_proj_column_tiling=use_tkg_down_proj_column_tiling,
            use_tkg_down_proj_optimized_layout=use_tkg_down_proj_optimized_layout,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            tensor_generator=tensor_generator,
        )

        # Build output placeholder tensors for shape/dtype metadata
        output_shape = (batch, seqlen, hidden)
        store_add_valid = fused_add and store_add
        output_ndarrays = {
            "out": np.zeros(output_shape, dtype=dtype),
        }
        if store_add_valid:
            output_ndarrays["add_out"] = np.zeros(output_shape, dtype=dtype)

        # Create lazy golden generator with closure capturing all necessary variables
        def create_lazy_golden():
            if quant_dtype is not None:
                return golden_quant_mlp(
                    inp_np=kernel_input,
                    fused_rmsnorm=(norm_type == NormType.RMS_NORM),
                    fused_add=fused_add,
                    store_add=store_add,
                    dtype=dtype,
                    quant_dtype=quant_dtype,
                    quantization_type=quantization_type,
                    skip_gate=skip_gate_proj,
                    act_fn_type=act_fn_type,
                    lnc=lnc_degree,
                    down_proj_lhs_rhs_swap_optimized_layout=use_tkg_down_proj_optimized_layout,
                    gate_clamp_lower_limit=gate_clamp_lower_limit,
                    gate_clamp_upper_limit=gate_clamp_upper_limit,
                    up_clamp_lower_limit=up_clamp_lower_limit,
                    up_clamp_upper_limit=up_clamp_upper_limit,
                    quantize_activation=True,
                )
            else:
                return golden_mlp(
                    inp_np=kernel_input,
                    norm_type=norm_type,
                    fused_add=fused_add,
                    store_add=store_add,
                    dtype=dtype,
                    skip_gate=skip_gate_proj,
                    act_fn_type=act_fn_type,
                    lnc=lnc_degree,
                    down_proj_lhs_rhs_swap_optimized_layout=use_tkg_down_proj_optimized_layout,
                    gate_clamp_lower_limit=gate_clamp_lower_limit,
                    gate_clamp_upper_limit=gate_clamp_upper_limit,
                    up_clamp_lower_limit=up_clamp_lower_limit,
                    up_clamp_upper_limit=up_clamp_upper_limit,
                )

        test_manager.execute(
            KernelArgs(
                kernel_func=mlp,
                compiler_input=compiler_args,
                kernel_input=kernel_input,
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(
                        lazy_golden_generator=create_lazy_golden,
                        output_ndarray=output_ndarrays,
                    ),
                    relative_accuracy=2e-2,
                    absolute_accuracy=1e-5,
                ),
                inference_args=TKG_INFERENCE_ARGS,
            )
        )

    @staticmethod
    def mlp_tkg_unit_config():
        """
        Create unit test config from all MLP TKG parameter sets.

        Returns:
            RangeTestConfig for unit tests
        """
        all_unit_test_params = (
            nki_tkg_fused_norm_mlp_kernel_spmd_vnc2_params
            + nki_tkg_fused_norm_mlp_kernel_spmd_vnc1_params
            + nki_tkg_fused_norm_mlp_kernel_spmd_vnc2_swap_perms
            + nki_tkg_fused_norm_mlp_kernel_spmd_skip_gate
            + nki_tkg_fused_norm_mlp_row_quant_kernel_params
            + nki_tkg_fused_norm_mlp_row_quant_kernel_layout_swap_perms
            + nki_tkg_fused_norm_mlp_static_quant_kernel_params
            + nki_tkg_fused_norm_mlp_static_quant_kernel_layout_swap_perms
        )
        # Create manual config with test_type="manual"
        manual_config = create_mlp_tkg_test_config(all_unit_test_params, test_type="manual")
        # Create model config with test_type=MODEL_TEST_TYPE
        model_config = create_mlp_tkg_test_config(mlp_tkg_model_configs, test_type=MODEL_TEST_TYPE)
        # Combine both configs by merging generators
        manual_config.global_tensor_configs.custom_generators.extend(
            model_config.global_tensor_configs.custom_generators
        )
        return manual_config

    @staticmethod
    def mlp_tkg_sweep_config():
        MAX_BATCH = TKG_BS_SEQLEN_THRESHOLD // 2
        MAX_SEQLEN = 2  # MLP TKG always processes BxS, so we keep max seqlen as 1
        MAX_HIDDEN = 32768
        MAX_INTERMEDIATE = 1024

        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=TensorRangeConfig(
                tensor_configs={
                    MLP_TKG_CONFIG: TensorConfig(
                        [
                            DimensionRangeConfig(min=1, max=MAX_BATCH, power_of=2, name=BATCH_DIM_NAME),
                            DimensionRangeConfig(
                                min=1,
                                max=MAX_SEQLEN,
                                power_of=2,
                                name=SEQUENCE_LEN_DIM_NAME,
                            ),
                            DimensionRangeConfig(
                                min=128,
                                max=MAX_HIDDEN,
                                power_of=2,
                                name=HIDDEN_DIM_NAME,
                            ),
                            DimensionRangeConfig(
                                min=128,
                                max=MAX_INTERMEDIATE,
                                multiple_of=128,
                                name=INTERMEDIATE_DIM_NAME,
                            ),
                        ]
                    ),
                },
                monotonic_step_size=DUMMY_STEP_SIZE,
                custom_generators=[RangeMonotonicGeneratorStrategy(step_size=DUMMY_STEP_SIZE)],
            ),
        )

    @staticmethod
    def mlp_tkg_basic_sweep_config():
        # One representative test vector
        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=TensorRangeConfig(
                tensor_configs={
                    MLP_TKG_CONFIG: TensorConfig(
                        [
                            DimensionRangeConfig(min=4, max=4, power_of=2, name=BATCH_DIM_NAME),
                            DimensionRangeConfig(min=1, max=1, power_of=2, name=SEQUENCE_LEN_DIM_NAME),
                            DimensionRangeConfig(min=8192, max=8192, power_of=2, name=HIDDEN_DIM_NAME),
                            DimensionRangeConfig(min=416, max=416, multiple_of=128, name=INTERMEDIATE_DIM_NAME),
                        ]
                    ),
                },
                monotonic_step_size=DUMMY_STEP_SIZE,
                custom_generators=[RangeMonotonicGeneratorStrategy(step_size=DUMMY_STEP_SIZE)],
            ),
        )

    @staticmethod
    def mlp_tkg_I_non_multiple_of_128_sweep_config():
        # testing large I, and I non multiple of 128
        MAX_BATCH = 16
        MAX_SEQLEN = 1  # MLP TKG always processes BxS, so we keep max seqlen as 1
        MIN_HIDDEN = 8192
        MAX_HIDDEN = 16384
        MIN_INTERMEDIATE = 412
        MAX_INTERMEDIATE = 5120

        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=TensorRangeConfig(
                tensor_configs={
                    MLP_TKG_CONFIG: TensorConfig(
                        [
                            DimensionRangeConfig(min=1, max=MAX_BATCH, power_of=2, name=BATCH_DIM_NAME),
                            DimensionRangeConfig(min=1, max=MAX_SEQLEN, name=SEQUENCE_LEN_DIM_NAME),
                            DimensionRangeConfig(min=MIN_HIDDEN, max=MAX_HIDDEN, power_of=2, name=HIDDEN_DIM_NAME),
                            DimensionRangeConfig(
                                min=MIN_INTERMEDIATE,
                                max=MAX_INTERMEDIATE,
                                multiple_of=896,
                                name=INTERMEDIATE_DIM_NAME,
                            ),
                        ]
                    ),
                },
                monotonic_step_size=DUMMY_STEP_SIZE,
                custom_generators=[RangeMonotonicGeneratorStrategy(step_size=DUMMY_STEP_SIZE)],
            ),
        )

    @staticmethod
    def mlp_tkg_store_add_false_sweep_config():
        # testing large I, and I non multiple of 128
        MAX_BATCH = 4
        MAX_SEQLEN = 1  # MLP TKG always processes BxS, so we keep max seqlen as 1
        MIN_HIDDEN = 8192
        MAX_HIDDEN = 32768
        INTERMEDIATE = 1024  # Intermediate size is not related with this test

        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=TensorRangeConfig(
                tensor_configs={
                    MLP_TKG_CONFIG: TensorConfig(
                        [
                            DimensionRangeConfig(min=1, max=MAX_BATCH, power_of=2, name=BATCH_DIM_NAME),
                            DimensionRangeConfig(min=1, max=MAX_SEQLEN, name=SEQUENCE_LEN_DIM_NAME),
                            DimensionRangeConfig(
                                min=MIN_HIDDEN,
                                max=MAX_HIDDEN,
                                power_of=2,
                                name=HIDDEN_DIM_NAME,
                            ),
                            DimensionRangeConfig(
                                min=INTERMEDIATE,
                                max=INTERMEDIATE,
                                name=INTERMEDIATE_DIM_NAME,
                            ),
                        ]
                    ),
                },
                monotonic_step_size=DUMMY_STEP_SIZE,
                custom_generators=[RangeMonotonicGeneratorStrategy(step_size=DUMMY_STEP_SIZE)],
            ),
        )

    # Unit Tests Entry Point
    @pytest.mark.fast
    @range_test_config(mlp_tkg_unit_config())
    def test_mlp_tkg_unit(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: IMetricsCollector,
        request,
    ):
        # Get the tensor config
        mlp_config = range_test_options.tensors[MLP_TKG_CONFIG]

        # Apply xfail for model configs and add metadata dimensions
        if range_test_options.test_type == MODEL_TEST_TYPE:
            request.node.add_marker(pytest.mark.xfail(strict=False, reason="Model coverage test"))
            test_metadata_key = {
                "vnc": mlp_config[VNC_DEGREE_DIM_NAME],
                "b": mlp_config[BATCH_DIM_NAME],
                "s": mlp_config[SEQUENCE_LEN_DIM_NAME],
                "h": mlp_config[HIDDEN_DIM_NAME],
                "i": mlp_config[INTERMEDIATE_DIM_NAME],
            }
            collector.match_and_add_metadata_dimensions(test_metadata_key, mlp_tkg_metadata_list)

        lnc_count = mlp_config[VNC_DEGREE_DIM_NAME]
        compiler_args = CompilerArgs(logical_nc_config=lnc_count)
        self.run_range_mlp_tkg_test(
            test_manager=test_manager,
            dtype=mlp_config[DTYPE_DIM_NAME],
            test_options=range_test_options,
            lnc_degree=lnc_count,
            compiler_args=compiler_args,
            norm_type=NormType(mlp_config[NORM_TYPE_DIM_NAME]),
            fused_add=bool(mlp_config[FUSED_ADD_DIM_NAME]),
            store_add=bool(mlp_config[STORE_ADD_DIM_NAME]),
            act_fn_type=ActFnType(mlp_config[ACT_FN_TYPE_DIM_NAME]),
            gate_bias=bool(mlp_config[GATE_BIAS_DIM_NAME]),
            up_bias=bool(mlp_config[UP_BIAS_DIM_NAME]),
            down_bias=bool(mlp_config[DOWN_BIAS_DIM_NAME]),
            norm_bias=bool(mlp_config[NORM_BIAS_DIM_NAME]),
            quant_dtype=mlp_config[QUANTIZATION_DTYPE_DIM_NAME],
            quantization_type=QuantizationType(mlp_config[QUANTIZATION_TYPE_DIM_NAME]),
            skip_gate_proj=bool(mlp_config[SKIP_GATE_DIM_NAME]),
            use_tkg_gate_up_proj_column_tiling=bool(mlp_config[USE_TKG_GATE_UP_PROJ_COLUMN_TILING_DIM_NAME]),
            use_tkg_down_proj_column_tiling=bool(mlp_config[USE_TKG_DOWN_PROJ_COLUMN_TILING_DIM_NAME]),
            use_tkg_down_proj_optimized_layout=bool(mlp_config[USE_TKG_DOWN_PROJ_OPTIMIZED_LAYOUT_DIM_NAME]),
            collector=collector,
        )

    @range_test_config(mlp_tkg_sweep_config())
    @pytest.mark.parametrize(
        "config_name, config",
        [
            ("column_tiling_basic", COLUMN_TILING_BASIC_CONFIG),
            ("column_tiling_full_features_rmsnorm", COLUMN_TILING_FULL_FEATURE_RMSNORM_CONFIG),
            ("column_tiling_full_features_layernorm", COLUMN_TILING_FULL_FEATURE_LAYERNORM_CONFIG),
            ("non_column_tiling_basic", NON_COLUMN_TILING_BASIC_CONFIG),
            ("non_column_tiling_full_features", NON_COLUMN_TILING_FULL_FEATURE_CONFIG),
        ],
    )
    def test_mlp_tkg_kernel_sweep(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: IMetricsCollector,
        config_name,
        config,
    ):
        compiler_args = CompilerArgs()
        self.run_range_mlp_tkg_test(
            test_manager=test_manager,
            dtype=config["dtype"],
            test_options=range_test_options,
            lnc_degree=compiler_args.logical_nc_config,
            compiler_args=compiler_args,
            norm_type=config["norm_type"],
            fused_add=config["fused_add"],
            store_add=config["store_add"],
            act_fn_type=config["act_fn_type"],
            gate_bias=config["gate_bias"],
            up_bias=config["up_bias"],
            down_bias=config["down_bias"],
            norm_bias=config["norm_bias"],
            quant_dtype=config["quant_dtype"],
            quantization_type=QuantizationType.NONE,
            skip_gate_proj=False,
            use_tkg_gate_up_proj_column_tiling=config["use_tkg_gate_up_proj_column_tiling"],
            use_tkg_down_proj_column_tiling=config["use_tkg_down_proj_column_tiling"],
            use_tkg_down_proj_optimized_layout=config["use_tkg_down_proj_optimized_layout"],
            collector=collector,
        )

    @range_test_config(mlp_tkg_I_non_multiple_of_128_sweep_config())
    @pytest.mark.parametrize(
        "config_name, config",
        [
            ("column_tiling_basic", COLUMN_TILING_BASIC_CONFIG),
            ("column_tiling_full_features_rmsnorm", COLUMN_TILING_FULL_FEATURE_RMSNORM_CONFIG),
            ("column_tiling_full_features_layernorm", COLUMN_TILING_FULL_FEATURE_LAYERNORM_CONFIG),
            ("non_column_tiling_basic", NON_COLUMN_TILING_BASIC_CONFIG),
            ("non_column_tiling_full_features", NON_COLUMN_TILING_FULL_FEATURE_CONFIG),
        ],
    )
    def test_mlp_tkg_kernel_sweep_I_non_multiple_of_128(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: IMetricsCollector,
        config_name,
        config,
    ):
        compiler_args = CompilerArgs()
        self.run_range_mlp_tkg_test(
            test_manager=test_manager,
            dtype=config["dtype"],
            test_options=range_test_options,
            lnc_degree=compiler_args.logical_nc_config,
            compiler_args=compiler_args,
            norm_type=config["norm_type"],
            fused_add=config["fused_add"],
            store_add=config["store_add"],
            act_fn_type=config["act_fn_type"],
            gate_bias=config["gate_bias"],
            up_bias=config["up_bias"],
            down_bias=config["down_bias"],
            norm_bias=config["norm_bias"],
            quant_dtype=config["quant_dtype"],
            quantization_type=QuantizationType.NONE,
            skip_gate_proj=False,
            use_tkg_gate_up_proj_column_tiling=config["use_tkg_gate_up_proj_column_tiling"],
            use_tkg_down_proj_column_tiling=config["use_tkg_down_proj_column_tiling"],
            use_tkg_down_proj_optimized_layout=config["use_tkg_down_proj_optimized_layout"],
            collector=collector,
        )

    @range_test_config(mlp_tkg_store_add_false_sweep_config())
    @pytest.mark.parametrize(
        "config_name, config",
        [
            ("column_tiling_basic", COLUMN_TILING_BASIC_CONFIG),
            ("column_tiling_full_features_rmsnorm", COLUMN_TILING_FULL_FEATURE_RMSNORM_CONFIG),
            ("column_tiling_full_features_layernorm", COLUMN_TILING_FULL_FEATURE_LAYERNORM_CONFIG),
        ],
    )
    def test_mlp_tkg_kernel_sweep_store_add_false(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: IMetricsCollector,
        config_name,
        config,
    ):
        # If store_add is False, rmsnorm_tkg and layernorm_tkg should accept SBUF input
        force_fused_add = True
        force_store_add = False
        # Bias is not relevant when checking fused_add=True, store_add=False configs
        force_bias = False

        compiler_args = CompilerArgs()
        self.run_range_mlp_tkg_test(
            test_manager=test_manager,
            dtype=config["dtype"],
            test_options=range_test_options,
            lnc_degree=compiler_args.logical_nc_config,
            compiler_args=compiler_args,
            norm_type=config["norm_type"],
            fused_add=force_fused_add,
            store_add=force_store_add,
            act_fn_type=config["act_fn_type"],
            gate_bias=force_bias,
            up_bias=force_bias,
            down_bias=force_bias,
            norm_bias=config["norm_bias"],
            quant_dtype=config["quant_dtype"],
            quantization_type=QuantizationType.NONE,
            skip_gate_proj=False,
            use_tkg_gate_up_proj_column_tiling=config["use_tkg_gate_up_proj_column_tiling"],
            use_tkg_down_proj_column_tiling=config["use_tkg_down_proj_column_tiling"],
            use_tkg_down_proj_optimized_layout=config["use_tkg_down_proj_optimized_layout"],
            collector=collector,
        )

    @range_test_config(mlp_tkg_basic_sweep_config())
    @pytest.mark.parametrize(
        "use_tkg_gate_up_proj_column_tiling, use_tkg_down_proj_column_tiling, skip_gate_proj, clamp",
        [
            [True, True, True, False],
            [False, True, True, False],
            [True, True, True, True],
            [False, True, True, True],
            [True, True, False, True],
            [False, True, False, True],
        ],
    )
    def test_mlp_tkg_kernel_sweep_feature_test(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: IMetricsCollector,
        use_tkg_gate_up_proj_column_tiling,
        use_tkg_down_proj_column_tiling,
        skip_gate_proj,
        clamp,
    ):
        gate_clamp_upper_limit = float(8.0) if clamp else None
        gate_clamp_lower_limit = float(-6.0) if clamp else None
        up_clamp_upper_limit = float(8.0) if clamp else None
        up_clamp_lower_limit = float(-6.0) if clamp else None

        compiler_args = CompilerArgs()
        self.run_range_mlp_tkg_test(
            test_manager=test_manager,
            dtype=nl.bfloat16,
            test_options=range_test_options,
            compiler_args=compiler_args,
            lnc_degree=compiler_args.logical_nc_config,
            norm_type=NormType.NO_NORM,
            fused_add=False,
            store_add=False,
            act_fn_type=ActFnType.SiLU,
            gate_bias=False,
            up_bias=False,
            down_bias=False,
            norm_bias=False,
            quant_dtype=None,
            quantization_type=QuantizationType.NONE,
            skip_gate_proj=skip_gate_proj,
            use_tkg_gate_up_proj_column_tiling=use_tkg_gate_up_proj_column_tiling,
            use_tkg_down_proj_column_tiling=use_tkg_down_proj_column_tiling,
            use_tkg_down_proj_optimized_layout=False,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            collector=collector,
        )

    @range_test_config(mlp_tkg_sweep_config())
    @pytest.mark.parametrize(
        "config_name, config",
        [
            ("column_tiling_basic", COLUMN_TILING_BASIC_CONFIG),
            ("column_tiling_full_features_rmsnorm", COLUMN_TILING_FULL_FEATURE_RMSNORM_CONFIG),
            ("non_column_tiling_basic", NON_COLUMN_TILING_BASIC_CONFIG),
            ("non_column_tiling_full_features", NON_COLUMN_TILING_FULL_FEATURE_CONFIG),
        ],
    )
    def test_mlp_tkg_kernel_sweep_fp8_row_quant(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: IMetricsCollector,
        config_name,
        config,
    ):
        compiler_args = CompilerArgs()
        self.run_range_mlp_tkg_test(
            test_manager=test_manager,
            dtype=config["dtype"],
            test_options=range_test_options,
            lnc_degree=compiler_args.logical_nc_config,
            compiler_args=compiler_args,
            norm_type=config["norm_type"],
            fused_add=config["fused_add"],
            store_add=config["store_add"],
            act_fn_type=config["act_fn_type"],
            gate_bias=config["gate_bias"],
            up_bias=config["up_bias"],
            down_bias=config["down_bias"],
            norm_bias=config["norm_bias"],
            quant_dtype=nl.float8_e4m3,
            quantization_type=QuantizationType.ROW,
            skip_gate_proj=False,
            use_tkg_gate_up_proj_column_tiling=config["use_tkg_gate_up_proj_column_tiling"],
            use_tkg_down_proj_column_tiling=config["use_tkg_down_proj_column_tiling"],
            use_tkg_down_proj_optimized_layout=config["use_tkg_down_proj_optimized_layout"],
            collector=collector,
        )

    @range_test_config(mlp_tkg_I_non_multiple_of_128_sweep_config())
    @pytest.mark.parametrize(
        "config_name, config",
        [
            ("column_tiling_basic", COLUMN_TILING_BASIC_CONFIG),
            ("column_tiling_full_features_rmsnorm", COLUMN_TILING_FULL_FEATURE_RMSNORM_CONFIG),
            ("non_column_tiling_basic", NON_COLUMN_TILING_BASIC_CONFIG),
            ("non_column_tiling_full_features", NON_COLUMN_TILING_FULL_FEATURE_CONFIG),
        ],
    )
    def test_mlp_tkg_kernel_sweep_I_non_multiple_of_128_fp8_row_quant(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: IMetricsCollector,
        config_name,
        config,
    ):
        compiler_args = CompilerArgs()
        self.run_range_mlp_tkg_test(
            test_manager=test_manager,
            dtype=config["dtype"],
            test_options=range_test_options,
            lnc_degree=compiler_args.logical_nc_config,
            compiler_args=compiler_args,
            norm_type=config["norm_type"],
            fused_add=config["fused_add"],
            store_add=config["store_add"],
            act_fn_type=config["act_fn_type"],
            gate_bias=config["gate_bias"],
            up_bias=config["up_bias"],
            down_bias=config["down_bias"],
            norm_bias=config["norm_bias"],
            quant_dtype=nl.float8_e4m3,
            quantization_type=QuantizationType.ROW,
            skip_gate_proj=False,
            use_tkg_gate_up_proj_column_tiling=config["use_tkg_gate_up_proj_column_tiling"],
            use_tkg_down_proj_column_tiling=config["use_tkg_down_proj_column_tiling"],
            use_tkg_down_proj_optimized_layout=config["use_tkg_down_proj_optimized_layout"],
            collector=collector,
        )

    @range_test_config(mlp_tkg_sweep_config())
    @pytest.mark.parametrize(
        "config_name, config",
        [
            ("column_tiling_basic", COLUMN_TILING_BASIC_CONFIG),
            ("column_tiling_full_features_rmsnorm", COLUMN_TILING_FULL_FEATURE_RMSNORM_CONFIG),
            ("non_column_tiling_basic", NON_COLUMN_TILING_BASIC_CONFIG),
            ("non_column_tiling_full_features", NON_COLUMN_TILING_FULL_FEATURE_CONFIG),
        ],
    )
    def test_mlp_tkg_kernel_sweep_fp8_static_quant(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: IMetricsCollector,
        config_name,
        config,
    ):
        compiler_args = CompilerArgs()
        self.run_range_mlp_tkg_test(
            test_manager=test_manager,
            dtype=config["dtype"],
            test_options=range_test_options,
            lnc_degree=compiler_args.logical_nc_config,
            compiler_args=compiler_args,
            norm_type=config["norm_type"],
            fused_add=config["fused_add"],
            store_add=config["store_add"],
            act_fn_type=config["act_fn_type"],
            gate_bias=config["gate_bias"],
            up_bias=config["up_bias"],
            down_bias=config["down_bias"],
            norm_bias=config["norm_bias"],
            quant_dtype=nl.float8_e4m3,
            quantization_type=QuantizationType.STATIC,
            skip_gate_proj=False,
            use_tkg_gate_up_proj_column_tiling=config["use_tkg_gate_up_proj_column_tiling"],
            use_tkg_down_proj_column_tiling=config["use_tkg_down_proj_column_tiling"],
            use_tkg_down_proj_optimized_layout=config["use_tkg_down_proj_optimized_layout"],
            collector=collector,
        )

    @range_test_config(mlp_tkg_I_non_multiple_of_128_sweep_config())
    @pytest.mark.parametrize(
        "config_name, config",
        [
            ("column_tiling_basic", COLUMN_TILING_BASIC_CONFIG),
            ("column_tiling_full_features_rmsnorm", COLUMN_TILING_FULL_FEATURE_RMSNORM_CONFIG),
            ("non_column_tiling_basic", NON_COLUMN_TILING_BASIC_CONFIG),
            ("non_column_tiling_full_features", NON_COLUMN_TILING_FULL_FEATURE_CONFIG),
        ],
    )
    def test_mlp_tkg_kernel_sweep_I_non_multiple_of_128_fp8_static_quant(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: IMetricsCollector,
        config_name,
        config,
    ):
        compiler_args = CompilerArgs()
        self.run_range_mlp_tkg_test(
            test_manager=test_manager,
            dtype=config["dtype"],
            test_options=range_test_options,
            lnc_degree=compiler_args.logical_nc_config,
            compiler_args=compiler_args,
            norm_type=config["norm_type"],
            fused_add=config["fused_add"],
            store_add=config["store_add"],
            act_fn_type=config["act_fn_type"],
            gate_bias=config["gate_bias"],
            up_bias=config["up_bias"],
            down_bias=config["down_bias"],
            norm_bias=config["norm_bias"],
            quant_dtype=nl.float8_e4m3,
            quantization_type=QuantizationType.STATIC,
            skip_gate_proj=False,
            use_tkg_gate_up_proj_column_tiling=config["use_tkg_gate_up_proj_column_tiling"],
            use_tkg_down_proj_column_tiling=config["use_tkg_down_proj_column_tiling"],
            use_tkg_down_proj_optimized_layout=config["use_tkg_down_proj_optimized_layout"],
            collector=collector,
        )
