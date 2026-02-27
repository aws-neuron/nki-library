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
from test.integration.nkilib.utils.dtype_helper import dt
from test.integration.nkilib.utils.tensor_generators import (
    gaussian_tensor_generator,
)
from test.integration.nkilib.utils.test_kernel_common import rms_norm
from test.utils.common_dataclasses import (
    TKG_INFERENCE_ARGS,
    CompilerArgs,
    KernelArgs,
    LazyGoldenGenerator,
    ValidationArgs,
)
from test.utils.metrics_collector import MetricName, MetricsCollector
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.ranged_test_harness import (
    DimensionRangeConfig,
    RangeMonotonicGeneratorStrategy,
    RangeRandomGeneratorStrategy,
    RangeTestCase,
    RangeTestConfig,
    TensorConfig,
    TensorRangeConfig,
    assert_negative_test_case,
    range_test_config,
)
from test.utils.test_orchestrator import Orchestrator
from typing import Any, Optional, final

import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.core.subkernels.rmsnorm_tkg import (
    rmsnorm_tkg,
)
from typing_extensions import override

RMSNORM_TKG_CONFIG = "rmsnorm_tkg_config"
LNC_DEGREE_DIM_NAME = "lnc"
BATCH_DIM_NAME = "b"
SEQUENCE_LEN_DIM_NAME = "s"
HIDDEN_DIM_NAME = "h"
DTYPE_DIM_NAME = "dt"
DUMMY_STEP_SIZE = 1  # Placeholder value that will be overridden by custom generators and/or DimensionRangeConfigs
DTYPE_TYPE_TO_INT = {nl.float16: 0, nl.bfloat16: 1, nl.float32: 2}
DTYPE_INT_TO_TYPE = {0: nl.float16, 1: nl.bfloat16, 2: nl.float32}
assert len(DTYPE_TYPE_TO_INT) == len(DTYPE_INT_TO_TYPE)
for k, v in DTYPE_TYPE_TO_INT.items():
    assert DTYPE_INT_TO_TYPE[v] == k


def golden_output_validator(
    inp: dict[str, Any],
    lnc_degree,
    hidden_actual,
    hidden_dim_tp: bool = False,
):
    hidden = inp['input']
    gamma = inp['gamma']
    dtype = hidden.dtype
    lnc_degree = lnc_degree
    hidden_actual = hidden_actual
    B, S, H = hidden.shape
    BxS = B * S
    H0, H1 = 128, H // 128

    # normalization
    result = rms_norm(hidden, gamma, hidden_actual=hidden_actual)
    result = result.reshape((BxS, -1))
    if hidden_dim_tp:
        result = result.reshape((BxS, H1, H0)).transpose((2, 0, 1))
    elif lnc_degree == 2:
        t0 = result[:, 0 : H // 2]
        t1 = result[:, H // 2 :]
        t0 = t0.reshape((BxS, H0, H1 // 2)).transpose((1, 0, 2))
        t1 = t1.reshape((BxS, H0, H1 // 2)).transpose((1, 0, 2))
        result = np.concatenate([t0, t1], axis=2)
    else:
        result = result.reshape((BxS, H0, H1)).transpose((1, 0, 2))

    return {"out": dt.static_cast(result, dtype)}


def build_rmsnorm_tkg_kernel_input(batch, seqlen, hidden_dim, hidden_actual_dim, hidden_dim_tp, dtype, tensor_gen):
    # RMSNorm kernel inputs
    input = tensor_gen(shape=(batch, seqlen, hidden_dim), dtype=dtype, name="input")
    gamma = tensor_gen(shape=(1, hidden_dim), dtype=dtype, name="gamma")
    output = tensor_gen(shape=(128, batch * seqlen, hidden_dim // 128), dtype=dtype, name='output')

    return {
        "input": input,
        "gamma": gamma,
        "output": output,
        "hidden_actual": hidden_actual_dim,
        "hidden_dim_tp": hidden_dim_tp,
    }


@pytest_test_metadata(
    name="RMSNorm TKG",
    pytest_marks=["rmsnorm", "tkg"],
)
@final
class TestRmsNormTKGKernel:
    def run_range_rmsnorm_tkg_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        test_options: RangeTestCase,
        lnc_degree,
        batch: int,
        seqlen: int,
        hidden: int,
        hidden_actual: int,
        hidden_dim_tp: bool,
        dtype,
        collector: MetricsCollector,
    ):
        is_negative_test_case = test_options.is_negative_test_case
        with assert_negative_test_case(is_negative_test_case):
            self.run_rmsnorm_tkg_test(
                test_manager=test_manager,
                compiler_args=compiler_args,
                collector=collector,
                lnc_degree=lnc_degree,
                batch=batch,
                seqlen=seqlen,
                hidden=hidden,
                hidden_actual=hidden_actual,
                hidden_dim_tp=hidden_dim_tp,
                dtype=dtype,
            )

    def run_rmsnorm_tkg_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        collector: MetricsCollector,
        lnc_degree,
        batch: int,
        seqlen: int,
        hidden: int,
        hidden_actual: int,
        hidden_dim_tp: bool,
        dtype,
        tensor_gen=gaussian_tensor_generator(),
    ):
        kernel_input = build_rmsnorm_tkg_kernel_input(
            batch=batch,
            seqlen=seqlen,
            hidden_dim=hidden,
            hidden_actual_dim=hidden_actual,
            hidden_dim_tp=hidden_dim_tp,
            dtype=dtype,
            tensor_gen=tensor_gen,
        )

        # Create lazy golden generator to defer computation until needed
        def create_lazy_golden():
            return golden_output_validator(
                inp=kernel_input,
                lnc_degree=lnc_degree,
                hidden_actual=hidden_actual,
                hidden_dim_tp=hidden_dim_tp,
            )

        output_placeholder = {"out": np.zeros((128, batch * seqlen, hidden // 128), dtype=dtype)}

        test_manager.execute(
            KernelArgs(
                kernel_func=rmsnorm_tkg,
                compiler_input=compiler_args,
                kernel_input=kernel_input,
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(
                        output_ndarray=output_placeholder,
                        lazy_golden_generator=create_lazy_golden,
                    ),
                    relative_accuracy=2e-2,
                    absolute_accuracy=1e-5,
                ),
                inference_args=TKG_INFERENCE_ARGS,
            )
        )

    # fmt: off
    rmsnorm_tkg_lnc_params = "lnc_degree, batch, seqlen, hidden, hidden_actual, hidden_dim_tp, dtype, tpbSgCyclesSum"
    rmsnorm_tkg_lnc_perms = [
        (1, 1, 1, 8192, None, False, np.float16, 18694971),
        (1, 1, 8, 8192, None, False, np.float16, 21886633),
        (1, 1, 1, 4096, 3072, False, np.float16, 19264137),
        (1, 1, 4, 2048, 2048, False, np.float16, 19369136),
        (1, 1, 4, 2048, 1920, False, np.float16, 17940805),
        # LNC2 batch 1
        (2, 1, 1, 5120, None, False, np.float16, 21691633),
        (2, 1, 1, 8192, None, False, np.float16, 21884966),
        # LNC2 higher batch
        (2, 2, 8, 8192, None, False, np.float16, 28388289),
        (2, 4, 8, 8192, None, False, np.float16, 27588290),
        (2, 128, 1, 8192, None, False, np.float16, 49899089),
        (2, 4, 1, 5120, None, False, np.float16, 22559132),
        # LNC2 BxS tiling
        (2, 128, 5, 3072, None, False, np.float16, 128576466),
        (2, 256, 5, 3072, None, False, np.float16, 246268782),
        (2, 63, 32, 3072, None, False, np.float16, 369565256),
        # LNC2 higher hidden
        (2, 1, 1, 16384, None, False, np.float16, 21884966),
        (2, 1, 1, 4096, 3072, False, np.float16, 20479968),
        # Sharding threshold
        (2, 2, 8, 16384, None, False, np.float16, 35641611),
        (2, 2, 8, 3072, 2880, False, np.float16, 26030793),
        # hidden_dim_tp = True
        (1, 1, 1, 4096, 3072, True, np.float16, 19264137),
        (2, 1, 1, 5120, None, True, np.float16, 21691633),
        (2, 128, 1, 8192, None, True, np.float16, 49899089),
        (2, 2, 8, 3072, 2880, True, np.float16, 26030793),
    ]
    # fmt: on

    @pytest.mark.fast
    @pytest.mark.parametrize(rmsnorm_tkg_lnc_params, rmsnorm_tkg_lnc_perms)
    def test_rmsnorm_tkg_unit(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        lnc_degree,
        batch,
        seqlen,
        hidden,
        hidden_actual,
        hidden_dim_tp,
        dtype,
        tpbSgCyclesSum,
    ):
        compiler_args = CompilerArgs(logical_nc_config=lnc_degree)
        self.run_rmsnorm_tkg_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
            lnc_degree=compiler_args.logical_nc_config,
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            hidden_actual=hidden_actual,
            hidden_dim_tp=hidden_dim_tp,
            dtype=dtype,
        )

    @staticmethod
    def rmsnorm_tkg_sweep_config() -> RangeTestConfig:
        # Test-specific dimension values
        MAX_BATCH = 128 // 2
        MAX_SEQLEN = 2
        MAX_HIDDEN = 32768

        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=TensorRangeConfig(
                tensor_configs={
                    RMSNORM_TKG_CONFIG: TensorConfig(
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
                            DimensionRangeConfig(min=0, max=len(DTYPE_INT_TO_TYPE) - 1, name=DTYPE_DIM_NAME),
                            DimensionRangeConfig(min=1, max=2, name=LNC_DEGREE_DIM_NAME),
                        ]
                    ),
                },
                monotonic_step_size=DUMMY_STEP_SIZE,
                custom_generators=[RangeMonotonicGeneratorStrategy(step_size=DUMMY_STEP_SIZE)],
            ),
        )

    @staticmethod
    def rmsnorm_tkg_sweep_config_large_batch() -> RangeTestConfig:
        # Test-specific dimension values
        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=TensorRangeConfig(
                tensor_configs={
                    RMSNORM_TKG_CONFIG: TensorConfig(
                        [
                            DimensionRangeConfig(min=128, max=640, multiple_of=128, name=BATCH_DIM_NAME),
                            DimensionRangeConfig(min=1, max=1, name=SEQUENCE_LEN_DIM_NAME),
                            DimensionRangeConfig(min=3072, max=8192, power_of=2, name=HIDDEN_DIM_NAME),
                        ]
                    ),
                },
                monotonic_step_size=DUMMY_STEP_SIZE,
                custom_generators=[RangeMonotonicGeneratorStrategy(step_size=DUMMY_STEP_SIZE)],
            ),
        )

    @range_test_config(rmsnorm_tkg_sweep_config())
    def test_rmsnorm_tkg_sweep(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: MetricsCollector,
    ):
        config = range_test_options.tensors['rmsnorm_tkg_config']
        compiler_args = CompilerArgs(logical_nc_config=config[LNC_DEGREE_DIM_NAME])
        self.run_range_rmsnorm_tkg_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=range_test_options,
            lnc_degree=config[LNC_DEGREE_DIM_NAME],
            batch=config[BATCH_DIM_NAME],
            seqlen=config[SEQUENCE_LEN_DIM_NAME],
            hidden=config[HIDDEN_DIM_NAME],
            hidden_actual=None,
            hidden_dim_tp=False,
            dtype=DTYPE_INT_TO_TYPE[config[DTYPE_DIM_NAME]],
            collector=collector,
        )

    @range_test_config(rmsnorm_tkg_sweep_config_large_batch())
    def test_rmsnorm_tkg_sweep_large_batch(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: MetricsCollector,
    ):
        config = range_test_options.tensors['rmsnorm_tkg_config']
        compiler_args = CompilerArgs(logical_nc_config=2)
        self.run_range_rmsnorm_tkg_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=range_test_options,
            lnc_degree=2,
            batch=config[BATCH_DIM_NAME],
            seqlen=config[SEQUENCE_LEN_DIM_NAME],
            hidden=config[HIDDEN_DIM_NAME],
            hidden_actual=None,
            hidden_dim_tp=False,
            dtype=nl.float16,
            collector=collector,
        )
