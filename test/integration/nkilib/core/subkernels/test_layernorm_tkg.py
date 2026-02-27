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

"""Integration tests for the LayerNorm TKG kernel with various LNC configurations and batch sizes."""

from test.integration.nkilib.utils.dtype_helper import dt
from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.integration.nkilib.utils.test_kernel_common import layer_norm
from test.utils.common_dataclasses import (
    CompilerArgs,
    KernelArgs,
    LazyGoldenGenerator,
    ValidationArgs,
)
from test.utils.metrics_collector import MetricsCollector
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.ranged_test_harness import (
    DimensionRangeConfig,
    RangeMonotonicGeneratorStrategy,
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
from nkilib_src.nkilib.core.subkernels.layernorm_tkg import layernorm_tkg

LAYERNORM_TKG_CONFIG = "layernorm_tkg_config"
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
    lnc_degree: int,
):
    """
    Compute expected LAYERNORM output for validation.

    Args:
        inp: Dictionary containing input tensors (input, gamma, beta)
        lnc_degree: LNC sharding degree (1 or 2)

    Returns:
        Dictionary with 'out' key containing expected output tensor
    """
    hidden = inp['input']
    gamma = inp['gamma']
    beta = inp['beta']
    dtype = hidden.dtype
    B, S, H = hidden.shape
    BxS = B * S
    H0, H1 = 128, H // 128

    # Compute layer normalization with beta bias
    result = layer_norm(hidden, gamma, norm_b=beta)
    result = result.reshape((BxS, -1))

    # Apply output layout transformation for LNC sharding
    if lnc_degree == 2:
        t0 = result[:, 0 : H // 2]
        t1 = result[:, H // 2 :]
        t0 = t0.reshape((BxS, H0, H1 // 2)).transpose((1, 0, 2))
        t1 = t1.reshape((BxS, H0, H1 // 2)).transpose((1, 0, 2))
        result = np.concatenate([t0, t1], axis=2)
    else:
        result = result.reshape((BxS, H0, H1)).transpose((1, 0, 2))

    return {"out": dt.static_cast(result, dtype)}


def build_layernorm_tkg_kernel_input(batch, seqlen, hidden_dim, dtype, tensor_gen):
    """
    Build input tensors for LAYERNORM TKG kernel.

    Args:
        batch: Batch size
        seqlen: Sequence length
        hidden_dim: Hidden dimension size
        dtype: Data type for tensors
        tensor_gen: Tensor generator function

    Returns:
        Dictionary containing input, gamma, output, and beta tensors
    """
    input_tensor = tensor_gen(shape=(batch, seqlen, hidden_dim), dtype=dtype, name="input")
    gamma = tensor_gen(shape=(1, hidden_dim), dtype=dtype, name="gamma")
    output = tensor_gen(shape=(128, batch * seqlen, hidden_dim // 128), dtype=dtype, name='output')
    beta = tensor_gen(shape=(1, hidden_dim), dtype=dtype, name="beta")

    return {"input": input_tensor, "gamma": gamma, "output": output, "beta": beta}


@pytest_test_metadata(
    name="LayerNorm TKG",
    pytest_marks=["layernorm", "tkg"],
)
@final
class TestLayerNormTKGKernel:
    def run_range_layernorm_tkg_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        test_options: RangeTestCase,
        lnc_degree: int,
        batch: int,
        seqlen: int,
        hidden: int,
        dtype,
        collector: MetricsCollector,
    ):
        """Run LayerNorm TKG test with range test options and negative test case handling."""
        is_negative_test_case = test_options.is_negative_test_case
        with assert_negative_test_case(is_negative_test_case):
            self.run_layernorm_tkg_test(
                test_manager=test_manager,
                compiler_args=compiler_args,
                collector=collector,
                lnc_degree=lnc_degree,
                batch=batch,
                seqlen=seqlen,
                hidden=hidden,
                dtype=dtype,
            )

    def run_layernorm_tkg_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        collector: MetricsCollector,
        lnc_degree: int,
        batch: int,
        seqlen: int,
        hidden: int,
        dtype,
        tensor_gen=gaussian_tensor_generator(),
    ):
        """
        Execute LayerNorm TKG kernel test with specified parameters.

        Args:
            test_manager: Test orchestrator for kernel execution
            compiler_args: Compiler configuration arguments
            collector: Metrics collector for test results
            lnc_degree: LNC sharding degree (1 or 2)
            batch: Batch size
            seqlen: Sequence length
            hidden: Hidden dimension size
            dtype: Data type for tensors
            tensor_gen: Tensor generator function for creating test inputs
        """
        kernel_input = build_layernorm_tkg_kernel_input(
            batch=batch,
            seqlen=seqlen,
            hidden_dim=hidden,
            dtype=dtype,
            tensor_gen=tensor_gen,
        )

        # Create lazy golden generator to defer computation until needed
        def create_lazy_golden():
            return golden_output_validator(
                inp=kernel_input,
                lnc_degree=lnc_degree,
            )

        output_placeholder = {"out": np.zeros((128, batch * seqlen, hidden // 128), dtype=dtype)}

        test_manager.execute(
            KernelArgs(
                kernel_func=layernorm_tkg,
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
            )
        )

    # fmt: off
    layernorm_tkg_lnc_params = "lnc_degree, batch, seqlen, hidden, dtype, tpbSgCyclesSum"
    layernorm_tkg_lnc_perms = [
        # LNC1 / Trn1
        (1, 1, 1, 8192, np.float16, 18694971),
        (1, 1, 8, 8192, np.float16, 21886633),
        # LNC2 batch 1
        (2, 1, 1, 5120, np.float16, 21691633),
        (2, 1, 1, 8192, np.float16, 21884966),
        # LNC2 higher batch
        (2, 2, 8, 8192, np.float16, 28388289),
        (2, 4, 8, 8192, np.float16, 27588290),
        (2, 128, 1, 8192, np.float16, 49899089),
        (2, 4, 1, 5120, np.float16, 22559132),
        # LNC2 higher hidden
        (2, 1, 1, 16384, np.float16, 21884966),
        # Sharding threshold
        (2, 2, 5, 16384, np.float16, 35641611),
    ]
    # fmt: on

    @pytest.mark.fast
    @pytest.mark.parametrize(layernorm_tkg_lnc_params, layernorm_tkg_lnc_perms)
    def test_layernorm_tkg_unit(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        lnc_degree,
        batch,
        seqlen,
        hidden,
        dtype,
        tpbSgCyclesSum,
    ):
        """Test LayerNorm TKG kernel with parametrized LNC configurations."""
        compiler_args = CompilerArgs(logical_nc_config=lnc_degree)
        self.run_layernorm_tkg_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
            lnc_degree=compiler_args.logical_nc_config,
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            dtype=dtype,
        )

    @staticmethod
    def layernorm_tkg_sweep_config() -> RangeTestConfig:
        """
        Create range test configuration for LayerNorm TKG sweep tests.

        Returns:
            RangeTestConfig with batch, sequence, hidden, dtype, and LNC dimension ranges.
        """
        # Test-specific dimension values
        MAX_BATCH = 128 // 2
        MAX_SEQLEN = 2
        MAX_HIDDEN = 32768

        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=TensorRangeConfig(
                tensor_configs={
                    LAYERNORM_TKG_CONFIG: TensorConfig(
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
    def layernorm_tkg_sweep_config_large_batch() -> RangeTestConfig:
        """
        Create range test configuration for large batch LayerNorm TKG sweep tests.

        Returns:
            RangeTestConfig with large batch sizes (128-640) and fixed LNC2 configuration.
        """
        # Test-specific dimension values
        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=TensorRangeConfig(
                tensor_configs={
                    LAYERNORM_TKG_CONFIG: TensorConfig(
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

    @range_test_config(layernorm_tkg_sweep_config())
    def test_layernorm_tkg_sweep(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: MetricsCollector,
    ):
        """Run LayerNorm TKG sweep test across multiple dimension configurations."""
        config = range_test_options.tensors['layernorm_tkg_config']
        compiler_args = CompilerArgs(logical_nc_config=config[LNC_DEGREE_DIM_NAME])
        self.run_range_layernorm_tkg_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=range_test_options,
            lnc_degree=config[LNC_DEGREE_DIM_NAME],
            batch=config[BATCH_DIM_NAME],
            seqlen=config[SEQUENCE_LEN_DIM_NAME],
            hidden=config[HIDDEN_DIM_NAME],
            dtype=DTYPE_INT_TO_TYPE[config[DTYPE_DIM_NAME]],
            collector=collector,
        )

    @range_test_config(layernorm_tkg_sweep_config_large_batch())
    def test_layernorm_tkg_sweep_large_batch(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: MetricsCollector,
    ):
        """Run LayerNorm TKG sweep test with large batch configurations."""
        config = range_test_options.tensors['layernorm_tkg_config']
        compiler_args = CompilerArgs(logical_nc_config=2)
        self.run_range_layernorm_tkg_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=range_test_options,
            lnc_degree=2,
            batch=config[BATCH_DIM_NAME],
            seqlen=config[SEQUENCE_LEN_DIM_NAME],
            hidden=config[HIDDEN_DIM_NAME],
            dtype=nl.float16,
            collector=collector,
        )
