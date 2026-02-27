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
from test.integration.nkilib.utils.comparators import maxAllClose
from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import (
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    KernelArgs,
    ValidationArgs,
)
from test.utils.metrics_collector import MetricName, MetricsCollector
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.ranged_test_harness import (
    assert_negative_test_case,
)
from test.utils.test_orchestrator import Orchestrator
from typing import Any, final

import ml_dtypes
import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.core.topk.rotational_topk import (
    RotationalTopkConfig,
    TopkConfig,
    cleanup_rotational_constants,
    prepare_rotational_constants,
    rotational_topk,
)
from typing_extensions import override

INPUT_TENSOR_NAME = "input_tensor"
BATCH_DIM_NAME = "batch"
VOCAB_DIM_NAME = "vocab"
SEQUENCE_LEN_DIM_NAME = "seqlen"


class MaxClassification(enum.Enum):
    SMALL = 1
    MEDIUM = 2
    LARGE = 3

    @staticmethod
    def classify(batch_size: int, seqlen: int, vocab_size: int):
        total_elements = batch_size * seqlen * vocab_size

        if total_elements <= 100000:
            return MaxClassification.SMALL
        elif total_elements <= 1000000:
            return MaxClassification.MEDIUM
        else:
            return MaxClassification.LARGE

    @override
    def __str__(self):
        return self.name


def topk_ref(inp: np.ndarray, k) -> tuple[np.ndarray, np.ndarray]:
    """Topk reference implementation.

    Args:
        inp: Input tensor of shape [..., vocab_size]

    Returns:
        tuple: (topk_values, topk_indices) where:
            - topk_values: shape [..., k] containing topk values
            - topk_indices: shape [..., k] containing indices of topk indices
    """

    # Get TopK indices and values on inp
    ind = np.argsort(-inp, axis=-1)[..., :k]
    val = np.take_along_axis(inp, ind, axis=-1)

    return {"topk_values": val, "topk_indices": ind.astype(np.uint32)}


def golden_output_generator(inp: dict[str, Any]):
    def generate(inp: npt.NDArray[Any]):
        input_tensor = inp["input"]
        k = inp["K"]
        output = topk_ref(input_tensor, k=k)
        return output

    out = lambda: generate(inp)
    return out


def golden_output_validator_values(
    inp: dict[str, Any],
):
    class RotationalTopkValueValidator(CustomValidator):
        @override
        def validate(self, actual_raw_output: npt.NDArray[Any]):
            input_tensor = inp["input"]
            BxS, _ = input_tensor.shape
            K = inp["K"]

            output = np.frombuffer(actual_raw_output, dtype=input_tensor.dtype).reshape(BxS, K)

            output_ref = topk_ref(input_tensor, K)

            passed = maxAllClose(output, output_ref["topk_values"], verbose=1)
            return passed

    return RotationalTopkValueValidator


def golden_output_validator_indices(
    inp: dict[str, Any],
):
    class RotationalTopkIndicesValidator(CustomValidator):
        @override
        def validate(self, actual_raw_output: npt.NDArray[Any]):
            input_tensor = inp["input"]
            BxS, _ = input_tensor.shape
            K = inp["K"]

            output = np.frombuffer(actual_raw_output, dtype=np.uint32).reshape(BxS, K)

            output_ref = topk_ref(input_tensor, K)
            val = np.take_along_axis(input_tensor, output.astype(np.uint64), axis=-1)

            passed = maxAllClose(output_ref["topk_values"], val, verbose=1)
            return passed

    return RotationalTopkIndicesValidator


def golden_output_validator_values_unsorted(
    inp: dict[str, Any],
):
    class RotationalTopkValueValidatorUnsorted(CustomValidator):
        @override
        def validate(self, actual_raw_output: npt.NDArray[Any]):
            input_tensor = inp["input"]
            BxS, _ = input_tensor.shape
            K = inp["K"]

            output = np.frombuffer(actual_raw_output, dtype=input_tensor.dtype).reshape(BxS, K)
            output_ref = topk_ref(input_tensor, K)

            # Compare sorted versions since output may be unsorted
            passed = maxAllClose(np.sort(output, axis=-1), np.sort(output_ref["topk_values"], axis=-1), verbose=1)
            return passed

    return RotationalTopkValueValidatorUnsorted


def golden_output_validator_indices_unsorted(
    inp: dict[str, Any],
):
    class RotationalTopkIndicesValidatorUnsorted(CustomValidator):
        @override
        def validate(self, actual_raw_output: npt.NDArray[Any]):
            input_tensor = inp["input"]
            BxS, _ = input_tensor.shape
            K = inp["K"]

            output = np.frombuffer(actual_raw_output, dtype=np.uint32).reshape(BxS, K)
            output_ref = topk_ref(input_tensor, K)
            val = np.take_along_axis(input_tensor, output.astype(np.uint64), axis=-1)

            # Compare sorted versions since output may be unsorted
            passed = maxAllClose(np.sort(output_ref["topk_values"], axis=-1), np.sort(val, axis=-1), verbose=1)
            return passed

    return RotationalTopkIndicesValidatorUnsorted


def _get_np_dtype(dtype):
    """Convert NKI dtype to numpy dtype for tensor generation."""
    if dtype == nl.bfloat16:
        return ml_dtypes.bfloat16
    return dtype


def build_topk_kernel_input(lnc_degree: int, batch: int, seqlen: int, vocab_size: int, K: int, dtype, tensor_gen):
    """Build input for cascaded max kernel."""
    np_dtype = _get_np_dtype(dtype)
    input_tensor = tensor_gen(shape=(batch * seqlen, vocab_size), dtype=np_dtype, name="input")
    return {"input": input_tensor, "K": K}


@pytest_test_metadata(
    name="Rotational TopK",
    pytest_marks=["topk", "rotational"],
)
@final
class TestTopKKernel:
    def run_topk_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        collector: MetricsCollector,
        lnc_degree: int,
        batch_size: int,
        seqlen: int,
        vocab_size: int,
        K: int,
        dtype,
        tensor_gen=gaussian_tensor_generator(),
        sorted: bool = True,
    ):
        kernel_input = build_topk_kernel_input(
            lnc_degree=lnc_degree,
            batch=batch_size,
            seqlen=seqlen,
            vocab_size=vocab_size,
            K=K,
            dtype=dtype,
            tensor_gen=tensor_gen,
        )

        # Preprocessing steps matching topk wrapper
        input_tensor = kernel_input["input"]
        inp_3d = input_tensor.reshape((batch_size, seqlen, vocab_size))

        topk_config = TopkConfig(
            inp_shape=inp_3d.shape,
            inp_dtype=dtype,
            k=K,
            sorted=sorted,
            num_programs=lnc_degree,
        )
        inp_reshaped = inp_3d.reshape((topk_config.BxS, topk_config.vocab_size))
        config = RotationalTopkConfig(inp_shape=inp_reshaped.shape, topk_config=topk_config)
        prepare_rotational_constants(config)

        # Update kernel input for jitted function
        kernel_input_jit = {"inp": inp_reshaped, "config": config}

        np_dtype = _get_np_dtype(dtype)
        placeholder_output = {
            "topk_values": np.ndarray(shape=(batch_size * seqlen, K), dtype=np_dtype),
            "topk_indices": np.ndarray(shape=(batch_size * seqlen, K), dtype=np.uint32),
        }

        if sorted:
            golden_validator_values = golden_output_validator_values(inp=kernel_input)
            golden_validator_indices = golden_output_validator_indices(inp=kernel_input)
        else:
            golden_validator_values = golden_output_validator_values_unsorted(inp=kernel_input)
            golden_validator_indices = golden_output_validator_indices_unsorted(inp=kernel_input)
        test_manager.execute(
            KernelArgs(
                kernel_func=rotational_topk,
                compiler_input=compiler_args,
                kernel_input=kernel_input_jit,
                validation_args=ValidationArgs(
                    golden_output={
                        "topk_values": CustomValidatorWithOutputTensorData(
                            validator=golden_validator_values,
                            output_ndarray=placeholder_output["topk_values"],
                        ),
                        "topk_indices": CustomValidatorWithOutputTensorData(
                            validator=golden_validator_indices,
                            output_ndarray=placeholder_output["topk_indices"],
                        ),
                    }
                ),
            )
        )

        cleanup_rotational_constants()

    # fmt: off
    topk_unit_params = "lnc_degree, batch, seqlen, vocab_size, K, dtype"

    large_batch_perms  = [
            [2, 150, 1, 1000, 50, nl.float32],   # BxS=150
            [2, 200, 1, 2000, 64, nl.float32],   # BxS=200
            [2, 1024, 1, 5000, 128, nl.float32],  # BxS=256
        ]

    topk_unit_perms = [
        # Llama 3 76B before global gather
        [2, 8, 5, 4058, 256, nl.float32],
        [2, 5, 5, 4058, 256, nl.float32],

        # # Llama 3 76B after global gather
        [1, 4, 5, 8192, 256, nl.float32],
        [1, 8, 5, 8192, 256, nl.float32],
        [2, 8, 5, 8192, 256, nl.float32],
        [2, 5, 5, 8192, 256, nl.float32],

        # Functionality tests
        
        [2, 1, 1, 3168, 256, nl.float32],

        # Vocab size generalization
        [2, 1, 1, 256, 256, nl.float32],
        [2, 1, 1, 16000, 256, nl.float32],

        # Max stage num batch sizes
        [2, 3, 1, 3168, 256, nl.float32],
        [2, 7, 1, 3168, 256, nl.float32],
        [2, 8, 1, 3168, 256, nl.float32],

        # Medium stage num batch sizes
        [2, 10, 1, 3168, 256, nl.float32],
        [2, 16, 1, 3168, 256, nl.float32],
        [2, 32, 1, 3168, 256, nl.float32],
        [2, 63, 1, 3168, 256, nl.float32],

        # Scanning approach batch sizes
        [2, 65, 1, 3168, 256, nl.float32],
        [2, 99, 1, 3168, 256, nl.float32],
        [2, 128, 1, 3168, 256, nl.float32],
        [2, 256, 1, 3168, 256, nl.float32],

        # K generalization nominal
        [2, 1, 1, 3168, 8, nl.float32],
        [2, 1, 1, 3168, 64, nl.float32],
        [2, 1, 1, 3168, 192, nl.float32],

        # K generalization hard
        [2, 1, 1, 3168, 1, nl.float32],
        [2, 1, 1, 3168, 7, nl.float32],
        [2, 1, 1, 3168, 60, nl.float32],
        [2, 1, 1, 3168, 99, nl.float32],

        # Mixed tests
        [1, 1, 7, 3999, 256, nl.float32],
        [1, 1, 63, 3999, 20, nl.float32],
        [2, 1, 127, 3999, 256, nl.float32],
        [2, 1, 127, 3999, 1, nl.float32],
        [1, 1, 127, 3999, 20, nl.float32],

        # bfloat16 tests
        [2, 1, 1, 3168, 256, nl.bfloat16],
        [2, 8, 5, 4058, 256, nl.bfloat16],
        [2, 1, 1, 16000, 256, nl.bfloat16],
        [2, 32, 1, 3168, 256, nl.bfloat16],

        # Large vocab test 
        [2, 1, 1, 25600, 256, nl.float32],
        [2, 8, 1, 25600, 256, nl.float32],
        [2, 1, 1, 2048, 256, nl.float32],

    ]
    # fmt: on
    topk_unit_perms.extend(large_batch_perms)

    @pytest.mark.fast
    @pytest.mark.parametrize(topk_unit_params, topk_unit_perms)
    def test_topk_unit(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        lnc_degree,
        batch,
        seqlen,
        vocab_size,
        K,
        dtype,
    ):
        compiler_args = CompilerArgs(logical_nc_config=lnc_degree)
        is_negative_test_case = False
        if batch * seqlen > 256:
            is_negative_test_case = True
        with assert_negative_test_case(is_negative_test_case):
            self.run_topk_test(
                test_manager=test_manager,
                compiler_args=compiler_args,
                collector=collector,
                lnc_degree=lnc_degree,
                batch_size=batch,
                seqlen=seqlen,
                vocab_size=vocab_size,
                K=K,
                dtype=dtype,
                tensor_gen=gaussian_tensor_generator(),
            )

    topk_unsorted_params = "lnc_degree, batch, seqlen, vocab_size, K, dtype"
    topk_unsorted_perms = [
        [2, 1, 1, 25136, 256, nl.float32],
        [2, 8, 5, 4058, 256, nl.float32],
        [2, 5, 5, 4058, 256, nl.float32],
    ]

    @pytest.mark.fast
    @pytest.mark.parametrize(topk_unsorted_params, topk_unsorted_perms)
    def test_topk_unsorted(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        lnc_degree,
        batch,
        seqlen,
        vocab_size,
        K,
        dtype,
    ):
        compiler_args = CompilerArgs(logical_nc_config=lnc_degree)
        self.run_topk_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
            lnc_degree=lnc_degree,
            batch_size=batch,
            seqlen=seqlen,
            vocab_size=vocab_size,
            K=K,
            dtype=dtype,
            sorted=False,
        )
