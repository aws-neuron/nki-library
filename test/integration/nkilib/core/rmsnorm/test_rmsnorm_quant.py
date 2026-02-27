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
import random
from test.integration.nkilib.utils.comparators import maxAllClose
from test.integration.nkilib.utils.tensor_generators import (
    duplicate_row_rmsnorm_inp_generator,
    gaussian_tensor_generator,
)
from test.utils.common_dataclasses import (
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    KernelArgs,
    ValidationArgs,
)
from test.utils.coverage_parametrized_tests import BoundedRange, assert_negative_test_case
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from typing import Any, Optional, final

import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.core.rmsnorm.rmsnorm_quant import (
    RmsNormQuantKernelArgs,
    rmsnorm_quant_kernel,
)
from nkilib_src.nkilib.core.utils.common_types import NormType, QuantizationType
from typing_extensions import override


def rmsnorm_quant_ref(
    inp: np.ndarray,
    gamma: np.ndarray,
    input_dequant_scale: np.ndarray | None,
    lower_bound: float,
    quant_only: bool,
    quant_type: QuantizationType,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """RMSNorm + Quantization reference impl.

    - inp: shape [B, S, H]
    - output[0]: shape [B, S, H] in fp8e4, representing the quantized RMSNorm output of input
    - output[1]: shape [B, S, 4] in fp32 representing the per-row dequantization scale
    """
    FP8_RANGE = 240.0

    assert len(inp.shape) == 3
    inp = inp.astype(np.float32)
    gamma = gamma.astype(np.float32)
    if input_dequant_scale is not None:
        input_dequant_scale = input_dequant_scale.astype(np.float32)

    # Perform RMSNorm
    if quant_only:
        norm = inp
    else:
        rms = np.sqrt(np.mean(np.square(inp), axis=-1, keepdims=True))
        norm = inp * np.reciprocal(rms + eps)
        norm *= gamma

    # Perform quantization
    if quant_type == QuantizationType.ROW:
        norm_abs_max = np.abs(norm).max(axis=-1, keepdims=True)
        if lower_bound > 0:
            norm_abs_max = np.clip(norm_abs_max, a_min=None, a_max=lower_bound)
            norm = np.clip(norm, a_min=-lower_bound, a_max=lower_bound)
        dequant_scale = norm_abs_max / FP8_RANGE
        quant_scale = np.reciprocal(dequant_scale)
        norm_quant = norm * quant_scale
        assert np.allclose(norm, norm_quant * dequant_scale)  # dequantization should yield same norm
        dequant_scale = dt.static_cast(dequant_scale, np.float32)
    elif quant_type == QuantizationType.STATIC:
        quant_scale = np.reciprocal(input_dequant_scale[0, 0])
        norm = norm * quant_scale
        norm_quant = np.clip(norm, a_min=-FP8_RANGE, a_max=FP8_RANGE)
        dequant_scale = None

    # Cast and return
    norm_quant = dt.static_cast(norm_quant, nl.float8_e4m3)

    return norm_quant, dequant_scale


def golden_output_validator(
    inp: dict[str, Any],
    lower_bound: float,
    norm_values_rtol: float,
    deq_scale_rtol: float,
    quant_only: Optional[bool],
):
    class RmsNormQuantValidator(CustomValidator):
        @override
        def validate(self, actual_raw_output: npt.NDArray[Any]):
            hidden = inp["hidden"]
            gamma = inp["ln_w"]
            quant_type = inp["kargs"].quantization_type
            is_quant_only = inp["kargs"].norm_type == NormType.NO_NORM if quant_only is None else quant_only
            eps = inp["kargs"].eps
            input_dequant_scale = inp["input_dequant_scale"]

            B, S, H = hidden.shape
            if quant_type == QuantizationType.ROW:
                # Get the full fp8 tensor including the extra 4 bytes
                full_fp8_tensor = np.frombuffer(
                    actual_raw_output.view(dtype=nl.bfloat16), dtype=nl.float8_e4m3
                ).reshape(B, S, H + 4)
                # Extract the fp8 values (first H elements of each vector)
                norm_out_hw_fp8 = full_fp8_tensor[:, :, :H]
                # Extract the dequantization scale (last 4 bytes of each vector)
                # Extract the last 4 fp8 elements (which represent the 4 bytes of the fp32 scale)
                last_4_fp8 = full_fp8_tensor[:, :, H : H + 4]  # Shape: (B, S, 4)
                # Reinterpret these 4 fp8 bytes as 1 fp32 value
                # Make sure the array is contiguous before using view
                last_4_fp8_contiguous = np.ascontiguousarray(last_4_fp8)
                norm_deq_scale_hw_fp32 = last_4_fp8_contiguous.view(dtype=np.float32).reshape(B, S, 1)
            else:
                norm_out_hw_fp8 = np.frombuffer(
                    actual_raw_output.view(dtype=nl.bfloat16), dtype=nl.float8_e4m3
                ).reshape(B, S, H)

            norm_out_golden, norm_deq_scale_golden = rmsnorm_quant_ref(
                hidden, gamma, input_dequant_scale, lower_bound, is_quant_only, quant_type, eps
            )

            # Do the comparison
            # Cast Norm outputs from fp8 to fp32
            norm_out_hw_fp32 = dt.static_cast(norm_out_hw_fp8, np.float32)
            norm_out_golden_fp32 = dt.static_cast(norm_out_golden, np.float32)

            passed = maxAllClose(norm_out_hw_fp32, norm_out_golden_fp32, rtol=norm_values_rtol, verbose=1)
            if quant_type == QuantizationType.ROW:
                passed &= maxAllClose(
                    norm_deq_scale_hw_fp32,
                    norm_deq_scale_golden,
                    rtol=deq_scale_rtol,
                    verbose=1,
                )

            return passed

    return RmsNormQuantValidator


def static_scale_wrapper(default_tensor_generator):
    rng = np.random.default_rng(0)

    def tensor_generator(shape, dtype, name):
        if name == "input_dequant_scale":
            fill_value = rng.normal(loc=0.5, scale=0.1)
            return np.full(shape, fill_value, dtype)
        else:
            return default_tensor_generator(shape, dtype, name)

    return tensor_generator


def build_rmsnorm_quant_kernel_input(
    lnc_degree: int,
    batch,
    seqlen,
    hidden_dim,
    dtype,
    lower_bound: float,
    tensor_gen,
    quant_type: QuantizationType,
    quant_only: bool = False,
):
    gamma_shape = [hidden_dim]

    # RMSNorm kernel inputs
    hidden = tensor_gen(shape=(batch, seqlen, hidden_dim), dtype=dtype, name="hidden")
    if lnc_degree > 1:
        gamma_shape = [1, hidden_dim]

    gamma = tensor_gen(shape=gamma_shape, dtype=dtype, name="gamma")
    if quant_type == QuantizationType.STATIC:
        scale = tensor_gen(shape=(128, 1), dtype=nl.float32, name="input_dequant_scale")
    else:
        scale = None
    norm_type = NormType.NO_NORM if quant_only else NormType.RMS_NORM
    kernel_args = RmsNormQuantKernelArgs(
        quantization_type=quant_type, lower_bound=lower_bound, norm_type=norm_type, eps=1e-6
    )
    return {"hidden": hidden, "ln_w": gamma, "kargs": kernel_args, "input_dequant_scale": scale}


def quant_input_tensor_generator(use_rng: bool = True):
    """Generate quantized inputs.

    use_rng: when False, generate all ones
    """
    rng = np.random.default_rng(0)

    def tensor_generator(shape, dtype, name=None):
        if use_rng:
            return rng.uniform(0, 10, shape).astype(dtype)
        else:
            return np.full(shape=shape, fill_value=rng.random(), dtype=dtype)

    return tensor_generator


@pytest_test_metadata(
    name="RMSNorm Quantization",
    pytest_marks=["rmsnorm", "quantization"],
)
@final
class TestRmsNormQuantKernel:
    def run_rmsnorm_quant_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        collector: IMetricsCollector,
        batch_size: int,
        hidden_size: int,
        dtype,
        lnc_degree,
        lower_bound: float,
        quant_type: QuantizationType,
        quant_dtype,
        seqlen: int,
        quant_only: bool | None = None,
        tensor_gen=static_scale_wrapper(gaussian_tensor_generator()),
    ):
        kernel_input = build_rmsnorm_quant_kernel_input(
            lnc_degree=lnc_degree,
            batch=batch_size,
            seqlen=seqlen,
            hidden_dim=hidden_size,
            dtype=dtype,
            lower_bound=lower_bound,
            tensor_gen=tensor_gen,
            quant_type=quant_type,
            quant_only=quant_only if quant_only is not None else False,
        )

        assert dt.sizeinbytes(np.float32) % dt.sizeinbytes(quant_dtype) == 0
        if quant_type == QuantizationType.ROW:
            dtype_size_scale = dt.sizeinbytes(np.float32) // dt.sizeinbytes(quant_dtype)
            rmsnorm_output = np.ndarray(shape=[batch_size, seqlen, hidden_size + dtype_size_scale], dtype=quant_dtype)
        else:
            rmsnorm_output = np.ndarray(shape=[batch_size, seqlen, hidden_size], dtype=quant_dtype)

        golden_validator = golden_output_validator(
            inp=kernel_input,
            lower_bound=lower_bound,
            norm_values_rtol=0.072,
            deq_scale_rtol=0.007,
            quant_only=quant_only,
        )

        test_manager.execute(
            KernelArgs(
                kernel_func=rmsnorm_quant_kernel,
                compiler_input=compiler_args,
                kernel_input=kernel_input,
                validation_args=ValidationArgs(
                    golden_output={
                        "out": CustomValidatorWithOutputTensorData(
                            validator=golden_validator,
                            output_ndarray=rmsnorm_output,
                        )
                    }
                ),
            )
        )

    # fmt: off
    rmsnorm_quant_lnc_params = "seqlen, quant_only, quant_type, tpbSgCyclesSum, tensor_gen, lower_bound, lnc_degree, batch, hidden"
    rmsnorm_quant_lnc_perms = [
        # seqlen, quant_only, quant_type tpbSgCyclesSum, tensor_gen, lower_bound, lnc_degree, batch, hidden
        (128 // 64, True, QuantizationType.ROW, 51_564_336, gaussian_tensor_generator(), 0.0, 2, 1, 16384),
        (128 // 64, True, QuantizationType.ROW, 55_688_496, gaussian_tensor_generator(), 0.5, 2, 1, 16384),
        (128 // 64, False, QuantizationType.ROW, 89_102_777, gaussian_tensor_generator(), 0.0, 2, 1, 16384),
        (128 // 64, False, QuantizationType.ROW, 93_851_770, gaussian_tensor_generator(), 0.5, 2, 1, 16384),
        (2048, False, QuantizationType.ROW, 683_281_265, gaussian_tensor_generator(), 0.0, 2, 1, 16384),
        (160, False, QuantizationType.ROW, 106_546_750, gaussian_tensor_generator(), 0.0, 2, 1, 16384),
        (160, False, QuantizationType.ROW, 105_370_418, duplicate_row_rmsnorm_inp_generator(), 0.0, 2, 1, 16384),
        (160, False, QuantizationType.ROW, 104_917_919, duplicate_row_rmsnorm_inp_generator(all_ones=True), 0.0, 2, 1, 16384),
        (160, True, QuantizationType.ROW, 68_355_309, gaussian_tensor_generator(), 0.0, 2, 1, 16384),
        (128 // 64, True, QuantizationType.STATIC, None, gaussian_tensor_generator(), 0.0, 2, 1, 16384),
        (128 // 64, False, QuantizationType.STATIC, None, gaussian_tensor_generator(), 0.0, 2, 1, 16384),
        (2048, False, QuantizationType.STATIC, None, gaussian_tensor_generator(), 0.0, 2, 1, 16384),
        (160, False, QuantizationType.STATIC, None, gaussian_tensor_generator(), 0.0, 2, 1, 16384),
        (160, False, QuantizationType.STATIC, None, duplicate_row_rmsnorm_inp_generator(), 0.0, 2, 1, 16384),
        (160, False, QuantizationType.STATIC, None, duplicate_row_rmsnorm_inp_generator(all_ones=True), 0.0, 2, 1, 16384),
        (160, True, QuantizationType.STATIC, None, gaussian_tensor_generator(), 0.0, 2, 1, 16384),
    ]
    # fmt: on

    @pytest.mark.fast
    @pytest.mark.parametrize(rmsnorm_quant_lnc_params, rmsnorm_quant_lnc_perms)
    def test_rmsnorm_quant_lnc_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        seqlen,
        quant_only,
        quant_type,
        tpbSgCyclesSum,
        tensor_gen,
        lower_bound,
        lnc_degree,
        batch,
        hidden,
    ):
        compiler_args = CompilerArgs(logical_nc_config=lnc_degree)
        self.run_rmsnorm_quant_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
            batch_size=batch,
            hidden_size=hidden,
            dtype=nl.bfloat16,
            lnc_degree=compiler_args.logical_nc_config,
            lower_bound=lower_bound,
            quant_type=quant_type,
            quant_dtype=nl.float8_e4m3,
            quant_only=quant_only,
            seqlen=seqlen,
            tensor_gen=static_scale_wrapper(tensor_gen) if quant_type == QuantizationType.STATIC else tensor_gen,
        )

    # fmt: off
    rmsnorm_quant_vnc_params = "seqlen, lower_bound, lnc_degree, batch, hidden, intermediate, quant_type"
    rmsnorm_quant_vnc_perms = [
        # seqlen, lower_bound, lnc_degree, batch, hidden, intermediate, quant_type
        (128, 0.0, 1, 1, 16384, 896, QuantizationType.ROW),
        (128, 0.5, 1, 1, 16384, 896, QuantizationType.ROW),
        (128, 0.0, 2, 1, 16384, 896, QuantizationType.ROW),
        (128, 0.5, 2, 1, 16384, 896, QuantizationType.ROW),
        (256, 0.0, 2, 1, 8192, 512, QuantizationType.ROW),
        ####################################
        # Model-specific test cases based on https://tiny.amazon.com/mdelqa5t/quipaFNb
        ####################################
        # Llama models
        (2048, 0.0, 2, 1, 8192, 896, QuantizationType.ROW),
        (2048, 0.0, 2, 1, 16384, 896, QuantizationType.ROW),
        (32768, 0.0, 2, 1, 8192, 896, QuantizationType.ROW),
        (32768, 0.0, 2, 1, 16384, 896, QuantizationType.ROW),
    ]
    # fmt: on

    @pytest.mark.fast
    @pytest.mark.parametrize(rmsnorm_quant_vnc_params, rmsnorm_quant_vnc_perms)
    def test_rmsnorm_quant_vnc_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        seqlen,
        lower_bound,
        lnc_degree,
        batch,
        hidden,
        intermediate,
        quant_type,
    ):
        compiler_args = CompilerArgs(logical_nc_config=lnc_degree)
        self.run_rmsnorm_quant_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
            batch_size=batch,
            hidden_size=hidden,
            dtype=nl.float16,
            lnc_degree=compiler_args.logical_nc_config,
            lower_bound=lower_bound,
            quant_type=quant_type,
            quant_dtype=nl.float8_e4m3,
            seqlen=seqlen,
            tensor_gen=quant_input_tensor_generator(use_rng=True),
        )

    _sweep_seqlen_values = sorted(set(random.sample(range(128, 32769, 128), 15) + [128, 32768]))
    _sweep_hidden_values = sorted(set(random.sample(range(1024, 16385, 512), 8) + [1024, 16384]))
    _sweep_lower_bound_values = sorted(set([round(x, 2) for x in [random.uniform(0.0, 1.0) for _ in range(5)]] + [0.0]))

    @pytest.mark.coverage_parametrize(
        # MAX_S=32768, MAX_H=16384, MAX_B=2
        seqlen=BoundedRange(_sweep_seqlen_values, boundary_values=[32769]),
        hidden=BoundedRange(_sweep_hidden_values, boundary_values=[16385]),
        batch=BoundedRange([1, 2], boundary_values=[0, 3]),
        lower_bound=BoundedRange(_sweep_lower_bound_values, boundary_values=[-0.1]),
        coverage="pairs",
    )
    def test_rmsnorm_row_quant_sweep(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        seqlen,
        hidden,
        batch,
        lower_bound,
        is_negative_test_case,
    ):
        """Sweep test for ROW quantization with pairwise coverage."""
        with assert_negative_test_case(is_negative_test_case):
            self.run_rmsnorm_quant_test(
                test_manager=test_manager,
                compiler_args=CompilerArgs(),
                collector=collector,
                batch_size=batch,
                hidden_size=hidden,
                dtype=nl.bfloat16,
                lnc_degree=1,
                lower_bound=lower_bound,
                quant_type=QuantizationType.ROW,
                quant_dtype=nl.float8_e4m3,
                seqlen=seqlen,
            )
