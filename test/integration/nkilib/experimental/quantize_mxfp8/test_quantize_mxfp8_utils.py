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

"""Common utilities for quantize_mxfp8 kernel tests: input generation, validation helpers, and re-exports."""

from typing import Any, TextIO

import nki.language as nl
import numpy as np
import numpy.typing as npt
import torch
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils._dense_scales import get_dense_scale_shape
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils._quantize_mxfp8_utils import (
    INTERLEAVE_FACTOR,
    QUANT_SCHEME_1X32,  # noqa: F401 - re-exported for test_quantize_mxfp8
    QUANT_SCHEME_WRAPX,
    get_fp8_dtype,
    get_fp8_dtype_x4,
)
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.common_dataclasses import TensorOrientation
from nkilib_src.nkilib.experimental.quantize_mxfp8.quantize_mxfp8_torch import (
    _quantize_mx_alt_emax,
    interleave_tensor,
    quantize_block_mxfp8_torch_ref,  # noqa: F401 - re-exported for test_quantize_mxfp8
)
from typing_extensions import override

from test.integration.nkilib.experimental.matmul_mxfp8.random_input_generator import (
    DistributionRegistry,
    get_random_distributions,
)
from test.utils.common_dataclasses import (
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    ValidationArgs,
)


def _generate_golden(
    src_tensor: np.ndarray, K: int, F: int, return_fp8_dtype: str, quant_scheme: str
) -> tuple[np.ndarray, np.ndarray]:
    """Compute golden data and scales using _quantize_mx_alt_emax."""
    x4_dtype = get_fp8_dtype_x4(return_fp8_dtype)
    non_x4_dtype = get_fp8_dtype(return_fp8_dtype)
    interleaved = interleave_tensor(src_tensor.T, quant_scheme)  # (K, F) -> (K//4, F*4)
    golden_data_x4, golden_scales = _quantize_mx_alt_emax(interleaved, x4_dtype)
    golden_data = golden_data_x4.view(non_x4_dtype).reshape(K // INTERLEAVE_FACTOR, F * INTERLEAVE_FACTOR)
    return golden_data, golden_scales


# ============================================================================
# Custom validators
# ============================================================================


class _DataValidator(CustomValidator):
    def __init__(self, logfile: TextIO | None, golden_data: np.ndarray, return_fp8_dtype: str):
        super().__init__(logfile)
        self.golden_data = golden_data
        self.return_fp8_dtype = return_fp8_dtype

    @override
    def validate(self, inference_output: npt.NDArray[Any]) -> bool:
        non_x4_dtype = get_fp8_dtype(self.return_fp8_dtype)
        output_data = inference_output.view(dtype=non_x4_dtype).reshape(self.golden_data.shape)
        passed = np.array_equal(output_data.astype(np.float32), self.golden_data.astype(np.float32))
        if not passed:
            bad = np.argwhere(output_data.astype(np.float32) != self.golden_data.astype(np.float32))
            rows = np.unique(bad[:, 0])
            cols = np.unique(bad[:, 1])
            self._print_with_log(
                f"Data comparison failed: shape={self.golden_data.shape} "
                f"n_bad={len(bad)} bad_rows={len(rows)}{rows[:16].tolist()}..{rows[-4:].tolist()} "
                f"bad_cols={len(cols)}{cols[:16].tolist()}..{cols[-4:].tolist()}"
            )
        return passed


class _ScalesValidator(CustomValidator):
    """Compares the dense [K // 32, F] scales; every row holds a real scale, so all are checked."""

    def __init__(self, logfile: TextIO | None, golden_scales: np.ndarray):
        super().__init__(logfile)
        self.golden_scales = golden_scales

    @override
    def validate(self, inference_output: npt.NDArray[Any]) -> bool:
        # Compare raw bytes to avoid NaN != NaN issues with float8_e8m0fnu
        # (value 255 encodes NaN, and np.array_equal would return False for NaN==NaN)
        output_bytes = inference_output.reshape(self.golden_scales.shape).view(np.uint8)
        golden_bytes = self.golden_scales.view(np.uint8)
        passed = np.array_equal(output_bytes, golden_bytes)
        if not passed:
            diff = np.abs(output_bytes.astype(np.int32) - golden_bytes.astype(np.int32))
            bad = np.argwhere(output_bytes != golden_bytes)
            rows = np.unique(bad[:, 0])
            cols = np.unique(bad[:, 1])
            self._print_with_log(
                f"Scales mismatch: max diff={np.max(diff)} shape={self.golden_scales.shape} "
                f"n_bad={len(bad)} bad_rows={len(rows)}{rows[:16].tolist()}..{rows[-4:].tolist()} "
                f"bad_cols={len(cols)}{cols[:16].tolist()}..{cols[-4:].tolist()}"
            )
        return passed


# ============================================================================
# Input generation and output tensor descriptors for UnitTestFramework
# ============================================================================


def generate_quantize_mxfp8_inputs(
    input_dtype: str,
    input_range_low: float,
    input_range_high: float,
    K: int,
    F: int,
    return_fp8_dtype: str,
    run_with_lnc2: bool,
    quant_scheme: str = QUANT_SCHEME_WRAPX,
    use_fast_dma_transpose: bool = False,
    src_orientation: TensorOrientation | None = None,
) -> dict:
    """Generate kernel inputs as a dict matching kernel signature.

    With src_orientation K_BY_F the kernel input is the [K, F] transpose of the generated
    [F, K] tensor, and the key is passed through; otherwise it is omitted (kernel default).
    """
    dist_name, params = get_random_distributions(num=1, seed=42)[0]
    dist = DistributionRegistry.get(dist_name)
    input_tensor = dist.init_fn(torch.empty((F, K)), **params).to(torch.bfloat16).float().numpy().astype(nl.bfloat16)
    kernel_input = {
        "src_tensor": input_tensor,
        "return_fp8_dtype": return_fp8_dtype,
        "run_with_lnc2": run_with_lnc2,
        "quant_scheme": quant_scheme,
        "use_fast_dma_transpose": use_fast_dma_transpose,
    }
    if src_orientation == TensorOrientation.K_BY_F:
        kernel_input["src_tensor"] = np.ascontiguousarray(input_tensor.T)
        kernel_input["src_orientation"] = src_orientation
    return kernel_input


def _f_by_k_source(kernel_input: dict) -> np.ndarray:
    """The kernel input as the logical [F, K] tensor, whatever orientation it was passed in."""
    src_tensor = kernel_input["src_tensor"]
    if kernel_input.get("src_orientation") == TensorOrientation.K_BY_F:
        return np.ascontiguousarray(src_tensor.T)
    return src_tensor


def build_output_tensors(kernel_input: dict) -> dict:
    """Build output tensor placeholders from kernel inputs."""
    F, K = _f_by_k_source(kernel_input).shape
    return_fp8_dtype = kernel_input["return_fp8_dtype"]

    non_x4_dtype = get_fp8_dtype(return_fp8_dtype)
    scale_P, scale_F = get_dense_scale_shape(K, F)

    return {
        "quantized_scales_hbm": np.zeros((scale_P, scale_F), dtype=nl.float8_e8m0fnu),
        "quantized_data_hbm": np.zeros((K // INTERLEAVE_FACTOR, F * INTERLEAVE_FACTOR), dtype=non_x4_dtype),
    }


def build_custom_validation_args(kernel_input: dict) -> ValidationArgs:
    """Build ValidationArgs using the COPT golden method and custom validators."""
    src_tensor = _f_by_k_source(kernel_input)
    F, K = src_tensor.shape
    return_fp8_dtype = kernel_input["return_fp8_dtype"]
    quant_scheme = kernel_input.get("quant_scheme", QUANT_SCHEME_WRAPX)

    non_x4_dtype = get_fp8_dtype(return_fp8_dtype)
    scale_P, scale_F = get_dense_scale_shape(K, F)

    golden_data, golden_scales_raw = _generate_golden(src_tensor, K, F, return_fp8_dtype, quant_scheme)
    # Dense [K // 32, F] exponents, viewed as e8m0 to keep the bytes.
    golden_scales = golden_scales_raw.view(nl.float8_e8m0fnu)

    # The validation contract takes a validator class constructed with just the logfile,
    # so bind this test case's expectations into subclasses.
    class _BoundScalesValidator(_ScalesValidator):
        def __init__(self, logfile: TextIO | None = None):
            super().__init__(logfile, golden_scales)

    class _BoundDataValidator(_DataValidator):
        def __init__(self, logfile: TextIO | None = None):
            super().__init__(logfile, golden_data, return_fp8_dtype)

    return ValidationArgs(
        golden_output={
            "quantized_scales_hbm": CustomValidatorWithOutputTensorData(
                validator=_BoundScalesValidator,
                output_ndarray=np.ndarray(shape=(scale_P, scale_F), dtype=nl.float8_e8m0fnu),
            ),
            "quantized_data_hbm": CustomValidatorWithOutputTensorData(
                validator=_BoundDataValidator,
                output_ndarray=np.ndarray(shape=(K // INTERLEAVE_FACTOR, F * INTERLEAVE_FACTOR), dtype=non_x4_dtype),
            ),
        },
    )
