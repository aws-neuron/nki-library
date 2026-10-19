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
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.quantize_mxfp8_utils import (
    INTERLEAVE_FACTOR,
    get_fp8_dtype,
    get_fp8_dtype_x4,
    get_scale_output_shape,
    get_scale_packing_info,
)
from nkilib_src.nkilib.experimental.quantize_mxfp8.quantize_mxfp8_torch import (
    Q_TILE_K,
    _interleave_tensor,
    _pack_scales,
    _quantize_mx_alt_emax,
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


def generate_golden_packed_scales(golden_scales: np.ndarray, K: int, F: int, Q_TILE_K: int) -> np.ndarray:
    """Re-export for test_matmul_mxfp8_generic."""
    return _pack_scales(golden_scales, K, F, enable_scale_packing=True)


def _generate_golden(
    src_tensor: np.ndarray, K: int, F: int, return_fp8_dtype: str, enable_scale_packing: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Compute golden data and scales using _quantize_mx_alt_emax."""
    x4_dtype = get_fp8_dtype_x4(return_fp8_dtype)
    non_x4_dtype = get_fp8_dtype(return_fp8_dtype)
    interleaved = _interleave_tensor(src_tensor.T)  # (K, F) -> (K//4, F*4)
    golden_data_x4, golden_scales = _quantize_mx_alt_emax(interleaved, x4_dtype)
    golden_data = golden_data_x4.view(non_x4_dtype).reshape(K // INTERLEAVE_FACTOR, F * INTERLEAVE_FACTOR)
    return golden_data, golden_scales


def _get_non_padded_rows(K: int, F: int, enable_scale_packing: bool) -> list[int]:
    """Return indices of non-padded rows in the scale output tensor."""
    if not enable_scale_packing:
        golden_num_rows = K // 32  # K // INTERLEAVE_FACTOR // 8
        return [(i // 4) * 32 + (i % 4) for i in range(golden_num_rows)]

    L_TILE_K = 512
    NUM_TILES_IN_K = K // L_TILE_K
    REMAINDER_K = K % L_TILE_K
    HAS_REMAINDER_256 = REMAINDER_K >= 256
    HAS_REMAINDER_128 = REMAINDER_K % 256 >= 128
    non_padded_rows = []
    tile_idx = 0

    def _collect(tile_k_size, tile_idx, k_idx_within_tile=0):
        scaling_group_idx, _, slot_partition_offset = get_scale_packing_info(tile_idx, True)
        scale_p_start = scaling_group_idx * Q_TILE_K
        num_rows = (tile_k_size // INTERLEAVE_FACTOR) // 8
        for row_idx in range(num_rows):
            packed_row = (row_idx // 4) * 32 + (row_idx % 4)
            non_padded_rows.append(scale_p_start + slot_partition_offset + packed_row)

    for _i in range(NUM_TILES_IN_K):
        _collect(L_TILE_K, tile_idx)
        tile_idx += 1

    if HAS_REMAINDER_256:
        _collect(256, tile_idx)
    if HAS_REMAINDER_128:
        _collect(128, tile_idx)

    return non_padded_rows


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
            self._print_with_log("Data comparison failed")
        return passed


class _ScalesValidator(CustomValidator):
    def __init__(
        self,
        logfile: TextIO | None,
        golden_scales: np.ndarray,
        K: int,
        F: int,
        enable_scale_packing: bool,
        non_padded_rows: list[int],
    ):
        super().__init__(logfile)
        self.golden_scales = golden_scales
        self.K = K
        self.F = F
        self.enable_scale_packing = enable_scale_packing
        self.non_padded_rows = non_padded_rows

    @override
    def validate(self, inference_output: npt.NDArray[Any]) -> bool:
        scale_P, scale_F = get_scale_output_shape(self.K, self.F, Q_TILE_K, self.enable_scale_packing)
        output_scales = inference_output.reshape(scale_P, scale_F)

        # Compare raw bytes to avoid NaN != NaN issues with float8_e8m0fnu
        # (value 255 encodes NaN, and np.array_equal would return False for NaN==NaN)
        output_bytes = output_scales[self.non_padded_rows].view(np.uint8)
        golden_bytes = self.golden_scales[self.non_padded_rows].view(np.uint8)
        passed = np.array_equal(output_bytes, golden_bytes)
        if not passed:
            diff = np.abs(output_bytes.astype(np.int32) - golden_bytes.astype(np.int32))
            self._print_with_log(f"Scales mismatch: max diff={np.max(diff)}")
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
    enable_scale_packing: bool,
) -> dict:
    """Generate kernel inputs as a dict matching kernel signature."""
    dist_name, params = get_random_distributions(num=1, seed=None)[0]
    dist = DistributionRegistry.get(dist_name)
    input_tensor = dist.init_fn(torch.empty((F, K)), **params).to(torch.bfloat16).float().numpy().astype(nl.bfloat16)
    return {
        "src_tensor": input_tensor,
        "return_fp8_dtype": return_fp8_dtype,
        "run_with_lnc2": run_with_lnc2,
        "enable_scale_packing": enable_scale_packing,
    }


def build_output_tensors(kernel_input: dict) -> dict:
    """Build output tensor placeholders from kernel inputs."""
    src_tensor = kernel_input["src_tensor"]
    F, K = src_tensor.shape
    return_fp8_dtype = kernel_input["return_fp8_dtype"]
    enable_scale_packing = kernel_input["enable_scale_packing"]

    non_x4_dtype = get_fp8_dtype(return_fp8_dtype)
    scale_P, scale_F = get_scale_output_shape(K, F, Q_TILE_K, enable_scale_packing)

    return {
        "quantized_scales_hbm": np.zeros((scale_P, scale_F), dtype=nl.float8_e8m0fnu),
        "quantized_data_hbm": np.zeros((K // INTERLEAVE_FACTOR, F * INTERLEAVE_FACTOR), dtype=non_x4_dtype),
    }


def build_custom_validation_args(kernel_input: dict) -> ValidationArgs:
    """Build ValidationArgs using the COPT golden method and custom validators."""
    src_tensor = kernel_input["src_tensor"]
    F, K = src_tensor.shape
    return_fp8_dtype = kernel_input["return_fp8_dtype"]
    enable_scale_packing = kernel_input["enable_scale_packing"]

    non_x4_dtype = get_fp8_dtype(return_fp8_dtype)
    scale_P, scale_F = get_scale_output_shape(K, F, Q_TILE_K, enable_scale_packing)

    golden_data, golden_scales_raw = _generate_golden(src_tensor, K, F, return_fp8_dtype, enable_scale_packing)
    golden_scales = _pack_scales(golden_scales_raw, K, F, enable_scale_packing)
    non_padded_rows = _get_non_padded_rows(K, F, enable_scale_packing)

    # The validation contract takes a validator class constructed with just the logfile,
    # so bind this test case's expectations into subclasses.
    class _BoundScalesValidator(_ScalesValidator):
        def __init__(self, logfile: TextIO | None = None):
            super().__init__(logfile, golden_scales, K, F, enable_scale_packing, non_padded_rows)

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
