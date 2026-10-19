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

"""Constants for FP8 quantization."""

"""
Minimum scale value to prevent division by zero in row-wise quantization.
When per-row absmax is zero (all-zero row), dequant_scale is clamped to this value.
Matches the existing 1e-5 floor used in test_mlp_common.py's perform_row_quant.
"""
MINVAL = 1e-5

# FP8_MAXVAL is NOT hardcoded here. Use get_max_positive_value_for_dtype(dtype)
# from kernel_helpers.py, which returns 240.0 for float8_e4m3 or 448.0 for float8_e4m3fn.
