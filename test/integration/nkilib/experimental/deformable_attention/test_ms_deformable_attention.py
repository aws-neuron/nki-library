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

"""Tests for Multi-Scale Deformable Attention kernel using UnitTestFramework."""

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.deformable_attention.ms_deformable_attention import (
    ms_deformable_attention,
)
from nkilib_src.nkilib.experimental.deformable_attention.ms_deformable_attention_torch import (
    ms_deformable_attention_torch_ref,
)

from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import CompilerArgs, LazyGoldenGenerator, Platforms, ValidationArgs
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def generate_ms_deformable_attention_inputs(
    batch: int,
    n_queries: int,
    n_heads: int,
    c_head: int,
    n_levels: int,
    n_points: int,
    spatial_shapes: list,
    dtype,
    value_layout: str = "BLNC",
    sampling_locations_layout: str = "BQHLP2",
    align_corners: bool = False,
    padding_mode: str = "zeros",
):
    """
    Generate inputs for multi-scale deformable attention kernel test.

    Args:
        batch: Batch size (B)
        n_queries: Number of queries (N_q)
        n_heads: Number of attention heads (N_h)
        c_head: Channels per head (C_h)
        n_levels: Number of feature pyramid levels (N_l)
        n_points: Number of sampling points per query per head per level (N_p)
        spatial_shapes: List of (H_i, W_i) tuples for each level
        dtype: Data type for tensors
        value_layout: "BLNC" or "BNLC"
        sampling_locations_layout: "BQHLP2" or "B2QHLP"
        align_corners: If True, [0,1] maps to [0, H-1]. If False, [0,1] maps to [-0.5, H-0.5]
        padding_mode: One of "zeros", "border", "reflection"

    Returns:
        dict: Dictionary of input tensors for kernel
    """
    np.random.seed(42)
    generate_tensor = gaussian_tensor_generator()

    # Compute total flattened length L and level start indices
    level_start_index_list = [0]
    for h, w in spatial_shapes:
        level_start_index_list.append(level_start_index_list[-1] + h * w)
    L = level_start_index_list[-1]
    level_start_index_list = level_start_index_list[:-1]

    # Generate value tensor based on layout
    if value_layout == "BLNC":
        value = generate_tensor(name="value", shape=(batch, L, n_heads, c_head), dtype=dtype)
    else:  # BNLC
        value = generate_tensor(name="value", shape=(batch, n_heads, L, c_head), dtype=dtype)

    spatial_shapes_tuple = tuple(tuple(shape) for shape in spatial_shapes)
    level_start_index_tuple = tuple(level_start_index_list)

    # Generate sampling_locations based on layout
    if sampling_locations_layout == "BQHLP2":
        sampling_locations = np.random.uniform(0.0, 1.0, size=(batch, n_queries, n_heads, n_levels, n_points, 2))
    else:  # B2QHLP
        sampling_locations = np.random.uniform(0.0, 1.0, size=(batch, 2, n_queries, n_heads, n_levels, n_points))
    sampling_locations = sampling_locations.astype(np.float32 if dtype == nl.float32 else np.float16)

    # Generate random attention_weights (B, N_q, N_h, N_l, N_p) that sum to 1 over (N_l, N_p)
    attention_weights = np.random.uniform(0.1, 1.0, size=(batch, n_queries, n_heads, n_levels, n_points))

    # Normalize to sum to 1 over (l, p) for each (b, q, h)
    attention_weights = attention_weights / attention_weights.sum(axis=(3, 4), keepdims=True)
    attention_weights = attention_weights.astype(np.float32 if dtype == nl.float32 else np.float16)

    return {
        "value": value,
        "spatial_shapes": spatial_shapes_tuple,
        "level_start_index": level_start_index_tuple,
        "sampling_locations": sampling_locations,
        "attention_weights": attention_weights,
        "value_layout": value_layout,
        "sampling_locations_layout": sampling_locations_layout,
        "align_corners": align_corners,
        "padding_mode": padding_mode,
    }


# fmt: off
MS_DEFORM_ATTN_PARAM_NAMES = (
    "batch, n_queries, n_heads, c_head, n_levels, n_points, spatial_shapes, dtype, value_layout, sampling_locations_layout, align_corners, padding_mode"
)

# Basic test parameters
MS_DEFORM_ATTN_BASIC_PARAMS = [
    # align_corners = False, padding_mode = zeros, BLNC layout
    (2, 256, 8, 32, 4, 8, [(50, 45), (25, 21), (13, 9), (5, 7)], nl.bfloat16, "BLNC", "B2QHLP", False, "zeros"),
    (2, 256, 8, 32, 4, 8, [(50, 45), (25, 21), (13, 9), (5, 7)], nl.bfloat16, "BLNC", "BQHLP2", False, "zeros"),
    # align_corners = False, padding_mode = zeros, BNLC layout
    (1, 256, 8, 32, 4, 8, [(50, 45), (25, 21), (13, 9), (5, 7)], nl.bfloat16, "BNLC", "B2QHLP", False, "zeros"),
    (1, 256, 8, 32, 4, 8, [(50, 45), (25, 21), (13, 9), (5, 7)], nl.bfloat16, "BNLC", "BQHLP2", False, "zeros"),
    # align_corners = False, padding_mode = border
    (1, 256, 8, 32, 4, 8, [(50, 45), (25, 21), (13, 9), (5, 7)], nl.bfloat16, "BLNC", "B2QHLP", False, "border"),
    (1, 512, 8, 32, 4, 8, [(50, 45), (25, 21), (13, 9), (5, 7)], nl.bfloat16, "BLNC", "BQHLP2", False, "border"),
    # align_corners = True, padding_mode = zeros
    (1, 256, 8, 32, 4, 8, [(50, 45), (25, 21), (13, 9), (5, 7)], nl.bfloat16, "BLNC", "B2QHLP", True, "zeros"),
    (1, 256, 8, 32, 4, 8, [(50, 45), (25, 21), (13, 9), (5, 7)], nl.bfloat16, "BLNC", "BQHLP2", True, "zeros"),
    # align_corners = True, padding_mode = border
    (1, 256, 8, 32, 4, 8, [(50, 45), (25, 21), (13, 9), (5, 7)], nl.bfloat16, "BLNC", "B2QHLP", True, "border"),
    (1, 256, 8, 32, 4, 8, [(50, 45), (25, 21), (13, 9), (5, 7)], nl.bfloat16, "BLNC", "BQHLP2", True, "border"),
    # C_h > 128
    (1, 256, 4, 256, 4, 4, [(50, 45), (25, 21), (13, 9), (5, 7)], nl.bfloat16, "BLNC", "B2QHLP", False, "zeros"),
    (1, 256, 4, 144, 4, 4, [(50, 45), (25, 21), (13, 9), (5, 7)], nl.bfloat16, "BNLC", "BQHLP2", False, "zeros"),
    # N_q % 128 != 0
    (1, 263, 8, 32, 4, 8, [(50, 45), (25, 21), (13, 9), (5, 7)], nl.bfloat16, "BLNC", "B2QHLP", False, "zeros"),
    (1, 501, 8, 32, 4, 8, [(50, 45), (25, 21), (13, 9), (5, 7)], nl.bfloat16, "BNLC", "B2QHLP", False, "zeros"),
]
MS_DEFORM_ATTN_BASIC_ALL_PARAMS = [
    pytest.param(*c, marks=pytest.mark.fast) for c in MS_DEFORM_ATTN_BASIC_PARAMS
]

# BEVFormer test parameters
MS_DEFORM_ATTN_BEVFORMER_FAST_PARAMS = [
    (2, 512, 8, 32, 4, 8, [(50, 45), (25, 21), (13, 9), (5, 7)], nl.bfloat16, "BLNC", "B2QHLP", False, "zeros"),
]
MS_DEFORM_ATTN_BEVFORMER_SLOW_PARAMS = [
    # BEVFormer Base (200x200)
    (1, 18000, 8, 32, 4, 8, [(113, 200), (57, 100), (29, 50), (15, 25)], nl.bfloat16, "BNLC", "B2QHLP", False, "zeros"), # SCA
    (1, 40000, 8, 32, 1, 4, [(200, 200)], nl.bfloat16, "BLNC", "B2QHLP", False, "zeros"), # TSA
    (1, 900, 8, 32, 1, 4, [(200, 200)], nl.bfloat16, "BLNC", "B2QHLP", False, "zeros"), # Decoder
    # BEVFormer Small (150x150)
    (1, 13500, 8, 32, 1, 8, [(113, 200)], nl.bfloat16, "BLNC", "B2QHLP", False, "zeros"), # SCA
    (1, 22500, 8, 32, 1, 4, [(150, 150)], nl.bfloat16, "BLNC", "B2QHLP", False, "zeros"), # TSA
    (1, 900, 8, 32, 1, 4, [(150, 150)], nl.bfloat16, "BLNC", "B2QHLP", False, "zeros") # Decoder
]

MS_DEFORM_ATTN_BEVFORMER_ALL_PARAMS = [
    pytest.param(*c, marks=pytest.mark.fast) for c in MS_DEFORM_ATTN_BEVFORMER_FAST_PARAMS
] + MS_DEFORM_ATTN_BEVFORMER_SLOW_PARAMS
# fmt: on


@pytest_test_metadata(name="MS Deformable Attention")
@pytest_marks(["deformable_attention", "ms_deform_attn"])
class TestMSDeformableAttention:
    @pytest_marks(["basic"])
    @pytest.mark.parametrize(MS_DEFORM_ATTN_PARAM_NAMES, MS_DEFORM_ATTN_BASIC_ALL_PARAMS)
    def test_ms_deformable_attention_basic(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        n_queries,
        n_heads,
        c_head,
        n_levels,
        n_points,
        spatial_shapes,
        dtype,
        value_layout,
        sampling_locations_layout,
        align_corners,
        padding_mode,
    ):
        """Test multi-scale deformable attention basic configurations."""

        def input_generator(test_config):
            return generate_ms_deformable_attention_inputs(
                batch=batch,
                n_queries=n_queries,
                n_heads=n_heads,
                c_head=c_head,
                n_levels=n_levels,
                n_points=n_points,
                spatial_shapes=spatial_shapes,
                dtype=dtype,
                value_layout=value_layout,
                sampling_locations_layout=sampling_locations_layout,
                align_corners=align_corners,
                padding_mode=padding_mode,
            )

        def output_tensors(kernel_input):
            return {
                "out": np.zeros((batch, n_queries, n_heads * c_head), dtype=dtype),
            }

        def golden_generator():
            """Generate golden output using torch reference."""
            kernel_input = input_generator(None)
            return torch_ref_wrapper(ms_deformable_attention_torch_ref)(**kernel_input)

        # Create and run test framework
        test_framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=ms_deformable_attention,
            torch_ref=torch_ref_wrapper(ms_deformable_attention_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )

        custom_validation = ValidationArgs(
            golden_output=LazyGoldenGenerator(
                output_ndarray=output_tensors(None),
                lazy_golden_generator=golden_generator,
            ),
            relative_accuracy=0.01,
            absolute_accuracy=1e-06,
        )

        # Run test with custom validation
        test_framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            custom_validation_args=custom_validation,
        )

    @pytest_marks(["bevformer"])
    @pytest.mark.parametrize(MS_DEFORM_ATTN_PARAM_NAMES, MS_DEFORM_ATTN_BEVFORMER_ALL_PARAMS)
    def test_ms_deformable_attention_bevformer(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        n_queries,
        n_heads,
        c_head,
        n_levels,
        n_points,
        spatial_shapes,
        dtype,
        value_layout,
        sampling_locations_layout,
        align_corners,
        padding_mode,
    ):
        """Test multi-scale deformable attention with BEVFormer configs."""

        def input_generator(test_config):
            return generate_ms_deformable_attention_inputs(
                batch=batch,
                n_queries=n_queries,
                n_heads=n_heads,
                c_head=c_head,
                n_levels=n_levels,
                n_points=n_points,
                spatial_shapes=spatial_shapes,
                dtype=dtype,
                value_layout=value_layout,
                sampling_locations_layout=sampling_locations_layout,
                align_corners=align_corners,
                padding_mode=padding_mode,
            )

        def output_tensors(kernel_input):
            return {
                "out": np.zeros((batch, n_queries, n_heads * c_head), dtype=dtype),
            }

        def golden_generator():
            """Generate golden output using torch reference."""
            kernel_input = input_generator(None)
            return torch_ref_wrapper(ms_deformable_attention_torch_ref)(**kernel_input)

        # Create test framework
        test_framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=ms_deformable_attention,
            torch_ref=torch_ref_wrapper(ms_deformable_attention_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )

        custom_validation = ValidationArgs(
            golden_output=LazyGoldenGenerator(
                output_ndarray=output_tensors(None),
                lazy_golden_generator=golden_generator,
            ),
            relative_accuracy=0.01,
            absolute_accuracy=1e-06,
        )

        # Run test with custom validation
        test_framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            custom_validation_args=custom_validation,
        )
