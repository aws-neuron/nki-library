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

"""Tests for the generic pad kernel — all modes, dimensionalities, and sizes."""

from typing import final

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.pad.pad import pad
from nkilib_src.nkilib.experimental.pad.pad_torch import pad_torch_ref

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.coverage_parametrized_tests import FilterResult
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _output_shape(shape, padding):
    out = list(shape)
    n_padded = len(padding) // 2
    for i in range(n_padded):
        dim = len(shape) - 1 - i
        out[dim] += padding[2 * i] + padding[2 * i + 1]
    return tuple(out)


def _filter_pad_combinations(shape_and_pad, mode, dtype=None):
    """Filter out invalid combinations."""
    shape, padding = shape_and_pad
    ndim = len(shape)
    pad_len = len(padding)

    expected_pad_len = (ndim - 2) * 2
    if pad_len != expected_pad_len:
        return FilterResult.INVALID

    if mode == "reflect":
        for i in range(pad_len // 2):
            dim = ndim - 1 - i
            if shape[dim] <= 1:
                return FilterResult.INVALID
            if padding[2 * i] >= shape[dim] or padding[2 * i + 1] >= shape[dim]:
                return FilterResult.INVALID

    if mode == "circular":
        for i in range(pad_len // 2):
            dim = ndim - 1 - i
            if padding[2 * i] > shape[dim] or padding[2 * i + 1] > shape[dim]:
                return FilterResult.INVALID

    return FilterResult.VALID


# fmt: off
_SHAPES_AND_PADS_FAST = [
    # --- Small correctness tests ---
    ((1, 2, 4, 4, 4),       (1, 1, 1, 1, 1, 1)),
    ((1, 2, 4, 4, 4),       (1, 2, 0, 0, 0, 0)),
    ((1, 2, 4, 4, 4),       (0, 0, 0, 0, 2, 3)),
    ((1, 2, 8, 2, 2),       (1, 1, 1, 1, 1, 1)),
    ((1, 2, 8, 8),          (1, 1, 1, 1)),
    ((1, 2, 8, 8),          (2, 3, 0, 0)),
    ((1, 2, 16),            (1, 1)),
    ((1, 2, 16),            (3, 2)),
    ((1, 4, 4, 4),          (1, 1, 1, 1)),
    # tile_dim=0, ts>1 (D tiled)
    ((1, 1, 64, 8, 8),      (0, 0, 0, 0, 3, 3)),
    ((1, 1, 64, 8, 8),      (1, 1, 1, 1, 3, 3)),
    ((1, 1, 128, 4, 4),     (1, 1, 1, 1, 2, 2)),
    # tile_dim=0, ts=1 (D fully sliced)
    ((1, 4, 8, 8, 8),       (2, 0, 0, 3, 1, 0)),
    ((1, 2, 8, 64, 64),     (1, 1, 1, 1, 1, 1)),
    ((1, 2, 8, 64, 64),     (0, 0, 0, 0, 3, 3)),
    ((1, 1, 16, 32, 64),    (1, 1, 2, 2, 3, 3)),
    # tile_dim=1, ts=1 (H fully sliced)
    ((1, 2, 4, 8, 2048),    (1, 1, 1, 1, 1, 1)),
    ((1, 2, 2, 4, 2048),    (1, 1, 0, 0, 0, 0)),
    ((1, 1, 2, 4, 4096),    (0, 0, 2, 2, 1, 1)),
    # 4D/3D
    ((1, 2, 32, 64),        (1, 1, 1, 1)),
    ((1, 4, 2048),          (3, 3)),
    # Large spatial
    ((1, 2, 2, 128, 128),   (1, 1, 1, 1, 0, 0)),
    ((1, 2, 2, 2, 4096),    (1, 1, 0, 0, 0, 0)),
    # --- Large perf-meaningful tests ---
    ((1, 512, 121, 40, 40), (1, 1, 1, 1, 2, 0)),
    ((1, 256, 8, 64, 64),   (1, 1, 1, 1, 1, 1)),
    ((1, 256, 16, 32, 64),  (1, 1, 2, 2, 3, 3)),
    ((1, 256, 64, 8, 8),    (1, 1, 1, 1, 3, 3)),
    ((1, 256, 4, 8, 2048),  (1, 1, 1, 1, 1, 1)),
    ((1, 256, 64, 64),      (1, 1, 1, 1)),
    ((1, 256, 4096),         (3, 3)),
    ((1, 128, 1, 1024, 1024), (1, 1, 1, 1, 0, 0)),
]

_SHAPES_AND_PADS_FULL = [
    ((1, 256, 1, 1024, 1024), (1, 1, 1, 1, 0, 0)),
    ((1, 256, 1, 1000, 1000), (1, 1, 1, 1, 0, 0)),
]
# fmt: on


@final
@pytest_test_metadata(name="Generic Pad", pytest_marks=["pad"])
class TestPad:
    @staticmethod
    def generate_inputs(shape_and_pad, mode, dtype):
        shape, padding = shape_and_pad
        np.random.seed(42)
        return {
            "x_ref": np.random.randn(*shape).astype(dtype),
            "padding": padding,
            "mode": mode,
            "value": 0,
        }

    @staticmethod
    def output_tensors(shape_and_pad, dtype):
        shape, padding = shape_and_pad
        return {"out": np.zeros(_output_shape(shape, padding), dtype=dtype)}

    @pytest.mark.fast
    @pytest.mark.coverage_parametrize(
        shape_and_pad=_SHAPES_AND_PADS_FAST,
        mode=["constant", "replicate", "reflect", "circular"],
        dtype=[nl.bfloat16],
        filter=_filter_pad_combinations,
        enable_automatic_boundary_tests=False,
        enable_invalid_combination_tests=False,
    )
    def test_pad_fast(
        self, test_manager: Orchestrator, platform_target: Platforms, shape_and_pad, mode, dtype, is_negative_test_case
    ):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=pad,
            torch_ref=torch_ref_wrapper(pad_torch_ref),
            kernel_input_generator=lambda _: self.generate_inputs(shape_and_pad, mode, dtype),
            output_tensor_descriptor=lambda _: self.output_tensors(shape_and_pad, dtype),
        )
        framework.run_test(
            test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=0, atol=0
        )

    @pytest.mark.coverage_parametrize(
        shape_and_pad=_SHAPES_AND_PADS_FULL,
        mode=["constant", "replicate", "reflect", "circular"],
        dtype=[nl.bfloat16],
        filter=_filter_pad_combinations,
        enable_automatic_boundary_tests=False,
        enable_invalid_combination_tests=False,
    )
    def test_pad_full(
        self, test_manager: Orchestrator, platform_target: Platforms, shape_and_pad, mode, dtype, is_negative_test_case
    ):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=pad,
            torch_ref=torch_ref_wrapper(pad_torch_ref),
            kernel_input_generator=lambda _: self.generate_inputs(shape_and_pad, mode, dtype),
            output_tensor_descriptor=lambda _: self.output_tensors(shape_and_pad, dtype),
        )
        framework.run_test(
            test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=0, atol=0
        )
