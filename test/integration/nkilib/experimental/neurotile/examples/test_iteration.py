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
"""Integration tests for tutorials in test/docs/neurotile/examples/01_iteration/."""

import ml_dtypes
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.neurotile.examples._01_iteration import (
    _01_tiles as tiles_mod,
)
from nkilib_src.nkilib.experimental.neurotile.examples._01_iteration import (
    _01_tiles_torch as tiles_refs,
)
from nkilib_src.nkilib.experimental.neurotile.examples._01_iteration import (
    _02_blocks as blocks_mod,
)
from nkilib_src.nkilib.experimental.neurotile.examples._01_iteration import (
    _02_blocks_torch as blocks_refs,
)
from nkilib_src.nkilib.experimental.neurotile.examples._01_iteration import (
    _03_streaming as streaming_mod,
)
from nkilib_src.nkilib.experimental.neurotile.examples._01_iteration import (
    _03_streaming_torch as streaming_refs,
)
from nkilib_src.nkilib.experimental.neurotile.examples._01_iteration import (
    _04_pipeline as pipeline_mod,
)
from nkilib_src.nkilib.experimental.neurotile.examples._01_iteration import (
    _04_pipeline_torch as pipeline_refs,
)
from nkilib_src.nkilib.experimental.neurotile.examples._01_iteration import (
    _05_access_patterns as access_mod,
)
from nkilib_src.nkilib.experimental.neurotile.examples._01_iteration import (
    _05_access_patterns_torch as access_refs,
)
from nkilib_src.nkilib.experimental.neurotile.examples._01_iteration import (
    _06_sliced_sources as sliced_mod,
)
from nkilib_src.nkilib.experimental.neurotile.examples._01_iteration import (
    _06_sliced_sources_torch as sliced_refs,
)

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Default 4×4 tile grid input (512×512 bf16) — matches _make_src() in 01_tiles.py / 02_blocks.py.
_TILE_P, _TILE_F = 128, 128
_DEFAULT_M, _DEFAULT_N = 4 * _TILE_P, 4 * _TILE_F


def _bf16_input(shape, key="src", seed=42):
    np.random.seed(seed)
    return {key: np.random.randn(*shape).astype(ml_dtypes.bfloat16)}


def _zeros_like(input_key="src"):
    def _gen(kernel_input):
        return {"out": np.zeros_like(kernel_input[input_key])}

    return _gen


def _default_src_inputs(_):
    return _bf16_input((_DEFAULT_M, _DEFAULT_N))


def _run(test_manager, platform_target, kernel, ref, inputs, outputs, rtol=1e-2, atol=1e-2):
    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=kernel,
        torch_ref=torch_ref_wrapper(ref),
        kernel_input_generator=inputs,
        output_tensor_descriptor=outputs,
    )
    framework.run_test(
        test_config=None,
        compiler_args=CompilerArgs(platform_target=platform_target),
        rtol=rtol,
        atol=atol,
    )


@pytest_marks(["neurotile"])
class TestNeurotileTiles:
    """Tutorials in 01_tiles.py — tile-iteration kernels."""

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "kernel,ref",
        [
            (tiles_mod.basic_tile_copy, tiles_refs.basic_tile_copy_torch_ref),
            (tiles_mod.direct_indexing, tiles_refs.direct_indexing_torch_ref),
            (tiles_mod.scale_kernel, tiles_refs.scale_torch_ref),
            (tiles_mod.load_tile_row, tiles_refs.load_tile_row_torch_ref),
            (tiles_mod.load_tile_column, tiles_refs.load_tile_column_torch_ref),
            (tiles_mod.load_subgrid, tiles_refs.load_subgrid_torch_ref),
            (tiles_mod.load_negative_index, tiles_refs.load_negative_index_torch_ref),
            (tiles_mod.load_strided_rows, tiles_refs.load_strided_rows_torch_ref),
        ],
    )
    def test_tile_kernel(self, test_manager: Orchestrator, platform_target: Platforms, kernel, ref):
        _run(test_manager, platform_target, kernel, ref, _default_src_inputs, _zeros_like("src"))

    @pytest.mark.fast
    def test_batched_tile_iteration(self, test_manager, platform_target):
        _run(
            test_manager,
            platform_target,
            tiles_mod.batched_tile_iteration,
            tiles_refs.batched_tile_iteration_torch_ref,
            lambda _: _bf16_input((4, 256, 256)),
            _zeros_like("src"),
        )

    @pytest.mark.fast
    def test_higher_rank_tile(self, test_manager, platform_target):
        _run(
            test_manager,
            platform_target,
            tiles_mod.higher_rank_tile,
            tiles_refs.higher_rank_tile_torch_ref,
            lambda _: _bf16_input((256, 4, 64)),
            _zeros_like("src"),
        )

    @pytest.mark.fast
    def test_partition_tile(self, test_manager, platform_target):
        _run(
            test_manager,
            platform_target,
            tiles_mod.partition_tile,
            tiles_refs.partition_tile_torch_ref,
            lambda _: _bf16_input((256, 4)),
            _zeros_like("src"),
        )

    @pytest.mark.fast
    def test_column_tile(self, test_manager, platform_target):
        _run(
            test_manager,
            platform_target,
            tiles_mod.column_tile,
            tiles_refs.column_tile_torch_ref,
            lambda _: _bf16_input((4, 256)),
            _zeros_like("src"),
        )


@pytest_marks(["neurotile"])
class TestNeurotileBlocks:
    """Tutorials in 02_blocks.py — block-iteration kernels.

    All use src shape (4*2*128, 2*2*128) = (1024, 512), per _make_src() in 02_blocks.py.
    """

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "kernel,ref",
        [
            (blocks_mod.block_level_scale, blocks_refs.block_level_scale_torch_ref),
            (blocks_mod.load_block_row, blocks_refs.load_block_row_torch_ref),
            (blocks_mod.load_block_subgrid, blocks_refs.load_block_subgrid_torch_ref),
            (blocks_mod.promote_tile_to_block_view, blocks_refs.promote_tile_to_block_view_torch_ref),
            (blocks_mod.descend_block_to_tile_view, blocks_refs.descend_block_to_tile_view_torch_ref),
            (blocks_mod.retile_block_view, blocks_refs.retile_block_view_torch_ref),
        ],
    )
    def test_block_kernel(self, test_manager, platform_target, kernel, ref):
        _run(
            test_manager,
            platform_target,
            kernel,
            ref,
            lambda _: _bf16_input((1024, 512)),
            _zeros_like("src"),
        )


@pytest_marks(["neurotile"])
class TestNeurotileStreaming:
    """Tutorials in 03_streaming.py — streaming kernels."""

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "kernel,ref",
        [
            (streaming_mod.stream_input_output_scale, streaming_refs.stream_input_output_scale_torch_ref),
            (streaming_mod.stream_dim_walks_columns, streaming_refs.stream_dim_walks_columns_torch_ref),
            (streaming_mod.stream_2d_blocks, streaming_refs.stream_2d_blocks_torch_ref),
        ],
    )
    def test_streaming_kernel(self, test_manager, platform_target, kernel, ref):
        _run(
            test_manager,
            platform_target,
            kernel,
            ref,
            lambda _: _bf16_input((512, 1024)),
            _zeros_like("src"),
        )

    @pytest.mark.fast
    def test_stream_with_dtype_conversion(self, test_manager, platform_target):
        # Output shape is (128, 1024), not src.shape — kernel reduces along block dim.
        def _outputs(kernel_input):
            return {"out": np.zeros((128, 1024), dtype=ml_dtypes.bfloat16)}

        _run(
            test_manager,
            platform_target,
            streaming_mod.stream_with_dtype_conversion,
            streaming_refs.stream_with_dtype_conversion_torch_ref,
            lambda _: _bf16_input((512, 1024)),
            _outputs,
        )


@pytest_marks(["neurotile"])
class TestNeurotilePipeline:
    """Tutorials in 04_pipeline.py — pipelined kernels."""

    @pytest.mark.fast
    def test_explicit_prefetch_matmul(self, test_manager, platform_target):
        def _inputs(_):
            np.random.seed(42)
            return {
                "AT_hbm": np.random.randn(256, 256).astype(ml_dtypes.bfloat16),
                "B_hbm": np.random.randn(256, 256).astype(ml_dtypes.bfloat16),
            }

        def _outputs(kernel_input):
            return {"out": np.zeros((256, 256), dtype=ml_dtypes.bfloat16)}

        _run(
            test_manager,
            platform_target,
            pipeline_mod.explicit_prefetch_matmul,
            pipeline_refs.explicit_prefetch_matmul_torch_ref,
            _inputs,
            _outputs,
        )

    @pytest.mark.fast
    def test_triple_buffer_pipeline(self, test_manager, platform_target):
        _run(
            test_manager,
            platform_target,
            pipeline_mod.triple_buffer_pipeline,
            pipeline_refs.triple_buffer_pipeline_torch_ref,
            lambda _: _bf16_input((512, 256)),
            _zeros_like("src"),
        )


@pytest_marks(["neurotile"])
class TestNeurotileAccessPatterns:
    """Tutorials in 05_access_patterns.py."""

    @pytest.mark.fast
    def test_contiguous_access_pattern(self, test_manager, platform_target):
        _run(
            test_manager,
            platform_target,
            access_mod.contiguous_access_pattern,
            access_refs.contiguous_access_pattern_torch_ref,
            lambda _: _bf16_input((256, 512)),
            _zeros_like("src"),
        )

    @pytest.mark.fast
    def test_strided_access_pattern(self, test_manager, platform_target):
        # Output is half the rows: src[::2, :]
        def _outputs(kernel_input):
            shape = kernel_input["src"].shape
            return {"out": np.zeros((shape[0] // 2, shape[1]), dtype=kernel_input["src"].dtype)}

        _run(
            test_manager,
            platform_target,
            access_mod.strided_access_pattern,
            access_refs.strided_access_pattern_torch_ref,
            lambda _: _bf16_input((512, 512)),
            _outputs,
        )


@pytest_marks(["neurotile"])
class TestNeurotileSlicedSources:
    """Tutorials in 06_sliced_sources.py — fp32 inputs."""

    @pytest.mark.fast
    def test_window_scale(self, test_manager, platform_target):
        def _inputs(_):
            np.random.seed(42)
            return {"src": np.random.randn(256, 512).astype(np.float32)}

        def _outputs(_kernel_input):
            return {"out": np.zeros((128, 512), dtype=np.float32)}

        _run(
            test_manager,
            platform_target,
            sliced_mod.window_scale,
            sliced_refs.window_scale_torch_ref,
            _inputs,
            _outputs,
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.fast
    def test_two_windows(self, test_manager, platform_target):
        def _inputs(_):
            np.random.seed(42)
            return {"src": np.random.randn(256, 512).astype(np.float32)}

        _run(
            test_manager,
            platform_target,
            sliced_mod.two_windows,
            sliced_refs.two_windows_torch_ref,
            _inputs,
            _zeros_like("src"),
            rtol=1e-5,
            atol=1e-5,
        )
