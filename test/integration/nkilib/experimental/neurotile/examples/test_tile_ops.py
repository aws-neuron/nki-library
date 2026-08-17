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
"""Integration tests for tutorials in test/docs/neurotile/examples/03_tile_ops/."""

import ml_dtypes
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.neurotile.examples._03_tile_ops import (
    _01_reshape_permute as reshape_mod,
)
from nkilib_src.nkilib.experimental.neurotile.examples._03_tile_ops import (
    _01_reshape_permute_torch as reshape_refs,
)
from nkilib_src.nkilib.experimental.neurotile.examples._03_tile_ops import (
    _02_fold_pattern_override as fold_mod,
)
from nkilib_src.nkilib.experimental.neurotile.examples._03_tile_ops import (
    _02_fold_pattern_override_torch as fold_refs,
)
from nkilib_src.nkilib.experimental.neurotile.examples._03_tile_ops import (
    _03_tensor_view as view_mod,
)
from nkilib_src.nkilib.experimental.neurotile.examples._03_tile_ops import (
    _03_tensor_view_torch as view_refs,
)
from nkilib_src.nkilib.experimental.neurotile.examples._03_tile_ops import (
    _04_transpose as transpose_mod,
)
from nkilib_src.nkilib.experimental.neurotile.examples._03_tile_ops import (
    _04_transpose_torch as transpose_refs,
)

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _run(test_manager, platform_target, kernel, ref, inputs, outputs, rtol=1e-2, atol=1e-2, logical_nc_config=None):
    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=kernel,
        torch_ref=torch_ref_wrapper(ref),
        kernel_input_generator=inputs,
        output_tensor_descriptor=outputs,
    )
    framework.run_test(
        test_config=None,
        compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=logical_nc_config),
        rtol=rtol,
        atol=atol,
    )


@pytest_marks(["neurotile"])
class TestNeurotileReshapePermute:
    """Tutorials in 01_reshape_permute.py."""

    @pytest.mark.fast
    def test_norm_reshape_transpose(self, test_manager: Orchestrator, platform_target: Platforms):
        # src: [4, 1024] fp32, out: [128, 32]
        _run(
            test_manager,
            platform_target,
            reshape_mod.norm_reshape_transpose,
            reshape_refs.norm_reshape_transpose_torch_ref,
            lambda _: {"src_hbm": np.random.RandomState(42).randn(4, 1024).astype(np.float32)},
            lambda _: {"out": np.zeros((128, 32), dtype=np.float32)},
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.fast
    def test_reshape_permute_load(self, test_manager, platform_target):
        # src: [4, 1024] fp32, out: [128, 4, 8]
        _run(
            test_manager,
            platform_target,
            reshape_mod.reshape_permute_load,
            reshape_refs.reshape_permute_load_torch_ref,
            lambda _: {"src_hbm": np.random.RandomState(42).randn(4, 1024).astype(np.float32)},
            lambda _: {"out": np.zeros((128, 4, 8), dtype=np.float32)},
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.fast
    def test_bsh_to_h0_bs_h1(self, test_manager, platform_target):
        # B=4, S=32, H=128 => out: [H0=8, B*S=128, H1=16]
        _run(
            test_manager,
            platform_target,
            reshape_mod.bsh_to_h0_bs_h1,
            reshape_refs.bsh_to_h0_bs_h1_torch_ref,
            lambda _: {"src_hbm": np.random.RandomState(42).randn(4, 32, 128).astype(np.float32)},
            lambda _: {"out": np.zeros((8, 128, 16), dtype=np.float32)},
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.fast
    def test_attention_qk_layout(self, test_manager, platform_target):
        # q: [B=4, S=32, H=128], out: [head_dim=16, B*S=128]
        _run(
            test_manager,
            platform_target,
            reshape_mod.attention_qk_layout,
            reshape_refs.attention_qk_layout_torch_ref,
            lambda _: {"q_hbm": np.random.RandomState(42).randn(4, 32, 128).astype(np.float32)},
            lambda _: {"out": np.zeros((16, 128), dtype=np.float32)},
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "kernel,ref,shape",
        [
            (reshape_mod.tile_reshape_elementwise, reshape_refs.tile_reshape_elementwise_torch_ref, (128, 512)),
            (reshape_mod.block_reshape_elementwise, reshape_refs.block_reshape_elementwise_torch_ref, (256, 512)),
            (reshape_mod.block_reshape_chunked, reshape_refs.block_reshape_chunked_torch_ref, (256, 512)),
        ],
    )
    def test_reshape_elementwise(self, test_manager, platform_target, kernel, ref, shape):
        _run(
            test_manager,
            platform_target,
            kernel,
            ref,
            lambda _: {"src": np.random.RandomState(42).randn(*shape).astype(ml_dtypes.bfloat16)},
            lambda ki: {"out": np.zeros_like(ki["src"])},
        )


@pytest_marks(["neurotile"])
class TestNeurotileFoldPattern:
    """Tutorials in 02_fold_pattern_override.py."""

    @pytest.mark.fast
    def test_fold_into_partition(self, test_manager, platform_target):
        # src: [P=32, F=512, K=4] -> out: [K*P=128, F=512]
        _run(
            test_manager,
            platform_target,
            fold_mod.fold_into_partition,
            fold_refs.fold_into_partition_torch_ref,
            lambda _: {"src_hbm": np.random.RandomState(42).randn(32, 512, 4).astype(np.float32)},
            lambda _: {"out": np.zeros((128, 512), dtype=np.float32)},
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.fast
    def test_fold_free_dim(self, test_manager, platform_target):
        # src: [P=32, F=128, K=4] -> out: [P, F*K] = [32, 512]
        _run(
            test_manager,
            platform_target,
            fold_mod.fold_free_dim,
            fold_refs.fold_free_dim_torch_ref,
            lambda _: {"src_hbm": np.random.RandomState(42).randn(32, 128, 4).astype(np.float32)},
            lambda _: {"out": np.zeros((32, 512), dtype=np.float32)},
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "kernel,ref",
        [
            (fold_mod.fold_partition_roundtrip, fold_refs.fold_partition_roundtrip_torch_ref),
            (fold_mod.fold_free_dim_roundtrip, fold_refs.fold_free_dim_roundtrip_torch_ref),
        ],
    )
    def test_fold_roundtrip(self, test_manager, platform_target, kernel, ref):
        # src: [P=32, F=128, K=4]; out shape == in shape
        _run(
            test_manager,
            platform_target,
            kernel,
            ref,
            lambda _: {"src_hbm": np.random.RandomState(42).randn(32, 128, 4).astype(np.float32)},
            lambda ki: {"out": np.zeros_like(ki["src_hbm"])},
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.fast
    def test_fold_chain_4d(self, test_manager, platform_target):
        # src: [4, 8, 64, 8] -> out: [32, 512]
        _run(
            test_manager,
            platform_target,
            fold_mod.fold_chain_4d,
            fold_refs.fold_chain_4d_torch_ref,
            lambda _: {"src_hbm": np.random.RandomState(42).randn(4, 8, 64, 8).astype(np.float32)},
            lambda _: {"out": np.zeros((32, 512), dtype=np.float32)},
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.fast
    def test_pattern_override_load(self, test_manager, platform_target):
        # src: [P=32, F_src=1024] -> out: [P, F_dst=512] strided.
        _run(
            test_manager,
            platform_target,
            fold_mod.pattern_override_load,
            fold_refs.pattern_override_load_torch_ref,
            lambda _: {"src_hbm": np.random.RandomState(42).randn(32, 1024).astype(np.float32)},
            lambda _: {"out": np.zeros((32, 512), dtype=np.float32)},
            rtol=1e-5,
            atol=1e-5,
        )


@pytest_marks(["neurotile"])
class TestNeurotileTensorView:
    """Tutorials in 03_tensor_view.py."""

    @pytest.mark.fast
    def test_tensor_view_chain(self, test_manager, platform_target):
        # src: [B=1, S=4, H=256], out: [H0=128, B*S*H1=8]
        _run(
            test_manager,
            platform_target,
            view_mod.tensor_view_chain,
            view_refs.tensor_view_chain_torch_ref,
            lambda _: {"src": np.random.RandomState(42).randn(1, 4, 256).astype(ml_dtypes.bfloat16)},
            lambda _: {"out": np.zeros((128, 8), dtype=ml_dtypes.bfloat16)},
        )

    @pytest.mark.fast
    def test_tensor_view_select(self, test_manager, platform_target):
        # src: [4, 128, 16] bf16 -> out: src[0] * 3
        _run(
            test_manager,
            platform_target,
            view_mod.tensor_view_select,
            view_refs.tensor_view_select_torch_ref,
            lambda _: {"src": np.random.RandomState(42).randn(4, 128, 16).astype(ml_dtypes.bfloat16)},
            lambda _: {"out": np.zeros((128, 16), dtype=ml_dtypes.bfloat16)},
        )

    @pytest.mark.fast
    def test_tensor_view_3d_direct(self, test_manager, platform_target):
        # src: [128, 4, 32] fp32 -> out: same shape
        _run(
            test_manager,
            platform_target,
            view_mod.tensor_view_3d_direct,
            view_refs.tensor_view_3d_direct_torch_ref,
            lambda _: {"src": np.random.RandomState(42).randn(128, 4, 32).astype(np.float32)},
            lambda ki: {"out": np.zeros_like(ki["src"])},
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.fast
    def test_nd_tile_alloc_sbuf(self, test_manager, platform_target):
        # src: [128, 4, 32] fp32 -> out: [128, 128]
        _run(
            test_manager,
            platform_target,
            view_mod.nd_tile_alloc_sbuf,
            view_refs.nd_tile_alloc_sbuf_torch_ref,
            lambda _: {"src": np.random.RandomState(42).randn(128, 4, 32).astype(np.float32)},
            lambda _: {"out": np.zeros((128, 128), dtype=np.float32)},
            rtol=1e-5,
            atol=1e-5,
        )


def _xpose_in(p, f, seed=42):
    def _gen(_):
        np.random.seed(seed)
        return {"src": np.random.randn(p, f).astype(ml_dtypes.bfloat16)}

    return _gen


def _xpose_out(p, f):
    return lambda _: {"out": np.zeros((f, p), dtype=ml_dtypes.bfloat16)}


def _gather_in(*dims, seed=42):
    def _gen(_):
        np.random.seed(seed)
        rows = dims[0]
        data = np.random.randn(*dims).astype(ml_dtypes.bfloat16)
        indices = np.random.permutation(rows).astype(np.uint32).reshape(rows, 1)
        return {"data": data, "indices": indices}

    return _gen


@pytest_marks(["neurotile"])
class TestNeurotileTranspose:
    """Tutorials in _04_transpose.py -- every way to express a DMA transpose."""

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "m,n",
        [
            (128, 64),  # direct: a single tile, F<=128
            (256, 512),  # multi-tile: 2x4 tile grid
            (128, 400),  # remainder: trailing column tile is narrower (400=3*128+16)
        ],
    )
    def test_transpose_tiled(self, test_manager: Orchestrator, platform_target: Platforms, m, n):
        _run(
            test_manager,
            platform_target,
            transpose_mod.transpose_tiled,
            transpose_refs.transpose_tiled_torch_ref,
            _xpose_in(m, n),
            _xpose_out(m, n),
            logical_nc_config=1,
        )

    @pytest.mark.fast
    def test_transpose_coalesced(self, test_manager, platform_target):
        # One transpose DMA per (128, N) row-block, then carve per-tile stores.
        _run(
            test_manager,
            platform_target,
            transpose_mod.transpose_coalesced,
            transpose_refs.transpose_coalesced_torch_ref,
            _xpose_in(256, 512),
            _xpose_out(256, 512),
            logical_nc_config=1,
        )

    @pytest.mark.fast
    def test_transpose_coalesced_single_store_example(self, test_manager, platform_target):
        # One transpose DMA per tile-row AND one store DMA per row via rearrange scatter.
        _run(
            test_manager,
            platform_target,
            transpose_mod.transpose_coalesced_single_store,
            transpose_refs.transpose_coalesced_single_store_torch_ref,
            _xpose_in(256, 512),
            _xpose_out(256, 512),
            logical_nc_config=1,
        )

    @pytest.mark.fast
    def test_transpose_into_dst(self, test_manager, platform_target):
        _run(
            test_manager,
            platform_target,
            transpose_mod.transpose_into_dst,
            transpose_refs.transpose_into_dst_torch_ref,
            _xpose_in(256, 512),
            _xpose_out(256, 512),
            logical_nc_config=1,
        )

    @pytest.mark.fast
    def test_transpose_into_dst_partial(self, test_manager, platform_target):
        # dst= sized to a non-128-square transposed shape (gaps doc 4.6#5).
        _run(
            test_manager,
            platform_target,
            transpose_mod.transpose_into_dst_partial,
            transpose_refs.transpose_into_dst_partial_torch_ref,
            _xpose_in(128, 100),
            _xpose_out(128, 100),
            logical_nc_config=1,
        )

    def test_transpose_streamed(self, test_manager, platform_target):
        _run(
            test_manager,
            platform_target,
            transpose_mod.transpose_streamed,
            transpose_refs.transpose_streamed_torch_ref,
            _xpose_in(256, 512, seed=7),
            _xpose_out(256, 512),
            logical_nc_config=1,
        )

    def test_transpose_block_streamed(self, test_manager, platform_target):
        # Contiguous block-stream: 6 seq-tiles, blocks of 2, one transpose DMA per block.
        _run(
            test_manager,
            platform_target,
            transpose_mod.transpose_block_streamed,
            transpose_refs.transpose_block_streamed_torch_ref,
            _xpose_in(128 * 6, 96, seed=7),
            _xpose_out(128 * 6, 96),
            logical_nc_config=1,
        )

    def test_transpose_block_streamed_sharded(self, test_manager, platform_target):
        # Tile-sharded 2-D grid (8x4 tiles), shard 0 of 2 owns even M-tiles;
        # iterate block-cols, stream block-rows, per-owned-tile transpose.
        _run(
            test_manager,
            platform_target,
            transpose_mod.transpose_block_streamed_sharded,
            transpose_refs.transpose_block_streamed_sharded_torch_ref,
            _xpose_in(128 * 8, 128 * 4, seed=7),
            lambda _: {"out": np.zeros((128 * 4, 4 * 128), dtype=ml_dtypes.bfloat16)},
            logical_nc_config=1,
        )

    def test_gather_transpose(self, test_manager, platform_target):
        n, d = 16, 64
        _run(
            test_manager,
            platform_target,
            transpose_mod.gather_transpose,
            transpose_refs.gather_transpose_torch_ref,
            _gather_in(n, d),
            lambda _: {"out": np.zeros((d, n), dtype=ml_dtypes.bfloat16)},
            logical_nc_config=1,
        )

    # Two shapes each to de-risk N-D AP orientation (gaps doc App.D: hardware-only
    # verifiable). rows % 16 == 0, rows in [16, 128], free dim <= 128.
    @pytest.mark.parametrize("rows,n_tiles,tile", [(16, 4, 64), (32, 2, 128)])
    def test_gather_transpose_3d(self, test_manager, platform_target, rows, n_tiles, tile):
        _run(
            test_manager,
            platform_target,
            transpose_mod.gather_transpose_3d,
            transpose_refs.gather_transpose_3d_torch_ref,
            _gather_in(rows, n_tiles, tile),
            lambda _: {"out": np.zeros((tile, n_tiles, rows), dtype=ml_dtypes.bfloat16)},
            logical_nc_config=1,
        )

    @pytest.mark.parametrize("rows,f_tiles,p", [(16, 8, 128), (32, 3, 128)])
    def test_gather_transpose_4d(self, test_manager, platform_target, rows, f_tiles, p):
        _run(
            test_manager,
            platform_target,
            transpose_mod.gather_transpose_4d,
            transpose_refs.gather_transpose_4d_torch_ref,
            _gather_in(rows, f_tiles, p),
            lambda _: {"out": np.zeros((p, 1, f_tiles, rows), dtype=ml_dtypes.bfloat16)},
            logical_nc_config=1,
        )

    @pytest.mark.fast
    def test_transpose_multi_dma_rejected(self):
        """A transposed free dim >128 and not a multiple of 128 would need 2 DMAs;
        the single-DMA contract must reject it at trace time and point at the split."""
        import nki
        import nki.language as nl
        from nkilib_src.nkilib.experimental import neurotile as nt

        @nki.jit
        def bad(src):  # src [128, 200] -> transposed F=200 (>128, not mult of 128)
            t = nt.tiles(src, tile_size=(128, 200))
            xposed = t[0, 0].load(transpose=True)
            out = nl.ndarray((200, 128), dtype=src.dtype, buffer=nl.shared_hbm)
            nt.tiles(out, tile_size=(200, 128))[0, 0].store(xposed.data)
            return out

        src = np.random.RandomState(42).randn(128, 200).astype(ml_dtypes.bfloat16)
        with pytest.raises(AssertionError, match="single-DMA contract"):
            nki.simulate(bad)(src)
