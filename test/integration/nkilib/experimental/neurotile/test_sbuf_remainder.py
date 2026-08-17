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

"""NKI integration tests for SBUF remainder tile handling.

Tests alloc_tiles with element_shape, SBUF tile indexing/data shapes,
retile of remainder tiles, and end-to-end transpose with remainder.

Note: NeuroTile's mlp_cte_remainder_* tests imported from a fgcc_mlp
example kernel that is not part of this migration. Those tests are
deliberately omitted from this port.
"""

import ml_dtypes
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.experimental import neurotile as nt

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


@nki.jit
def _alloc_tiles_remainder_kernel(src):
    """Allocate tiled SBUF with remainder, verify shapes via trace-time asserts."""
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    gated = nt.alloc_tiles(
        tile_size=(128, 512),
        buffer_type=nl.sbuf,
        dtype=nl.bfloat16,
        element_shape=(128, 1792),
    )
    assert gated.shape == (1, 4), f"Expected (1,4), got {gated.shape}"
    assert gated.data.shape == (128, 1792), f"Expected (128,1792), got {gated.data.shape}"

    for i in range(3):
        t = gated[0, i]
        assert t.element_shape == (128, 512), f"Tile [0,{i}] expected (128,512)"
        assert t.data.shape == (128, 512), f"Tile [0,{i}] data expected (128,512)"
        assert not t.is_remainder

    rem = gated[0, 3]
    assert rem.element_shape == (128, 256), "Remainder expected (128,256)"
    assert rem.data.shape == (128, 256), "Remainder data expected (128,256)"
    assert rem.is_remainder

    nisa.dma_copy(dst, src)
    return dst


@nki.jit
def _alloc_tiles_multirow_kernel(src):
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    gated = nt.alloc_tiles(
        tile_size=(128, 512),
        buffer_type=nl.sbuf,
        dtype=nl.bfloat16,
        element_shape=(256, 1792),
    )
    assert gated.shape == (2, 4)

    for row in range(2):
        assert gated[row, 0].data.shape == (128, 512)
        assert gated[row, 3].data.shape == (128, 256)
        assert gated[row, 3].is_remainder

    nisa.dma_copy(dst, src)
    return dst


@nki.jit
def _alloc_tiles_exact_kernel(src):
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    gated = nt.alloc_tiles(
        tile_size=(128, 512),
        buffer_type=nl.sbuf,
        dtype=nl.bfloat16,
        element_shape=(128, 2048),
    )
    assert gated.shape == (1, 4)
    assert gated.data.shape == (128, 2048)
    assert not gated.is_remainder

    for i in range(4):
        assert gated[0, i].data.shape == (128, 512)
        assert not gated[0, i].is_remainder

    nisa.dma_copy(dst, src)
    return dst


@nki.jit
def _alloc_partial_partition_kernel(src):
    """Regression: element_shape whose PARTITION extent (44) is smaller than
    tile_size[0] (128). p_tile_count floor-divided 44 // 128 -> 0, which
    produced an invalid (partition_stride, 0) AP level and crashed at index
    time; ceiling_div now yields 1 tile. alloc_tiles and alloc_blocks both.
    """
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    a = nt.alloc_tiles(
        tile_size=(128, 128),
        buffer_type=nl.sbuf,
        dtype=nl.bfloat16,
        element_shape=(44, 256),
    )
    assert a.shape == (1, 2), f"Expected (1,2), got {a.shape}"
    # indexing the tiles is what crashed pre-fix
    nisa.memset(a[0, 0].data, 0.0)
    nisa.memset(a[0, 1].data, 0.0)

    b = nt.alloc_blocks(
        tile_size=(128, 128),
        block_size=(1, 2),
        buffer_type=nl.sbuf,
        dtype=nl.bfloat16,
        element_shape=(44, 256),
    )
    blk = b[0, 0]
    nisa.memset(blk[0, 0].data, 0.0)
    nisa.memset(blk[0, 1].data, 0.0)

    nisa.dma_copy(dst, src)
    return dst


@nki.jit
def _retile_remainder_kernel(src):
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    gated = nt.alloc_tiles(
        tile_size=(128, 512),
        buffer_type=nl.sbuf,
        dtype=nl.bfloat16,
        element_shape=(128, 1792),
    )

    sub_full = nt.tiles(gated[0, 0], tile_size=(128, 128))
    assert sub_full.shape == (1, 4)
    for j in range(4):
        assert sub_full[0, j].data.shape == (128, 128)

    sub_rem = nt.tiles(gated[0, 3], tile_size=(128, 128))
    assert sub_rem.shape == (1, 2)
    for j in range(2):
        assert sub_rem[0, j].data.shape == (128, 128)

    nisa.dma_copy(dst, src)
    return dst


@nki.jit
def _transpose_remainder_kernel(src):
    M, I = src.shape
    dst = nl.ndarray((M, I), dtype=src.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src, tile_size=(128, 512))

    gated = nt.alloc_tiles(
        tile_size=(128, 512),
        buffer_type=nl.sbuf,
        dtype=src.dtype,
        element_shape=(128, I),
    )

    for i in range(src_tiles.shape[1]):
        t = src_tiles[0, i].load()
        actual_f = t.data.shape[1]
        nisa.tensor_copy(gated[0, i].data[:, :actual_f], t.data)

    for i in range(gated.shape[1]):
        block = gated[0, i]
        subtiles = nt.tiles(block, tile_size=(128, 128))
        for j in range(subtiles.shape[1]):
            st = subtiles[0, j]
            psum_tmp = nl.ndarray((128, 128), dtype=src.dtype, buffer=nl.psum)
            nisa.nc_transpose(psum_tmp, st.data)
            nisa.tensor_copy(st.data, psum_tmp)

    for i in range(gated.shape[1]):
        actual_f = gated[0, i].data.shape[1]
        nisa.dma_copy(dst[:128, nl.ds(i * 512, actual_f)], gated[0, i].data)

    return dst


def _small_src_inputs(_):
    np.random.seed(42)
    return {"src": np.random.randn(128, 128).astype(ml_dtypes.bfloat16)}


def _small_src_output(kernel_input):
    return {"out": np.zeros_like(kernel_input["src"])}


def _passthrough_ref(src: torch.Tensor) -> torch.Tensor:
    return src.clone()


def _transpose_inputs(_):
    np.random.seed(42)
    return {"src": np.random.randn(128, 1792).astype(ml_dtypes.bfloat16)}


def _transpose_output(kernel_input):
    return {"out": np.zeros_like(kernel_input["src"])}


def _transpose_ref(src: torch.Tensor) -> torch.Tensor:
    """Per-tile in-place transpose: each (128, 128) sub-tile gets transposed.

    The kernel transposes within each tile_size=(128, 128) sub-tile. For
    a (128, 1792) source: 14 sub-tiles per row, each transposed.
    """
    M, I = src.shape
    out = src.clone()
    sub_w = 128
    for i in range(I // sub_w):
        s = i * sub_w
        e = s + sub_w
        out[:, s:e] = src[:, s:e].t()
    # Remainder columns (1792 % 128 = 0 here, so no extras)
    return out


@pytest_marks(["neurotile"])
class TestSBUFRemainderShapes:
    """alloc_tiles with element_shape: shape and data shape pinning."""

    @pytest.mark.fast
    def test_alloc_tiles_remainder_shapes(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_alloc_tiles_remainder_kernel,
            torch_ref=torch_ref_wrapper(_passthrough_ref),
            kernel_input_generator=_small_src_inputs,
            output_tensor_descriptor=_small_src_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_alloc_tiles_multirow_remainder(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_alloc_tiles_multirow_kernel,
            torch_ref=torch_ref_wrapper(_passthrough_ref),
            kernel_input_generator=_small_src_inputs,
            output_tensor_descriptor=_small_src_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_alloc_tiles_exact_fit(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_alloc_tiles_exact_kernel,
            torch_ref=torch_ref_wrapper(_passthrough_ref),
            kernel_input_generator=_small_src_inputs,
            output_tensor_descriptor=_small_src_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.fast
    def test_alloc_partial_partition(self, test_manager: Orchestrator, platform_target: Platforms):
        """element_shape partition extent < tile_size[0] (44 < 128): regression for
        the p_tile_count floor-div-to-zero (128, 0) crash on alloc_tiles/alloc_blocks."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_alloc_partial_partition_kernel,
            torch_ref=torch_ref_wrapper(_passthrough_ref),
            kernel_input_generator=_small_src_inputs,
            output_tensor_descriptor=_small_src_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_retile_remainder(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_retile_remainder_kernel,
            torch_ref=torch_ref_wrapper(_passthrough_ref),
            kernel_input_generator=_small_src_inputs,
            output_tensor_descriptor=_small_src_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )


@pytest_marks(["neurotile"])
class TestSBUFRemainderTranspose:
    """End-to-end transpose with remainder-allocated SBUF."""

    @pytest.mark.fast
    def test_transpose_remainder(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_transpose_remainder_kernel,
            torch_ref=torch_ref_wrapper(_transpose_ref),
            kernel_input_generator=_transpose_inputs,
            output_tensor_descriptor=_transpose_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )


# ============================================================================
# DMA transpose-load (.load(transpose=True)) on real hardware.
#
# The HBM->SBUF DMA-transpose path had no direct integration coverage; these
# kernels exercise it end-to-end and validate against src.T. They are the
# numeric oracle for:
#   - the F<=128 direct path and the F%128==0 tiled path (baseline / regression)
#   - the F%128!=0 tiled path (Gap 4) -- currently a hard assert, fails today
# ============================================================================


def _dma_xpose_ref(src: torch.Tensor) -> torch.Tensor:
    """Full 2-D transpose: out[f, p] = src[p, f]."""
    return src.t().contiguous()


# The transpose-load tiles F into chunks of <=128. For chunk c the buffer holds
# src[:, c*128:(c+1)*128].T  (shape (chunk_w, p_dim)), which is exactly row-block
# [c*128 : c*128+chunk_w] of the true transpose src.T. So we recombine the chunk
# blocks into a (f_dim, p_dim) output and validate against src.t(). Shapes are
# derived from src.shape inside the kernel (NKI tracer does not close over free
# Python variables).


@nki.jit
def _xpose_kernel_direct(src):
    """F<=128 direct path: buffer is already (f_dim, p_dim) == src.T."""
    p_dim, f_dim = src.shape
    out = nl.ndarray((f_dim, p_dim), dtype=src.dtype, buffer=nl.shared_hbm)
    xposed = nt.tiles(src, tile_size=src.shape)[0, 0].load(transpose=True)
    nisa.dma_copy(out, xposed.data)
    return out


@nki.jit
def _xpose_kernel_tiled(src):
    """F%128==0 tiled path: buffer is (128, p_dim*num_chunks); recombine the
    per-chunk transposed blocks into the true (f_dim, p_dim) transpose."""
    p_dim, f_dim = src.shape
    out = nl.ndarray((f_dim, p_dim), dtype=src.dtype, buffer=nl.shared_hbm)
    xposed = nt.tiles(src, tile_size=src.shape)[0, 0].load(transpose=True)
    num_chunks = f_dim // 128
    for c in range(num_chunks):
        # buffer block c == src[:, c*128:(c+1)*128].T == rows [c*128:(c+1)*128] of src.T
        nisa.dma_copy(out[nl.ds(c * 128, 128), :], xposed.data[:, nl.ds(c * p_dim, p_dim)])
    return out


@nki.jit
def _xpose_kernel_subtile_dst(src):
    """dst= is a sub-tile of a WIDER block buffer (physical width > logical tile).

    Transpose-loads each of 2 column-tiles of src (128, 256) into its own tile
    slot of a (128, 2*128) block buffer via dst=slot[ti].data. Exercises the
    _dma_transpose dst partition-stride fix: the emitted AP's level-0 stride must
    follow the slot's physical 256 width, not the logical 128.
    """
    p_dim, f_dim = src.shape  # (128, 256)
    n_tiles = f_dim // 128
    out = nl.ndarray((p_dim, f_dim), dtype=src.dtype, buffer=nl.shared_hbm)
    blk = nt.alloc_blocks(
        tile_size=(p_dim, 128),
        block_size=(n_tiles, 1),
        grid=(1, 1),
        buffer_type=nl.sbuf,
        dtype=src.dtype,
    )
    slot_tiles = blk[0, 0]  # (n_tiles, 1) interior grid of the wide slot
    src_tiles = nt.tiles(src, tile_size=(p_dim, 128))  # grid (1, n_tiles): column-tiles on dim 1
    for ti in range(n_tiles):
        src_tiles[0, ti].load(transpose=True, dst=slot_tiles[ti, 0].data)
    nisa.dma_copy(out, blk[0, 0].data)
    return out


def _subtile_dst_ref(src):
    # Each column-tile transposed, packed side-by-side: [t0.T | t1.T].
    import torch

    n_tiles = src.shape[1] // 128
    return torch.cat([src[:, c * 128 : (c + 1) * 128].t().contiguous() for c in range(n_tiles)], dim=1)


@nki.jit
def _xpose_kernel_remainder_f(src):
    """Gap 4: F%128!=0 (128,400). A transposed free dim >128 and not a multiple
    of 128 cannot be one coalesced DMA (the single-DMA contract), so tile F into
    128-wide column tiles plus a trailing remainder tile and transpose-load each
    as its own DMA. Each column tile c == src[:, c].T == a row band of src.T."""
    p_dim, f_dim = src.shape
    out = nl.ndarray((f_dim, p_dim), dtype=src.dtype, buffer=nl.shared_hbm)
    col_tiles = nt.tiles(src, tile_size=(p_dim, 128))  # grid (1, ceil(f_dim/128))
    num_full = f_dim // 128
    rem = f_dim % 128
    for c in range(num_full):
        xposed = col_tiles[0, c].load(transpose=True)  # (128, p_dim)
        nisa.dma_copy(out[nl.ds(c * 128, 128), :], xposed.data)
    if rem > 0:
        xposed = col_tiles[0, num_full].load(transpose=True)  # (rem, p_dim)
        nisa.dma_copy(out[nl.ds(num_full * 128, rem), :], xposed.data)
    return out


def _xpose_inputs_for(p_dim, f_dim):
    def _gen(_):
        np.random.seed(42)
        return {"src": np.random.randn(p_dim, f_dim).astype(ml_dtypes.bfloat16)}

    return _gen


def _xpose_output_for(p_dim, f_dim):
    def _out(kernel_input):
        return {"out": np.zeros((f_dim, p_dim), dtype=ml_dtypes.bfloat16)}

    return _out


@pytest_marks(["neurotile"])
class TestDmaTransposeLoad:
    """Numeric correctness of view.load(transpose=True) on hardware."""

    @pytest.mark.fast
    def test_transpose_load_direct(self, test_manager: Orchestrator, platform_target: Platforms):
        """F<=128 direct path: (128,128) transpose-load == src.T."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_xpose_kernel_direct,
            torch_ref=torch_ref_wrapper(_dma_xpose_ref),
            kernel_input_generator=_xpose_inputs_for(128, 128),
            output_tensor_descriptor=_xpose_output_for(128, 128),
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.fast
    def test_transpose_load_tiled(self, test_manager: Orchestrator, platform_target: Platforms):
        """F%128==0 tiled path: (128,512) transpose-load recombined == src.T."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_xpose_kernel_tiled,
            torch_ref=torch_ref_wrapper(_dma_xpose_ref),
            kernel_input_generator=_xpose_inputs_for(128, 512),
            output_tensor_descriptor=_xpose_output_for(128, 512),
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_transpose_load_remainder_f(self, test_manager: Orchestrator, platform_target: Platforms):
        """Gap 4: F%128!=0 (128,400) transpose-load, split into per-tile DMAs
        (128-wide column tiles + remainder), recombined == src.T."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_xpose_kernel_remainder_f,
            torch_ref=torch_ref_wrapper(_dma_xpose_ref),
            kernel_input_generator=_xpose_inputs_for(128, 400),
            output_tensor_descriptor=_xpose_output_for(128, 400),
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.fast
    def test_transpose_load_subtile_dst(self, test_manager: Orchestrator, platform_target: Platforms):
        """dst= a sub-tile of a wider block buffer: transpose-load each column-tile
        of (128,256) into its slot. Exercises the _dma_transpose dst partition-
        stride fix (level-0 must follow the slot's physical 256 width)."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_xpose_kernel_subtile_dst,
            torch_ref=torch_ref_wrapper(_subtile_dst_ref),
            kernel_input_generator=_xpose_inputs_for(128, 256),
            output_tensor_descriptor=lambda _: {"out": np.zeros((128, 256), dtype=ml_dtypes.bfloat16)},
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_transpose_load_f_under_128(self, test_manager: Orchestrator, platform_target: Platforms):
        """Gap 4 doc §4.6#3: F<128 single-sub-chunk path (128,100) -> (100,128).
        Buffer is (f_dim, p_dim); regression guard for the direct (single-chunk)
        branch of the unified chunker."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_xpose_kernel_direct,
            torch_ref=torch_ref_wrapper(_dma_xpose_ref),
            kernel_input_generator=_xpose_inputs_for(128, 100),
            output_tensor_descriptor=_xpose_output_for(128, 100),
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )


# ============================================================================
# Gap 1 on hardware: DMA-control params (dge_mode / priority) actually reach
# nisa.dma_transpose and produce correct results. Unit tests verify the
# validation gate in Python; these verify on-device that a passed param does
# not get silently dropped and the transpose is still numerically correct.
# ============================================================================


@nki.jit
def _xpose_kernel_dge_unknown(src):
    """Tiled transpose with explicit dge_mode=unknown (the default, but passed
    explicitly to prove the param threads through to the DMA)."""
    p_dim, f_dim = src.shape
    out = nl.ndarray((f_dim, p_dim), dtype=src.dtype, buffer=nl.shared_hbm)
    xposed = nt.tiles(src, tile_size=src.shape)[0, 0].load(transpose=True, dge_mode=nisa.dge_mode.unknown)
    num_chunks = f_dim // 128
    for c in range(num_chunks):
        nisa.dma_copy(out[nl.ds(c * 128, 128), :], xposed.data[:, nl.ds(c * p_dim, p_dim)])
    return out


@nki.jit
def _xpose_kernel_priority(src):
    """Tiled transpose with priority=0 -- a QoS hint; must not change results."""
    p_dim, f_dim = src.shape
    out = nl.ndarray((f_dim, p_dim), dtype=src.dtype, buffer=nl.shared_hbm)
    xposed = nt.tiles(src, tile_size=src.shape)[0, 0].load(transpose=True, priority=0)
    num_chunks = f_dim // 128
    for c in range(num_chunks):
        nisa.dma_copy(out[nl.ds(c * 128, 128), :], xposed.data[:, nl.ds(c * p_dim, p_dim)])
    return out


@nki.jit
def _xpose_kernel_remainder_f_dge(src):
    """Gap 1 + Gap 4 combined: F%128!=0 (128,400) transpose with dge_mode set.
    Proves DMA-control threads through each per-tile transpose DMA when F is
    split into 128-wide column tiles plus a trailing remainder tile."""
    p_dim, f_dim = src.shape
    out = nl.ndarray((f_dim, p_dim), dtype=src.dtype, buffer=nl.shared_hbm)
    col_tiles = nt.tiles(src, tile_size=(p_dim, 128))
    num_full = f_dim // 128
    rem = f_dim % 128
    for c in range(num_full):
        xposed = col_tiles[0, c].load(transpose=True, dge_mode=nisa.dge_mode.unknown)
        nisa.dma_copy(out[nl.ds(c * 128, 128), :], xposed.data)
    if rem > 0:
        xposed = col_tiles[0, num_full].load(transpose=True, dge_mode=nisa.dge_mode.unknown)
        nisa.dma_copy(out[nl.ds(num_full * 128, rem), :], xposed.data)
    return out


@nki.jit
def _xpose_kernel_hwdge(src):
    """hwdge transpose with the required shape: src P==16, F%128==0, 2-byte dtype.
    Source (16, 128) -> single 128-chunk -> buffer (128, 16) == src.T."""
    p_dim, f_dim = src.shape
    out = nl.ndarray((f_dim, p_dim), dtype=src.dtype, buffer=nl.shared_hbm)
    xposed = nt.tiles(src, tile_size=src.shape)[0, 0].load(transpose=True, dge_mode=nisa.dge_mode.hwdge)
    nisa.dma_copy(out, xposed.data)
    return out


@pytest_marks(["neurotile"])
class TestDmaTransposeLoadDmaControl:
    """Gap 1: DMA-control params reach nisa.dma_transpose on-device."""

    def test_transpose_load_dge_unknown(self, test_manager: Orchestrator, platform_target: Platforms):
        """dge_mode=unknown threads through; (128,512) transpose == src.T."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_xpose_kernel_dge_unknown,
            torch_ref=torch_ref_wrapper(_dma_xpose_ref),
            kernel_input_generator=_xpose_inputs_for(128, 512),
            output_tensor_descriptor=_xpose_output_for(128, 512),
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    def test_transpose_load_priority(self, test_manager: Orchestrator, platform_target: Platforms):
        """priority=0 accepted on-device; (128,512) transpose == src.T.
        dma_transpose priority is NeuronCore-v4+ only, so trn1/trn2 are excluded
        (run on trn3 via --shared-fleet)."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_xpose_kernel_priority,
            torch_ref=torch_ref_wrapper(_dma_xpose_ref),
            kernel_input_generator=_xpose_inputs_for(128, 512),
            output_tensor_descriptor=_xpose_output_for(128, 512),
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_transpose_load_hwdge(self, test_manager: Orchestrator, platform_target: Platforms):
        """dge_mode=hwdge with constraint-satisfying shape (16,128) -> (128,16)
        == src.T. Proves the hwdge path lowers and is numerically correct."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_xpose_kernel_hwdge,
            torch_ref=torch_ref_wrapper(_dma_xpose_ref),
            kernel_input_generator=_xpose_inputs_for(16, 128),
            output_tensor_descriptor=_xpose_output_for(16, 128),
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_transpose_load_remainder_dge(self, test_manager: Orchestrator, platform_target: Platforms):
        """Gap 1 + Gap 4 doc §4.6#4: F%128!=0 (128,400) with dge_mode set --
        DMA-control honored across both the full-chunk and partial-chunk DMAs."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_xpose_kernel_remainder_f_dge,
            torch_ref=torch_ref_wrapper(_dma_xpose_ref),
            kernel_input_generator=_xpose_inputs_for(128, 400),
            output_tensor_descriptor=_xpose_output_for(128, 400),
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )


# ============================================================================
# Indirect (gather) transpose via neurotile data_view[idx_tile].load(transpose=True).
#   flat = indices.T.flatten()[:N]; dst = src[flat, :].T   -> (D, N)
# src (N=16, D=64) HBM, N%16==0, D<=128, 2-byte; indices (N,1) uint32 SBUF.
#
# Note: oob_mode.skip on this op is all-or-nothing on trn2 (verified by device
# dump: any out-of-bounds index skips the WHOLE gather, not per-row), so a
# per-row oob_value fill is not a supported semantic and is not tested here.
# ============================================================================
_IXPOSE_N = 16
_IXPOSE_D = 64


@nki.jit
def _xpose_kernel_indirect(data, indices):
    """Indirect (gather) transpose: gather N rows of data via `indices`,
    transpose to (D, N)."""
    N, D = data.shape
    out = nl.ndarray((D, N), dtype=data.dtype, buffer=nl.shared_hbm)

    data_iter = nt.tiles(data, tile_size=(N, D))
    idx_tile = nt.tiles(indices, tile_size=(N, 1))[0, 0].load()
    xposed = data_iter[idx_tile, 0].load(transpose=True)
    nisa.dma_copy(out, xposed.data)
    return out


def _indirect_inputs(_):
    np.random.seed(42)
    data = np.random.randn(_IXPOSE_N, _IXPOSE_D).astype(ml_dtypes.bfloat16)
    idx = np.random.permutation(_IXPOSE_N).astype(np.uint32).reshape(_IXPOSE_N, 1)
    return {"data": data, "indices": idx}


def _indirect_output(kernel_input):
    return {"out": np.zeros((_IXPOSE_D, _IXPOSE_N), dtype=ml_dtypes.bfloat16)}


def _indirect_ref(data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    flat = indices.t().flatten()[:_IXPOSE_N]
    return data[flat.long(), :].t().contiguous()


# ============================================================================
# N-D gather-transpose: transpose_axes= selects the reshape-trick rank.
#   3-D src (rows, n_tiles, tile), axes=(2,1,0) -> (tile, n_tiles, rows)
#   4-D src (rows, f_tiles, P),    axes=(3,1,2,0) -> (P, 1, f_tiles, rows)
# Mirrors production sites bwmm_bwd_dropless (3-D) / moe_cte_mx_utils (4-D).
# rows gathered via the vector index; the 4-D size-1 dummy dim is synthesized
# by the AP builder, so the view itself stays 3-D. rows%16==0, 2-byte dtype.
# ============================================================================
_GX3_ROWS, _GX3_NT, _GX3_TILE = 16, 4, 64
_GX4_ROWS, _GX4_FT, _GX4_P = 16, 8, 128


@nki.jit
def _xpose_kernel_indirect_3d(data, indices):
    """3-D gather-transpose: gather rows, axes=(2,1,0) -> (tile, n_tiles, rows)."""
    rows, n_tiles, tile = data.shape
    out = nl.ndarray((tile, n_tiles, rows), dtype=data.dtype, buffer=nl.shared_hbm)

    data_iter = nt.tiles(data, tile_size=(rows, n_tiles, tile))
    idx_tile = nt.tiles(indices, tile_size=(rows, 1))[0, 0].load()
    xposed = data_iter[idx_tile, 0, 0].load(transpose=True, transpose_axes=(2, 1, 0))
    nisa.dma_copy(out, xposed.data)
    return out


@nki.jit
def _xpose_kernel_indirect_4d(data, indices):
    """4-D gather-transpose: gather rows, axes=(3,1,2,0) -> (P, 1, f_tiles, rows)."""
    rows, f_tiles, p = data.shape
    out = nl.ndarray((p, 1, f_tiles, rows), dtype=data.dtype, buffer=nl.shared_hbm)

    data_iter = nt.tiles(data, tile_size=(rows, f_tiles, p))
    idx_tile = nt.tiles(indices, tile_size=(rows, 1))[0, 0].load()
    xposed = data_iter[idx_tile, 0, 0].load(transpose=True, transpose_axes=(3, 1, 2, 0))
    nisa.dma_copy(out, xposed.data)
    return out


def _indirect_3d_inputs(_):
    np.random.seed(42)
    data = np.random.randn(_GX3_ROWS, _GX3_NT, _GX3_TILE).astype(ml_dtypes.bfloat16)
    idx = np.random.permutation(_GX3_ROWS).astype(np.uint32).reshape(_GX3_ROWS, 1)
    return {"data": data, "indices": idx}


def _indirect_3d_output(kernel_input):
    return {"out": np.zeros((_GX3_TILE, _GX3_NT, _GX3_ROWS), dtype=ml_dtypes.bfloat16)}


def _indirect_3d_ref(data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    flat = indices.t().flatten()[:_GX3_ROWS]
    return data[flat.long()].permute(2, 1, 0).contiguous()


def _indirect_4d_inputs(_):
    np.random.seed(42)
    data = np.random.randn(_GX4_ROWS, _GX4_FT, _GX4_P).astype(ml_dtypes.bfloat16)
    idx = np.random.permutation(_GX4_ROWS).astype(np.uint32).reshape(_GX4_ROWS, 1)
    return {"data": data, "indices": idx}


def _indirect_4d_output(kernel_input):
    return {"out": np.zeros((_GX4_P, 1, _GX4_FT, _GX4_ROWS), dtype=ml_dtypes.bfloat16)}


def _indirect_4d_ref(data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    flat = indices.t().flatten()[:_GX4_ROWS]
    gathered = data[flat.long()]
    return gathered.reshape(_GX4_ROWS, 1, _GX4_FT, _GX4_P).permute(3, 1, 2, 0).contiguous()


@pytest_marks(["neurotile"])
class TestDmaTransposeLoadIndirect:
    """Indirect (gather) transpose via neurotile .load(transpose=True)."""

    def test_indirect_transpose(self, test_manager: Orchestrator, platform_target: Platforms):
        """Gather-transpose == src[idx].T."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_xpose_kernel_indirect,
            torch_ref=torch_ref_wrapper(_indirect_ref),
            kernel_input_generator=_indirect_inputs,
            output_tensor_descriptor=_indirect_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_indirect_transpose_3d(self, test_manager: Orchestrator, platform_target: Platforms):
        """3-D gather-transpose (transpose_axes=(2,1,0)) == src[idx].permute(2,1,0)."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_xpose_kernel_indirect_3d,
            torch_ref=torch_ref_wrapper(_indirect_3d_ref),
            kernel_input_generator=_indirect_3d_inputs,
            output_tensor_descriptor=_indirect_3d_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_indirect_transpose_4d(self, test_manager: Orchestrator, platform_target: Platforms):
        """4-D gather-transpose (transpose_axes=(3,1,2,0)) == src[idx] reshaped+permuted."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_xpose_kernel_indirect_4d,
            torch_ref=torch_ref_wrapper(_indirect_4d_ref),
            kernel_input_generator=_indirect_4d_inputs,
            output_tensor_descriptor=_indirect_4d_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )
