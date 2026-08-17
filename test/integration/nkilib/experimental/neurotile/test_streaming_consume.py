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

"""Behavioral tests for the post-stream-load view state.

`stream(dim=).load(k)` is a "consume one step on dim" operation. These
kernels assert the post-consume Grid/Layout state inside the kernel body
(via Python assert / NKI tracer), then dma_copy(out, src) so we can
validate end-to-end with a passthrough torch reference.
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

# Tile-grid streaming kernels: src=(512, 1024), tile_size=(128, 128) -> (4, 8) grid


@nki.jit
def _kernel_stream_tiles_default_dim0(src):
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.tiles(src, tile_size=(128, 128))
    s = v.stream(buffer_count=2)
    assert s.count == 4, str(s.count)
    slot = s.load(0)
    assert slot.element_shape == (128, 1024), str(slot.element_shape)
    assert slot._grid.cursor == 1, str(slot._grid.cursor)
    assert slot.shape == (8,), str(slot.shape)
    nisa.dma_copy(out, src)
    return out


@nki.jit
def _kernel_stream_tiles_explicit_dim1(src):
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.tiles(src, tile_size=(128, 128))
    s = v.stream(dim=1, buffer_count=2)
    assert s.count == 8, str(s.count)
    slot = s.load(0)
    assert slot.element_shape == (512, 128), str(slot.element_shape)
    assert slot._grid.cursor == 0, str(slot._grid.cursor)
    assert slot.shape == (4, 1), str(slot.shape)
    nisa.dma_copy(out, src)
    return out


@nki.jit
def _kernel_stream_tile_row_via_int(src):
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.tiles(src, tile_size=(128, 128))
    row = v[2]
    assert row._grid.cursor == 1, str(row._grid.cursor)
    s = row.stream(buffer_count=2)
    assert s.count == 8, str(s.count)
    slot = s.load(0)
    assert slot.element_shape == (128, 128), str(slot.element_shape)
    assert slot._grid.cursor == slot.ndim, str(slot._grid.cursor)
    nisa.dma_copy(out, src)
    return out


@nki.jit
def _kernel_stream_tile_col_via_int(src):
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.tiles(src, tile_size=(128, 128))
    col = v[:, 3]
    assert col._grid.cursor == 0, str(col._grid.cursor)
    s = col.stream(buffer_count=2)
    assert s.count == 4, str(s.count)
    nisa.dma_copy(out, src)
    return out


@nki.jit
def _kernel_stream_tile_subgrid(src):
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.tiles(src, tile_size=(128, 128))
    sub = v[1:3, :]
    s = sub.stream(buffer_count=2)
    assert s.count == 2, str(s.count)
    nisa.dma_copy(out, src)
    return out


# Block-grid streaming: src=(512, 1024), tile_size=(128, 256), block_size=(2, 2) -> (2, 2) blocks


@nki.jit
def _kernel_stream_blocks_default_dim0(src):
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.blocks(src, tile_size=(128, 256), block_size=(2, 2))
    s = v.stream(buffer_count=2)
    assert s.count == 2, str(s.count)
    nisa.dma_copy(out, src)
    return out


@nki.jit
def _kernel_stream_blocks_explicit_dim1(src):
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.blocks(src, tile_size=(128, 256), block_size=(2, 2))
    s = v.stream(dim=1, buffer_count=2)
    assert s.count == 2, str(s.count)
    nisa.dma_copy(out, src)
    return out


@nki.jit
def _kernel_stream_block_row_via_int(src):
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.blocks(src, tile_size=(128, 256), block_size=(2, 2))
    row = v[0]
    assert row._grid.cursor == 1, str(row._grid.cursor)
    s = row.stream(buffer_count=2)
    assert s.count == 2, str(s.count)
    nisa.dma_copy(out, src)
    return out


@nki.jit
def _kernel_stream_block_col_via_int(src):
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.blocks(src, tile_size=(128, 256), block_size=(2, 2))
    col = v[:, 1]
    assert col._grid.cursor == 0, str(col._grid.cursor)
    s = col.stream(buffer_count=2)
    assert s.count == 2, str(s.count)
    nisa.dma_copy(out, src)
    return out


# Common fixture: passthrough kernel takes src, returns src.clone() unmodified


def _src_inputs(_):
    np.random.seed(42)
    return {"src": np.random.randn(512, 1024).astype(ml_dtypes.bfloat16)}


def _src_output(kernel_input):
    return {"out": np.zeros_like(kernel_input["src"])}


def _passthrough_ref(src: torch.Tensor) -> torch.Tensor:
    return src.clone()


_KERNELS = [
    _kernel_stream_tiles_default_dim0,
    _kernel_stream_tiles_explicit_dim1,
    _kernel_stream_tile_row_via_int,
    _kernel_stream_tile_col_via_int,
    _kernel_stream_tile_subgrid,
    _kernel_stream_blocks_default_dim0,
    _kernel_stream_blocks_explicit_dim1,
    _kernel_stream_block_row_via_int,
    _kernel_stream_block_col_via_int,
]


@pytest_marks(["neurotile"])
class TestStreamingConsume:
    """Pin the post-stream view state across pre-narrow / stream-dim variants."""

    @pytest.mark.fast
    @pytest.mark.parametrize("kernel", _KERNELS, ids=lambda k: k.__name__.lstrip("_"))
    def test_stream_consume(self, test_manager: Orchestrator, platform_target: Platforms, kernel):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref_wrapper(_passthrough_ref),
            kernel_input_generator=_src_inputs,
            output_tensor_descriptor=_src_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )


# ============================================================================
# Streamed (double-buffered) transpose-load -- Gap 5.
#
# x_stream.load(i, transpose=True) loads source tile i into a rotating slot
# already shaped for the transposed output. Today BlockStream.load has no
# transpose= param (TypeError) and the slots are sized to the non-transposed
# load shape, so this whole flow is unreachable. Post-fix it must:
#   5a -- forward transpose= to the child load,
#   5b -- size the rotating slots to the transposed shape,
# and produce the per-tile transpose numerically.
# ============================================================================

# Source (256, 128): 2 S-tiles of (128, 128) on dim 0. Each tile transposes
# to (128, 128); we write each transposed tile back to its row block.
_XPOSE_STREAM_TS = 128
_XPOSE_STREAM_H = 128
_XPOSE_STREAM_NTILES = 2


@nki.jit
def _kernel_stream_transpose(src):
    """Stream each (TS, H) S-tile with transpose=True into a rotating slot,
    store the (H, TS) transposed tile back to the matching output block."""
    M, H = src.shape  # (256, 128)
    out = nl.ndarray((M, H), dtype=src.dtype, buffer=nl.shared_hbm)

    s_tiles = nt.tiles(src, tile_size=(_XPOSE_STREAM_TS, _XPOSE_STREAM_H))
    x_stream = s_tiles.stream(buffer_count=2)
    assert x_stream.count == _XPOSE_STREAM_NTILES, str(x_stream.count)

    for i in range(_XPOSE_STREAM_NTILES):
        slot = x_stream.load(i, transpose=True)  # first load fixes transpose mode
        # Transposed slot is (H, TS); write it to out[i*TS:(i+1)*TS, :] as the
        # transpose of source tile i.
        nisa.dma_copy(out[nl.ds(i * _XPOSE_STREAM_TS, _XPOSE_STREAM_TS), :], slot.data)
    return out


def _stream_transpose_inputs(_):
    np.random.seed(42)
    shape = (_XPOSE_STREAM_TS * _XPOSE_STREAM_NTILES, _XPOSE_STREAM_H)
    return {"src": np.random.randn(*shape).astype(ml_dtypes.bfloat16)}


def _stream_transpose_output(kernel_input):
    return {"out": np.zeros_like(kernel_input["src"])}


def _stream_transpose_ref(src: torch.Tensor) -> torch.Tensor:
    """Per-S-tile transpose: each (TS, H) tile is transposed to (H, TS) and
    written back to the same row block."""
    out = src.clone()
    ts = _XPOSE_STREAM_TS
    n = _XPOSE_STREAM_NTILES
    for i in range(n):
        block = src[i * ts : (i + 1) * ts, :]
        out[i * ts : (i + 1) * ts, :] = block.t().contiguous()
    return out


# Multi-tile rotation: 5 S-tiles through buffer_count=2 slots -> each slot
# reused ~3x. Proves the rotating pool wraps correctly for transpose loads and
# that the slot-identity contract (slot i and slot i+buffer_count share one
# SBUF buffer) holds -- doc §5.x#4. NTILES > buffer_count is the point.
_ROT_TS = 128
_ROT_H = 128
_ROT_NTILES = 5
_ROT_BUFFERS = 2


@nki.jit
def _kernel_stream_transpose_rotate(src):
    """Transpose-stream 5 S-tiles through 2 rotating slots. Trace-time asserts
    pin slot identity (i and i+buffer_count are the same buffer); numeric output
    proves rotation does not corrupt data across the wrap."""
    M, H = src.shape  # (5*128, 128)
    out = nl.ndarray((M, H), dtype=src.dtype, buffer=nl.shared_hbm)

    s_tiles = nt.tiles(src, tile_size=(_ROT_TS, _ROT_H))
    x_stream = s_tiles.stream(buffer_count=_ROT_BUFFERS)
    assert x_stream.count == _ROT_NTILES, str(x_stream.count)

    for i in range(_ROT_NTILES):
        slot = x_stream.load(i, transpose=True)  # first load fixes transpose mode + sizes slots
        nisa.dma_copy(out[nl.ds(i * _ROT_TS, _ROT_TS), :], slot.data)

    # Slot-identity (checked after loads have sized the transposed pool):
    # stream[i] and stream[i + buffer_count] are the same SBUF buffer (the
    # rotating pool wraps modulo buffer_count). Assert on the internal _source
    # -- the stable rotating-buffer ndarray -- since .data is a fresh AP per call.
    assert x_stream[0]._source is x_stream[_ROT_BUFFERS]._source, "slot 0 != slot buffer_count (no rotation)"
    assert x_stream[1]._source is x_stream[1 + _ROT_BUFFERS]._source, "slot 1 != slot 1+buffer_count"
    return out


def _rotate_inputs(_):
    np.random.seed(7)
    shape = (_ROT_TS * _ROT_NTILES, _ROT_H)
    return {"src": np.random.randn(*shape).astype(ml_dtypes.bfloat16)}


def _rotate_output(kernel_input):
    return {"out": np.zeros_like(kernel_input["src"])}


def _rotate_ref(src: torch.Tensor) -> torch.Tensor:
    out = src.clone()
    for i in range(_ROT_NTILES):
        block = src[i * _ROT_TS : (i + 1) * _ROT_TS, :]
        out[i * _ROT_TS : (i + 1) * _ROT_TS, :] = block.t().contiguous()
    return out


@nki.jit
def _kernel_stream_mix_load_types(src):
    """Misuse: a normal load then a transpose load on the same stream. The
    rotating slots are sized by the first load, so the second must be rejected
    at trace time (the gate in BlockStream._ensure_buffers)."""
    m, h = src.shape
    out = nl.ndarray((m, h), dtype=src.dtype, buffer=nl.shared_hbm)
    s_tiles = nt.tiles(src, tile_size=(_ROT_TS, _ROT_H))
    stream = s_tiles.stream(buffer_count=2)
    a = stream.load(0)  # locks normal mode + sizes slots
    b = stream.load(1, transpose=True)  # conflicting -> AssertionError
    nisa.dma_copy(out[nl.ds(0, _ROT_TS), :], a.data)
    nisa.dma_copy(out[nl.ds(_ROT_TS, _ROT_TS), :], b.data)
    return out


# stream(transpose=True) + grid-typed slot: the slot is allocated AND typed to
# the transposed block up front, so stream[bi] is tile-addressable and each
# interior tile can be a per-tile transpose-load dst. Block of BS=2 (TS, H)
# seq-tiles; H=256 -> H1=2 transposed chunks per tile.
_SLOT_TS = 128
_SLOT_H0 = 128
_SLOT_H = 256
_SLOT_H1 = _SLOT_H // _SLOT_H0
_SLOT_BS = 2


@nki.jit
def _kernel_stream_transpose_slot_dst(src):
    """stream(transpose=True): read the grid-typed rotating slot, transpose-load
    each owned seq-tile into the slot's interior tile via slot[0, ti].data."""
    out = nl.ndarray((_SLOT_H0, _SLOT_BS * _SLOT_H1 * _SLOT_TS), dtype=src.dtype, buffer=nl.shared_hbm)
    s_tiles = nt.tiles(src, tile_size=(_SLOT_TS, _SLOT_H))
    x_blocks = nt.blocks(s_tiles, block_size=(_SLOT_BS, 1))
    x_stream = x_blocks.stream(buffer_count=2, transpose=True)

    slot = x_stream[0]  # grid-typed transposed-block slot: (1, BS) tile grid
    blk_tiles = nt.tiles(x_blocks[0])  # the block's interior seq-tiles
    for ti in range(_SLOT_BS):
        blk_tiles[ti].load(transpose=True, dst=slot[0, ti].data)
    nisa.dma_copy(out, slot.data)
    return out


def _slot_dst_inputs(_):
    np.random.seed(42)
    return {"src": np.random.randn(_SLOT_TS * _SLOT_BS, _SLOT_H).astype(ml_dtypes.bfloat16)}


def _slot_dst_output(_):
    return {"out": np.zeros((_SLOT_H0, _SLOT_BS * _SLOT_H1 * _SLOT_TS), dtype=ml_dtypes.bfloat16)}


def _slot_dst_ref(src: torch.Tensor) -> torch.Tensor:
    """Each seq-tile transposed + H-chunk-packed (H0, H1*TS), tiles side by side."""
    ts, h0, h1 = _SLOT_TS, _SLOT_H0, _SLOT_H1

    def tile_xpose(rows):
        return torch.cat([rows[:, c * h0 : (c + 1) * h0].t().contiguous() for c in range(h1)], dim=1)

    return torch.cat([tile_xpose(src[i * ts : (i + 1) * ts, :]) for i in range(_SLOT_BS)], dim=1)


# Tile-sharded (interleaved) transpose stream: shard 0 of 2 owns the EVEN seq-tiles,
# so a block's owned tiles are non-contiguous (gapped) in HBM. The transpose slot
# must be sized to the OWNED tiles (dense), not the gapped span -- if it were the
# gapped span the per-tile dst writes would land into a 2x-inflated slot and the
# dma_copy of slot.data would read uninitialized gap rows. 8 seq-tiles, 2 shards,
# BS=2 -> shard 0 owns 4 tiles -> 2 blocks; this kernel transposes block 0.
_SH_TS = 128
_SH_H = 256
_SH_H1 = _SH_H // 128
_SH_NSH = 2
_SH_NTILES = 8
_SH_BS = 2


@nki.jit
def _kernel_stream_transpose_sharded_slot(src):
    """Interleaved tile-sharded transpose stream: per-owned-tile dst into the
    dense slot. Exercises owned-extent (not gapped-span) slot sizing."""
    out = nl.ndarray((128, _SH_BS * _SH_H1 * _SH_TS), dtype=src.dtype, buffer=nl.shared_hbm)
    own_r = nt.interleaved_range(rank=0, num_shards=_SH_NSH, total=_SH_NTILES)
    s_tiles = nt.tiles(src, tile_size=(_SH_TS, _SH_H))[own_r, :]
    x_blocks = nt.blocks(s_tiles, block_size=(_SH_BS, 1))
    x_stream = x_blocks.stream(buffer_count=2, transpose=True)

    slot = x_stream[0]  # dense grid-typed slot over the 2 OWNED tiles
    blk_tiles = nt.tiles(x_blocks[0])
    for ti in range(_SH_BS):
        blk_tiles[ti].load(transpose=True, dst=slot[0, ti].data)
    nisa.dma_copy(out, slot.data)
    return out


def _sharded_slot_inputs(_):
    np.random.seed(42)
    return {"src": np.random.randn(_SH_TS * _SH_NTILES, _SH_H).astype(ml_dtypes.bfloat16)}


def _sharded_slot_output(_):
    return {"out": np.zeros((128, _SH_BS * _SH_H1 * _SH_TS), dtype=ml_dtypes.bfloat16)}


def _sharded_slot_ref(src: torch.Tensor) -> torch.Tensor:
    """Block 0's owned (even) seq-tiles 0 and 2, each transposed + H-chunk-packed."""
    ts, h1 = _SH_TS, _SH_H1
    owned = [0, _SH_NSH]  # block 0's two owned tiles (interleaved: 0, 2)

    def tile_xpose(rows):
        return torch.cat([rows[:, c * 128 : (c + 1) * 128].t().contiguous() for c in range(h1)], dim=1)

    return torch.cat([tile_xpose(src[t * ts : (t + 1) * ts, :]) for t in owned], dim=1)


@pytest_marks(["neurotile"])
class TestStreamLoadTypeConsistency:
    """Mixing transpose and non-transpose loads on one stream is rejected at
    trace time -- the rotating slots are sized by the first load."""

    @pytest.mark.fast
    def test_mixed_load_types_rejected(self):
        src = np.zeros((_ROT_TS * 2, _ROT_H), dtype=ml_dtypes.bfloat16)
        with pytest.raises(AssertionError, match="same transpose"):
            _kernel_stream_mix_load_types(src)


@pytest_marks(["neurotile"])
class TestStreamTransposeLoad:
    """Gap 5: double-buffered transpose-load via BlockStream."""

    def test_stream_transpose(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_stream_transpose,
            torch_ref=torch_ref_wrapper(_stream_transpose_ref),
            kernel_input_generator=_stream_transpose_inputs,
            output_tensor_descriptor=_stream_transpose_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_stream_transpose_rotation(self, test_manager: Orchestrator, platform_target: Platforms):
        """5 tiles through 2 slots: slot-identity asserts (trace-time) + numeric
        correctness across the rotation wrap (doc §5.x#4)."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_stream_transpose_rotate,
            torch_ref=torch_ref_wrapper(_rotate_ref),
            kernel_input_generator=_rotate_inputs,
            output_tensor_descriptor=_rotate_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_stream_transpose_slot_dst(self, test_manager: Orchestrator, platform_target: Platforms):
        """stream(transpose=True): the grid-typed slot is tile-addressable, and
        each interior tile is a valid per-tile transpose-load dst (slot[0, ti].data).
        Validates the construction-time transpose declaration + slot grid-typing."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_stream_transpose_slot_dst,
            torch_ref=torch_ref_wrapper(_slot_dst_ref),
            kernel_input_generator=_slot_dst_inputs,
            output_tensor_descriptor=_slot_dst_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_stream_transpose_sharded_slot(self, test_manager: Orchestrator, platform_target: Platforms):
        """Interleaved tile-sharded transpose stream: the slot is sized to the
        block's OWNED tiles (dense), not the gapped iteration span. Per-owned-tile
        dst writes land in the dense slot and dma_copy reads no uninitialized gap."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_stream_transpose_sharded_slot,
            torch_ref=torch_ref_wrapper(_sharded_slot_ref),
            kernel_input_generator=_sharded_slot_inputs,
            output_tensor_descriptor=_sharded_slot_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )
