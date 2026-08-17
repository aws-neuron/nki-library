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

"""SBUF / HBM AP correctness on partial-trailing-tile and sliced views.

Failure modes that were caught at production-kernel scale:

1. Partition stride (`_build_ap`): when the rotating buffer slot is
   wider than one walked block (e.g., partial trailing I-tile on a
   sliced view), the AP partition stride must equal the SBUF ndarray's
   allocated F width, not the walked F.

2. HBM AP merge clamp (`ap_emitter`): on a sharded view, a gappy
   tile-walk axis (step > nested walk) merged with an outer block axis
   must clamp the merged count using inner-walk-aware arithmetic.

3. HBM addressable extent (`dim_addressable`): an `nt.tensor_view`
   chain that `slice`s a reshaped dim must keep the full sliced extent
   addressable -- the offset is subtracted only for remainder dims. The
   under-report collapsed a dim and made `dma_copy` reject the element
   count at compile time. Exercised through the two conv3d_neurotile
   chains: output store (flatten -> reshape_dim -> slice) and filter
   load (3x slice+squeeze_dim).
"""

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
def _stream_partial_tile_kernel(src):
    """Stream-load a partial trailing I-tile from a wider rotating buffer."""
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    src_blocks = nt.blocks(src, tile_size=(128, 256), block_size=(4, 1))
    src_partial = src_blocks[:, 1:2]
    src_stream = src_partial.stream(buffer_count=2)

    for k_block_idx in range(src_partial.shape[0]):
        loaded = src_stream.load(k_block_idx)
        tiles = nt.tiles(loaded)
        for _k in range(tiles.shape[0]):
            psum = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(psum, 0.0)

    nisa.dma_copy(dst, src)
    return dst


@nki.jit
def _interleaved_block_load_kernel(src):
    """Sharded interleaved + multi-block load with tolist iteration."""
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    shard_id = nl.program_id(0)
    num_shards = nl.num_programs(0)

    src_flat = src.flatten_dims(0, 1)
    r = nt.interleaved_range(rank=shard_id, num_shards=num_shards, total=8)
    tiles_view = nt.tiles(src_flat, tile_size=(128, 128))[r, :]
    blocks = nt.blocks(tiles_view, block_size=(2, 1))

    items = blocks.tolist()
    for i in range(len(items)):
        block_row = items[i]
        sb_block = block_row.load()  # noqa: F841

    nisa.dma_copy(dst, src)
    return dst


def _stream_partial_inputs(_):
    np.random.seed(42)
    return {
        "src": np.random.randn(1024, 384)
        .astype(np.float32)
        .astype(np.dtype("bfloat16") if hasattr(np, "bfloat16") else np.float32)
    }


def _stream_partial_inputs_simple(_):
    np.random.seed(42)
    import ml_dtypes

    return {"src": np.random.randn(1024, 384).astype(ml_dtypes.bfloat16)}


def _stream_partial_output(kernel_input):
    return {"out": np.zeros(kernel_input["src"].shape, dtype=kernel_input["src"].dtype)}


def _stream_partial_ref(src: torch.Tensor) -> torch.Tensor:
    """The kernel's only observable behavior is `nisa.dma_copy(dst, src)`."""
    return src.clone()


def _interleaved_block_inputs(_):
    np.random.seed(42)
    import ml_dtypes

    return {"src": np.random.randn(1, 1024, 8192).astype(ml_dtypes.bfloat16)}


def _interleaved_block_output(kernel_input):
    return {"out": np.zeros(kernel_input["src"].shape, dtype=kernel_input["src"].dtype)}


def _interleaved_block_ref(src: torch.Tensor) -> torch.Tensor:
    """The kernel's only observable behavior is `nisa.dma_copy(dst, src)`."""
    return src.clone()


# --- nt.tensor_view addressable-extent chains (conv3d_neurotile) ---

_P_MAX = 128
_NUM_DH_STACKED = 4


def _dh_group_width(dh, dh_end, h_out):
    """Width of the next dh group from ``dh``: capped at the stacking width,
    the shard end, and the D boundary (a group never spans two D_out
    positions -- mirrors conv3d_neurotile's window-reuse constraint)."""
    count = min(_NUM_DH_STACKED, dh_end - dh)
    d_first = dh // h_out
    d_last = (dh + count - 1) // h_out
    if d_last != d_first:
        count = (d_first + 1) * h_out - dh
    return count


@nki.jit
def _output_store_chain_kernel(inp):
    """Copy ``inp`` -> ``out`` by storing each (c_out_tile, dh_group) region
    through the nt.tensor_view flatten -> reshape_dim -> slice output-store
    chain. dh positions split into two shards (mirroring lnc_shard) so the
    second shard's first group starts at a non-multiple of the stacking
    width -- the condition that exposed the addressable-extent bug."""
    B, C_out, D_out, H_out, W_out = inp.shape
    out = nl.ndarray(inp.shape, dtype=inp.dtype, buffer=nl.shared_hbm)
    total_dh = D_out * H_out
    shard_dh = (total_dh + 1) // 2

    for b in range(B):
        for shard in range(2):
            dh = shard * shard_dh
            dh_end = min((shard + 1) * shard_dh, total_dh)
            while dh < dh_end:
                count = _dh_group_width(dh, dh_end, H_out)
                eff = count * W_out
                c_start = 0
                while c_start < C_out:
                    c_end = min(c_start + _P_MAX, C_out)
                    c_size = c_end - c_start

                    # Load the matching source region with native NkiTensor
                    # views (the known-good path) -- contiguous in flat D*H*W.
                    src_sbuf = nl.ndarray((c_size, eff), dtype=inp.dtype, buffer=nl.sbuf)
                    src_view = (
                        inp.select(0, b)
                        .slice(0, c_start, c_end)
                        .flatten_dims(1, 3)
                        .slice(1, dh * W_out, (dh + count) * W_out)
                    )
                    nisa.dma_copy(dst=src_sbuf, src=src_view)

                    # Store it back through the NkiTensor transform chain under test.
                    result_view = src_sbuf.reshape_dim(1, (count, W_out))
                    out_nt = (
                        out.slice(0, b, b + 1)
                        .squeeze_dim(0)
                        .slice(0, c_start, c_end)
                        .flatten_dims(1, 3)
                        .reshape_dim(1, (total_dh, W_out))
                        .slice(1, dh, dh + count)
                        .slice(2, 0, W_out)
                    )
                    nisa.dma_copy(dst=out_nt, src=result_view)

                    c_start = c_start + _P_MAX
                dh = dh + count
    return out


@nki.jit
def _filter_load_chain_kernel(filters):
    """Gather each (C_in_tile, C_out_tile) plane of a 5-D filter through the
    nt.tensor_view 3x slice+squeeze_dim chain, then store it to ``out``."""
    K_d, K_h, K_w, C_in, C_out = filters.shape
    c_in_tile = min(_P_MAX, C_in)
    c_out_tile = min(256, C_out)
    num_pos = K_d * K_h * K_w
    out = nl.ndarray((num_pos, c_in_tile, c_out_tile), dtype=filters.dtype, buffer=nl.shared_hbm)

    pos = 0
    for kd in range(K_d):
        for kh in range(K_h):
            for kw in range(K_w):
                sbuf = nl.ndarray((c_in_tile, c_out_tile), dtype=filters.dtype, buffer=nl.sbuf)
                filter_view = (
                    filters.slice(0, kd, kd + 1)
                    .squeeze_dim(0)
                    .slice(0, kh, kh + 1)
                    .squeeze_dim(0)
                    .slice(0, kw, kw + 1)
                    .squeeze_dim(0)
                    .slice(0, 0, c_in_tile)
                    .slice(1, 0, c_out_tile)
                )
                nisa.dma_copy(dst=sbuf, src=filter_view)

                out_plane = out.select(0, pos)
                nisa.dma_copy(dst=out_plane, src=sbuf)
                pos = pos + 1
    return out


@nki.jit
def _psum_slice_reshape_kernel(src):
    """Allocate a PSUM bank at the MAX free width, slice its .data
    to a narrower effective width, then ``.reshape_dim(...)`` on the bank's
    NkiTensor and use the reshaped view as a DMA source. The reshape must thread
    the bank's physical row width (512), not the contiguous-of-logical 144, or the
    on-chip partition-stride check ("Partition step 144 must equal tensor free
    dimension size 512") fails at compile."""
    P, MAX_F, EFF, NUM_DH, W = 128, 512, 144, 12, 12
    out = nl.ndarray((P, EFF), dtype=src.dtype, buffer=nl.shared_hbm)

    # Pool allocated at the max free width (512); one bank used here.
    psums = nt.psum_pool(tile_size=(P, MAX_F), grid=(1, 1), bank_ids=(0,), dtype=nl.float32)
    # Compute the effective region via a real matmul into the bank's first EFF cols.
    lhsT = nl.ndarray((P, P), dtype=src.dtype, buffer=nl.sbuf)
    rhs = nl.ndarray((P, EFF), dtype=src.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=rhs, src=src)
    nisa.memset(lhsT, 0.0)
    nisa.nc_matmul(psums[0, 0].data[:, :EFF], lhsT, rhs)

    # Slice the 512-wide bank .data to the narrower EFF=144, then reshape and
    # use the view as a compute operand -- the slice is backed by a 512-wide
    # buffer, so the reshape must thread the physical row width 512 (not
    # contiguous-of-logical 144) or the on-chip partition-stride check fails at
    # compile (PSUM -> SBUF via the reshaped view, then SBUF -> HBM store).
    psum_eff = psums[0, 0].data[:, :EFF]  # logical (128, 144), physical row width 512
    psum_view = psum_eff.reshape_dim(1, (NUM_DH, W))  # [128, 12, 12]; threads phys 512
    eff_sb = nl.ndarray((P, EFF), dtype=src.dtype, buffer=nl.sbuf)
    sb_view = eff_sb.reshape_dim(1, (NUM_DH, W))
    nisa.tensor_copy(sb_view, psum_view)
    nisa.dma_copy(dst=out, src=eff_sb)
    return out


def _psum_slice_reshape_ref(src: torch.Tensor) -> torch.Tensor:
    # lhsT is zeros -> matmul output is zeros; kernel writes zeros to out.
    return torch.zeros((128, 144), dtype=src.dtype)


# conv3d W-padding: a free-dim slice AFTER reshape_dim must narrow
# WITHIN the PSUM tile (advance the free offset), not move the tile-array cursor.
_PW_P, _PW_NUM_DH, _PW_W = 128, 4, 12
_PW_EFF = _PW_NUM_DH * _PW_W  # 48 valid free elements
_PW_MAX_F = 512  # bank over-allocated to the max (the failing condition)
_PW_DH_START, _PW_DH_COUNT = 1, 2
_PW_W_START, _PW_W_COUNT = 3, 7


@nki.jit
def _psum_strided_wpad_kernel(src):
    """conv3d W-padded strided matmul into an over-allocated PSUM bank.

    Writes only the valid sub-region dh[1:3] x w[3:10] of a (4, 12) free layout
    via slice -> reshape_dim -> slice -> slice, leaving padded positions zero.
    The post-reshape free-dim slices must narrow within the tile -- regression
    for the tile-index/in-tile-offset conflation that crashed at
    ``tile_arrays[offset]`` (index OOB) on a single-tile PSUM view."""
    out = nl.ndarray((_PW_P, _PW_EFF), dtype=nl.float32, buffer=nl.shared_hbm)

    psums = nt.psum_pool(tile_size=(_PW_P, _PW_MAX_F), grid=(1, 1), bank_ids=(0,), dtype=nl.float32)
    nisa.memset(psums[0, 0].data, 0.0)

    # All-ones operands: each valid output position = contraction sum = P (128).
    stat = nl.ndarray((_PW_P, _PW_P), dtype=nl.float32, buffer=nl.sbuf)
    moving = nl.ndarray((_PW_P, _PW_DH_COUNT * _PW_W_COUNT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(stat, 1.0)
    nisa.memset(moving, 1.0)

    # The conv3d W-padded strided view chain, on the PSUM bank's NkiTensor.
    view = (
        psums[0, 0]
        .data[:, :_PW_EFF]
        .slice(1, 0, _PW_EFF)
        .reshape_dim(1, (_PW_NUM_DH, _PW_W))
        .slice(1, _PW_DH_START, _PW_DH_START + _PW_DH_COUNT)
        .slice(2, _PW_W_START, _PW_W_START + _PW_W_COUNT)
    )
    nisa.nc_matmul(view, stat, moving)

    sb = nl.ndarray((_PW_P, _PW_EFF), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(sb, psums[0, 0].data[:, :_PW_EFF])
    nisa.dma_copy(dst=out, src=sb)
    return out


def _psum_strided_wpad_ref(src: torch.Tensor) -> torch.Tensor:
    # All-ones (128-contraction) matmul -> valid positions hold 128.0; padded 0.
    out = torch.zeros((_PW_P, _PW_NUM_DH, _PW_W), dtype=torch.float32)
    out[:, _PW_DH_START : _PW_DH_START + _PW_DH_COUNT, _PW_W_START : _PW_W_START + _PW_W_COUNT] = float(_PW_P)
    return out.reshape(_PW_P, _PW_EFF)


# (B, C_out, D_out, H_out, W_out); total_dh chosen so the shard split lands a
# group start at a non-multiple of the stacking width (the failing condition).
_OUTPUT_STORE_SHAPES = [
    (1, 256, 1, 30, 104),  # total_dh=30 -> shard start 15, 15 % 4 = 3 (the C1024 H/W)
    (1, 128, 1, 26, 100),  # total_dh=26 -> shard start 13, 13 % 4 = 1
    (2, 192, 2, 12, 12),  # total_dh=24, D_out=2 -> exercises the D-boundary group split
]

# (K_d, K_h, K_w, C_in, C_out)
_FILTER_SHAPES = [
    (3, 3, 3, 1024, 1024),
    (3, 3, 3, 256, 512),
]


def _identity_ref(inp: torch.Tensor) -> torch.Tensor:
    return inp.clone()


def _filter_planes_ref(filters: torch.Tensor) -> torch.Tensor:
    K_d, K_h, K_w, C_in, C_out = filters.shape
    c_in_tile = min(_P_MAX, C_in)
    c_out_tile = min(256, C_out)
    planes = []
    for kd in range(K_d):
        for kh in range(K_h):
            for kw in range(K_w):
                planes.append(filters[kd, kh, kw, :c_in_tile, :c_out_tile])
    return torch.stack(planes, dim=0)


@pytest_marks(["neurotile"])
class TestSBUFAPPartitionStride:
    """Validation that AP construction emits correct strides for partial tiles."""

    @pytest.mark.fast
    def test_stream_partial_tile_partition_stride(self, test_manager: Orchestrator, platform_target: Platforms):
        """SBUFLayout._build_ap: partition stride uses allocated F, not walked F."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_stream_partial_tile_kernel,
            torch_ref=torch_ref_wrapper(_stream_partial_ref),
            kernel_input_generator=_stream_partial_inputs_simple,
            output_tensor_descriptor=_stream_partial_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.fast
    def test_interleaved_block_load_merge_clamp(self, test_manager: Orchestrator, platform_target: Platforms):
        """APEmitter._merge_contiguous: clamp accounts for inner-walk on gappy axes."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_interleaved_block_load_kernel,
            torch_ref=torch_ref_wrapper(_interleaved_block_ref),
            kernel_input_generator=_interleaved_block_inputs,
            output_tensor_descriptor=_interleaved_block_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=2),
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("shape", _OUTPUT_STORE_SHAPES)
    def test_tensor_view_output_store_chain(self, test_manager: Orchestrator, platform_target: Platforms, shape):
        """HBMLayout.dim_addressable: flatten_dims -> reshape_dim -> slice output
        store keeps the full sliced extent addressable (round-trips identity)."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_output_store_chain_kernel,
            torch_ref=torch_ref_wrapper(_identity_ref),
            kernel_input_generator=lambda _: {"inp": np.random.randn(*shape).astype(np.float32)},
            output_tensor_descriptor=lambda _: {"out": np.zeros(shape, dtype=np.float32)},
        )
        np.random.seed(42)
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("shape", _FILTER_SHAPES)
    def test_tensor_view_filter_load_chain(self, test_manager: Orchestrator, platform_target: Platforms, shape):
        """HBMLayout.dim_addressable: 3x slice+squeeze_dim filter load gathers
        the correct C_in/C_out plane from a 5-D filter."""
        K_d, K_h, K_w, C_in, C_out = shape
        c_in_tile = min(_P_MAX, C_in)
        c_out_tile = min(256, C_out)
        num_pos = K_d * K_h * K_w

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_filter_load_chain_kernel,
            torch_ref=torch_ref_wrapper(_filter_planes_ref),
            kernel_input_generator=lambda _: {"filters": np.random.randn(*shape).astype(np.float32)},
            output_tensor_descriptor=lambda _: {"out": np.zeros((num_pos, c_in_tile, c_out_tile), dtype=np.float32)},
        )
        np.random.seed(42)
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.fast
    def test_psum_slice_reshape_threads_physical_stride(self, test_manager: Orchestrator, platform_target: Platforms):
        """Slicing a max-allocated PSUM bank, then ``.reshape_dim(...)`` on its
        NkiTensor, must thread the bank's physical row width so the reshaped
        view's AP has a valid on-chip partition stride (regression: used
        contiguous-of-logical 144 != row width 512)."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_psum_slice_reshape_kernel,
            torch_ref=torch_ref_wrapper(_psum_slice_reshape_ref),
            kernel_input_generator=lambda _: {"src": np.zeros((128, 144), dtype=np.float32)},
            output_tensor_descriptor=lambda _: {"out": np.zeros((128, 144), dtype=np.float32)},
        )
        np.random.seed(42)
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.fast
    def test_psum_strided_wpad_matmul(self, test_manager: Orchestrator, platform_target: Platforms):
        """conv3d W-padded strided matmul into an over-allocated
        PSUM bank. A free-dim slice after reshape_dim must narrow within the
        tile (advance the free offset), not index the tile array -- regression
        for the crash at ``tile_arrays[offset]`` on a single-tile PSUM view.
        Verifies the matmul lands only in the valid strided sub-region."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_psum_strided_wpad_kernel,
            torch_ref=torch_ref_wrapper(_psum_strided_wpad_ref),
            kernel_input_generator=lambda _: {"src": np.zeros((128, _PW_EFF), dtype=np.float32)},
            output_tensor_descriptor=lambda _: {"out": np.zeros((128, _PW_EFF), dtype=np.float32)},
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-5,
            atol=1e-5,
        )
