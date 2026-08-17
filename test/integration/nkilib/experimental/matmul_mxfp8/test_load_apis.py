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

"""Integration tests for load_tile_PE_swizzle_wrapX.

Validates that load_tile_PE_swizzle_wrapX produces the correct swizzled SBUF
layout [tile_k // 4, tile_f * 4] from an unswizzled [F, K] bf16 input tensor.

Tests cover:
  - Direct DMA mode (contiguous rows) with tile_k in {128, 256, 512}
  - Indirect DMA mode (gathered rows) with tile_k in {128, 256, 512}
  - OOB skip mode (skip_token=True with -1 padding indices)
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.moe.bwd.moe_bwd_parameters import SkipMode
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.common_dataclasses import (
    P_MAX,
    TensorDescriptor,
    TileLocation,
)
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.common_utils import (
    create_and_set_active_sbm,
    get_active_sbm,
    with_active_sbm,
)
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.load_apis import (
    load_tile_PE_swizzle_wrapX,
)
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.quantize_mxfp8_utils import (
    INTERLEAVE_FACTOR,
)

from test.integration.nkilib.experimental.matmul_mxfp8 import utils as matmul_utils
from test.utils import common_dataclasses
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.unit_test_framework import UnitTestFramework

# ============================================================================
# NKI Kernels
# ============================================================================


@nki.jit
@with_active_sbm
def pe_transpose_load_kernel(
    tensor_fk,
    F: int,
    K: int,
    tile_f: int,
    tile_k: int,
    tiles_in_k: int = 1,
    k_offset: int = 0,
    f_offset: int = 0,
    store_f_offset: int = 0,
    row_indices=None,
    skip_token: bool = False,
):
    """Load tile(s) using BF16 PE transpose and copy the swizzled result to HBM.

    Supports both direct and indirect DMA modes:
      - row_indices=None: Direct DMA, loads contiguous rows from f_offset.
      - row_indices=tensor: Indirect DMA, gathers rows by index from SBUF.

    When tiles_in_k > 1, loads multiple consecutive k-tiles into successive
    slots of the 4D SBUF buffer.

    When store_f_offset > 0, writes the loaded tile at a non-zero F position
    within the destination buffer.

    When skip_token=True, enables oob_mode.skip so that out-of-bounds indices
    (e.g., -1) are skipped by the DMA engine. The destination is pre-zeroed so
    skipped rows produce zeros in the output.

    Args:
        tensor_fk: Input [F, K] bf16 tensor in HBM.
        F: F dimension size.
        K: K dimension size.
        tile_f: Tile size in F dimension.
        tile_k: Tile size in K dimension (per k-tile).
        tiles_in_k: Number of consecutive k-tiles to load (default 1).
        k_offset: Starting offset in K dimension (default 0).
        f_offset: Offset in F dimension (default 0, used in direct mode only).
        store_f_offset: F offset in the destination buffer (default 0).
        row_indices: Optional [P_MAX, NUM_SUB_TILES] int32 tensor in HBM.
            If provided, enables indirect DMA gather mode.
        skip_token: If True, uses oob_mode.skip for DMA (default False).

    Returns:
        output: [physical_tile_k * tiles_in_k, total_physical_f] bf16 in HBM.
    """
    create_and_set_active_sbm()
    sbm = get_active_sbm()
    sbm.open_scope("PE_TRANSPOSE_LOAD")

    physical_tile_k = tile_k // INTERLEAVE_FACTOR
    total_physical_f = (store_f_offset + tile_f) * INTERLEAVE_FACTOR

    # Create TensorDescriptor for the input
    td = TensorDescriptor(
        data=tensor_fk,
        load_with_PE_swizzle=True,
        skip_dma=SkipMode(skip_token=skip_token),
    )

    # If row_indices provided, load them to SBUF
    indices_sbuf = None
    if row_indices is not None:
        NUM_SUB_TILES = tile_f // P_MAX
        indices_sbuf = nl.ndarray((P_MAX, NUM_SUB_TILES), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=indices_sbuf, src=row_indices)

    # Allocate 4D destination SBUF: [physical_tile_k, tiles_in_k, 1, total_physical_f]
    data_sbuf = sbm.alloc_stack(
        shape=(physical_tile_k, tiles_in_k, 1, total_physical_f),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )

    for k_idx in nl.affine_range(tiles_in_k):
        current_k_offset = k_offset + k_idx * tile_k

        if indices_sbuf is not None:
            load_loc = TileLocation(
                tensor=td,
                tile_k=tile_k,
                tile_f=tile_f,
                k_offset=current_k_offset,
                f_offset=0,
                vector_offset=indices_sbuf,
            )
        else:
            load_loc = TileLocation(
                tensor=td,
                tile_k=tile_k,
                tile_f=tile_f,
                k_offset=current_k_offset,
                f_offset=f_offset,
            )

        data_store_loc = TileLocation(
            tensor=TensorDescriptor(data=data_sbuf, is_swizzled=True, is_f_by_k=False),
            tile_k=physical_tile_k,
            tile_f=tile_f,
            k_offset=k_idx,
            f_offset=store_f_offset,
        )

        load_tile_PE_swizzle_wrapX(load_loc, data_store_loc=data_store_loc)

    # Copy to HBM: store each k-tile slot separately since SBUF reshape
    # cannot change the partition dimension count.
    output = nl.ndarray(
        (physical_tile_k * tiles_in_k, total_physical_f),
        dtype=nl.bfloat16,
        buffer=nl.shared_hbm,
    )
    for k_idx in nl.affine_range(tiles_in_k):
        src_slice = data_sbuf[nl.ds(0, physical_tile_k), nl.ds(k_idx, 1), 0, nl.ds(0, total_physical_f)]
        dst_slice = output[nl.ds(k_idx * physical_tile_k, physical_tile_k), nl.ds(0, total_physical_f)]
        nisa.dma_copy(dst=dst_slice, src=src_slice)

    sbm.close_scope()
    return output


# ============================================================================
# Torch reference functions
# ============================================================================


def pe_transpose_torch_ref(
    tensor_fk,
    F: int,
    K: int,
    tile_f: int,
    tile_k: int,
    tiles_in_k: int = 1,
    k_offset: int = 0,
    f_offset: int = 0,
    store_f_offset: int = 0,
    row_indices=None,
    skip_token: bool = False,
):
    """Golden reference for PE transpose output.

    Supports both direct and indirect modes:
      - row_indices=None: Slices [f_offset:f_offset+tile_f, k_offset:k_offset+tile_k].
      - row_indices provided: Gathers rows by index, slices K at k_offset.

    When tiles_in_k > 1, produces the concatenation of swizzled tiles along the
    partition dimension. When store_f_offset > 0, the swizzled data is placed at
    the corresponding F position within a wider zero-initialized output.

    When skip_token=True, indices with value -1 are treated as zero rows
    (matching oob_mode.skip behavior where skipped DMA leaves pre-zeroed memory).

    Then transposes to [tile_k, tile_f] and swizzles to the interleaved layout.
    """
    physical_tile_k = tile_k // INTERLEAVE_FACTOR
    physical_f = tile_f * INTERLEAVE_FACTOR
    total_physical_f = (store_f_offset + tile_f) * INTERLEAVE_FACTOR

    all_k_tiles = []
    for k_idx in range(tiles_in_k):
        current_k_offset = k_offset + k_idx * tile_k

        if row_indices is not None:
            indices = row_indices.flatten(order='F').astype(np.int64)
            if skip_token:
                gathered = np.zeros((tile_f, tile_k), dtype=tensor_fk.dtype)
                for i, idx in enumerate(indices):
                    if idx >= 0:
                        gathered[i, :] = tensor_fk[idx, current_k_offset : current_k_offset + tile_k]
            else:
                gathered = tensor_fk[indices, current_k_offset : current_k_offset + tile_k]
            tensor_kf = gathered.T.copy()
        else:
            tensor_kf = tensor_fk[f_offset : f_offset + tile_f, current_k_offset : current_k_offset + tile_k].T.copy()

        swizzled = matmul_utils.swizzle_tensor(tensor_kf, TILE_P=tile_k)

        if store_f_offset > 0:
            row = np.zeros((physical_tile_k, total_physical_f), dtype=tensor_fk.dtype)
            f_start = store_f_offset * INTERLEAVE_FACTOR
            row[:, f_start : f_start + physical_f] = swizzled
            all_k_tiles.append(row)
        else:
            all_k_tiles.append(swizzled)

    # The kernel stores each k-tile contiguously: k_tile 0 at rows [0:P],
    # k_tile 1 at rows [P:2P], etc.
    result = np.concatenate(all_k_tiles, axis=0).astype(nl.bfloat16)
    return {"out": result}


# ============================================================================
# Test shapes — Direct DMA
# ============================================================================

# (F, K, tile_f, tile_k) — covers all tile_k values and varying tile_f
_TEST_SHAPES_DIRECT = [
    # tile_k=512
    (128, 512, 128, 512),
    (256, 1024, 256, 512),
    (512, 2048, 512, 512),
    # tile_k=256
    (128, 512, 128, 256),
    (256, 1024, 256, 256),
    (512, 512, 128, 256),
    # tile_k=128
    (128, 512, 128, 128),
    (256, 1024, 256, 128),
    (512, 512, 128, 128),
]

# (F, K, tile_f, tile_k, k_offset, f_offset) — covers offset math for all tile_k
_TEST_SHAPES_WITH_OFFSETS = [
    # tile_k=512: k_offset only, f_offset only, both
    (128, 1024, 128, 512, 512, 0),
    (512, 512, 128, 512, 0, 256),
    (512, 1024, 256, 512, 512, 256),
    # tile_k=256: k_offset only, both
    (256, 1024, 128, 256, 512, 0),
    (512, 1024, 256, 256, 256, 128),
    # tile_k=128: k_offset only, both
    (256, 512, 128, 128, 256, 0),
    (512, 1024, 256, 128, 384, 256),
]


# ============================================================================
# Test shapes — Indirect DMA
# ============================================================================

# (F, K, tile_f, tile_k, k_offset) — covers all tile_k values
_TEST_SHAPES_INDIRECT = [
    # tile_k=512
    (512, 512, 128, 512, 0),
    (1024, 512, 256, 512, 0),
    (1024, 1024, 128, 512, 512),
    # tile_k=256
    (512, 512, 128, 256, 0),
    (1024, 512, 256, 256, 0),
    (1024, 1024, 128, 256, 256),
    # tile_k=128
    (512, 512, 128, 128, 0),
    (1024, 512, 256, 128, 0),
    (1024, 1024, 128, 128, 512),
]


# ============================================================================
# Test shapes — OOB Skip
# ============================================================================

# (F, K, tile_f, tile_k, k_offset, num_padding_rows) — trailing -1 padding
_TEST_SHAPES_OOB_SKIP_TRAILING = [
    # tile_k=512: all padding, half, sparse, multi-subtile
    (512, 512, 128, 512, 0, 128),
    (512, 512, 128, 512, 0, 64),
    (1024, 512, 256, 512, 0, 64),
    (512, 1024, 128, 512, 512, 32),
    # tile_k=256
    (512, 512, 128, 256, 0, 64),
    (1024, 1024, 128, 256, 512, 32),
    # tile_k=128
    (512, 512, 128, 128, 0, 64),
    (1024, 1024, 128, 128, 512, 32),
    # tile_f=256 (multi-subtile, tile_f > 128)
    (1024, 512, 256, 512, 0, 128),
    (1024, 1024, 256, 256, 512, 64),
    (1024, 512, 256, 128, 0, 64),
    # tile_f=512 (multi-subtile, tile_f > 128)
    (1024, 512, 512, 512, 0, 256),
    (1024, 1024, 512, 256, 512, 128),
    (1024, 512, 512, 128, 0, 128),
]

# (F, K, tile_f, tile_k, k_offset, num_padding_rows) — scattered -1 padding
_TEST_SHAPES_OOB_SKIP_SCATTERED = [
    # tile_k=512
    (512, 512, 128, 512, 0, 32),
    (1024, 1024, 256, 512, 512, 100),
    # tile_k=256
    (512, 512, 128, 256, 0, 32),
    (1024, 1024, 128, 256, 512, 64),
    # tile_k=128
    (512, 512, 128, 128, 0, 32),
    (1024, 1024, 128, 128, 512, 64),
]


# ============================================================================
# Test shapes — Multi K-tile (tiles_in_k > 1)
# ============================================================================

# (F, K, tile_f, tile_k, tiles_in_k, k_offset) — direct DMA, multiple k-tiles
_TEST_SHAPES_MULTI_K_DIRECT = [
    # tiles_in_k=2
    (128, 1024, 128, 512, 2, 0),
    (256, 1024, 128, 256, 2, 0),
    (256, 512, 128, 128, 2, 0),
    (256, 1024, 128, 256, 2, 512),
    # tiles_in_k=4
    (128, 2048, 128, 512, 4, 0),
    (256, 1024, 128, 256, 4, 0),
    (256, 512, 128, 128, 4, 0),
    # tiles_in_k=2, tile_f > 128
    (512, 1024, 256, 512, 2, 0),
    (512, 1024, 512, 256, 2, 0),
]

# (F, K, tile_f, tile_k, tiles_in_k, k_offset) — indirect DMA, multiple k-tiles
_TEST_SHAPES_MULTI_K_INDIRECT = [
    # tiles_in_k=2
    (512, 1024, 128, 512, 2, 0),
    (512, 1024, 128, 256, 2, 0),
    (512, 512, 128, 128, 2, 0),
    (512, 1024, 128, 256, 2, 512),
    # tiles_in_k=4
    (512, 2048, 128, 512, 4, 0),
    (512, 1024, 128, 256, 4, 0),
    # tiles_in_k=2, tile_f > 128
    (1024, 1024, 256, 512, 2, 0),
    (1024, 1024, 512, 256, 2, 0),
]

# (F, K, tile_f, tile_k, tiles_in_k, k_offset, store_f_offset) — non-zero store_f_offset
_TEST_SHAPES_MULTI_K_STORE_F_OFFSET = [
    # tiles_in_k=1, store_f_offset > 0
    (256, 512, 128, 512, 1, 0, 128),
    (256, 512, 128, 256, 1, 0, 128),
    # tiles_in_k=2, store_f_offset > 0
    (256, 1024, 128, 512, 2, 0, 128),
    (256, 1024, 128, 256, 2, 0, 256),
    # tiles_in_k=2, tile_f > 128, store_f_offset > 0
    (512, 1024, 256, 512, 2, 0, 128),
    (1024, 1024, 512, 256, 2, 0, 128),
]


# ============================================================================
# Test class — Direct DMA
# ============================================================================


@pytest_test_metadata(name="BF16 PE Transpose")
@pytest_marks(["bf16_pe_transpose"])
@pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN1, common_dataclasses.Platforms.TRN2])
class TestBF16PETranspose:
    """
    Correctness tests for load_tile_PE_swizzle_wrapX (direct DMA mode).
    """

    @pytest.mark.fast
    @pytest.mark.parametrize("F,K,tile_f,tile_k", _TEST_SHAPES_DIRECT)
    def test_pe_transpose_vs_golden(self, test_manager, platform_target, F, K, tile_f, tile_k):
        """Verify BF16 PE transpose load matches CPU swizzle golden reference."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")

        compiler_args = common_dataclasses.CompilerArgs(
            platform_target=platform_target,
            logical_nc_config=1,
        )

        physical_tile_k = tile_k // INTERLEAVE_FACTOR
        physical_f = tile_f * INTERLEAVE_FACTOR

        rng = np.random.RandomState(42)
        tensor_fk = rng.uniform(-1, 1, (F, K)).astype(nl.bfloat16)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=pe_transpose_load_kernel,
            torch_ref=pe_transpose_torch_ref,
            kernel_input_generator=lambda _: {
                "tensor_fk": tensor_fk,
                "F": F,
                "K": K,
                "tile_f": tile_f,
                "tile_k": tile_k,
            },
            output_tensor_descriptor=lambda _: {
                "out": np.zeros((physical_tile_k, physical_f), dtype=nl.bfloat16),
            },
        )
        framework.run_test(test_config=None, compiler_args=compiler_args, rtol=0, atol=0)

    @pytest.mark.fast
    @pytest.mark.parametrize("F,K,tile_f,tile_k,k_offset,f_offset", _TEST_SHAPES_WITH_OFFSETS)
    def test_pe_transpose_with_offsets(self, test_manager, platform_target, F, K, tile_f, tile_k, k_offset, f_offset):
        """Verify BF16 PE transpose correctly handles non-zero k/f offsets."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")

        compiler_args = common_dataclasses.CompilerArgs(
            platform_target=platform_target,
            logical_nc_config=1,
        )

        physical_tile_k = tile_k // INTERLEAVE_FACTOR
        physical_f = tile_f * INTERLEAVE_FACTOR

        rng = np.random.RandomState(42)
        tensor_fk = rng.uniform(-1, 1, (F, K)).astype(nl.bfloat16)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=pe_transpose_load_kernel,
            torch_ref=pe_transpose_torch_ref,
            kernel_input_generator=lambda _: {
                "tensor_fk": tensor_fk,
                "F": F,
                "K": K,
                "tile_f": tile_f,
                "tile_k": tile_k,
                "k_offset": k_offset,
                "f_offset": f_offset,
            },
            output_tensor_descriptor=lambda _: {
                "out": np.zeros((physical_tile_k, physical_f), dtype=nl.bfloat16),
            },
        )
        framework.run_test(test_config=None, compiler_args=compiler_args, rtol=0, atol=0)


# ============================================================================
# Test class — Indirect DMA
# ============================================================================


@pytest_marks(["bf16_pe_transpose_indirect"])
@pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN1, common_dataclasses.Platforms.TRN2])
class TestBF16PETransposeIndirect:
    """
    Correctness tests for load_tile_PE_swizzle_wrapX with indirect DMA (gather mode).
    """

    @pytest.mark.fast
    @pytest.mark.parametrize("F,K,tile_f,tile_k,k_offset", _TEST_SHAPES_INDIRECT)
    def test_pe_transpose_indirect_random_perm(self, test_manager, platform_target, F, K, tile_f, tile_k, k_offset):
        """Verify indirect DMA gathers the correct rows and swizzles them."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")

        compiler_args = common_dataclasses.CompilerArgs(
            platform_target=platform_target,
            logical_nc_config=1,
        )

        physical_tile_k = tile_k // INTERLEAVE_FACTOR
        physical_f = tile_f * INTERLEAVE_FACTOR

        rng = np.random.RandomState(42)
        tensor_fk = rng.uniform(-1, 1, (F, K)).astype(nl.bfloat16)

        NUM_SUB_TILES = tile_f // P_MAX
        row_indices_flat = rng.choice(F, size=tile_f, replace=False).astype(np.int32)
        row_indices = row_indices_flat.reshape(P_MAX, NUM_SUB_TILES, order='F')

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=pe_transpose_load_kernel,
            torch_ref=pe_transpose_torch_ref,
            kernel_input_generator=lambda _: {
                "tensor_fk": tensor_fk,
                "row_indices": row_indices,
                "F": F,
                "K": K,
                "tile_f": tile_f,
                "tile_k": tile_k,
                "k_offset": k_offset,
            },
            output_tensor_descriptor=lambda _: {
                "out": np.zeros((physical_tile_k, physical_f), dtype=nl.bfloat16),
            },
        )
        framework.run_test(test_config=None, compiler_args=compiler_args, rtol=0, atol=0)


# ============================================================================
# Test class — Indirect DMA with OOB skip
# ============================================================================


@pytest_marks(["bf16_pe_transpose_oob_skip"])
@pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN1, common_dataclasses.Platforms.TRN2])
class TestBF16PETransposeOOBSkip:
    """
    Correctness tests for load_tile_PE_swizzle_wrapX with oob_mode.skip.

    Validates that when skip_token=True and some row indices are -1 (out of bounds),
    those rows produce zeros in the swizzled output while valid rows are gathered
    and swizzled correctly.
    """

    @pytest.mark.fast
    @pytest.mark.parametrize("F,K,tile_f,tile_k,k_offset,num_padding_rows", _TEST_SHAPES_OOB_SKIP_TRAILING)
    def test_pe_transpose_oob_skip_trailing_padding(
        self, test_manager, platform_target, F, K, tile_f, tile_k, k_offset, num_padding_rows
    ):
        """Verify that trailing -1 indices produce zeros and valid rows are correct."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")

        compiler_args = common_dataclasses.CompilerArgs(
            platform_target=platform_target,
            logical_nc_config=1,
        )

        physical_tile_k = tile_k // INTERLEAVE_FACTOR
        physical_f = tile_f * INTERLEAVE_FACTOR

        rng = np.random.RandomState(42)
        tensor_fk = rng.uniform(-1, 1, (F, K)).astype(nl.bfloat16)

        num_valid = tile_f - num_padding_rows
        valid_indices = rng.choice(F, size=num_valid, replace=False).astype(np.int32)
        padding_indices = np.full(num_padding_rows, -1, dtype=np.int32)
        row_indices_flat = np.concatenate([valid_indices, padding_indices])

        NUM_SUB_TILES = tile_f // P_MAX
        row_indices = row_indices_flat.reshape(P_MAX, NUM_SUB_TILES, order='F')

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=pe_transpose_load_kernel,
            torch_ref=pe_transpose_torch_ref,
            kernel_input_generator=lambda _: {
                "tensor_fk": tensor_fk,
                "row_indices": row_indices,
                "F": F,
                "K": K,
                "tile_f": tile_f,
                "tile_k": tile_k,
                "k_offset": k_offset,
                "skip_token": True,
            },
            output_tensor_descriptor=lambda _: {
                "out": np.zeros((physical_tile_k, physical_f), dtype=nl.bfloat16),
            },
        )
        framework.run_test(test_config=None, compiler_args=compiler_args, rtol=0, atol=0)

    @pytest.mark.fast
    @pytest.mark.parametrize("F,K,tile_f,tile_k,k_offset,num_padding_rows", _TEST_SHAPES_OOB_SKIP_SCATTERED)
    def test_pe_transpose_oob_skip_scattered_padding(
        self, test_manager, platform_target, F, K, tile_f, tile_k, k_offset, num_padding_rows
    ):
        """Verify that randomly scattered -1 indices produce zeros while valid rows are correct."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")

        compiler_args = common_dataclasses.CompilerArgs(
            platform_target=platform_target,
            logical_nc_config=1,
        )

        physical_tile_k = tile_k // INTERLEAVE_FACTOR
        physical_f = tile_f * INTERLEAVE_FACTOR

        rng = np.random.RandomState(123)
        tensor_fk = rng.uniform(-1, 1, (F, K)).astype(nl.bfloat16)

        num_valid = tile_f - num_padding_rows
        valid_indices = rng.choice(F, size=num_valid, replace=False).astype(np.int32)
        row_indices_flat = np.full(tile_f, -1, dtype=np.int32)
        valid_positions = rng.choice(tile_f, size=num_valid, replace=False)
        row_indices_flat[valid_positions] = valid_indices

        NUM_SUB_TILES = tile_f // P_MAX
        row_indices = row_indices_flat.reshape(P_MAX, NUM_SUB_TILES, order='F')

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=pe_transpose_load_kernel,
            torch_ref=pe_transpose_torch_ref,
            kernel_input_generator=lambda _: {
                "tensor_fk": tensor_fk,
                "row_indices": row_indices,
                "F": F,
                "K": K,
                "tile_f": tile_f,
                "tile_k": tile_k,
                "k_offset": k_offset,
                "skip_token": True,
            },
            output_tensor_descriptor=lambda _: {
                "out": np.zeros((physical_tile_k, physical_f), dtype=nl.bfloat16),
            },
        )
        framework.run_test(test_config=None, compiler_args=compiler_args, rtol=0, atol=0)


# ============================================================================
# Test class — Multi K-tile (tiles_in_k > 1)
# ============================================================================


@pytest_marks(["bf16_pe_transpose_multi_k"])
@pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN1, common_dataclasses.Platforms.TRN2])
class TestBF16PETransposeMultiK:
    """
    Correctness tests for load_tile_PE_swizzle_wrapX with tiles_in_k > 1.

    Validates that loading multiple consecutive k-tiles into the same 4D SBUF
    buffer produces correct swizzled output for each k-tile slot.
    """

    @pytest.mark.fast
    @pytest.mark.parametrize("F,K,tile_f,tile_k,tiles_in_k,k_offset", _TEST_SHAPES_MULTI_K_DIRECT)
    def test_pe_transpose_multi_k_direct(
        self, test_manager, platform_target, F, K, tile_f, tile_k, tiles_in_k, k_offset
    ):
        """Verify multiple k-tiles are loaded and swizzled correctly (direct DMA)."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")

        compiler_args = common_dataclasses.CompilerArgs(
            platform_target=platform_target,
            logical_nc_config=1,
        )

        physical_tile_k = tile_k // INTERLEAVE_FACTOR
        physical_f = tile_f * INTERLEAVE_FACTOR

        rng = np.random.RandomState(42)
        tensor_fk = rng.uniform(-1, 1, (F, K)).astype(nl.bfloat16)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=pe_transpose_load_kernel,
            torch_ref=pe_transpose_torch_ref,
            kernel_input_generator=lambda _: {
                "tensor_fk": tensor_fk,
                "F": F,
                "K": K,
                "tile_f": tile_f,
                "tile_k": tile_k,
                "tiles_in_k": tiles_in_k,
                "k_offset": k_offset,
            },
            output_tensor_descriptor=lambda _: {
                "out": np.zeros((physical_tile_k * tiles_in_k, physical_f), dtype=nl.bfloat16),
            },
        )
        framework.run_test(test_config=None, compiler_args=compiler_args, rtol=0, atol=0)

    @pytest.mark.fast
    @pytest.mark.parametrize("F,K,tile_f,tile_k,tiles_in_k,k_offset", _TEST_SHAPES_MULTI_K_INDIRECT)
    def test_pe_transpose_multi_k_indirect(
        self, test_manager, platform_target, F, K, tile_f, tile_k, tiles_in_k, k_offset
    ):
        """Verify multiple k-tiles are loaded and swizzled correctly (indirect DMA)."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")

        compiler_args = common_dataclasses.CompilerArgs(
            platform_target=platform_target,
            logical_nc_config=1,
        )

        physical_tile_k = tile_k // INTERLEAVE_FACTOR
        physical_f = tile_f * INTERLEAVE_FACTOR

        rng = np.random.RandomState(42)
        tensor_fk = rng.uniform(-1, 1, (F, K)).astype(nl.bfloat16)

        NUM_SUB_TILES = tile_f // P_MAX
        row_indices_flat = rng.choice(F, size=tile_f, replace=False).astype(np.int32)
        row_indices = row_indices_flat.reshape(P_MAX, NUM_SUB_TILES, order='F')

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=pe_transpose_load_kernel,
            torch_ref=pe_transpose_torch_ref,
            kernel_input_generator=lambda _: {
                "tensor_fk": tensor_fk,
                "row_indices": row_indices,
                "F": F,
                "K": K,
                "tile_f": tile_f,
                "tile_k": tile_k,
                "tiles_in_k": tiles_in_k,
                "k_offset": k_offset,
            },
            output_tensor_descriptor=lambda _: {
                "out": np.zeros((physical_tile_k * tiles_in_k, physical_f), dtype=nl.bfloat16),
            },
        )
        framework.run_test(test_config=None, compiler_args=compiler_args, rtol=0, atol=0)

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "F,K,tile_f,tile_k,tiles_in_k,k_offset,store_f_offset", _TEST_SHAPES_MULTI_K_STORE_F_OFFSET
    )
    def test_pe_transpose_multi_k_store_f_offset(
        self, test_manager, platform_target, F, K, tile_f, tile_k, tiles_in_k, k_offset, store_f_offset
    ):
        """Verify non-zero store_f_offset places swizzled data at correct position."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")

        compiler_args = common_dataclasses.CompilerArgs(
            platform_target=platform_target,
            logical_nc_config=1,
        )

        physical_tile_k = tile_k // INTERLEAVE_FACTOR
        total_physical_f = (store_f_offset + tile_f) * INTERLEAVE_FACTOR

        rng = np.random.RandomState(42)
        tensor_fk = rng.uniform(-1, 1, (F, K)).astype(nl.bfloat16)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=pe_transpose_load_kernel,
            torch_ref=pe_transpose_torch_ref,
            kernel_input_generator=lambda _: {
                "tensor_fk": tensor_fk,
                "F": F,
                "K": K,
                "tile_f": tile_f,
                "tile_k": tile_k,
                "tiles_in_k": tiles_in_k,
                "k_offset": k_offset,
                "store_f_offset": store_f_offset,
            },
            output_tensor_descriptor=lambda _: {
                "out": np.zeros((physical_tile_k * tiles_in_k, total_physical_f), dtype=nl.bfloat16),
            },
        )
        framework.run_test(test_config=None, compiler_args=compiler_args, rtol=0, atol=0)


# ============================================================================
# K-by-F NKI Kernel
# ============================================================================


@nki.jit
@with_active_sbm
def pe_transpose_load_kf_kernel(
    tensor_kf,
    F: int,
    K: int,
    tile_f: int,
    tile_k: int,
    k_offset: int = 0,
    f_offset: int = 0,
):
    """Load a K-by-F tile using BF16 PE transpose and copy the swizzled result to HBM.

    The input tensor is [K, F] (K-by-F layout). The function loads and transposes
    it into the same swizzled [tile_k//4, tile_f*4] layout as the F-by-K path.

    Args:
        tensor_kf: Input [K, F] bf16 tensor in HBM.
        F: F dimension size.
        K: K dimension size.
        tile_f: Tile size in F dimension.
        tile_k: Tile size in K dimension.
        k_offset: Starting offset in K dimension (default 0).
        f_offset: Offset in F dimension (default 0).

    Returns:
        output: [physical_tile_k, physical_f] bf16 in HBM.
    """
    create_and_set_active_sbm()
    sbm = get_active_sbm()
    sbm.open_scope("PE_TRANSPOSE_LOAD_KF")

    physical_tile_k = tile_k // INTERLEAVE_FACTOR
    physical_f = tile_f * INTERLEAVE_FACTOR

    td = TensorDescriptor(
        data=tensor_kf,
        is_f_by_k=False,
    )

    data_sbuf = sbm.alloc_stack(
        shape=(physical_tile_k, 1, 1, physical_f),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )

    load_loc = TileLocation(
        tensor=td,
        tile_k=tile_k,
        tile_f=tile_f,
        k_offset=k_offset,
        f_offset=f_offset,
    )

    data_store_loc = TileLocation(
        tensor=TensorDescriptor(data=data_sbuf, is_swizzled=True, is_f_by_k=False),
        tile_k=physical_tile_k,
        tile_f=tile_f,
        k_offset=0,
        f_offset=0,
    )

    load_tile_PE_swizzle_wrapX(load_loc, data_store_loc=data_store_loc)

    output = nl.ndarray(
        (physical_tile_k, physical_f),
        dtype=nl.bfloat16,
        buffer=nl.shared_hbm,
    )
    nisa.dma_copy(dst=output, src=data_sbuf[:, 0, 0, :physical_f])

    sbm.close_scope()
    return output


# ============================================================================
# K-by-F Torch reference
# ============================================================================


def pe_transpose_kf_torch_ref(
    tensor_kf,
    F: int,
    K: int,
    tile_f: int,
    tile_k: int,
    k_offset: int = 0,
    f_offset: int = 0,
):
    """Golden reference for K-by-F PE transpose.

    Takes [K, F] input, extracts [tile_k, tile_f] slice, transposes to [tile_k, tile_f]
    (which is already K-major), then swizzles to interleaved layout.
    """
    # Extract tile from [K, F] and transpose to [tile_k, tile_f] (K on rows)
    tile_kf = tensor_kf[k_offset : k_offset + tile_k, f_offset : f_offset + tile_f]
    # tile_kf is already [tile_k, tile_f] = [K-rows, F-cols]
    # swizzle_tensor expects [K, F] input
    swizzled = matmul_utils.swizzle_tensor(tile_kf, TILE_P=tile_k)
    return {"out": swizzled.astype(nl.bfloat16)}


# ============================================================================
# K-by-F Test shapes
# ============================================================================

# (F, K, tile_f, tile_k, k_offset, f_offset)
_TEST_SHAPES_KF = [
    # tile_k=512, tile_f=128 (single sub-tile)
    (512, 512, 128, 512, 0, 0),
    (512, 1024, 128, 512, 512, 0),
    (1024, 512, 128, 512, 0, 128),
    (1024, 1024, 128, 512, 512, 256),
    # tile_k=512, tile_f=256 (two sub-tiles)
    (512, 512, 256, 512, 0, 0),
    (1024, 1024, 256, 512, 512, 256),
    # tile_k=512, tile_f=512 (four sub-tiles)
    (512, 512, 512, 512, 0, 0),
    (1024, 1024, 512, 512, 512, 0),
    # tile_k=256
    (512, 512, 128, 256, 0, 0),
    (512, 512, 256, 256, 0, 0),
    (1024, 1024, 128, 256, 256, 128),
    # tile_k=128
    (512, 512, 128, 128, 0, 0),
    (1024, 1024, 128, 128, 384, 256),
    # Non-square F (F not power-of-2, mimics backward kernel shapes)
    (1536, 2048, 128, 512, 0, 0),
    (1536, 2048, 128, 512, 512, 256),
    (1536, 2048, 128, 512, 1536, 0),
    (1536, 2048, 512, 512, 0, 0),
    (1536, 2048, 512, 512, 1536, 0),
    # Non-divisible K (K not divisible by 512, requires remainder tiles)
    # K=768: one 512-tile + one 256-remainder
    (512, 768, 128, 512, 0, 0),
    (512, 768, 128, 256, 512, 0),
    (512, 768, 512, 512, 0, 0),
    (512, 768, 512, 256, 512, 0),
    # K=640: one 512-tile + one 128-remainder
    (512, 640, 128, 512, 0, 0),
    (512, 640, 128, 128, 512, 0),
    (512, 640, 512, 512, 0, 0),
    (512, 640, 512, 128, 512, 0),
    # K=896: one 512-tile + one 256-tile + one 128-remainder
    (512, 896, 128, 512, 0, 0),
    (512, 896, 128, 256, 512, 0),
    (512, 896, 128, 128, 768, 0),
    # Large non-square
    (2048, 2048, 512, 512, 0, 0),
    (2048, 2048, 512, 512, 1536, 512),
]


# ============================================================================
# Test class — K-by-F option (a): 4x fast DMA direct transpose
# ============================================================================


@pytest_marks(["bf16_pe_transpose_kf"])
@pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN1, common_dataclasses.Platforms.TRN2])
class TestBF16PETransposeKbyF:
    """
    Correctness tests for load_tile_PE_swizzle_wrapX with K-by-F input layout,
    using option (a): 4x fast DMA direct transpose.
    """

    @pytest.mark.fast
    @pytest.mark.parametrize("F,K,tile_f,tile_k,k_offset,f_offset", _TEST_SHAPES_KF)
    def test_pe_transpose_kf_fast_dma(self, test_manager, platform_target, F, K, tile_f, tile_k, k_offset, f_offset):
        """Verify K-by-F PE transpose with 4x fast DMA transpose matches golden."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")

        compiler_args = common_dataclasses.CompilerArgs(
            platform_target=platform_target,
            logical_nc_config=1,
        )

        physical_tile_k = tile_k // INTERLEAVE_FACTOR
        physical_f = tile_f * INTERLEAVE_FACTOR

        rng = np.random.RandomState(42)
        tensor_kf = rng.uniform(-1, 1, (K, F)).astype(nl.bfloat16)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=pe_transpose_load_kf_kernel,
            torch_ref=pe_transpose_kf_torch_ref,
            kernel_input_generator=lambda _: {
                "tensor_kf": tensor_kf,
                "F": F,
                "K": K,
                "tile_f": tile_f,
                "tile_k": tile_k,
                "k_offset": k_offset,
                "f_offset": f_offset,
            },
            output_tensor_descriptor=lambda _: {
                "out": np.zeros((physical_tile_k, physical_f), dtype=nl.bfloat16),
            },
        )
        framework.run_test(test_config=None, compiler_args=compiler_args, rtol=0, atol=0)


# ============================================================================
# K-by-F Indirect NKI Kernel
# ============================================================================


@nki.jit
@with_active_sbm
def pe_transpose_load_kf_indirect_kernel(
    tensor_kf,
    F: int,
    K: int,
    tile_f: int,
    tile_k: int,
    k_offset: int = 0,
    row_indices=None,
    skip_token: bool = False,
):
    """Load a K-by-F tile using indirect DMA gather and copy swizzled result to HBM.

    The input tensor is [K, F] (e.g., [S, H] where S=tokens, H=hidden). The
    function gathers tile_f scattered K-rows (tokens) via vector_offset, reading
    tile_k contiguous F-elements from each row, then PE-swizzles into
    [tile_k//4, tile_f*4] layout.

    Args:
        tensor_kf: Input [K, F] bf16 tensor in HBM.
        F: F dimension size (second dim, e.g. hidden).
        K: K dimension size (first dim, e.g. sequence/tokens).
        tile_f: Tile size in F dimension (number of rows to gather).
        tile_k: Tile size in K dimension (number of elements per row to read).
        k_offset: Starting offset in F dimension (default 0).
        row_indices: [P_MAX, NUM_SUB_TILES] int32 tensor in HBM holding
            global K-row indices for indirect DMA gather.
        skip_token: If True, uses oob_mode.skip for DMA (default False).

    Returns:
        output: [physical_tile_k, physical_f] bf16 in HBM.
    """
    create_and_set_active_sbm()
    sbm = get_active_sbm()
    sbm.open_scope("PE_TRANSPOSE_LOAD_KF_INDIRECT")

    physical_tile_k = tile_k // INTERLEAVE_FACTOR
    physical_f = tile_f * INTERLEAVE_FACTOR

    td = TensorDescriptor(
        data=tensor_kf,
        is_f_by_k=False,
        skip_dma=SkipMode(skip_token=skip_token),
    )

    # Load indices to SBUF
    NUM_SUB_TILES = tile_f // P_MAX
    indices_sbuf = nl.ndarray((P_MAX, NUM_SUB_TILES), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=indices_sbuf, src=row_indices)

    data_sbuf = sbm.alloc_stack(
        shape=(physical_tile_k, 1, 1, physical_f),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )

    load_loc = TileLocation(
        tensor=td,
        tile_k=tile_k,
        tile_f=tile_f,
        k_offset=k_offset,
        f_offset=0,
        vector_offset=indices_sbuf,
    )

    data_store_loc = TileLocation(
        tensor=TensorDescriptor(data=data_sbuf, is_swizzled=True, is_f_by_k=False),
        tile_k=physical_tile_k,
        tile_f=tile_f,
        k_offset=0,
        f_offset=0,
    )

    load_tile_PE_swizzle_wrapX(load_loc, data_store_loc=data_store_loc)

    output = nl.ndarray(
        (physical_tile_k, physical_f),
        dtype=nl.bfloat16,
        buffer=nl.shared_hbm,
    )
    nisa.dma_copy(dst=output, src=data_sbuf[:, 0, 0, :physical_f])

    sbm.close_scope()
    return output


# ============================================================================
# K-by-F Indirect Torch reference
# ============================================================================


def pe_transpose_kf_indirect_torch_ref(
    tensor_kf,
    F: int,
    K: int,
    tile_f: int,
    tile_k: int,
    k_offset: int = 0,
    row_indices=None,
    skip_token: bool = False,
):
    """Golden reference for K-by-F indirect PE transpose.

    Takes [K, F] input (e.g., [S, H]), gathers tile_f scattered K-rows using
    row_indices, reads tile_k elements from each row starting at k_offset,
    then swizzles to interleaved layout.

    When skip_token=True, indices with value -1 are treated as zero rows
    (matching oob_mode.skip behavior).
    """
    # row_indices is (P_MAX, NUM_SUB_TILES) in column-major (Fortran) order
    indices = row_indices.flatten(order='F').astype(np.int64)

    if skip_token:
        # Gather with OOB handling: -1 indices produce zero rows
        tile_kf = np.zeros((tile_k, tile_f), dtype=tensor_kf.dtype)
        for i, idx in enumerate(indices):
            if idx >= 0:
                tile_kf[:, i] = tensor_kf[idx, k_offset : k_offset + tile_k]
    else:
        # Gather K-rows and slice F-elements: tensor_kf[indices, k_offset:k_offset+tile_k]
        gathered = tensor_kf[indices, k_offset : k_offset + tile_k]
        # gathered is [tile_f, tile_k], transpose to [tile_k, tile_f] for swizzle
        tile_kf = gathered.T.copy()

    # tile_kf is [tile_k, tile_f] — swizzle_tensor expects [K, F] input
    swizzled = matmul_utils.swizzle_tensor(tile_kf, TILE_P=tile_k)
    return {"out": swizzled.astype(nl.bfloat16)}


# ============================================================================
# K-by-F Indirect Test shapes
# ============================================================================

# (F, K, tile_f, tile_k, k_offset) — covers all tile_k values
# K = number of rows (tokens), F = row width (hidden), tile_f = rows to gather,
# tile_k = elements per row to read, k_offset = starting element within row
_TEST_SHAPES_KF_INDIRECT = [
    # tile_k=512
    (512, 512, 128, 512, 0),
    (1024, 512, 256, 512, 0),
    (1024, 1024, 128, 512, 512),
    (1024, 1024, 256, 512, 512),
    # tile_k=256
    (512, 512, 128, 256, 0),
    (1024, 512, 256, 256, 0),
    (1024, 1024, 128, 256, 256),
    # tile_k=128
    (512, 512, 128, 128, 0),
    (1024, 512, 256, 128, 0),
    (1024, 1024, 128, 128, 512),
    # tile_f=512 (four sub-tiles)
    (1024, 512, 512, 512, 0),
    (1024, 1024, 512, 256, 512),
]

# (F, K, tile_f, tile_k, k_offset, num_padding_rows) — OOB skip with -1 padding
_TEST_SHAPES_KF_INDIRECT_OOB_SKIP = [
    # tile_k=512
    (512, 512, 128, 512, 0, 64),
    (512, 512, 128, 512, 0, 128),
    (1024, 1024, 256, 512, 512, 64),
    # tile_k=256
    (512, 512, 128, 256, 0, 64),
    (1024, 1024, 128, 256, 256, 32),
    # tile_k=128
    (512, 512, 128, 128, 0, 64),
    (1024, 1024, 128, 128, 512, 32),
    # tile_f=256 multi-subtile
    (1024, 512, 256, 512, 0, 128),
    (1024, 1024, 256, 256, 512, 64),
]


# ============================================================================
# Test class — K-by-F Indirect DMA
# ============================================================================


@pytest_marks(["bf16_pe_transpose_kf_indirect"])
@pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN1, common_dataclasses.Platforms.TRN2])
class TestBF16PETransposeKbyFIndirect:
    """
    Correctness tests for load_tile_PE_swizzle_wrapX with K-by-F input layout
    and indirect DMA (gather mode for scattered K-rows/tokens).
    """

    @pytest.mark.fast
    @pytest.mark.parametrize("F,K,tile_f,tile_k,k_offset", _TEST_SHAPES_KF_INDIRECT)
    def test_pe_transpose_kf_indirect_random_perm(self, test_manager, platform_target, F, K, tile_f, tile_k, k_offset):
        """Verify K-by-F indirect DMA gathers the correct rows and swizzles them."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")

        compiler_args = common_dataclasses.CompilerArgs(
            platform_target=platform_target,
            logical_nc_config=1,
        )

        physical_tile_k = tile_k // INTERLEAVE_FACTOR
        physical_f = tile_f * INTERLEAVE_FACTOR

        rng = np.random.RandomState(42)
        tensor_kf = rng.uniform(-1, 1, (K, F)).astype(nl.bfloat16)

        NUM_SUB_TILES = tile_f // P_MAX
        row_indices_flat = rng.choice(K, size=tile_f, replace=False).astype(np.int32)
        row_indices = row_indices_flat.reshape(P_MAX, NUM_SUB_TILES, order='F')

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=pe_transpose_load_kf_indirect_kernel,
            torch_ref=pe_transpose_kf_indirect_torch_ref,
            kernel_input_generator=lambda _: {
                "tensor_kf": tensor_kf,
                "row_indices": row_indices,
                "F": F,
                "K": K,
                "tile_f": tile_f,
                "tile_k": tile_k,
                "k_offset": k_offset,
            },
            output_tensor_descriptor=lambda _: {
                "out": np.zeros((physical_tile_k, physical_f), dtype=nl.bfloat16),
            },
        )
        framework.run_test(test_config=None, compiler_args=compiler_args, rtol=0, atol=0)

    @pytest.mark.fast
    @pytest.mark.parametrize("F,K,tile_f,tile_k,k_offset,num_padding_rows", _TEST_SHAPES_KF_INDIRECT_OOB_SKIP)
    def test_pe_transpose_kf_indirect_oob_skip(
        self, test_manager, platform_target, F, K, tile_f, tile_k, k_offset, num_padding_rows
    ):
        """Verify K-by-F indirect DMA with oob_mode.skip handles -1 indices correctly."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")

        compiler_args = common_dataclasses.CompilerArgs(
            platform_target=platform_target,
            logical_nc_config=1,
        )

        physical_tile_k = tile_k // INTERLEAVE_FACTOR
        physical_f = tile_f * INTERLEAVE_FACTOR

        rng = np.random.RandomState(42)
        tensor_kf = rng.uniform(-1, 1, (K, F)).astype(nl.bfloat16)

        # Create indices with trailing -1 padding
        num_valid = tile_f - num_padding_rows
        valid_indices = rng.choice(K, size=num_valid, replace=False).astype(np.int32)
        padding_indices = np.full(num_padding_rows, -1, dtype=np.int32)
        row_indices_flat = np.concatenate([valid_indices, padding_indices])

        NUM_SUB_TILES = tile_f // P_MAX
        row_indices = row_indices_flat.reshape(P_MAX, NUM_SUB_TILES, order='F')

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=pe_transpose_load_kf_indirect_kernel,
            torch_ref=pe_transpose_kf_indirect_torch_ref,
            kernel_input_generator=lambda _: {
                "tensor_kf": tensor_kf,
                "row_indices": row_indices,
                "F": F,
                "K": K,
                "tile_f": tile_f,
                "tile_k": tile_k,
                "k_offset": k_offset,
                "skip_token": True,
            },
            output_tensor_descriptor=lambda _: {
                "out": np.zeros((physical_tile_k, physical_f), dtype=nl.bfloat16),
            },
        )
        framework.run_test(test_config=None, compiler_args=compiler_args, rtol=0, atol=0)
