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
"""
Integration tests for gen_mask_tkg standalone mask generation kernel.

This test suite validates the mask generation algorithm used by attention_tkg,
covering all code paths:
- Flat KV cache (block_len=0) with strided and non-strided MM1 layouts
- Block KV cache (block_len>0) with shuffled index generation
- LNC sharding configurations (lnc=1 and lnc=2)
- Sliding window attention (SWA) mask generation

The golden function uses gen_mask_tkg_torch_ref from gen_mask_tkg_torch.py.
"""

from test.integration.nkilib.core.attention.test_attention_tkg import build_active_attention_mask, build_swa_positions
from test.utils.common_dataclasses import CompilerArgs
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper
from typing import Optional, final

import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.core.attention.attention_tkg_utils import (
    AttnTKGConfig,
    is_s_prior_sharded,
    resize_cache_block_len_for_attention_tkg_kernel,
)
from nkilib_src.nkilib.core.attention.attention_tkg_utils import (
    is_batch_sharded as is_batch_sharded_fn,
)
from nkilib_src.nkilib.core.attention.gen_mask_tkg import gen_mask_tkg
from nkilib_src.nkilib.core.attention.gen_mask_tkg_torch import gen_mask_tkg_torch_ref
from nkilib_src.nkilib.core.utils.allocator import SbufManager
from nkilib_src.nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
from nkilib_src.nkilib.core.utils.logging import Logger

# Hardware constants
P_MAX = 128


def gen_mask_tkg_wrapper(
    pos_ids_hbm: nl.ndarray,
    mask_out_hbm: nl.ndarray,
    bs: int,
    q_head: int,
    s_active: int,
    is_s_prior_sharded: bool,
    s_prior_per_shard: int,
    start_pos_hbm: nl.ndarray = None,
    s_prior_offset: int = 0,
    block_len: int = 0,
    strided_mm1: bool = True,
    active_mask_hbm: nl.ndarray = None,
    is_batch_sharded: bool = False,
    batch_offset: int = 0,
) -> nl.ndarray:
    """Wrapper kernel that handles HBM↔SBUF transfers for testing gen_mask_tkg.

    For LNC=1: Output shape is [P_MAX, n_sprior_tile, bs, q_head, s_active]
    For LNC=2: Output shape is [2, P_MAX, n_sprior_tile_per_shard, bs, q_head, s_active]
    """
    _, lnc, shard_id = get_verified_program_sharding_info("gen_mask_wrapper", (0, 1))

    if lnc == 1:
        # Shape: [P_MAX, n_sprior_tile, bs, q_head, s_active]
        _, n_sprior_tile_per_shard, _, _, _ = mask_out_hbm.shape
    else:
        # Shape: [lnc, P_MAX, n_sprior_tile_per_shard, bs, q_head, s_active]
        _, _, n_sprior_tile_per_shard, _, _, _ = mask_out_hbm.shape

    sbm = SbufManager(
        0, P_MAX * n_sprior_tile_per_shard * bs * q_head * s_active * 8, Logger("gen_mask_wrapper"), use_auto_alloc=True
    )
    sbm.open_scope(name="gen_mask_wrapper")

    pos_ids_sbuf = sbm.alloc_stack((P_MAX, bs * s_active), dtype=pos_ids_hbm.dtype, buffer=nl.sbuf, name="pos_ids_sbuf")
    mask_out_sbuf = sbm.alloc_stack(
        (P_MAX, n_sprior_tile_per_shard, bs, q_head, s_active),
        dtype=mask_out_hbm.dtype,
        buffer=nl.sbuf,
        name="mask_out_sbuf",
    )

    nisa.dma_copy(dst=pos_ids_sbuf, src=pos_ids_hbm)

    start_pos_sbuf = None
    if start_pos_hbm is not None:
        start_pos_sbuf = sbm.alloc_stack(
            (P_MAX, bs * s_active), dtype=start_pos_hbm.dtype, buffer=nl.sbuf, name="start_pos_sbuf"
        )
        nisa.dma_copy(dst=start_pos_sbuf, src=start_pos_hbm)

    gen_mask_tkg(
        pos_ids=pos_ids_sbuf,
        mask_out=mask_out_sbuf,
        bs=bs,
        q_head=q_head,
        s_active=s_active,
        is_s_prior_sharded=is_s_prior_sharded,
        s_prior_per_shard=s_prior_per_shard,
        start_pos=start_pos_sbuf,
        s_prior_offset=s_prior_offset,
        block_len=block_len,
        strided_mm1=strided_mm1,
        active_mask=active_mask_hbm,
        sbm=sbm,
        is_batch_sharded=is_batch_sharded,
        batch_offset=batch_offset,
    )

    if lnc == 1:
        golden_mask = nl.ndarray(
            mask_out_sbuf.shape, dtype=mask_out_sbuf.dtype, buffer=nl.shared_hbm, name="golden_mask"
        )
        nisa.dma_copy(dst=golden_mask, src=mask_out_sbuf)
    else:
        golden_mask = nl.ndarray(mask_out_hbm.shape, dtype=mask_out_hbm.dtype, buffer=nl.shared_hbm, name="golden_mask")
        nisa.dma_copy(dst=golden_mask[shard_id, :, :, :, :, :], src=mask_out_sbuf)

    sbm.close_scope()

    return golden_mask


def gen_mask_tkg_torch_ref_adapter(
    pos_ids_hbm: torch.Tensor,
    mask_out_hbm: torch.Tensor,
    bs: int,
    q_head: int,
    s_active: int,
    is_s_prior_sharded: bool,
    s_prior_per_shard: int,
    start_pos_hbm: torch.Tensor = None,
    s_prior_offset: int = 0,
    block_len: int = 0,
    strided_mm1: bool = True,
    active_mask_hbm: torch.Tensor = None,
    is_batch_sharded: bool = False,
    batch_offset: int = 0,
) -> dict[str, torch.Tensor]:
    """Torch ref adapter matching gen_mask_tkg_wrapper signature.

    Bridges the LncSubscriptable gen_mask_tkg_torch_ref to the flat-call
    interface expected by torch_ref_wrapper / UnitTestFramework.
    """
    if mask_out_hbm.dim() == 5:
        lnc = 1
        _, n_sprior_tile, _, _, _ = mask_out_hbm.shape
    else:
        lnc = mask_out_hbm.shape[0]
        _, _, n_sprior_tile, _, _, _ = mask_out_hbm.shape

    pos_ids = pos_ids_hbm.float()
    active_mask = active_mask_hbm.float() if active_mask_hbm is not None else None
    start_pos = start_pos_hbm.float() if start_pos_hbm is not None else None

    if lnc == 1:
        mask_out = torch.zeros((P_MAX, n_sprior_tile, bs, q_head, s_active), dtype=torch.float32)
        gen_mask_tkg_torch_ref.shard_id = 0
        gen_mask_tkg_torch_ref[lnc](
            pos_ids=pos_ids,
            mask_out=mask_out,
            bs=bs,
            q_head=q_head,
            s_active=s_active,
            is_s_prior_sharded=is_s_prior_sharded,
            s_prior_per_shard=s_prior_per_shard,
            start_pos=start_pos,
            s_prior_offset=s_prior_offset,
            block_len=block_len,
            strided_mm1=strided_mm1,
            active_mask=active_mask,
            is_batch_sharded=is_batch_sharded,
            batch_offset=batch_offset,
        )
        return {"golden_mask": mask_out}
    else:
        result = torch.zeros((lnc, P_MAX, n_sprior_tile, bs, q_head, s_active), dtype=torch.float32)
        for shard_idx in range(lnc):
            mask_out = torch.zeros((P_MAX, n_sprior_tile, bs, q_head, s_active), dtype=torch.float32)
            gen_mask_tkg_torch_ref.shard_id = shard_idx
            gen_mask_tkg_torch_ref[lnc](
                pos_ids=pos_ids,
                mask_out=mask_out,
                bs=bs,
                q_head=q_head,
                s_active=s_active,
                is_s_prior_sharded=is_s_prior_sharded,
                s_prior_per_shard=s_prior_per_shard,
                start_pos=start_pos,
                s_prior_offset=s_prior_offset,
                block_len=block_len,
                strided_mm1=strided_mm1,
                active_mask=active_mask,
                is_batch_sharded=is_batch_sharded,
            )
            result[shard_idx] = mask_out
        return {"golden_mask": result}


def create_pos_ids_tensor(
    cache_lens: np.ndarray,
    batch: int,
    s_active: int,
    dtype=np.float32,
    per_active: bool = False,
) -> np.ndarray:
    """
    Create pos_ids tensor from cache lengths.

    Args:
        cache_lens: [batch] array of cache lengths per batch.
        batch: Batch size.
        s_active: Active sequence length.
        dtype: Output dtype.
        per_active: If True, each s_active slot gets cache_lens[b] + i
            (for SWA). If False, all slots get the same cache_lens[b] value.

    Returns:
        pos_ids: [P_MAX, batch * s_active] tensor where all partitions get
                 the same cache_len value (broadcasted).
    """
    if per_active:
        row = np.zeros(batch * s_active, dtype=dtype)
        for b in range(batch):
            for i in range(s_active):
                row[b * s_active + i] = cache_lens[b] + i
    else:
        row = np.repeat(cache_lens, s_active).astype(dtype)  # [batch * s_active]
    return np.broadcast_to(row[np.newaxis, :], (P_MAX, batch * s_active)).copy()


def generate_gen_mask_inputs(
    batch: int,
    q_head: int,
    s_ctx: int,
    s_active: int,
    block_len: int,
    lnc: int,
    strided_mm1: bool,
    dtype=np.float32,
    s_prior_offset: int = 0,
    fa_tile_size: int = 0,
    sliding_window: int = 0,
    include_active_mask: bool = False,
    batch_offset: int = 0,
    bs_full: Optional[int] = None,
):
    """Build kernel inputs for gen_mask_tkg test, compatible with UnitTestFramework.

    Returns dict with keys matching gen_mask_tkg_wrapper signature.
    mask_out_hbm uses .must_alias_input suffix for mutable output.
    """
    cfg = AttnTKGConfig(bs=batch, q_head=q_head, s_active=s_active, curr_sprior=s_ctx)
    sprior_sharded = is_s_prior_sharded(cfg, P_MAX) if lnc > 1 else False

    if sprior_sharded:
        s_prior_per_shard = s_ctx // lnc
    else:
        s_prior_per_shard = s_ctx

    if fa_tile_size > 0:
        n_sprior_tile_per_shard = fa_tile_size // P_MAX
        assert (
            s_prior_offset + fa_tile_size <= s_prior_per_shard
        ), f"FA tile (offset={s_prior_offset}, size={fa_tile_size}) exceeds s_prior_per_shard ({s_prior_per_shard})"
    else:
        n_sprior_tile_per_shard = s_prior_per_shard // P_MAX

    adjusted_block_len = block_len
    if block_len > 0:
        num_blocks_total = s_ctx // block_len
        adjusted_block_len, _ = resize_cache_block_len_for_attention_tkg_kernel(num_blocks_total, block_len, lnc, P_MAX)
        if fa_tile_size > 0:
            fold_size = adjusted_block_len * P_MAX
            assert (
                fa_tile_size % fold_size == 0
            ), f"fa_tile_size ({fa_tile_size}) must be divisible by block_len * P_MAX ({fold_size})"
            assert (
                s_prior_offset % fold_size == 0
            ), f"s_prior_offset ({s_prior_offset}) must be divisible by block_len * P_MAX ({fold_size})"

    np.random.seed(42)
    if sliding_window > 0:
        # SWA circular buffer invariant: pos_id must be < s_ctx - s_active
        cache_lens = np.random.randint(1, s_ctx - s_active, size=(batch,)).astype(np.int32)
    else:
        cache_lens = np.random.randint(1, s_ctx, size=(batch,)).astype(np.int32)

    if sliding_window > 0:
        pos_ids_data = create_pos_ids_tensor(cache_lens, batch, s_active, dtype, per_active=True)
        pos_id_2d = cache_lens.reshape(batch, 1)
        start_pos_ids, _ = build_swa_positions(
            pos_id=pos_id_2d,
            bs=batch,
            s_active=s_active,
            sliding_window=sliding_window,
            cache_len=s_ctx,
            block_len=block_len,
        )
        start_pos_flat = start_pos_ids.reshape(batch * s_active).astype(dtype)
        start_pos_data = np.broadcast_to(start_pos_flat[np.newaxis, :], (P_MAX, batch * s_active)).copy()
    else:
        pos_ids_data = create_pos_ids_tensor(cache_lens, batch, s_active, dtype)
        start_pos_data = None

    if lnc == 1:
        mask_out_data = np.zeros((P_MAX, n_sprior_tile_per_shard, batch, q_head, s_active), dtype=dtype)
    else:
        mask_out_data = np.zeros((lnc, P_MAX, n_sprior_tile_per_shard, batch, q_head, s_active), dtype=dtype)

    result = {
        "pos_ids_hbm": pos_ids_data,
        "mask_out_hbm.must_alias_input": mask_out_data,
        "bs": batch,
        "q_head": q_head,
        "s_active": s_active,
        "is_s_prior_sharded": sprior_sharded,
        "s_prior_per_shard": s_prior_per_shard,
        "s_prior_offset": s_prior_offset,
        "block_len": adjusted_block_len,
        "strided_mm1": strided_mm1,
    }

    if start_pos_data is not None:
        result["start_pos_hbm"] = start_pos_data

    if include_active_mask:
        batch_sharded = is_batch_sharded_fn(cfg, P_MAX) if lnc > 1 else False

        if fa_tile_size > 0:
            tile_end = s_prior_offset + fa_tile_size
            active_region_start = s_prior_per_shard - s_active
            assert tile_end >= s_prior_per_shard and s_prior_offset <= active_region_start, (
                f"FA tile (offset={s_prior_offset}, end={tile_end}) must completely include active region "
                f"[{active_region_start}, {s_prior_per_shard}) when active_mask is provided"
            )

        bs_full_computed = batch * lnc if batch_sharded else batch
        if bs_full is not None:
            bs_full_computed = bs_full
        active_mask = (
            build_active_attention_mask(batch=bs_full_computed, num_heads=q_head, s_active=s_active, transposed=True)
            .numpy()
            .astype(dtype)
        )
        result["active_mask_hbm"] = active_mask
        result["is_batch_sharded"] = batch_sharded
        if batch_offset > 0:
            result["batch_offset"] = batch_offset

    return result


@pytest_test_metadata(
    name="Gen Mask TKG",
    pytest_marks=["attention", "tkg", "subkernel"],
)
@final
class TestGenMaskTkg:
    """
    Integration test suite for gen_mask_tkg kernel.

    Tests run the actual NKI kernel on Neuron hardware and compare output
    against the torch reference implementation (gen_mask_tkg_torch_ref).
    """

    @staticmethod
    def _run_test(test_manager, lnc, input_generator):
        def output_tensors(kernel_input):
            return {"golden_mask": kernel_input["mask_out_hbm.must_alias_input"]}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=gen_mask_tkg_wrapper,
            torch_ref=torch_ref_wrapper(gen_mask_tkg_torch_ref_adapter),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(test_config=None, compiler_args=CompilerArgs(logical_nc_config=lnc), rtol=0, atol=0)

    # ============================================================================
    # FLAT KV CACHE TESTS (block_len = 0) - STRIDED MM1
    # ============================================================================

    # fmt: off
    # Test parameters for flat KV cache with strided MM1 layout
    # (batch, q_head, s_ctx, s_active, sliding_window, s_prior_offset, fa_tile_size, lnc)
    # fa_tile_size=0 means full s_prior, >0 means FA tile mode
    flat_kv_strided_test_params = "batch, q_head, s_ctx, s_active, sliding_window, s_prior_offset, fa_tile_size, lnc"
    flat_kv_strided_test_perms = [
        # LNC=1 basic tests (full s_prior)
        (4, 1, 256, 1, 0, 0, 0, 1),   # Minimal s_ctx
        (4, 1, 512, 1, 0, 0, 0, 1),   # Small s_ctx
        (4, 1, 1024, 5, 0, 0, 0, 1),  # With multiple active tokens
        (4, 2, 2048, 7, 0, 0, 0, 1),  # Multiple heads
        (4, 1, 4096, 5, 0, 0, 0, 1),  # Larger s_ctx
        # LNC=1 FA tile tests
        (4, 1, 1024, 5, 0, 0, 256, 1),    # First tile
        (4, 1, 1024, 5, 0, 256, 256, 1),  # Second tile
        (4, 1, 1024, 5, 0, 512, 256, 1),  # Third tile
        (4, 1, 1024, 5, 0, 768, 256, 1),  # Last tile
        # LNC=2 tests - matching test_attention_tkg.py configurations
        (4, 2, 16384, 7, 0, 0, 0, 2),     # Match attention_tkg test
        (4, 1, 4096, 5, 0, 0, 0, 2),      # Medium s_ctx
        (4, 1, 4096, 5, 0, 768, 256, 2),  # FA test
        (8, 8, 4096, 5, 0, 1536, 256, 2),  # Batch sharding
    ]
    # fmt: on

    @pytest.mark.fast
    @pytest.mark.parametrize(flat_kv_strided_test_params, flat_kv_strided_test_perms)
    def test_flat_kv_strided_mask_generation(
        self,
        test_manager: Orchestrator,
        batch: int,
        q_head: int,
        s_ctx: int,
        s_active: int,
        sliding_window: int,
        s_prior_offset: int,
        fa_tile_size: int,
        lnc: int,
    ):
        """
        Test flat KV cache mask generation with strided MM1 layout (block_len=0).
        """

        def input_generator(test_config, input_tensor_def=None):
            return generate_gen_mask_inputs(
                batch=batch,
                q_head=q_head,
                s_ctx=s_ctx,
                s_active=s_active,
                block_len=0,
                lnc=lnc,
                strided_mm1=True,
                s_prior_offset=s_prior_offset,
                fa_tile_size=fa_tile_size,
                sliding_window=sliding_window,
            )

        self._run_test(test_manager, lnc, input_generator)

    # ============================================================================
    # FLAT KV CACHE TESTS (block_len = 0) - NON-STRIDED MM1
    # ============================================================================

    # fmt: off
    # Test parameters for flat KV cache with non-strided MM1 layout
    flat_kv_nonstrided_test_params = "batch, q_head, s_ctx, s_active, sliding_window, s_prior_offset, fa_tile_size, lnc"
    flat_kv_nonstrided_test_perms = [
        # LNC=1 basic tests (full s_prior)
        (4, 1, 256, 1, 0, 0, 0, 1),   # Minimal s_ctx
        (4, 1, 512, 1, 0, 0, 0, 1),   # Small s_ctx
        (4, 1, 1024, 5, 0, 0, 0, 1),  # With multiple active tokens
        # LNC=1 FA tile tests
        (4, 1, 1024, 5, 0, 256, 256, 1),  # FA tile with offset
        # LNC=2 tests
        (4, 1, 4096, 5, 0, 0, 0, 2),  # Medium s_ctx with LNC=2
        (4, 1, 4096, 5, 0, 512, 512, 2),  # FA tile with offset
        (8, 8, 4096, 5, 0, 1536, 512, 2),  # Batch sharding
    ]
    # fmt: on

    @pytest.mark.parametrize(flat_kv_nonstrided_test_params, flat_kv_nonstrided_test_perms)
    def test_flat_kv_nonstrided_mask_generation(
        self,
        test_manager: Orchestrator,
        batch: int,
        q_head: int,
        s_ctx: int,
        s_active: int,
        sliding_window: int,
        s_prior_offset: int,
        fa_tile_size: int,
        lnc: int,
    ):
        """
        Test flat KV cache mask generation with non-strided MM1 layout (block_len=0, strided_mm1=False).
        """

        def input_generator(test_config, input_tensor_def=None):
            return generate_gen_mask_inputs(
                batch=batch,
                q_head=q_head,
                s_ctx=s_ctx,
                s_active=s_active,
                block_len=0,
                lnc=lnc,
                strided_mm1=False,
                s_prior_offset=s_prior_offset,
                fa_tile_size=fa_tile_size,
                sliding_window=sliding_window,
            )

        self._run_test(test_manager, lnc, input_generator)

    # ============================================================================
    # BLOCK KV CACHE TESTS (block_len > 0)
    # ============================================================================

    # fmt: off
    # Test parameters for block KV cache
    block_kv_test_params = "batch, q_head, s_ctx, s_active, block_len, sliding_window, s_prior_offset, fa_tile_size, lnc"
    block_kv_test_perms = [
        # LNC=1 block KV tests (full s_prior)
        (4, 1, 256, 5, 16, 0, 0, 0, 1),
        (4, 1, 512, 5, 16, 0, 0, 0, 1),
        (4, 1, 1024, 5, 16, 0, 0, 0, 1),
        (4, 1, 2048, 5, 32, 0, 0, 0, 1),  # Larger block_len
        # LNC=1 block KV FA tile tests
        (4, 1, 4096, 5, 16, 0, 0, 2048, 1),    # First tile
        (4, 1, 4096, 5, 16, 0, 2048, 2048, 1), # Second tile
        (4, 1, 10240, 5, 16, 0, 8192, 2048, 1),   # Second tile (small last tile)
        # LNC=2 block KV tests - matching test_attention_tkg.py configurations
        (4, 1, 8192, 5, 16, 0, 0, 0, 2),
        (4, 1, 4096, 5, 16, 0, 0, 0, 2),
        (4, 1, 20480, 5, 16, 0, 8192, 2048, 2),  # FA tile with offset
        (8, 8, 10240, 5, 16, 0, 8192, 2048, 2),  # Batch sharding
    ]
    # fmt: on

    @pytest.mark.parametrize(block_kv_test_params, block_kv_test_perms)
    def test_block_kv_mask_generation(
        self,
        test_manager: Orchestrator,
        batch: int,
        q_head: int,
        s_ctx: int,
        s_active: int,
        block_len: int,
        sliding_window: int,
        s_prior_offset: int,
        fa_tile_size: int,
        lnc: int,
    ):
        """
        Test block KV cache mask generation (block_len>0).

        Validates the shuffled index mask generation that matches the K cache
        block layout used by the attention kernel.
        """

        def input_generator(test_config, input_tensor_def=None):
            return generate_gen_mask_inputs(
                batch=batch,
                q_head=q_head,
                s_ctx=s_ctx,
                s_active=s_active,
                block_len=block_len,
                lnc=lnc,
                strided_mm1=False,
                s_prior_offset=s_prior_offset,
                fa_tile_size=fa_tile_size,
                sliding_window=sliding_window,
            )

        self._run_test(test_manager, lnc, input_generator)

    # ============================================================================
    # SWA (SLIDING WINDOW ATTENTION) TESTS
    # ============================================================================

    # fmt: off
    # Test parameters for SWA mask generation
    # (batch, q_head, s_ctx, s_active, block_len, sliding_window, strided_mm1, s_prior_offset, fa_tile_size, lnc)
    swa_test_params = "batch, q_head, s_ctx, s_active, block_len, sliding_window, strided_mm1, s_prior_offset, fa_tile_size, lnc"
    swa_test_perms = [
        # Flat KV strided, LNC=1
        (4, 1, 1024, 5, 0,  64, True, 0, 0, 1),
        (4, 1, 1024, 5, 0, 128, True, 0, 0, 1),
        (4, 1, 1024, 5, 0, 256, True, 0, 0, 1),
        (4, 1, 1024, 5, 0, 512, True, 0, 0, 1),
        (4, 2, 2048, 7, 0, 128, True, 0, 0, 1),
        # Flat KV strided, LNC=2
        (4, 1, 4096, 5, 0, 128, True, 0, 0, 2),
        (4, 2, 4096, 7, 0, 256, True, 0, 0, 2),
        # Flat KV non-strided, LNC=1
        (4, 1, 1024, 5, 0, 128, False, 0, 0, 1),
        (4, 1, 1024, 5, 0, 256, False, 0, 0, 1),
        # Flat KV non-strided, LNC=2
        (4, 1, 4096, 5, 0, 128, False, 0, 0, 2),
        # Block KV, LNC=1
        (4, 1, 1024, 5, 16, 128, False, 0, 0, 1),
        (4, 1, 2048, 5, 16, 256, False, 0, 0, 1),
        (4, 1, 2048, 5, 32, 512, False, 0, 0, 1),
        # Block KV, LNC=2
        (4, 1, 4096, 5, 16, 128, False, 0, 0, 2),
        (4, 1, 8192, 5, 16, 256, False, 0, 0, 2),
        # SWA + FA tiling (flat KV strided)
        (4, 1, 1024, 5, 0, 128, True, 0, 256, 1),    # First FA tile
        (4, 1, 1024, 5, 0, 128, True, 512, 256, 1),   # Middle FA tile
        (4, 1, 1024, 5, 0, 128, True, 768, 256, 1),   # Last FA tile
        # SWA + FA tiling (block KV)
        (4, 1, 4096, 5, 16, 128, False, 0, 2048, 1),     # First FA tile, block KV
        (4, 1, 4096, 5, 16, 128, False, 2048, 2048, 1),  # Second FA tile, block KV
        # SWA + FA tiling + LNC=2
        (4, 1, 4096, 5, 0, 128, True, 0, 256, 2),     # FA tile, sprior-sharded
        # SWA + s_active=1
        (4, 1, 1024, 1, 0, 128, True, 0, 0, 1),
        (4, 1, 2048, 1, 16, 256, False, 0, 0, 1),
        # SWA + LNC=2 batch-sharded
        (8, 8, 4096, 5, 0, 128, True, 0, 0, 2),       # BQS=320>128, batch-sharded
        (8, 8, 4096, 5, 16, 128, False, 0, 0, 2),      # BQS=320>128, batch-sharded, block KV
    ]
    # fmt: on

    @pytest.mark.parametrize(swa_test_params, swa_test_perms)
    def test_swa_mask_generation(
        self,
        test_manager: Orchestrator,
        batch: int,
        q_head: int,
        s_ctx: int,
        s_active: int,
        block_len: int,
        sliding_window: int,
        strided_mm1: bool,
        s_prior_offset: int,
        fa_tile_size: int,
        lnc: int,
    ):
        """
        Test SWA (sliding window attention) mask generation.

        Validates per-query windowed masks for both flat and block KV layouts.
        """

        def input_generator(test_config, input_tensor_def=None):
            return generate_gen_mask_inputs(
                batch=batch,
                q_head=q_head,
                s_ctx=s_ctx,
                s_active=s_active,
                block_len=block_len,
                lnc=lnc,
                strided_mm1=strided_mm1,
                s_prior_offset=s_prior_offset,
                fa_tile_size=fa_tile_size,
                sliding_window=sliding_window,
            )

        self._run_test(test_manager, lnc, input_generator)

    # ============================================================================
    # ACTIVE MASK TESTS - Testing _load_active_mask code path
    # ============================================================================

    # fmt: off
    # Test parameters for active_mask tests (flat KV)
    # These test the _load_active_mask code path for flat KV cache
    active_mask_test_params = "batch, q_head, s_ctx, s_active, strided_mm1, sliding_window, s_prior_offset, fa_tile_size, lnc, batch_offset, bs_full"
    active_mask_test_perms = [
        # Strided MM1 with active_mask (LNC=1, full s_prior)
        (4, 1, 256, 5, True, 0, 0, 0, 1, 0, None),
        (4, 2, 512, 5, True, 0, 0, 0, 1, 0, None),
        # Non-strided MM1 with active_mask (LNC=1, full s_prior)
        (4, 1, 256, 5, False, 0, 0, 0, 1, 0, None),
        (4, 2, 512, 5, False, 0, 0, 0, 1, 0, None),
        # s_active=1 edge cases with active_mask (tests the expand_dim fix in _load_active_mask)
        (4, 1, 256, 1, True, 0, 0, 0, 1, 0, None),   # Strided MM1, s_active=1
        (4, 1, 256, 1, False, 0, 0, 0, 1, 0, None),  # Non-strided MM1, s_active=1
        # LNC=1 FA tile tests with active_mask (tile must include active region)
        (4, 1, 1024, 5, True, 0, 768, 256, 1, 0, None),   # Last tile (where active mask applies)
        (4, 1, 1024, 5, False, 0, 768, 256, 1, 0, None),  # Last tile (where active mask applies)
        # LNC=2 sprior-sharded with active_mask (flat KV doesn't support batch-sharded active_mask)
        (4, 2, 4096, 5, True, 0, 0, 0, 2, 0, None),       # Sprior sharded: 4*2*5=40 <= 128

        # Batch-sharded LNC=2 with strided MM1 and FA tiling
        (80, 8, 256, 8, True, 0, 0, 256, 2, 0, None),

        # Strided MM1 with load1_nrows = 0 (s_active % n_sprior_tile = 0)
        # These test the edge case where only load2 path is used in _load_active_mask
        # s_active=8, s_ctx=256 -> n_sprior_tile=2, load1_nrows=8%2=0
        (4, 1, 256, 8, True, 0, 0, 0, 1, 0, None),    # load1_nrows=0: 8%2=0
        (4, 2, 256, 4, True, 0, 0, 0, 1, 0, None),    # load1_nrows=0: 4%2=0
        (80, 8, 256, 8, True, 0, 0, 0, 1, 0, None),   # Matches failing attention_tkg test: bs=80, s_a=8, s_p=256

        # Strided MM1 with load1_nrows > 0 (s_active % n_sprior_tile != 0)
        # These test the edge case where both load1 and load2 paths are used
        # s_active=7, s_ctx=256 -> n_sprior_tile=2, load1_nrows=7%2=1, load2_nrows=6
        (4, 1, 256, 7, True, 0, 0, 0, 1, 0, None),    # load1_nrows=1, load2_nrows=6
        (80, 8, 256, 7, True, 0, 0, 0, 1, 0, None),   # Large batch, odd s_active

        # Strided MM1 with s_active < n_sprior_tile (load2_nrows = 0, only load1 used)
        # s_active=1, s_ctx=256 -> n_sprior_tile=2, load1_nrows=1%2=1, load2_nrows=0
        (4, 1, 256, 1, True, 0, 0, 0, 1, 0, None),    # Minimal s_active, only load1 path
        (80, 8, 256, 1, True, 0, 0, 0, 1, 0, None),   # Large batch, minimal s_active

        # Batch-sharded LNC=2 with load1_nrows = 0 (exercises the fixed DMA stride path)
        # bs*q_head*s_active > 128 -> batch-sharded, s_active % n_sprior_tile = 0
        (80, 8, 256, 4, True, 0, 0, 256, 2, 0, None), # bs_full=160, load1_nrows=4%2=0, only load2
        (32, 8, 256, 8, True, 0, 0, 256, 2, 0, None), # bs_full=64, BQS=2048>128, load1_nrows=8%2=0

        # Batch-sharded LNC=2 with load1_nrows > 0 (both DMA paths with stride fix)
        (80, 8, 256, 7, True, 0, 0, 256, 2, 0, None), # bs_full=160, load1_nrows=7%2=1, load2_nrows=6
        (32, 8, 256, 5, True, 0, 0, 256, 2, 0, None), # bs_full=64, BQS=1280>128, load1_nrows=5%2=1

        # Batch-sharded LNC=2 with s_active=1 (only load1 path with stride fix)
        (80, 8, 256, 1, True, 0, 0, 256, 2, 0, None), # bs_full=160, load1_nrows=1, load2_nrows=0

        # Non-strided MM1 with batch-sharded LNC=2 (tests the TensorView slice path)
        (80, 8, 256, 8, False, 0, 0, 256, 2, 0, None), # Batch-sharded, non-strided, BQS=5120>128
        (32, 8, 256, 5, False, 0, 0, 256, 2, 0, None), # Batch-sharded, non-strided, BQS=1280>128

        # LNC=2 sprior-sharded with strided MM1 and load1_nrows = 0
        # bs*q_head*s_active <= 128 -> sprior-sharded
        (4, 2, 4096, 8, True, 0, 0, 0, 2, 0, None),   # BQS=64<=128, sprior-sharded, load1_nrows=8%16=8!=0
        (4, 1, 4096, 16, True, 0, 0, 0, 2, 0, None),  # BQS=64<=128, sprior-sharded, load1_nrows=16%16=0

        # FA tiling with batch-sharded LNC=2 (different tile sizes)
        # batch-sharded: s_prior_per_shard = s_ctx = 512, active region at [504, 512)
        # FA tile must cover the active region, so offset must be at end
        (80, 8, 512, 8, True, 0, 256, 256, 2, 0, None), # FA tile at [256, 512), covers active region [504, 512)

        # Boundary: BQS exactly at P_MAX threshold for sharding decision
        # bs*q_head*s_active = 128 -> NOT batch-sharded (needs > 128), sprior-sharded
        (16, 1, 4096, 8, True, 0, 0, 0, 2, 0, None),  # BQS=128, sprior-sharded
        # bs*q_head*s_active = 130 -> batch-sharded (> 128)
        (26, 1, 256, 5, True, 0, 0, 256, 2, 0, None), # BQS=130>128, batch-sharded

        # SWA + active_mask (flat KV strided)
        (4, 1, 1024, 5, True, 128, 768, 256, 1, 0, None),   # SWA + active_mask, last FA tile
        (4, 2, 1024, 5, True, 256, 0, 0, 1, 0, None),        # SWA + active_mask, full s_prior
        # SWA + active_mask (flat KV non-strided)
        (4, 1, 1024, 5, False, 128, 768, 256, 1, 0, None),
        # SWA + active_mask + LNC=2 sprior-sharded
        (4, 2, 4096, 5, True, 128, 0, 0, 2, 0, None),        # BQS=40<=128, sprior-sharded

        # batch_offset > 0 (batch tiling): tile_bs=batch, offset into larger active_mask
        (2, 1, 256, 5, True,  0, 0, 0, 1, 2, 4),   # Strided, second tile of bs_full=4
        (3, 2, 512, 5, True,  0, 0, 0, 1, 3, 7),   # Strided, odd total batch, second tile
        (1, 1, 256, 1, True,  0, 0, 0, 1, 3, 4),   # Strided, last single-batch tile
        (2, 1, 256, 5, False, 0, 0, 0, 1, 2, 4),   # Non-strided, second tile
        # batch_offset > 0, LNC=2 sprior-sharded (BQS <= 128)
        (2, 2, 4096, 5, True, 0, 0, 0, 2, 2, 4),   # Sprior-sharded, second tile
        # batch_offset > 0, LNC=2 batch-sharded (BQS > 128)
        (40, 8, 256, 8, True, 0, 0, 256, 2, 40, 160),  # Batch-sharded, second tile
        # batch_offset > 0 + s_prior_offset > 0 (FA tiling + batch tiling combined)
        (2, 1, 1024, 5, True,  0, 768, 256, 1, 2, 4),  # Strided, FA last tile + batch offset
        (2, 2, 4096, 5, True, 0, 1792, 256, 2, 2, 4),  # Sprior-sharded LNC=2, FA last tile + batch offset
        # batch_offset > 0 + SWA (sliding window + batch tiling combined)
        (2, 1, 1024, 5, True, 128, 768, 256, 1, 2, 4),   # SWA + batch offset, strided, FA last tile
        (2, 1, 1024, 5, False, 128, 768, 256, 1, 2, 4),   # SWA + batch offset, non-strided, FA last tile
        (2, 2, 4096, 5, True, 256, 0, 0, 2, 2, 4),        # SWA + batch offset, sprior-sharded LNC=2
    ]
    # fmt: on

    @pytest.mark.parametrize(active_mask_test_params, active_mask_test_perms)
    def test_flat_kv_with_active_mask(
        self,
        test_manager: Orchestrator,
        batch: int,
        q_head: int,
        s_ctx: int,
        s_active: int,
        strided_mm1: bool,
        sliding_window: int,
        s_prior_offset: int,
        fa_tile_size: int,
        lnc: int,
        batch_offset: int,
        bs_full: Optional[int],
    ):
        """
        Test flat KV cache mask generation with active_mask provided.

        This tests the _load_active_mask code path which loads the causal
        active mask onto the last section of the prior mask.
        """

        def input_generator(test_config, input_tensor_def=None):
            return generate_gen_mask_inputs(
                batch=batch,
                q_head=q_head,
                s_ctx=s_ctx,
                s_active=s_active,
                block_len=0,
                lnc=lnc,
                strided_mm1=strided_mm1,
                s_prior_offset=s_prior_offset,
                fa_tile_size=fa_tile_size,
                sliding_window=sliding_window,
                include_active_mask=True,
                batch_offset=batch_offset,
                bs_full=bs_full,
            )

        self._run_test(test_manager, lnc, input_generator)

    # ============================================================================
    # BLOCK KV WITH ACTIVE MASK TESTS - Testing _load_active_mask_block_kv
    # ============================================================================
    # These tests validate the batch sharding fix in _load_active_mask_block_kv.
    # Previously, batch-sharded mode with block KV silently skipped loading the
    # active mask, causing test failures.

    # fmt: off
    # Test parameters for block KV with active_mask tests
    # (batch, q_head, s_ctx, s_active, block_len, sliding_window, s_prior_offset, fa_tile_size, lnc)
    block_kv_active_mask_test_params = "batch, q_head, s_ctx, s_active, block_len, sliding_window, s_prior_offset, fa_tile_size, lnc, batch_offset, bs_full"
    block_kv_active_mask_test_perms = [
        # LNC=1 block KV with active_mask (full s_prior)
        (4, 1, 2048, 5, 16, 0, 0, 0, 1, 0, None),
        (4, 2, 4096, 5, 16, 0, 0, 0, 1, 0, None),
        # s_active=1 edge case with block KV active_mask
        (4, 1, 2048, 1, 16, 0, 0, 0, 1, 0, None),
        # LNC=2 sprior-sharded (bs * q_head * s_active <= P_MAX)
        (4, 1, 8192, 5, 16, 0, 0, 0, 2, 0, None),  # Sprior sharded, full s_prior
        (4, 1, 4096, 5, 16, 0, 0, 0, 2, 0, None),  # Sprior sharded, smaller s_ctx
        # LNC=2 batch-sharded (bs * q_head * s_active > P_MAX)
        # These test cases replicate the previously failing attention_tkg test vectors:
        # - [64, 8, 1, 2048, 2048, 128, 16, True, True, ...] (bs*q*s=512 > 128)
        (64, 8, 2048, 1, 16, 0, 0, 0, 2, 0, None),  # Batch sharding: 64*8*1=512 > 128 - matches failing test!
        (8, 8, 4096, 5, 16, 0, 0, 0, 2, 0, None),   # Batch sharding: 8*8*5=320 > 128
        (4, 8, 4096, 7, 16, 0, 0, 0, 2, 0, None),   # Batch sharding: 4*8*7=224 > 128
        # Block KV FA tile with active_mask (tile must include active region)
        # For FA tile tests, active positions are at END of s_prior_per_shard
        # sprior-sharded: s_prior_per_shard = s_ctx/lnc = 8192/2 = 4096, so tile [2048, 4096] is at end
        (4, 1, 8192, 5, 16, 0, 2048, 2048, 2, 0, None),  # FA tile, sprior sharded
        # batch-sharded: s_prior_per_shard = s_ctx = 8192, so tile must be at [6144, 8192]
        (8, 8, 8192, 5, 16, 0, 6144, 2048, 2, 0, None),  # FA tile, batch sharded

        # batch_offset > 0 (batch tiling): block KV
        (2, 1, 2048, 5, 16, 0, 0, 0, 1, 2, 4),   # Block KV, second tile
        (3, 2, 4096, 5, 16, 0, 0, 0, 1, 3, 7),   # Block KV, odd batch, second tile
        # batch_offset > 0, LNC=2 sprior-sharded block KV
        (2, 1, 8192, 5, 16, 0, 0, 0, 2, 2, 4),   # Sprior-sharded, second tile
        # batch_offset > 0, LNC=2 batch-sharded block KV
        (4, 8, 4096, 5, 16, 0, 0, 0, 2, 4, 16),  # Batch-sharded, second tile
        # batch_offset > 0 + s_prior_offset > 0 (FA tiling + batch tiling combined), block KV
        (2, 1, 4096, 5, 16, 0, 2048, 2048, 1, 2, 4),  # Block KV, FA last tile + batch offset
        # batch_offset > 0 + SWA (sliding window + batch tiling combined), block KV
        (2, 1, 2048, 5, 16, 128, 0, 0, 1, 2, 4),   # Block KV + SWA + batch offset
        (2, 1, 8192, 5, 16, 128, 0, 0, 2, 2, 4),   # Block KV + SWA + batch offset, sprior-sharded LNC=2
    ]
    # fmt: on

    @pytest.mark.parametrize(block_kv_active_mask_test_params, block_kv_active_mask_test_perms)
    def test_block_kv_with_active_mask(
        self,
        test_manager: Orchestrator,
        batch: int,
        q_head: int,
        s_ctx: int,
        s_active: int,
        block_len: int,
        sliding_window: int,
        s_prior_offset: int,
        fa_tile_size: int,
        lnc: int,
        batch_offset: int,
        bs_full: Optional[int],
    ):
        """
        Test block KV cache mask generation with active_mask provided.

        This tests the _load_active_mask_block_kv code path which was fixed
        to properly handle batch sharding. Before the fix, batch-sharded mode
        would silently skip loading active_mask, causing test failures.
        """

        def input_generator(test_config, input_tensor_def=None):
            return generate_gen_mask_inputs(
                batch=batch,
                q_head=q_head,
                s_ctx=s_ctx,
                s_active=s_active,
                block_len=block_len,
                lnc=lnc,
                strided_mm1=False,
                s_prior_offset=s_prior_offset,
                fa_tile_size=fa_tile_size,
                sliding_window=sliding_window,
                include_active_mask=True,
                batch_offset=batch_offset,
                bs_full=bs_full,
            )

        self._run_test(test_manager, lnc, input_generator)
