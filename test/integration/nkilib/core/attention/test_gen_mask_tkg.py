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
Integration tests for gen_mask_tkg standalone mask generation kernel and
gen_mask_tkg_hbm HBM wrapper.

This test suite validates the mask generation algorithm used by attention_tkg,
covering all code paths:
- Flat KV cache (block_len=0) with strided and non-strided MM1 layouts
- Block KV cache (block_len>0) with shuffled index generation
- LNC sharding configurations (lnc=1 and lnc=2)
- Sliding window attention (SWA) mask generation
- HBM wrapper with SBUF allocation, P_MAX broadcast, LNC sharding, and tiling

The golden function uses gen_mask_tkg_torch_ref from gen_mask_tkg_torch.py.
"""

import os
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
from nkilib_src.nkilib.core.attention.gen_mask_tkg import gen_mask_tkg, gen_mask_tkg_hbm
from nkilib_src.nkilib.core.attention.gen_mask_tkg_torch import gen_mask_tkg_hbm_torch_ref, gen_mask_tkg_torch_ref
from nkilib_src.nkilib.core.utils.allocator import SbufManager
from nkilib_src.nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info, reduce
from nkilib_src.nkilib.core.utils.logging import Logger

from test.integration.nkilib.core.attention.test_attention_tkg import build_active_attention_mask, build_swa_positions
from test.utils.common_dataclasses import (
    TKG_INFERENCE_ARGS,
    CompilerArgs,
    Platforms,
)
from test.utils.metrics_collector import MetricsCollector
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.simulation_setup import simulate_kernel
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, filter_kernel_input, torch_ref_wrapper

# Hardware constants
P_MAX = 128

_ABBREVS = {
    "batch": "b",
    "q_head": "qh",
    "s_ctx": "sc",
    "s_active": "sa",
    "block_len": "bl",
    "sliding_window": "sw",
    "strided_mm1": "smm1",
    "s_prior_offset": "spo",
    "fa_tile_size": "fa",
    "lnc": "lnc",
    "batch_offset": "bo",
    "bs_full": "bsf",
    "cp_seq_offset": "cpso",
}


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
    transposed_out: bool = False,
) -> nl.ndarray:
    """Wrapper kernel that handles HBM↔SBUF transfers for testing gen_mask_tkg.

    transposed_out=False:
        LNC=1: Output shape is [P_MAX, n_sprior_tile, bs, q_head, s_active]
        LNC=2: Output shape is [2, P_MAX, n_sprior_tile_per_shard, bs, q_head, s_active]

    transposed_out=True:
        LNC=1: [P_MAX, n_bsq_tiles_shard, s_prior_this]
        LNC=2: [lnc, P_MAX, n_bsq_tiles_shard, s_prior_this] (each shard writes golden[shard_id])
    """
    _, lnc, shard_id = get_verified_program_sharding_info("gen_mask_wrapper", (0, 1))

    mask_out_sbuf_shape = mask_out_hbm.shape if lnc == 1 else mask_out_hbm.shape[1:]
    num_elts = reduce('mul', mask_out_sbuf_shape, 1)

    sbm = SbufManager(0, num_elts * 8, Logger("gen_mask_wrapper"), use_auto_alloc=True)
    sbm.open_scope(name="gen_mask_wrapper")

    pos_ids_sbuf = sbm.alloc_stack((P_MAX, bs * s_active), dtype=pos_ids_hbm.dtype, buffer=nl.sbuf, name="pos_ids_sbuf")
    mask_out_sbuf = sbm.alloc_stack(
        mask_out_sbuf_shape,
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
        transposed_out=transposed_out,
    )

    if lnc == 1:
        golden_mask = nl.ndarray(
            mask_out_sbuf_shape, dtype=mask_out_sbuf.dtype, buffer=nl.shared_hbm, name="golden_mask"
        )
        nisa.dma_copy(dst=golden_mask, src=mask_out_sbuf)
    else:
        golden_mask = nl.ndarray(mask_out_hbm.shape, dtype=mask_out_hbm.dtype, buffer=nl.shared_hbm, name="golden_mask")
        nisa.dma_copy(dst=golden_mask.select(0, shard_id), src=mask_out_sbuf)

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
    start_pos_hbm: torch.Tensor | None = None,
    s_prior_offset: int = 0,
    block_len: int = 0,
    strided_mm1: bool = True,
    active_mask_hbm: torch.Tensor | None = None,
    is_batch_sharded: bool = False,
    batch_offset: int = 0,
    transposed_out: bool = False,
) -> dict[str, torch.Tensor]:
    """Torch ref adapter matching gen_mask_tkg_wrapper signature.

    Bridges the LncSubscriptable gen_mask_tkg_torch_ref to the flat-call
    interface expected by torch_ref_wrapper / UnitTestFramework.
    """
    pos_ids = pos_ids_hbm.float()
    active_mask = active_mask_hbm.float() if active_mask_hbm is not None else None
    start_pos = start_pos_hbm.float() if start_pos_hbm is not None else None

    # hacky LNC check since UnitTestFramework does not allow deviating APIs
    if transposed_out:
        lnc = mask_out_hbm.shape[0] if mask_out_hbm.dim() == 4 else 1
    else:
        lnc = mask_out_hbm.shape[0] if mask_out_hbm.dim() == 6 else 1

    def gen_mask(mask_out: torch.Tensor, shard_id: int) -> None:
        gen_mask_tkg_torch_ref.shard_id = shard_id
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
            transposed_out=transposed_out,
        )

    if lnc == 1:
        mask_out = torch.zeros(mask_out_hbm.shape, dtype=torch.float32)
        gen_mask(mask_out, 0)
        return {"golden_mask": mask_out}
    else:
        result = torch.zeros(mask_out_hbm.shape, dtype=torch.float32)
        for shard_idx in range(lnc):
            mask_out = torch.zeros(mask_out_hbm.shape[1:], dtype=torch.float32)
            gen_mask(mask_out, shard_idx)
            result[shard_idx] = mask_out
        return {"golden_mask": result}


def gen_mask_tkg_hbm_torch_ref_adapter_factory(lnc: int):
    """Create a torch ref adapter for gen_mask_tkg_hbm at a given LNC.

    Returns a function matching the gen_mask_tkg_hbm kernel signature that
    delegates to gen_mask_tkg_hbm_torch_ref[lnc].
    """

    def gen_mask_tkg_hbm_torch_ref_adapter(
        pos_ids_hbm: torch.Tensor,
        bs: int,
        q_head: int,
        s_active: int,
        s_prior: int,
        start_pos_hbm: torch.Tensor | None = None,
        block_len: int = 0,
        active_mask: torch.Tensor | None = None,
        enable_fa_s_prior_tiling: bool = True,
        fuse_rope: bool = False,
        transposed_out: bool = False,
        cp_seq_offset: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Torch ref adapter matching gen_mask_tkg_hbm kernel signature."""
        mask = gen_mask_tkg_hbm_torch_ref[lnc](
            pos_ids_hbm=pos_ids_hbm,
            bs=bs,
            q_head=q_head,
            s_active=s_active,
            s_prior=s_prior,
            start_pos_hbm=start_pos_hbm,
            block_len=block_len,
            active_mask=active_mask,
            enable_fa_s_prior_tiling=enable_fa_s_prior_tiling,
            fuse_rope=fuse_rope,
            transposed_out=transposed_out,
            cp_seq_offset=cp_seq_offset,
        )
        # gen_mask_tkg_hbm emits its HBM output as uint8 (binary 0/1 mask);
        # cast the fp32 reference to uint8 so the exact-match comparison lines
        # up with the kernel's uint8 output edge.
        return {"mask_out_hbm": mask.to(torch.uint8)}

    # The torch-ref golden cache keys on the ref's __qualname__ (plus dep/input
    # hashes). lnc is captured in this closure and is NOT part of the adapter's
    # kwargs, so without stamping it here the lnc=1 and lnc=2 arms share one
    # cache key while producing different goldens (block_len resizes per lnc) ->
    # a warm cache serves the wrong sibling's golden. Stamp lnc into the qualname
    # so each lnc keys a distinct cache entry.
    gen_mask_tkg_hbm_torch_ref_adapter.__qualname__ += f"[lnc{lnc}]"
    gen_mask_tkg_hbm_torch_ref_adapter.__name__ += f"_lnc{lnc}"
    return gen_mask_tkg_hbm_torch_ref_adapter


def generate_gen_mask_hbm_inputs(
    batch: int,
    q_head: int,
    s_ctx: int,
    s_active: int,
    block_len: int,
    strided_mm1: bool,
    sliding_window: int = 0,
    include_active_mask: bool = False,
    lnc: int = 2,
    cp_seq_offset: int = 0,
    dtype=np.float32,
    transposed_out: bool = False,
):
    """Build kernel inputs for gen_mask_tkg_hbm test, compatible with UnitTestFramework.

    Generates SBUF-level inputs at lnc=1 and extracts HBM-level tensors
    matching the gen_mask_tkg_hbm kernel signature.
    """
    resolved_strided_mm1 = strided_mm1 if strided_mm1 is not None else (block_len == 0)

    sbuf_inp = generate_gen_mask_inputs(
        batch=batch,
        q_head=q_head,
        s_ctx=s_ctx,
        s_active=s_active,
        block_len=block_len,
        lnc=1,
        strided_mm1=resolved_strided_mm1,
        sliding_window=sliding_window,
        include_active_mask=include_active_mask,
        dtype=dtype,
    )

    s_prior = sbuf_inp["s_prior_per_shard"]  # lnc=1, no sharding

    result = {
        "pos_ids_hbm": sbuf_inp["pos_ids_hbm"][:1, :].copy(),
        "bs": batch,
        "q_head": q_head,
        "s_active": s_active,
        "s_prior": s_prior,
        "block_len": block_len,
        "transposed_out": transposed_out,
    }

    if "start_pos_hbm" in sbuf_inp:
        result["start_pos_hbm"] = sbuf_inp["start_pos_hbm"][:1, :].copy()

    if "active_mask_hbm" in sbuf_inp:
        result["active_mask"] = sbuf_inp["active_mask_hbm"]

    # Only surface cp_seq_offset when nonzero so the default (non-CP) call is
    # byte-identical to before this parameter existed.
    if cp_seq_offset != 0:
        result["cp_seq_offset"] = cp_seq_offset

    return result


def create_pos_ids_tensor(
    cache_lens: np.ndarray,
    batch: int,
    s_active: int,
    dtype=np.float32,
) -> np.ndarray:
    """
    Create pos_ids tensor from cache lengths.

    Args:
        cache_lens: [batch] array of cache lengths per batch.
        batch: Batch size.
        s_active: Active sequence length.
        dtype: Output dtype.

    Returns:
        pos_ids: [P_MAX, batch * s_active] tensor where all partitions get
                 the same cache_len value (broadcasted).
    """
    row = np.zeros(batch * s_active, dtype=dtype)
    for b in range(batch):
        for i in range(s_active):
            row[b * s_active + i] = cache_lens[b] + i
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
    transposed_out: bool = False,
):
    """Build kernel inputs for gen_mask_tkg test, compatible with UnitTestFramework.

    Returns dict with keys matching gen_mask_tkg_wrapper signature.

    transposed_out=True builds the QK-swap layout inputs: mask_out_hbm is 3D
    [P_MAX, n_bsq_tiles, s_prior] (s_active_bqh on partition, s_prior on free).
    """
    cfg = AttnTKGConfig(bs=batch, q_head=q_head, s_active=s_active, curr_sprior=s_ctx)
    sprior_sharded = is_s_prior_sharded(cfg.bs, cfg.q_head, cfg.s_active, cfg.curr_sprior, P_MAX) if lnc > 1 else False

    if sprior_sharded:
        s_prior_per_shard = s_ctx // lnc
    else:
        s_prior_per_shard = s_ctx

    if transposed_out:
        s_active_bqh = batch * q_head * s_active
        assert s_active_bqh % P_MAX == 0, (
            f"transposed_out requires batch*q_head*s_active ({s_active_bqh}) divisible by P_MAX ({P_MAX})"
        )
        n_bsq_tiles = s_active_bqh // P_MAX
        mask_out_base_shape = (P_MAX, n_bsq_tiles, s_prior_per_shard)
    else:
        if fa_tile_size > 0:
            n_sprior_tile_per_shard = fa_tile_size // P_MAX
            assert s_prior_offset + fa_tile_size <= s_prior_per_shard, (
                f"FA tile (offset={s_prior_offset}, size={fa_tile_size}) exceeds s_prior_per_shard ({s_prior_per_shard})"
            )
        else:
            n_sprior_tile_per_shard = s_prior_per_shard // P_MAX
        mask_out_base_shape = (P_MAX, n_sprior_tile_per_shard, batch, q_head, s_active)

    mask_out_shape = mask_out_base_shape if lnc == 1 else (lnc,) + mask_out_base_shape

    adjusted_block_len = block_len
    if block_len > 0:
        num_blocks_total = s_ctx // block_len
        adjusted_block_len, _ = resize_cache_block_len_for_attention_tkg_kernel(
            num_blocks_total,
            block_len,
            lnc,
            P_MAX,
            batch,
            q_head,
            s_active,
            enable_fa_s_prior_tiling=fa_tile_size > 0,
        )
        if fa_tile_size > 0:
            fold_size = adjusted_block_len * P_MAX
            assert fa_tile_size % fold_size == 0, (
                f"fa_tile_size ({fa_tile_size}) must be divisible by block_len * P_MAX ({fold_size})"
            )
            assert s_prior_offset % fold_size == 0, (
                f"s_prior_offset ({s_prior_offset}) must be divisible by block_len * P_MAX ({fold_size})"
            )

    np.random.seed(42)
    # pos_ids[b, i] = cache_lens[b] + i, so cache_lens[b] + s_active - 1 <= s_ctx - 1
    cache_lens = np.random.randint(1, s_ctx - s_active + 1, size=(batch,)).astype(np.int32)

    pos_ids_data = create_pos_ids_tensor(cache_lens, batch, s_active, dtype)

    if sliding_window > 0:
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
        start_pos_data = None

    mask_out_data = np.zeros(mask_out_shape, dtype=dtype)

    result = {
        "pos_ids_hbm": pos_ids_data,
        "mask_out_hbm": mask_out_data,
        "bs": batch,
        "q_head": q_head,
        "s_active": s_active,
        "is_s_prior_sharded": sprior_sharded,
        "s_prior_per_shard": s_prior_per_shard,
        "s_prior_offset": s_prior_offset,
        "block_len": adjusted_block_len,
        "strided_mm1": strided_mm1,
        "transposed_out": transposed_out,
    }

    if start_pos_data is not None:
        result["start_pos_hbm"] = start_pos_data

    if include_active_mask:
        batch_sharded = (
            is_batch_sharded_fn(cfg.bs, cfg.q_head, cfg.s_active, cfg.curr_sprior, P_MAX) if lnc > 1 else False
        )

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


@pytest_test_metadata(name="Gen Mask TKG")
@pytest_marks(["attention", "tkg", "subkernel"])
@final
class TestGenMaskTkg:
    """
    Integration test suite for gen_mask_tkg kernel.

    Tests run the actual NKI kernel on Neuron hardware and compare output
    against the torch reference implementation (gen_mask_tkg_torch_ref).
    """

    def _run_test(
        self, test_manager: Orchestrator, collector: MetricsCollector, platform_target: Platforms, lnc, input_generator
    ):
        def output_tensors(kernel_input):
            return {"golden_mask": kernel_input["mask_out_hbm"]}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=gen_mask_tkg_wrapper,
            torch_ref=torch_ref_wrapper(gen_mask_tkg_torch_ref_adapter),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
            collector=collector,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
            rtol=0,
            atol=0,
        )

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
        pytest.param(4, 2, 2048, 7, 0, 0, 0, 1, marks=pytest.mark.fast),  # Multiple heads
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

    @pytest_parametrize(flat_kv_strided_test_params, flat_kv_strided_test_perms, abbrevs=_ABBREVS)
    def test_flat_kv_strided_mask_generation(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
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

        self._run_test(test_manager, collector, platform_target, lnc, input_generator)

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

    @pytest_parametrize(flat_kv_nonstrided_test_params, flat_kv_nonstrided_test_perms, abbrevs=_ABBREVS)
    def test_flat_kv_nonstrided_mask_generation(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
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

        self._run_test(test_manager, collector, platform_target, lnc, input_generator)

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

    @pytest_parametrize(block_kv_test_params, block_kv_test_perms, abbrevs=_ABBREVS)
    def test_block_kv_mask_generation(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
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

        self._run_test(test_manager, collector, platform_target, lnc, input_generator)

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

    @pytest_parametrize(swa_test_params, swa_test_perms, abbrevs=_ABBREVS)
    def test_swa_mask_generation(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
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

        self._run_test(test_manager, collector, platform_target, lnc, input_generator)

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

        # Non-strided MM1 with batch-sharded LNC=2 (tests the strided slice path)
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

    @pytest_parametrize(active_mask_test_params, active_mask_test_perms, abbrevs=_ABBREVS)
    def test_flat_kv_with_active_mask(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
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

        self._run_test(test_manager, collector, platform_target, lnc, input_generator)

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

    @pytest_parametrize(block_kv_active_mask_test_params, block_kv_active_mask_test_perms, abbrevs=_ABBREVS)
    def test_block_kv_with_active_mask(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
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

        self._run_test(test_manager, collector, platform_target, lnc, input_generator)

    # ============================================================================
    # TRANSPOSED (transposed_out) MASK TESTS - the s_active_bqh-partition layout
    # (s_active_bqh on partition, s_prior on free), the transposed counterpart of the default
    # s_prior-partition layout. Covers LNC=1 and LNC=2 (both s_prior- and batch-sharded).
    # ============================================================================

    # fmt: off
    transposed_test_params = "batch, q_head, s_ctx, s_active, block_len, sliding_window, lnc"
    transposed_test_perms = [
        # s_active_qh = q_head * s_active; batch * s_active_qh a multiple of P_MAX.
        # LNC=1
        (4, 16, 2048, 8, 32, 0, 1),      # s_active_qh=128, full causal
        (4, 16, 2048, 8, 32, 128, 1),    # s_active_qh=128, SWA
        (4, 16, 2048, 8, 32, 256, 1),    # s_active_qh=128, wider window
        (8, 16, 4096, 8, 32, 0, 1),      # more folds
        (8, 16, 4096, 8, 32, 128, 1),    # more folds, SWA
        (16, 8, 2048, 4, 32, 0, 1),      # s_active_qh=32
        (16, 8, 2048, 4, 32, 128, 1),    # s_active_qh=32, SWA
        # LNC=2 s_prior-sharded (large s_ctx, small bs*s_active_bqh -> shards s_prior on the free axis)
        (4, 16, 4096, 8, 32, 0, 2),      # sprior-sharded, full causal
        (4, 16, 4096, 8, 32, 128, 2),    # sprior-sharded, SWA
        # LNC=2 batch-sharded (bs*s_active_bqh large -> shards batch on the partition/grp axis)
        (16, 8, 2048, 4, 32, 0, 2),      # batch-sharded, full causal
        (16, 8, 2048, 4, 32, 128, 2),    # batch-sharded, SWA
        # Flat KV (block_len=0): contiguous s_prior on the free axis.
        (4, 16, 2048, 8, 0, 0, 1),       # flat KV, full causal
        (4, 16, 2048, 8, 0, 128, 1),     # flat KV, SWA
        (4, 16, 4096, 8, 0, 0, 2),       # flat KV, sprior-sharded
        (16, 8, 2048, 4, 0, 0, 2),       # flat KV, batch-sharded
    ]
    # fmt: on

    @pytest_parametrize(transposed_test_params, transposed_test_perms, abbrevs=_ABBREVS)
    def test_transposed_mask_generation(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        batch: int,
        q_head: int,
        s_ctx: int,
        s_active: int,
        block_len: int,
        sliding_window: int,
        lnc: int,
    ):
        """Directly test the transposed_out (s_active_bqh-partition) mask layout, kernel vs torch ref."""

        def input_generator(test_config, input_tensor_def=None):
            return generate_gen_mask_inputs(
                batch=batch,
                q_head=q_head,
                s_ctx=s_ctx,
                s_active=s_active,
                block_len=block_len,
                lnc=lnc,
                strided_mm1=False,
                sliding_window=sliding_window,
                transposed_out=True,
            )

        self._run_test(test_manager, collector, platform_target, lnc, input_generator)


# ============================================================================
# HBM WRAPPER TESTS
# ============================================================================


@pytest_marks(["attention", "tkg", "subkernel"])
class TestGenMaskTkgHbm:
    """
    Integration test suite for gen_mask_tkg_hbm HBM wrapper kernel.

    Tests run the actual NKI kernel on Neuron hardware and compare output
    against the torch reference implementation. The HBM wrapper handles
    SBUF allocation, P_MAX broadcast, LNC sharding, and FA tile looping
    internally, so tests only need to provide HBM-level inputs.
    """

    def _run_test(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        lnc: int,
        input_generator,
    ):
        def output_tensors(kernel_input):
            s_prior = kernel_input["s_prior"]
            bs = kernel_input["bs"]
            q_head = kernel_input["q_head"]
            s_active = kernel_input["s_active"]
            # gen_mask_tkg_hbm emits its HBM output as uint8 (binary 0/1 mask),
            # not fp32.  The descriptor dtype tells the validator how to
            # reinterpret the kernel's raw HBM bytes, so it must be uint8: with
            # the exact-match comparison below (rtol=0/atol=0), a regression back
            # to an fp32 output would misalign the raw bytes and fail hard.
            if kernel_input["transposed_out"]:
                shape = (bs, q_head, s_active, s_prior)
            else:
                shape = (s_prior, bs, q_head, s_active)
            return {"mask_out_hbm": np.zeros(shape, dtype=np.uint8)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=gen_mask_tkg_hbm,
            torch_ref=torch_ref_wrapper(gen_mask_tkg_hbm_torch_ref_adapter_factory(lnc)),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
            collector=collector,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
            # Exact match: the mask is binary (0/1), so no tolerance is allowed.
            # This transitively verifies the kernel's uint8 output equals the
            # uint8-cast reference byte-for-byte (dtype + value equivalence).
            rtol=0,
            atol=0,
            inference_args=TKG_INFERENCE_ARGS,
        )

    # ========================================================================
    # OUTPUT DTYPE + VALUE-RANGE REGRESSION (uint8 mask)
    # ========================================================================

    # fmt: off
    output_dtype_test_params = "batch, q_head, s_ctx, s_active, block_len, sliding_window, strided_mm1"
    output_dtype_test_perms = [
        (4, 1, 256, 1, 0, 0, True),      # full-context, flat KV
        (4, 1, 256, 5, 0, 256, True),    # SWA (sliding_window=256), flat KV
        (4, 1, 2048, 5, 128, 0, False),  # full-context, block KV
    ]
    # fmt: on

    @pytest_parametrize(output_dtype_test_params, output_dtype_test_perms, abbrevs=_ABBREVS)
    def test_output_dtype_and_range(
        self,
        platform_target: Platforms,
        batch: int,
        q_head: int,
        s_ctx: int,
        s_active: int,
        block_len: int,
        sliding_window: int,
        strided_mm1: bool,
    ):
        """Regression guard: gen_mask_tkg_hbm emits a uint8 binary (0/1) mask.

        Directly simulates the kernel (mirroring the framework's sim path) and
        asserts the HBM output tensor is uint8 (was fp32) and that every value
        is in {0, 1}.  This pins the dtype narrowing and proves it did not
        corrupt the mask contents, independent of the golden comparison.
        """
        kernel_input = generate_gen_mask_hbm_inputs(
            batch=batch,
            q_head=q_head,
            s_ctx=s_ctx,
            s_active=s_active,
            block_len=block_len,
            strided_mm1=strided_mm1,
            sliding_window=sliding_window,
            lnc=1,
        )
        os.environ["NKI_NC_VERSION"] = platform_target.get_nc_gen()
        outputs = simulate_kernel(
            gen_mask_tkg_hbm,
            filter_kernel_input(kernel_input, gen_mask_tkg_hbm),
            1,  # lnc=1
        )
        assert len(outputs) == 1, f"expected 1 output tensor, got {len(outputs)}"
        mask = np.asarray(outputs[0])
        assert mask.dtype == np.uint8, f"mask HBM output dtype must be uint8, got {mask.dtype}"
        unique_vals = np.unique(mask)
        assert np.all(np.isin(unique_vals, [0, 1])), f"uint8 mask must be binary (0/1); found values {unique_vals}"

    # ========================================================================
    # FLAT KV CACHE TESTS (block_len = 0)
    # ========================================================================

    # fmt: off
    flat_kv_test_params = "batch, q_head, s_ctx, s_active, strided_mm1, lnc"
    flat_kv_test_perms = [
        # Strided MM1, LNC=1
        (4, 1, 256, 1, True, 1),
        (4, 1, 1024, 5, True, 1),
        (4, 2, 2048, 7, True, 1),
        (16, 4, 4096, 5, True, 1),
        # Non-strided MM1, LNC=1
        pytest.param(4, 1, 256, 1, False, 1, marks=pytest.mark.fast),
        (4, 1, 1024, 5, False, 1),
        (8, 2, 4096, 7, False, 1),
        # LNC=2 s_prior-sharded (BQS <= 128)
        pytest.param(4, 1, 4096, 5, True, 2, marks=pytest.mark.fast),      # BQS=20
        (4, 2, 16384, 7, True, 2),     # BQS=56
        (4, 1, 4096, 5, False, 2),     # Non-strided, s_prior-sharded
        # LNC=2 batch-sharded (BQS > 128)
        pytest.param(8, 8, 4096, 5, True, 2, marks=pytest.mark.fast),      # BQS=320
        (8, 8, 4096, 5, False, 2),     # Non-strided, batch-sharded
    ]
    # fmt: on

    @pytest_parametrize(flat_kv_test_params, flat_kv_test_perms, abbrevs=_ABBREVS)
    def test_flat_kv(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        batch: int,
        q_head: int,
        s_ctx: int,
        s_active: int,
        strided_mm1: bool,
        lnc: int,
    ):
        """Test flat KV cache mask generation (block_len=0)."""

        def input_generator(test_config, input_tensor_def=None):
            return generate_gen_mask_hbm_inputs(
                batch=batch,
                q_head=q_head,
                s_ctx=s_ctx,
                s_active=s_active,
                block_len=0,
                strided_mm1=strided_mm1,
                lnc=lnc,
            )

        self._run_test(test_manager, collector, platform_target, lnc, input_generator)

    # ========================================================================
    # BLOCK KV CACHE TESTS (block_len > 0)
    # ========================================================================

    # fmt: off
    block_kv_test_params = "batch, q_head, s_ctx, s_active, block_len, lnc"
    block_kv_test_perms = [
        # LNC=1
        (4, 1, 256, 5, 16, 1),
        (4, 1, 2048, 5, 32, 1),
        (8, 2, 4096, 7, 16, 1),
        (4, 1, 4096, 5, 128, 1),       # block_len=128, resizes to 32
        (4, 1, 32768, 5, 128, 1),      # block_len=128, no resize, multi-fold
        # LNC=2 s_prior-sharded
        (4, 1, 8192, 5, 16, 2),
        (4, 1, 8192, 5, 128, 2),       # block_len=128, resizes to 32
        (4, 1, 32768, 5, 128, 2),      # block_len=128, no resize, multi-fold, sprior-sharded
        # LNC=2 batch-sharded
        (8, 8, 4096, 5, 16, 2),        # BQS=320
        (8, 8, 4096, 5, 128, 2),       # block_len=128, resizes to 16
        (8, 8, 32768, 5, 128, 2),      # block_len=128, no resize, multi-fold, batch-sharded
    ]
    # fmt: on

    @pytest_parametrize(block_kv_test_params, block_kv_test_perms, abbrevs=_ABBREVS)
    def test_block_kv(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        batch: int,
        q_head: int,
        s_ctx: int,
        s_active: int,
        block_len: int,
        lnc: int,
    ):
        """Test block KV cache mask generation (block_len>0)."""

        def input_generator(test_config, input_tensor_def=None):
            return generate_gen_mask_hbm_inputs(
                batch=batch,
                q_head=q_head,
                s_ctx=s_ctx,
                s_active=s_active,
                block_len=block_len,
                strided_mm1=False,
                lnc=lnc,
            )

        self._run_test(test_manager, collector, platform_target, lnc, input_generator)

    # ========================================================================
    # ACTIVE MASK TESTS
    # ========================================================================

    # fmt: off
    hbm_active_mask_test_params = "batch, q_head, s_ctx, s_active, block_len, strided_mm1, lnc"
    hbm_active_mask_test_perms = [
        # Flat KV, LNC=1
        (4, 1, 256, 5, 0, True, 1),
        (4, 2, 512, 5, 0, True, 1),
        (4, 1, 256, 5, 0, False, 1),
        (4, 1, 256, 1, 0, True, 1),     # s_active=1
        # Block KV with active mask, LNC=1
        (4, 1, 2048, 5, 16, False, 1),
        (4, 2, 4096, 5, 16, False, 1),
        (4, 1, 4096, 5, 128, False, 1),
        (4, 1, 32768, 5, 128, False, 1),  # block_len=128, no resize, multi-fold
        # Larger batch / s_ctx
        (16, 4, 4096, 5, 0, True, 1),
        (8, 2, 4096, 7, 16, False, 1),
        # LNC=2 s_prior-sharded
        (4, 2, 4096, 5, 0, True, 2),    # BQS=40
        (4, 1, 8192, 5, 16, False, 2),
        (4, 1, 8192, 5, 128, False, 2),
        (4, 1, 32768, 5, 128, False, 2),  # block_len=128, no resize, multi-fold, sprior-sharded
        # LNC=2 batch-sharded
        (8, 8, 256, 5, 0, True, 2),     # BQS=320
        (8, 8, 4096, 5, 16, False, 2),
        (8, 8, 4096, 5, 128, False, 2),
        (8, 8, 32768, 5, 128, False, 2),  # block_len=128, no resize, multi-fold, batch-sharded
    ]
    # fmt: on

    @pytest_parametrize(hbm_active_mask_test_params, hbm_active_mask_test_perms, abbrevs=_ABBREVS)
    def test_active_mask(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        batch: int,
        q_head: int,
        s_ctx: int,
        s_active: int,
        block_len: int,
        strided_mm1: bool,
        lnc: int,
    ):
        """Test mask generation with active_mask (cascaded attention)."""

        def input_generator(test_config, input_tensor_def=None):
            return generate_gen_mask_hbm_inputs(
                batch=batch,
                q_head=q_head,
                s_ctx=s_ctx,
                s_active=s_active,
                block_len=block_len,
                strided_mm1=strided_mm1,
                include_active_mask=True,
                lnc=lnc,
            )

        self._run_test(test_manager, collector, platform_target, lnc, input_generator)

    # ========================================================================
    # SWA (SLIDING WINDOW ATTENTION) TESTS
    # ========================================================================

    # fmt: off
    hbm_swa_test_params = "batch, q_head, s_ctx, s_active, sliding_window, block_len, strided_mm1, lnc"
    hbm_swa_test_perms = [
        # Production gpt-oss-120b decode SWA shape: sliding_window=256 over a
        # large context.  Pairs with the full-context ctx=10240 case in
        # test_fa_tiling to cover both mask variants the commit verified.
        (8, 8, 10240, 1, 256, 0, True, 1),
        (8, 8, 10240, 1, 256, 128, False, 1),  # block KV variant
        # Flat KV strided, LNC=1
        (4, 1, 1024, 5, 128, 0, True, 1),
        (4, 1, 1024, 5, 256, 0, True, 1),
        (4, 2, 2048, 7, 512, 0, True, 1),
        # Flat KV non-strided, LNC=1
        (4, 1, 1024, 5, 128, 0, False, 1),
        (4, 2, 2048, 7, 256, 0, False, 1),
        # Block KV with SWA, LNC=1
        (4, 1, 2048, 5, 128, 16, False, 1),
        (4, 1, 2048, 5, 256, 16, False, 1),
        (4, 1, 4096, 5, 256, 128, False, 1),   # block_len=128 + SWA
        # LNC=2 s_prior-sharded
        (4, 1, 4096, 5, 128, 0, True, 2),
        (4, 1, 8192, 5, 256, 128, False, 2),   # block_len=128 + SWA, sprior-sharded
        # LNC=2 batch-sharded
        (8, 8, 4096, 5, 128, 0, True, 2),
        (8, 8, 4096, 5, 256, 128, False, 2),   # block_len=128 + SWA, batch-sharded
    ]
    # fmt: on

    @pytest_parametrize(hbm_swa_test_params, hbm_swa_test_perms, abbrevs=_ABBREVS)
    def test_swa(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        batch: int,
        q_head: int,
        s_ctx: int,
        s_active: int,
        sliding_window: int,
        block_len: int,
        strided_mm1: bool,
        lnc: int,
    ):
        """Test SWA (sliding window attention) mask generation."""

        def input_generator(test_config, input_tensor_def=None):
            return generate_gen_mask_hbm_inputs(
                batch=batch,
                q_head=q_head,
                s_ctx=s_ctx,
                s_active=s_active,
                block_len=block_len,
                strided_mm1=strided_mm1,
                sliding_window=sliding_window,
                lnc=lnc,
            )

        self._run_test(test_manager, collector, platform_target, lnc, input_generator)

    # ========================================================================
    # CONTEXT-PARALLEL SEQUENCE OFFSET TESTS (cp_seq_offset)
    # ========================================================================
    # A context-parallel decode rank holds a disjoint global slice
    # [cp_seq_offset, cp_seq_offset + s_prior) of the prior KV context, so the
    # shard-local mask iota must be shifted by cp_seq_offset into global
    # coordinates before the causal iota < pos_ids compare. These tests pass a
    # nonzero cp_seq_offset to both the kernel and the torch reference and
    # require an exact match. Because slot k = cache_len - 1 lies inside the
    # prior region for every batch element, a dropped or wrong offset flips at
    # least that bit, so the nonzero case fails if the offset is not applied.
    # The cp_seq_offset=0 rows re-confirm the non-CP path stays byte-identical.

    # fmt: off
    cp_seq_offset_test_params = "batch, q_head, s_ctx, s_active, block_len, strided_mm1, cp_seq_offset, lnc"
    cp_seq_offset_test_perms = [
        # Flat KV strided, LNC=1 — 0 (no-op) then a nonzero global offset
        (4, 1, 1024, 5, 0, True, 0, 1),
        pytest.param(4, 1, 1024, 5, 0, True, 256, 1, marks=pytest.mark.fast),
        (4, 2, 2048, 7, 0, True, 512, 1),
        # Flat KV non-strided, LNC=1
        (4, 1, 1024, 5, 0, False, 256, 1),
        # Block KV, LNC=1 — offset applied before the block shuffle reshape
        (4, 1, 2048, 5, 16, False, 512, 1),
        # LNC=2 s_prior-sharded — offset added on top of the per-shard base
        (4, 1, 4096, 5, 0, True, 1024, 2),
        # LNC=2 batch-sharded
        (8, 8, 4096, 5, 0, True, 1024, 2),
    ]
    # fmt: on

    @pytest_parametrize(cp_seq_offset_test_params, cp_seq_offset_test_perms, abbrevs=_ABBREVS)
    def test_cp_seq_offset(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        batch: int,
        q_head: int,
        s_ctx: int,
        s_active: int,
        block_len: int,
        strided_mm1: bool,
        cp_seq_offset: int,
        lnc: int,
    ):
        """Test context-parallel global sequence offset (cp_seq_offset).

        Regression test for the CP mask offset: the kernel shifts the prior
        mask into global coordinates, and the torch reference must apply the
        same shift. A nonzero offset that is dropped or misapplied produces a
        mismatch against the reference (rtol=atol=0).
        """

        def input_generator(test_config, input_tensor_def=None):
            return generate_gen_mask_hbm_inputs(
                batch=batch,
                q_head=q_head,
                s_ctx=s_ctx,
                s_active=s_active,
                block_len=block_len,
                strided_mm1=strided_mm1,
                cp_seq_offset=cp_seq_offset,
            )

        self._run_test(test_manager, collector, platform_target, lnc, input_generator)

    # ========================================================================
    # FA TILING TESTS (large configs triggering multiple tiles)
    # ========================================================================

    # fmt: off
    fa_tiling_test_params = "batch, q_head, s_ctx, s_active, block_len, strided_mm1, lnc"
    fa_tiling_test_perms = [
        # Production gpt-oss-120b decode shape: full-context s_prior=10240.
        # Pins the uint8 mask on the exact context length the commit verified.
        (8, 8, 10240, 1, 0, True, 1),     # full-context, ctx=10240 (gpt-oss decode)
        (8, 8, 10240, 1, 128, False, 1),  # full-context, ctx=10240, block KV
        # LNC=1 (strided uses batch tiling, non-strided uses s_prior tiling)
        (8, 8, 8192, 5, 0, True, 1),
        (8, 8, 8192, 7, 0, True, 1),      # last tile reduced
        (8, 8, 8192, 5, 0, False, 1),
        (8, 8, 8192, 5, 16, False, 1),    # block KV tiling
        (8, 8, 8192, 5, 128, False, 1),   # block_len=128 + tiling (resize to 64, batch tiling)
        (8, 8, 32768, 5, 128, False, 1),  # block_len=128, no resize, s_prior tiling
        # LNC=2 + tiling
        (8, 8, 8192, 5, 0, True, 2),      # batch-sharded + batch tiling
        (4, 4, 32768, 7, 0, True, 2),     # s_prior-sharded + batch tiling
        (4, 1, 32768, 5, 128, False, 2),  # block_len=128, no resize, sprior-sharded (single tile)
        (4, 4, 65536, 5, 128, False, 2),  # block_len=128, no resize, sprior-sharded + actual tiling
        (8, 8, 32768, 5, 128, False, 2),  # block_len=128, no resize, batch-sharded + tiling
    ]
    # fmt: on

    @pytest_parametrize(fa_tiling_test_params, fa_tiling_test_perms, abbrevs=_ABBREVS)
    def test_fa_tiling(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        batch: int,
        q_head: int,
        s_ctx: int,
        s_active: int,
        block_len: int,
        strided_mm1: bool,
        lnc: int,
    ):
        """Test FA tiling with large configs that trigger multiple tiles."""

        def input_generator(test_config, input_tensor_def=None):
            return generate_gen_mask_hbm_inputs(
                batch=batch,
                q_head=q_head,
                s_ctx=s_ctx,
                s_active=s_active,
                block_len=block_len,
                strided_mm1=strided_mm1,
                lnc=lnc,
            )

        self._run_test(test_manager, collector, platform_target, lnc, input_generator)

    # ========================================================================
    # FA TILING + ACTIVE MASK TESTS
    # ========================================================================

    # fmt: off
    fa_tiling_active_mask_test_params = "batch, q_head, s_ctx, s_active, block_len, strided_mm1, lnc"
    fa_tiling_active_mask_test_perms = [
        # LNC=1 — active mask at end of s_prior, only last tile loads it
        (8, 8, 8192, 5, 0, True, 1),
        (8, 8, 8192, 5, 0, False, 1),
        (8, 8, 8192, 5, 16, False, 1),     # block KV
        (8, 8, 8192, 5, 128, False, 1),    # block_len=128 + tiling + active mask
        (8, 8, 32768, 5, 128, False, 1),   # block_len=128, no resize + tiling + active mask
        # LNC=2 batch-sharded + tiling
        (8, 8, 8192, 5, 0, True, 2),
        (8, 8, 32768, 5, 128, False, 2),   # block_len=128, no resize, batch-sharded + tiling + active mask
        # LNC=2 s_prior-sharded + tiling
        (4, 4, 32768, 7, 0, True, 2),
        (4, 4, 65536, 5, 128, False, 2),   # block_len=128, no resize, sprior-sharded + tiling + active mask
    ]
    # fmt: on

    @pytest_parametrize(fa_tiling_active_mask_test_params, fa_tiling_active_mask_test_perms, abbrevs=_ABBREVS)
    def test_fa_tiling_with_active_mask(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        batch: int,
        q_head: int,
        s_ctx: int,
        s_active: int,
        block_len: int,
        strided_mm1: bool,
        lnc: int,
    ):
        """Test FA tiling combined with active_mask."""

        def input_generator(test_config, input_tensor_def=None):
            return generate_gen_mask_hbm_inputs(
                batch=batch,
                q_head=q_head,
                s_ctx=s_ctx,
                s_active=s_active,
                block_len=block_len,
                strided_mm1=strided_mm1,
                include_active_mask=True,
                lnc=lnc,
            )

        self._run_test(test_manager, collector, platform_target, lnc, input_generator)

    # ========================================================================
    # SWA + ACTIVE MASK TESTS
    # ========================================================================

    # fmt: off
    swa_active_mask_test_params = "batch, q_head, s_ctx, s_active, sliding_window, block_len, strided_mm1, lnc"
    swa_active_mask_test_perms = [
        (4, 2, 1024, 5, 256, 0, True, 1),
        (4, 1, 1024, 5, 128, 0, False, 1),
        (4, 1, 4096, 5, 256, 128, False, 1),   # block_len=128 + SWA + active mask
        (4, 2, 4096, 5, 128, 0, True, 2),
        (4, 1, 8192, 5, 256, 128, False, 2),   # block_len=128 + SWA + active mask, sprior-sharded
    ]
    # fmt: on

    @pytest_parametrize(swa_active_mask_test_params, swa_active_mask_test_perms, abbrevs=_ABBREVS)
    def test_swa_with_active_mask(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        batch: int,
        q_head: int,
        s_ctx: int,
        s_active: int,
        sliding_window: int,
        block_len: int,
        strided_mm1: bool,
        lnc: int,
    ):
        """Test SWA mask generation combined with active_mask."""

        def input_generator(test_config, input_tensor_def=None):
            return generate_gen_mask_hbm_inputs(
                batch=batch,
                q_head=q_head,
                s_ctx=s_ctx,
                s_active=s_active,
                block_len=block_len,
                strided_mm1=strided_mm1,
                sliding_window=sliding_window,
                include_active_mask=True,
                lnc=lnc,
            )

        self._run_test(test_manager, collector, platform_target, lnc, input_generator)

    # ========================================================================
    # LNC SWEEP TESTS
    # ========================================================================
    # These tests verify that gen_mask_tkg_hbm produces correct masks at both
    # LNC=1 and LNC=2 by parametrizing lnc as a grid dimension and delegating
    # to _run_test (which compares kernel output against the torch reference).
    #
    # This catches the P_MAX-major HBM layout bug: when the mask was stored as
    # [P_MAX, n_sprior_tile, ...], LNC=2 shard boundaries did not align with
    # row boundaries, producing wrong mask bits after reshape.

    # fmt: off
    lnc_sweep_test_params = "batch, q_head, s_ctx, s_active, block_len, strided_mm1, sliding_window"
    lnc_sweep_test_perms = [
        # Flat KV, strided
        (4, 1, 4096, 5, 0, True, 0),
        (4, 2, 16384, 7, 0, True, 0),
        # Flat KV, non-strided
        (4, 1, 4096, 5, 0, False, 0),
        # Block KV — the primary bug scenario
        (4, 1, 8192, 5, 16, False, 0),
        (4, 1, 12288, 5, 16, False, 0),
        (8, 8, 4096, 5, 16, False, 0),
        (4, 1, 8192, 5, 128, False, 0),
        # Block KV + SWA
        (4, 1, 8192, 5, 16, False, 128),
        (4, 1, 12288, 5, 16, False, 128),
        # Flat KV + SWA
        (4, 1, 4096, 5, 0, True, 128),
        (4, 2, 4096, 7, 0, True, 256),
    ]
    # fmt: on

    @pytest.mark.parametrize("lnc", [1, 2])
    @pytest_parametrize(lnc_sweep_test_params, lnc_sweep_test_perms, abbrevs=_ABBREVS)
    def test_gen_mask_hbm_lnc_sweep(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        batch: int,
        q_head: int,
        s_ctx: int,
        s_active: int,
        block_len: int,
        strided_mm1: bool,
        sliding_window: int,
        lnc: int,
    ):
        """Test gen_mask_tkg_hbm at both LNC=1 and LNC=2 against torch reference.

        Regression test for the P_MAX-major HBM layout bug. Each (config, lnc)
        pair is validated independently against the torch reference, so any
        shard-boundary misalignment shows up as a mismatch.
        """

        def input_generator(test_config, input_tensor_def=None):
            return generate_gen_mask_hbm_inputs(
                batch=batch,
                q_head=q_head,
                s_ctx=s_ctx,
                s_active=s_active,
                block_len=block_len,
                strided_mm1=strided_mm1,
                sliding_window=sliding_window,
                lnc=lnc,
            )

        self._run_test(test_manager, collector, platform_target, lnc, input_generator)

    # ========================================================================
    # TRANSPOSED (transposed_out) HBM TESTS
    # ========================================================================
    # The QK-swap HBM layout [bs, q_head, s_active, s_prior] (s_active_bqh-major),
    # the transposed counterpart of the default [s_prior, bs, q_head, s_active].
    # Covers block KV (the production swap path), flat KV (free-axis P_MAX tiling),
    # SWA, and both LNC sharding modes.

    # fmt: off
    transposed_hbm_test_params = "batch, q_head, s_ctx, s_active, block_len, sliding_window, lnc"
    transposed_hbm_test_perms = [
        # block KV, LNC=1 (batch * s_active_qh a multiple of P_MAX)
        (4, 16, 2048, 8, 32, 0, 1),      # s_active_qh=128, full causal
        (4, 16, 2048, 8, 32, 128, 1),    # s_active_qh=128, SWA
        (8, 16, 4096, 8, 32, 0, 1),      # more folds
        (16, 8, 2048, 4, 32, 128, 1),    # s_active_qh=32, SWA
        # flat KV, LNC=1 (free-axis P_MAX tiling)
        (4, 16, 2048, 8, 0, 0, 1),       # flat KV, full causal
        (4, 16, 2048, 8, 0, 128, 1),     # flat KV, SWA
        # LNC=2 s_prior-sharded (large s_ctx, small bs*s_active_bqh)
        (4, 16, 4096, 8, 32, 0, 2),      # block KV, sprior-sharded
        (4, 16, 4096, 8, 0, 0, 2),       # flat KV, sprior-sharded
        # LNC=2 batch-sharded (bs*s_active_bqh large)
        (16, 8, 2048, 4, 32, 0, 2),      # block KV, batch-sharded
        (16, 8, 2048, 4, 0, 128, 2),     # flat KV, batch-sharded, SWA
    ]
    # fmt: on

    @pytest_parametrize(transposed_hbm_test_params, transposed_hbm_test_perms, abbrevs=_ABBREVS)
    def test_transposed_hbm(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        batch: int,
        q_head: int,
        s_ctx: int,
        s_active: int,
        block_len: int,
        sliding_window: int,
        lnc: int,
    ):
        """Test the transposed_out (QK-swap) HBM mask layout, kernel vs torch ref."""

        def input_generator(test_config, input_tensor_def=None):
            return generate_gen_mask_hbm_inputs(
                batch=batch,
                q_head=q_head,
                s_ctx=s_ctx,
                s_active=s_active,
                block_len=block_len,
                strided_mm1=False,
                sliding_window=sliding_window,
                lnc=lnc,
                transposed_out=True,
            )

        self._run_test(test_manager, collector, platform_target, lnc, input_generator)
