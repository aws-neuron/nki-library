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

The golden function is reused from test_attention_tkg.py (numpy_gen_attention_cache_mask).
"""

from test.integration.nkilib.core.attention.test_attention_tkg import (
    numpy_gen_attention_active_mask,
    numpy_gen_attention_cache_mask,
)
from test.utils.common_dataclasses import (
    TKG_INFERENCE_ARGS,
    CompilerArgs,
    KernelArgs,
    LazyGoldenGenerator,
    ValidationArgs,
)
from test.utils.metrics_collector import MetricsCollector
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from typing import Any, final

import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.attention.attention_tkg_utils import (
    AttnTKGConfig,
    is_s_prior_sharded,
    resize_cache_block_len_for_attention_tkg_kernel,
)
from nkilib_src.nkilib.core.attention.attention_tkg_utils import (
    is_batch_sharded as is_batch_sharded_fn,
)
from nkilib_src.nkilib.core.attention.gen_mask_tkg import gen_mask_tkg
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
    s_prior_offset: int = 0,
    block_len: int = 0,
    strided_mm1: bool = True,
) -> nl.ndarray:
    """
    Wrapper kernel that handles HBM↔SBUF transfers for testing gen_mask_tkg.

    For LNC=1: Output shape is [P_MAX, n_sprior_tile, bs, q_head, s_active]
    For LNC=2: Output shape is [2, P_MAX, n_sprior_tile_per_shard, bs, q_head, s_active]
               Each shard writes to its portion of the shared buffer.

    Args:
        pos_ids_hbm: Position IDs tensor in HBM. Shape [P_MAX, bs * s_active].
        mask_out_hbm: Output mask buffer in shared HBM.
        bs: Batch size.
        q_head: Number of query heads.
        s_active: Active sequence length.
        is_s_prior_sharded: Whether s_prior dimension is sharded across LNCs.
        s_prior_per_shard: Total s_prior per shard (NC's full s_prior).
        s_prior_offset: Offset within current shard (for flash attention tiling).
        block_len: Block length for block KV cache (0 = flat cache).
        strided_mm1: Whether to use strided MM1 layout.

    Returns:
        mask_out_hbm: Generated mask tensor with combined output from all shards
    """
    # Get sharding info
    _, lnc, shard_id = get_verified_program_sharding_info("gen_mask_wrapper", (0, 1))

    # Extract dimensions from mask_out_hbm shape based on lnc
    if lnc == 1:
        # Shape: [P_MAX, n_sprior_tile, bs, q_head, s_active]
        _, n_sprior_tile_per_shard, _, _, _ = mask_out_hbm.shape
    else:
        # Shape: [lnc, P_MAX, n_sprior_tile_per_shard, bs, q_head, s_active]
        _, _, n_sprior_tile_per_shard, _, _, _ = mask_out_hbm.shape

    # Initialize SBUF manager
    sbm = SbufManager(
        0, P_MAX * n_sprior_tile_per_shard * bs * q_head * s_active * 8, Logger("gen_mask_wrapper"), use_auto_alloc=True
    )
    sbm.open_scope(name="gen_mask_wrapper")

    # Allocate SBUF tensors
    pos_ids_sbuf = sbm.alloc_stack((P_MAX, bs * s_active), dtype=pos_ids_hbm.dtype, buffer=nl.sbuf, name="pos_ids_sbuf")
    mask_out_sbuf = sbm.alloc_stack(
        (P_MAX, n_sprior_tile_per_shard, bs, q_head, s_active),
        dtype=mask_out_hbm.dtype,
        buffer=nl.sbuf,
        name="mask_out_sbuf",
    )

    # Copy pos_ids from HBM to SBUF
    nisa.dma_copy(dst=pos_ids_sbuf, src=pos_ids_hbm)

    # Call the actual gen_mask_tkg kernel
    gen_mask_tkg(
        pos_ids=pos_ids_sbuf,
        mask_out=mask_out_sbuf,
        bs=bs,
        q_head=q_head,
        s_active=s_active,
        is_s_prior_sharded=is_s_prior_sharded,
        s_prior_per_shard=s_prior_per_shard,
        s_prior_offset=s_prior_offset,
        block_len=block_len,
        strided_mm1=strided_mm1,
        active_mask=None,
        sbm=sbm,
    )

    # Create output tensor with name matching golden output key
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


def build_gen_mask_kernel_input(
    batch: int,
    q_head: int,
    s_ctx: int,
    s_active: int,
    block_len: int,
    lnc: int,
    strided_mm1: bool,
    dtype,
    s_prior_offset: int = 0,
    fa_tile_size: int = 0,
):
    """Build kernel inputs for gen_mask_tkg test."""
    # Determine sharding mode using the same logic as the kernel
    cfg = AttnTKGConfig(bs=batch, q_head=q_head, s_active=s_active, curr_sprior=s_ctx)
    sprior_sharded = is_s_prior_sharded(cfg, P_MAX) if lnc > 1 else False

    # For LNC sharding, each shard produces output based on sharding mode
    if sprior_sharded:
        # Sprior-sharded: each shard processes s_ctx // lnc
        s_prior_per_shard = s_ctx // lnc
    else:
        # Batch-sharded or no sharding: both shards process full s_prior
        s_prior_per_shard = s_ctx

    # Determine output tile size
    if fa_tile_size > 0:
        # FA tile mode: mask_out is for a specific tile
        n_sprior_tile_per_shard = fa_tile_size // P_MAX
        # FA tile must fit within shard's s_prior portion
        assert (
            s_prior_offset + fa_tile_size <= s_prior_per_shard
        ), f"FA tile (offset={s_prior_offset}, size={fa_tile_size}) exceeds s_prior_per_shard ({s_prior_per_shard})"
    else:
        # Full s_prior mode
        n_sprior_tile_per_shard = s_prior_per_shard // P_MAX

    # For block KV cache, compute the adjusted block_len
    adjusted_block_len = block_len
    if block_len > 0:
        num_blocks_total = s_ctx // block_len
        adjusted_block_len, _ = resize_cache_block_len_for_attention_tkg_kernel(num_blocks_total, block_len, lnc, P_MAX)
        # FA tile size and offset must be divisible by fold_size (block_len * P_MAX)
        if fa_tile_size > 0:
            fold_size = adjusted_block_len * P_MAX
            assert (
                fa_tile_size % fold_size == 0
            ), f"fa_tile_size ({fa_tile_size}) must be divisible by block_len * P_MAX ({fold_size})"
            assert (
                s_prior_offset % fold_size == 0
            ), f"s_prior_offset ({s_prior_offset}) must be divisible by block_len * P_MAX ({fold_size})"

    # Generate position IDs (cache lengths)
    # These represent how many prior tokens are valid for each batch
    np.random.seed(42)
    # Cache lens should be less than s_ctx to have meaningful masking
    cache_lens = np.random.randint(1, s_ctx, size=(batch,)).astype(np.int32)

    # Create pos_ids tensor: shape [P_MAX, bs * s_active]
    # All partitions get the same cache_len value (broadcasted)
    pos_ids_data = np.zeros((P_MAX, batch * s_active), dtype=dtype)
    for batch_idx in range(batch):
        for s_active_idx in range(s_active):
            pos_ids_data[:, batch_idx * s_active + s_active_idx] = cache_lens[batch_idx]

    # Output mask buffer shape depends on lnc and sharding mode
    if lnc == 1:
        # Shape: [P_MAX, n_sprior_tile, bs, q_head, s_active]
        mask_out_data = np.zeros((P_MAX, n_sprior_tile_per_shard, batch, q_head, s_active), dtype=dtype)
    else:
        # Shape: [lnc, P_MAX, n_sprior_tile_per_shard, bs, q_head, s_active]
        # Each shard writes to its portion of this combined buffer
        mask_out_data = np.zeros((lnc, P_MAX, n_sprior_tile_per_shard, batch, q_head, s_active), dtype=dtype)

    return {
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
        # Metadata for golden computation
        "_cache_lens": cache_lens,
        "_n_sprior_tile_per_shard": n_sprior_tile_per_shard,
        "_s_ctx_total": s_ctx,
        "_lnc": lnc,
        "_original_block_len": block_len,
        "_adjusted_block_len": adjusted_block_len,
    }


def golden_mask_output(
    inp: dict[str, Any],
    lnc: int,
) -> dict[str, np.ndarray]:
    """
    Compute golden mask output using numpy_gen_attention_cache_mask from test_attention_tkg.py.

    This function uses the same golden as the attention kernel tests, then transforms
    the output to match the gen_mask_tkg kernel's output format:
    - numpy_gen_attention_cache_mask output: (batch, num_heads, S_tkg, S_ctx)
    - gen_mask_tkg kernel output: (P_MAX, n_sprior_tile, bs, q_head, s_active)

    The transformation depends on the layout mode:
    - Strided MM1: kernel uses iota[p, f] = f + p * n_sprior_tile
    - Non-strided MM1: kernel uses iota[p, f] = p + f * P_MAX
    - Block KV: kernel uses shuffled indices matching K cache block layout

    For flash attention tiling (s_prior_offset > 0), the mask is generated for a tile
    starting at s_prior_offset within the shard's s_prior portion.
    """
    cache_lens = inp["_cache_lens"]
    batch = inp["bs"]
    q_head = inp["q_head"]
    s_active = inp["s_active"]
    s_ctx = inp["_s_ctx_total"]
    strided_mm1 = inp["strided_mm1"]
    original_block_len = inp["_original_block_len"]
    adjusted_block_len = inp["_adjusted_block_len"]
    n_sprior_tile_per_shard = inp["_n_sprior_tile_per_shard"]
    s_prior_offset = inp.get("s_prior_offset", 0)
    dtype = inp["pos_ids_hbm"].dtype

    # Determine sharding mode using the same logic as the kernel
    cfg = AttnTKGConfig(bs=batch, q_head=q_head, s_active=s_active, curr_sprior=s_ctx)
    sprior_sharded = is_s_prior_sharded(cfg, P_MAX) if lnc > 1 else False

    def transform_mask_to_kernel_format(
        mask: np.ndarray, n_tiles: int, block_len: int, strided: bool, shard_offset: int = 0
    ) -> np.ndarray:
        """
        Transform numpy_gen_attention_cache_mask output to gen_mask_tkg kernel format.

        Args:
            mask: Input mask with shape (batch, q_head, s_active, s_ctx)
            n_tiles: Number of s_prior tiles for this shard
            block_len: Adjusted block length (0 for flat KV)
            strided: Whether strided MM1 layout is used
            shard_offset: Offset for this shard's s_prior portion (includes both NC offset and FA tile offset)

        Returns:
            Mask with shape (P_MAX, n_tiles, batch, q_head, s_active)
        """
        s_prior_shard = n_tiles * P_MAX  # s_prior for this shard

        if block_len > 0:
            # Block KV: numpy_gen_attention_cache_mask already applied swapaxes(-1, -2)
            # so mask layout is (batch, q_head, s_active, num_folds, block_len, P_MAX) flattened
            # Slice on the fold dimension, then reshape to kernel format
            fold_size = block_len * P_MAX
            num_folds_total = mask.shape[-1] // fold_size
            num_folds_shard = s_prior_shard // fold_size
            fold_start = shard_offset // fold_size
            # Reshape to expose fold dimension, slice, then reshape to kernel format
            mask_by_fold = mask.reshape(batch, q_head, s_active, num_folds_total, block_len, P_MAX)
            mask_shard = mask_by_fold[:, :, :, fold_start : fold_start + num_folds_shard, :, :]
            # Shape: (batch, q_head, s_active, num_folds_shard, block_len, P_MAX)
            # Merge num_folds_shard and block_len to get n_tiles
            result = mask_shard.reshape(batch, q_head, s_active, num_folds_shard * block_len, P_MAX)
            result = result.transpose(4, 3, 0, 1, 2)  # -> (P_MAX, n_tiles, batch, q_head, s_active)
        else:
            # Flat KV: extract the relevant s_prior portion for this shard
            mask_shard = mask[:, :, :, shard_offset : shard_offset + s_prior_shard]

            if strided:
                # Strided MM1: kernel uses iota[p, f] = f + p * n_tiles
                # So position pos maps to (p, f) = (pos // n_tiles, pos % n_tiles)
                # mask_shard shape: (batch, q_head, s_active, s_prior_shard)
                result = mask_shard.reshape(batch, q_head, s_active, P_MAX, n_tiles)
                result = result.transpose(3, 4, 0, 1, 2)  # -> (P_MAX, n_tiles, batch, q_head, s_active)
            else:
                # Non-strided MM1: kernel uses iota[p, f] = p + f * P_MAX
                # So position pos maps to (p, f) = (pos % P_MAX, pos // P_MAX)
                # mask_shard shape: (batch, q_head, s_active, s_prior_shard)
                result = mask_shard.reshape(batch, q_head, s_active, n_tiles, P_MAX)
                result = result.transpose(4, 3, 0, 1, 2)  # -> (P_MAX, n_tiles, batch, q_head, s_active)

        return result.astype(dtype)

    # Generate full mask using numpy_gen_attention_cache_mask
    # Note: For block_len > 0, it already handles the swapaxes transformation internally
    full_mask = numpy_gen_attention_cache_mask(
        cache_len=cache_lens,
        batch=batch,
        num_heads=q_head,
        S_tkg=s_active,
        S_ctx=s_ctx,
        lnc=lnc,
        block_len=original_block_len,  # Use original block_len, function handles resizing
        unify_for_cascaded=False,  # Don't add active mask, we only test prior mask
    )
    # full_mask shape: (batch, q_head, s_active, s_ctx)

    if lnc == 1:
        # For LNC=1, s_prior_offset is the FA tile offset within the full s_prior
        mask_out = transform_mask_to_kernel_format(
            full_mask, n_sprior_tile_per_shard, adjusted_block_len, strided_mm1, shard_offset=s_prior_offset
        )
    elif sprior_sharded:
        # Sprior-sharded: each shard processes different s_prior portion
        s_prior_per_shard = s_ctx // lnc
        mask_out = np.zeros((lnc, P_MAX, n_sprior_tile_per_shard, batch, q_head, s_active), dtype=dtype)

        for lnc_shard_idx in range(lnc):
            # NC offset + FA tile offset
            shard_offset = lnc_shard_idx * s_prior_per_shard + s_prior_offset
            shard_mask = transform_mask_to_kernel_format(
                full_mask, n_sprior_tile_per_shard, adjusted_block_len, strided_mm1, shard_offset=shard_offset
            )
            mask_out[lnc_shard_idx] = shard_mask
    else:
        # Batch-sharded or no sharding: both shards process full s_prior (or FA tile), produce identical output
        mask_single = transform_mask_to_kernel_format(
            full_mask, n_sprior_tile_per_shard, adjusted_block_len, strided_mm1, shard_offset=s_prior_offset
        )

        mask_out = np.zeros((lnc, P_MAX, n_sprior_tile_per_shard, batch, q_head, s_active), dtype=dtype)
        for lnc_shard_idx in range(lnc):
            mask_out[lnc_shard_idx] = mask_single

    return {"golden_mask": mask_out}


@pytest_test_metadata(
    name="Gen Mask TKG",
    pytest_marks=["attention", "tkg", "subkernel"],
)
@final
class TestGenMaskTkg:
    """
    Integration test suite for gen_mask_tkg kernel.

    Tests run the actual NKI kernel on Neuron hardware and compare output
    against numpy golden implementations (using the same golden as test_attention_tkg.py).
    """

    def run_gen_mask_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        collector: MetricsCollector,
        batch: int,
        q_head: int,
        s_ctx: int,
        s_active: int,
        block_len: int,
        lnc: int,
        strided_mm1: bool = True,
        s_prior_offset: int = 0,
        fa_tile_size: int = 0,
        dtype=np.float32,
    ):
        """Run a single gen_mask_tkg test case."""
        kernel_input = build_gen_mask_kernel_input(
            batch=batch,
            q_head=q_head,
            s_ctx=s_ctx,
            s_active=s_active,
            block_len=block_len,
            lnc=lnc,
            strided_mm1=strided_mm1,
            dtype=dtype,
            s_prior_offset=s_prior_offset,
            fa_tile_size=fa_tile_size,
        )

        # Tolerances for mask validation (exact match expected for binary mask)
        relative_tolerance, absolute_tolerance = 0, 0

        # Create output placeholder for LazyGoldenGenerator
        output_placeholder = {"golden_mask": kernel_input["mask_out_hbm"]}

        # Create lazy golden generator function (takes no arguments)
        def create_lazy_golden() -> dict[str, np.ndarray]:
            return golden_mask_output(inp=kernel_input, lnc=lnc)

        # Extract kernel-only arguments (remove metadata)
        kernel_only_input = {k: v for k, v in kernel_input.items() if not k.startswith("_")}

        test_manager.execute(
            KernelArgs(
                kernel_func=gen_mask_tkg_wrapper,
                compiler_input=compiler_args,
                kernel_input=kernel_only_input,
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(
                        output_ndarray=output_placeholder,
                        lazy_golden_generator=create_lazy_golden,
                    ),
                    relative_accuracy=relative_tolerance,
                    absolute_accuracy=absolute_tolerance,
                ),
                inference_args=TKG_INFERENCE_ARGS,
            )
        )

    # ============================================================================
    # FLAT KV CACHE TESTS (block_len = 0) - STRIDED MM1
    # ============================================================================

    # fmt: off
    # Test parameters for flat KV cache with strided MM1 layout
    # (batch, q_head, s_ctx, s_active, s_prior_offset, fa_tile_size, lnc)
    # fa_tile_size=0 means full s_prior, >0 means FA tile mode
    flat_kv_strided_test_params = "batch, q_head, s_ctx, s_active, s_prior_offset, fa_tile_size, lnc"
    flat_kv_strided_test_perms = [
        # LNC=1 basic tests (full s_prior)
        (4, 1, 256, 1, 0, 0, 1),   # Minimal s_ctx
        (4, 1, 512, 1, 0, 0, 1),   # Small s_ctx
        (4, 1, 1024, 5, 0, 0, 1),  # With multiple active tokens
        (4, 2, 2048, 7, 0, 0, 1),  # Multiple heads
        (4, 1, 4096, 5, 0, 0, 1),  # Larger s_ctx
        # LNC=1 FA tile tests
        (4, 1, 1024, 5, 0, 256, 1),    # First tile
        (4, 1, 1024, 5, 256, 256, 1),  # Second tile
        (4, 1, 1024, 5, 512, 256, 1),  # Third tile
        (4, 1, 1024, 5, 768, 256, 1),  # Last tile
        # LNC=2 tests - matching test_attention_tkg.py configurations
        (4, 2, 16384, 7, 0, 0, 2),     # Match attention_tkg test
        (4, 1, 4096, 5, 0, 0, 2),      # Medium s_ctx
        (4, 1, 4096, 5, 768, 256, 2),  # FA test
        (8, 8, 4096, 5, 1536, 256, 2),  # Batch sharding
    ]
    # fmt: on

    @pytest.mark.fast
    @pytest.mark.parametrize(flat_kv_strided_test_params, flat_kv_strided_test_perms)
    def test_flat_kv_strided_mask_generation(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        batch: int,
        q_head: int,
        s_ctx: int,
        s_active: int,
        s_prior_offset: int,
        fa_tile_size: int,
        lnc: int,
    ):
        """
        Test flat KV cache mask generation with strided MM1 layout (block_len=0).
        """
        compiler_args = CompilerArgs(logical_nc_config=lnc)
        self.run_gen_mask_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
            batch=batch,
            q_head=q_head,
            s_ctx=s_ctx,
            s_active=s_active,
            block_len=0,
            lnc=lnc,
            strided_mm1=True,
            s_prior_offset=s_prior_offset,
            fa_tile_size=fa_tile_size,
        )

    # ============================================================================
    # FLAT KV CACHE TESTS (block_len = 0) - NON-STRIDED MM1
    # ============================================================================

    # fmt: off
    # Test parameters for flat KV cache with non-strided MM1 layout
    flat_kv_nonstrided_test_params = "batch, q_head, s_ctx, s_active, s_prior_offset, fa_tile_size, lnc"
    flat_kv_nonstrided_test_perms = [
        # LNC=1 basic tests (full s_prior)
        (4, 1, 256, 1, 0, 0, 1),   # Minimal s_ctx
        (4, 1, 512, 1, 0, 0, 1),   # Small s_ctx
        (4, 1, 1024, 5, 0, 0, 1),  # With multiple active tokens
        # LNC=1 FA tile tests
        (4, 1, 1024, 5, 256, 256, 1),  # FA tile with offset
        # LNC=2 tests
        (4, 1, 4096, 5, 0, 0, 2),  # Medium s_ctx with LNC=2
        (4, 1, 4096, 5, 512, 512, 2),  # FA tile with offset
        (8, 8, 4096, 5, 1536, 512, 2),  # Batch sharding
    ]
    # fmt: on

    @pytest.mark.parametrize(flat_kv_nonstrided_test_params, flat_kv_nonstrided_test_perms)
    def test_flat_kv_nonstrided_mask_generation(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        batch: int,
        q_head: int,
        s_ctx: int,
        s_active: int,
        s_prior_offset: int,
        fa_tile_size: int,
        lnc: int,
    ):
        """
        Test flat KV cache mask generation with non-strided MM1 layout (block_len=0, strided_mm1=False).
        """
        compiler_args = CompilerArgs(logical_nc_config=lnc)
        self.run_gen_mask_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
            batch=batch,
            q_head=q_head,
            s_ctx=s_ctx,
            s_active=s_active,
            block_len=0,
            lnc=lnc,
            strided_mm1=False,
            s_prior_offset=s_prior_offset,
            fa_tile_size=fa_tile_size,
        )

    # ============================================================================
    # BLOCK KV CACHE TESTS (block_len > 0)
    # ============================================================================

    # fmt: off
    # Test parameters for block KV cache
    block_kv_test_params = "batch, q_head, s_ctx, s_active, block_len, s_prior_offset, fa_tile_size, lnc"
    block_kv_test_perms = [
        # LNC=1 block KV tests (full s_prior)
        (4, 1, 256, 5, 16, 0, 0, 1),
        (4, 1, 512, 5, 16, 0, 0, 1),
        (4, 1, 1024, 5, 16, 0, 0, 1),
        (4, 1, 2048, 5, 32, 0, 0, 1),  # Larger block_len
        # LNC=1 block KV FA tile tests
        (4, 1, 4096, 5, 16, 0, 2048, 1),    # First tile
        (4, 1, 4096, 5, 16, 2048, 2048, 1), # Second tile
        (4, 1, 10240, 5, 16, 8192, 2048, 1),   # Second tile (small last tile)
        # LNC=2 block KV tests - matching test_attention_tkg.py configurations
        (4, 1, 8192, 5, 16, 0, 0, 2),
        (4, 1, 4096, 5, 16, 0, 0, 2),
        (4, 1, 20480, 5, 16, 8192, 2048, 2),  # FA tile with offset
        (8, 8, 10240, 5, 16, 8192, 2048, 2),  # Batch sharding
    ]
    # fmt: on

    @pytest.mark.parametrize(block_kv_test_params, block_kv_test_perms)
    def test_block_kv_mask_generation(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        batch: int,
        q_head: int,
        s_ctx: int,
        s_active: int,
        block_len: int,
        s_prior_offset: int,
        fa_tile_size: int,
        lnc: int,
    ):
        """
        Test block KV cache mask generation (block_len>0).

        Validates the shuffled index mask generation that matches the K cache
        block layout used by the attention kernel.
        """
        compiler_args = CompilerArgs(logical_nc_config=lnc)
        self.run_gen_mask_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
            batch=batch,
            q_head=q_head,
            s_ctx=s_ctx,
            s_active=s_active,
            block_len=block_len,
            lnc=lnc,
            strided_mm1=False,  # Block KV doesn't use strided MM1
            s_prior_offset=s_prior_offset,
            fa_tile_size=fa_tile_size,
        )

    # ============================================================================
    # ACTIVE MASK TESTS - Testing _load_active_mask code path
    # ============================================================================

    def run_gen_mask_with_active_mask_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        collector: MetricsCollector,
        batch: int,
        q_head: int,
        s_ctx: int,
        s_active: int,
        block_len: int,
        lnc: int,
        strided_mm1: bool = True,
        s_prior_offset: int = 0,
        fa_tile_size: int = 0,
        dtype=np.float32,
    ):
        """Run a gen_mask_tkg test case with active_mask provided."""
        kernel_input = build_gen_mask_with_active_mask_input(
            batch=batch,
            q_head=q_head,
            s_ctx=s_ctx,
            s_active=s_active,
            block_len=block_len,
            lnc=lnc,
            strided_mm1=strided_mm1,
            dtype=dtype,
            s_prior_offset=s_prior_offset,
            fa_tile_size=fa_tile_size,
        )

        # Tolerances for mask validation (exact match expected for binary mask)
        relative_tolerance, absolute_tolerance = 0, 0

        # Create output placeholder for LazyGoldenGenerator
        output_placeholder = {"golden_mask": kernel_input["mask_out_hbm"]}

        # Create lazy golden generator function (takes no arguments)
        def create_lazy_golden() -> dict[str, np.ndarray]:
            return golden_mask_with_active_mask_output(inp=kernel_input, lnc=lnc)

        # Extract kernel-only arguments (remove metadata)
        kernel_only_input = {k: v for k, v in kernel_input.items() if not k.startswith("_")}

        test_manager.execute(
            KernelArgs(
                kernel_func=gen_mask_tkg_wrapper_with_active_mask,
                compiler_input=compiler_args,
                kernel_input=kernel_only_input,
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(
                        output_ndarray=output_placeholder,
                        lazy_golden_generator=create_lazy_golden,
                    ),
                    relative_accuracy=relative_tolerance,
                    absolute_accuracy=absolute_tolerance,
                ),
                inference_args=TKG_INFERENCE_ARGS,
            )
        )

    # fmt: off
    # Test parameters for active_mask tests (flat KV)
    # These test the _load_active_mask code path for flat KV cache
    active_mask_test_params = "batch, q_head, s_ctx, s_active, strided_mm1, s_prior_offset, fa_tile_size, lnc"
    active_mask_test_perms = [
        # Strided MM1 with active_mask (LNC=1, full s_prior)
        (4, 1, 256, 5, True, 0, 0, 1),
        (4, 2, 512, 5, True, 0, 0, 1),
        # Non-strided MM1 with active_mask (LNC=1, full s_prior)
        (4, 1, 256, 5, False, 0, 0, 1),
        (4, 2, 512, 5, False, 0, 0, 1),
        # LNC=1 FA tile tests with active_mask (tile must include active region)
        (4, 1, 1024, 5, True, 768, 256, 1),   # Last tile (where active mask applies)
        (4, 1, 1024, 5, False, 768, 256, 1),  # Last tile (where active mask applies)
        # LNC=2 sprior-sharded with active_mask (flat KV doesn't support batch-sharded active_mask)
        (4, 2, 4096, 5, True, 0, 0, 2),       # Sprior sharded: 4*2*5=40 <= 128

        # Batch-sharded LNC=2 with strided MM1 and FA tiling
        (80, 8, 256, 8, True, 0, 256, 2),

        # Strided MM1 with load1_nrows = 0 (s_active % n_sprior_tile = 0)
        # These test the edge case where only load2 path is used in _load_active_mask
        # s_active=8, s_ctx=256 -> n_sprior_tile=2, load1_nrows=8%2=0
        (4, 1, 256, 8, True, 0, 0, 1),    # load1_nrows=0: 8%2=0
        (4, 2, 256, 4, True, 0, 0, 1),    # load1_nrows=0: 4%2=0
        (80, 8, 256, 8, True, 0, 0, 1),   # Matches failing attention_tkg test: bs=80, s_a=8, s_p=256

        # Strided MM1 with load1_nrows > 0 (s_active % n_sprior_tile != 0)
        # These test the edge case where both load1 and load2 paths are used
        # s_active=7, s_ctx=256 -> n_sprior_tile=2, load1_nrows=7%2=1, load2_nrows=6
        (4, 1, 256, 7, True, 0, 0, 1),    # load1_nrows=1, load2_nrows=6
        (80, 8, 256, 7, True, 0, 0, 1),   # Large batch, odd s_active

        # Strided MM1 with s_active < n_sprior_tile (load2_nrows = 0, only load1 used)
        # s_active=1, s_ctx=256 -> n_sprior_tile=2, load1_nrows=1%2=1, load2_nrows=0
        (4, 1, 256, 1, True, 0, 0, 1),    # Minimal s_active, only load1 path
        (80, 8, 256, 1, True, 0, 0, 1),   # Large batch, minimal s_active

        # Batch-sharded LNC=2 with load1_nrows = 0 (exercises the fixed DMA stride path)
        # bs*q_head*s_active > 128 -> batch-sharded, s_active % n_sprior_tile = 0
        (80, 8, 256, 4, True, 0, 256, 2), # bs_full=160, load1_nrows=4%2=0, only load2
        (32, 8, 256, 8, True, 0, 256, 2), # bs_full=64, BQS=2048>128, load1_nrows=8%2=0

        # Batch-sharded LNC=2 with load1_nrows > 0 (both DMA paths with stride fix)
        (80, 8, 256, 7, True, 0, 256, 2), # bs_full=160, load1_nrows=7%2=1, load2_nrows=6
        (32, 8, 256, 5, True, 0, 256, 2), # bs_full=64, BQS=1280>128, load1_nrows=5%2=1

        # Batch-sharded LNC=2 with s_active=1 (only load1 path with stride fix)
        (80, 8, 256, 1, True, 0, 256, 2), # bs_full=160, load1_nrows=1, load2_nrows=0

        # Non-strided MM1 with batch-sharded LNC=2 (tests the TensorView slice path)
        (80, 8, 256, 8, False, 0, 256, 2), # Batch-sharded, non-strided, BQS=5120>128
        (32, 8, 256, 5, False, 0, 256, 2), # Batch-sharded, non-strided, BQS=1280>128

        # LNC=2 sprior-sharded with strided MM1 and load1_nrows = 0
        # bs*q_head*s_active <= 128 -> sprior-sharded
        (4, 2, 4096, 8, True, 0, 0, 2),   # BQS=64<=128, sprior-sharded, load1_nrows=8%16=8!=0
        (4, 1, 4096, 16, True, 0, 0, 2),  # BQS=64<=128, sprior-sharded, load1_nrows=16%16=0

        # FA tiling with batch-sharded LNC=2 (different tile sizes)
        # batch-sharded: s_prior_per_shard = s_ctx = 512, active region at [504, 512)
        # FA tile must cover the active region, so offset must be at end
        (80, 8, 512, 8, True, 256, 256, 2), # FA tile at [256, 512), covers active region [504, 512)

        # Boundary: BQS exactly at P_MAX threshold for sharding decision
        # bs*q_head*s_active = 128 -> NOT batch-sharded (needs > 128), sprior-sharded
        (16, 1, 4096, 8, True, 0, 0, 2),  # BQS=128, sprior-sharded
        # bs*q_head*s_active = 130 -> batch-sharded (> 128)
        (26, 1, 256, 5, True, 0, 256, 2), # BQS=130>128, batch-sharded
    ]
    # fmt: on

    @pytest.mark.parametrize(active_mask_test_params, active_mask_test_perms)
    def test_flat_kv_with_active_mask(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        batch: int,
        q_head: int,
        s_ctx: int,
        s_active: int,
        strided_mm1: bool,
        s_prior_offset: int,
        fa_tile_size: int,
        lnc: int,
    ):
        """
        Test flat KV cache mask generation with active_mask provided.

        This tests the _load_active_mask code path which loads the causal
        active mask onto the last section of the prior mask.
        """
        compiler_args = CompilerArgs(logical_nc_config=lnc)
        self.run_gen_mask_with_active_mask_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
            batch=batch,
            q_head=q_head,
            s_ctx=s_ctx,
            s_active=s_active,
            block_len=0,  # Flat KV only for active_mask tests
            lnc=lnc,
            strided_mm1=strided_mm1,
            s_prior_offset=s_prior_offset,
            fa_tile_size=fa_tile_size,
        )

    # ============================================================================
    # BLOCK KV WITH ACTIVE MASK TESTS - Testing _load_active_mask_block_kv
    # ============================================================================
    # These tests validate the batch sharding fix in _load_active_mask_block_kv.
    # Previously, batch-sharded mode with block KV silently skipped loading the
    # active mask, causing test failures.

    # fmt: off
    # Test parameters for block KV with active_mask tests
    # (batch, q_head, s_ctx, s_active, block_len, s_prior_offset, fa_tile_size, lnc)
    block_kv_active_mask_test_params = "batch, q_head, s_ctx, s_active, block_len, s_prior_offset, fa_tile_size, lnc"
    block_kv_active_mask_test_perms = [
        # LNC=1 block KV with active_mask (full s_prior)
        (4, 1, 2048, 5, 16, 0, 0, 1),
        (4, 2, 4096, 5, 16, 0, 0, 1),
        # LNC=2 sprior-sharded (bs * q_head * s_active <= P_MAX)
        (4, 1, 8192, 5, 16, 0, 0, 2),  # Sprior sharded, full s_prior
        (4, 1, 4096, 5, 16, 0, 0, 2),  # Sprior sharded, smaller s_ctx
        # LNC=2 batch-sharded (bs * q_head * s_active > P_MAX) - these were failing before the fix!
        # These test cases replicate the previously failing attention_tkg test vectors:
        # - [64, 8, 1, 2048, 2048, 128, 16, True, True, ...] (bs*q*s=512 > 128)
        (64, 8, 2048, 1, 16, 0, 0, 2),  # Batch sharding: 64*8*1=512 > 128 - matches failing test!
        (8, 8, 4096, 5, 16, 0, 0, 2),   # Batch sharding: 8*8*5=320 > 128
        (4, 8, 4096, 7, 16, 0, 0, 2),   # Batch sharding: 4*8*7=224 > 128
        # Block KV FA tile with active_mask (tile must include active region)
        # For FA tile tests, active positions are at END of s_prior_per_shard
        # sprior-sharded: s_prior_per_shard = s_ctx/lnc = 8192/2 = 4096, so tile [2048, 4096] is at end
        (4, 1, 8192, 5, 16, 2048, 2048, 2),  # FA tile, sprior sharded
        # batch-sharded: s_prior_per_shard = s_ctx = 8192, so tile must be at [6144, 8192]
        (8, 8, 8192, 5, 16, 6144, 2048, 2),  # FA tile, batch sharded
    ]
    # fmt: on

    @pytest.mark.parametrize(block_kv_active_mask_test_params, block_kv_active_mask_test_perms)
    def test_block_kv_with_active_mask(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        batch: int,
        q_head: int,
        s_ctx: int,
        s_active: int,
        block_len: int,
        s_prior_offset: int,
        fa_tile_size: int,
        lnc: int,
    ):
        """
        Test block KV cache mask generation with active_mask provided.

        This tests the _load_active_mask_block_kv code path which was fixed
        to properly handle batch sharding. Before the fix, batch-sharded mode
        would silently skip loading active_mask, causing test failures.
        """
        compiler_args = CompilerArgs(logical_nc_config=lnc)
        self.run_gen_mask_with_active_mask_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
            batch=batch,
            q_head=q_head,
            s_ctx=s_ctx,
            s_active=s_active,
            block_len=block_len,
            lnc=lnc,
            strided_mm1=False,  # Block KV doesn't use strided MM1
            s_prior_offset=s_prior_offset,
            fa_tile_size=fa_tile_size,
        )


# ============================================================================
# Helper functions for active_mask tests
# ============================================================================


def gen_mask_tkg_wrapper_with_active_mask(
    pos_ids_hbm: nl.ndarray,
    mask_out_hbm: nl.ndarray,
    active_mask_hbm: nl.ndarray,
    bs: int,
    q_head: int,
    s_active: int,
    is_s_prior_sharded: bool,
    is_batch_sharded: bool,
    s_prior_per_shard: int,
    s_prior_offset: int = 0,
    block_len: int = 0,
    strided_mm1: bool = True,
) -> nl.ndarray:
    """
    Wrapper kernel that handles HBM↔SBUF transfers for testing gen_mask_tkg with active_mask.

    Similar to gen_mask_tkg_wrapper but also passes active_mask to the kernel.
    """
    # Get sharding info
    _, lnc, shard_id = get_verified_program_sharding_info("gen_mask_wrapper_active", (0, 1))

    # Extract dimensions from mask_out_hbm shape based on lnc
    if lnc == 1:
        _, n_sprior_tile_per_shard, _, _, _ = mask_out_hbm.shape
    else:
        _, _, n_sprior_tile_per_shard, _, _, _ = mask_out_hbm.shape

    # Initialize SBUF manager
    sbm = SbufManager(
        0, P_MAX * n_sprior_tile_per_shard * bs * q_head * s_active * 8, Logger("gen_mask_wrapper"), use_auto_alloc=True
    )
    sbm.open_scope(name="gen_mask_wrapper_active")

    # Allocate SBUF tensors
    pos_ids_sbuf = sbm.alloc_stack((P_MAX, bs * s_active), dtype=pos_ids_hbm.dtype, buffer=nl.sbuf, name="pos_ids_sbuf")
    mask_out_sbuf = sbm.alloc_stack(
        (P_MAX, n_sprior_tile_per_shard, bs, q_head, s_active),
        dtype=mask_out_hbm.dtype,
        buffer=nl.sbuf,
        name="mask_out_sbuf",
    )

    # Copy pos_ids from HBM to SBUF
    nisa.dma_copy(dst=pos_ids_sbuf, src=pos_ids_hbm)

    # Only pass active_mask to the last shard in sprior-sharded mode,
    # matching attention_tkg behavior: active positions live at the end of
    # the full s_prior sequence, which only the last shard processes.
    # For LNC=1 or batch-sharded mode, always pass the mask.
    load_active_mask = not (is_s_prior_sharded and shard_id < lnc - 1)

    # Call the actual gen_mask_tkg kernel with active_mask
    gen_mask_tkg(
        pos_ids=pos_ids_sbuf,
        mask_out=mask_out_sbuf,
        bs=bs,
        q_head=q_head,
        s_active=s_active,
        is_s_prior_sharded=is_s_prior_sharded,
        s_prior_per_shard=s_prior_per_shard,
        s_prior_offset=s_prior_offset,
        block_len=block_len,
        strided_mm1=strided_mm1,
        active_mask=active_mask_hbm if load_active_mask else None,
        sbm=sbm,
        is_batch_sharded=is_batch_sharded,
    )

    # Create output tensor with name matching golden output key
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


def build_gen_mask_with_active_mask_input(
    batch: int,
    q_head: int,
    s_ctx: int,
    s_active: int,
    block_len: int,
    lnc: int,
    strided_mm1: bool,
    dtype,
    s_prior_offset: int = 0,
    fa_tile_size: int = 0,
):
    """Build kernel inputs for gen_mask_tkg test with active_mask."""
    # For active_mask tests with FA tiling, the tile must completely include the active region
    # (last s_active positions). Otherwise the kernel will incorrectly load active_mask.
    if fa_tile_size > 0:
        # Determine s_prior_per_shard based on sharding mode
        cfg = AttnTKGConfig(bs=batch, q_head=q_head, s_active=s_active, curr_sprior=s_ctx)
        sprior_sharded = is_s_prior_sharded(cfg, P_MAX) if lnc > 1 else False
        s_prior_per_shard = s_ctx // lnc if sprior_sharded else s_ctx
        tile_end = s_prior_offset + fa_tile_size
        active_region_start = s_prior_per_shard - s_active
        assert tile_end >= s_prior_per_shard and s_prior_offset <= active_region_start, (
            f"FA tile (offset={s_prior_offset}, end={tile_end}) must completely include active region "
            f"[{active_region_start}, {s_prior_per_shard}) when active_mask is provided"
        )

    # Start with base input
    base_input = build_gen_mask_kernel_input(
        batch=batch,
        q_head=q_head,
        s_ctx=s_ctx,
        s_active=s_active,
        block_len=block_len,
        lnc=lnc,
        strided_mm1=strided_mm1,
        dtype=dtype,
        s_prior_offset=s_prior_offset,
        fa_tile_size=fa_tile_size,
    )

    # Determine batch sharding mode
    cfg = AttnTKGConfig(bs=batch, q_head=q_head, s_active=s_active, curr_sprior=s_ctx)
    batch_sharded = is_batch_sharded_fn(cfg, P_MAX) if lnc > 1 else False

    # Generate active_mask: lower triangular causal mask
    # Shape expected by kernel: [s_active, bs_full, q_head, s_active]
    # For batch-sharded mode, bs_full = bs * lnc (each NC processes different batch portion)
    # For sprior-sharded mode, bs_full = bs (both NCs process same batches)
    bs_full = batch * lnc if batch_sharded else batch
    active_mask = numpy_gen_attention_active_mask(shape=(bs_full, q_head, s_active, s_active), transposed=True).astype(
        dtype
    )

    base_input["active_mask_hbm"] = active_mask
    base_input["is_batch_sharded"] = batch_sharded
    return base_input


def golden_mask_with_active_mask_output(
    inp: dict[str, Any],
    lnc: int,
) -> dict[str, np.ndarray]:
    """
    Compute golden mask output with active_mask merged.

    Uses numpy_gen_attention_cache_mask with unify_for_cascaded=True to get
    the full mask including the active portion.

    Note: numpy_gen_attention_cache_mask with unify_for_cascaded=True returns
    shape (S_ctx, batch, num_heads, S_tkg), which needs to be transposed back
    to (batch, num_heads, S_tkg, S_ctx) for the transform function.
    """
    cache_lens = inp["_cache_lens"]
    batch = inp["bs"]
    q_head = inp["q_head"]
    s_active = inp["s_active"]
    s_ctx = inp["_s_ctx_total"]
    strided_mm1 = inp["strided_mm1"]
    original_block_len = inp["_original_block_len"]
    adjusted_block_len = inp["_adjusted_block_len"]
    n_sprior_tile_per_shard = inp["_n_sprior_tile_per_shard"]
    s_prior_offset = inp.get("s_prior_offset", 0)
    dtype = inp["pos_ids_hbm"].dtype

    # Determine sharding mode using the same logic as the kernel
    cfg = AttnTKGConfig(bs=batch, q_head=q_head, s_active=s_active, curr_sprior=s_ctx)
    sprior_sharded = is_s_prior_sharded(cfg, P_MAX) if lnc > 1 else False

    def transform_mask_to_kernel_format(
        mask: np.ndarray, n_tiles: int, block_len: int, strided: bool, shard_offset: int = 0
    ) -> np.ndarray:
        """Transform mask to kernel output format."""
        s_prior_shard = n_tiles * P_MAX

        if block_len > 0:
            # Block KV: slice on the fold dimension
            fold_size = block_len * P_MAX
            num_folds_total = mask.shape[-1] // fold_size
            num_folds_shard = s_prior_shard // fold_size
            fold_start = shard_offset // fold_size
            mask_by_fold = mask.reshape(batch, q_head, s_active, num_folds_total, block_len, P_MAX)
            mask_shard = mask_by_fold[:, :, :, fold_start : fold_start + num_folds_shard, :, :]
            result = mask_shard.reshape(batch, q_head, s_active, num_folds_shard * block_len, P_MAX)
            result = result.transpose(4, 3, 0, 1, 2)
        else:
            # Flat KV: extract the relevant s_prior portion for this shard
            mask_shard = mask[:, :, :, shard_offset : shard_offset + s_prior_shard]

            if strided:
                result = mask_shard.reshape(batch, q_head, s_active, P_MAX, n_tiles)
                result = result.transpose(3, 4, 0, 1, 2)
            else:
                result = mask_shard.reshape(batch, q_head, s_active, n_tiles, P_MAX)
                result = result.transpose(4, 3, 0, 1, 2)

        return result.astype(dtype)

    # Generate full mask with active portion using unify_for_cascaded=True
    # Returns shape (S_ctx, batch, num_heads, S_tkg) when unify_for_cascaded=True
    full_mask_transposed = numpy_gen_attention_cache_mask(
        cache_len=cache_lens,
        batch=batch,
        num_heads=q_head,
        S_tkg=s_active,
        S_ctx=s_ctx,
        lnc=lnc,
        block_len=original_block_len,
        unify_for_cascaded=True,  # Include active mask in the output
    )
    # Transpose back to (batch, num_heads, S_tkg, S_ctx) for transform function
    full_mask = full_mask_transposed.transpose(1, 2, 3, 0)

    if lnc == 1:
        # For LNC=1, s_prior_offset is the FA tile offset within the full s_prior
        mask_out = transform_mask_to_kernel_format(
            full_mask, n_sprior_tile_per_shard, adjusted_block_len, strided_mm1, shard_offset=s_prior_offset
        )
    elif sprior_sharded:
        # Sprior-sharded: each shard processes different s_prior portion
        s_prior_per_shard = s_ctx // lnc
        mask_out = np.zeros((lnc, P_MAX, n_sprior_tile_per_shard, batch, q_head, s_active), dtype=dtype)

        for lnc_shard_idx in range(lnc):
            # NC offset + FA tile offset
            shard_offset = lnc_shard_idx * s_prior_per_shard + s_prior_offset
            shard_mask = transform_mask_to_kernel_format(
                full_mask, n_sprior_tile_per_shard, adjusted_block_len, strided_mm1, shard_offset=shard_offset
            )
            mask_out[lnc_shard_idx] = shard_mask
    else:
        # Batch-sharded or no sharding: both shards process full s_prior (or FA tile), produce identical output
        mask_single = transform_mask_to_kernel_format(
            full_mask, n_sprior_tile_per_shard, adjusted_block_len, strided_mm1, shard_offset=s_prior_offset
        )

        mask_out = np.zeros((lnc, P_MAX, n_sprior_tile_per_shard, batch, q_head, s_active), dtype=dtype)
        for lnc_shard_idx in range(lnc):
            mask_out[lnc_shard_idx] = mask_single

    return {"golden_mask": mask_out}
