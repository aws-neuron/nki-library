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

"""Integration tests for attention_reduce utilities.

Exercises reduce_one_batch (online-softmax rescale + per-group output
correction, writing combined stats back to HBM) followed by normalize_one_batch
(final division by the softmax denominator and LSE computation for training).
The two functions together cover the full unnormalized-accumulation reduction
path used by the segmented-attention kernels.
"""

from typing import final

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.core.utils.attention_reduce import (
    _MAX_FREE_TILES,
    normalize_one_batch,
    reduce_one_batch,
)
from nkilib_src.nkilib.core.utils.modular_allocator import ModularAllocator

from test.utils.common_dataclasses import CompilerArgs
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# =============================================================================
# Test Kernel
# =============================================================================


@nki.jit
def kernel_attention_reduce(
    o_prev_hbm,
    neg_max_prev_hbm,
    sum_prev_hbm,
    o_curr_hbm,
    neg_max_curr_hbm,
    sum_curr_hbm,
    o_out,
    lse_out,
    d,
    num_grps,
):
    """Reduce prev+curr attention stats for batch 0, then normalize and emit LSE."""
    sb_p = nl.tile_size.pmax
    softmax_pat = [[num_grps, sb_p], [1, num_grps]]
    o_pat = [[d, sb_p], [1, d]]
    num_free = min(num_grps, _MAX_FREE_TILES)

    # reduce_one_batch overwrites the "prev" HBM tensors in place. Copy them into
    # internal HBM scratch so the caller-provided inputs stay pristine for the
    # torch reference (the framework shares plain input arrays with torch_ref).
    o_prev_scratch = nl.ndarray(o_prev_hbm.shape, dtype=o_prev_hbm.dtype, buffer=nl.hbm)
    neg_max_prev_scratch = nl.ndarray(neg_max_prev_hbm.shape, dtype=neg_max_prev_hbm.dtype, buffer=nl.hbm)
    sum_prev_scratch = nl.ndarray(sum_prev_hbm.shape, dtype=sum_prev_hbm.dtype, buffer=nl.hbm)
    nisa.dma_copy(dst=o_prev_scratch, src=o_prev_hbm)
    nisa.dma_copy(dst=neg_max_prev_scratch, src=neg_max_prev_hbm)
    nisa.dma_copy(dst=sum_prev_scratch, src=sum_prev_hbm)

    allocator = ModularAllocator()
    neg_max_prev_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
    sum_prev_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
    neg_max_curr_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
    sum_curr_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
    o_prev_sb = allocator.alloc_sbuf_tensor(
        shape=(sb_p, d), dtype=nl.float32, block_dim=[num_grps], num_free_tiles=[num_free]
    )
    o_curr_sb = allocator.alloc_sbuf_tensor(
        shape=(sb_p, d), dtype=nl.float32, block_dim=[num_grps], num_free_tiles=[num_free]
    )
    o_new_sb = allocator.alloc_sbuf_tensor(
        shape=(sb_p, d), dtype=nl.float32, block_dim=[num_grps], num_free_tiles=[num_free]
    )
    reduce_batch_addr = allocator.get_current_address()

    reduce_one_batch(
        o_prev_scratch,
        neg_max_prev_scratch,
        sum_prev_scratch,
        o_curr_hbm,
        neg_max_curr_hbm,
        sum_curr_hbm,
        0,
        0,
        num_grps,
        d,
        num_grps,
        sb_p,
        softmax_pat,
        o_pat,
        neg_max_prev_sb,
        sum_prev_sb,
        neg_max_curr_sb,
        sum_curr_sb,
        o_prev_sb,
        o_curr_sb,
        o_new_sb,
        reduce_batch_addr,
        allocator,
    )

    allocator.set_current_address(reduce_batch_addr)
    normalize_one_batch(
        o_prev_scratch,
        neg_max_prev_scratch,
        sum_prev_scratch,
        o_out,
        lse_out,
        0,
        d,
        num_grps,
        True,
        nl.float32,
        allocator,
    )
    return o_out, lse_out


@nki.jit
def kernel_normalize_inference(
    o_prev_hbm,
    sum_prev_hbm,
    o_out,
    d,
    num_grps,
):
    """Normalize a pre-accumulated batch with training=False and an explicit
    grp_end (=num_grps), so no LSE is emitted and neg_max is unused."""
    allocator = ModularAllocator()
    # neg_max / lse are unread when training=False; pass sum_prev_hbm as a stand-in.
    normalize_one_batch(
        o_prev_hbm,
        sum_prev_hbm,
        sum_prev_hbm,
        o_out,
        sum_prev_hbm,
        0,
        d,
        num_grps,
        False,
        nl.float32,
        allocator,
        0,
        num_grps,
    )
    return o_out


# =============================================================================
# Torch Reference
# =============================================================================


def normalize_inference_torch_ref(o_prev_hbm, sum_prev_hbm, o_out, d, num_grps):
    """Reference: o_out[g] = o_unnorm[g] / S[:, g] per 128-row group."""
    sb_p = 128
    o_prev = o_prev_hbm.to(torch.float32)
    denom = sum_prev_hbm.to(torch.float32)
    o_result = torch.zeros_like(o_prev)
    for g in range(num_grps):
        rows = slice(g * sb_p, (g + 1) * sb_p)
        o_result[0, rows, :] = o_prev[0, rows, :] / denom[0, :, g : g + 1]
    return {"o_out": o_result}


def attention_reduce_torch_ref(
    o_prev_hbm,
    neg_max_prev_hbm,
    sum_prev_hbm,
    o_curr_hbm,
    neg_max_curr_hbm,
    sum_curr_hbm,
    o_out,
    lse_out,
    d,
    num_grps,
):
    """Online-softmax reduce + normalize reference (unnormalized accumulation)."""
    sb_p = 128
    o_prev = o_prev_hbm.to(torch.float32)
    o_curr = o_curr_hbm.to(torch.float32)
    nmp = neg_max_prev_hbm.to(torch.float32)
    nmc = neg_max_curr_hbm.to(torch.float32)
    sp = sum_prev_hbm.to(torch.float32)
    sc = sum_curr_hbm.to(torch.float32)

    nm_new = torch.minimum(nmp, nmc)
    corr_p = torch.exp(nm_new - nmp)
    corr_c = torch.exp(nm_new - nmc)
    sum_new = sp * corr_p + sc * corr_c

    o_result = torch.zeros_like(o_prev)
    lse_result = torch.zeros((1, sb_p, num_grps), dtype=torch.float32)
    for g in range(num_grps):
        rows = slice(g * sb_p, (g + 1) * sb_p)
        o_new = o_prev[0, rows, :] * corr_p[0, :, g : g + 1] + o_curr[0, rows, :] * corr_c[0, :, g : g + 1]
        o_result[0, rows, :] = o_new / sum_new[0, :, g : g + 1]
        lse_result[0, :, g] = -nm_new[0, :, g] + torch.log(sum_new[0, :, g])

    return {"o_out": o_result, "lse_out": lse_result}


# =============================================================================
# Tests
# =============================================================================


@final
@pytest_test_metadata(name="AttentionReduce")
@pytest_marks(["attention_reduce"])
class TestAttentionReduce:
    """Tests for reduce_one_batch + normalize_one_batch with full data validation."""

    _ABBREVS = {
        "num_grps": "g",
        "d": "d",
    }

    @pytest_parametrize(
        "num_grps,d",
        [
            pytest.param(2, 8, marks=pytest.mark.fast),
            (1, 16),
            (3, 16),
        ],
        abbrevs=_ABBREVS,
    )
    def test_attention_reduce(self, test_manager, platform_target, num_grps, d):
        """Reduce prev+curr stats then normalize + emit LSE over all groups."""
        sb_p = 128
        seqlen = num_grps * sb_p

        def input_generator(test_config):
            np.random.seed(42)
            o_prev = np.random.randn(1, seqlen, d).astype(np.float32)
            o_curr = np.random.randn(1, seqlen, d).astype(np.float32)
            neg_max_prev = (-np.abs(np.random.randn(1, sb_p, num_grps))).astype(np.float32)
            neg_max_curr = (-np.abs(np.random.randn(1, sb_p, num_grps))).astype(np.float32)
            sum_prev = (np.abs(np.random.randn(1, sb_p, num_grps)) + 0.5).astype(np.float32)
            sum_curr = (np.abs(np.random.randn(1, sb_p, num_grps)) + 0.5).astype(np.float32)
            return {
                "o_prev_hbm": o_prev,
                "neg_max_prev_hbm": neg_max_prev,
                "sum_prev_hbm": sum_prev,
                "o_curr_hbm": o_curr,
                "neg_max_curr_hbm": neg_max_curr,
                "sum_curr_hbm": sum_curr,
                "o_out.must_alias_input": np.zeros((1, seqlen, d), dtype=np.float32),
                "lse_out.must_alias_input": np.zeros((1, sb_p, num_grps), dtype=np.float32),
                "d": d,
                "num_grps": num_grps,
            }

        def output_tensors(kernel_input):
            return {
                "o_out": np.zeros((1, seqlen, d), dtype=np.float32),
                "lse_out": np.zeros((1, sb_p, num_grps), dtype=np.float32),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_attention_reduce,
            torch_ref=torch_ref_wrapper(attention_reduce_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-3,
            atol=1e-3,
        )

    @pytest.mark.fast
    @pytest_parametrize(
        "num_grps,d",
        [
            (2, 8),
        ],
        abbrevs=_ABBREVS,
    )
    def test_normalize_inference_explicit_group_range(self, test_manager, platform_target, num_grps, d):
        """Normalize with training=False and an explicit grp_end: exercises the
        inference path (no LSE) over an explicitly-bounded group range."""
        sb_p = 128
        seqlen = num_grps * sb_p

        def input_generator(test_config):
            np.random.seed(42)
            o_unnorm = np.random.randn(1, seqlen, d).astype(np.float32)
            denom = (np.abs(np.random.randn(1, sb_p, num_grps)) + 0.5).astype(np.float32)
            return {
                "o_prev_hbm": o_unnorm,
                "sum_prev_hbm": denom,
                "o_out.must_alias_input": np.zeros((1, seqlen, d), dtype=np.float32),
                "d": d,
                "num_grps": num_grps,
            }

        def output_tensors(kernel_input):
            return {"o_out": np.zeros((1, seqlen, d), dtype=np.float32)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_normalize_inference,
            torch_ref=torch_ref_wrapper(normalize_inference_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-3,
            atol=1e-3,
        )
