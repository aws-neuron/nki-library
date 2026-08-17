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

"""Integration tests for segmented attention with block-based KV cache."""

from typing import final

import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.attention.attention_segmented_cte import attention_segmented_cte
from nkilib_src.nkilib.core.attention.attention_segmented_cte_torch import attention_segmented_cte_torch_ref

try:
    from test.integration.nkilib.core.attention.test_attention_segmented_cte_model_config import (
        segmented_attention_cte_model_configs,
    )
except ImportError:
    # SKIP_MODEL_TESTS=1 raises ImportError from the model_config module.
    # Fall back to an empty dict so downstream model-driven parametrize becomes empty.
    segmented_attention_cte_model_configs = {}
from test.integration.nkilib.utils.tensor_generators import np_random_sample
from test.utils.common_dataclasses import (
    CompilerArgs,
    ModelTestType,
    Platforms,
    prepare_model_parametrize,
)
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def generate_inputs(
    bs: int,
    num_q_heads: int,
    num_kv_heads: int,
    block_size: int,
    prior_seg_size: int,
    head_dim: int,
    prior_tokens: int,
    tp_q: bool,
    tp_out: bool,
    dtype,
    sliding_window: int | None = None,
    seqlen_q: int | None = None,
    use_sink: bool = False,
    sink_value: float | None = None,
    active_seqlen: int | None = None,
    k_pre_transposed: bool = False,
    fp8_packed: bool = False,
    k_scale_val: float | None = None,
    v_scale_val: float | None = None,
    kv_dtype=None,
):
    """Generate input tensors for segmented attention test.

    sink_value: if not None, overrides the random sink with a constant scalar
        (broadcast to shape (bs_q, 1)). Used to reproduce bugs sensitive to
        sink magnitude (e.g. sink ≈ 2.0).
    active_seqlen: if not None, zero-fills Q past this position so the caller
        can simulate a padded input where only the first active_seqlen rows
        are meaningful. Does not shrink the kernel's working size.
    k_pre_transposed: if True, reshape k_cache into (num_blocks*num_kv_heads,
        head_dim, block_size) layout — exercises the kernel's transposed-K path.
    k_scale_val / v_scale_val: if not None, populate k_scale / v_scale
        dequantization tensors ((128, 1) fp32) for the FP8 KV path.
    kv_dtype: if not None, use this dtype for K/V cache tensors (e.g.
        nl.float8_e4m3 for true fp8 KV cache). Q remains in `dtype`.
    """
    np.random.seed(42)
    random_gen = np_random_sample()

    bs_q = bs * num_q_heads
    if seqlen_q is None:
        seqlen_q = prior_seg_size
    active_tokens = seqlen_q

    # Calculate total blocks needed
    num_prior_blocks = prior_tokens // block_size
    num_active_blocks = active_tokens // block_size
    total_blocks = num_prior_blocks + num_active_blocks

    # max_blocks_per_seq only needs to cover the real blocks (prior + active).
    # The kernel pads block_tables internally when seqlen_q < prior_seg_size to
    # cover the fused partial-prior helper's traced speculative one-past read.
    num_blocks_per_seg = prior_seg_size // block_size
    num_active_blocks_per_seg = seqlen_q // block_size
    prior_segs = (prior_tokens + prior_seg_size - 1) // prior_seg_size if prior_tokens > 0 else 0
    max_blocks_per_seq = prior_segs * num_blocks_per_seg + num_active_blocks_per_seg

    # Generate Q tensor
    q = random_gen(shape=(bs_q, seqlen_q, head_dim) if tp_q else (bs_q, head_dim, seqlen_q), dtype=dtype)

    # Zero-fill the padded region of Q when active_seqlen < seqlen_q, so that
    # downstream tests can treat the first active_seqlen rows as "real" input
    # and ignore outputs beyond that prefix.
    if active_seqlen is not None and active_seqlen < seqlen_q:
        q_fp = dt.static_cast(q, np.float32)
        if tp_q:
            q_fp[:, active_seqlen:, :] = 0.0
        else:
            q_fp[:, :, active_seqlen:] = 0.0
        q = dt.static_cast(q_fp, dtype)

    # Generate block-based KV cache with enough blocks
    # KV cache layout: (num_blocks, num_kv_head, block_size, head_dim)
    cache_dtype = kv_dtype if kv_dtype is not None else dtype
    k_cache = random_gen(shape=(total_blocks, num_kv_heads, block_size, head_dim), dtype=cache_dtype)
    v_cache = random_gen(shape=(total_blocks, num_kv_heads, block_size, head_dim), dtype=cache_dtype)

    # Generate block tables: [prior_blocks..., active_blocks..., duplicate-block-0 padding]
    # Format: [0, 1, ..., prior_blocks-1, prior_blocks, ..., total_blocks-1, 0, 0, ...]
    # Where blocks [0...prior_blocks-1] are prior KV and [prior_blocks...total_blocks-1] are active KV.
    # The tail [total_blocks:max_blocks_per_seq] is padded with valid block index 0
    # instead of -1. If a kernel-traced DMA fires speculatively at runtime with
    # an index from the tail, k_cache[0] is an in-bounds real block. The
    # attention mask excludes those positions from final output, so
    # correctness is preserved. The torch reference uses `total_blocks =
    # (prior_tokens + seqlen_q) // block_size` to determine the real range.
    block_tables = np.zeros((bs, max_blocks_per_seq), dtype=np.int32)

    # Fill [0..total_blocks) with sequential real block indices.
    block_tables[:, :total_blocks] = np.arange(total_blocks, dtype=np.int32)

    block_tables = dt.static_cast(block_tables, nl.int32)

    # Prior tokens tensor
    prior_tokens_tensor = dt.static_cast(np.full(shape=(1, 1), fill_value=prior_tokens, dtype=np.int32), nl.int32)

    # Sink tensor: constant override takes priority over the default random [0,1)
    # sampling so we can exercise sink magnitudes outside the default range.
    if sink_value is not None:
        sink = dt.static_cast(np.full(shape=(bs_q, 1), fill_value=sink_value, dtype=np.float32), dtype)
    elif sliding_window is not None or use_sink:
        sink = random_gen(shape=(bs_q, 1), dtype=dtype)
    else:
        sink = None

    # Transposed K cache layout for the fused KV-dequant / pre-transposed path.
    if fp8_packed:
        # Pack K cache head-major: (num_blocks, num_kv_heads, block_size, head_dim) fp8
        # -> (num_blocks, num_kv_heads, block_size//2, head_dim, 2) fp8
        # Position 2i goes to [..., 0], position 2i+1 goes to [..., 1].
        k_flat = k_cache.reshape(total_blocks, num_kv_heads, block_size, head_dim)
        # Split even/odd positions along block_size: (num_blocks, num_kv_heads, block_size//2, head_dim)
        k_even = k_flat[:, :, 0::2, :]  # positions 0, 2, 4, ...
        k_odd = k_flat[:, :, 1::2, :]  # positions 1, 3, 5, ...
        # Stack on new last axis: (num_blocks, num_kv_heads, block_size//2, head_dim, 2)
        k_packed = np.stack([k_even, k_odd], axis=-1)
        k_cache = dt.static_cast(k_packed, cache_dtype)
    elif k_pre_transposed:
        num_blocks_k = k_cache.shape[0]
        k_tp = k_cache.reshape(num_blocks_k * num_kv_heads, block_size, head_dim)
        k_tp = np.transpose(k_tp, (0, 2, 1))
        k_cache = dt.static_cast(k_tp, cache_dtype)

    k_scale = np.full((128, 1), k_scale_val, dtype=np.float32) if k_scale_val is not None else None
    v_scale = np.full((128, 1), v_scale_val, dtype=np.float32) if v_scale_val is not None else None

    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "block_tables": block_tables,
        "prior_tokens": prior_tokens_tensor,
        "block_size": block_size,
        "prior_seg_size": prior_seg_size,
        "scale": 1.0,
        "tp_q": tp_q,
        "tp_out": tp_out,
        "sliding_window": sliding_window,
        "sink": sink,
        "num_q_heads": num_q_heads,
        "k_pre_transposed": k_pre_transposed,
        "fp8_packed": fp8_packed,
        "k_scale": k_scale,
        "v_scale": v_scale,
    }


def output_tensors(kernel_input, bs_q, prior_seg_size, head_dim, tp_out, dtype):
    """Generate output tensor descriptor."""
    if tp_out:
        return {"out": np.zeros((bs_q, head_dim, prior_seg_size), dtype=dtype)}
    else:
        return {"out": np.zeros((bs_q, prior_seg_size, head_dim), dtype=dtype)}


# Shared head-ratio lists for matrix tests. Each entry is (num_q_heads, num_kv_heads).
# GQA_HEAD_RATIOS: model/sharding-specific ratios seen in production (Llama3, gpt-oss).
# HEAD_RATIOS: num_q_heads coverage test — kept narrow (MHA + production GQA)
# to bound test-matrix size and sweep time.
GQA_HEAD_RATIOS = [
    (8, 1),  # 8Q:1KV — e.g., Llama-70B/-405B sharded configs
    (1, 1),  # 1Q:1KV — single-head / min shard
]

HEAD_RATIOS = [
    (1, 1),  # MHA single head
    (8, 1),  # GQA 8:1
]


def _block_aligned(value, block_size):
    return value > 0 and value % block_size == 0


def _default_skip(*, q_active, prior_seg_size, prior_tokens, block_size, num_q_heads, lnc, **_):
    """Default constraint-based skip rules for _build_perms.

    - q_active, prior_seg_size, prior_tokens must be multiples of block_size.
    - q_active must be at least one block.
    - LNC2 + odd num_q_heads needs q_active >= 2 * block_size (50/50 sequence shard).
    - Total sequence length prior_tokens + q_active must fit within 128k context.
    """
    if q_active % block_size != 0 or q_active < block_size:
        return True
    if prior_seg_size % block_size != 0 or prior_seg_size < block_size:
        return True
    if prior_tokens % block_size != 0:
        return True
    if lnc == 2 and num_q_heads % 2 == 1 and q_active < 2 * block_size:
        return True
    if prior_tokens + q_active > 131072:
        return True
    return False


def _prior_tokens_full(prior_seg_size, q_active, max_context=131072):
    """Prior-token sweep covering 0 / partial / full prior at 32k and 128k total length.

    Total sequence length = prior_tokens + q_active. Candidates:
      - 0                                       (no prior segments)
      - prior_seg_size + prior_seg_size // 2    (one full prior segment + a partial;
                                                  exercises both the full-prior loop
                                                  AND the partial-prior branch)
      - 32768 - q_active                        (total = 32k)
      - max_context - q_active                  (total = 128k)

    Filters negatives, dedups, sorts.
    """
    candidates = [
        0,
        prior_seg_size + prior_seg_size // 2,
        32768 - q_active,
        max_context - q_active,
    ]
    return sorted({p for p in candidates if p >= 0})


def _build_perms(
    *,
    q_actives,
    prior_seg_sizes,
    block_sizes,
    head_ratios,
    head_dims,
    prior_fn=_prior_tokens_full,
    tp_q=True,
    tp_out=True,
    dtype=nl.bfloat16,
    lnc=2,
    sliding_window=None,
    use_sink=False,
    sink_value=None,
    extra_cols=None,
    extra_col_variants=None,
    extra_col_names=(),
    skip=_default_skip,
    is_fast=lambda **_: False,
):
    """Cartesian product over the provided axes with constraint-based skipping.

    Each yielded row matches the canonical signature:

        bs, num_q_heads, num_kv_heads, block_size, prior_seg_size,
        head_dim, prior_tokens, tp_q, tp_out, dtype, q_active[, *extra]

    where `extra` comes from either `extra_cols` (a single fixed tuple
    appended to every row) or `extra_col_variants` (a list of tuples;
    each tuple is its own row, expanding the cartesian product). Pass
    `extra_col_names` so `is_fast` receives the variant values as
    kwargs (e.g., `extra_col_names=("k_scale_val", "v_scale_val")`).
    Rows that pass `is_fast(**kwargs)` are wrapped with the fast pytest mark.
    Default: no rows are marked fast — opt in per callsite via `is_fast=`.
    """
    if extra_cols is not None and extra_col_variants is not None:
        raise ValueError("pass either extra_cols or extra_col_variants, not both")
    variants = extra_col_variants if extra_col_variants is not None else [extra_cols]
    rows = []
    for q_active in q_actives:
        for prior_seg_size in prior_seg_sizes:
            priors = prior_fn(prior_seg_size, q_active)
            for prior_tokens in priors:
                for block_size in block_sizes:
                    for num_q_heads, num_kv_heads in head_ratios:
                        for head_dim in head_dims:
                            ctx = {
                                "q_active": q_active,
                                "prior_seg_size": prior_seg_size,
                                "prior_tokens": prior_tokens,
                                "block_size": block_size,
                                "num_q_heads": num_q_heads,
                                "num_kv_heads": num_kv_heads,
                                "head_dim": head_dim,
                                "lnc": lnc,
                            }
                            if skip(**ctx):
                                continue
                            base_row = [
                                1,  # bs
                                num_q_heads,
                                num_kv_heads,
                                block_size,
                                prior_seg_size,
                                head_dim,
                                prior_tokens,
                                tp_q,
                                tp_out,
                                dtype,
                                q_active,
                            ]
                            for extra in variants:
                                row = list(base_row)
                                if extra is not None:
                                    row.extend(extra)
                                variant_ctx = ctx | dict(zip(extra_col_names, extra, strict=True)) if extra else ctx
                                if is_fast(**variant_ctx):
                                    rows.append(pytest.param(*row, marks=pytest.mark.fast))
                                else:
                                    rows.append(pytest.param(*row))
    return rows


# Canonical parametrize signature for _build_perms-driven tests.
# Extra columns (e.g., sliding_window, attn_mode) follow q_active.
_MATRIX_PARAMS_BASE = (
    "bs, num_q_heads, num_kv_heads, block_size, prior_seg_size, head_dim, prior_tokens, tp_q, tp_out, dtype, q_active"
)


# --- Fast-slice predicates -------------------------------------------------
# Predicates are module-level so they can be referenced inside class-body
# _build_perms() calls (class locals aren't visible inside such expressions).
# _build_perms defaults to no-fast; methods opt in via `is_fast=<predicate>`.


def _lnc1_slice_min_set_is_fast(*, num_q_heads, num_kv_heads, prior_seg_size, q_active, prior_tokens, **_):
    return (num_q_heads, num_kv_heads) == (1, 1) and prior_seg_size == 2048 and q_active == 512 and prior_tokens == 2048


def _remainder_batch_min_set_is_fast(*, num_q_heads, num_kv_heads, q_active, prior_tokens, **_):
    return (num_q_heads, num_kv_heads) == (1, 1) and q_active == 128 and prior_tokens == 2048


def _swa_k_pre_transposed_min_set_is_fast(*, num_q_heads, num_kv_heads, head_dim, q_active, prior_tokens, **_):
    return (num_q_heads, num_kv_heads) == (3, 1) and head_dim == 128 and q_active == 2048 and prior_tokens == 0


def _tp_out_false_slice_min_set_is_fast(*, num_q_heads, num_kv_heads, q_active, prior_tokens, **_):
    return (num_q_heads, num_kv_heads) == (2, 1) and q_active == 512 and prior_tokens == 0


def _fp8_kv_min_set_is_fast(*, num_q_heads, num_kv_heads, k_scale_val, v_scale_val, **_):
    return (num_q_heads, num_kv_heads) == (3, 1) and (k_scale_val, v_scale_val) == (1.5, 0.5)


def _swa_fp8_kv_min_set_is_fast(*, head_dim, k_scale_val, v_scale_val, **_):
    return head_dim == 64 and (k_scale_val, v_scale_val) == (1.67, 1.67)


def _large_d_min_set_is_fast(
    *, num_q_heads, num_kv_heads, head_dim, q_active, prior_seg_size, prior_tokens, block_size, **_
):
    if (num_q_heads, num_kv_heads) != (1, 1) or head_dim != 256 or q_active != 2048:
        return False
    return (block_size, prior_seg_size, prior_tokens) in {
        (16, 2048, 3072),
        (16, 4096, 30720),
        (128, 2048, 3072),
    }


def _output_tensor_for_q_active(bs_q, q_active, head_dim, tp_out, dtype):
    """Output tensor descriptor when q_active may differ from prior_seg_size.

    Output seqlen dim always matches Q's active length.
    """
    if tp_out:
        return {"out": np.zeros((bs_q, head_dim, q_active), dtype=dtype)}
    else:
        return {"out": np.zeros((bs_q, q_active, head_dim), dtype=dtype)}


@pytest_test_metadata(
    name="Attention Segmented CTE",
    tags=["model"],
    pytest_marks=["attention", "segmented", "cte"],
)
@final
class TestSegmentedAttentionCTE:
    """Test class for segmented attention with block-based KV cache."""

    # =========================================================================
    # Matrix-driven coverage tests (builder-based).
    #
    # These replace the hand-written blocks above by driving parametrize from
    # `_build_perms`. The general tests (#2, #4, #5, #5b, #6) cover cross-cutting
    # coverage; the model-specific tests (#7) below own the full q_active x
    # prior_seg_size sweep for Llama3 and gpt-oss.
    # =========================================================================

    # --- Shared helpers for builder-driven tests ---------------------------

    def _matrix_input_generator(
        self,
        *,
        bs,
        num_q_heads,
        num_kv_heads,
        block_size,
        prior_seg_size,
        head_dim,
        prior_tokens,
        tp_q,
        tp_out,
        dtype,
        q_active,
        sliding_window=None,
        use_sink=False,
        sink_value=None,
        k_pre_transposed=False,
        fp8_packed=False,
        k_scale_val=None,
        v_scale_val=None,
        prescale_q=False,
        kv_dtype=None,
    ):
        def _gen(test_config, input_tensor_def=None):
            inputs = generate_inputs(
                bs=bs,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                block_size=block_size,
                prior_seg_size=prior_seg_size,
                head_dim=head_dim,
                prior_tokens=prior_tokens,
                tp_q=tp_q,
                tp_out=tp_out,
                dtype=dtype,
                seqlen_q=q_active,
                sliding_window=sliding_window,
                use_sink=use_sink,
                sink_value=sink_value,
                k_pre_transposed=k_pre_transposed,
                fp8_packed=fp8_packed,
                k_scale_val=k_scale_val,
                v_scale_val=v_scale_val,
                kv_dtype=kv_dtype,
            )
            if prescale_q:
                # Multiply Q by 1/sqrt(head_dim) and pass scale=1.0 — caller
                # has already absorbed the softmax scale into Q.
                inv_sqrt_d = 1.0 / np.sqrt(head_dim)
                q_fp = dt.static_cast(inputs["q"], np.float32)
                q_fp = q_fp * inv_sqrt_d
                inputs["q"] = dt.static_cast(q_fp, dtype)
                inputs["scale"] = 1.0
            return inputs

        return _gen

    def _matrix_output_descriptor(self, *, bs_q, q_active, head_dim, tp_out, dtype):
        def _desc(kernel_input):
            return _output_tensor_for_q_active(bs_q, q_active, head_dim, tp_out, dtype)

        return _desc

    def _run_matrix_test(
        self,
        *,
        test_manager,
        platform_target,
        bs,
        num_q_heads,
        num_kv_heads,
        block_size,
        prior_seg_size,
        head_dim,
        prior_tokens,
        tp_q,
        tp_out,
        dtype,
        q_active,
        lnc=2,
        sliding_window=None,
        use_sink=False,
        sink_value=None,
        k_pre_transposed=False,
        fp8_packed=False,
        k_scale_val=None,
        v_scale_val=None,
        prescale_q=False,
        kv_dtype=None,
        rtol=1e-2,
        atol=1e-2,
    ):
        bs_q = bs * num_q_heads
        input_gen = self._matrix_input_generator(
            bs=bs,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            prior_seg_size=prior_seg_size,
            head_dim=head_dim,
            prior_tokens=prior_tokens,
            tp_q=tp_q,
            tp_out=tp_out,
            dtype=dtype,
            q_active=q_active,
            sliding_window=sliding_window,
            use_sink=use_sink,
            sink_value=sink_value,
            k_pre_transposed=k_pre_transposed,
            fp8_packed=fp8_packed,
            k_scale_val=k_scale_val,
            v_scale_val=v_scale_val,
            prescale_q=prescale_q,
            kv_dtype=kv_dtype,
        )
        output_desc = self._matrix_output_descriptor(
            bs_q=bs_q,
            q_active=q_active,
            head_dim=head_dim,
            tp_out=tp_out,
            dtype=dtype,
        )
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=attention_segmented_cte,
            torch_ref=torch_ref_wrapper(attention_segmented_cte_torch_ref),
            kernel_input_generator=input_gen,
            output_tensor_descriptor=output_desc,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
            rtol=rtol,
            atol=atol,
        )

    # --- #7a. Llama3 model-specific test ----------------------------------

    llama3_perms = _build_perms(
        q_actives=[128, 1024, 2048, 8192],
        prior_seg_sizes=[512, 2048, 4096, 8192],
        block_sizes=[16, 128],
        head_ratios=GQA_HEAD_RATIOS,
        head_dims=[128],
        prior_fn=_prior_tokens_full,
        tp_q=True,
        tp_out=True,
        dtype=nl.bfloat16,
        lnc=2,
    )

    @pytest_marks(["model", "optimal"])
    @pytest.mark.parametrize(_MATRIX_PARAMS_BASE, llama3_perms)
    def test_attention_segmented_cte_llama3(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        bs,
        num_q_heads,
        num_kv_heads,
        block_size,
        prior_seg_size,
        head_dim,
        prior_tokens,
        tp_q,
        tp_out,
        dtype,
        q_active,
    ):
        """Llama3 model-specific coverage: head_dim=128, GQA 8:1 and 1:1, full q_active x prior_seg sweep."""
        self._run_matrix_test(
            test_manager=test_manager,
            platform_target=platform_target,
            bs=bs,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            prior_seg_size=prior_seg_size,
            head_dim=head_dim,
            prior_tokens=prior_tokens,
            tp_q=tp_q,
            tp_out=tp_out,
            dtype=dtype,
            q_active=q_active,
            lnc=2,
        )

    # --- #7b. gpt-oss model-specific test ---------------------------------

    gpt_oss_full_perms = _build_perms(
        q_actives=[128, 1024, 2048, 8192],
        prior_seg_sizes=[512, 2048, 4096, 8192],
        block_sizes=[16, 128],
        head_ratios=GQA_HEAD_RATIOS,
        head_dims=[64],
        prior_fn=_prior_tokens_full,
        tp_q=True,
        tp_out=True,
        dtype=nl.bfloat16,
        lnc=2,
        use_sink=True,
        sink_value=2.0,
    )

    gpt_oss_swa_perms = _build_perms(
        q_actives=[128, 1024, 2048, 8192],
        prior_seg_sizes=[512, 2048, 4096, 8192],
        block_sizes=[16, 128],
        head_ratios=GQA_HEAD_RATIOS,
        head_dims=[64],
        prior_fn=_prior_tokens_full,
        tp_q=True,
        tp_out=True,
        dtype=nl.bfloat16,
        lnc=2,
        sliding_window=128,
        use_sink=True,
        sink_value=2.0,
    )

    @pytest_marks(["model", "optimal"])
    @pytest.mark.parametrize(_MATRIX_PARAMS_BASE, gpt_oss_full_perms)
    def test_attention_segmented_cte_gpt_oss_full(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        bs,
        num_q_heads,
        num_kv_heads,
        block_size,
        prior_seg_size,
        head_dim,
        prior_tokens,
        tp_q,
        tp_out,
        dtype,
        q_active,
    ):
        """gpt-oss full-attention with sink=2.0. head_dim=64, GQA 8:1 and 1:1."""
        self._run_matrix_test(
            test_manager=test_manager,
            platform_target=platform_target,
            bs=bs,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            prior_seg_size=prior_seg_size,
            head_dim=head_dim,
            prior_tokens=prior_tokens,
            tp_q=tp_q,
            tp_out=tp_out,
            dtype=dtype,
            q_active=q_active,
            lnc=2,
            use_sink=True,
            sink_value=2.0,
        )

    @pytest_marks(["model", "optimal"])
    @pytest.mark.parametrize(_MATRIX_PARAMS_BASE, gpt_oss_swa_perms)
    def test_attention_segmented_cte_gpt_oss_swa(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        bs,
        num_q_heads,
        num_kv_heads,
        block_size,
        prior_seg_size,
        head_dim,
        prior_tokens,
        tp_q,
        tp_out,
        dtype,
        q_active,
    ):
        """gpt-oss SWA (window=128) with sink=2.0. head_dim=64, GQA 8:1 and 1:1."""
        self._run_matrix_test(
            test_manager=test_manager,
            platform_target=platform_target,
            bs=bs,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            prior_seg_size=prior_seg_size,
            head_dim=head_dim,
            prior_tokens=prior_tokens,
            tp_q=tp_q,
            tp_out=tp_out,
            dtype=dtype,
            q_active=q_active,
            lnc=2,
            sliding_window=128,
            use_sink=True,
            sink_value=2.0,
        )

    # --- #7c. Prescaled-Q small q_active regression ------------------------
    # Covers the regime that previously triggered an SBUF address collision
    # between extra_kv_used_len (allocated by the seg_cte caller) and
    # _attention_cte's internal kv_used_len_sb. q_active in {256, 512} forces
    # the Core 1 extra-iteration path (q_active < 2 * _K_TILE_SZ, partial last
    # K tile). Q is multiplied by 1/sqrt(head_dim) at the call site with
    # scale=1.0, mirroring how production callers absorb the softmax scale
    # into Q.

    llama3_prescale_perms = _build_perms(
        q_actives=[256, 512],
        prior_seg_sizes=[1024, 2048],
        block_sizes=[128],
        head_ratios=GQA_HEAD_RATIOS,
        head_dims=[128],
        prior_fn=_prior_tokens_full,
        tp_q=True,
        tp_out=True,
        dtype=nl.bfloat16,
        lnc=2,
    )

    @pytest_marks(["model", "optimal"])
    @pytest.mark.parametrize(_MATRIX_PARAMS_BASE, llama3_prescale_perms)
    def test_attention_segmented_cte_llama3_prescaled_q(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        bs,
        num_q_heads,
        num_kv_heads,
        block_size,
        prior_seg_size,
        head_dim,
        prior_tokens,
        tp_q,
        tp_out,
        dtype,
        q_active,
    ):
        """Llama3 small-q_active prescaled-Q: head_dim=128, q_active in {256, 512}."""
        self._run_matrix_test(
            test_manager=test_manager,
            platform_target=platform_target,
            bs=bs,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            prior_seg_size=prior_seg_size,
            head_dim=head_dim,
            prior_tokens=prior_tokens,
            tp_q=tp_q,
            tp_out=tp_out,
            dtype=dtype,
            q_active=q_active,
            lnc=2,
            prescale_q=True,
        )

    gpt_oss_prescale_perms = _build_perms(
        q_actives=[256, 512],
        prior_seg_sizes=[1024, 2048],
        block_sizes=[128],
        head_ratios=GQA_HEAD_RATIOS,
        head_dims=[64],
        prior_fn=_prior_tokens_full,
        tp_q=True,
        tp_out=True,
        dtype=nl.bfloat16,
        lnc=2,
        use_sink=True,
        sink_value=2.0,
    )

    @pytest_marks(["model", "optimal"])
    @pytest.mark.parametrize(_MATRIX_PARAMS_BASE, gpt_oss_prescale_perms)
    def test_attention_segmented_cte_gpt_oss_prescaled_q(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        bs,
        num_q_heads,
        num_kv_heads,
        block_size,
        prior_seg_size,
        head_dim,
        prior_tokens,
        tp_q,
        tp_out,
        dtype,
        q_active,
    ):
        """gpt-oss small-q_active prescaled-Q with sink=2.0: head_dim=64, q_active in {256, 512}."""
        self._run_matrix_test(
            test_manager=test_manager,
            platform_target=platform_target,
            bs=bs,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            prior_seg_size=prior_seg_size,
            head_dim=head_dim,
            prior_tokens=prior_tokens,
            tp_q=tp_q,
            tp_out=tp_out,
            dtype=dtype,
            q_active=q_active,
            lnc=2,
            use_sink=True,
            sink_value=2.0,
            prescale_q=True,
        )

    # --- #7d. d>128 (d-tiled K/V SBUF, e.g. Gemma4/Qwen3.6) ---------------
    # d=512 with prior_seg_size>2048 exceeds SBUF budget (K tiles alone need 4MB).
    # A kernel_assert guards this at runtime. See TODO in attention_segmented_cte.py.

    large_d_perms = _build_perms(
        q_actives=[2048],
        prior_seg_sizes=[2048, 4096],
        block_sizes=[16, 128],
        head_ratios=[(2, 1), (1, 1)],
        head_dims=[256],
        prior_fn=_prior_tokens_full,
        tp_q=True,
        tp_out=True,
        dtype=nl.bfloat16,
        lnc=2,
        is_fast=_large_d_min_set_is_fast,
    ) + _build_perms(
        q_actives=[2048],
        prior_seg_sizes=[2048],
        block_sizes=[16, 128],
        head_ratios=[(2, 1), (1, 1)],
        head_dims=[512],
        prior_fn=_prior_tokens_full,
        tp_q=True,
        tp_out=True,
        dtype=nl.bfloat16,
        lnc=2,
    )

    @pytest_marks(["model", "optimal"])
    @pytest.mark.parametrize(_MATRIX_PARAMS_BASE, large_d_perms)
    def test_attention_segmented_cte_large_d(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        bs,
        num_q_heads,
        num_kv_heads,
        block_size,
        prior_seg_size,
        head_dim,
        prior_tokens,
        tp_q,
        tp_out,
        dtype,
        q_active,
    ):
        """Test segmented attention with head_dim > 128 (d-tiled K/V SBUF)."""
        self._run_matrix_test(
            test_manager=test_manager,
            platform_target=platform_target,
            bs=bs,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            prior_seg_size=prior_seg_size,
            head_dim=head_dim,
            prior_tokens=prior_tokens,
            tp_q=tp_q,
            tp_out=tp_out,
            dtype=dtype,
            q_active=q_active,
            lnc=2,
        )

    # --- #2. General matrix (non-model head ratios / dims) ----------------

    general_matrix_perms = _build_perms(
        q_actives=[1024, 2048, 4096],
        prior_seg_sizes=[1024, 2048, 4096],
        block_sizes=[16, 128],
        head_ratios=GQA_HEAD_RATIOS,
        head_dims=[64, 128],
        prior_fn=_prior_tokens_full,
        tp_q=True,
        tp_out=True,
        dtype=nl.bfloat16,
        lnc=2,
    )

    @pytest.mark.parametrize(_MATRIX_PARAMS_BASE, general_matrix_perms)
    def test_attention_segmented_cte_q_active_prior_seg_matrix(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        bs,
        num_q_heads,
        num_kv_heads,
        block_size,
        prior_seg_size,
        head_dim,
        prior_tokens,
        tp_q,
        tp_out,
        dtype,
        q_active,
    ):
        """Decoupled q_active x prior_seg_size matrix with production defaults (LNC2, tp_out=True, GQA)."""
        self._run_matrix_test(
            test_manager=test_manager,
            platform_target=platform_target,
            bs=bs,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            prior_seg_size=prior_seg_size,
            head_dim=head_dim,
            prior_tokens=prior_tokens,
            tp_q=tp_q,
            tp_out=tp_out,
            dtype=dtype,
            q_active=q_active,
            lnc=2,
        )

    # --- #4. num_q_heads coverage (MHA + GQA) -----------------------------

    num_q_heads_perms = _build_perms(
        q_actives=[512, 2048, 8192],
        prior_seg_sizes=[2048, 4096],
        block_sizes=[128],
        head_ratios=HEAD_RATIOS,
        head_dims=[128],
        prior_fn=lambda seg, q: [0, max(0, 131072 - q)],
        tp_q=True,
        tp_out=True,
        dtype=nl.bfloat16,
        lnc=2,
    )

    @pytest.mark.parametrize(_MATRIX_PARAMS_BASE, num_q_heads_perms)
    def test_attention_segmented_cte_num_q_heads(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        bs,
        num_q_heads,
        num_kv_heads,
        block_size,
        prior_seg_size,
        head_dim,
        prior_tokens,
        tp_q,
        tp_out,
        dtype,
        q_active,
    ):
        """num_q_heads coverage: (1,1), (2,1), (4,2), (8,1) across LNC2 production defaults."""
        self._run_matrix_test(
            test_manager=test_manager,
            platform_target=platform_target,
            bs=bs,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            prior_seg_size=prior_seg_size,
            head_dim=head_dim,
            prior_tokens=prior_tokens,
            tp_q=tp_q,
            tp_out=tp_out,
            dtype=dtype,
            q_active=q_active,
            lnc=2,
        )

    # --- #4b. Remainder-batch (bs_q % 2 != 0) coverage -------------------
    # Focused test for the LNC2 50/50 remainder shard, exercising small
    # num_grps (1, 2, 3) so we cover the single-core fallback (num_grps=1),
    # the even-split base case (num_grps=2), and the asymmetric split
    # (num_grps=3). Odd num_q_heads gives bs_q % 2 != 0.

    @staticmethod
    def _remainder_skip(*, q_active, prior_seg_size, prior_tokens, block_size, num_q_heads, lnc, **_):
        # Reuse default alignment rules but drop the 'q_active >= 2*block_size'
        # constraint so num_grps = 1 (q_active = 128) is generated.
        if q_active % block_size != 0 or q_active < block_size:
            return True
        if prior_seg_size % block_size != 0 or prior_seg_size < block_size:
            return True
        if prior_tokens % block_size != 0:
            return True
        return False

    remainder_perms = _build_perms(
        q_actives=[128, 256, 384],  # num_grps = 1, 2, 3
        prior_seg_sizes=[2048],
        block_sizes=[128],
        head_ratios=[(1, 1), (3, 1)],  # odd num_q_heads → bs_q % 2 != 0
        head_dims=[128],
        prior_fn=lambda seg, q: [0, seg],
        tp_q=True,
        tp_out=True,
        dtype=nl.bfloat16,
        lnc=2,
        skip=_remainder_skip,
        is_fast=_remainder_batch_min_set_is_fast,
    )

    @pytest.mark.parametrize(_MATRIX_PARAMS_BASE, remainder_perms)
    def test_attention_segmented_cte_remainder_batch(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        bs,
        num_q_heads,
        num_kv_heads,
        block_size,
        prior_seg_size,
        head_dim,
        prior_tokens,
        tp_q,
        tp_out,
        dtype,
        q_active,
    ):
        """LNC2 remainder-shard coverage: odd num_q_heads × num_grps ∈ {1, 2, 3}."""
        self._run_matrix_test(
            test_manager=test_manager,
            platform_target=platform_target,
            bs=bs,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            prior_seg_size=prior_seg_size,
            head_dim=head_dim,
            prior_tokens=prior_tokens,
            tp_q=tp_q,
            tp_out=tp_out,
            dtype=dtype,
            q_active=q_active,
            lnc=2,
        )

    # --- #5. SWA + sink (generic, head_dim=64, block_size=128) ------------

    swa_sink_perms = _build_perms(
        q_actives=[128, 512, 1024, 2048, 4096, 8192],
        prior_seg_sizes=[512],
        block_sizes=[128],
        head_ratios=GQA_HEAD_RATIOS,
        head_dims=[64],
        prior_fn=lambda seg, q: [0, max(0, 131072 - q)],
        tp_q=True,
        tp_out=True,
        dtype=nl.bfloat16,
        lnc=2,
        sliding_window=128,
        use_sink=True,
        sink_value=2.0,
    )

    @pytest.mark.parametrize(_MATRIX_PARAMS_BASE, swa_sink_perms)
    def test_attention_segmented_cte_swa_sink(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        bs,
        num_q_heads,
        num_kv_heads,
        block_size,
        prior_seg_size,
        head_dim,
        prior_tokens,
        tp_q,
        tp_out,
        dtype,
        q_active,
    ):
        """SWA (window=128) + sink=2.0 with head_dim=64, block_size=128 (gpt-oss-like generic coverage)."""
        self._run_matrix_test(
            test_manager=test_manager,
            platform_target=platform_target,
            bs=bs,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            prior_seg_size=prior_seg_size,
            head_dim=head_dim,
            prior_tokens=prior_tokens,
            tp_q=tp_q,
            tp_out=tp_out,
            dtype=dtype,
            q_active=q_active,
            lnc=2,
            sliding_window=128,
            use_sink=True,
            sink_value=2.0,
        )

    # --- #5b. Non-SWA sink -----------------------------------------------

    sink_perms = _build_perms(
        q_actives=[512, 2048, 8192],
        prior_seg_sizes=[2048, 4096],
        block_sizes=[128],
        head_ratios=GQA_HEAD_RATIOS,
        head_dims=[64, 128],
        prior_fn=lambda seg, q: [0, max(0, 131072 - q)],
        tp_q=True,
        tp_out=True,
        dtype=nl.bfloat16,
        lnc=2,
        use_sink=True,
        sink_value=2.0,
    )

    @pytest.mark.parametrize(_MATRIX_PARAMS_BASE, sink_perms)
    def test_attention_segmented_cte_sink_matrix(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        bs,
        num_q_heads,
        num_kv_heads,
        block_size,
        prior_seg_size,
        head_dim,
        prior_tokens,
        tp_q,
        tp_out,
        dtype,
        q_active,
    ):
        """Non-SWA sink=2.0 across head_dim in {64, 128}, GQA 8:1 and 1:1."""
        self._run_matrix_test(
            test_manager=test_manager,
            platform_target=platform_target,
            bs=bs,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            prior_seg_size=prior_seg_size,
            head_dim=head_dim,
            prior_tokens=prior_tokens,
            tp_q=tp_q,
            tp_out=tp_out,
            dtype=dtype,
            q_active=q_active,
            lnc=2,
            use_sink=True,
            sink_value=2.0,
        )

    # --- #5c. Multi-batch sink (bs>1) — exercises per-batch sink slicing ---

    @pytest.mark.parametrize(
        "bs, num_q_heads, num_kv_heads, block_size, prior_seg_size, head_dim, prior_tokens, tp_q, tp_out, dtype, q_active, sliding_window",
        [
            # Non-SWA, bs=2, GQA 8:1, with prior
            pytest.param(2, 8, 1, 128, 2048, 128, 2048, True, True, nl.bfloat16, 512, None),
            # Non-SWA, bs=2, GQA 8:1, no prior
            pytest.param(2, 8, 1, 128, 2048, 128, 0, True, True, nl.bfloat16, 512, None),
            # Non-SWA, bs=2, GQA 8:1, num_full=2 prior (prior_tokens=1024, prior_seg_size=512
            # -> floor(1024/512)=2, partial=0). Exercises the opt5 peeled Region C on the
            # multihead path: 1 accumulate-only segment + 1 final accumulate+normalize
            # segment. Kept non-SWA on purpose: SWA returns early and never reaches Region C.
            pytest.param(2, 8, 1, 128, 512, 64, 1024, True, True, nl.bfloat16, 512, None),
            # Non-SWA, bs=2, GQA 8:1, num_full=4 prior (prior_tokens=2048, prior_seg_size=512
            # -> floor(2048/512)=4, partial=0). Deeper guard for the opt5 peel: 3
            # accumulate-only segments + 1 final accumulate+normalize segment; the
            # interleaved normalize must fire exactly once (on the last prior segment).
            pytest.param(2, 8, 1, 128, 512, 64, 2048, True, True, nl.bfloat16, 512, None),
            # SWA, bs=2, GQA 8:1, with prior
            pytest.param(2, 8, 1, 128, 512, 64, 512, True, True, nl.bfloat16, 512, 128),
            # SWA, bs=2, GQA 8:1, no prior
            pytest.param(2, 8, 1, 128, 512, 64, 0, True, True, nl.bfloat16, 512, 128),
        ],
    )
    def test_attention_segmented_cte_multi_batch_sink(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        bs,
        num_q_heads,
        num_kv_heads,
        block_size,
        prior_seg_size,
        head_dim,
        prior_tokens,
        tp_q,
        tp_out,
        dtype,
        q_active,
        sliding_window,
    ):
        """Multi-batch (bs>1) with sink=2.0 — verifies per-batch sink slicing correctness."""
        self._run_matrix_test(
            test_manager=test_manager,
            platform_target=platform_target,
            bs=bs,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            prior_seg_size=prior_seg_size,
            head_dim=head_dim,
            prior_tokens=prior_tokens,
            tp_q=tp_q,
            tp_out=tp_out,
            dtype=dtype,
            q_active=q_active,
            lnc=2,
            sliding_window=sliding_window,
            use_sink=True,
            sink_value=2.0,
        )

    # --- #6a. tp_out=False slice ------------------------------------------

    tp_out_false_perms = _build_perms(
        q_actives=[512, 2048],
        prior_seg_sizes=[2048],
        block_sizes=[128],
        head_ratios=[(2, 1), (8, 1)],
        head_dims=[128],
        prior_fn=lambda seg, q: [0, seg],
        tp_q=True,
        tp_out=False,
        dtype=nl.bfloat16,
        lnc=2,
        is_fast=_tp_out_false_slice_min_set_is_fast,
    )

    @pytest.mark.parametrize(_MATRIX_PARAMS_BASE, tp_out_false_perms)
    def test_attention_segmented_cte_tp_out_false_slice(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        bs,
        num_q_heads,
        num_kv_heads,
        block_size,
        prior_seg_size,
        head_dim,
        prior_tokens,
        tp_q,
        tp_out,
        dtype,
        q_active,
    ):
        """Small tp_out=False slice — non-default config, representative coverage only."""
        self._run_matrix_test(
            test_manager=test_manager,
            platform_target=platform_target,
            bs=bs,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            prior_seg_size=prior_seg_size,
            head_dim=head_dim,
            prior_tokens=prior_tokens,
            tp_q=tp_q,
            tp_out=tp_out,
            dtype=dtype,
            q_active=q_active,
            lnc=2,
        )

    # --- #6b. LNC1 slice --------------------------------------------------

    lnc1_slice_perms = _build_perms(
        q_actives=[512, 2048],
        prior_seg_sizes=[2048, 4096],
        block_sizes=[128],
        head_ratios=[(1, 1), (4, 1)],
        head_dims=[128],
        prior_fn=lambda seg, q: [0, seg, 2 * seg],
        tp_q=True,
        tp_out=True,
        dtype=nl.bfloat16,
        lnc=1,
        is_fast=_lnc1_slice_min_set_is_fast,
    )

    @pytest.mark.parametrize(_MATRIX_PARAMS_BASE, lnc1_slice_perms)
    def test_attention_segmented_cte_lnc1_slice(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        bs,
        num_q_heads,
        num_kv_heads,
        block_size,
        prior_seg_size,
        head_dim,
        prior_tokens,
        tp_q,
        tp_out,
        dtype,
        q_active,
    ):
        """Small LNC1 slice with tp_out=True — non-default config, representative coverage only."""
        self._run_matrix_test(
            test_manager=test_manager,
            platform_target=platform_target,
            bs=bs,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            prior_seg_size=prior_seg_size,
            head_dim=head_dim,
            prior_tokens=prior_tokens,
            tp_q=tp_q,
            tp_out=tp_out,
            dtype=dtype,
            q_active=q_active,
            lnc=1,
        )

    # --- #7. Transposed K-cache slice (k_pre_transposed=True) ------------
    # Exercises the transposed K-cache layout introduced in 28132257. Narrow
    # slice: fixed pss=2048 with 0/partial/full prior, representative GQA
    # ratios, both head_dims and both block_sizes. ~12 tests.
    k_pre_transposed_perms = _build_perms(
        q_actives=[2048],
        prior_seg_sizes=[2048],
        block_sizes=[128],
        head_ratios=[(2, 1), (3, 1), (4, 2)],
        head_dims=[64, 128],
        prior_fn=lambda seg, q: [0, seg],
        tp_q=True,
        tp_out=True,
        dtype=nl.bfloat16,
        lnc=2,
    )

    @pytest.mark.parametrize(_MATRIX_PARAMS_BASE, k_pre_transposed_perms)
    def test_attention_segmented_cte_k_pre_transposed(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        bs,
        num_q_heads,
        num_kv_heads,
        block_size,
        prior_seg_size,
        head_dim,
        prior_tokens,
        tp_q,
        tp_out,
        dtype,
        q_active,
    ):
        """Segmented attention with transposed K cache (k_pre_transposed=True)."""
        self._run_matrix_test(
            test_manager=test_manager,
            platform_target=platform_target,
            bs=bs,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            prior_seg_size=prior_seg_size,
            head_dim=head_dim,
            prior_tokens=prior_tokens,
            tp_q=tp_q,
            tp_out=tp_out,
            dtype=dtype,
            q_active=q_active,
            lnc=2,
            k_pre_transposed=True,
        )

    # --- #8. FP8 KV cache dequantization (k_pre_transposed=True + scales) -
    # Covers both identity-scale (must match non-scaled result) and non-
    # identity scales. Parametrized over (k_scale_val, v_scale_val) × base
    # config slice. ~14 tests.
    # Compile-time-weighted minimum set: only the (3,1)-heads × (1.5, 0.5)-scales
    # combo adds branch coverage.
    fp8_kv_perms = _build_perms(
        q_actives=[2048],
        prior_seg_sizes=[2048],
        block_sizes=[128],
        head_ratios=[(2, 1), (3, 1)],
        head_dims=[128],
        prior_fn=lambda seg, q: [0, seg],
        tp_q=True,
        tp_out=True,
        dtype=nl.bfloat16,
        lnc=2,
        extra_col_variants=[
            (1.0, 1.0),  # identity
            (1.67, 1.67),  # symmetric non-identity
            (1.5, 0.5),  # asymmetric
        ],
        extra_col_names=("k_scale_val", "v_scale_val"),
        is_fast=_fp8_kv_min_set_is_fast,
    )

    @pytest.mark.parametrize(
        _MATRIX_PARAMS_BASE + ", k_scale_val, v_scale_val",
        fp8_kv_perms,
    )
    def test_attention_segmented_cte_fp8_kv(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        bs,
        num_q_heads,
        num_kv_heads,
        block_size,
        prior_seg_size,
        head_dim,
        prior_tokens,
        tp_q,
        tp_out,
        dtype,
        q_active,
        k_scale_val,
        v_scale_val,
    ):
        """FP8 KV cache dequantization path (k_pre_transposed=True + per-head-dim scales)."""
        self._run_matrix_test(
            test_manager=test_manager,
            platform_target=platform_target,
            bs=bs,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            prior_seg_size=prior_seg_size,
            head_dim=head_dim,
            prior_tokens=prior_tokens,
            tp_q=tp_q,
            tp_out=tp_out,
            dtype=dtype,
            q_active=q_active,
            lnc=2,
            k_pre_transposed=True,
            k_scale_val=k_scale_val,
            v_scale_val=v_scale_val,
        )

    # --- #8b. True FP8 KV cache dtype (k_pre_transposed=True + fp8 dtype + scales) -
    # Exercises actual float8_e4m3 k_cache/v_cache tensors loaded into fp8 SBUF tiles.
    fp8_kv_dtype_perms = [
        pytest.param(*base.values, k_scale_val, v_scale_val, marks=pytest.mark.fast)
        for base in _build_perms(
            q_actives=[2048],
            prior_seg_sizes=[2048],
            block_sizes=[128],
            head_ratios=[(2, 1)],
            head_dims=[128],
            prior_fn=lambda seg, q: [0, seg],
            tp_q=True,
            tp_out=True,
            dtype=nl.bfloat16,
            lnc=2,
            is_fast=lambda **_: True,
        )
        for k_scale_val, v_scale_val in [
            (1.0, 1.0),
            (1.5, 0.5),
        ]
    ]

    @pytest.mark.parametrize(
        _MATRIX_PARAMS_BASE + ", k_scale_val, v_scale_val",
        fp8_kv_dtype_perms,
    )
    def test_attention_segmented_cte_fp8_kv_dtype(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        bs,
        num_q_heads,
        num_kv_heads,
        block_size,
        prior_seg_size,
        head_dim,
        prior_tokens,
        tp_q,
        tp_out,
        dtype,
        q_active,
        k_scale_val,
        v_scale_val,
    ):
        """True FP8 KV cache: k_cache/v_cache are float8_e4m3 with k_pre_transposed=True."""
        self._run_matrix_test(
            test_manager=test_manager,
            platform_target=platform_target,
            bs=bs,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            prior_seg_size=prior_seg_size,
            head_dim=head_dim,
            prior_tokens=prior_tokens,
            tp_q=tp_q,
            tp_out=tp_out,
            dtype=dtype,
            q_active=q_active,
            lnc=2,
            k_pre_transposed=True,
            k_scale_val=k_scale_val,
            v_scale_val=v_scale_val,
            kv_dtype=nl.float8_e4m3,
            rtol=5e-2,
            atol=5e-2,
        )

    # --- #8c. FP8 packed KV cache (fp8_packed=True + scales) ----------------
    # Exercises both the batched dma_transpose path (16-aligned block counts: 2048)
    # and the per-block fallback path (non-16-aligned: 1920->15, 640->5, 1408->11).
    # head_ratios includes GQA (8,2) and (4,2) so the head-major packed K addressing
    # (per-head row offset, block index scaled by num_kv_heads) is exercised with
    # num_kv_heads > 1 — (2,1) alone leaves that addressing a no-op.
    fp8_packed_perms = [
        pytest.param(*base.values, k_scale_val, v_scale_val, marks=pytest.mark.fast)
        for base in _build_perms(
            q_actives=[2048],
            prior_seg_sizes=[2048, 1920, 640, 1408],
            block_sizes=[128],
            head_ratios=[(2, 1), (8, 2), (4, 2)],
            head_dims=[64, 128],
            prior_fn=lambda seg, q: [0, seg],
            tp_q=True,
            tp_out=True,
            dtype=nl.bfloat16,
            lnc=2,
            is_fast=lambda **_: True,
        )
        for k_scale_val, v_scale_val in [
            (1.0, 1.0),
            (1.67, 1.67),
            (1.5, 0.5),
        ]
    ]

    @pytest.mark.parametrize(
        _MATRIX_PARAMS_BASE + ", k_scale_val, v_scale_val",
        fp8_packed_perms,
    )
    def test_attention_segmented_cte_fp8_packed(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        bs,
        num_q_heads,
        num_kv_heads,
        block_size,
        prior_seg_size,
        head_dim,
        prior_tokens,
        tp_q,
        tp_out,
        dtype,
        q_active,
        k_scale_val,
        v_scale_val,
    ):
        """FP8 packed KV cache path — batched transpose and per-block fallback."""
        self._run_matrix_test(
            test_manager=test_manager,
            platform_target=platform_target,
            bs=bs,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            prior_seg_size=prior_seg_size,
            head_dim=head_dim,
            prior_tokens=prior_tokens,
            tp_q=tp_q,
            tp_out=tp_out,
            dtype=dtype,
            q_active=q_active,
            lnc=2,
            fp8_packed=True,
            k_scale_val=k_scale_val,
            v_scale_val=v_scale_val,
            kv_dtype=nl.float8_e4m3,
            rtol=5e-2,
            atol=5e-2,
        )

    # --- #9. SWA + transposed K cache ------------------------------------
    swa_k_pre_transposed_perms = _build_perms(
        q_actives=[2048],
        prior_seg_sizes=[2048],
        block_sizes=[128],
        head_ratios=[(2, 1), (3, 1)],
        head_dims=[64, 128],
        prior_fn=lambda seg, q: [0, seg],
        tp_q=True,
        tp_out=True,
        dtype=nl.bfloat16,
        lnc=2,
        sliding_window=1024,
        is_fast=_swa_k_pre_transposed_min_set_is_fast,
    )

    @pytest.mark.parametrize(_MATRIX_PARAMS_BASE, swa_k_pre_transposed_perms)
    def test_attention_segmented_cte_swa_k_pre_transposed(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        bs,
        num_q_heads,
        num_kv_heads,
        block_size,
        prior_seg_size,
        head_dim,
        prior_tokens,
        tp_q,
        tp_out,
        dtype,
        q_active,
    ):
        """SWA with transposed K cache."""
        self._run_matrix_test(
            test_manager=test_manager,
            platform_target=platform_target,
            bs=bs,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            prior_seg_size=prior_seg_size,
            head_dim=head_dim,
            prior_tokens=prior_tokens,
            tp_q=tp_q,
            tp_out=tp_out,
            dtype=dtype,
            q_active=q_active,
            lnc=2,
            sliding_window=1024,
            k_pre_transposed=True,
        )

    # --- #10. SWA + FP8 KV cache ------------------------------------------
    # Compile-time-weighted minimum set: only head_dim=64 × (1.67, 1.67) scales
    # adds branch coverage.
    swa_fp8_kv_perms = _build_perms(
        q_actives=[2048],
        prior_seg_sizes=[2048],
        block_sizes=[128],
        head_ratios=[(2, 1)],
        head_dims=[64, 128],
        prior_fn=lambda seg, q: [0, seg],
        tp_q=True,
        tp_out=True,
        dtype=nl.bfloat16,
        lnc=2,
        sliding_window=1024,
        extra_col_variants=[(1.0, 1.0), (1.67, 1.67), (1.5, 0.5)],
        extra_col_names=("k_scale_val", "v_scale_val"),
        is_fast=_swa_fp8_kv_min_set_is_fast,
    )

    @pytest.mark.parametrize(
        _MATRIX_PARAMS_BASE + ", k_scale_val, v_scale_val",
        swa_fp8_kv_perms,
    )
    def test_attention_segmented_cte_swa_fp8_kv(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        bs,
        num_q_heads,
        num_kv_heads,
        block_size,
        prior_seg_size,
        head_dim,
        prior_tokens,
        tp_q,
        tp_out,
        dtype,
        q_active,
        k_scale_val,
        v_scale_val,
    ):
        """SWA + FP8 KV cache dequantization."""
        self._run_matrix_test(
            test_manager=test_manager,
            platform_target=platform_target,
            bs=bs,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            prior_seg_size=prior_seg_size,
            head_dim=head_dim,
            prior_tokens=prior_tokens,
            tp_q=tp_q,
            tp_out=tp_out,
            dtype=dtype,
            q_active=q_active,
            lnc=2,
            sliding_window=1024,
            k_pre_transposed=True,
            k_scale_val=k_scale_val,
            v_scale_val=v_scale_val,
        )

    # --- Compile-only probe: q_active=16384, prior_seg_size=8192 ---
    # The existing matrix caps q_active at 8192. Verify whether q_active=16384
    # with prior_seg_size=8192 compiles, in both bs_q layouts:
    #   - "with_remainder":  bs=1, num_q_heads=1 → bs_q=1 → LNC2 remainder path
    #   - "wo_remainder":    bs=1, num_q_heads=2 → bs_q=2 → primary-only
    # prior_tokens=8192 gives num_full=1 (one full-prior iteration). This is a
    # probe — correctness check is via simulator to avoid long HW compile times.
    #
    # Currently skipped: both variants fail compile with NCC_IBIR228 on trn2
    # (SBUF per-partition overflow: 231,112B needed vs 229,376B capacity,
    # +~1.7KB over budget). Keeping the test case as a regression probe for
    # when SBUF usage is trimmed.
    @pytest.mark.skip(reason="q_active=16384 exceeds trn2 SBUF budget (NCC_IBIR228); retry once SBUF usage is trimmed")
    @pytest.mark.parametrize(
        "remainder_mode,num_q_heads",
        [
            ("with_remainder", 1),
            ("wo_remainder", 2),
        ],
    )
    def test_qa16384_probe(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        remainder_mode,
        num_q_heads,
    ):
        """Compile/run probe for q_active=16384 with prior_seg_size=8192."""
        self._run_matrix_test(
            test_manager=test_manager,
            platform_target=platform_target,
            bs=1,
            num_q_heads=num_q_heads,
            num_kv_heads=1,
            block_size=128,
            prior_seg_size=8192,
            head_dim=128,
            prior_tokens=8192,
            tp_q=True,
            tp_out=True,
            dtype=nl.bfloat16,
            q_active=16384,
            lnc=2,
        )

    # --- LNC2 extra iteration K/V tile zeroing ----------------------------

    @pytest.mark.fast
    @pytest.mark.simulation
    @pytest.mark.parametrize(
        "q_active, prior_seg_size",
        [
            (256, 1024),
            (256, 2048),
        ],
    )
    def test_lnc2_extra_iter_kv_tile_zeroing(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        q_active,
        prior_seg_size,
    ):
        """Verify all K/V tiles are zeroed in the LNC2 extra iteration path.

        In LNC2, Core 1 runs one extra iteration to attend over Core 0's active
        KV segment. This test ensures the K/V tiles are fully zeroed before that
        load, preventing stale NaN data from being visible to the matmul.
        """
        self._run_matrix_test(
            test_manager=test_manager,
            platform_target=platform_target,
            bs=1,
            num_q_heads=1,
            num_kv_heads=1,
            block_size=128,
            prior_seg_size=prior_seg_size,
            head_dim=128,
            prior_tokens=0,
            tp_q=True,
            tp_out=True,
            dtype=nl.bfloat16,
            q_active=q_active,
            lnc=2,
        )


@pytest_marks(["attention", "segmented", "cte", "model"])
@final
class TestSegmentedAttentionCteModel:
    """Model-driven tests for segmented attention CTE kernel, organized by tier."""

    _MODEL_PARAMS = (
        "bs, num_q_heads, num_kv_heads, block_size, prior_seg_size, head_dim, prior_tokens, tp_q, tp_out, dtype"
    )

    _OPTIMAL_PARAMS, _OPTIMAL_IDS = (
        prepare_model_parametrize(
            {ModelTestType.OPTIMAL: segmented_attention_cte_model_configs.get(ModelTestType.OPTIMAL, [])}
        )
        if segmented_attention_cte_model_configs
        else ([], [])
    )

    def _run_model_test(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        bs,
        num_q_heads,
        num_kv_heads,
        block_size,
        prior_seg_size,
        head_dim,
        prior_tokens,
        tp_q,
        tp_out,
        dtype,
    ):
        bs_q = bs * num_q_heads

        def input_generator(test_config, input_tensor_def=None):
            return generate_inputs(
                bs=bs,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                block_size=block_size,
                prior_seg_size=prior_seg_size,
                head_dim=head_dim,
                prior_tokens=prior_tokens,
                tp_q=tp_q,
                tp_out=tp_out,
                dtype=dtype,
            )

        def output_tensor_descriptor(kernel_input):
            return output_tensors(kernel_input, bs_q, prior_seg_size, head_dim, tp_out, dtype)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=attention_segmented_cte,
            torch_ref=torch_ref_wrapper(attention_segmented_cte_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensor_descriptor,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.optimal
    @pytest.mark.parametrize(_MODEL_PARAMS, _OPTIMAL_PARAMS, ids=_OPTIMAL_IDS)
    def test_optimal(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        bs,
        num_q_heads,
        num_kv_heads,
        block_size,
        prior_seg_size,
        head_dim,
        prior_tokens,
        tp_q,
        tp_out,
        dtype,
    ):
        """OPTIMAL: Performance-optimized model configs."""
        self._run_model_test(**{k: v for k, v in locals().items() if k != "self"})
