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

"""Integration tests for the absorbed-latent MLA QKV CTE kernel."""

import functools
from typing import Any, final

import ml_dtypes
import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
import torch
from nkilib_src.nkilib.experimental.mla.deepseek.mla_common_cte import MlaPrecision
from nkilib_src.nkilib.experimental.mla.deepseek.mla_qkv_cte import mla_qkv_cte_kernel
from nkilib_src.nkilib.experimental.mla.deepseek.mla_qkv_cte_torch import mla_qkv_cte_torch_ref
from typing_extensions import override

from test.integration.nkilib.core.moe.moe_cte.test_utils import build_prequantized_hidden_concat
from test.integration.nkilib.core.qkv.test_qkv_common import (
    _reduce_mx_scale_to_compact_block128,
    _swizzle_mla_cols,
    build_qkv_mla_input,
)
from test.integration.nkilib.experimental.sparse_attention_indexer.test_sparse_attention_indexer_common import (
    _dequant_swizzled_qr_for_torch_ref,
)
from test.integration.nkilib.utils.tensor_generators import generate_stabilized_mx_data
from test.utils.common_dataclasses import (
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    ModelTestType,
    Platforms,
    prepare_model_parametrize,
)
from test.utils.comparators import maxAllClose
from test.utils.golden_provider import mark_uncacheable
from test.utils.model_test_configs import no_model_configs
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

try:
    from test.integration.nkilib.experimental.mla.config.test_mla_qkv_cte import (
        MODEL_PARAM_NAMES,
        mla_qkv_cte_model_configs,
    )
except ImportError:
    MODEL_PARAM_NAMES = (
        "vnc_degree, batch, seqlen, hidden, n_heads, qk_lora_rank, "
        "kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim, norm_eps, precision"
    )
    mla_qkv_cte_model_configs = no_model_configs()


@functools.wraps(mla_qkv_cte_torch_ref)
def _qkv_ref_with_qr(*args, **kwargs):
    """Remap the shared ref's outputs to the kernel's 6 returns. The shared ref emits
    q_lift/q_pe/c_kv/k_pe plus the q-normed ``qr`` [B, S, qk_lora_rank]; the kernel now
    always allocates + returns the MX-quantized qr export (``qr_qtz`` / ``qr_scale``). Carry
    the fp32 qr under ``qr_qtz`` (the comparator dequantizes the kernel's packed export and
    compares to it) and a placeholder ``qr_scale`` (validated implicitly via that dequant).
    functools.wraps preserves the signature the test framework validates against the kernel."""
    ref = mla_qkv_cte_torch_ref(*args, **kwargs)
    qr = ref["qr"]
    qr_np = qr.to("cpu", torch.float32).numpy() if isinstance(qr, torch.Tensor) else np.asarray(qr, dtype=np.float32)
    b, s, ql = qr_np.shape
    # New dict (not the ref's Dict[str, Tensor]) so the numpy qr goldens type-check cleanly.
    out: dict[str, Any] = {k: ref[k] for k in ("q_lift", "q_pe", "c_kv", "k_pe")}
    out["qr_qtz"] = qr_np.reshape(b * s, ql).astype(np.float32)
    out["qr_scale"] = np.zeros((1,), dtype=np.uint8)  # unused golden; qr_scale checked via dequant
    return out


_FP8_AMAX_TEST = {"float8_e4m3": 240.0, "float8_e4m3fn": 448.0}
_AMAX_FLOOR_TEST = 1e-30


def _np_fp8(nl_dtype):
    """numpy view dtype for an NKI fp8 dtype, and its max finite magnitude."""
    name = str(nl_dtype)
    if "e4m3fn" in name:
        return ml_dtypes.float8_e4m3fn, _FP8_AMAX_TEST["float8_e4m3fn"]
    # Legacy (non-OCP) e4m3 caps at 240; ml_dtypes' closest encoding is float8_e4m3b11fnuz, but
    # only the AMAX matters here -- the comparison decodes both sides through the same dtype.
    return ml_dtypes.float8_e4m3, _FP8_AMAX_TEST["float8_e4m3"]


def _expected_cache_images(c_kv_golden, k_pe_golden, slot_mapping, num_slots, kv_lora_rank, cache_dtype, scale_region):
    """Replay the kernel's paged store in numpy: per-token fp8 quantize, then scatter by slot.

    Returns ``(c_data, c_scale, kpe)`` where ``c_data`` is the DEQUANTIZED latent per slot
    (fp32), ``c_scale`` the per-slot stored scale (fp32, or None for a bf16 cache) and ``kpe``
    the per-slot RoPE key. Comparing dequantized values rather than raw bytes keeps the check
    robust to fp8 rounding ties while still proving slot addressing, oob_mode.skip of padded
    tokens, the scale field, and that untouched slots stay zero.
    """
    c = np.asarray(
        c_kv_golden.to(torch.float32) if isinstance(c_kv_golden, torch.Tensor) else c_kv_golden, dtype=np.float32
    ).reshape(-1, kv_lora_rank)
    kp = np.asarray(
        k_pe_golden.to(torch.float32) if isinstance(k_pe_golden, torch.Tensor) else k_pe_golden, dtype=np.float32
    )
    kp = kp.reshape(len(c), -1)
    slots = np.asarray(slot_mapping).reshape(-1).astype(np.int64)

    c_data = np.zeros((num_slots, kv_lora_rank), dtype=np.float32)
    kpe = np.zeros((num_slots, kp.shape[1]), dtype=np.float32)
    c_scale = np.zeros((num_slots,), dtype=np.float32) if scale_region else None

    if scale_region:
        np_fp8, fp8_amax = _np_fp8(cache_dtype)
        amax = np.maximum(np.abs(c).max(axis=1), _AMAX_FLOOR_TEST)
        scale = (amax / fp8_amax).astype(np.float32)
        deq = (c / scale[:, None]).astype(np_fp8).astype(np.float32) * scale[:, None]
    else:
        scale = None
        deq = c.astype(ml_dtypes.bfloat16).astype(np.float32)

    for tok, slot in enumerate(slots):
        if slot < 0 or slot >= num_slots:
            continue
        c_data[slot] = deq[tok]
        kpe[slot] = kp[tok].astype(ml_dtypes.bfloat16).astype(np.float32)
        # Both are gated on scale_region, but narrowing does not carry across variables.
        if c_scale is not None and scale is not None:
            c_scale[slot] = scale[tok]
    return c_data, c_scale, kpe


def _ds128_to_native32(compact_scale, in_dim, out_dim):
    """Broadcast a DeepSeek compact block-128 scale [in//128, ceil(out/128)] uint8 to the
    NATIVE block-32 MX layout [in//32, out] the kernel consumes with compact_scales=False.

    Lossless: one 128x128 DS block covers 4 identical 32-K sub-blocks and 128 identical
    N columns. Mirrors the offline weight-prep broadcast, so the native tensor is exactly
    what the compact path would materialize in-kernel -> identical golden."""
    s = compact_scale.cpu().numpy() if hasattr(compact_scale, "cpu") else np.asarray(compact_scale)
    full = np.repeat(s, 4, axis=0)  # K: block-128 -> block-32 (4 sub-blocks)
    full = np.repeat(full, 128, axis=1)  # N: per-128-block -> per-column
    return np.ascontiguousarray(full[: in_dim // 32, :out_dim]).astype(np.uint8)


def _build_mla_qkv_weights(hidden, n_heads, qk_lora_rank, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim):
    """Build this kernel's wqkv_a / wq_b MX weights + BOTH compact-scale orderings.

    Local to the MLA kernel (rather than the shared ``build_qkv_mla_input``) because it must
    reduce the wqkv_a compact scale from the NATURAL columns, i.e. BEFORE the output-column
    swizzle: a swizzled 128-block spans 4 DIFFERENT natural blocks, so reducing after the
    swizzle discards 3 of every 4 block scales and cannot be inverted. Reuses the shared
    ``_swizzle_mla_cols`` / ``_reduce_mx_scale_to_compact_block128`` primitives.

    The WEIGHT is always pre-swizzled offline (qr + kv blocks; k_pe untouched), matching the
    production loader. Returns both scale forms so the test can drive either kernel path:
      - ``natural``: compact block-128 of the un-swizzled columns (compact_scales=True; the
        kernel reproduces the swizzled per-column scale in-kernel.
      - ``swizzled``: compact block-128 reduced from the swizzled columns (the source for the
        native block-32 layout, which ships pre-broadcast in swizzled order).
    """
    q_out_dim = n_heads * (qk_nope_head_dim + qk_rope_head_dim)
    kv_a_out_dim = kv_lora_rank + qk_rope_head_dim
    fused_qkv_dim = qk_lora_rank + kv_a_out_dim

    _, wq_a, wq_a_scale = generate_stabilized_mx_data(nl.float8_e4m3fn_x4, (hidden // 4, qk_lora_rank * 4), val_range=5)
    _, wkv_a, wkv_a_scale = generate_stabilized_mx_data(
        nl.float8_e4m3fn_x4, (hidden // 4, kv_a_out_dim * 4), val_range=5
    )
    _, wq_b, wq_b_scale = generate_stabilized_mx_data(
        nl.float8_e4m3fn_x4, (qk_lora_rank // 4, q_out_dim * 4), val_range=5
    )

    wqkv_a = np.concatenate([wq_a, wkv_a], axis=-1)
    wqkv_a_scale = np.concatenate([wq_a_scale, wkv_a_scale], axis=-1)

    # Natural-order compact scale: reduced BEFORE the swizzle (see docstring).
    scale_natural = _reduce_mx_scale_to_compact_block128(wqkv_a_scale.reshape(-1, fused_qkv_dim), hidden, fused_qkv_dim)

    def _swz_blocks(arr):
        qr = _swizzle_mla_cols(arr[..., :qk_lora_rank], qk_lora_rank)
        kv = _swizzle_mla_cols(arr[..., qk_lora_rank : qk_lora_rank + kv_lora_rank], kv_lora_rank)
        return np.concatenate([qr, kv, arr[..., qk_lora_rank + kv_lora_rank :]], axis=-1)

    wqkv_a = _swz_blocks(wqkv_a)
    scale_swizzled = _reduce_mx_scale_to_compact_block128(
        _swz_blocks(wqkv_a_scale).reshape(-1, fused_qkv_dim), hidden, fused_qkv_dim
    )
    wq_b_scale_compact = _reduce_mx_scale_to_compact_block128(
        wq_b_scale.reshape(-1, q_out_dim), qk_lora_rank, q_out_dim
    )
    return {
        "wqkv_a": wqkv_a,
        "wqkv_a_scale_natural": scale_natural,
        "wqkv_a_scale_swizzled": scale_swizzled,
        "wq_b": wq_b,
        "wq_b_scale": wq_b_scale_compact,
        "fused_qkv_dim": fused_qkv_dim,
        "q_out_dim": q_out_dim,
    }


def _build_rope_caches_bf16(batch, seqlen, qk_rope_head_dim, base=10000.0):
    """Build realistic ``[B, S, R]`` bf16 cos/sin RoPE caches.

    Mirrors HuggingFace: ``inv_freq[j] = base ** (-2j / R)`` for ``j`` in ``[0, R/2)``,
    ``emb = cat(freqs, freqs)``, so the per-pair angles live in the FIRST half of each row --
    which is the half both the kernel and the torch ref read. Identical construction for
    DeepSeek-V3.2 and GLM-MoE-DSA (both set ``config.head_dim = qk_rope_head_dim``).
    """
    inv_freq = 1.0 / (base ** (np.arange(0, qk_rope_head_dim, 2, dtype=np.float64) / qk_rope_head_dim))
    angles = np.arange(seqlen, dtype=np.float64)[:, None] * inv_freq[None, :]  # [S, R/2]
    emb = np.concatenate([angles, angles], axis=-1)  # [S, R], HF cat(freqs, freqs)
    cos = np.broadcast_to(np.cos(emb), (batch, seqlen, qk_rope_head_dim))
    sin = np.broadcast_to(np.sin(emb), (batch, seqlen, qk_rope_head_dim))
    return np.ascontiguousarray(cos).astype(nl.bfloat16), np.ascontiguousarray(sin).astype(nl.bfloat16)


def _build_mla_qkv_inputs_bf16(
    batch, seqlen, hidden, n_heads, qk_lora_rank, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim
):
    """Build the full BF16 (``MlaPrecision.BF16``) input set for the MLA QKV CTE kernel.

    Everything is plain bf16 in NATURAL column order -- no 4-packing, no MX scales, no MLA
    output-column swizzle -- which is the whole point of the bf16 path, so this is a handful
    of ``randn`` calls rather than the MX builders above.

    Each weight is scaled by ``1 / sqrt(K)`` so every matmul output stays roughly unit
    variance; bf16 has ~3 decimal digits of mantissa, so letting activations drift by orders
    of magnitude across the three chained projections would make the comparison measure
    dynamic range rather than the kernel.
    """
    np.random.seed(42)
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    qkv_out_dim = qk_lora_rank + kv_lora_rank + qk_rope_head_dim
    q_out_dim = n_heads * qk_head_dim

    def _randn(shape, scale):
        return (np.random.randn(*shape) * scale).astype(nl.bfloat16)

    cos_cache, sin_cache = _build_rope_caches_bf16(batch, seqlen, qk_rope_head_dim)
    return {
        "x_hbm": _randn((batch, seqlen, hidden), 1.0),
        "wqkv_a": _randn((hidden, qkv_out_dim), hidden**-0.5),
        "wq_b": _randn((qk_lora_rank, q_out_dim), qk_lora_rank**-0.5),
        # Gammas centred on 1.0: a norm scale, not a random projection.
        "q_norm_gamma": (1.0 + 0.1 * np.random.randn(1, qk_lora_rank)).astype(nl.bfloat16),
        "kv_norm_gamma": (1.0 + 0.1 * np.random.randn(1, kv_lora_rank)).astype(nl.bfloat16),
        "wuk": _randn((qk_nope_head_dim, n_heads * kv_lora_rank), qk_nope_head_dim**-0.5),
        "cos_cache": cos_cache,
        "sin_cache": sin_cache,
    }


def _build_wuk_bf16(n_heads, qk_nope_head_dim, kv_lora_rank):
    """Build the absorption weight ``W_uk`` (the K_b half of kv_b_proj) as a
    plain bf16 array for nope-contraction.

    W_uk has logical shape [nope, n_heads * kv_lora]; contraction = nope, head
    ``h`` owns columns [h*kv_lora, (h+1)*kv_lora). Values are small (randn*0.1)
    to keep the bf16 absorption in a benign numerical range.
    """
    out_dim = n_heads * kv_lora_rank
    wuk = (np.random.randn(qk_nope_head_dim, out_dim) * 0.1).astype(nl.bfloat16)
    return wuk


@pytest_test_metadata(name="MLA QKV CTE", tags=["model"])
@pytest_marks(["qkv", "cte", "mx", "mla"])
@final
class TestMlaQkvCteKernel:
    # fmt: off
    # scale_mode selects the wqkv_a/wq_b MX scale layout:
    #   "compact" - DeepSeek compact block-128 (default). wqkv_a's scale is the NATURAL
    #               (un-swizzled) column order and the kernel reproduces the swizzled
    #               per-column scale via a 32-run-tiled broadcast, so the offline loader
    #               never does the dequant->swizzle->REQUANT.
    #   "native"  - pre-broadcast native block-32 [K//32, N] already in swizzled column
    #               order: one DMA, no in-kernel broadcast.
    qkv_mla_absorbed_test_params = (
        "vnc_degree, batch, seqlen, hidden, n_heads, qk_lora_rank, "
        "kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim, norm_eps, scale_mode"
    )
    qkv_mla_absorbed_test_perms = [
        # Small H (2048) avoids the simulator's large-shape auto-skip (H=7168 is
        # in the skip list) so these cases run numerically in sim. Full 128 heads.
        pytest.param(2, 1, 256, 2048, 128, 1536, 512, 64, 128, 1e-6, "compact", marks=pytest.mark.fast),
        [2, 1, 512, 2048, 128, 1536, 512, 64, 128, 1e-6, "compact"],
        [2, 1, 128, 2048, 128, 1536, 512, 64, 128, 1e-6, "compact"],
        [1, 1, 128, 2048, 16, 1536, 512, 64, 128, 1e-6, "compact"],
        [2, 1, 128, 7168, 128, 1536, 512, 64, 128, 1e-6, "compact"],
        [2, 1, 256, 7168, 128, 1536, 512, 64, 128, 1e-6, "compact"],
        [2, 1, 512, 7168, 128, 1536, 512, 64, 128, 1e-6, "compact"],
        [2, 1, 256, 7168, 64, 1536, 512, 64, 128, 1e-6, "compact"],
        [2, 1, 512, 7168, 32, 1536, 512, 64, 128, 1e-6, "compact"],
        # S_local = 2048 at the production head count: the 128k-global/CP64 shard. Baseline for
        # the fp8 paged-store A/B (see qkv_mla_paged_test_perms).
        [2, 1, 2048, 7168, 128, 1536, 512, 64, 128, 1e-6, "compact"],
        [2, 1, 4096, 7168, 4, 1536, 512, 64, 128, 1e-6, "compact"],
        [2, 1, 8192, 7168, 2, 1536, 512, 64, 128, 1e-6, "compact"],
        # ---- SMALL sequence lengths at the full DeepSeek config (H=7168, 128 heads):
        [2, 1, 2,  7168, 128, 1536, 512, 64, 128, 1e-6, "compact"],
        [2, 1, 16, 7168, 128, 1536, 512, 64, 128, 1e-6, "compact"],
        [2, 1, 32, 7168, 128, 1536, 512, 64, 128, 1e-6, "compact"],
        # ---- NATIVE block-32 scales (compact_scales=False): pre-broadcast [K//32, N] in
        # swizzled column order, one DMA, no in-kernel broadcast.
        pytest.param(2, 1, 256, 2048, 128, 1536, 512, 64, 128, 1e-6, "native", marks=pytest.mark.fast),
        [2, 1, 128, 7168, 128, 1536, 512, 64, 128, 1e-6, "native"],
    ]

    # fmt: on

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest_parametrize(qkv_mla_absorbed_test_params, qkv_mla_absorbed_test_perms)
    def test_mla_qkv_cte_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden,
        n_heads,
        qk_lora_rank,
        kv_lora_rank,
        qk_rope_head_dim,
        qk_nope_head_dim,
        norm_eps,
        scale_mode,
    ):
        compact_scales = scale_mode != "native"
        compiler_args = CompilerArgs(
            logical_nc_config=vnc_degree,
            platform_target=platform_target,
            additional_cmd_args=["--enable-ocp-compliant-scale-computation"],
        )

        def input_generator(test_config):
            # Reuse the v32 builder for the shared first/second Q projections,
            # Shared v32 builder is used ONLY for the norm gammas + RoPE caches; the wqkv_a /
            # wq_b weights and scales come from _build_mla_qkv_weights below (this kernel
            # needs the natural-order scale, which must be reduced pre-swizzle).
            v32 = build_qkv_mla_input(
                batch=batch,
                seqlen=seqlen,
                hidden_dim=hidden,
                qk_lora_rank=qk_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                kv_lora_rank=kv_lora_rank,
                n_heads=n_heads,
                variant="v32",
                qk_nope_head_dim=qk_nope_head_dim,
                v_head_dim=qk_nope_head_dim,
            )
            np.random.seed(42)
            wuk_hbm = _build_wuk_bf16(n_heads, qk_nope_head_dim, kv_lora_rank)
            # PACKED MX input (rmsnorm_mx_prefill pack_scales=True), built by the proven MoE
            # helper; the torch ref decodes the same concat so both see identical activations.
            n_H512 = hidden // (128 * 4)
            concat, _ = build_prequantized_hidden_concat(batch * seqlen, n_H512, hidden)

            # wqkv_a / wq_b + both compact-scale orderings, built locally (the natural-order
            # scale must be reduced pre-swizzle; see _build_mla_qkv_weights).
            w = _build_mla_qkv_weights(hidden, n_heads, qk_lora_rank, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim)
            wq_b_scale = w["wq_b_scale"]
            if compact_scales:
                # Compact path: the NATURAL (un-swizzled) column-order scale.
                wqkv_a_scale = w["wqkv_a_scale_natural"]
            else:
                # Native block-32: broadcast the swizzled-order compact scale to the native
                # [K//32, N] layout offline (lossless), matching real weight-prep.
                wqkv_a_scale = _ds128_to_native32(w["wqkv_a_scale_swizzled"], hidden, w["fused_qkv_dim"])
                wq_b_scale = _ds128_to_native32(wq_b_scale, qk_lora_rank, w["q_out_dim"])
            return {
                "x_hbm_mx": np.asarray(concat).reshape(batch, seqlen, -1),
                "wqkv_a_hbm": w["wqkv_a"],
                "wqkv_a_scale_hbm": wqkv_a_scale,
                "wq_b_hbm": w["wq_b"],
                "wq_b_scale_hbm": wq_b_scale,
                "q_norm_gamma_hbm": v32["q_norm_gamma_hbm"],
                "kv_norm_gamma_hbm": v32["kv_norm_gamma_hbm"],
                "wuk_hbm": wuk_hbm,
                "cos_cache_hbm": v32["cos_cache_hbm"],
                "sin_cache_hbm": v32["sin_cache_hbm"],
                "n_heads": n_heads,
                "qk_nope_head_dim": qk_nope_head_dim,
                "qk_rope_head_dim": qk_rope_head_dim,
                "kv_lora_rank": kv_lora_rank,
                "qk_lora_rank": qk_lora_rank,
                "norm_eps": norm_eps,
                "compact_scales": compact_scales,
            }

        # qr export buffer shapes: one [P_MAX, qk_lora_rank//512, P_MAX] block per global
        # 128-query s-tile; fp8x4 packed as uint32 with paired uint8 block-32 scales.
        num_s_tiles = (seqlen + 127) // 128
        qr_lora_512 = qk_lora_rank // 512
        M = batch * seqlen

        def output_tensor_descriptor(kernel_input):
            return {
                "q_lift": np.zeros((batch, seqlen, n_heads, kv_lora_rank), dtype=nl.bfloat16),
                "q_pe": np.zeros((batch, seqlen, n_heads, qk_rope_head_dim), dtype=nl.bfloat16),
                "c_kv": np.zeros((batch, seqlen, kv_lora_rank), dtype=nl.bfloat16),
                "k_pe": np.zeros((batch, seqlen, qk_rope_head_dim), dtype=nl.bfloat16),
                "qr_qtz": np.zeros((num_s_tiles, 128, qr_lora_512, 128), dtype=np.uint32),
                "qr_scale": np.zeros((num_s_tiles, 128, qr_lora_512, 128), dtype=np.uint8),
            }

        # qr export check: dequantize the kernel's packed qr_qtz/qr_scale back to [M, qk_lora]
        # and compare to the reference q-normed qr with  the MX tolerance.
        M_dec = num_s_tiles * 128
        _qr_scale_holder: dict[str, npt.NDArray[Any]] = {}

        def _comparator(golden_dict, output_tensors):
            qr_golden = golden_dict["qr_qtz"]  # fp32 [M, qk_lora_rank] reference q-normed qr

            # Custom validators receive RAW uint8 bytes (load_output_tensor_as_bytes), so
            # reinterpret + reshape to the [num_s_tiles, 128, qr_lora_512, 128] export layout.
            _qr_shape = (num_s_tiles, 128, qr_lora_512, 128)

            class _ScaleStash(CustomValidator):
                # qr_scale carries no standalone golden; stash it for the qr_qtz dequant below
                # (validated in this dict BEFORE qr_qtz, so the buffer is available there).
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    _qr_scale_holder["scale"] = inference_output.view(np.uint8).reshape(_qr_shape)
                    return True

            class _QrValidator(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    qtz = inference_output.view(np.uint32).reshape(_qr_shape)
                    deq = _dequant_swizzled_qr_for_torch_ref(qtz, _qr_scale_holder["scale"], M_dec, qk_lora_rank)
                    return maxAllClose(deq[:M], qr_golden, rtol=8e-2, atol=1e-2, verbose=1, min_pass_rate=0.98)

            # Latent outputs: default allclose against the fp32 reference (plain ndarray golden).
            # qr_scale before qr_qtz so the stash runs first.
            return {
                "q_lift": golden_dict["q_lift"],
                "q_pe": golden_dict["q_pe"],
                "c_kv": golden_dict["c_kv"],
                "k_pe": golden_dict["k_pe"],
                "qr_scale": CustomValidatorWithOutputTensorData(_ScaleStash, output_tensors["qr_scale"]),
                "qr_qtz": CustomValidatorWithOutputTensorData(_QrValidator, output_tensors["qr_qtz"]),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=mla_qkv_cte_kernel,
            torch_ref=torch_ref_wrapper(_qkv_ref_with_qr),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensor_descriptor,
            check_unused_params=True,
        )
        # q_lift passes through two MX matmuls (stage1 + Q stage2) then a bf16
        # absorption matmul; the bf16 final stage adds little error beyond the
        # upstream MX rounding, so the standard MX rtol covers it.
        framework.run_test(
            test_config=None, compiler_args=compiler_args, rtol=8e-2, atol=1e-2, custom_comparator=_comparator
        )

    # fmt: off
    # BF16 path (MlaPrecision.BF16): unquantized bf16 activations + bf16 [K, N]
    qkv_mla_bf16_test_params = (
        "vnc_degree, batch, seqlen, hidden, n_heads, qk_lora_rank, "
        "kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim, norm_eps"
    )
    qkv_mla_bf16_test_perms = [
        # ---- DeepSeek-V3.2 dims in bf16: same shapes as the MX cases, different arithmetic. ----
        # Small H (2048) avoids the simulator's large-shape auto-skip (7168 is in the skip list).
        pytest.param(2, 1, 256, 2048, 128, 1536, 512, 64, 128, 1e-6, marks=pytest.mark.fast),
        [2, 1, 128, 2048, 128, 1536, 512, 64, 128, 1e-6],
        [2, 1, 512, 2048, 128, 1536, 512, 64, 128, 1e-6],
        [1, 1, 128, 2048,  16, 1536, 512, 64, 128, 1e-6],
        [2, 1, 128, 7168, 128, 1536, 512, 64, 128, 1e-6],
        [2, 1, 256, 7168,  64, 1536, 512, 64, 128, 1e-6],
        [2, 1, 4096, 7168,  4, 1536, 512, 64, 128, 1e-6],
        # ---- GLM-MoE-DSA (GLM 5.2 / 5.3). qk_nope_head_dim=192 -> absorption tiles 128 + 64. ----
        pytest.param(2, 1, 256, 2048, 64, 2048, 512, 64, 192, 1e-5, marks=pytest.mark.fast),
        [2, 1,    2, 6144,  64, 2048, 512, 64, 192, 1e-5],
        [2, 1,   16, 6144,  64, 2048, 512, 64, 192, 1e-5],
        [2, 1,   32, 6144,  64, 2048, 512, 64, 192, 1e-5],
        [2, 1,   64, 6144,  64, 2048, 512, 64, 192, 1e-5],
        [2, 1, 128, 2048,  64, 2048, 512, 64, 192, 1e-5],
        [2, 1, 128, 6144,  16, 2048, 512, 64, 192, 1e-5],
        [2, 1, 128, 6144,  64, 2048, 512, 64, 192, 1e-5],
        [2, 1, 256, 6144,  64, 2048, 512, 64, 192, 1e-5],
        [2, 1, 512, 2048,  64, 2048, 512, 64, 192, 1e-5], 
        [2, 1, 512, 6144,  64, 2048, 512, 64, 192, 1e-5],
        [2, 1, 1024, 6144, 64, 2048, 512, 64, 192, 1e-5],
        [2, 1, 2048, 6144, 64, 2048, 512, 64, 192, 1e-5],
        [2, 1, 4096, 6144, 64, 2048, 512, 64, 192, 1e-5],
        [2, 1, 4096, 6144,  8, 2048, 512, 64, 192, 1e-5],
        [2, 1, 8192, 6144, 64, 2048, 512, 64, 192, 1e-5],
        [2, 1, 16384, 6144, 64, 2048, 512, 64, 192, 1e-5],
    ]
    # fmt: on

    def _run_bf16(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden,
        n_heads,
        qk_lora_rank,
        kv_lora_rank,
        qk_rope_head_dim,
        qk_nope_head_dim,
        norm_eps,
    ):
        """BF16-path body, shared by the unit sweep below and the model-driven class.

        Extracted so the model configs the benchmark dashboard tracks run the SAME code as the
        functional sweep. A model class with its own copy of the harness setup would let the two
        drift, and a measurement taken on a path the functional tests do not exercise is not
        attributable to a validated kernel.
        """
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)

        def input_generator(test_config):
            inp = _build_mla_qkv_inputs_bf16(
                batch, seqlen, hidden, n_heads, qk_lora_rank, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim
            )
            return {
                # x_hbm_mx carries the PLAIN bf16 activation on this path (the param keeps its
                # MX-era name so one entry point serves both precisions).
                "x_hbm_mx": inp["x_hbm"],
                "wqkv_a_hbm": inp["wqkv_a"],
                "wqkv_a_scale_hbm": None,  # unused in bf16
                "wq_b_hbm": inp["wq_b"],
                "wq_b_scale_hbm": None,  # unused in bf16
                "q_norm_gamma_hbm": inp["q_norm_gamma"],
                "kv_norm_gamma_hbm": inp["kv_norm_gamma"],
                "wuk_hbm": inp["wuk"],
                "cos_cache_hbm": inp["cos_cache"],
                "sin_cache_hbm": inp["sin_cache"],
                "n_heads": n_heads,
                "qk_nope_head_dim": qk_nope_head_dim,
                "qk_rope_head_dim": qk_rope_head_dim,
                "kv_lora_rank": kv_lora_rank,
                "qk_lora_rank": qk_lora_rank,
                "norm_eps": norm_eps,
                "compact_scales": True,  # ignored in bf16
                "precision": MlaPrecision.BF16,
            }

        def output_tensor_descriptor(kernel_input):
            # BF16 returns 5 tensors: the four latents plus the plain-bf16 qr indexer export
            # (the MX path instead returns the pre-transposed, MX-quantized qr_qtz + qr_scale).
            return {
                "q_lift": np.zeros((batch, seqlen, n_heads, kv_lora_rank), dtype=nl.bfloat16),
                "q_pe": np.zeros((batch, seqlen, n_heads, qk_rope_head_dim), dtype=nl.bfloat16),
                "c_kv": np.zeros((batch, seqlen, kv_lora_rank), dtype=nl.bfloat16),
                "k_pe": np.zeros((batch, seqlen, qk_rope_head_dim), dtype=nl.bfloat16),
                "qr": np.zeros((batch, seqlen, qk_lora_rank), dtype=nl.bfloat16),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=mla_qkv_cte_kernel,
            torch_ref=torch_ref_wrapper(mla_qkv_cte_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensor_descriptor,
            check_unused_params=True,
        )
        """
        Tighter than the MX path's 8e-2: bf16 has ~8 mantissa bits against MX fp8's ~3 plus a
        block scale, so the only error here is bf16 rounding of the three chained matmuls'
        operands (the tensor engine accumulates in fp32).

        4e-2 against a measured worst case of 2.43%, which is the ``qr`` export at
        seqlen=2/GLM dims. ``qr`` is consistently the loosest output because the kernel runs
        its RMSNorm on a bf16 buffer -- two bf16 roundings of a near-unit quantity reduced
        over a 2048-wide row -- while the reference normalizes in fp32 and rounds once. The
        four latents production actually consumes stay at or under 1.52% even though they are
        computed FROM qr, since the following matmul averages that noise away.
        """
        framework.run_test(test_config=None, compiler_args=compiler_args, rtol=4e-2, atol=1e-2)

    # fmt: off
    # GLM-MoE-DSA (MlaPrecision.BF16) + the paged fp8 KV store -- the production config for GLM.
    # seqlen is S_local: 128 = 8k global context at CP64, 2048 = 128k. Each perm pairs with the
    # same-shape perm in qkv_mla_bf16_test_perms for an exact no-store / store A/B.
    qkv_mla_bf16_paged_test_params = (
        "vnc_degree, batch, seqlen, hidden, n_heads, qk_lora_rank, "
        "kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim, norm_eps"
    )
    qkv_mla_bf16_paged_test_perms = [
        # ---- SPARSE regime: S_local per CP64 rank for each global context length. ----
        # seqlen == S_local; global = S_local * 64.
        pytest.param(2, 1, 128, 6144, 64, 2048, 512, 64, 192, 1e-5, marks=pytest.mark.fast),  # 8k
        [2, 1, 256, 6144, 64, 2048, 512, 64, 192, 1e-5],   # 16k
        [2, 1, 512, 6144, 64, 2048, 512, 64, 192, 1e-5],   # 32k
        [2, 1, 1024, 6144, 64, 2048, 512, 64, 192, 1e-5],  # 64k
        [2, 1, 2048, 6144, 64, 2048, 512, 64, 192, 1e-5],  # 128k
        # ---- DENSE regime: global context <= the indexer's topk (2048), so attention runs
        # dense and S_local is tiny. These are also the store's tail-path cases: at vnc_degree 2
        # a seqlen of 2 leaves one token per core, i.e. s_tile_sz == 1.
        pytest.param(2, 1, 32, 6144, 64, 2048, 512, 64, 192, 1e-5, marks=pytest.mark.fast),  # 2k
        [2, 1, 16, 6144, 64, 2048, 512, 64, 192, 1e-5],  # 1k
        [2, 1, 2, 6144, 64, 2048, 512, 64, 192, 1e-5],   # 128, one token per core
    ]

    # fmt: on

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest_parametrize(qkv_mla_bf16_paged_test_params, qkv_mla_bf16_paged_test_perms)
    def test_mla_qkv_cte_bf16_paged_kv_store(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden,
        n_heads,
        qk_lora_rank,
        kv_lora_rank,
        qk_rope_head_dim,
        qk_nope_head_dim,
        norm_eps,
    ):
        """BF16 (GLM) path + the paged KV cache store: fp8 quantize then scatter to slot_mapping.

        Same contract as the MX variant: this checks the kernel still produces the SAME contiguous
        latents with the scatter enabled. The cache CONTENTS are a mutated input, which the harness
        does not compare, so the scatter's addressing is validated end-to-end once attention reads
        the paged cache.
        """
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)
        M = batch * seqlen
        # Legacy (non-OCP) fp8 is fine on this path: bf16 carries no float8_e4m3fn_x4 weights, so
        # NCC_EOCP001 -- which forces OCP e4m3fn on the MX path -- does not apply here.
        # Legacy (non-OCP) e4m3: the bf16 path carries no float8_e4m3fn_x4 weights, so
        # NCC_EOCP001 -- which forces OCP on the MX path -- does not apply here.
        latent_cache_dtype = nl.float8_e4m3
        num_slots = 2 * M
        n_pad = min(8, M)

        # Reversed token -> slot order so a contiguous scatter cannot pass; tail is padding
        # (slot -1), which the store must drop via oob_mode.skip.
        slot_mapping_ref = np.arange(M - 1, -1, -1, dtype=np.int32) * 2
        slot_mapping_ref[M - n_pad :] = -1
        # One array each, shared by the input generator and the output descriptor: the aliased
        # output IS the input buffer, so the descriptor must hand back the same object.
        # One fused bf16 row per token: fp8 latent bitcast 2-per-bf16, then the scale, then k_pe,
        # padded to a 16-byte stride.
        kv_row_bf16 = -(-(kv_lora_rank // 2 + 1 + qk_rope_head_dim) // 8) * 8
        kv_cache = np.zeros((num_slots, kv_row_bf16), dtype=nl.bfloat16)

        def input_generator(test_config):
            inp = _build_mla_qkv_inputs_bf16(
                batch, seqlen, hidden, n_heads, qk_lora_rank, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim
            )
            slot_mapping = slot_mapping_ref
            kernel_input = {
                "x_hbm_mx": inp["x_hbm"],
                "wqkv_a_hbm": inp["wqkv_a"],
                "wqkv_a_scale_hbm": None,  # unused in bf16
                "wq_b_hbm": inp["wq_b"],
                "wq_b_scale_hbm": None,  # unused in bf16
                "q_norm_gamma_hbm": inp["q_norm_gamma"],
                "kv_norm_gamma_hbm": inp["kv_norm_gamma"],
                "wuk_hbm": inp["wuk"],
                "cos_cache_hbm": inp["cos_cache"],
                "sin_cache_hbm": inp["sin_cache"],
                "n_heads": n_heads,
                "qk_nope_head_dim": qk_nope_head_dim,
                "qk_rope_head_dim": qk_rope_head_dim,
                "kv_lora_rank": kv_lora_rank,
                "qk_lora_rank": qk_lora_rank,
                "norm_eps": norm_eps,
                "compact_scales": True,  # ignored in bf16
                "precision": MlaPrecision.BF16,
                # ".must_alias_input" is what actually establishes the in/out alias (the suffix
                # the BIR emitter looks for); without it the harness hands the validator a buffer
                # unrelated to what the scatter wrote. Same convention as qkv_cte's k_cache.
                "kv_cache_hbm.must_alias_input": kv_cache,
                "slot_mapping_hbm": slot_mapping.reshape(batch, seqlen),
            }
            return kernel_input

        # The framework requires the ref's key set to equal the descriptor's, and the kernel writes
        # no contiguous c_kv / k_pe in this mode -- but the cache expectation is built from exactly
        # those reference latents. Capture them, then strip them from the returned dict.
        kv_golden: dict[str, Any] = {}

        @functools.wraps(mla_qkv_cte_torch_ref)
        def _ref_paged(*args, **kwargs):
            ref = dict(mla_qkv_cte_torch_ref(*args, **kwargs))
            kv_golden["c_kv"] = ref.pop("c_kv")
            kv_golden["k_pe"] = ref.pop("k_pe")
            return ref

        def output_tensor_descriptor(kernel_input):
            # The caches are MUTATED INPUTS the kernel also returns; the tracer's detect_must_alias
            # turns them into aliased outputs, which is what lets the harness read back what the
            # scatter actually wrote.
            # No c_kv / k_pe: with the paged store on, the kernel writes and returns only the
            # fp8 cache, so those two contiguous outputs no longer exist.
            return {
                "q_lift": np.zeros((batch, seqlen, n_heads, kv_lora_rank), dtype=nl.bfloat16),
                "q_pe": np.zeros((batch, seqlen, n_heads, qk_rope_head_dim), dtype=nl.bfloat16),
                "qr": np.zeros((batch, seqlen, qk_lora_rank), dtype=nl.bfloat16),
                "kv_cache": kv_cache,
            }

        def _comparator(golden_dict, output_tensors):
            """Validate what the scatter actually wrote into the paged cache.

            The cache comes back as an aliased output (the kernel returns the mutated input and the
            tracer's detect_must_alias turns it into a real one). This checks that the store puts the
            right VALUES in the right SLOTS: addressing, oob_mode.skip of padded tokens, the
            per-token scale field, and untouched slots staying zero.
            """
            np_fp8 = _np_fp8(latent_cache_dtype)[0]

            class _KvCache(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    """Decode the fused row per field and check it against the reference latents.

                    Row is bf16-CONTAINED: [0:L//2] is the fp8 latent bitcast 2-per-bf16, [L//2] the
                    bf16 scale, then R bf16 k_pe. Each field is read with its own dtype -- the latent
                    span is never interpreted as bf16 numerics.
                    """
                    raw = inference_output.view(np.uint8).reshape(num_slots, kv_row_bf16 * 2)
                    lat = raw[:, :kv_lora_rank].copy().view(np_fp8).astype(np.float32)
                    scale = raw[:, kv_lora_rank : kv_lora_rank + 2].copy().view(ml_dtypes.bfloat16)
                    scale = scale.astype(np.float32).reshape(num_slots)
                    kpe = raw[:, kv_lora_rank + 2 : kv_lora_rank + 2 + qk_rope_head_dim * 2]
                    kpe = kpe.copy().view(ml_dtypes.bfloat16).astype(np.float32)

                    """
                    Expected image comes from the reference latents, so the latent and scale absorb
                    the upstream projection error and both carry the projection's rtol. The scale is
                    amax/fp8_amax over the row: a max is stable in VALUE to within that error even
                    where the argmax differs.

                    The raw fp8 codes are deliberately NOT compared -- a scale differing by a
                    fraction of a step re-rounds them by a whole code, so only the DEQUANTIZED latent
                    is meaningful.
                    """
                    exp_data, exp_scale, exp_kpe = _expected_cache_images(
                        kv_golden["c_kv"],
                        kv_golden["k_pe"],
                        slot_mapping_ref,
                        num_slots,
                        kv_lora_rank,
                        latent_cache_dtype,
                        1,
                    )
                    ok_scale = maxAllClose(scale, exp_scale, rtol=4e-2, atol=1e-2, verbose=1)
                    ok_kpe = maxAllClose(kpe, exp_kpe, rtol=4e-2, atol=1e-2, verbose=1)
                    ok_lat = maxAllClose(
                        lat * scale[:, None], exp_data, rtol=4e-2, atol=1e-2, verbose=1, min_pass_rate=0.98
                    )
                    return bool(ok_scale and ok_kpe and ok_lat)

            # No c_kv / k_pe entries: the kernel does not write them when the paged store is on.
            return {
                "q_lift": golden_dict["q_lift"],
                "q_pe": golden_dict["q_pe"],
                "qr": golden_dict["qr"],
                "kv_cache": CustomValidatorWithOutputTensorData(_KvCache, output_tensors["kv_cache"]),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=mla_qkv_cte_kernel,
            # Uncacheable: a cache hit skips the ref, leaving kv_golden empty for the comparator.
            torch_ref=mark_uncacheable(torch_ref_wrapper(_ref_paged)),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensor_descriptor,
            check_unused_params=True,
        )
        framework.run_test(
            test_config=None, compiler_args=compiler_args, rtol=4e-2, atol=1e-2, custom_comparator=_comparator
        )

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest_parametrize(qkv_mla_bf16_test_params, qkv_mla_bf16_test_perms)
    def test_mla_qkv_cte_bf16_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden,
        n_heads,
        qk_lora_rank,
        kv_lora_rank,
        qk_rope_head_dim,
        qk_nope_head_dim,
        norm_eps,
    ):
        self._run_bf16(
            test_manager,
            platform_target,
            vnc_degree,
            batch,
            seqlen,
            hidden,
            n_heads,
            qk_lora_rank,
            kv_lora_rank,
            qk_rope_head_dim,
            qk_nope_head_dim,
            norm_eps,
        )


@pytest_marks(["qkv", "cte", "model", "mx", "mla_qkv_cte_kernel"])
class TestMlaQkvCteModel:
    """Model-driven tests for the MLA QKV CTE kernel, organized by tier.

    Parametrized from configuration/test_vector_metadata/, which is also what feeds the
    kernel-to-model benchmark dashboard, so a config registered there is a config that runs.

    Carries no @pytest_test_metadata of its own: one annotation per file is enforced, and the
    kernel is already named by the class above.

    The ``mx`` mark does not describe this kernel's arithmetic: GLM-5.3 runs the bf16 MLA
    path. The trn3 model lane requires one of the MX marks to admit a test
    (``-m "model and (optimal or tier0) and (mx or mxfp4 or mxfp8)"``), and per the lane
    owner we carry ``mx`` for now rather than widening that group to accept ``bf16``. Do not
    read it as a precision declaration; the precision field in the config table is what
    identifies the path.
    """

    _OPTIMAL_PARAMS, _OPTIMAL_IDS = prepare_model_parametrize(
        {
            ModelTestType.OPTIMAL: mla_qkv_cte_model_configs.get(ModelTestType.OPTIMAL, []),
        }
    )

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.optimal
    @pytest.mark.parametrize(MODEL_PARAM_NAMES, _OPTIMAL_PARAMS, ids=_OPTIMAL_IDS)
    def test_optimal(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden,
        n_heads,
        qk_lora_rank,
        kv_lora_rank,
        qk_rope_head_dim,
        qk_nope_head_dim,
        norm_eps,
        precision,
    ):
        """OPTIMAL: the model configs the benchmark dashboard tracks for GLM-5.3.

        Only the BF16 path is registered. GLM's ``qk_nope_head_dim`` is 192 and the MX path
        asserts 128, so an MX config at GLM dims could not run; the precision field is carried
        anyway so a measurement is identified with the path that produced it, and so a future
        MX registration needs no schema change.
        """
        if MlaPrecision[precision] is not MlaPrecision.BF16:
            pytest.skip(f"no model-driven MX path registered for this kernel yet, got {precision}")
        TestMlaQkvCteKernel()._run_bf16(
            test_manager,
            platform_target,
            vnc_degree,
            batch,
            seqlen,
            hidden,
            n_heads,
            qk_lora_rank,
            kv_lora_rank,
            qk_rope_head_dim,
            qk_nope_head_dim,
            norm_eps,
        )
