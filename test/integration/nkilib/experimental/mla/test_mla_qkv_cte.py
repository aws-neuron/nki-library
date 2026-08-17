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

import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
import torch
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
from test.utils.common_dataclasses import CompilerArgs, CustomValidator, CustomValidatorWithOutputTensorData, Platforms
from test.utils.comparators import maxAllClose
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


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
