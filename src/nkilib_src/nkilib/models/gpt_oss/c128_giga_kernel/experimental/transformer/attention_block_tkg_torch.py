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

"""PyTorch reference implementation for the fused attention block TKG kernel.

Pipeline::

    X ─► RMSNorm ─► QKV projection ─► QK norm (pre-RoPE) ─► RoPE
      ─► QK norm (post-RoPE) ─► FP8 KV quantization (optional)
      ─► Attention (with KV cache) ─► KV cache update
      ─► Output projection ─► X_out

Dimensions used throughout:
    B: Batch size
    S: Active sequence length (tokens being generated)
    H: Hidden dimension
    D: Head dimension (d_head)
    N: Number of query heads
    S_ctx: Context length (KV cache length visible to attention)
    S_max: Maximum context length (KV cache allocation size)
"""

import inspect
import math
from typing import Dict, Optional, Tuple, Union

import nki.language as nl
import torch

from ...core.attention.attention_tkg import INACTIVE_BLOCK_IDX
from ...core.attention.attention_tkg_torch import attention_tkg_torch_ref
from ...core.attention.attention_tkg_utils import (
    AttnTKGConfig,
    is_batch_sharded,
    is_qk_swapped,
    uses_flash_attention,
)
from ...core.embeddings.rope_torch import _rope_single_head
from ...core.output_projection.output_projection_tkg_torch import output_projection_tkg_torch_ref
from ...core.qkv.qkv_tkg_torch import qkv_tkg_torch_ref
from ...core.utils.common_types import DtypeMode, NormType, QKVOutputLayout, QuantizationType
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import get_max_positive_value_for_dtype
from ...core.utils.logging import get_logger
from ..collectives.distributed_adapter import get_pg

logger = get_logger("attention_block_tkg_torch")

P_MAX = 128  # Partition dimension size (nl.tile_size.pmax)


class AttentionBlockTkgTorchRef(torch.nn.Module):
    """PyTorch reference for the fused attention block TKG kernel.

    Stateful ``nn.Module`` whose ``forward`` implements the full pipeline.
    ``self.lnc`` is set at construction and used by the inner attention ref.

    Usage::

        ref = AttentionBlockTkgTorchRef(lnc=2)
        out = ref(X=..., ...)
    """

    def __init__(self, lnc: int = 2, kv_quant_dtype: str = nl.float8_e4m3):
        super().__init__()
        self.lnc = lnc
        self.kv_quant_dtype = kv_quant_dtype
        self.__signature__ = inspect.signature(self.forward)

    def forward(
        self,
        # -- input
        X: torch.Tensor,
        *,
        X_hidden_dim_actual: Optional[int],
        # -- rmsnorm X
        rmsnorm_X_enabled: bool,
        rmsnorm_X_eps: Optional[float],
        rmsnorm_X_gamma: Optional[torch.Tensor],
        # -- qkv projections
        W_qkv: torch.Tensor,
        bias_qkv: Optional[torch.Tensor],
        quantization_type_qkv: QuantizationType,
        weight_dequant_scale_qkv: Optional[torch.Tensor],
        input_dequant_scale_qkv: Optional[torch.Tensor],
        # -- Q/K processing: pre-RoPE RMSNorm
        rmsnorm_QK_pre_rope_enabled: bool,
        rmsnorm_QK_pre_rope_eps: float,
        rmsnorm_QK_pre_rope_W_Q: Optional[torch.Tensor],
        rmsnorm_QK_pre_rope_W_K: Optional[torch.Tensor],
        # -- RoPE
        cos: Optional[torch.Tensor],
        sin: Optional[torch.Tensor],
        rope_contiguous_layout: bool,
        # -- Q/K processing: post-RoPE RMSNorm
        rmsnorm_QK_post_rope_enabled: bool,
        rmsnorm_QK_post_rope_eps: float,
        rmsnorm_QK_post_rope_W_Q: Optional[torch.Tensor],
        rmsnorm_QK_post_rope_W_K: Optional[torch.Tensor],
        # -- attention
        skip_attention: bool,
        K_cache_transposed: bool,
        active_blocks_table: Optional[torch.Tensor],
        K_cache: torch.Tensor,
        V_cache: torch.Tensor,
        attention_mask: torch.Tensor,
        sink: Optional[torch.Tensor],
        softmax_scale: Optional[float],
        # -- FP8 KV cache quantization
        k_scale: Optional[torch.Tensor],
        v_scale: Optional[torch.Tensor],
        # -- KV cache update
        update_cache: bool,
        kv_cache_update_idx: torch.Tensor,
        # -- output projection
        W_out: Optional[torch.Tensor],
        bias_out: Optional[torch.Tensor],
        quantization_type_out: QuantizationType,
        weight_dequant_scale_out: Optional[torch.Tensor],
        input_dequant_scale_out: Optional[torch.Tensor],
        # -- output
        transposed_out: bool,
        transposed_in: bool = False,
        out_in_sb: bool = False,
        # -- kernel-only params (accepted for signature compatibility, validated below)
        sbm: None = None,
        X_in_sb: bool = False,
        KVDP: int = 1,
        KVDP_replica_group: None = None,
        KVDP_collective_mode=None,
        KVDP_rank: Optional[torch.Tensor] = None,
        enable_fa_s_prior_tiling: bool = True,
        fp8_packed: bool = False,
        pos_ids: Optional[torch.Tensor] = None,
        swa_start_pos_ids: Optional[torch.Tensor] = None,
        S_ctx: Optional[int] = None,
        is_h_transposed_by_4: bool = False,
        max_context_len=None,
        dtype_mode: DtypeMode = DtypeMode.NON_OCP,
        CP: int = 1,
        CP_replica_group: None = None,
        CP_collective_mode: None = None,
    ) -> Dict[str, torch.Tensor]:
        """PyTorch reference for the fused attention block TKG kernel.

        Implements the full fused attention block pipeline for token generation:
        RMSNorm → QKV projection → QK norm → RoPE → QK norm → FP8 KV quantization
        → Attention (with KV cache) → KV cache update → Output projection.

        Each stage delegates to its standalone torch reference where one exists
        (QKV, RoPE, attention, output projection). Stages without a standalone
        kernel (QK norm, FP8 quantization, KV cache update) are implemented
        inline in this module.

        Dimensions:
            B: Batch size
            S: Active sequence length (tokens being generated)
            H: Hidden dimension
            D: Head dimension (d_head)
            N: Number of query heads
            S_ctx: Context length (KV cache length visible to attention)
            S_max: Maximum context length (KV cache allocation size)

        Args:
            X: Input hidden states.
                Shape: ``[B, S, H]`` (default) or
                ``[H0=pmax, n_prgs, H1_shard, BxS]`` when ``transposed_in=True``.
            X_hidden_dim_actual: Actual hidden dimension when H is padded.
                None means H is the true dimension.

            rmsnorm_X_enabled: Apply RMSNorm to X before QKV projection.
            rmsnorm_X_eps: Epsilon for input RMSNorm.
            rmsnorm_X_gamma: Gamma (scale) for input RMSNorm. Shape: ``[1, H]``.

            W_qkv: QKV projection weight. Shape: ``[H, (N + 2) * D]``.
            bias_qkv: Optional QKV bias. Shape: ``[1, (N + 2) * D]``.
            quantization_type_qkv: Quantization for QKV projection (NONE, ROW, STATIC).
            weight_dequant_scale_qkv: QKV weight dequantization scale.
            input_dequant_scale_qkv: QKV input dequantization scale.

            rmsnorm_QK_pre_rope_enabled: Apply per-head RMSNorm to Q/K before RoPE.
            rmsnorm_QK_pre_rope_eps: Epsilon for pre-RoPE QK norm.
            rmsnorm_QK_pre_rope_W_Q: Gamma for Q pre-RoPE norm. Shape: ``[1, D]``.
            rmsnorm_QK_pre_rope_W_K: Gamma for K pre-RoPE norm. Shape: ``[1, D]``.

            cos: RoPE cosine table. Shape: ``[D//2, B, S]``. None to skip RoPE.
            sin: RoPE sine table. Shape: ``[D//2, B, S]``. None to skip RoPE.
            rope_contiguous_layout: RoPE half-dimension split layout.

            rmsnorm_QK_post_rope_enabled: Apply per-head RMSNorm to Q/K after RoPE.
            rmsnorm_QK_post_rope_eps: Epsilon for post-RoPE QK norm.
            rmsnorm_QK_post_rope_W_Q: Gamma for Q post-RoPE norm. Shape: ``[1, D]``.
            rmsnorm_QK_post_rope_W_K: Gamma for K post-RoPE norm. Shape: ``[1, D]``.

            skip_attention: If True, skip attention and pass Q through directly.
            K_cache_transposed: K cache layout is ``[B, 1, D, S_max]`` vs ``[B, 1, S_max, D]``.
            active_blocks_table: Block-to-slot mapping for block KV cache. None for non-block KV.
                Shape: ``[B, S_ctx // block_len]`` when ``kv_heads == 1``;
                ``[B, kv_heads, S_ctx // block_len]`` when ``kv_heads > 1`` (per-head block tables).
            K_cache: Key cache tensor (see :func:`_update_kv_cache` for shapes).
            V_cache: Value cache tensor.
            attention_mask: Attention mask. Shape: ``[S_ctx, B, N, S]`` when
                pos_ids is None, ``[S, B, N, S]`` (active-only) when pos_ids is provided.
                When KVDP > 1: ``[S_ctx, B_attn, q_heads_attn, S_tkg]``.
            sink: Optional attention sink tensor. Forwarded to attention ref.
            softmax_scale: Custom softmax scale. None uses ``1/√D``. When using FP8
                KV cache with None, k_scale is automatically absorbed effectively setting
                ``softmax_scale = (1/√D) / k_scale``. When explicitly provided, caller
                must incorporate k_scale: ``softmax_scale = softmax_scale / k_scale``.
            pos_ids: Optional position IDs for in-kernel mask generation.
                Shape: ``[B, S]``. When provided, the prior causal mask is generated
                on-chip from position IDs and attention_mask carries only the
                active-to-active portion.
            swa_start_pos_ids: Optional per-query sliding window start positions.
                Shape: ``[B, S]``. When provided with pos_ids, generates a banded
                SWA mask attending to positions in ``[start_pos, pos_id)``.
            S_ctx: Optional explicit context length for flat KV with pos_ids.
                Must be provided when using flat KV with pos_ids, otherwise None.
            is_h_transposed_by_4: Whether input X and RMSNorm gamma have been
                pre-shuffled along the H dimension for MXFP quantization.
                Forwarded to the QKV projection reference. Default: False.
            max_context_len: Optional int32 array [1] for dynamic FA early exit.
                Limits KV context processed to at most max_context_len positions.

            k_scale: FP8 quantization scale for K. Shape: ``[1, 1]`` or ``[P, 1]``.
                None to disable FP8 KV quantization.
            v_scale: FP8 quantization scale for V. Same shape convention as k_scale.

            update_cache: Whether to write new K/V into the cache.
            kv_cache_update_idx: Cache write positions.
                Shape (kv_heads == 1): ``[B, S_tkg]`` for block KV, ``[B, 1]`` for flat KV.
                Shape (kv_heads >  1): ``[B, kv_heads, S_tkg]`` for block KV (per-head physical
                slots); ``[B, 1]`` for flat KV (start position is shared across kv heads).
                For flat KV, only the start position is needed; consecutive tokens are assumed.

            W_out: Output projection weight. Shape: ``[N*D, H]``. None to skip.
                When using FP8 KV cache, should incorporate v_scale:
                ``W_out / v_scale`` (NONE) or absorb into weight_dequant_scale_out (ROW/STATIC).
            bias_out: Output projection bias. Shape: ``[1, H]``.
            quantization_type_out: Quantization for output projection.
            weight_dequant_scale_out: Output projection weight dequantization scale.
            input_dequant_scale_out: Output projection input dequantization scale.

            transposed_out: Transpose the final output.
            transposed_in: When True, X is in transposed HBM layout
                ``[H0=pmax, n_prgs, H1_shard, BxS]``. The ref converts back to
                ``[B, S, H]`` before processing. Default: False.
            out_in_sb: Output in SBUF layout (``[D, B, N, S]`` instead of ``[B, N, D, S]``).

            sbm: Accepted for signature compatibility (kernel-only, SBUF memory handle).
            KVDP: Number of KV data parallelism ranks. When > 1, the torch ref
                performs KVDP collectives (Q all_to_all, K/V batch slice,
                output all_to_all) to match the kernel's per-rank behavior.
            KVDP_replica_group: Replica group for KVDP collectives. Used to
                obtain the process group via ``get_pg()``.
            KVDP_collective_mode: Accepted for signature compatibility (kernel-only).
            KVDP_rank: This rank's position within its KVDP replica group (0 to KVDP-1).
                Used for K/V batch slicing. Shape: ``[1]``, dtype ``uint32``.
            enable_fa_s_prior_tiling: Accepted for signature compatibility
                 (kernel-only, whether to enable flash attention in kernel).
            CP: Context parallelism degree (1 = disabled). When > 1, the torch
                ref performs CP collectives (Q all_gather, distributed softmax
                correction, output all_to_all) to match the kernel's per-rank behavior.
            CP_replica_group: Replica group for CP collectives.
            CP_collective_mode: Accepted for signature compatibility (kernel-only).
            Dict with keys:
                - ``"X_out"``: Output tensor. Shape depends on ``W_out``, ``transposed_out``,
                  and ``out_in_sb`` settings.
                - ``"K_cache_updated"`` / ``"K_tkg"``: Updated K cache (or raw K if
                  ``update_cache`` is False).
                - ``"V_cache_updated"`` / ``"V_tkg"``: Updated V cache (or raw V).
        """
        X = X.float()
        # The transposed s_active_bqh-partition mask layout [B, N, S, S_ctx] applies ONLY to the
        # pre-generated full mask (pos_ids is None). This reference only implements the default
        # [S_ctx, B, N, S] layout, so transpose it back before anything (the transposed_in X reshape,
        # _extract_shapes) reads the mask axes. With pos_ids the HBM tensor is the active mask
        # [S_tkg, B, N, S] (same for swap and default), so no transpose. Swap is always block-KV, so
        # B/q_heads/S_ctx come from layout-independent tensors; S_tkg is X.shape[1] or BxS // B when
        # transposed_in.
        if active_blocks_table is not None and pos_ids is None:
            B = active_blocks_table.shape[0]
            kv_heads_mask = V_cache.shape[1] if V_cache.dim() == 4 else 1
            d_head_mask = V_cache.shape[-1]
            q_heads_mask = W_qkv.shape[1] // d_head_mask - 2 * kv_heads_mask
            S_ctx_mask = active_blocks_table.shape[-1] * V_cache.shape[-2]
            S_tkg_mask = X.shape[-1] // B if transposed_in else X.shape[1]
            if is_qk_swapped(
                bs=B,
                q_head=q_heads_mask,
                d_head=d_head_mask,
                s_active=S_tkg_mask,
                curr_sprior=S_ctx_mask,
                lnc=self.lnc,
                p_max=P_MAX,
                block_len=V_cache.shape[-2],
                is_2byte_kv=K_cache.dtype in (torch.bfloat16, torch.float16),
                fp8_packed=fp8_packed,
                fuse_rope=False,
                kv_heads=kv_heads_mask,
            ):
                # Partition banding: for small per-NC batch the kernel folds s_prior onto the partition
                # axis, so the mask arrives banded as [bs_n_prgs * P_MAX, S_ctx // band_factor] (2D).
                # Un-band it back to the logical [B, N, S, S_ctx] this ref computes on — inverse of the
                # banding in gen_mask_tkg_hbm_torch_ref (and gen_mask_tkg_hbm), using the same (bs, q_head)
                # the mask was generated with (B == B_attn, q_heads_mask == num_mask_heads).
                if attention_mask.dim() == 2:
                    sqh = q_heads_mask * S_tkg_mask
                    batch_sharded = self.lnc > 1 and is_batch_sharded(B, q_heads_mask, S_tkg_mask, S_ctx_mask, P_MAX)
                    bs_n_prgs = self.lnc if batch_sharded else 1
                    bs_per_nc = B // bs_n_prgs
                    batches_per_psum = P_MAX // sqh
                    bf = batches_per_psum // min(bs_per_nc, batches_per_psum) if bs_per_nc < batches_per_psum else 1
                    banded_sprior = S_ctx_mask // bf
                    p_max = bs_per_nc * bf * sqh  # == P_MAX in the banding regime
                    banded = attention_mask.reshape(bs_n_prgs, p_max, banded_sprior)
                    logical = torch.zeros((bs_n_prgs, bs_per_nc * sqh, S_ctx_mask), dtype=banded.dtype)
                    use_fa, fa_tile_size = uses_flash_attention(enable_fa_s_prior_tiling, S_ctx_mask)
                    if not use_fa:
                        fa_tile_size = S_ctx_mask
                    fa_offset = 0
                    band_free = 0
                    while fa_offset < S_ctx_mask:
                        tile_sp = min(fa_tile_size, S_ctx_mask - fa_offset)
                        band_sp = tile_sp // bf
                        # banded chunk -> [n_prgs, bs_per_nc, bf, sqh, band_sp], inverse of the gen-side permute.
                        chunk = banded[:, :, band_free : band_free + band_sp].reshape(
                            bs_n_prgs, bs_per_nc, bf, sqh, band_sp
                        )
                        chunk = chunk.permute(0, 1, 3, 2, 4)  # [n_prgs, bs_per_nc, sqh, bf, band_sp]
                        logical[:, :, fa_offset : fa_offset + tile_sp] = chunk.reshape(
                            bs_n_prgs, bs_per_nc * sqh, tile_sp
                        )
                        fa_offset += tile_sp
                        band_free += band_sp
                    attention_mask = logical.reshape(B * sqh, S_ctx_mask).reshape(
                        B, q_heads_mask, S_tkg_mask, S_ctx_mask
                    )
                attention_mask = attention_mask.permute(3, 0, 1, 2)

        if transposed_in:
            # Convert [H0, n_prgs, H1_shard, BxS] back to [B, S_tkg, H]
            H0, n_prgs, H1_shard, BxS = X.shape
            H = H0 * n_prgs * H1_shard
            _, _, _, S_tkg_mask = attention_mask.shape
            B_from_x = BxS // S_tkg_mask
            X = X.permute(3, 1, 0, 2).reshape(B_from_x, S_tkg_mask, H)
        # Infer kv_heads from the K_cache rank. When kv_heads > 1, the explicit kv-head axis at dim 1 is required.
        # base_ndim is the rank without a head dimension; k_ndim_with_head is one rank above it.
        base_ndim = 4 if fp8_packed else 3
        k_ndim_with_head = base_ndim + 1
        if K_cache.dim() == k_ndim_with_head:
            kv_heads = K_cache.shape[1]
        else:
            kernel_assert(
                K_cache.dim() == base_ndim,
                f"K_cache must have {base_ndim} (kv_heads==1) or {k_ndim_with_head} (explicit kv_heads) dims, "
                f"got {K_cache.dim()}",
            )
            kv_heads = 1

        # Normalize an explicit kv-head axis (dim 1) out of a block-KV cache by merging it
        block_kv_had_head_dim = active_blocks_table is not None and K_cache.dim() == k_ndim_with_head
        if block_kv_had_head_dim:
            K_cache = K_cache.reshape((K_cache.shape[0] * kv_heads,) + tuple(K_cache.shape[2:]))
            V_cache = V_cache.reshape((V_cache.shape[0] * kv_heads,) + tuple(V_cache.shape[2:]))
        # fp8_packed: unpack [num_blocks, block_len//2, d_head, 2] -> [num_blocks, block_len, d_head]
        if fp8_packed and active_blocks_table is not None and K_cache.dim() == 4:
            num_blocks, half_bl, d_head_k, _ = K_cache.shape
            K_cache = K_cache.permute(0, 1, 3, 2).reshape(num_blocks, half_bl * 2, d_head_k)
        B, S_tkg, H, d_head, q_heads, S_ctx, S_max_ctx, blk_len = self._extract_shapes(
            X,
            W_qkv,
            K_cache,
            K_cache_transposed,
            attention_mask,
            active_blocks_table,
            pos_ids,
            S_ctx,
            kv_heads=kv_heads,
        )

        # -- QKV projection (reuse qkv_tkg_torch_ref)

        qkv_result = qkv_tkg_torch_ref(
            hidden=X,
            qkv_w=W_qkv,
            norm_w=rmsnorm_X_gamma if rmsnorm_X_enabled else None,
            norm_type=NormType.RMS_NORM if rmsnorm_X_enabled else NormType.NO_NORM,
            eps=rmsnorm_X_eps if rmsnorm_X_eps else 1e-6,
            d_head=d_head,
            num_kv_heads=kv_heads,
            num_q_heads=q_heads,
            output_layout=QKVOutputLayout.NBSd,
            quantization_type=quantization_type_qkv,
            qkv_w_scale=weight_dequant_scale_qkv,
            qkv_in_scale=input_dequant_scale_qkv,
            qkv_bias=bias_qkv,
            hidden_actual=X_hidden_dim_actual,
            is_h_dim_4h_transposed=is_h_transposed_by_4,
            dtype_mode=dtype_mode,
        )
        QKV = qkv_result['out'].float()  # [N+2, B, S, D]
        # Free large intermediate to reduce peak memory in SimDistRunner sequential passes
        del qkv_result

        # -- QK norm pre RoPE (in-place on QKV)
        QKV = self._qk_norm_pre_rope(
            QKV,
            q_heads,
            kv_heads,
            d_head,
            rmsnorm_QK_pre_rope_enabled,
            rmsnorm_QK_pre_rope_eps,
            rmsnorm_QK_pre_rope_W_Q,
            rmsnorm_QK_pre_rope_W_K,
        )

        # -- RoPE → Q [D, B, N, S], K [D, B, kv_heads, S] (always 4D)
        cos_f = cos.float() if cos is not None else None
        sin_f = sin.float() if sin is not None else None
        Q, K = self._apply_rope_to_qk(QKV, cos_f, sin_f, q_heads, kv_heads, d_head, rope_contiguous_layout)
        del cos_f, sin_f  # Free to reduce peak memory in SimDistRunner

        # -- QK norm post RoPE
        Q, K = self._qk_norm_post_rope(
            Q,
            K,
            d_head,
            rmsnorm_QK_post_rope_enabled,
            rmsnorm_QK_post_rope_eps,
            rmsnorm_QK_post_rope_W_Q,
            rmsnorm_QK_post_rope_W_K,
        )

        # V: last kv_heads rows of QKV [kv_heads, B, S, D] -> [B, kv_heads, S, D]. Always 4D
        # (kv_heads == 1 produces [B, 1, S, D]). Clone so `del QKV` actually frees QKV's storage.
        V = QKV[q_heads + kv_heads :].permute(1, 0, 2, 3).clone()  # [B, kv_heads, S, D]
        del QKV

        # -- FP8 KV cache quantization (optional)
        kv_quant = k_scale is not None and v_scale is not None
        K_new = K.clone()
        V_new = V.clone()
        if kv_quant:
            K_new, V_new = self._quantize_kv_to_fp8(K_new, V_new, k_scale, v_scale)

        # -- Collective setup
        is_KVDP = KVDP > 1
        is_CP = CP > 1
        # CP + pos_ids (in-kernel mask gen) is supported; the caller owns CP semantics:
        # pass pos_ids = local_filled (rank-local cache is contiguous) and an
        # ownership-gated active-only mask (see attention_block_tkg for the contract).
        kvdp_pg = get_pg(KVDP_replica_group) if is_KVDP else None
        cp_pg = get_pg(CP_replica_group) if is_CP else None

        # -- KVDP input collectives: redistribute Q heads across ranks, slice K/V batch.
        # KV heads are replicated across ranks (not gathered), so K/V are only batch-sliced.
        if is_KVDP:
            kernel_assert(KVDP_rank is not None, "KVDP_rank tensor is required when KVDP > 1")
            kvdp_rank_int = int(KVDP_rank.item())
            Q, K_new, V_new, K_cache, V_cache = self._kvdp_input_collectives(
                Q,
                K_new,
                V_new,
                K_cache,
                V_cache,
                q_heads,
                kv_heads,
                KVDP,
                B,
                kvdp_pg,
                kvdp_rank_int,
            )

        B_attn = B // KVDP  # KVDP=1 when disabled
        q_heads_attn = q_heads * KVDP * CP  # KVDP=1, CP=1 when disabled
        kernel_assert(
            attention_mask.shape[2] == q_heads_attn,
            f"attention_mask head dim mismatch: expected {q_heads_attn}, got {attention_mask.shape[2]}",
        )

        # -- CP input collectives: all-gather Q heads across CP ranks
        if is_CP:
            # Q [D, B, q_heads_attn/CP, S] -> all_gather -> [D, B, q_heads_attn, S]
            Q = self._cp_input_collectives(Q, CP, cp_pg)

        # -- Attention (reuse attention_tkg_torch_ref)
        # Default output for the skip_attention + no output projection path.
        output = Q.clone()  # [D,B_attn,N_attn,S]
        if skip_attention:
            attn_out = Q.permute(1, 2, 0, 3)  # [D,B_attn,N_attn,S] -> [B_attn,N_attn,D,S]
        else:
            # Fold kv_heads into batch for attention, run, then unfold.
            # Inputs (post any KVDP/CP collectives):
            #   Q [D, B_attn, q_heads_attn, S],
            #   K_new [D, B_attn, S]      (kv_heads==1) or [D, B_attn, kv_heads, S]      (kv_heads>1),
            #   V_new [B_attn, S, D]      (kv_heads==1) or [B_attn, kv_heads, S, D]      (kv_heads>1),
            #   K_cache [B_attn, 1, ...]  (kv_heads==1, flat) or [B_attn, kv_heads, ...] (kv_heads>1, flat).
            d_head = Q.shape[0]
            S_tkg = Q.shape[-1]
            q_per_group = q_heads_attn // kv_heads
            B_folded = B_attn * kv_heads

            # Q/K/V/mask fold: reshapes (b-outer/kv-inner: B_folded = B_attn * kv_heads).
            Q_folded = Q.reshape(d_head, B_folded, q_per_group, S_tkg)
            K_folded = K_new.reshape(d_head, B_folded, S_tkg)
            V_folded = V_new.reshape(B_folded, S_tkg, d_head)
            mask_folded = attention_mask.reshape(attention_mask.shape[0], B_folded, q_per_group, S_tkg)
            # Cache fold: block KV uses a shared pool (only fold active_blocks_table); flat KV folds
            # the [B_attn, kv_heads, ...] cache to [B_folded, 1, ...].
            is_block_kv = blk_len > 0
            if is_block_kv:
                K_cache_folded = K_cache
                V_cache_folded = V_cache
                active_blocks_table_folded = active_blocks_table.reshape(B_folded, -1)
            else:
                K_cache_folded = K_cache.reshape(B_folded, 1, *K_cache.shape[2:])
                V_cache_folded = V_cache.reshape(B_folded, 1, *V_cache.shape[2:])
                active_blocks_table_folded = None
            # pos_ids / swa fold: head-independent, replicate each real batch's [S_tkg] across kv_heads.
            # TODO: drop the replication and broadcast at the consumption site (mirror sink in _prep_sink).
            pos_ids_folded = (
                pos_ids.reshape(B_attn, 1, S_tkg).expand(B_attn, kv_heads, S_tkg).reshape(B_folded, S_tkg)
                if pos_ids is not None
                else None
            )
            swa_folded = (
                swa_start_pos_ids.reshape(B_attn, 1, S_tkg).expand(B_attn, kv_heads, S_tkg).reshape(B_folded, S_tkg)
                if swa_start_pos_ids is not None
                else None
            )

            attn_result = self._run_attention(
                Q_folded,
                K_folded,
                V_folded,
                K_cache_folded,
                V_cache_folded,
                mask_folded,
                active_blocks_table_folded,
                K_cache_transposed,
                blk_len,
                S_ctx,
                S_max_ctx,
                softmax_scale,
                q_per_group,
                sink=sink,
                pos_ids=pos_ids_folded,
                swa_start_pos_ids=swa_folded,
                k_scale=k_scale,
                dtype_mode=dtype_mode,
                return_cp_softmax_stats=is_CP,
            )
            if is_CP:
                # Folded result: unnormalized output + local softmax stats, [B_folded, q_per_group, ...].
                attn_out_unnorm, softmax_max, softmax_sum = attn_result
                # Unfold to flat [B_attn, q_heads_attn] before the CP combine: the folded
                # [B_folded, q_per_group] view would chunk the CP all-to-all on the wrong head axis
                # (CP-replicated heads interleaved with kv-groups), sending heads to the wrong ranks.
                attn_unnorm_flat = attn_out_unnorm.reshape(B_attn, q_heads_attn, d_head, S_tkg)
                smax_flat = softmax_max.reshape(B_attn, q_heads_attn, S_tkg)
                ssum_flat = softmax_sum.reshape(B_attn, q_heads_attn, S_tkg)
                # Per-CP-rank head count after the CP reduce = q_heads_attn / CP = q_heads * KVDP.
                q_heads_per_cp_rank = q_heads_attn // CP
                attn_out = self._cp_output_collectives(
                    attn_unnorm_flat, smax_flat, ssum_flat, q_heads_per_cp_rank, CP, cp_pg
                )  # [B_attn, q_heads*KVDP, D, S]
            else:
                # Unfold: [B_folded, q_per_group, D, S] -> [B_attn, kv_heads*q_per_group, D, S].
                attn_out = attn_result.reshape(B_attn, kv_heads * q_per_group, d_head, S_tkg)
            output = attn_out.permute(2, 0, 1, 3) if out_in_sb else attn_out

        # -- KVDP output collectives: redistribute attention output back
        if is_KVDP:
            attn_out = self._kvdp_output_collectives(attn_out, KVDP, kvdp_pg, KVDP_collective_mode, kvdp_rank_int)
            output = attn_out.permute(2, 0, 1, 3) if out_in_sb else attn_out

        # -- KV cache update
        # After KVDP input collectives, K_new/V_new and the cache are per-rank (B_attn batches);
        # for KVDP=1 they have full batch B. With kv_heads>1 they carry the kv_heads dimension.
        K_out, V_out = self._update_kv_cache(
            K_new,
            V_new,
            K_cache.float(),
            V_cache.float(),
            kv_cache_update_idx,
            update_cache,
            K_cache_transposed,
            blk_len,
            d_head,
            kv_heads=kv_heads,
        )

        # -- Output projection (reuse output_projection_tkg_torch_ref)
        if W_out is not None:
            attn_D_B_N_S = attn_out.permute(2, 0, 1, 3)  # [B,N,D,S] -> [D,B,N,S]
            out_result = output_projection_tkg_torch_ref(
                attention=attn_D_B_N_S,
                weight=W_out,
                bias=bias_out,
                quantization_type=quantization_type_out,
                weight_scale=weight_dequant_scale_out,
                input_scale=input_dequant_scale_out,
                TRANSPOSE_OUT=transposed_out,
                dtype_mode=dtype_mode,
            )
            output = out_result['out']

        # -- Build result dict
        if update_cache:
            # fp8_packed: repack K_cache_updated [n, bl, d] -> [n, bl//2, d, 2]
            if fp8_packed and active_blocks_table is not None:
                num_blocks, block_len_full, d_head_k = K_out.shape
                K_out = K_out.reshape(num_blocks, block_len_full // 2, 2, d_head_k).permute(0, 1, 3, 2)
            # Restore the kv-head axis removed on input
            if block_kv_had_head_dim:
                K_out = K_out.reshape((K_out.shape[0] // kv_heads, kv_heads) + tuple(K_out.shape[1:]))
                V_out = V_out.reshape((V_out.shape[0] // kv_heads, kv_heads) + tuple(V_out.shape[1:]))
            return {"X_out": output, "K_cache_updated": K_out, "V_cache_updated": V_out}
        return {"X_out": output, "K_tkg": K_out, "V_tkg": V_out}

    # ── Helper methods ──────────────────────────────────────────────────────

    def _extract_shapes(
        self,
        X: torch.Tensor,
        W_qkv: torch.Tensor,
        K_cache: torch.Tensor,
        K_cache_transposed: bool,
        attention_mask: torch.Tensor,
        active_blocks_table: Optional[torch.Tensor],
        pos_ids: Optional[torch.Tensor],
        S_ctx_param: Optional[int],
        kv_heads: int = 1,
    ) -> Tuple[int, int, int, int, int, int, int, int]:
        """Extract and validate tensor dimensions, mirroring the kernel's
        ``_validate_and_extract_config``.

        Expected input shapes:

        - ``X``:  ``(B, S_tkg, H)``
        - ``W_qkv``:  ``(H, (q_heads + 2) * d_head)``
        - ``K_cache`` flat:  ``(B, 1, S_max_ctx, d_head)``  or transposed ``(B, 1, d_head, S_max_ctx)``
        - ``K_cache`` block:  ``(n_blocks, blk_len, d_head)``
        - ``attention_mask``:  ``(S_ctx, B, q_heads, S_tkg)`` or ``(S_tkg, B, q_heads, S_tkg)`` when pos_ids is provided

        Returns:
            ``(B, S_tkg, H, d_head, q_heads, S_ctx, S_max_ctx, blk_len)``
        """
        B, S_tkg, H = X.shape  # (B, S_tkg, H)

        is_block_kv = active_blocks_table is not None

        if is_block_kv:
            # block: (n_blocks, blk_len, d_head)
            d_head = K_cache.shape[2]
        elif K_cache_transposed:
            # flat transposed: (B, 1, d_head, S_max_ctx)
            d_head = K_cache.shape[2]
        else:
            # flat: (B, 1, S_max_ctx, d_head)
            d_head = K_cache.shape[3]

        # W_qkv packs Q, K, V heads: (H, (q_heads + 2*kv_heads) * d_head)
        q_heads = W_qkv.shape[1] // d_head - 2 * kv_heads
        kernel_assert(q_heads > 0, f"q_heads must be > 0, got {q_heads}")
        kernel_assert(d_head % 2 == 0, f"d_head must be even, got {d_head}")

        use_pos_id = pos_ids is not None
        if use_pos_id and not is_block_kv:
            kernel_assert(S_ctx_param is not None, "S_ctx is required when using pos_ids with flat KV")
        else:
            kernel_assert(S_ctx_param is None, "S_ctx must be None when not using pos_ids or with block KV")

        S_ctx = attention_mask.shape[0] if not use_pos_id else None
        blk_len = K_cache.shape[1] if is_block_kv else 0

        if is_block_kv:
            # active_blocks_table: [B, num_blocks] (kv_heads==1) or [B, kv_heads, num_blocks] (kv_heads>1)
            S_max_ctx = active_blocks_table.shape[-1] * blk_len
            S_ctx = S_max_ctx
        elif K_cache_transposed:
            # (B, 1, d_head, S_max_ctx)
            S_max_ctx = K_cache.shape[3]
        else:
            # (B, 1, S_max_ctx, d_head)
            S_max_ctx = K_cache.shape[2]

        if S_ctx is None:
            S_ctx = S_ctx_param

        return B, S_tkg, H, d_head, q_heads, S_ctx, S_max_ctx, blk_len

    def _rms_norm(self, x: torch.Tensor, axis: int, eps: float, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        """RMSNorm: ``x / sqrt(mean(x², axis) + eps) * w``.

        Args:
            x: Input tensor.
            axis: Dimension to normalize over.
            eps: Epsilon for numerical stability.
            w: Optional affine scale (gamma), broadcastable to *x*.
        """
        normalized = x / torch.sqrt(torch.mean(x**2, dim=axis, keepdim=True) + eps)
        if w is not None:
            normalized = normalized * w
        return normalized

    def _qk_norm_pre_rope(
        self,
        QKV: torch.Tensor,
        num_heads: int,
        kv_heads: int,
        d_head: int,
        enabled: bool,
        eps: float,
        W_Q: Optional[torch.Tensor],
        W_K: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply per-head RMSNorm to Q and K before RoPE (in-place).

        Args:
            QKV: Combined QKV tensor. Shape: ``[q_heads+2*kv_heads, B, S, D]``.
            num_heads: Number of query heads (N).
            d_head: Head dimension (D).
            enabled: Whether to apply normalization.
            eps: Epsilon for RMSNorm.
            W_Q: Optional gamma for Q heads. Shape: ``[1, D]``.
            W_K: Optional gamma for K heads. Shape: ``[1, D]``.
            kv_heads: Number of KV heads.

        Returns:
            QKV tensor (modified in-place).
        """
        if not enabled:
            return QKV
        w_Q = W_Q.float().reshape(1, 1, 1, d_head) if W_Q is not None else None
        w_K = W_K.float().reshape(1, 1, 1, d_head) if W_K is not None else None
        QKV[:num_heads] = self._rms_norm(QKV[:num_heads], axis=-1, eps=eps, w=w_Q)
        QKV[num_heads : num_heads + kv_heads] = self._rms_norm(
            QKV[num_heads : num_heads + kv_heads], axis=-1, eps=eps, w=w_K
        )
        return QKV

    def _apply_rope_to_qk(
        self,
        QKV: torch.Tensor,
        cos: Optional[torch.Tensor],
        sin: Optional[torch.Tensor],
        num_heads: int,
        kv_heads: int,
        d_head: int,
        rope_contiguous_layout: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Rotary Position Embedding to Q and K heads.

        Transposes from QKV's ``[q_heads+2*kv_heads, B, S, D]`` layout to the ``[D, B, *, S]``
        layout expected by downstream attention, applying RoPE per-head.

        Args:
            QKV: Combined QKV tensor. Shape: ``[q_heads+2*kv_heads, B, S, D]``.
            cos: Cosine table for RoPE. Shape: ``[D//2, B, S]``. None to skip RoPE.
            sin: Sine table for RoPE. Shape: ``[D//2, B, S]``. None to skip RoPE.
            num_heads: Number of query heads (N).
            d_head: Head dimension (D).
            rope_contiguous_layout: Whether RoPE uses contiguous (True) or
                interleaved (False) layout for the half-dimension split.
            kv_heads: Number of KV heads.

        Returns:
            Tuple of (Q, K):
                - Q: Shape ``[D, B, N, S]``.
                - K: Shape ``[D, B, kv_heads, S]``.
        """
        _, batch, S_tkg, _ = QKV.shape
        skip_rope = cos is None or sin is None
        total_heads = num_heads + kv_heads
        QK = torch.zeros(d_head, batch, total_heads, S_tkg)
        for b in range(batch):
            for h in range(total_heads):
                cur_head = QKV[h, b, :, :].T
                if skip_rope:
                    QK[:, b, h, :] = cur_head
                else:
                    QK[:, b, h, :] = _rope_single_head(cur_head, cos[:, b, :], sin[:, b, :], rope_contiguous_layout)
        return QK[:, :, :num_heads, :], QK[:, :, num_heads:, :]

    def _qk_norm_post_rope(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        d_head: int,
        enabled: bool,
        eps: float,
        W_Q: Optional[torch.Tensor],
        W_K: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply per-head RMSNorm to Q and K after RoPE.

        Args:
            Q: Query tensor. Shape: ``[D, B, N, S]``.
            K: Key tensor. Shape: ``[D, B, S]``.
            d_head: Head dimension (D).
            enabled: Whether to apply normalization.
            eps: Epsilon for RMSNorm.
            W_Q: Optional gamma for Q. Shape: ``[1, D]``.
            W_K: Optional gamma for K. Shape: ``[1, D]``.

        Returns:
            Tuple of (Q, K) with same shapes as inputs.
        """
        if not enabled:
            return Q, K
        w_Q = W_Q.float().reshape(d_head, 1, 1, 1) if W_Q is not None else None
        # K may be [D, B, S] (single KV head) or [D, B, kv_heads, S] (multi KV head).
        w_K = W_K.float().reshape([d_head] + [1] * (K.dim() - 1)) if W_K is not None else None
        return self._rms_norm(Q, axis=0, eps=eps, w=w_Q), self._rms_norm(K, axis=0, eps=eps, w=w_K)

    def _quantize_kv_to_fp8(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize K and V: scale and clamp to [-max, max].

        The clipping bound is derived from ``self.kv_quant_dtype``.

        Args:
            K: Key tensor (any shape, typically ``[D, B, S]``), in float32.
            V: Value tensor (any shape, typically ``[B, S, D]``), in float32.
            k_scale: Scale for K. Only ``k_scale[0, 0]`` is used (broadcast).
            v_scale: Scale for V. Only ``v_scale[0, 0]`` is used (broadcast).

        Returns:
            Tuple of (K_quantized, V_quantized) in float32, clamped to the
            representable range of ``self.kv_quant_dtype``.
        """
        fp8_max = get_max_positive_value_for_dtype(self.kv_quant_dtype)
        kernel_assert(fp8_max is not None, f"Unsupported kv_quant_dtype: {self.kv_quant_dtype}")
        K_scaled = K * k_scale[0, 0].float()
        V_scaled = V * v_scale[0, 0].float()
        k_clip = (K_scaled.abs() > fp8_max).float().mean().item()
        v_clip = (V_scaled.abs() > fp8_max).float().mean().item()
        kernel_assert(k_clip <= 0.01, f"Too many K values clipped ({k_clip:.1%}), scale may be inappropriate")
        kernel_assert(v_clip <= 0.01, f"Too many V values clipped ({v_clip:.1%}), scale may be inappropriate")
        return (
            torch.clamp(K_scaled, -fp8_max, fp8_max),
            torch.clamp(V_scaled, -fp8_max, fp8_max),
        )

    def _run_attention(
        self,
        Q: torch.Tensor,
        K_active: torch.Tensor,
        V_active: torch.Tensor,
        K_cache: torch.Tensor,
        V_cache: torch.Tensor,
        attention_mask: torch.Tensor,
        active_blocks_table: Optional[torch.Tensor],
        K_cache_transposed: bool,
        block_len: int,
        S_ctx: int,
        S_max_ctx: int,
        softmax_scale: Optional[float],
        num_heads: int,
        sink: Optional[torch.Tensor] = None,
        pos_ids: Optional[torch.Tensor] = None,
        swa_start_pos_ids: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        dtype_mode: DtypeMode = DtypeMode.NON_OCP,
        return_cp_softmax_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Run scaled dot-product attention via :func:`attention_tkg_torch_ref`.

        Applies softmax scaling to Q, then delegates to the standalone attention
        TKG reference which handles the KV cache lookup and masked attention.

        Args:
            Q: Query tensor. Shape: ``[D, B, N, S]``.
            K_active: Active key tensor. Shape: ``[D, B, S]``.
            V_active: Active value tensor. Shape: ``[B, S, D]``.
            K_cache: Key cache. Shape: ``[blocks, block_len, D]`` (block KV) or
                ``[B, 1, D, S_max]`` (transposed) or ``[B, 1, S_max, D]``.
            V_cache: Value cache. Shape: ``[blocks, block_len, D]`` (block KV) or
                ``[B, 1, S_max, D]``.
            attention_mask: Attention mask. Shape: ``[S_ctx, B, N, S]`` when
                pos_ids is None, ``[S, B, N, S]`` (active-only) when pos_ids is provided.
            active_blocks_table: Block-to-slot mapping for block KV cache. None for non-block KV.
                Shape: ``[B, S_ctx // block_len]`` when ``kv_heads == 1``;
                ``[B, kv_heads, S_ctx // block_len]`` when ``kv_heads > 1`` (per-head block tables).
            K_cache_transposed: Whether K cache has transposed layout.
            block_len: Block length for block KV cache (0 = non-block).
            S_ctx: Context length visible to attention.
            S_max_ctx: Maximum context length (cache allocation size).
            softmax_scale: Custom softmax scale. None uses ``1/√D``. When using FP8
                KV cache with None, k_scale is automatically absorbed effectively setting
                ``softmax_scale = (1/√D) / k_scale``. When explicitly provided, caller
                must incorporate k_scale: ``softmax_scale = softmax_scale / k_scale``.
            num_heads: Number of query heads (N).
            sink: Optional attention sink tensor. Forwarded to attention ref.
            return_cp_softmax_stats: When True, return unnormalized output with
                softmax stats (max, sum) for CP distributed correction.

        Returns:
            Attention output ``[B, N, D, S]``, or tuple of
            ``(out, softmax_max, softmax_sum)`` when return_cp_softmax_stats=True.
        """
        d_head, batch, _, S_tkg = Q.shape
        q = Q.permute(1, 2, 3, 0)
        k_active_attn = K_active.reshape(d_head, batch, 1, S_tkg).permute(1, 2, 3, 0)
        v_active_attn = V_active.reshape(batch, 1, S_tkg, d_head)

        if softmax_scale is not None:
            q = q * softmax_scale
        else:
            q = q / math.sqrt(d_head)
            # When using FP8 KV cache without explicit softmax_scale,
            # divide by k_scale to dequantize KV values in the QK matmul
            if k_scale is not None:
                q = q / float(k_scale.flat[0] if hasattr(k_scale, 'flat') else k_scale[0, 0])

        is_block_kv = block_len > 0
        use_pos_id = pos_ids is not None
        cfg = AttnTKGConfig(
            bs=batch,
            q_head=num_heads,
            s_active=S_tkg,
            curr_sprior=S_ctx,
            full_sprior=S_max_ctx,
            d_head=d_head,
            block_len=block_len,
            tp_k_prior=not K_cache_transposed,
            strided_mm1=not is_block_kv,
            use_pos_id=use_pos_id,
            fuse_rope=False,
            return_cp_softmax_stats=return_cp_softmax_stats,
        )
        out = torch.zeros(batch, num_heads, d_head, S_tkg, dtype=torch.float32)
        abt = active_blocks_table.to(torch.int32) if active_blocks_table is not None else None

        # When use_pos_id=True, attention_tkg_torch_ref generates the prior mask internally
        # from rope_pos_ids/start_pos_ids. attention_mask contains the active portion.
        mask_arg = attention_mask.to(torch.uint8)

        cp_softmax_stats_out = {} if return_cp_softmax_stats else None
        attention_tkg_torch_ref[self.lnc](
            q=q,
            k_active=k_active_attn,
            v_active=v_active_attn,
            k_prior=K_cache.float(),
            v_prior=V_cache.float(),
            mask=mask_arg,
            out=out,
            cfg=cfg,
            sbm=None,
            rope_pos_ids=pos_ids.float() if pos_ids is not None else None,
            start_pos_ids=swa_start_pos_ids.float() if swa_start_pos_ids is not None else None,
            sink=sink,
            active_blocks_table=abt if is_block_kv else None,
            cp_softmax_stats_out=cp_softmax_stats_out,
            dtype_mode=dtype_mode,
        )
        if return_cp_softmax_stats:
            return out, cp_softmax_stats_out["fa_running_max"], cp_softmax_stats_out["fa_running_sum"]
        return out

    def _update_kv_cache(
        self,
        K_new: torch.Tensor,
        V_new: torch.Tensor,
        K_cache: torch.Tensor,
        V_cache: torch.Tensor,
        kv_cache_update_idx: torch.Tensor,
        update_cache: bool,
        K_cache_transposed: bool,
        block_len: int,
        d_head: int,
        kv_heads: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Write newly projected K/V into the KV cache at the given positions.

        When ``update_cache`` is False, returns the new K/V tensors unchanged
        (they are still needed as outputs for the test harness).

        For block KV caches, ``kv_cache_update_idx`` contains *physical* flat
        indices into the block table, one per token. An index of
        ``INACTIVE_BLOCK_IDX`` means "skip".

        Args:
            K_new: New key tensor from projection. Shape: ``[D, B, S]``.
            V_new: New value tensor from projection. Shape: ``[B, S, D]``.
            K_cache: Key cache to update. Shape: ``[blocks, block_len, D]``
                (block KV) or ``[B, 1, D, S_max]`` / ``[B, 1, S_max, D]``.
            V_cache: Value cache to update. Same shape convention as K_cache.
            kv_cache_update_idx: Per-token write positions.
                Shape: ``[B, S_tkg]`` for block KV, ``[B, 1]`` for flat KV.
            update_cache: Whether to actually write into the cache.
            K_cache_transposed: Whether K cache uses ``[B, 1, D, S_max]`` layout.
            block_len: Block length (0 = non-block KV cache).
            d_head: Head dimension (D).

        Returns:
            Tuple of (K_cache_updated, V_cache_updated).
        """
        if not update_cache:
            return K_new, V_new

        K_cache = K_cache.clone()
        V_cache = V_cache.clone()

        # K_new is always 4D [D, B, kv_heads, S]; V_new is always 4D [B, kv_heads, S, D].
        # kv_cache_update_idx may be 2D [B, S_tkg] (when kv_heads==1) or 3D [B, kv_heads, S_tkg];
        # reshape kv_cache_update_idx to 3D so the loop is uniform.
        batch, S_tkg = K_new.shape[1], K_new.shape[-1]

        if block_len > 0:
            # Block KV: shared [num_blocks, block_len, D] between batch
            num_blocks = K_cache.shape[0]
            K_flat = K_cache.reshape(num_blocks * block_len, d_head)
            V_flat = V_cache.reshape(num_blocks * block_len, d_head)
            idx = kv_cache_update_idx.reshape(batch, kv_heads, S_tkg)
            for b in range(batch):
                for kv_h in range(kv_heads):
                    for s in range(S_tkg):
                        slot = int(idx[b, kv_h, s].item())
                        if slot == INACTIVE_BLOCK_IDX:
                            continue
                        K_flat[slot, :] = K_new[:, b, kv_h, s]
                        V_flat[slot, :] = V_new[b, kv_h, s, :]
            return K_flat.reshape(num_blocks, block_len, d_head), V_flat.reshape(num_blocks, block_len, d_head)

        # Flat KV: cache [B, kv_heads, S_max, D] (or transposed [B, kv_heads, D, S_max]).
        # idx[b, 0] is the per-batch start position (shared across kv heads and tokens).
        S_max = K_cache.shape[3] if K_cache_transposed else K_cache.shape[2]
        for b in range(batch):
            start_pos = int(kv_cache_update_idx[b, 0].item())
            # Out-of-bounds start: the kernel's flat cache write uses oob_mode.skip, so a CP
            # non-owning rank passes an OOB index to drop the write.
            if start_pos + S_tkg > S_max:
                continue
            for kv_h in range(kv_heads):
                if K_cache_transposed:
                    K_cache[b, kv_h, :, start_pos : start_pos + S_tkg] = K_new[:, b, kv_h, :]
                else:
                    K_cache[b, kv_h, start_pos : start_pos + S_tkg, :] = K_new[:, b, kv_h, :].T
                V_cache[b, kv_h, start_pos : start_pos + S_tkg, :] = V_new[b, kv_h, :, :]
        return K_cache, V_cache

    def _kvdp_input_collectives(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        K_cache: torch.Tensor,
        V_cache: torch.Tensor,
        q_heads: int,
        kv_heads: int,
        KVDP: int,
        B: int,
        pg,
        KVDP_rank: int,
    ):
        """KVDP input collectives: Q all_to_all, K/V batch slice.

        Before: each rank has Q [D, B, q_heads, S], K [D, B, kv_heads, S], V [B, kv_heads, S, D]
        After:  each rank has Q [D, B_attn, q_heads*KVDP, S], K [D, B_attn, kv_heads, S], V [B_attn, kv_heads, S, D]

        The Q all_to_all exchanges batch chunks for head chunks across ranks:
        each rank sends its B/KVDP batch slice to each other rank and receives
        q_heads from each other rank.

        KV heads are replicated across ranks, so K/V is batch-sliced.
        """
        B_attn = B // KVDP

        # Q all_to_all: [D, B, q_heads, S] -> split batch -> exchange -> cat heads -> [D, B_attn, q_heads*KVDP, S]
        # Split Q along batch dim into KVDP chunks, each [D, B_attn, q_heads, S]
        Q_send = list(Q.chunk(KVDP, dim=1))
        Q_recv = [torch.empty_like(Q_send[0]) for _ in range(KVDP)]
        pg.alltoall(Q_recv, Q_send)
        # Each received chunk has q_heads from a different rank -> cat along head dim
        Q = torch.cat(Q_recv, dim=2)  # [D, B_attn, q_heads*KVDP, S]

        # K batch slice: [D, B, kv_heads, S] -> [D, B_attn, kv_heads, S]
        K = K[:, KVDP_rank * B_attn : (KVDP_rank + 1) * B_attn, :, :]

        # V batch slice: [B, kv_heads, S, D] -> [B_attn, kv_heads, S, D]
        V = V[KVDP_rank * B_attn : (KVDP_rank + 1) * B_attn, :, :, :]

        # K_cache/V_cache are already per-rank (B_attn) from the input generator
        return Q, K, V, K_cache, V_cache

    def _kvdp_output_collectives(
        self,
        attn_out: torch.Tensor,
        KVDP: int,
        pg,
        collective_mode=None,
        kvdp_rank: int = 0,
    ) -> torch.Tensor:
        """KVDP output collectives: attention output redistribution.

        Before: each rank has attn_out [B_attn, q_heads*KVDP, D, S]
        After:  each rank has attn_out [B, q_heads, D, S]

        ALL_TO_ALL: exchanges head chunks for batch chunks.
        ALL_GATHER_SLICE: gathers all ranks' outputs, slices own heads by KVDP_rank.
        """
        from .attention_block_tkg_sharding import KVDPCollectiveMode

        if collective_mode == KVDPCollectiveMode.ALL_GATHER_SLICE:
            # all_gather on batch dim: [B_attn, q_attn, D, S] -> [B, q_attn, D, S]
            chunks = [torch.empty_like(attn_out) for _ in range(KVDP)]
            pg.allgather([chunks], [attn_out])
            gathered = torch.cat(chunks, dim=0)  # [B, q_attn, D, S]
            # Slice heads by KVDP_rank: q_attn = q_heads * KVDP, take q_heads starting at kvdp_rank*q_heads
            q_heads = attn_out.shape[1] // KVDP
            return gathered[:, kvdp_rank * q_heads : (kvdp_rank + 1) * q_heads, :, :]
        else:
            # ALL_TO_ALL: split heads into KVDP chunks, exchange for batch chunks
            attn_send = list(attn_out.chunk(KVDP, dim=1))
            attn_recv = [torch.empty_like(attn_send[0]) for _ in range(KVDP)]
            pg.alltoall(attn_recv, attn_send)
            # Each received chunk has B_attn batches from a different rank -> cat along batch dim
            return torch.cat(attn_recv, dim=0)  # [B, q_heads, D, S]

    def _cp_input_collectives(
        self,
        Q: torch.Tensor,
        CP: int,
        pg,
    ) -> torch.Tensor:
        """CP input collectives: all-gather Q heads across CP ranks.

        Before: Q [D, B, q_heads, s_active] — each rank has its own q_heads
        After:  Q [D, B, q_heads*CP, s_active] — each rank has all heads, attends to its s_prior/CP KV cache shard
        """
        # Q shape: [D, B, q_heads, S]
        # all_gather on head dim (dim=2)
        Q_chunks = [torch.empty_like(Q) for _ in range(CP)]  # CP x [D, B, q_heads, s_active]
        pg.allgather([Q_chunks], [Q])
        return torch.cat(Q_chunks, dim=2)  # [D, B, q_heads*CP, S]

    def _cp_output_collectives(
        self,
        attn_out_unnorm: torch.Tensor,
        softmax_max: torch.Tensor,
        softmax_sum: torch.Tensor,
        q_heads_per_cp_rank: int,
        CP: int,
        pg,
    ) -> torch.Tensor:
        """CP output collectives: softmax correction across CP ranks + all-to-all.

        Each rank has unnormalized attention output for all q_heads_per_cp_rank*CP heads
        computed against its s_prior/CP KV cache shard.

        Steps:
        1. All-gather softmax stats (max, sum) from all CP ranks
        2. Compute global max across all ranks
        3. Compute correction per rank: exp(rank_max - global_max)
        4. Compute global sum: sum of (correction_i * sum_i) across ranks
        5. Scale local output: attn_out_unnorm * correction / global_sum
        6. All-to-all: split q_heads_per_cp_rank*CP Q heads into CP chunks, send chunk i to rank i
        7. Sum the scaled outputs received from each rank

        Each rank computed attention for all heads but only against its KV shard,
        so steps 6-7 sum across shards to produce the final output per head.

        Args:
            attn_out_unnorm: [B, q_heads_per_cp_rank*CP, D, s_active] — unnormalized (not divided by softmax_sum)
            softmax_max: [B, q_heads_per_cp_rank*CP, s_active] — local softmax max
            softmax_sum: [B, q_heads_per_cp_rank*CP, s_active] — local softmax sum
            q_heads_per_cp_rank: number of Q heads per CP rank
            CP: context parallelism degree
        Returns:
            [B, q_heads_per_cp_rank, D, s_active] — normalized attention output
        """
        q_heads_attn = q_heads_per_cp_rank * CP

        # 1. All-gather softmax stats across CP ranks
        # softmax_max: [B, q_heads_attn, S] per rank
        all_max = [torch.empty_like(softmax_max) for _ in range(CP)]  # CP x [B, q_heads_attn, S]
        pg.allgather([all_max], [softmax_max])
        all_max = torch.stack(all_max, dim=0)  # [CP, B, q_heads_attn, S]

        all_sum = [torch.empty_like(softmax_sum) for _ in range(CP)]  # CP x [B, q_heads_attn, S]
        pg.allgather([all_sum], [softmax_sum])
        all_sum = torch.stack(all_sum, dim=0)  # [CP, B, q_heads_attn, S]

        # 2. Compute global max
        global_max = all_max.max(dim=0).values  # [B, q_heads_attn, S]

        # 3. Correction per rank: exp(rank_max - global_max)
        # correction[i] = exp(max_i - global_max)
        corrections = torch.exp(all_max - global_max.unsqueeze(0))  # [CP, B, q_heads_attn, S]

        # 4. Global sum: sum(correction_i * sum_i) across ranks
        corrected_sums = corrections * all_sum  # [CP, B, q_heads_attn, S]
        global_sum = corrected_sums.sum(dim=0)  # [B, q_heads_attn, S]

        # 5. Scale: attn_out_unnorm * (correction / global_sum)
        local_correction = torch.exp(softmax_max - global_max)  # [B, q_heads_attn, S]
        local_scale = local_correction / global_sum  # [B, q_heads_attn, S]
        # attn_out_unnorm: [B, q_heads_attn, D, S], scale: [B, q_heads_attn, S]
        attn_scaled = attn_out_unnorm * local_scale.unsqueeze(2)  # [B, q_heads_attn, D, S]

        # 6. All-to-all: split q_heads_per_cp_rank*CP Q heads into CP chunks, send chunk i to rank i
        attn_send = list(attn_scaled.chunk(CP, dim=1))  # CP x [B, q_heads_per_cp_rank, D, S]
        attn_recv = [torch.empty_like(attn_send[0]) for _ in range(CP)]  # CP x [B, q_heads_per_cp_rank, D, S]
        pg.alltoall(attn_recv, attn_send)

        # 7. Sum scaled outputs received from each rank
        attn_stacked = torch.stack(attn_recv, dim=0)  # [CP, B, q_heads_per_cp_rank, D, S]
        attn_out = attn_stacked.sum(dim=0)  # [B, q_heads_per_cp_rank, D, S]
        return attn_out


# Dispatch-compatible top-level function matching the attention_block_tkg kernel signature.
# Uses default lnc=2 which matches the most common deployment config.
# The test file instantiates AttentionBlockTkgTorchRef directly when it needs a specific lnc.
#
# Note: AttentionBlockTkgTorchRef.forward() accepts extra test-only params (X_in_sb)
# for UnitTestFramework signature compatibility. This wrapper exposes only the params
# present in the actual attention_block_tkg kernel for dispatch audit compliance.


def attention_block_tkg_torch_ref(**kwargs):
    """Dispatch-compatible torch reference for attention_block_tkg.

    Infers kv_quant_dtype from K_cache.dtype and dtype_mode, matching the
    kernel's resolution logic:
    - k_scale/v_scale provided + K_cache has float8 dtype -> use K_cache.dtype
    - k_scale/v_scale provided + K_cache is float32 (CPU test) -> resolve from dtype_mode
    - No k_scale/v_scale -> kv_quant_dtype=None (no KV quantization)

    Uses lnc=2 (standard LNC2 deployment config).
    """
    from ...core.utils.kernel_helpers import resolve_fp8_e4m3_dtype

    K_cache = kwargs.get("K_cache")
    k_scale = kwargs.get("k_scale")
    dtype_mode = kwargs.get("dtype_mode", DtypeMode.NON_OCP)

    if k_scale is not None and K_cache is not None:
        cache_dtype_str = str(K_cache.dtype)
        if "float8" in cache_dtype_str:
            kv_quant_dtype = cache_dtype_str
        else:
            # CPU/test path: K_cache is float32, resolve FP8 type from dtype_mode
            kv_quant_dtype = str(resolve_fp8_e4m3_dtype(dtype_mode))
    else:
        kv_quant_dtype = str(nl.float8_e4m3)

    ref = AttentionBlockTkgTorchRef(lnc=2, kv_quant_dtype=kv_quant_dtype)
    return ref(**kwargs)


# Set __signature__ to match the kernel (forward minus X_in_sb)
_fwd_sig = inspect.signature(AttentionBlockTkgTorchRef(lnc=2).forward)
_dispatch_params = [p for name, p in _fwd_sig.parameters.items() if name != "X_in_sb"]
attention_block_tkg_torch_ref.__signature__ = _fwd_sig.replace(parameters=_dispatch_params)
