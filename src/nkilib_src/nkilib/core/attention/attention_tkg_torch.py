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
PyTorch reference for attention_tkg kernel
"""

import math
from typing import Optional, Tuple

import torch

from ..utils.allocator import SbufManager
from ..utils.kernel_assert import kernel_assert
from .attention_tkg_utils import (
    AttnTKGConfig,
    is_batch_sharded,
    resize_cache_block_len_for_attention_tkg_kernel,
)


def attention_tkg_torch_ref(
    q: torch.Tensor,
    k_active: torch.Tensor,
    v_active: torch.Tensor,
    k_prior: torch.Tensor,
    v_prior: torch.Tensor,
    mask: torch.Tensor,
    out: torch.Tensor,
    cfg: AttnTKGConfig,
    sbm: SbufManager,
    inv_freqs: Optional[torch.Tensor] = None,
    rope_pos_ids: Optional[torch.Tensor] = None,
    sink: Optional[torch.Tensor] = None,
    active_blocks_table: Optional[torch.Tensor] = None,
    k_out: Optional[torch.Tensor] = None,
    DBG_TENSORS: Optional[tuple] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    PyTorch stub / reference for NKI kernel attention.attention_tkg.attention_tkg
    with identical interface to the kernel.
    """

    # Currently hardcoding LNC to 2 and P_MAX to 128, the kernel determines LNC with nl.num_programs
    # and P_MAX through nl.tile_size.pmax which isn't available in torch golden.
    # TODO: Need to find a better way to dynamically configure this. Confirm whether
    # we should have an ExtraParams structure/parameter shared by torch refs that
    # wraps this type of info
    LNC = 2
    P_MAX = 128

    batch = cfg.bs
    q_head = cfg.q_head
    d_head = cfg.d_head
    s_prior = cfg.curr_sprior
    s_prior_full = cfg.full_sprior
    s_active = cfg.s_active
    block_len = cfg.block_len
    is_block_kv = active_blocks_table is not None
    out_in_sb = cfg.out_in_sb
    attn_out_shape = (d_head, batch * q_head * s_active) if out_in_sb else (batch, q_head, d_head, s_active)
    k_out_shape = (d_head, batch * s_active) if out_in_sb else (batch, 1, d_head, s_active)

    if DBG_TENSORS:
        expected_len = 4 + (1 if is_block_kv else 0)
        kernel_assert(
            len(DBG_TENSORS) == expected_len,
            f"Received {len(DBG_TENSORS)} debug tensors, when {expected_len} are expected.",
        )

    # Convert to float32 because CPU doesn't support half precision
    q = q.to(torch.float32)
    k_active = k_active.to(torch.float32)
    v_active = v_active.to(torch.float32)
    k_prior = k_prior.to(torch.float32)
    v_prior = v_prior.to(torch.float32)
    mask = mask.to(torch.uint8)

    active_mask = mask
    if not cfg.use_pos_id:  # FIXME: Waiting for FE to remove the requirement for this reshape
        active_mask = active_mask.reshape(cfg.curr_sprior, cfg.bs, cfg.q_head, cfg.s_active)

    reduced_blk_len = None
    DBG_ACTIVE_TABLE = None
    if is_block_kv:
        kernel_assert(s_prior % block_len == 0, "s_prior must be divisible by block_len")
        kernel_assert(active_blocks_table is not None, "active_blocks_table must be provided for block KV")
        k_prior = _gather_block_kv_to_flat(k_prior, active_blocks_table, batch, s_prior_full, d_head, block_len)
        v_prior = _gather_block_kv_to_flat(v_prior, active_blocks_table, batch, s_prior_full, d_head, block_len)
        k_prior = k_prior.unsqueeze(1)
        v_prior = v_prior.unsqueeze(1)

        reduced_blk_len, resize_factor = resize_cache_block_len_for_attention_tkg_kernel(
            s_prior // block_len, block_len, LNC, P_MAX
        )

        # Only reshape mask if NOT using use_pos_id (i.e., pre-computed full cache mask)
        # When use_pos_id=True, the mask is a small active mask that will be expanded in _attention_tkg_fwd_ref
        if not cfg.use_pos_id:
            active_mask = (
                active_mask.permute(1, 2, 3, 0)
                .reshape((-1, reduced_blk_len, P_MAX))
                .swapaxes(-1, -2)
                .reshape((batch, q_head, s_active, s_prior))
                .permute(3, 0, 1, 2)
            )

        DBG_ACTIVE_TABLE = _get_dbg_active_table(active_blocks_table, resize_factor, batch, P_MAX)

    active_mask = active_mask.permute(1, 2, 0, 3)  # [b, n, s_prior, s_active]
    q, k_active = _reshape_q_and_k_active(q, k_active, cfg)
    k_prior, v_prior = _slice_and_reshape_kv_prior(k_prior, v_prior, cfg)

    inv_freqs = inv_freqs.to(torch.float32) if inv_freqs is not None else None
    rope_pos_ids = rope_pos_ids.to(torch.float32) if rope_pos_ids is not None else None
    sink = sink.to(torch.float32) if sink is not None else None

    attn_out, attn_k_out, DBG_QK, DBG_QK_MAX, DBG_QK_EXP, DBG_EXP_SUM = _attention_tkg_fwd_ref(
        q=q,
        k_active=k_active,
        v_active=v_active,
        k_prior=k_prior,
        v_prior=v_prior,
        active_mask=active_mask,
        inv_freqs=inv_freqs,
        rope_pos_ids=rope_pos_ids,
        sink=sink,
        cfg=cfg,
    )

    # Need to transpose and reshape if output is in sbuf
    if out_in_sb:
        attn_out = attn_out.permute(2, 0, 1, 3).reshape(attn_out_shape)
        attn_k_out = attn_k_out.permute(2, 0, 1, 3).reshape(k_out_shape)

    kernel_assert(
        out.shape == attn_out.shape, f"Output shape mismatch: out.shape={out.shape}, attn_out.shape={attn_out.shape}"
    )
    out.copy_(attn_out)

    if k_out is not None:
        kernel_assert(
            k_out.shape == attn_k_out.shape,
            f"Output shape mismatch: k_out.shape={k_out.shape}, attn_k_out.shape={attn_k_out.shape}",
        )
        k_out.copy_(attn_k_out)

    if DBG_TENSORS:
        DBG_QK = _reshape_debug_tensor(DBG_QK, cfg, P_MAX, LNC, is_block_kv, reduced_blk_len)
        DBG_QK_EXP = _reshape_debug_tensor(DBG_QK_EXP, cfg, P_MAX, LNC, is_block_kv, reduced_blk_len)

        DBG_RESULTS = [
            DBG_QK,
            DBG_QK_MAX,
            DBG_QK_EXP,
            DBG_EXP_SUM,
        ]
        if is_block_kv:
            DBG_RESULTS.append(DBG_ACTIVE_TABLE)

        for dst_tensor, dbg_result in zip(DBG_TENSORS, DBG_RESULTS):
            dst_tensor.view(dbg_result.shape).copy_(dbg_result)

    return out, k_out


def _reshape_q_and_k_active(q, k_active, cfg: AttnTKGConfig):
    # attention_tkg supports qk tensors starting out in sbuf instead of hbm
    # in this case, we need to transpose them into dimension config that torch expects
    d_head, batch, q_head, s_active = cfg.d_head, cfg.bs, cfg.q_head, cfg.s_active

    if cfg.qk_in_sb:
        # Need to reshape first because these tensors will be flattened if in sb
        q = q.reshape(d_head, batch, q_head, s_active).permute((1, 2, 0, 3))
        k_active = k_active.reshape(d_head, 1, batch, s_active).permute((2, 1, 0, 3))
    else:
        q = q.permute(0, 1, 3, 2)
        k_active = k_active.permute(0, 1, 3, 2)

    return q, k_active


def _slice_and_reshape_kv_prior(k_prior, v_prior, cfg: AttnTKGConfig):
    batch, s_prior = cfg.bs, cfg.curr_sprior

    k_prior = k_prior[:batch, ...]  # when has_cache_buffer, there is an extra buffer batch in the end
    v_prior = v_prior[:batch, ...]  # when has_cache_buffer, there is an extra buffer batch in the end

    if cfg.tp_k_prior:
        k_prior = k_prior[..., :s_prior, :].permute(0, 1, 3, 2)
    else:
        k_prior = k_prior[..., :s_prior]
    v_prior = v_prior[..., :s_prior, :]

    return k_prior, v_prior


def _gather_block_kv_to_flat(block_cache, active_blocks_table, batch, S_max_ctx, d_head, block_len):
    '''Gather block cache to flat layout'''
    flat_cache = torch.zeros((batch, S_max_ctx, d_head), dtype=block_cache.dtype)
    for b in range(batch):
        flat_cache[b][: active_blocks_table.shape[1] * block_len, :] = block_cache[active_blocks_table[b]].reshape(
            (-1, d_head)
        )
    return flat_cache


# Intermediate verification to verify active blocks reshape and load to SBUF.
def _get_dbg_active_table(active_blocks_table, resize_factor, batch, p_max):
    '''Get debug active table'''
    table_repeated = torch.repeat_interleave(active_blocks_table, resize_factor, axis=1) * resize_factor
    increment_pattern = torch.arange(resize_factor).repeat(active_blocks_table.shape[1])
    table_incremented = table_repeated + increment_pattern[None, :]
    dbg_active_table = table_incremented.reshape((batch, table_incremented.shape[1] // p_max, p_max)).permute(2, 1, 0)

    return dbg_active_table


def _gen_rope_coeff(inv_freq, pos_ids):
    d_head_half, _ = inv_freq.shape
    batch, seqlen = pos_ids.shape

    inv_freq_expanded = inv_freq[:, None, :].expand(d_head_half, batch, seqlen)
    pos_ids_expanded = pos_ids[None, :, :].expand(d_head_half, batch, seqlen)

    freqs = inv_freq_expanded * pos_ids_expanded
    emb = torch.cat([freqs, freqs], axis=0)

    freqs_cos = torch.cos(emb)
    freqs_sin = torch.sin(emb)

    return freqs_cos, freqs_sin


def _apply_rope(x, cos, sin, d_head):
    def _rotate_half(x):
        x1 = x[: d_head // 2, ...]
        x2 = x[d_head // 2 :, ...]
        return torch.cat((-x2, x1), axis=0)

    # Reshape x from BNdS to dBNS
    x = x.permute(2, 0, 1, 3)
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)

    embed = (x * cos) + (_rotate_half(x) * sin)
    embed = embed.permute(1, 2, 0, 3)

    return embed


def _attention_tkg_fwd_ref(
    q,
    k_active,
    v_active,
    k_prior,
    v_prior,
    active_mask,
    inv_freqs,
    rope_pos_ids,
    sink,
    cfg: AttnTKGConfig,
):
    batch, q_head, s_active, s_prior, d_head = cfg.bs, cfg.q_head, cfg.s_active, cfg.curr_sprior, cfg.d_head

    # clone to avoid inplace modification
    k_prior = k_prior.clone()
    v_prior = v_prior.clone()

    if cfg.fuse_rope:
        freq_cos, freq_sin = _gen_rope_coeff(inv_freqs, rope_pos_ids)
        q = _apply_rope(q, freq_cos, freq_sin, d_head)
        q = q / math.sqrt(d_head)
        k_active = _apply_rope(k_active, freq_cos, freq_sin, d_head)

    # If use_pos_id, generate the prior mask with pos_id, then overwrite the last portion with active_mask
    if cfg.use_pos_id:
        mask = torch.zeros((batch, q_head, s_prior, s_active), dtype=active_mask.dtype)
        for b in range(batch):
            pos_id = int(rope_pos_ids[b, 0])
            mask[b, :, :pos_id, :] = 1
            mask[..., -s_active:, :] = active_mask  # overwrite active mask onto the last portion of prior mask
    else:
        mask = active_mask

    # Overwrite active onto last portion of prior
    k_prior[..., -s_active:] = k_active
    v_prior[..., -s_active:, :] = v_active

    # K @ Q^T (note that k_prior and q here are both already transposed)
    score = k_prior.permute(0, 1, 3, 2) @ q  # [b, n, s_prior, s_active]

    score[mask == 0] = -torch.inf

    DBG_QK = torch.clone(score)

    # Pad sink to the end of s_prior dimension.
    if sink is not None:
        sink = sink.reshape((q_head, 1))[None, :, :, None].expand(batch, q_head, 1, s_active)
        score = torch.cat([score, sink], axis=2)

    # Take column wise max
    score_max = torch.max(score, dim=2, keepdim=True).values  # [b, n, 1, s_active]

    DBG_QK_MAX = -score_max  # The debug tensor is always negated

    # Subtract max from score, exp activate
    score -= score_max  # [b, n, s_prior, s_active]
    score = torch.exp(score)  # [b, n, s_prior, s_active]

    DBG_QK_EXP = score[:, :, :-1, :] if sink is not None else score  # drop the sink

    # Take column wise sum
    score_sum = torch.sum(score, dim=2, keepdim=True)  # [b, n, 1, s_active]

    DBG_EXP_SUM = score_sum

    # Divide score by exp sum
    score = score / score_sum  # [b, n, s_prior, s_active]

    # Remove sink from the end of s_prior dimension.
    if sink is not None:
        score = score[:, :, :-1, :]

    # Softmax @ V, transpose out
    out = score.permute(0, 1, 3, 2) @ v_prior  # [b, n, s_active, d_head]
    out = out.permute(0, 1, 3, 2)  # [b, n, d_head, s_active]

    return out, k_active, DBG_QK, DBG_QK_MAX, DBG_QK_EXP, DBG_EXP_SUM


# is_block_kv or bs_n_prgs == 1 : [b, n, s_prior, s_active] -> [P_MAX, lnc, sprior_tiles, [reduced_blk_len,] b, n, s_active]
# bs_n_prgs > 1: [b, n, s_prior, s_active] -> [s_prior, b, n, s_active]
def _reshape_debug_tensor(
    tensor: torch.Tensor,
    cfg: AttnTKGConfig,
    p_max: int,
    lnc: int,
    is_block_kv: bool,
    reduced_blk_len: int = None,
):
    batch, q_head, s_prior, s_active = cfg.bs, cfg.q_head, cfg.curr_sprior, cfg.s_active
    batch_sharded = is_batch_sharded(cfg, p_max)
    if is_block_kv:
        kernel_assert(reduced_blk_len is not None, "reduced_blk_len must be provided for block KV")
        return tensor.reshape(
            (batch, q_head, lnc, s_prior // lnc // p_max // reduced_blk_len, p_max, reduced_blk_len, s_active)
        ).permute(4, 2, 3, 5, 0, 1, 6)
    elif batch_sharded:
        if cfg.strided_mm1:
            return tensor.reshape((batch, q_head, s_prior, s_active)).permute(2, 0, 1, 3)
        else:
            return tensor.reshape((batch, q_head, s_prior // p_max, p_max, s_active)).permute(3, 2, 0, 1, 4)
    elif cfg.strided_mm1:
        return tensor.reshape((batch, q_head, lnc, p_max, s_prior // lnc // p_max, s_active)).permute(3, 2, 4, 0, 1, 5)
    else:
        return tensor.reshape((batch, q_head, lnc, s_prior // lnc // p_max, p_max, s_active)).permute(4, 2, 3, 0, 1, 5)
