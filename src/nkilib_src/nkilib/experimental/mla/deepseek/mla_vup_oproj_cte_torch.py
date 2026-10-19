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

"""PyTorch reference for sparse_mla_vupmx_oproj_cte (KERNEL B — MX V-up + MX o_proj).

Consumes the latent attention output out_attn[B, S, H*L] (from kernel A), applies the
per-head MX V-up then the MX o_proj over H*d_v. Replays the kernel's MX pipeline so
numerics match hardware.
"""

from typing import Dict

import numpy as np
import torch

from ....core.utils.mx_torch_common import mx_matmul, quantize_to_mx, unpack_float8_e4m3fn_x4
from .mla_common_cte import MlaPrecision

_Q_WIDTH = 4
_DS_SCALE_BLOCK = 128


def _as_bf16_fp32(t) -> torch.Tensor:
    """Round a tensor/array through bf16 and return it as fp32.

    The bf16 kernel feeds bf16 operands to a tensor engine that accumulates in fp32, so the
    reference rounds every matmul input to bf16 and does the product in fp32.
    """
    if not isinstance(t, torch.Tensor):
        t = torch.from_numpy(np.asarray(t).astype(np.float32))
    return t.to(torch.bfloat16).to(torch.float32)


def _mla_vup_oproj_cte_torch_ref_bf16(
    out_attn_hbm,  # [B, S, H*L] bf16 latent attention output, NATURAL latent column order
    wuv_hbm,  # [H*L, d_v] bf16 V-up weight
    wo_hbm,  # [H*d_v, HID] bf16 o_proj weight
    kv_lora_rank: int,
) -> Dict[str, torch.Tensor]:
    """PyTorch reference for the BF16 (``MlaPrecision.BF16``) V-up + o_proj path.

    Two plain matmuls. Unlike the MX reference there is no 4-pack de-permutation of the latent
    (bf16 kernel A writes natural order), no activation quantize and no weight dequant — only
    bf16 rounding at each matmul boundary.

    Returns a dict with ``out``: [B, S, HID] bf16.
    """
    attn = _as_bf16_fp32(out_attn_hbm)
    B, S, HL = attn.shape
    L = kv_lora_rank
    H = HL // L
    wuv = _as_bf16_fp32(wuv_hbm)
    d_v = wuv.shape[1]
    wo = _as_bf16_fp32(wo_hbm)
    HID = wo.shape[1]
    Hdv = H * d_v

    # ---- V-up per head: attn[b, :, h, :] @ W_uv[h] ----
    attn = attn.reshape(B, S, H, L)
    attn_v = torch.zeros((B, S, H, d_v), dtype=torch.float32)
    for h in range(H):
        attn_v[:, :, h, :] = attn[:, :, h, :] @ wuv[h * L : (h + 1) * L, :]

    # ---- o_proj over H*d_v ----
    attn_v = _as_bf16_fp32(attn_v).reshape(B * S, Hdv)
    out = (attn_v @ wo).reshape(B, S, HID)
    return {"out": out.to(torch.bfloat16)}


def _broadcast_compact_scales(compact_scale, in_dim, out_dim, compact_scales=True):
    """Return the MX matmul scale layout [in//32, out] uint8.

    compact_scales=True: input is DeepSeek compact block-128 [in//128, ceil(out/128)];
    broadcast to [in//32, out]. compact_scales=False: input is ALREADY native block-32
    [in//32, out] (offline lossless broadcast); pass through unchanged.
    """
    compact_np = compact_scale.cpu().numpy() if isinstance(compact_scale, torch.Tensor) else compact_scale
    if not compact_scales:
        return compact_np[: in_dim // 32, :out_dim].astype(np.uint8)
    full = np.repeat(compact_np, 4, axis=0)
    full = np.repeat(full, _DS_SCALE_BLOCK, axis=1)
    return full[: in_dim // 32, :out_dim].astype(np.uint8)


def _mx_quantize_activation(a_flat, in_dim, free):
    """MX-block-quantize a [free, in_dim] activation into the golden's stationary form."""
    a_np = a_flat.T.numpy()  # [in_dim, free]
    a_np = (
        a_np.reshape(in_dim // _Q_WIDTH, _Q_WIDTH, free)
        .transpose(0, 2, 1)
        .reshape(in_dim // _Q_WIDTH, _Q_WIDTH * free)
        .astype(np.float32)
    )
    a_mx, a_scale = quantize_to_mx(a_np, _fp8x4())
    return unpack_float8_e4m3fn_x4(a_mx), torch.from_numpy(a_scale.astype(np.float64))


def mla_vupmx_oproj_cte_torch_ref(
    out_attn_hbm: torch.Tensor,  # [B, S, H*L] bf16 latent attention output (from kernel A)
    wuv_qtz_hbm: np.ndarray,  # [H*L // 4, d_v] fp8x4 packed MX V-up weight
    wuv_scale_hbm: np.ndarray,  # [H*L // 128, ceil(d_v/128)] uint8 compact scales
    wo_qtz_hbm: np.ndarray,  # [H*d_v // 4, HID] fp8x4 packed MX o_proj weight
    wo_scale_hbm: np.ndarray,  # [H*d_v // 128, ceil(HID/128)] uint8 compact scales
    compact_scales: bool = True,
    precision: MlaPrecision = MlaPrecision.MX,
    kv_lora_rank: int = 512,
) -> Dict[str, torch.Tensor]:
    """Reference for V-up + o_proj, MX or BF16. Returns Dict with "out": [B, S, HID] bf16.

    With ``precision=MlaPrecision.BF16`` this delegates to
    :func:`_mla_vup_oproj_cte_torch_ref_bf16`; the weights are then bf16 ``[K, N]``, both scale
    args are ignored, and the latent is in natural (un-4-packed) column order.
    """
    if precision.is_bf16():
        return _mla_vup_oproj_cte_torch_ref_bf16(
            out_attn_hbm=out_attn_hbm,
            wuv_hbm=wuv_qtz_hbm,
            wo_hbm=wo_qtz_hbm,
            kv_lora_rank=kv_lora_rank,
        )

    B, S, HL = out_attn_hbm.shape
    L = 512  # fixed (one MX 512-tile per head); H recovered from H*L
    H = HL // L
    d_v = wuv_qtz_hbm.shape[1]
    Hdv = H * d_v
    HID = wo_qtz_hbm.shape[1]

    """
    out_attn carries the cross-kernel MX 4-pack column order written by kernel A's MM2:
    within each head's L block, physical column c = sub*(L//4) + group holds natural
    latent l = 4*group + sub. The kernel's V0 dma_transpose reads it back into natural
    4-pack-on-partition order; replay that here by de-permuting each head's L block to
    the natural latent order [.., l] that the V-up contracts. (inverse of attn kernel:
    natural[4*group+sub] = raw[sub*(L//4)+group].)
    """
    _H_PACK = 4
    attn = out_attn_hbm.to(torch.float32).reshape(B, S, H, _H_PACK, L // _H_PACK)
    attn = attn.permute(0, 1, 2, 4, 3).reshape(B, S, H, L)

    # ---- V-up per head: replay kernel's MX pipeline (quantize latent + dequant W_uv). ----
    attn_bf = attn.to(torch.bfloat16).to(torch.float32)
    wuv_unpacked = unpack_float8_e4m3fn_x4(wuv_qtz_hbm)  # [H*L//4, d_v*4]
    wuv_scale_full = _broadcast_compact_scales(
        wuv_scale_hbm, H * L, d_v, compact_scales=compact_scales
    )  # [H*L//32, d_v]
    wuv_scale_t = torch.from_numpy(wuv_scale_full.astype(np.float64))

    attn_v = torch.zeros((B, S, H, d_v), dtype=torch.float32)
    for b in range(B):
        for h in range(H):
            w_h = wuv_unpacked[h * (L // _Q_WIDTH) : (h + 1) * (L // _Q_WIDTH), :]
            w_scale_h = wuv_scale_t[h * (L // 32) : (h + 1) * (L // 32), :]
            a_mx_t, a_scale_t = _mx_quantize_activation(attn_bf[b, :, h, :], L, S)
            attn_v[b, :, h, :] = mx_matmul(
                stationary=a_mx_t, moving=w_h, stationary_scale=a_scale_t, moving_scale=w_scale_h
            )
    attn_v = attn_v.to(torch.bfloat16).to(torch.float32)

    # ---- MX o_proj over H*d_v. ----
    w_unpacked = unpack_float8_e4m3fn_x4(wo_qtz_hbm)
    wo_scale_full = _broadcast_compact_scales(wo_scale_hbm, Hdv, HID, compact_scales=compact_scales)
    w_scale_t = torch.from_numpy(wo_scale_full.astype(np.float64))

    out = torch.zeros((B, S, HID), dtype=torch.float32)
    for b in range(B):
        a_flat = attn_v[b].reshape(S, Hdv)
        a_mx_t, a_scale_t = _mx_quantize_activation(a_flat, Hdv, S)
        res = mx_matmul(stationary=a_mx_t, moving=w_unpacked, stationary_scale=a_scale_t, moving_scale=w_scale_t)
        out[b] = res.reshape(S, HID)

    return {"out": out.to(torch.bfloat16)}


def _fp8x4():
    import nki.language as nl

    return nl.float8_e4m3fn_x4
