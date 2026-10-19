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

"""PyTorch oracle for Kimi-K3 dense gated MLA decode."""

import numpy as np
import torch


def _as_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "dtype") and str(value.dtype) == "bfloat16":
        return torch.from_numpy(np.asarray(value, dtype=np.float32)).to(torch.bfloat16)
    return torch.as_tensor(value)


def kimi_k3_mla_decode_torch_ref(
    q_nope_absorbed_T: torch.Tensor,
    q_pe_T: torch.Tensor,
    ckv_prior_T: torch.Tensor,
    kpe_prior_T: torch.Tensor,
    ckv_cur_T: torch.Tensor,
    kpe_cur_T: torch.Tensor,
    positions: torch.Tensor,
    out_absorb_w: torch.Tensor,
    gate: torch.Tensor,
    o_proj_w: torch.Tensor,
    softmax_scale: float,
    key_pos: torch.Tensor | None = None,
    sparse_add: torch.Tensor | None = None,
):
    """Return the BF16 per-rank Kimi MLA decode output ``{"output": [B, H]}``."""
    if key_pos is not None:
        raise ValueError("Kimi-K3 MLA decode requires key_pos=None")
    if sparse_add is not None:
        raise ValueError("Kimi-K3 MLA decode requires sparse_add=None")

    q_nope = _as_tensor(q_nope_absorbed_T).permute(1, 2, 0).contiguous()
    q_pe = _as_tensor(q_pe_T).permute(1, 2, 0).contiguous()
    ckv_prior = _as_tensor(ckv_prior_T).permute(1, 2, 0).contiguous()
    kpe_prior = _as_tensor(kpe_prior_T).permute(1, 2, 0).contiguous()
    ckv_cur = _as_tensor(ckv_cur_T).permute(1, 2, 0).squeeze(1).contiguous()
    kpe_cur = _as_tensor(kpe_cur_T).permute(1, 2, 0).squeeze(1).contiguous()
    positions = _as_tensor(positions).to(torch.int32)
    out_absorb_w = _as_tensor(out_absorb_w)
    gate = _as_tensor(gate)
    o_proj_w = _as_tensor(o_proj_w)

    batch, heads, _ = q_nope.shape
    prior_len = ckv_prior.shape[1]
    value_width = out_absorb_w.shape[1]
    io_dtype = q_nope.dtype

    prior_scores = (
        torch.einsum("bhc,bsc->bhs", q_nope.float(), ckv_prior.float())
        + torch.einsum("bhr,bsr->bhs", q_pe.float(), kpe_prior.float())
    ) * float(softmax_scale)
    slots = torch.arange(prior_len, device=prior_scores.device).view(1, 1, prior_len)
    prior_scores = prior_scores.masked_fill(
        slots >= positions.view(batch, 1, 1),
        -30000.0,
    )
    current_scores = (
        torch.einsum("bhc,bc->bh", q_nope.float(), ckv_cur.float())
        + torch.einsum("bhr,br->bh", q_pe.float(), kpe_cur.float())
    ).unsqueeze(-1) * float(softmax_scale)

    probabilities = torch.softmax(
        torch.cat([prior_scores, current_scores], dim=-1),
        dim=-1,
        dtype=torch.float32,
    ).to(io_dtype)
    latent = torch.einsum("bhs,bsc->bhc", probabilities[..., :prior_len], ckv_prior) + probabilities[
        ..., prior_len:
    ] * ckv_cur.unsqueeze(1)
    attn = torch.einsum("bhc,hdc->bhd", latent, out_absorb_w)
    attn = attn.reshape(batch, heads * value_width)
    output = torch.matmul(attn * gate, o_proj_w)
    return {"output": output.to(io_dtype)}
