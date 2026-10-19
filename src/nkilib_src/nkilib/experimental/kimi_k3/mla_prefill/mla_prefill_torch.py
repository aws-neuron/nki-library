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

"""PyTorch oracle for Kimi-K3 dense causal MLA prefill."""

import numpy as np
import torch


def _as_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "dtype") and str(value.dtype) == "bfloat16":
        return torch.from_numpy(np.asarray(value, dtype=np.float32)).to(torch.bfloat16)
    return torch.as_tensor(value)


def kimi_k3_mla_prefill_torch_ref(
    q_lift_hbm: torch.Tensor,
    q_pe_hbm: torch.Tensor,
    c_kv_hbm: torch.Tensor,
    k_pe_hbm: torch.Tensor,
    softmax_scale: float,
    topk_indices_hbm: torch.Tensor | None = None,
    dense: bool = True,
    q_pos_offset_hbm: torch.Tensor | None = None,
):
    """Return natural-order BF16 latent attention ``{"out_attn": [B,S,H*L]}``."""
    if not dense:
        raise ValueError("Kimi-K3 MLA prefill requires dense=True")

    # Deliberately unused: dense mode must not reinterpret None as an empty
    # gather or make junk index values observable.
    del topk_indices_hbm

    q_lift = _as_tensor(q_lift_hbm)
    q_pe = _as_tensor(q_pe_hbm)
    c_kv = _as_tensor(c_kv_hbm)
    k_pe = _as_tensor(k_pe_hbm)
    offset = 0 if q_pos_offset_hbm is None else int(_as_tensor(q_pos_offset_hbm).reshape(-1)[0])

    batch, seqlen, heads, latent_width = q_lift.shape
    kv_seqlen = c_kv.shape[1]
    q_lift_bhsd = q_lift.permute(0, 2, 1, 3)
    q_pe_bhsd = q_pe.permute(0, 2, 1, 3)
    k_pe_b1sd = k_pe.unsqueeze(1)

    scores = torch.matmul(q_pe_bhsd, k_pe_b1sd.transpose(2, 3))
    scores = scores + torch.einsum(
        "bhqc,bsc->bhqs",
        q_lift_bhsd,
        c_kv,
    )
    scores = scores * float(softmax_scale)

    query_global = offset + torch.arange(seqlen, device=scores.device)
    future = torch.arange(kv_seqlen, device=scores.device)[None, :] > query_global[:, None]
    scores = torch.where(
        future[None, None, :, :],
        torch.finfo(scores.dtype).min,
        scores,
    )
    probabilities = torch.softmax(
        scores,
        dim=-1,
        dtype=torch.float32,
    ).to(scores.dtype)
    latent = torch.einsum("bhqs,bsc->bhqc", probabilities, c_kv)
    natural = latent.permute(0, 2, 1, 3).reshape(
        batch,
        seqlen,
        heads * latent_width,
    )
    return {"out_attn": natural.to(torch.bfloat16)}
