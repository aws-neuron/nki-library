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

"""MXFP4-packed -> dense weight adapters for the GPT-OSS MoE golden comparison.

The nkilib MoE kernel consumes MXFP4 weights in a tiled/shuffled layout; the
golden oracle wants dense fp32 ``[E,H,2,I]`` gate/up and ``[E,I,H]`` down.

These adapters reconstruct the *effective dense weight the kernel contracts with*
by replicating the unpack + block-scale of the nkilib MX torch refs
(``mlp_proj_mx_torch.py``) up to the point just before ``mx_matmul``, then folding
in ``mx_matmul``'s reshape/scale so the result is the plain dense matrix ``W`` such
that ``out = hidden @ W`` (gate/up) / ``out = inter @ W`` (down) — the same
contraction the kernel performs.

Key insight — the ``(partition, q)`` -> natural-index ordering.
    ``mx_matmul`` contracts over the partition axis ``K/4`` and the 4-wide ``q``
    axis. In ``gate_up_proj_mx_torch_ref`` the moving hidden operand is built
    (``_moe_tkg_mx_ref`` line ~598) as::

        h = active_in.reshape(BxS, q_width=4, n_H512, pmax=128).transpose(3,2,0,1)

    i.e. the *natural* hidden index decomposes as ``h_nat = q*(H//4) + row`` where
    ``row = h512*128 + p`` and ``q`` is the SLOWEST-varying axis. The stationary
    weight is unpacked to ``[K/4, F, 4]`` where the last (x4) axis IS that same
    ``q``. So to recover the dense weight in the golden's natural index order we
    must lay the weight out as ``h_nat = q*(K/4) + row`` (``permute(F, q, row)``),
    NOT ``row*4 + q``. The previous adapter used ``row*4 + q`` — hence the ~97%
    mismatch. Verified 100% against a one-hot brute-force probe of the refs for
    ``(H,I) in {512,1024}^2``.

Contraction shapes (from mx_matmul + the two projection refs):
  gate/up: w_flat[H/4, I*4] --(reshape [H/4,I,4], block-scale, permute [I,q,row])--> W_eff[I, H]
           => dense gate/up weight fed to the golden is W_eff.T = [H, I].
  down:    w_flat[Iq, H*4] --(reshape [Iq,H,4], block-scale, retile)--> W[I, H]
           where flat partition Iq = i512*p_I + row_p and golden index
           i_col = i512*512 + q*128 + row_p.
"""

import numpy as np
import torch
from nkilib_src.nkilib.core.utils.mx_torch_common import unpack_float4_x4

_Q_WIDTH = 4
_Q_HEIGHT = 8
_PMAX = 128


def _dequant_qkr(weight_qtz_flat, weight_scale_2d, F_out):
    """Unpack + block-scale an MX weight tile, returning ``w[K/4, F_out, 4]``.

    weight_qtz_flat: numpy float4_e2m1fn_x4 ``[K/4, F_out]`` (already transposed+reshaped)
    weight_scale_2d: numpy uint8 ``[K/8/4, F_out]``

    Mirrors ``mx_matmul``'s stationary handling: unpack to ``[K/4, F_out, 4]``, then
    apply ``2^(scale-127)`` per 8-row block. The returned axes are
    ``(partition=K/4, F_out, q=4)`` — the caller decides how ``(partition, q)`` map
    onto the natural contraction index.
    """
    w = unpack_float4_x4(weight_qtz_flat)  # [K/4, F_out*4] fp32
    Kq = w.shape[0]
    w = w.reshape(Kq, F_out, _Q_WIDTH)  # [K/4, F_out, 4]
    ss = torch.pow(2.0, torch.from_numpy(weight_scale_2d.astype(np.int32)).float() - 127.0)  # [K/8/4, F_out]
    SSP = ss.shape[0]
    w = (w.reshape(SSP, _Q_HEIGHT, F_out, _Q_WIDTH) * ss[:, None, :, None]).reshape(Kq, F_out, _Q_WIDTH)
    return w


def packed_mx_to_dense_gate_up(gate_up_w_qtz, gate_up_w_scale, H, I):
    """[E,128,2,n_H512,I] + [E,16,2,n_H512,I] -> dense [E,H,2,I] fp32.

    For each expert e and slice g in {gate(0), up(1)}: the ref does
    ``w_flat = unpack(weight_qtz.transpose(1,0,2).reshape(H//4, I))`` then
    ``mx_matmul`` contracts the hidden operand whose natural index is
    ``h_nat = q*(H//4) + row``. We unpack to ``w[H/4, I, 4]`` and lay it out as
    ``W[h_nat, i]`` via ``permute(I, q, row)`` so it matches the golden's natural
    hidden ordering, returning ``W.T = [H, I]``.
    """
    E = gate_up_w_qtz.shape[0]
    dense = torch.zeros(E, H, 2, I, dtype=torch.float32)
    for e in range(E):
        for g in range(2):
            wq = gate_up_w_qtz[e, :, g, :, :]  # numpy x4 [128, n_H512, I]
            wq_flat = wq.transpose(1, 0, 2).reshape(H // _Q_WIDTH, I)
            sc = gate_up_w_scale[e, :, g, :, :]  # numpy uint8 [16, n_H512, I]
            sc_flat = sc.transpose(1, 0, 2).reshape(H // _Q_HEIGHT // _Q_WIDTH, I)
            w = _dequant_qkr(wq_flat, sc_flat, I)  # [H/4, I, 4] = [row, i, q]
            Kq = w.shape[0]
            # h_nat = q*(H/4) + row  =>  [i, q, row] -> flatten -> [I, H] -> .T
            dense[e, :, g, :] = w.permute(1, 2, 0).reshape(I, _Q_WIDTH * Kq).T
    return dense


def packed_mx_to_dense_down(down_w_qtz, down_w_scale, H, I):
    """[E,p_I,n_I512,H] + [E,p_I//8,n_I512,H] -> dense [E,I,H] fp32.

    ``down_proj_mx_torch_ref`` does ``weight_np = weight_qtz.transpose(1,0,2).reshape(-1, H)``
    then ``mx_matmul`` contracts over the intermediate axis. The flat partition index
    is ``P_flat = i512*p_I + row_p`` and the 4-wide ``q`` axis maps onto the golden's
    natural intermediate index as ``i_col = i512*512 + q*128 + row_p`` (mirroring the
    ``[128, n_I512, BxS, 4]`` output tiling of the gate/up projection that feeds the
    down projection). We unpack to ``w[P_flat, H, 4]`` and re-tile to ``[I, H]``.
    """
    E, p_I, n_I512, Hc = down_w_qtz.shape
    assert Hc == H
    dense = torch.zeros(E, I, H, dtype=torch.float32)
    for e in range(E):
        wq = down_w_qtz[e]  # [p_I, n_I512, H] x4
        wq_flat = wq.transpose(1, 0, 2).reshape(n_I512 * p_I, H)  # [(n_I512*p_I), H]
        sc = down_w_scale[e]  # [p_I//8, n_I512, H]
        sc_flat = sc.transpose(1, 0, 2).reshape(n_I512 * (p_I // _Q_HEIGHT), H)
        w = _dequant_qkr(wq_flat, sc_flat, H)  # [(n_I512*p_I), H, 4] = [P_flat, h, q]
        # P_flat = i512*p_I + row_p ; i_col = i512*512 + q*128 + row_p (=i512*(4*p_I)+q*p_I+row_p)
        full = w.reshape(n_I512, p_I, H, _Q_WIDTH).permute(0, 3, 1, 2).reshape(n_I512 * _Q_WIDTH * p_I, H)
        dense[e] = full[:I, :]  # drop pad rows when I % 512 != 0
    return dense


def unpack_gate_up_bias(gate_up_bias, H, I):
    """MX fused gate/up bias [E, I_p, 2, n_I512, 4] -> golden dense [E, 2, I] fp32.

    The MX bias is added inside ``gate_up_proj_mx_torch_ref`` (line 113-117) in the
    tiled output space ``[128, n_I512, BxS, 4]`` via ``out += bias_t.unsqueeze(2)``
    with ``bias_t = bias[e][:, g, :, :]`` of shape ``[I_p, n_I512, 4]``. The output
    tiling maps ``out[row_p, i512, :, q]`` to logical intermediate index
    ``i_col = i512*512 + q*128 + row_p`` (= ``i512*(4*I_p) + q*I_p + row_p``). Invert
    that exact mapping to recover the ``[E, 2, I]`` dense bias the golden adds.
    """
    E, I_p, two, n_I512, q = gate_up_bias.shape
    assert two == 2 and q == _Q_WIDTH
    b = torch.as_tensor(np.asarray(gate_up_bias, dtype=np.float32))  # [E, I_p, 2, n_I512, 4]
    # [E, I_p, 2, n_I512, q] -> [E, 2, n_I512, q, I_p] -> flatten i_col, drop pad
    out = b.permute(0, 2, 3, 4, 1).reshape(E, 2, n_I512 * _Q_WIDTH * I_p)
    return out[:, :, :I].contiguous()
