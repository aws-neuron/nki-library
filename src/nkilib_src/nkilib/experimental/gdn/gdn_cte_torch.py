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
"""Torch reference for the gdn_cte prefill family -- ONE reference for BOTH algorithms.

It walks the recurrence ONE TOKEN AT A TIME, straight from the defining equations:

    S_t = g_t * S_{t-1} + beta_t * k_t (v_t - g_t (k_t . S_{t-1}))^T
    o_t = q_t . S_t
"""

from typing import Optional

import torch

from ...core.utils.kernel_assert import kernel_assert
from .gdn_cte_utils import ALGORITHM_NEUMANN, ALGORITHMS, DEFAULT_ALGORITHM, MAX_HEAD_DIM

_CHUNK = 128
_IDX_UPPER_NEG = 0
_IDX_TRIL_STRICT = 1
_IDX_EYE = 2
_NEG_INF_MASK_VALUE = -1e9
"""The pack's layout, MIRRORED from `gdn_cte_utils` rather than imported -- DO NOT "fix" this
into an import. `_validate_mask_pack` below checks the pack that module BUILDS, so importing
its layout would make the check compare the pack against itself: a plane reordering there
would then pass silently. Mirrored, the two must agree, and any divergence fails every test
that passes a pack. Contract values neither side validates (`ALGORITHMS`, `MAX_HEAD_DIM`) are
imported normally, above."""

# Mirrors `gdn_cte_schur.DEFAULT_GROUP_SIZE`, for the same reason the plane layout is mirrored:
# importing it would pull `nki` into a file that needs only torch. The value is inert here --
# the reference is per-token -- but it must track the kernel default it claims parity with.
_DEFAULT_GROUP_SIZE = 8


def gdn_cte_torch_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g_log: torch.Tensor,
    beta: torch.Tensor,
    scale: float = 1.0,
    algorithm: str = DEFAULT_ALGORITHM,
    mask_pack: Optional[torch.Tensor] = None,
    group_size: int = _DEFAULT_GROUP_SIZE,
    recurrent_state: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token fp32 reference for the chunked GDN prefill kernels.

    Dimensions:
        B: Batch size
        Hk: Key/query head count
        Hv: Value head count, a multiple of Hk
        Dk: Key/state head dimension
        Dv: Value/state head dimension
        S: Sequence length

    Args:
        query (torch.Tensor): [B, Hk, S, Dk], TOKEN-major, NOT pre-scaled (the kernels
            apply `scale` on device, and so does this reference).
        key (torch.Tensor): [B, Hk, S, Dk], TOKEN-major.
        value (torch.Tensor): [B, Hv, S, Dv].
        g_log (torch.Tensor): [B, Hv, S], per-token log decay (negative).
        beta (torch.Tensor): [B, Hv, S], per-token update strength in (0, 1).
        scale (float): Query scaling, applied here as it is on device.
        algorithm (str): Accepted for signature parity with the `gdn_cte` dispatcher.
            Both algorithms compute this same function.
        mask_pack (Optional[torch.Tensor]): [N_MASKS, CHUNK, CHUNK] constant mask pack,
            required for `"schur"` and omitted for `"neumann"`, matching the kernel.
            Carries no math here; its fixed planes are VALIDATED when present, so a
            corrupted pack fails with a clear message.
        group_size (int): Accepted for signature parity with the kernels. Grouping is a
            scheduling transform and does not affect the result.
        recurrent_state (Optional[torch.Tensor]): [B, Hv, Dk, Dv] initial state, or None
            for a zero state (as the kernels do when the argument is omitted).

    Returns:
        out (torch.Tensor): [B, Hv, S, Dv], head-major, UN-normalized.
        state_out (torch.Tensor): [B, Hv, Dk, Dv], final recurrent state.

    Notes:
        - The recurrence is genuinely sequential in t; the loop is vectorized over
          (B, Hv) only, which is what keeps it independent of either kernel's tiling.
        - GQA is modeled by materializing the key/query head repeat once, up front.
          The kernels instead index the shared head -- same values, and doing it
          differently here is part of the independence.
        - `scale` is applied to the OUTPUT projection of q only, never to the state
          update, mirroring the kernels: the carried state is scale-independent.

    Pseudocode:
        state = recurrent_state or 0
        for t in range(S):
            state = state * exp(g_log[t])
            v_old = (state * k[t]).sum(Dk)
            state = state + outer(k[t], (v[t] - v_old) * beta[t])
            out[t] = scale * (state * q[t]).sum(Dk)
    """
    batch, n_k_heads, seqlen, key_head_dim = query.shape
    n_v_heads = value.shape[1]
    value_head_dim = value.shape[-1]
    kernel_assert(
        n_v_heads % n_k_heads == 0,
        f"n_v_heads must be divisible by n_k_heads, got {n_v_heads} and {n_k_heads}",
    )
    # Mirrors the kernels' head-dim contract, so a bad shape fails here the same way rather
    # than as a torch broadcast error. Dk != Dv is allowed (the schur path supports it).
    kernel_assert(
        key_head_dim == key.shape[-1],
        f"query and key must have the same head dim, got {key_head_dim} and {key.shape[-1]}",
    )
    kernel_assert(
        key_head_dim <= MAX_HEAD_DIM and value_head_dim <= MAX_HEAD_DIM,
        f"head dims must be at most {MAX_HEAD_DIM}, got key_head_dim={key_head_dim} and "
        f"value_head_dim={value_head_dim}",
    )
    kernel_assert(
        algorithm in ALGORITHMS,
        f"algorithm must be one of {ALGORITHMS}, got {algorithm!r}",
    )
    kernel_assert(
        isinstance(group_size, int) and group_size >= 1,
        f"group_size must be a positive int, got {group_size}",
    )
    # Validated when supplied. `"schur"` requires one (the dispatcher asserts that); the
    # neumann path takes none, so there is nothing to check there.
    kernel_assert(
        mask_pack is not None or algorithm == ALGORITHM_NEUMANN,
        f"algorithm={algorithm!r} requires `mask_pack`",
    )
    if mask_pack is not None:
        _validate_mask_pack(mask_pack)
    v_heads_per_k_head = n_v_heads // n_k_heads

    query = query.float()
    key = key.float()
    value = value.float()
    beta = beta.float()

    # GQA: value head h shares key head h // v_heads_per_k_head. Materialized once so
    # the step loop is a plain vectorized expression over (B, Hv).
    query_rep = query.repeat_interleave(v_heads_per_k_head, dim=1)
    key_rep = key.repeat_interleave(v_heads_per_k_head, dim=1)

    if recurrent_state is None:
        state = torch.zeros((batch, n_v_heads, key_head_dim, value_head_dim), dtype=torch.float32)
    else:
        state = recurrent_state.float().clone()
    out = torch.zeros((batch, n_v_heads, seqlen, value_head_dim), dtype=torch.float32)

    gate = g_log.float().exp()

    for step in range(seqlen):
        query_step = query_rep[:, :, step, :]
        key_step = key_rep[:, :, step, :]
        value_step = value[:, :, step, :]
        gate_step = gate[:, :, step][..., None, None]
        beta_step = beta[:, :, step][..., None]

        state = state * gate_step
        value_old = (state * key_step[..., None]).sum(dim=-2)
        delta = (value_step - value_old) * beta_step
        state = state + key_step[..., None] * delta[..., None, :]
        out[:, :, step, :] = scale * (state * query_step[..., None]).sum(dim=-2)

    return out, state


def _validate_mask_pack(mask_pack: torch.Tensor) -> None:
    """Validate the constant mask pack the kernels cannot build for themselves.

    Args:
        mask_pack (torch.Tensor): [N_MASKS, CHUNK, CHUNK] fp32 pack.

    Notes:
        Only the three FIXED planes are checked, in the pack's documented order:
        0 upper_neg, 1 tril_strict, 2 eye. The per-level off masks are not re-derived
        here. A wrong pack would otherwise surface as a kernel accuracy failure with no
        hint of where it came from.
    """
    kernel_assert(
        mask_pack.ndim == 3 and mask_pack.shape[1] == mask_pack.shape[2],
        f"mask_pack must be (N_MASKS, CHUNK, CHUNK), got {tuple(mask_pack.shape)}",
    )
    chunk_len = mask_pack.shape[1]
    kernel_assert(
        chunk_len == _CHUNK,
        f"mask_pack planes must be {_CHUNK}x{_CHUNK}, got {chunk_len}x{chunk_len}",
    )
    pack = mask_pack.float()
    ones = torch.ones((chunk_len, chunk_len), dtype=torch.float32)
    expected_upper_neg = (ones - torch.tril(ones, diagonal=0)) * _NEG_INF_MASK_VALUE
    kernel_assert(
        torch.equal(pack[_IDX_UPPER_NEG], expected_upper_neg),
        "mask_pack plane 0 is not upper_neg -- rebuild it with build_mask_pack()",
    )
    kernel_assert(
        torch.equal(pack[_IDX_TRIL_STRICT], torch.tril(ones, diagonal=-1)),
        "mask_pack plane 1 is not tril_strict -- rebuild it with build_mask_pack()",
    )
    kernel_assert(
        torch.equal(pack[_IDX_EYE], torch.eye(chunk_len, dtype=torch.float32)),
        "mask_pack plane 2 is not the identity -- rebuild it with build_mask_pack()",
    )
