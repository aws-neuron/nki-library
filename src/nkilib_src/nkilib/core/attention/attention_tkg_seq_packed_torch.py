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

"""Torch utilities for constructing sequence-packed Attention TKG metadata."""

import math
from typing import NamedTuple

import torch


class PackedAccumulatorRouteTables(NamedTuple):
    """Tensor Indirection routes for row partials and persistent group state."""

    partial_route_table: torch.Tensor
    group_state_route_table: torch.Tensor


def build_packed_accumulator_route_tables(
    seq_id_table: torch.Tensor,
    *,
    d_head: int,
    s_active_qh: int,
    rows_per_nc: int,
    bs_per_nc: int,
) -> PackedAccumulatorRouteTables:
    """Derive packed-accumulator routes from the canonical sequence-ID table.

    ``seq_id_table`` uses absolute sequence IDs and ``-1`` for padding. Rows are
    contiguous by NeuronCore, with exactly ``rows_per_nc`` rows per core. Within
    a row, all slots for one sequence are assigned to the group named by their
    first slot. The partial route places each slot's record in that group; the
    group-state route maps the group to its NC-local persistent sequence state.
    Unused groups map to one extra dummy state record.

    Max, sum, and PV reuse both routes. Each returned table has shape
    ``[num_rows, d_head, ceil(num_slots * s_active_qh / 16)]`` and dtype
    ``torch.uint16``.
    """
    if seq_id_table.ndim != 2:
        raise ValueError(f"seq_id_table must be rank 2, got shape {tuple(seq_id_table.shape)}")
    if seq_id_table.dtype == torch.bool or torch.is_floating_point(seq_id_table) or torch.is_complex(seq_id_table):
        raise TypeError(f"seq_id_table must use an integer dtype, got {seq_id_table.dtype}")
    if rows_per_nc <= 0 or bs_per_nc <= 0 or s_active_qh <= 0:
        raise ValueError("rows_per_nc, bs_per_nc, and s_active_qh must be positive")
    if d_head <= 0 or d_head % 16 != 0:
        raise ValueError(f"d_head must be a positive multiple of 16, got {d_head}")

    num_rows, num_slots = seq_id_table.shape
    if num_rows == 0 or num_slots == 0:
        raise ValueError("seq_id_table must contain at least one row and one slot")
    if num_slots % 2 != 0:
        raise ValueError(f"num_slots ({num_slots}) must be even")
    if num_rows % rows_per_nc != 0:
        raise ValueError(f"num_rows ({num_rows}) must be divisible by rows_per_nc ({rows_per_nc})")
    if num_slots > bs_per_nc:
        raise ValueError(f"num_slots ({num_slots}) must not exceed bs_per_nc ({bs_per_nc})")

    device = seq_id_table.device
    seq_ids = seq_id_table.to(dtype=torch.int64)
    nc_ids = torch.arange(num_rows, device=device, dtype=torch.int64) // rows_per_nc
    sequence_offsets = nc_ids * bs_per_nc
    local_seq_ids = seq_ids - sequence_offsets[:, None]
    valid = seq_ids >= 0
    invalid_ids = (seq_ids < -1) | (valid & ((local_seq_ids < 0) | (local_seq_ids >= bs_per_nc)))
    if torch.any(invalid_ids).item():
        bad_row, bad_slot = torch.nonzero(invalid_ids, as_tuple=False)[0].tolist()
        raise ValueError(
            f"seq_id_table[{bad_row}, {bad_slot}]={seq_ids[bad_row, bad_slot].item()} "
            "is not padding or local to that row's NeuronCore"
        )
    # Padding needs an in-bounds route even though its masked partial is an
    # identity. Reusing local sequence zero keeps the route valid and harmless.
    local_seq_ids = torch.where(valid, local_seq_ids, torch.zeros_like(local_seq_ids))

    # Each distinct sequence in a row owns the group named by its first slot.
    slot_ids = torch.arange(num_slots, device=device, dtype=torch.int64)
    equal_ids = local_seq_ids[:, :, None] == local_seq_ids[:, None, :]
    candidate_groups = slot_ids.view(1, 1, num_slots).expand(num_rows, num_slots, num_slots)
    slot_groups = torch.where(
        equal_ids,
        candidate_groups,
        torch.full_like(candidate_groups, num_slots),
    ).amin(dim=2)

    slots = slot_ids.view(1, 1, num_slots, 1)
    lanes = torch.arange(s_active_qh, device=device, dtype=torch.int64).view(1, 1, 1, s_active_qh)
    groups = slot_groups[:, None, :, None]
    group_ids = slot_ids.view(1, 1, num_slots)
    used_groups = (slot_groups[:, :, None] == group_ids).any(dim=1)
    # A group that owns no slot routes to the extra persistent dummy record.
    # This gives every fixed-width route entry a legal destination.
    group_sequences = torch.where(
        used_groups,
        local_seq_ids,
        torch.full_like(local_seq_ids, bs_per_nc),
    )

    # Source zero in each group is reserved for its prior persistent state;
    # current row slots occupy sources 1..num_slots.
    partial_route = (groups * (num_slots + 1) + (slots + 1)) * s_active_qh + lanes
    group_state_route = group_sequences[:, :, None] * s_active_qh + lanes.reshape(1, 1, s_active_qh)

    max_offset = max(
        partial_route.max().item(),
        group_state_route.max().item(),
    )
    if max_offset >= 2**16:
        raise ValueError(f"packed accumulator route offset {max_offset} exceeds uint16")

    def to_snake_table(offsets: torch.Tensor) -> torch.Tensor:
        """Format logical offsets for 16-lane Tensor Indirection quadrants."""
        flat = offsets.reshape(num_rows, -1)
        index_cols = math.ceil(flat.shape[1] / 16)
        padded = torch.zeros(
            (num_rows, index_cols * 16),
            dtype=torch.int64,
            device=device,
        )
        padded[:, : flat.shape[1]] = flat
        first_quadrant = padded.reshape(num_rows, index_cols, 16).transpose(1, 2)
        return first_quadrant.repeat(1, d_head // 16, 1).to(dtype=torch.uint16)

    return PackedAccumulatorRouteTables(
        partial_route_table=to_snake_table(partial_route),
        group_state_route_table=to_snake_table(group_state_route),
    )
