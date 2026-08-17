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

"""Shared activation-checkpoint configuration for the MXFP8 MoE fwd/bwd pair.

Lives at the ``moe_mxfp8`` package root (not under ``fwd/`` or ``bwd/``) because
both passes share the same set of activation checkpoints: the forward produces
them and the backward consumes them, so a single source of truth keeps the two
sides in lockstep. The backward will consume this config in a future change to
decide which checkpoints it can rely on versus must recompute.
"""

from dataclasses import dataclass
from enum import Enum

import nki.language as nl


class CheckpointLayout(Enum):
    """Store layout for an activation checkpoint.

    Attributes:
        DIRECT: Store the per-block activation in its natural token-major layout
            ([..., B, I_TP]) with no transpose — a single plain DMA of the whole
            block. This is the default: writing full token rows keeps the
            destination contiguous, so the DMA coalesces into max-size packets.
        TRANSPOSED: Store the per-block activation transposed to an I_TP-major
            layout ([..., I_TP, B]), which is what the MXFP8 MoE backward consumes.
            NOT CURRENTLY IMPLEMENTED — the forward's block-granular store path
            supports DIRECT only and raises on TRANSPOSED. The previous per-tile
            transposed store was removed because storing a tile at a time wrote only
            ``tile_n`` of each token row, splitting every store into tiny
            (~1 KiB) DMA packets that bottlenecked the kernel.
    """

    TRANSPOSED = 0
    DIRECT = 1


def checkpoint_block_dims(layout: "CheckpointLayout", I_TP: int, block_size: int):
    """Per-block trailing dims (..., X, Y) for a checkpoint stored in ``layout``.

    Single source of truth for the on-HBM shape of a checkpoint tile, shared by
    the kernel allocation, the torch reference, and the test's output_shapes so a
    new layout only has to be registered here. TRANSPOSED is I_TP-major
    (``(I_TP, B)``); DIRECT is token-major (``(B, I_TP)``). Callers prepend the
    block/half dims (e.g. ``(N, 2, *dims)`` for gate/up, ``(N, *dims)`` for the
    scaled intermediate).
    """
    if layout == CheckpointLayout.DIRECT:
        return (block_size, I_TP)
    return (I_TP, block_size)


@dataclass(frozen=True)
class MXFP8MOECheckpointConfig(nl.NKIObject):
    """Which MXFP8 MoE activation checkpoints are exchanged between fwd and bwd.

    The forward can emit two activation checkpoints for the MXFP8 MoE backward.
    Each ``save_*`` flag independently controls whether its checkpoint is saved:
    on the forward side a disabled checkpoint is not computed/stored/allocated/
    returned; on the backward side (future) a disabled checkpoint is recomputed
    instead of being read. The gate/up checkpoint is required by the current
    backward, so it defaults to saved.

    Each ``*_layout`` field independently selects that checkpoint's store layout
    (see ``CheckpointLayout``). Both default to DIRECT, the only layout the
    forward's block-granular store currently implements; the tensors keep their
    ``_T`` output names for continuity with the backward's parameter names.
    TRANSPOSED (the layout the backward consumes) is not currently supported and
    raises at trace time — restoring it means adding a block-level transposed
    store to the forward, not a per-tile one.

    Args:
        save_gate_up_proj_act (bool): Save gate_up_proj_act_checkpoint_T (clamped
            gate/up pre-activations). Required by the current backward.
        save_scaled_intermediate (bool): Save scaled_intermediate_checkpoint_T
            (SiLU(gate)*up*EA) so the backward can skip its Phase-1 recompute +
            Phase-4 transpose.
        gate_up_proj_act_layout (CheckpointLayout): Store layout for the gate/up
            checkpoint. TRANSPOSED -> [N, 2, I_TP, B]; DIRECT -> [N, 2, B, I_TP].
        scaled_intermediate_layout (CheckpointLayout): Store layout for the scaled
            intermediate. TRANSPOSED -> [N, I_TP, B]; DIRECT -> [N, B, I_TP].
    """

    save_gate_up_proj_act: bool = True
    save_scaled_intermediate: bool = True
    gate_up_proj_act_layout: CheckpointLayout = CheckpointLayout.DIRECT
    scaled_intermediate_layout: CheckpointLayout = CheckpointLayout.DIRECT
