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
            Implemented by a block-granular store that PE-transposes the block in
            ``<=TILE_M`` I_TP chunks into a token-ordered staging buffer, then issues
            one coalesced DMA per chunk (each stored I_TP row is a contiguous token
            run). This is distinct from the removed per-tile transposed store, which
            wrote only ``tile_n`` of each token row and split every store into tiny
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
    """Whether the forward emits the gate/up activation checkpoint for the backward.

    ``save_gate_up_proj_act`` controls whether the forward saves
    gate_up_proj_act_checkpoint_T: when disabled the checkpoint is not computed/
    stored/allocated/returned (the backward recomputes it). The gate/up checkpoint
    is required by the current backward, so it defaults to saved.

    The checkpoint's store LAYOUT is no longer a flag here — it is derived from the
    ``gate_up_proj_act_td`` orientation on ``MXFP8MOEFwdConfig`` (F_BY_K -> DIRECT,
    K_BY_F -> TRANSPOSED). The scaled-intermediate checkpoint is not emitted (the
    backward never consumed it).

    Args:
        save_gate_up_proj_act (bool): Save gate_up_proj_act_checkpoint_T (clamped
            gate/up pre-activations). Required by the current backward.
    """

    save_gate_up_proj_act: bool = True
