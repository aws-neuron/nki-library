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

import nki.language as nl


@dataclass(frozen=True)
class MXFP8MOECheckpointConfig(nl.NKIObject):
    """Which MXFP8 MoE activation checkpoints are exchanged between fwd and bwd.

    The forward can emit two activation checkpoints for the MXFP8 MoE backward.
    Each flag independently controls whether its checkpoint is saved: on the
    forward side a disabled checkpoint is not computed/stored/allocated/returned;
    on the backward side (future) a disabled checkpoint is recomputed instead of
    being read. The gate/up checkpoint is required by the current backward, so it
    defaults to saved.

    Args:
        save_gate_up_proj_act (bool): Save gate_up_proj_act_checkpoint_T
            ([N, 2, I_TP, B], clamped gate/up pre-activations). Required by the
            current backward.
        save_scaled_intermediate (bool): Save scaled_intermediate_checkpoint_T
            ([N, I_TP, B], SiLU(gate)*up*EA transposed) so the backward can skip
            its Phase-1 recompute + Phase-4 transpose.
    """

    save_gate_up_proj_act: bool = True
    save_scaled_intermediate: bool = True
