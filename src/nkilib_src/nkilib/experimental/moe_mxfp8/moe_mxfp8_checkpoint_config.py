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

"""Shared activation-checkpoint configuration for the MXFP8 MoE fwd/bwd pair."""

from dataclasses import dataclass

import nki.language as nl


@dataclass(frozen=True)
class MXFP8MOECheckpointConfig(nl.NKIObject):
    """Whether the forward emits the gate/up activation checkpoint for the backward.

    ``save_gate_up_proj_act`` controls whether the forward saves
    gate_up_proj_act_checkpoint_T: when disabled the checkpoint is not computed/
    stored/allocated/returned (the backward recomputes it). The gate/up checkpoint
    is required by the current backward, so it defaults to saved.

    The checkpoint is always stored token-major as ``[N, 2, B, I_TP]``. The
    scaled-intermediate checkpoint is not emitted because the backward does not
    consume it.

    Args:
        save_gate_up_proj_act (bool): Save gate_up_proj_act_checkpoint_T (clamped
            gate/up pre-activations). Required by the current backward.
    """

    save_gate_up_proj_act: bool = True
