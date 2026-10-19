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

"""End-to-end coverage for the MXFP8 MoE forward/backward checkpoint contract."""

from typing import Optional, final

import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.moe_mxfp8.bwd.blockwise_mm_backward_mxfp8 import (
    blockwise_mm_bwd_mxfp8,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.bwd.blockwise_mm_backward_mxfp8_torch import (
    blockwise_mm_bwd_mxfp8_torch_ref,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.bwd.config import MXFP8MOEBwdConfig
from nkilib_src.nkilib.experimental.moe_mxfp8.fwd.blockwise_mm_forward_mxfp8 import (
    blockwise_mm_fwd_mxfp8,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.fwd.blockwise_mm_forward_mxfp8_torch import (
    blockwise_mm_fwd_mxfp8_torch_ref,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.fwd.config import MXFP8MOEFwdConfig
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.common_dataclasses import (
    TensorDescriptor,
    TensorOrientation,
)

from test.integration.nkilib.experimental.moe_mxfp8.mxfp8_moe_fwd_test_utils import (
    build_mxfp8_moe_fwd_inputs,
)
from test.integration.nkilib.experimental.moe_mxfp8.mxfp8_moe_validated_common import (
    make_moe_validated_comparator,
)
from test.utils import common_dataclasses
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

bfloat16 = nl.bfloat16

TOKENS = 1024
HIDDEN = 256
INTERMEDIATE = 256
BLOCK_SIZE = 512
EXPERTS = 1


def moe_mxfp8_fwd_bwd_e2e(
    hidden_states: nl.NkiTensor,
    expert_affinities_masked: nl.NkiTensor,
    fwd_gate_up_proj_weight: nl.NkiTensor,
    fwd_down_proj_weight: nl.NkiTensor,
    bwd_gate_up_proj_weight: nl.NkiTensor,
    bwd_down_proj_weight: nl.NkiTensor,
    token_position_to_id: nl.NkiTensor,
    block_to_expert: nl.NkiTensor,
    output_hidden_states_grad: nl.NkiTensor,
    block_size: int = BLOCK_SIZE,
    fwd_config: Optional[MXFP8MOEFwdConfig] = None,
    bwd_config: Optional[MXFP8MOEBwdConfig] = None,
) -> tuple:
    """Feed the forward-produced checkpoint directly into the backward kernel."""
    if fwd_config is None:
        output_hidden_states, gate_up_proj_act_checkpoint_T = blockwise_mm_fwd_mxfp8(
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=fwd_gate_up_proj_weight,
            down_proj_weight=fwd_down_proj_weight,
            token_position_to_id=token_position_to_id,
            block_to_expert=block_to_expert,
            block_size=block_size,
        )
    else:
        output_hidden_states, gate_up_proj_act_checkpoint_T = blockwise_mm_fwd_mxfp8(
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=fwd_gate_up_proj_weight,
            down_proj_weight=fwd_down_proj_weight,
            token_position_to_id=token_position_to_id,
            block_to_expert=block_to_expert,
            block_size=block_size,
            config=fwd_config,
        )
    # Forward shards the checkpoint across both NeuronCores. Ensure every
    # shared-HBM store is visible before either core starts consuming it.
    nisa.core_barrier(
        data=gate_up_proj_act_checkpoint_T,
        cores=(0, 1),
        name="mxfp8_fwd_bwd_checkpoint_handoff",
    )
    bwd_outputs = blockwise_mm_bwd_mxfp8(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=bwd_gate_up_proj_weight,
        down_proj_weight=bwd_down_proj_weight,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        output_hidden_states_grad=output_hidden_states_grad,
        block_size=block_size,
        gate_up_proj_act_checkpoint_T=gate_up_proj_act_checkpoint_T,
        config=bwd_config,
    )
    return (output_hidden_states, gate_up_proj_act_checkpoint_T) + bwd_outputs


def moe_mxfp8_fwd_bwd_e2e_torch_ref(
    hidden_states,
    expert_affinities_masked,
    fwd_gate_up_proj_weight,
    fwd_down_proj_weight,
    bwd_gate_up_proj_weight,
    bwd_down_proj_weight,
    token_position_to_id,
    block_to_expert,
    output_hidden_states_grad,
    block_size: int = BLOCK_SIZE,
    fwd_config: Optional[MXFP8MOEFwdConfig] = None,
    bwd_config: Optional[MXFP8MOEBwdConfig] = None,
) -> dict:
    """Chain the matching PyTorch references through the same checkpoint."""
    fwd_result = blockwise_mm_fwd_mxfp8_torch_ref(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=fwd_gate_up_proj_weight,
        down_proj_weight=fwd_down_proj_weight,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        block_size=block_size,
        config=fwd_config,
    )
    checkpoint = fwd_result["gate_up_proj_act_checkpoint_T"]
    bwd_result = blockwise_mm_bwd_mxfp8_torch_ref(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=bwd_gate_up_proj_weight,
        down_proj_weight=bwd_down_proj_weight,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        output_hidden_states_grad=output_hidden_states_grad,
        block_size=block_size,
        gate_up_proj_act_checkpoint_T=checkpoint,
        config=bwd_config,
    )
    return {
        "output_hidden_states": fwd_result["output_hidden_states"],
        "gate_up_proj_act_checkpoint_T": checkpoint,
        **bwd_result,
    }


def moe_mxfp8_fwd_bwd_shared_weights(
    hidden_states: nl.NkiTensor,
    expert_affinities_masked: nl.NkiTensor,
    gate_up_proj_weight: nl.NkiTensor,
    down_proj_weight: nl.NkiTensor,
    token_position_to_id: nl.NkiTensor,
    block_to_expert: nl.NkiTensor,
    output_hidden_states_grad: nl.NkiTensor,
    block_size: int = BLOCK_SIZE,
    fwd_config: Optional[MXFP8MOEFwdConfig] = None,
    bwd_config: Optional[MXFP8MOEBwdConfig] = None,
) -> tuple:
    """Feed one physical BF16 weight copy unchanged to both passes."""
    return moe_mxfp8_fwd_bwd_e2e(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        fwd_gate_up_proj_weight=gate_up_proj_weight,
        fwd_down_proj_weight=down_proj_weight,
        bwd_gate_up_proj_weight=gate_up_proj_weight,
        bwd_down_proj_weight=down_proj_weight,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        output_hidden_states_grad=output_hidden_states_grad,
        block_size=block_size,
        fwd_config=fwd_config,
        bwd_config=bwd_config,
    )


def moe_mxfp8_fwd_bwd_shared_weights_torch_ref(
    hidden_states,
    expert_affinities_masked,
    gate_up_proj_weight,
    down_proj_weight,
    token_position_to_id,
    block_to_expert,
    output_hidden_states_grad,
    block_size: int = BLOCK_SIZE,
    fwd_config: Optional[MXFP8MOEFwdConfig] = None,
    bwd_config: Optional[MXFP8MOEBwdConfig] = None,
) -> dict:
    """Reference for the shared physical-weight wrapper."""
    return moe_mxfp8_fwd_bwd_e2e_torch_ref(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        fwd_gate_up_proj_weight=gate_up_proj_weight,
        fwd_down_proj_weight=down_proj_weight,
        bwd_gate_up_proj_weight=gate_up_proj_weight,
        bwd_down_proj_weight=down_proj_weight,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        output_hidden_states_grad=output_hidden_states_grad,
        block_size=block_size,
        fwd_config=fwd_config,
        bwd_config=bwd_config,
    )


def build_e2e_inputs():
    """Build one dense layer with each pass receiving its natural BF16 weights."""
    fwd_inputs = build_mxfp8_moe_fwd_inputs(
        tokens=TOKENS,
        hidden=HIDDEN,
        intermediate=INTERMEDIATE,
        expert=EXPERTS,
        block_size=BLOCK_SIZE,
        top_k=1,
        single_expert_dense=True,
    )
    fwd_gate_up = fwd_inputs["gate_up_proj_weight"]
    fwd_down = fwd_inputs["down_proj_weight"]

    # The backward contracts over the opposite axes, so give it the natural
    # transpose of the same logical weights.
    bwd_gate_up = np.ascontiguousarray(fwd_gate_up.transpose(0, 3, 2, 1))
    bwd_down = np.ascontiguousarray(fwd_down.transpose(0, 2, 1))

    rng = np.random.default_rng(301974009)
    return {
        "hidden_states": fwd_inputs["hidden_states"],
        "expert_affinities_masked": fwd_inputs["expert_affinities_masked"],
        "fwd_gate_up_proj_weight": fwd_gate_up,
        "fwd_down_proj_weight": fwd_down,
        "bwd_gate_up_proj_weight": bwd_gate_up,
        "bwd_down_proj_weight": bwd_down,
        "token_position_to_id": fwd_inputs["token_position_to_id"],
        "block_to_expert": fwd_inputs["block_to_expert"],
        "output_hidden_states_grad": rng.uniform(-1.0, 1.0, size=(TOKENS, HIDDEN)).astype(bfloat16),
        "block_size": BLOCK_SIZE,
        "fwd_config": None,
        "bwd_config": None,
    }


def build_shared_weight_e2e_inputs(
    weight_orientation: TensorOrientation,
):
    """Build one physical weight layout consumed unchanged by both passes."""
    fwd_inputs = build_mxfp8_moe_fwd_inputs(
        tokens=TOKENS,
        hidden=HIDDEN,
        intermediate=INTERMEDIATE,
        expert=EXPERTS,
        block_size=BLOCK_SIZE,
        top_k=1,
        single_expert_dense=True,
        weights_k_by_f=weight_orientation == TensorOrientation.K_BY_F,
    )
    # A forward-natural physical copy is K-by-F for the backward matmuls. A
    # backward-natural copy is the backward's default F-by-K input.
    bwd_weight_td = (
        TensorDescriptor(orientation=TensorOrientation.K_BY_F)
        if weight_orientation == TensorOrientation.F_BY_K
        else None
    )
    bwd_config = MXFP8MOEBwdConfig(gate_up_weight_td=bwd_weight_td, down_weight_td=bwd_weight_td)

    rng = np.random.default_rng(301974009)
    return {
        "hidden_states": fwd_inputs["hidden_states"],
        "expert_affinities_masked": fwd_inputs["expert_affinities_masked"],
        "gate_up_proj_weight": fwd_inputs["gate_up_proj_weight"],
        "down_proj_weight": fwd_inputs["down_proj_weight"],
        "token_position_to_id": fwd_inputs["token_position_to_id"],
        "block_to_expert": fwd_inputs["block_to_expert"],
        "output_hidden_states_grad": rng.uniform(-1.0, 1.0, size=(TOKENS, HIDDEN)).astype(bfloat16),
        "block_size": BLOCK_SIZE,
        "fwd_config": fwd_inputs["config"],
        "bwd_config": bwd_config,
    }


@pytest_test_metadata(name="MoE MXFP8 Blockwise MatMul FWD BWD E2E")
@pytest_marks(["moe_mxfp8", "moe", "blockwise_mm_fwd", "blockwise_mm_bwd", "mxfp8", "e2e"])
@pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN1, common_dataclasses.Platforms.TRN2])
@final
class TestMoeMxfp8FwdBwdE2E:
    def _run_test(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        weight_orientation: Optional[TensorOrientation] = None,
    ):
        if weight_orientation is None:
            kernel_inputs = build_e2e_inputs()
            kernel_entry = moe_mxfp8_fwd_bwd_e2e
            torch_ref = moe_mxfp8_fwd_bwd_e2e_torch_ref
        else:
            kernel_inputs = build_shared_weight_e2e_inputs(weight_orientation)
            kernel_entry = moe_mxfp8_fwd_bwd_shared_weights
            torch_ref = moe_mxfp8_fwd_bwd_shared_weights_torch_ref
        num_blocks = TOKENS // BLOCK_SIZE
        checkpoint_shape = (num_blocks, 2, BLOCK_SIZE, INTERMEDIATE)
        output_shapes = {
            "output_hidden_states": ((TOKENS, HIDDEN), bfloat16),
            "gate_up_proj_act_checkpoint_T": (checkpoint_shape, bfloat16),
            "hidden_states_grad": ((TOKENS, HIDDEN), bfloat16),
            "expert_affinities_masked_grad": ((TOKENS * EXPERTS, 1), bfloat16),
            "gate_up_proj_weight_grad": ((EXPERTS, HIDDEN, 2, INTERMEDIATE), bfloat16),
            "down_proj_weight_grad": ((EXPERTS, INTERMEDIATE, HIDDEN), bfloat16),
        }

        def input_gen(_):
            return dict(kernel_inputs)

        def output_tensors(_):
            return {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in output_shapes.items()}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_entry,
            torch_ref=torch_ref_wrapper(torch_ref),
            kernel_input_generator=input_gen,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=common_dataclasses.CompilerArgs(
                logical_nc_config=2,
                platform_target=platform_target,
            ),
            custom_comparator=make_moe_validated_comparator(output_shapes),
        )

    @pytest.mark.fast
    def test_default_checkpoint_chain(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
    ):
        self._run_test(test_manager, platform_target)

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "weight_orientation",
        [
            pytest.param(TensorOrientation.F_BY_K, id="forward_natural"),
            pytest.param(TensorOrientation.K_BY_F, id="backward_natural"),
        ],
    )
    def test_shared_bf16_weight_copy(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        weight_orientation: TensorOrientation,
    ):
        self._run_test(test_manager, platform_target, weight_orientation)
