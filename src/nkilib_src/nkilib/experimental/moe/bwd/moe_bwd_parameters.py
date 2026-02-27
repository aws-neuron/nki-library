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

"""Parameter classes for MoE (Mixture of Experts) backward pass kernels."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import nki
import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert


@dataclass(frozen=True)
class SkipMode(nl.NKIObject):
    """
    Configuration for skipping DMA operations during OOB handling.

    Args:
        skip_token (bool): Skip token-related DMA operations.
        skip_weight (bool): Skip weight-related DMA operations.
    """

    skip_token: bool = False
    skip_weight: bool = False


class ActFnType(Enum):
    """
    Activation function types for MoE layers.

    Attributes:
        SiLU: Sigmoid Linear Unit activation.
        GELU: Gaussian Error Linear Unit activation.
        GELU_Tanh_Approx: GELU with tanh approximation.
        Swish: Swish activation (sigmoid-weighted linear unit).
    """

    SiLU = 0
    GELU = 1
    GELU_Tanh_Approx = 2
    Swish = 3


@dataclass(frozen=True)
class ClampLimits(nl.NKIObject):
    """
    Gradient clamping limits for numerical stability.

    Args:
        linear_clamp_upper_limit (float): Upper clamp limit for linear operations.
        linear_clamp_lower_limit (float): Lower clamp limit for linear operations.
        non_linear_clamp_upper_limit (float): Upper clamp limit for non-linear operations.
        non_linear_clamp_lower_limit (float): Lower clamp limit for non-linear operations.
    """

    linear_clamp_upper_limit: float = None
    linear_clamp_lower_limit: float = None
    non_linear_clamp_upper_limit: float = None
    non_linear_clamp_lower_limit: float = None

    def __repr__(self):
        return (
            f"linear_{self.linear_clamp_lower_limit}_{self.linear_clamp_upper_limit}_"
            f"non_linear_{self.non_linear_clamp_lower_limit}_{self.non_linear_clamp_upper_limit}"
        )


class ShardOption(Enum):
    """
    Sharding strategies for blockwise backward kernel.

    Attributes:
        AUTO: Automatically select best implementation based on dimensions.
        SHARD_ON_HIDDEN: Shard across hidden dimension (simpler, no H-tiling).
        SHARD_ON_INTERMEDIATE: Shard across intermediate dimension (better memory).
        BASELINE_LNC1: Single-core baseline implementation.
    """

    AUTO = 0
    SHARD_ON_HIDDEN = 1
    SHARD_ON_INTERMEDIATE = 2
    BASELINE_LNC1 = 3


class AffinityOption(Enum):
    """
    Affinity scaling dimension options.

    Attributes:
        AFFINITY_ON_H: Scale affinity on hidden dimension.
        AFFINITY_ON_I: Scale affinity on intermediate dimension.
    """

    AFFINITY_ON_H = 0
    AFFINITY_ON_I = 1


class KernelTypeOption(Enum):
    """
    Token dropping strategy options.

    Attributes:
        DROPPING: Dropping kernel with number of blocks = number of experts.
        DROPLESS: Dropless kernel with variable number of blocks per expert.
    """

    DROPPING = 0
    DROPLESS = 1


@dataclass
class DownProjOutputGradBlocking(nl.NKIObject):
    """
    Blocking parameters for compute_down_projection_output_grad.

    Args:
        block_h (int): Block size for hidden dimension.
    """

    block_h: int = 8


@dataclass
class GateUpOutputGradBlocking(nl.NKIObject):
    """
    Blocking parameters for compute_gate_up_projection_output_grad.

    Args:
        block_h (int): Block size for hidden dimension.
        block_b (int): Block size for batch dimension.
        block_i (int): Block size for intermediate dimension.
    """

    block_h: int = 8
    block_b: int = 2
    block_i: int = 2


@dataclass
class DownWeightGradBlocking(nl.NKIObject):
    """
    Blocking parameters for compute_down_projection_weight_grad.

    Args:
        block_h (int): Block size for hidden dimension.
        block_b (int): Block size for batch dimension.
        block_i (int): Block size for intermediate dimension.
    """

    block_h: int = 2
    block_b: int = 4
    block_i: int = 8


@dataclass
class HiddenGradBlocking(nl.NKIObject):
    """
    Blocking parameters for compute_hidden_states_grad.

    Args:
        block_h (int): Block size for hidden dimension.
        block_b (int): Block size for batch dimension.
        block_i (int): Block size for intermediate dimension.
    """

    block_h: int = 2
    block_b: int = 4
    block_i: int = 8


@dataclass
class GateUpWeightGradBlocking(nl.NKIObject):
    """
    Blocking parameters for compute_gate_up_projection_weight_grad.

    Args:
        block_h (int): Block size for hidden dimension.
        block_b (int): Block size for batch dimension.
        block_i (int): Block size for intermediate dimension.
    """

    block_h: int = 4
    block_b: int = 4
    block_i: int = 4


@dataclass
class MOEBwdDroplessBlockingParams(nl.NKIObject):
    """
    Blocking hyperparameters for all MoE backward pass compute functions.

    Args:
        down_proj_output_grad (DownProjOutputGradBlocking): Blocking for down projection output grad.
        gate_up_output_grad (GateUpOutputGradBlocking): Blocking for gate/up output grad.
        down_weight_grad (DownWeightGradBlocking): Blocking for down weight grad.
        hidden_grad (HiddenGradBlocking): Blocking for hidden states grad.
        gate_up_weight_grad (GateUpWeightGradBlocking): Blocking for gate/up weight grad.
    """

    down_proj_output_grad: DownProjOutputGradBlocking = None
    gate_up_output_grad: GateUpOutputGradBlocking = None
    down_weight_grad: DownWeightGradBlocking = None
    hidden_grad: HiddenGradBlocking = None
    gate_up_weight_grad: GateUpWeightGradBlocking = None

    def __post_init__(self):
        if self.down_proj_output_grad is None:
            self.down_proj_output_grad = DownProjOutputGradBlocking()
        if self.gate_up_output_grad is None:
            self.gate_up_output_grad = GateUpOutputGradBlocking()
        if self.down_weight_grad is None:
            self.down_weight_grad = DownWeightGradBlocking()
        if self.hidden_grad is None:
            self.hidden_grad = HiddenGradBlocking()
        if self.gate_up_weight_grad is None:
            self.gate_up_weight_grad = GateUpWeightGradBlocking()


@dataclass
class MOEBwdParameters(nl.NKIObject):
    """
    Parameters for blockwise MoE backward pass kernel.

    Groups all input tensors, output gradient tensors, and configuration options
    for the dropless MoE backward kernel.

    Dimensions:
        T: Total number of input tokens
        H: Hidden dimension size
        I_TP: Intermediate size / tensor parallel degree
        E: Number of experts
        B: Block size (tokens per block)
        N: Number of blocks

    Args:
        hidden_states (nl.ndarray): [T, H], Input hidden states.
        hidden_states_grad (nl.ndarray): [T, H], Output gradient for hidden states.
        expert_affinities_masked (nl.ndarray): [T * E, 1], Expert affinities.
        expert_affinities_masked_grad (nl.ndarray): [T * E, 1], Output gradient for affinities.
        gate_up_proj_weight (nl.ndarray): [E, H, 2, I_TP], Gate/up projection weights.
        gate_up_proj_weight_grad (nl.ndarray): [E, H, 2, I_TP], Output gradient for gate/up weights.
        gate_up_proj_act_checkpoint_T (nl.ndarray): [N, 2, I_TP, B], Checkpointed activations.
        down_proj_weight (nl.ndarray): [E, I_TP, H], Down projection weights.
        down_proj_weight_grad (nl.ndarray): [E, I_TP, H], Output gradient for down weights.
        down_proj_act_checkpoint (nl.ndarray): [N, B, H], Checkpointed down activations.
        token_position_to_id (nl.ndarray): [N * B], Token position mapping.
        block_to_expert (nl.ndarray): [N, 1], Expert index per block.
        output_hidden_states_grad (nl.ndarray): [T, H], Upstream gradient.
        block_size (int): Tokens per block (128, 256, 512, or 1024).
        skip_dma (SkipMode): OOB handling mode.
        compute_dtype (nki.dtype): Computation dtype (default: nl.bfloat16).
        is_tensor_update_accumulating (bool): Accumulate into existing gradients.
        clamp_limits (ClampLimits): Gradient clamping limits.
        gate_and_up_proj_bias_grad (nl.ndarray, optional): [E, 2, I_TP], Bias gradients.
        down_proj_bias_grad (nl.ndarray, optional): [E, H], Down bias gradients.
        activation_type (ActFnType): Activation function type.
        blocking_params (MOEBwdDroplessBlockingParams): Blocking hyperparameters.

    Notes:
        - block_size must be one of: 128, 256, 512, 1024.
        - H must be divisible by num_shards for LNC sharding.
        - Derived dimensions (T, H, I_TP, E, N) are computed in __post_init__.
    """

    # Input tensors
    hidden_states: nl.ndarray
    expert_affinities_masked: nl.ndarray
    gate_up_proj_weight: nl.ndarray
    gate_up_proj_act_checkpoint_T: nl.ndarray
    down_proj_weight: nl.ndarray
    token_position_to_id: nl.ndarray
    block_to_expert: nl.ndarray
    output_hidden_states_grad: nl.ndarray

    # Output gradient tensors
    hidden_states_grad: nl.ndarray
    expert_affinities_masked_grad: nl.ndarray
    gate_up_proj_weight_grad: nl.ndarray
    down_proj_weight_grad: nl.ndarray

    # Optional bias gradients
    gate_and_up_proj_bias_grad: Optional[nl.ndarray] = None
    down_proj_bias_grad: Optional[nl.ndarray] = None

    # Optional Down Projection Activation Checkpoint
    down_proj_act_checkpoint: Optional[nl.ndarray] = None

    # Configuration
    block_size: int = 512
    skip_dma: SkipMode = None
    compute_dtype: nki.dtype = nl.bfloat16
    is_tensor_update_accumulating: bool = True
    clamp_limits: ClampLimits = None
    activation_type: ActFnType = ActFnType.SiLU
    blocking_params: MOEBwdDroplessBlockingParams = None

    # Derived dimensions (computed in __post_init__)
    T: int = None
    H: int = None
    I_TP: int = None
    E: int = None
    N: int = None

    def __post_init__(self):
        """Initialize default values and derive dimensions from tensor shapes."""
        if self.skip_dma is None:
            self.skip_dma = SkipMode()
        if self.clamp_limits is None:
            self.clamp_limits = ClampLimits()
        if self.blocking_params is None:
            self.blocking_params = MOEBwdDroplessBlockingParams()

        # Derive dimensions from tensor shapes
        self.T = self.hidden_states.shape[0]
        self.H = self.hidden_states.shape[1]
        self.E = self.down_proj_weight.shape[0]
        self.I_TP = self.down_proj_weight.shape[1]
        self.N = self.token_position_to_id.shape[0] // self.block_size

    def validate(self):
        """
        Validate parameter constraints.

        Raises:
            AssertionError: If any validation check fails.
        """
        kernel_assert(
            self.block_size in (128, 256, 512, 1024),
            f"block_size must be 128, 256, 512, or 1024, got {self.block_size}",
        )
        kernel_assert(self.I_TP % 2 == 0, f"I_TP must be divisible by 2, got {self.I_TP}")
        kernel_assert(self.H % 2 == 0, f"H must be divisible by 2, got {self.H}")

    def validate_sharding(self, num_shards: int):
        """
        Validate sharding constraints.

        Args:
            num_shards (int): Number of shards for LNC sharding.

        Raises:
            AssertionError: If sharding constraints are violated.
        """
        kernel_assert(
            self.H % num_shards == 0,
            f"Hidden dim H={self.H} must be divisible by num_shards={num_shards} to initialize the gradients",
        )

        sharded_h = self.H // num_shards
        kernel_assert(
            sharded_h % 32 == 0,
            f"H dim when sharded by num_shards={num_shards} must be divisible by 32 for DMA transpose, "
            f"got sharded H as {sharded_h}",
        )

        sharded_i_tp = self.I_TP // num_shards
        kernel_assert(
            sharded_i_tp % 32 == 0,
            f"I_TP dim when sharded by num_shards={num_shards} must be divisible by 32 for DMA transpose, "
            f"got sharded I_TP as {sharded_i_tp}",
        )

    def get_activation_ops(self):
        """
        Get forward and backward activation functions based on activation_type.

        Returns:
            tuple: (forward_fn, backward_fn) activation function pair.
        """
        if self.activation_type == ActFnType.SiLU:
            return nl.silu, nl.silu_dx
        elif self.activation_type == ActFnType.Swish:
            return nl.gelu_apprx_sigmoid, nl.gelu_apprx_sigmoid_dx
        else:
            return nl.silu, nl.silu_dx
