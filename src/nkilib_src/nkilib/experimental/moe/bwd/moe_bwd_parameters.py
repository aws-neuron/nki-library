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

from ....core.utils.allocator import align_to, sizeinbytes
from ....core.utils.common_types import ActFnType
from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil


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
    Sharding strategies for blockwise kernels.

    Attributes:
        SHARD_ON_FREE: Shard across the free (output) dimension of each matmul.
        SHARD_ON_HIDDEN: Shard across hidden dimension for all functions.
        SHARD_ON_BLOCK: Partition whole token-blocks across cores — each core
            computes a disjoint subset of the N blocks end-to-end and writes its
            own output slab; the per-shard slabs are reduced at the end. Used by
            the MXFP8 MoE forward kernel (blockwise_mm_fwd_mxfp8).
    """

    SHARD_ON_FREE = 0
    SHARD_ON_HIDDEN = 1
    SHARD_ON_BLOCK = 2


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
        buffer_degree (int): Number of multi-buffer sections for interleaved execution.
    """

    block_h: int = 8
    buffer_degree: int = 4

    def estimate_sbuf_usage(
        self,
        B,
        H,
        I_TP,
        num_shards,
        dtype,
        shard_option=None,
        affinity_option=None,
        is_tensor_update_accumulating=True,
    ):
        """
        Estimate peak SBUF bytes for _compute_down_projection_output_grad.

        Computes persistent_bytes + buffer_degree * inner_section_bytes, mirroring
        the allocation pattern: small per-token buffers persist across the loop,
        while large H-blocked tiles cycle inside the interleave scope.

        Args:
            B (int): Block size.
            H (int): Hidden dimension.
            I_TP (int): Intermediate dimension (unused, kept for uniform interface).
            num_shards (int): Number of LNC shards.
            dtype: Compute data type.
            shard_option: Unused, kept for uniform interface.
            affinity_option: Unused, kept for uniform interface.
            is_tensor_update_accumulating (bool): Unused, kept for uniform interface.

        Returns:
            int: Estimated peak SBUF bytes.
        """
        TILE_SIZE = 128
        B_TILE_SIZE = min(TILE_SIZE, B)
        H_TILE_SIZE = TILE_SIZE
        H_BLOCK_SIZE = min(self.block_h * H_TILE_SIZE, H)
        elem = sizeinbytes(dtype)
        ALIGN = 32

        # Persistent: buffer_degree × 6 buffers of (B_TILE_SIZE, 1)
        # Each small allocation (2-4 bytes) gets 32-byte aligned by SbufManager,
        # so effective size per allocation is align_to(bytes, 32).
        # Types: ea_dtype(4) + fp32(4) + fp32(4) + dtype(elem) + int32(4) + int32(4)
        persistent_bytes = self.buffer_degree * (
            align_to(4, ALIGN)
            + align_to(4, ALIGN)
            + align_to(4, ALIGN)
            + align_to(elem, ALIGN)
            + align_to(4, ALIGN)
            + align_to(4, ALIGN)
        )

        # Inner section: 4 tiles of (B_TILE_SIZE, H_BLOCK_SIZE) + ea_grad_local (B_TILE_SIZE, 1)
        inner_section_bytes = 4 * H_BLOCK_SIZE * elem + align_to(4, ALIGN)

        return persistent_bytes + self.buffer_degree * inner_section_bytes


@dataclass
class GateUpOutputGradBlocking(nl.NKIObject):
    """
    Blocking parameters for compute_gate_up_projection_output_grad.

    Args:
        block_h (int): Block size for hidden dimension.
        block_b (int): Block size for batch dimension.
        block_i (int): Block size for intermediate dimension.
        buffer_degree (int): Number of multi-buffer sections for interleaved execution.
    """

    block_h: int = 8
    block_b: int = 2
    block_i: int = 2
    buffer_degree: int = 3  # not reduced for fp32: this pipelined tile's peak SBUF is non-monotonic in degree

    def estimate_sbuf_usage(
        self,
        B,
        H,
        I_TP,
        num_shards,
        dtype,
        shard_option=None,
        affinity_option=None,
        is_tensor_update_accumulating=True,
        grad_accum_dtype=None,
    ):
        """
        Estimate peak SBUF bytes for _compute_gate_up_projection_output_grad.

        Args:
            B (int): Block size.
            H (int): Hidden dimension.
            I_TP (int): Intermediate dimension.
            num_shards (int): Number of LNC shards.
            dtype: Compute data type.
            shard_option (ShardOption): Sharding strategy.
            affinity_option (AffinityOption): Affinity scaling dimension.
            is_tensor_update_accumulating (bool): Unused, kept for uniform interface.

        Returns:
            int: Estimated peak SBUF bytes.
        """
        TILE_SIZE = 128
        PSUM_SIZE = 512
        elem = sizeinbytes(dtype)
        # fp32-capable tiles (silu_dx_gate, silu_gate_grad, gate_silu_activation,
        # partial_block_gate_up_mult_grad, sendrecv_recv_buf) follow grad_accum_dtype.
        grad_elem = sizeinbytes(grad_accum_dtype) if grad_accum_dtype is not None else elem

        is_shard_on_h = shard_option == ShardOption.SHARD_ON_HIDDEN if shard_option is not None else False

        if is_shard_on_h:
            H_SHARDED = H // num_shards
            I_TP_SHARDED = I_TP
        else:
            H_SHARDED = H
            I_TP_SHARDED = I_TP // num_shards

        B_TILE_SIZE = min(TILE_SIZE, B)
        H_TILE_SIZE = min(TILE_SIZE, H_SHARDED)
        I_TP_TILE_SIZE = min(PSUM_SIZE, I_TP_SHARDED)

        H_BLOCK_SIZE = min(self.block_h * H_TILE_SIZE, H_SHARDED)
        I_TP_BLOCK_SIZE = min(self.block_i * I_TP_TILE_SIZE, I_TP_SHARDED)
        B_BLOCK_SIZE = min(self.block_b * B_TILE_SIZE, B)

        NUM_B_TILES = div_ceil(B_BLOCK_SIZE, B_TILE_SIZE)
        NUM_H_TILES = div_ceil(H_BLOCK_SIZE, H_TILE_SIZE)
        NUM_B_TILES_TOTAL = div_ceil(B, B_TILE_SIZE)

        # SBUF bytes = free dims only (exclude partition dim shape[0])
        # For (P, NUM_B_TILES, I_TP_BLOCK_SIZE): bytes = NUM_B_TILES * I_TP_BLOCK_SIZE * elem

        # 6 persistent tensors: (B_TILE, NUM_B_TILES, I_TP_BLOCK_SIZE)
        # 3 follow grad_accum_dtype (silu_dx_gate, silu_gate_grad, gate_silu_activation),
        # 3 stay compute_dtype (up_activation_grad, gate_activation_grad, block_gate_up_mult).
        base_persistent = (3 * grad_elem + 3 * elem) * NUM_B_TILES * I_TP_BLOCK_SIZE

        # buffer_degree × (3 large + 1 weight_temp); of the 3 large,
        # partial_block_gate_up_mult_grad follows grad_accum_dtype, the other 2 stay compute_dtype.
        buffered = self.buffer_degree * ((grad_elem + 2 * elem) * NUM_B_TILES * I_TP_BLOCK_SIZE + H_BLOCK_SIZE * elem)

        # Affinity I adds significant persistent buffers
        is_affinity_i = affinity_option == AffinityOption.AFFINITY_ON_I if affinity_option is not None else False
        affinity_i_bytes = 0
        if is_affinity_i:
            affinity_i_bytes = (
                self.buffer_degree * H_BLOCK_SIZE * elem  # grad_load_temp (P, H_BLOCK)
                + NUM_B_TILES_TOTAL * 4  # recv_buf (P, 1) fp32
                + NUM_B_TILES_TOTAL * elem  # ea_grad_reduced (P, 1)
                + NUM_B_TILES_TOTAL * 4  # ea_grad_accum (P, 1) fp32
                + I_TP_BLOCK_SIZE * 4  # ea_product_temp (P, I_TP_BLOCK) fp32
                + 1 * 4  # ea_reduce_temp (P, 1) fp32
                + I_TP_BLOCK_SIZE * elem  # scaled_gate_up_mult_tile (P, I_TP_BLOCK)
                + NUM_B_TILES_TOTAL * 4  # ea_tiles_all (P, NUM_B_TILES_TOTAL) fp32
                + NUM_B_TILES_TOTAL * 4  # ea_offsets_all (P, 1) int32 × NUM_B_TILES_TOTAL
                + NUM_B_TILES_TOTAL * 4  # addr_tmp (P, 1) int32 × NUM_B_TILES_TOTAL
                + NUM_B_TILES_TOTAL * 4  # ea_load (P, 1) fp32 × NUM_B_TILES_TOTAL
            )

        # Shard-on-H adds sendrecv buffer (P, b_tiles_per_core, I_TP_BLOCK_SIZE)
        shard_h_bytes = 0
        if is_shard_on_h:
            b_tiles_per_core = max(1, NUM_B_TILES // num_shards)
            shard_h_bytes = b_tiles_per_core * I_TP_BLOCK_SIZE * grad_elem

        persistent_bytes = base_persistent + buffered + affinity_i_bytes + shard_h_bytes

        # Inner section: weight_transposed (H_TILE, NUM_H_TILES, I_TP_BLOCK) + grad_transposed (H_TILE, NUM_H_TILES, B_BLOCK)
        inner_section_bytes = NUM_H_TILES * I_TP_BLOCK_SIZE * elem + NUM_H_TILES * B_BLOCK_SIZE * elem

        return persistent_bytes + self.buffer_degree * inner_section_bytes


@dataclass
class DownWeightGradBlocking(nl.NKIObject):
    """
    Blocking parameters for compute_down_projection_weight_grad.

    Args:
        block_h (int): Block size for hidden dimension.
        block_b (int): Block size for batch dimension.
        block_i (int): Block size for intermediate dimension.
        buffer_degree (int): Number of multi-buffer sections for interleaved execution.
    """

    block_h: int = 2  # bf16 / TOT
    block_b: int = 4
    block_i: int = 8
    buffer_degree: int = 3  # bf16 / TOT
    buffer_degree_fp32: int = 1  # fp32: weight-grad accumulators are 2x bytes -> less buffering
    block_h_fp32: int = 1  # fp32: halve block_h to fit the SBUF budget

    def get_buffer_degree(self, grad_accum_dtype):
        """Return the dtype-appropriate buffer degree (fp32 accumulators are 2x bytes)."""
        return self.buffer_degree_fp32 if grad_accum_dtype == nl.float32 else self.buffer_degree

    def set_buffer_degree(self, buffer_degree, buffer_degree_fp32=None):
        """Configure the bf16 (default) and optional fp32 buffer degrees."""
        self.buffer_degree = buffer_degree
        if buffer_degree_fp32 is not None:
            self.buffer_degree_fp32 = buffer_degree_fp32

    def get_block_h(self, grad_accum_dtype):
        """Return the dtype-appropriate block_h (fp32 accumulator is 2x bytes)."""
        return self.block_h_fp32 if grad_accum_dtype == nl.float32 else self.block_h

    def set_block_h(self, block_h, block_h_fp32=None):
        """Configure the bf16 (default) and optional fp32 block_h."""
        self.block_h = block_h
        if block_h_fp32 is not None:
            self.block_h_fp32 = block_h_fp32

    def estimate_sbuf_usage(
        self,
        B,
        H,
        I_TP,
        num_shards,
        dtype,
        shard_option=None,
        affinity_option=None,
        is_tensor_update_accumulating=True,
        grad_accum_dtype=None,
    ):
        """
        Estimate peak SBUF bytes for _compute_down_projection_weight_grad.

        Args:
            B (int): Block size.
            H (int): Hidden dimension.
            I_TP (int): Intermediate dimension.
            num_shards (int): Number of LNC shards.
            dtype: Compute data type.
            shard_option: Unused, kept for uniform interface.
            affinity_option: Unused, kept for uniform interface.
            is_tensor_update_accumulating (bool): Unused, kept for uniform interface.

        Returns:
            int: Estimated peak SBUF bytes.
        """
        TILE_SIZE = 128
        PSUM_SIZE = 512
        elem = sizeinbytes(dtype)

        H_SHARDED = H // num_shards
        B_TILE_SIZE = min(TILE_SIZE, B)
        H_TILE_SIZE = min(PSUM_SIZE, H_SHARDED)
        I_TP_TILE_SIZE = TILE_SIZE

        H_BLOCK_SIZE = min(self.get_block_h(grad_accum_dtype) * H_TILE_SIZE, H_SHARDED)
        I_TP_BLOCK_SIZE = min(self.block_i * I_TP_TILE_SIZE, I_TP)
        B_BLOCK_SIZE = min(self.block_b * B_TILE_SIZE, B)

        NUM_I_TP_TILES = div_ceil(I_TP_BLOCK_SIZE, I_TP_TILE_SIZE)
        NUM_B_TILES = div_ceil(B_BLOCK_SIZE, B_TILE_SIZE)

        # Persistent: buffer_degree × (result_tiles + existing_weight_grad)
        # Shape: (I_TP_TILE_SIZE, NUM_I_TP_TILES, H_BLOCK_SIZE) → free = NUM_I_TP_TILES * H_BLOCK_SIZE
        # Both accumulators follow the grad-output dtype (fp32 when caller preallocates fp32 grad HBM).
        grad_elem = sizeinbytes(grad_accum_dtype) if grad_accum_dtype is not None else elem
        persistent_bytes = 2 * self.get_buffer_degree(grad_accum_dtype) * NUM_I_TP_TILES * H_BLOCK_SIZE * grad_elem

        # Inner section: lhs_tiles (B_TILE, NUM_B_TILES, I_TP_BLOCK) + rhs_tiles (B_TILE, NUM_B_TILES, H_BLOCK)
        inner_section_bytes = NUM_B_TILES * (I_TP_BLOCK_SIZE + H_BLOCK_SIZE) * elem

        return persistent_bytes + self.get_buffer_degree(grad_accum_dtype) * inner_section_bytes


@dataclass
class HiddenGradBlocking(nl.NKIObject):
    """
    Blocking parameters for compute_hidden_states_grad.

    Args:
        block_h (int): Block size for hidden dimension.
        block_b (int): Block size for batch dimension.
        block_i (int): Block size for intermediate dimension.
        buffer_degree (int): Number of multi-buffer sections for interleaved execution.
    """

    block_h: int = 2
    block_b: int = 4
    block_i: int = 8
    buffer_degree: int = 3
    block_h_fp32: int = 4  # fp32: widen block_h so H fits in fewer H-blocks -> fewer grad-out scatter intervals
    buffer_degree_fp32: int = 2  # fp32: less buffering to offset the wider block_h and stay in SBUF

    def get_buffer_degree(self, grad_accum_dtype):
        """Return the dtype-appropriate buffer degree (fp32 accumulators are 2x bytes)."""
        return self.buffer_degree_fp32 if grad_accum_dtype == nl.float32 else self.buffer_degree

    def set_buffer_degree(self, buffer_degree, buffer_degree_fp32=None):
        """Configure the bf16 (default) and optional fp32 buffer degrees."""
        self.buffer_degree = buffer_degree
        if buffer_degree_fp32 is not None:
            self.buffer_degree_fp32 = buffer_degree_fp32

    def get_block_h(self, grad_accum_dtype):
        """Return the dtype-appropriate block_h (fp32 grad-out is 2x bytes)."""
        return self.block_h_fp32 if grad_accum_dtype == nl.float32 else self.block_h

    def set_block_h(self, block_h, block_h_fp32=None):
        """Configure the bf16 (default) and optional fp32 block_h."""
        self.block_h = block_h
        if block_h_fp32 is not None:
            self.block_h_fp32 = block_h_fp32

    def estimate_sbuf_usage(
        self,
        B,
        H,
        I_TP,
        num_shards,
        dtype,
        shard_option=None,
        affinity_option=None,
        is_tensor_update_accumulating=True,
        grad_accum_dtype=None,
    ):
        """
        Estimate peak SBUF bytes for _compute_hidden_states_grad.

        Args:
            B (int): Block size.
            H (int): Hidden dimension.
            I_TP (int): Intermediate dimension.
            num_shards (int): Number of LNC shards.
            dtype: Compute data type.
            shard_option: Unused, kept for uniform interface.
            affinity_option: Unused, kept for uniform interface.
            is_tensor_update_accumulating (bool): Whether accumulation buffers are allocated.

        Returns:
            int: Estimated peak SBUF bytes.
        """
        TILE_SIZE = 128
        PSUM_SIZE = 512
        elem = sizeinbytes(dtype)

        H_SHARDED = H // num_shards
        B_TILE_SIZE = min(TILE_SIZE, B)
        H_TILE_SIZE = min(PSUM_SIZE, H_SHARDED)
        I_TP_TILE_SIZE = min(TILE_SIZE, I_TP)

        _bh = self.get_block_h(grad_accum_dtype)
        _bd = self.get_buffer_degree(grad_accum_dtype)

        H_BLOCK_SIZE = min(_bh * H_TILE_SIZE, H_SHARDED)
        I_TP_BLOCK_SIZE = min(self.block_i * I_TP_TILE_SIZE, I_TP)
        B_BLOCK_SIZE = min(self.block_b * B_TILE_SIZE, B)

        NUM_B_TILES = div_ceil(B_BLOCK_SIZE, B_TILE_SIZE)
        NUM_I_TP_TILES = div_ceil(I_TP_BLOCK_SIZE, I_TP_TILE_SIZE)
        NUM_H_INNER_TILES = div_ceil(H_TILE_SIZE, TILE_SIZE)

        # Persistent: buffer_degree × (rhs_temp + result_tiles + optional existing_hidden_grad)
        # rhs_temp: (TILE_SIZE, NUM_H_INNER_TILES, I_TP_BLOCK_SIZE) → free = NUM_H_INNER_TILES * I_TP_BLOCK_SIZE
        # result_tiles: (B_TILE, NUM_B_TILES, H_BLOCK_SIZE) → free = NUM_B_TILES * H_BLOCK_SIZE
        # rhs_temp stays compute_dtype; the result/existing accumulators follow the grad-output dtype.
        grad_elem = sizeinbytes(grad_accum_dtype) if grad_accum_dtype is not None else elem
        result_count = 2 if is_tensor_update_accumulating else 1
        persistent_bytes = _bd * (
            NUM_H_INNER_TILES * I_TP_BLOCK_SIZE * elem + result_count * NUM_B_TILES * H_BLOCK_SIZE * grad_elem
        )

        # Inner section: lhs_tiles (I_TP_TILE, NUM_I_TP_TILES, B_BLOCK) + rhs_tiles (I_TP_TILE, NUM_I_TP_TILES, H_BLOCK)
        inner_section_bytes = NUM_I_TP_TILES * (B_BLOCK_SIZE + H_BLOCK_SIZE) * elem

        return persistent_bytes + _bd * inner_section_bytes


@dataclass
class GateUpWeightGradBlocking(nl.NKIObject):
    """
    Blocking parameters for compute_gate_up_projection_weight_grad.

    Args:
        block_h (int): Block size for hidden dimension.
        block_b (int): Block size for batch dimension.
        block_i (int): Block size for intermediate dimension.
        buffer_degree (int): Number of multi-buffer sections for interleaved execution.
    """

    block_h: int = 4
    block_b: int = 4
    block_i: int = 4
    buffer_degree: int = 3  # bf16 / TOT
    buffer_degree_fp32: int = 1  # fp32: weight-grad accumulators are 2x bytes -> less buffering

    def get_buffer_degree(self, grad_accum_dtype):
        """Return the dtype-appropriate buffer degree (fp32 accumulators are 2x bytes)."""
        return self.buffer_degree_fp32 if grad_accum_dtype == nl.float32 else self.buffer_degree

    def set_buffer_degree(self, buffer_degree, buffer_degree_fp32=None):
        """Configure the bf16 (default) and optional fp32 buffer degrees."""
        self.buffer_degree = buffer_degree
        if buffer_degree_fp32 is not None:
            self.buffer_degree_fp32 = buffer_degree_fp32

    def estimate_sbuf_usage(
        self,
        B,
        H,
        I_TP,
        num_shards,
        dtype,
        shard_option=None,
        affinity_option=None,
        is_tensor_update_accumulating=True,
        grad_accum_dtype=None,
    ):
        """
        Estimate peak SBUF bytes for _compute_gate_up_projection_weight_grad.

        Args:
            B (int): Block size.
            H (int): Hidden dimension.
            I_TP (int): Intermediate dimension.
            num_shards (int): Number of LNC shards.
            dtype: Compute data type.
            shard_option: Unused, kept for uniform interface.
            affinity_option: Unused, kept for uniform interface.
            is_tensor_update_accumulating (bool): Unused, kept for uniform interface.

        Returns:
            int: Estimated peak SBUF bytes.
        """
        TILE_SIZE = 128
        PSUM_SIZE = 512
        elem = sizeinbytes(dtype)

        H_SHARDED = H // num_shards
        B_TILE_SIZE = min(TILE_SIZE, B)
        H_TILE_SIZE = min(TILE_SIZE, H_SHARDED)
        I_TP_TILE_SIZE = min(PSUM_SIZE, I_TP)

        H_BLOCK_SIZE = min(self.block_h * H_TILE_SIZE, H_SHARDED)
        I_TP_BLOCK_SIZE = min(self.block_i * I_TP_TILE_SIZE, I_TP)
        B_BLOCK_SIZE = min(self.block_b * B_TILE_SIZE, B)

        NUM_H_TILES = div_ceil(H_BLOCK_SIZE, H_TILE_SIZE)
        NUM_B_TILES = div_ceil(B_BLOCK_SIZE, B_TILE_SIZE)

        # Persistent: buffer_degree × (weight_grad_accum + existing_weight_grad)
        # Shape: (H_TILE, NUM_H_TILES, I_TP_BLOCK_SIZE) → free = NUM_H_TILES * I_TP_BLOCK_SIZE
        # Both accumulators follow the grad-output dtype (fp32 when caller preallocates fp32 grad HBM).
        grad_elem = sizeinbytes(grad_accum_dtype) if grad_accum_dtype is not None else elem
        persistent_bytes = 2 * self.get_buffer_degree(grad_accum_dtype) * NUM_H_TILES * I_TP_BLOCK_SIZE * grad_elem

        # Inner section: gate_up_proj_output_grad (B_TILE, NUM_B_TILES, I_TP_BLOCK) + block_hidden_states (B_TILE, NUM_B_TILES, H_BLOCK)
        inner_section_bytes = NUM_B_TILES * (I_TP_BLOCK_SIZE + H_BLOCK_SIZE) * elem

        return persistent_bytes + self.get_buffer_degree(grad_accum_dtype) * inner_section_bytes


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
        if self.down_proj_output_grad == None:
            self.down_proj_output_grad = DownProjOutputGradBlocking()
        if self.gate_up_output_grad == None:
            self.gate_up_output_grad = GateUpOutputGradBlocking()
        if self.down_weight_grad == None:
            self.down_weight_grad = DownWeightGradBlocking()
        if self.hidden_grad == None:
            self.hidden_grad = HiddenGradBlocking()
        if self.gate_up_weight_grad == None:
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
    skip_grad_initialization: bool = False
    clamp_limits: ClampLimits = None
    activation_type: ActFnType = ActFnType.SiLU
    skip_gate_proj: bool = False
    blocking_params: MOEBwdDroplessBlockingParams = None
    affinity_option: AffinityOption = AffinityOption.AFFINITY_ON_H
    shard_option: ShardOption = ShardOption.SHARD_ON_FREE
    # Opt-in high-precision bias-grad accumulation. None = compute_dtype (baseline, unchanged).
    # Set to nl.float32 to accumulate bias gradients in fp32 (aligns with fp32 reference accumulation).
    accumulation_dtype: nki.dtype = None

    # Derived dimensions (computed in __post_init__)
    T: int = None
    H: int = None
    I_TP: int = None
    E: int = None
    N: int = None

    def __post_init__(self):
        """Initialize default values and derive dimensions from tensor shapes."""
        if self.skip_dma == None:
            self.skip_dma = SkipMode()
        if self.clamp_limits == None:
            self.clamp_limits = ClampLimits()
        if self.blocking_params == None:
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
        if self.affinity_option == AffinityOption.AFFINITY_ON_I:
            kernel_assert(
                self.down_proj_act_checkpoint == None,
                "down_proj_act_checkpoint must be None for AFFINITY_ON_I",
            )
        if self.shard_option == ShardOption.SHARD_ON_HIDDEN:
            kernel_assert(
                self.affinity_option == AffinityOption.AFFINITY_ON_I,
                "SHARD_ON_HIDDEN only supports AFFINITY_ON_I",
            )

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
        if self.shard_option != ShardOption.SHARD_ON_HIDDEN and sharded_i_tp % 32 != 0:
            kernel_assert(
                self.block_size == 128,
                f"I_TP dim when sharded by num_shards={num_shards} must be divisible by 32. "
                f"If not, block size must be 128 for DMA transpose. "
                f"Got sharded I_TP={sharded_i_tp}, block_size={self.block_size}",
            )

    def get_activation_ops(self):
        """
        Get the activation type for forward and backward pass dispatch.

        Returns:
            ActFnType: The activation function type. Callers use apply_activation()
            and apply_activation_dx() from kernel_helpers with this type.
        """
        kernel_assert(
            self.activation_type in (ActFnType.SiLU, ActFnType.Swish, ActFnType.SquaredReLU),
            f"moe dropless backward supports only SiLU, Swish, and SquaredReLU activations, got {self.activation_type}",
        )
        return self.activation_type
