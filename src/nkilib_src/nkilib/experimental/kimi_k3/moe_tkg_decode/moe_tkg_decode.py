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

"""Kimi-K3 routed-expert decode with serving-layout MXFP4 weights."""

import nki
import nki.isa as nisa
import nki.language as nl

from ....core.moe.moe_tkg.down_projection_mx import load_broadcast_down_weight_scale_bias
from ....core.moe.moe_tkg.gate_up_projection_mx import (
    _projection_matmul_mx,
    load_gate_up_weight_scale_bias,
)
from ....core.moe.moe_tkg.projection_mx_constants import (
    GATE_FUSED_IDX,
    MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM,
    UP_FUSED_IDX,
    _q_width,
)
from ....core.utils.common_types import MoELNCShardingStrategy
from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil

_PMAX = 128
_MX_SCALE_PARTITIONS = 16
_DOWN_TILE_H = 1024


@nki.jit
def moe_tkg_decode(
    latent: nl.NkiTensor,
    local_affinities: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    beta: float = 4.0,
    linear_beta: float = 25.0,
) -> nl.NkiTensor:
    """Compute Kimi-K3's unnormalized local routed-expert decode contribution.

    This LNC1 Trn3 kernel consumes the model's natural latent layout and serving
    MXFP4 weights directly. Gate/up projections and per-expert down outputs have
    BF16 boundaries. The two-sided SiTU activation, affinity scaling, and
    cross-expert accumulation use FP32.

    Args:
        latent: BF16 ``[T, D]`` latent activations, where ``1 <= T <= 128``.
        local_affinities: BF16 or FP32 ``[T, E_L]`` local affinities. Entries
            outside the selected local top-k must be zero.
        expert_gate_up_weights: Packed MXFP4
            ``[E_L, 128, 2, D/512, I]`` gate/up weights.
        expert_down_weights: Packed MXFP4
            ``[E_L, 128, I/512, D]`` down weights.
        expert_gate_up_weights_scale: UINT8 E8M0 scales
            ``[E_L, 16, 2, D/512, I]``.
        expert_down_weights_scale: UINT8 E8M0 scales
            ``[E_L, 16, I/512, D]``.
        beta: SiTU gate-branch soft-clipping factor.
        linear_beta: SiTU up-branch soft-clipping factor.

    Returns:
        BF16 ``[T, D]`` unnormalized local routed-expert contribution.
    """
    kernel_assert(
        nl.program_ndim() == 0 or (nl.program_ndim() == 1 and nl.num_programs(0) == 1),
        "moe_tkg_decode supports LNC1 only",
    )
    kernel_assert(nisa.get_nc_version() >= nisa.nc_version.gen4, "moe_tkg_decode requires Trn3 or newer")
    kernel_assert(len(latent.shape) == 2, f"Expected latent [T, D], got {latent.shape}")
    kernel_assert(len(local_affinities.shape) == 2, f"Expected local_affinities [T, E_L], got {local_affinities.shape}")
    kernel_assert(
        len(expert_gate_up_weights.shape) == 5,
        f"Expected expert_gate_up_weights rank 5, got {expert_gate_up_weights.shape}",
    )
    kernel_assert(
        len(expert_down_weights.shape) == 4,
        f"Expected expert_down_weights rank 4, got {expert_down_weights.shape}",
    )
    kernel_assert(latent.dtype == nl.bfloat16, f"Expected BF16 latent, got {latent.dtype}")
    kernel_assert(
        local_affinities.dtype == nl.bfloat16 or local_affinities.dtype == nl.float32,
        f"Expected BF16 or FP32 local_affinities, got {local_affinities.dtype}",
    )
    kernel_assert(
        expert_gate_up_weights.dtype == nl.float4_e2m1fn_x4,
        f"Expected MXFP4 gate/up weights, got {expert_gate_up_weights.dtype}",
    )
    kernel_assert(
        expert_down_weights.dtype == nl.float4_e2m1fn_x4,
        f"Expected MXFP4 down weights, got {expert_down_weights.dtype}",
    )
    kernel_assert(
        expert_gate_up_weights_scale.dtype == nl.uint8,
        f"Expected UINT8 gate/up MX scales, got {expert_gate_up_weights_scale.dtype}",
    )
    kernel_assert(
        expert_down_weights_scale.dtype == nl.uint8,
        f"Expected UINT8 down MX scales, got {expert_down_weights_scale.dtype}",
    )
    kernel_assert(beta > 0.0, f"Expected beta > 0, got {beta}")
    kernel_assert(linear_beta > 0.0, f"Expected linear_beta > 0, got {linear_beta}")

    T_physical, D = latent.shape
    E_L = expert_gate_up_weights.shape[0]
    I = expert_gate_up_weights.shape[-1]
    n_D512 = D // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
    n_I512 = I // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
    T = div_ceil(T_physical, _q_width) * _q_width

    kernel_assert(T_physical > 0 and T_physical <= _PMAX, f"Expected 1 <= T <= 128, got T={T_physical}")
    kernel_assert(
        D > 0 and D % MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM == 0,
        f"Expected D divisible by 512, got D={D}",
    )
    kernel_assert(
        I > 0 and I % MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM == 0,
        f"Expected I divisible by 512, got I={I}",
    )
    kernel_assert(E_L > 0, f"Expected at least one local expert, got E_L={E_L}")
    kernel_assert(
        local_affinities.shape == (T_physical, E_L),
        f"Expected local_affinities shape {(T_physical, E_L)}, got {local_affinities.shape}",
    )
    kernel_assert(
        expert_gate_up_weights.shape == (E_L, _PMAX, 2, n_D512, I),
        f"Unexpected gate/up weight shape {expert_gate_up_weights.shape}",
    )
    kernel_assert(
        expert_gate_up_weights_scale.shape == (E_L, _MX_SCALE_PARTITIONS, 2, n_D512, I),
        f"Unexpected gate/up scale shape {expert_gate_up_weights_scale.shape}",
    )
    kernel_assert(
        expert_down_weights.shape == (E_L, _PMAX, n_I512, D),
        f"Unexpected down weight shape {expert_down_weights.shape}",
    )
    kernel_assert(
        expert_down_weights_scale.shape == (E_L, _MX_SCALE_PARTITIONS, n_I512, D),
        f"Unexpected down scale shape {expert_down_weights_scale.shape}",
    )

    input_quant_sb, input_scale_sb = _quantize_natural_latent(latent=latent, T=T, T_physical=T_physical, D=D)
    output_accum_sb = nl.ndarray((T, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=output_accum_sb, value=0.0)

    affinities_flat = local_affinities.reshape((T_physical * E_L,))
    for expert_idx in nl.sequential_range(E_L):
        gate_weight_sb, gate_weight_scale_sb, _ = load_gate_up_weight_scale_bias(
            weight=expert_gate_up_weights,
            scale=expert_gate_up_weights_scale,
            bias=None,
            expert_idx=expert_idx,
            gate_or_up_idx=GATE_FUSED_IDX,
            H=D,
            n_I512_tiles_local=n_I512,
            I_local=I,
            I_offset=0,
        )
        gate_term_f32 = _project_to_fp32(
            input_quant_sb=input_quant_sb,
            input_scale_sb=input_scale_sb,
            weight_sb=gate_weight_sb,
            weight_scale_sb=gate_weight_scale_sb,
            n_H512=n_D512,
            n_I512=n_I512,
            I=I,
            T=T,
        )
        for tile_i in nl.sequential_range(n_I512):
            gate_sigmoid_f32 = nl.ndarray((_PMAX, T, _q_width), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                dst=gate_sigmoid_f32,
                op=nl.sigmoid,
                data=gate_term_f32[:, tile_i, :, :],
            )
            nisa.activation(
                dst=gate_term_f32[:, tile_i, :, :],
                op=nl.tanh,
                data=gate_term_f32[:, tile_i, :, :],
                scale=1.0 / beta,
            )
            nisa.scalar_tensor_tensor(
                dst=gate_term_f32[:, tile_i, :, :],
                data=gate_term_f32[:, tile_i, :, :],
                op0=nl.multiply,
                operand0=beta,
                op1=nl.multiply,
                operand1=gate_sigmoid_f32,
            )

        up_weight_sb, up_weight_scale_sb, _ = load_gate_up_weight_scale_bias(
            weight=expert_gate_up_weights,
            scale=expert_gate_up_weights_scale,
            bias=None,
            expert_idx=expert_idx,
            gate_or_up_idx=UP_FUSED_IDX,
            H=D,
            n_I512_tiles_local=n_I512,
            I_local=I,
            I_offset=0,
        )
        up_f32 = _project_to_fp32(
            input_quant_sb=input_quant_sb,
            input_scale_sb=input_scale_sb,
            weight_sb=up_weight_sb,
            weight_scale_sb=up_weight_scale_sb,
            n_H512=n_D512,
            n_I512=n_I512,
            I=I,
            T=T,
        )

        activation_bf16 = nl.ndarray((_PMAX, n_I512, T, _q_width), dtype=nl.bfloat16, buffer=nl.sbuf)
        for tile_i in nl.sequential_range(n_I512):
            nisa.activation(
                dst=up_f32[:, tile_i, :, :],
                op=nl.tanh,
                data=up_f32[:, tile_i, :, :],
                scale=1.0 / linear_beta,
            )
            nisa.scalar_tensor_tensor(
                dst=activation_bf16[:, tile_i, :, :],
                data=up_f32[:, tile_i, :, :],
                op0=nl.multiply,
                operand0=linear_beta,
                op1=nl.multiply,
                operand1=gate_term_f32[:, tile_i, :, :],
            )

        activation_quant_sb = nl.ndarray(
            (_PMAX, n_I512, T),
            dtype=nl.float8_e4m3fn_x4,
            buffer=nl.sbuf,
        )
        activation_scale_sb = nl.ndarray((_PMAX, n_I512, T), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.quantize_mx(
            src=activation_bf16,
            dst=activation_quant_sb,
            dst_scale=activation_scale_sb,
        )

        down_weight_sb, down_weight_scale_sb, _ = load_broadcast_down_weight_scale_bias(
            weight=expert_down_weights,
            scale=expert_down_weights_scale,
            bias=None,
            expert_idx=expert_idx,
            H=D,
            tile_I=_PMAX,
            n_I512_tiles=n_I512,
            tile_offset=0,
            tile_T=T,
            sharding_strategy=MoELNCShardingStrategy.NO_SHARD,
        )

        affinity_raw_sb = nl.ndarray((T, 1), dtype=local_affinities.dtype, buffer=nl.sbuf)
        affinity_f32_sb = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
        if T > T_physical:
            nisa.memset(dst=affinity_raw_sb, value=0.0)
        nisa.dma_copy(
            dst=affinity_raw_sb[:T_physical, :],
            src=affinities_flat.ap(
                pattern=[[E_L, T_physical], [1, 1]],
                offset=expert_idx,
            ),
        )
        nisa.tensor_copy(dst=affinity_f32_sb, src=affinity_raw_sb)

        tile_H = min(D, _DOWN_TILE_H)
        n_H_tiles = div_ceil(D, tile_H)
        for tile_h in nl.sequential_range(n_H_tiles):
            tile_H_offset = tile_h * tile_H
            tile_H_actual = min(tile_H, D - tile_H_offset)
            H_slice = nl.ds(tile_H_offset, tile_H_actual)
            expert_out_psum = nl.ndarray((T, tile_H_actual), dtype=nl.bfloat16, buffer=nl.psum)
            for tile_i in nl.sequential_range(n_I512):
                nisa.nc_matmul_mx(
                    dst=expert_out_psum,
                    stationary=activation_quant_sb[:, tile_i, :],
                    moving=down_weight_sb[:, tile_i, H_slice],
                    stationary_scale=activation_scale_sb[:, tile_i, :],
                    moving_scale=down_weight_scale_sb[:, tile_i, H_slice],
                )

            expert_out_f32 = nl.ndarray((T, tile_H_actual), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=expert_out_f32, src=expert_out_psum)
            nisa.scalar_tensor_tensor(
                dst=output_accum_sb[:, H_slice],
                data=expert_out_f32,
                op0=nl.multiply,
                operand0=affinity_f32_sb,
                op1=nl.add,
                operand1=output_accum_sb[:, H_slice],
            )

    output = nl.ndarray((T_physical, D), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    tile_H = min(D, _DOWN_TILE_H)
    n_H_tiles = div_ceil(D, tile_H)
    for tile_h in nl.sequential_range(n_H_tiles):
        tile_H_offset = tile_h * tile_H
        tile_H_actual = min(tile_H, D - tile_H_offset)
        H_slice = nl.ds(tile_H_offset, tile_H_actual)
        output_bf16_sb = nl.ndarray((T, tile_H_actual), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=output_bf16_sb, src=output_accum_sb[:, H_slice])
        nisa.dma_copy(
            dst=output[:, H_slice],
            src=output_bf16_sb[:T_physical, :],
        )
    return output


def _quantize_natural_latent(
    latent: nl.NkiTensor,
    T: int,
    T_physical: int,
    D: int,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Swizzle natural ``[T,D]`` input and quantize it to MXFP8."""
    n_D512 = D // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
    latent_flat = latent.reshape((T_physical * D,))
    input_swizzled_sb = nl.ndarray((_PMAX, n_D512, T, _q_width), dtype=nl.bfloat16, buffer=nl.sbuf)
    if T > T_physical:
        nisa.memset(dst=input_swizzled_sb, value=0.0)
    for tile_h in nl.affine_range(n_D512):
        for q_idx in nl.affine_range(_q_width):
            nisa.dma_copy(
                dst=input_swizzled_sb[:, tile_h, :T_physical, q_idx],
                src=latent_flat.ap(
                    pattern=[[_q_width, _PMAX], [D, T_physical]],
                    offset=tile_h * MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM + q_idx,
                ),
            )

    input_quant_sb = nl.ndarray((_PMAX, n_D512, T), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    input_scale_sb = nl.ndarray((_PMAX, n_D512, T), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.quantize_mx(
        src=input_swizzled_sb,
        dst=input_quant_sb,
        dst_scale=input_scale_sb,
    )
    return input_quant_sb, input_scale_sb


def _project_to_fp32(
    input_quant_sb: nl.NkiTensor,
    input_scale_sb: nl.NkiTensor,
    weight_sb: nl.NkiTensor,
    weight_scale_sb: nl.NkiTensor,
    n_H512: int,
    n_I512: int,
    I: int,
    T: int,
) -> nl.NkiTensor:
    """Run one MX projection and expose its BF16 PSUM boundary as FP32."""
    projection_psums = []
    for _ in range(n_I512):
        projection_psums.append(nl.ndarray((_PMAX, _q_width, T), dtype=nl.bfloat16, buffer=nl.psum))

    _projection_matmul_mx(
        out_psum_lst=projection_psums,
        weight_sb=weight_sb,
        weight_scale_sb=weight_scale_sb,
        input_quant_sb=input_quant_sb,
        input_scale_sb=input_scale_sb,
        tile_T_slice=nl.ds(0, T),
        tile_T_actual=T,
        n_H512_tiles=n_H512,
        n_total_I512_tiles=n_I512,
        I_local=I,
    )

    projection_f32 = nl.ndarray((_PMAX, n_I512, T, _q_width), dtype=nl.float32, buffer=nl.sbuf)
    for tile_i in nl.sequential_range(n_I512):
        nisa.tensor_copy(
            dst=projection_f32[:, tile_i, :, :],
            src=projection_psums[tile_i].ap(
                pattern=[[_q_width * T, _PMAX], [1, T], [T, _q_width]],
            ),
        )
    return projection_f32
