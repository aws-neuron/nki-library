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

"""Latent-MoE routed-expert prefill with packed MXFP4 weights and exact SiTU."""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.compiler._builtins import get_backend
from nki.experimental._sync_type import sync_type
from nki.language.tensor import tensor_ap_view

try:
    from nki.experimental._dequantize import _make_simulate_fn
except ImportError:  # older nki releases exposed the module without the underscore
    from nki.experimental.dequantize import _make_simulate_fn

from ....core.utils.kernel_assert import kernel_assert

_PMAX = 128
_MATMUL_K = 128
_MX_PACK_WIDTH = 4
_MX_GROUP_SIZE = 32
_MX_TILE = 512
_MX_PACKED_TILE = _MX_TILE // _MX_PACK_WIDTH
_MX_SCALE_ROWS = _MX_TILE // _MX_GROUP_SIZE
_I_LAYOUT_TILES_PER_I512 = _MX_PACK_WIDTH


@nki.jit
def latent_moe_mx_prefill(
    latent: nl.NkiTensor,
    local_affinities: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    beta: float = 4.0,
    linear_beta: float = 25.0,
) -> nl.NkiTensor:
    """Compute unnormalized local routed-expert contributions for Kimi-K3 prefill.

    The kernel consumes the model's serving MXFP4 layout directly. Each 512-wide
    weight tile is unpacked and scaled only in SBUF, used by BF16 matmuls, and
    discarded. No dense weight tensor is materialized in HBM.

    Gate, up, activation, per-expert output, and the returned tensor are BF16.
    SiTU intermediates, projection PSUMs, affinity scaling, and cross-expert
    accumulation are FP32.

    Args:
        latent: BF16 ``[T, D]`` latent activations. ``T <= 128``.
        local_affinities: BF16 or FP32 ``[T, E_L]`` local expert affinities.
            Unselected and remote experts must have exactly-zero entries.
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
        "latent_moe_mx_prefill supports LNC1 only",
    )
    kernel_assert(nisa.get_nc_version() >= nisa.nc_version.gen4, "latent_moe_mx_prefill requires Trn3 or newer")
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

    T, D = latent.shape
    E_L = expert_gate_up_weights.shape[0]
    I = expert_gate_up_weights.shape[-1]
    n_D512 = D // _MX_TILE
    n_D128 = D // _MATMUL_K
    n_I512 = I // _MX_TILE
    n_I_layout_tiles = I // _PMAX

    kernel_assert(T > 0 and T <= _PMAX, f"Expected 1 <= T <= 128, got T={T}")
    kernel_assert(D > 0 and D % _MX_TILE == 0, f"Expected D divisible by 512, got D={D}")
    kernel_assert(I > 0 and I % _MX_TILE == 0, f"Expected I divisible by 512, got I={I}")
    kernel_assert(E_L > 0, f"Expected at least one local expert, got E_L={E_L}")
    kernel_assert(
        local_affinities.shape == (T, E_L),
        f"Expected local_affinities shape {(T, E_L)}, got {local_affinities.shape}",
    )
    kernel_assert(
        expert_gate_up_weights.shape == (E_L, _MX_PACKED_TILE, 2, n_D512, I),
        f"Unexpected gate/up weight shape {expert_gate_up_weights.shape}",
    )
    kernel_assert(
        expert_gate_up_weights_scale.shape == (E_L, _MX_SCALE_ROWS, 2, n_D512, I),
        f"Unexpected gate/up scale shape {expert_gate_up_weights_scale.shape}",
    )
    kernel_assert(
        expert_down_weights.shape == (E_L, _MX_PACKED_TILE, n_I512, D),
        f"Unexpected down weight shape {expert_down_weights.shape}",
    )
    kernel_assert(
        expert_down_weights_scale.shape == (E_L, _MX_SCALE_ROWS, n_I512, D),
        f"Unexpected down scale shape {expert_down_weights_scale.shape}",
    )

    gate_up_weights_flat = expert_gate_up_weights.reshape((E_L * _MX_PACKED_TILE * 2 * n_D512 * I,))
    gate_up_scales_flat = expert_gate_up_weights_scale.reshape((E_L * _MX_SCALE_ROWS * 2 * n_D512 * I,))
    down_weights_flat = expert_down_weights.reshape((E_L * _MX_PACKED_TILE * n_I512 * D,))
    down_scales_flat = expert_down_weights_scale.reshape((E_L * _MX_SCALE_ROWS * n_I512 * D,))
    latent_flat = latent.reshape((T * D,))
    affinities_flat = local_affinities.reshape((T * E_L,))

    latent_tiles = nl.ndarray((_MATMUL_K, n_D128, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    for d128_idx in nl.affine_range(n_D128):
        nisa.dma_copy(
            dst=latent_tiles[:, d128_idx, :],
            src=latent_flat.ap(
                pattern=[[1, _MATMUL_K], [D, T]],
                offset=d128_idx * _MATMUL_K,
            ),
        )

    dequant_indices = _build_dequant_indices()
    activation = nl.ndarray((_PMAX, n_I_layout_tiles, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    output_accum = nl.ndarray((_PMAX, n_D128, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=output_accum, value=0.0)

    for expert_idx in nl.sequential_range(E_L):
        for i_layout_idx in nl.sequential_range(n_I_layout_tiles):
            gate_bf16 = _gate_up_projection(
                gate_up_weights_flat=gate_up_weights_flat,
                gate_up_scales_flat=gate_up_scales_flat,
                latent_tiles=latent_tiles,
                dequant_indices=dequant_indices,
                expert_idx=expert_idx,
                branch_idx=0,
                i_layout_idx=i_layout_idx,
                n_D512=n_D512,
                I=I,
                T=T,
            )
            up_bf16 = _gate_up_projection(
                gate_up_weights_flat=gate_up_weights_flat,
                gate_up_scales_flat=gate_up_scales_flat,
                latent_tiles=latent_tiles,
                dequant_indices=dequant_indices,
                expert_idx=expert_idx,
                branch_idx=1,
                i_layout_idx=i_layout_idx,
                n_D512=n_D512,
                I=I,
                T=T,
            )

            gate_f32 = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
            up_f32 = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
            gate_sigmoid = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gate_f32, src=gate_bf16)
            nisa.tensor_copy(dst=up_f32, src=up_bf16)
            nisa.activation(dst=gate_sigmoid, op=nl.sigmoid, data=gate_f32)
            nisa.activation(dst=gate_f32, op=nl.tanh, data=gate_f32, scale=1.0 / beta)
            nisa.scalar_tensor_tensor(
                dst=gate_f32,
                data=gate_f32,
                op0=nl.multiply,
                operand0=beta,
                op1=nl.multiply,
                operand1=gate_sigmoid,
            )
            nisa.activation(dst=up_f32, op=nl.tanh, data=up_f32, scale=1.0 / linear_beta)
            nisa.scalar_tensor_tensor(
                dst=activation[:, i_layout_idx, :],
                data=up_f32,
                op0=nl.multiply,
                operand0=linear_beta,
                op1=nl.multiply,
                operand1=gate_f32,
            )

        affinity_raw = nl.ndarray((_PMAX, T), dtype=local_affinities.dtype, buffer=nl.sbuf)
        affinity_f32 = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=affinity_raw,
            src=affinities_flat.ap(
                pattern=[[0, _PMAX], [E_L, T]],
                offset=expert_idx,
            ),
        )
        nisa.tensor_copy(dst=affinity_f32, src=affinity_raw)

        for d128_idx in nl.sequential_range(n_D128):
            expert_out_bf16 = _down_projection(
                down_weights_flat=down_weights_flat,
                down_scales_flat=down_scales_flat,
                activation=activation,
                dequant_indices=dequant_indices,
                expert_idx=expert_idx,
                d128_idx=d128_idx,
                n_I512=n_I512,
                D=D,
                T=T,
            )
            expert_out_f32 = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
            weighted = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=expert_out_f32, src=expert_out_bf16)
            nisa.tensor_tensor(
                dst=weighted,
                data1=expert_out_f32,
                data2=affinity_f32,
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=output_accum[:, d128_idx, :],
                data1=output_accum[:, d128_idx, :],
                data2=weighted,
                op=nl.add,
            )

    output = nl.ndarray((T, D), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for d128_idx in nl.sequential_range(n_D128):
        output_dt = nl.ndarray((_PMAX, T), dtype=nl.bfloat16, buffer=nl.sbuf)
        output_td_psum = nl.ndarray((T, _PMAX), dtype=nl.bfloat16, buffer=nl.psum)
        output_td = nl.ndarray((T, _PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=output_dt, src=output_accum[:, d128_idx, :])
        nisa.nc_transpose(dst=output_td_psum, data=output_dt)
        nisa.tensor_copy(dst=output_td, src=output_td_psum)
        nisa.dma_copy(
            dst=output[:, nl.ds(d128_idx * _MATMUL_K, _MATMUL_K)],
            src=output_td,
        )
    return output


def _gate_up_projection(
    gate_up_weights_flat: nl.NkiTensor,
    gate_up_scales_flat: nl.NkiTensor,
    latent_tiles: nl.NkiTensor,
    dequant_indices: nl.NkiTensor,
    expert_idx: int,
    branch_idx: int,
    i_layout_idx: int,
    n_D512: int,
    I: int,
    T: int,
) -> nl.NkiTensor:
    """Project one 128-row gate/up output tile and round it to BF16."""
    projection_psum = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.psum)
    weight_transposed_psum = nl.ndarray((_MATMUL_K, _PMAX), dtype=nl.bfloat16, buffer=nl.psum)
    weight_matmul = nl.ndarray((_MATMUL_K, _PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)

    weight = _load_gate_up_weight_tile(
        gate_up_weights_flat=gate_up_weights_flat,
        gate_up_scales_flat=gate_up_scales_flat,
        dequant_indices=dequant_indices,
        expert_idx=expert_idx,
        branch_idx=branch_idx,
        d512_idx=0,
        i_layout_idx=i_layout_idx,
        n_D512=n_D512,
        I=I,
    )
    for d128_sub_idx in range(_MX_TILE // _MATMUL_K):
        nisa.nc_transpose(
            dst=weight_transposed_psum,
            data=weight[:, nl.ds(d128_sub_idx * _MATMUL_K, _MATMUL_K)],
        )
        nisa.tensor_copy(dst=weight_matmul, src=weight_transposed_psum)
        nisa.nc_matmul(
            dst=projection_psum,
            stationary=weight_matmul,
            moving=latent_tiles[:, d128_sub_idx, :],
            accumulate=d128_sub_idx != 0,
        )

    for d512_idx in nl.sequential_range(1, n_D512):
        weight = _load_gate_up_weight_tile(
            gate_up_weights_flat=gate_up_weights_flat,
            gate_up_scales_flat=gate_up_scales_flat,
            dequant_indices=dequant_indices,
            expert_idx=expert_idx,
            branch_idx=branch_idx,
            d512_idx=d512_idx,
            i_layout_idx=i_layout_idx,
            n_D512=n_D512,
            I=I,
        )
        for d128_sub_idx in range(_MX_TILE // _MATMUL_K):
            nisa.nc_transpose(
                dst=weight_transposed_psum,
                data=weight[:, nl.ds(d128_sub_idx * _MATMUL_K, _MATMUL_K)],
            )
            nisa.tensor_copy(dst=weight_matmul, src=weight_transposed_psum)
            nisa.nc_matmul(
                dst=projection_psum,
                stationary=weight_matmul,
                moving=latent_tiles[:, d512_idx * (_MX_TILE // _MATMUL_K) + d128_sub_idx, :],
                accumulate=True,
            )

    projection_bf16 = nl.ndarray((_PMAX, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=projection_bf16, src=projection_psum)
    return projection_bf16


def _down_projection(
    down_weights_flat: nl.NkiTensor,
    down_scales_flat: nl.NkiTensor,
    activation: nl.NkiTensor,
    dequant_indices: nl.NkiTensor,
    expert_idx: int,
    d128_idx: int,
    n_I512: int,
    D: int,
    T: int,
) -> nl.NkiTensor:
    """Project one 128-column down output tile and round it to BF16."""
    projection_psum = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.psum)
    weight_q = nl.ndarray((_PMAX, _MATMUL_K), dtype=nl.bfloat16, buffer=nl.sbuf)
    weight_transposed_psum = nl.ndarray((_MATMUL_K, _PMAX), dtype=nl.bfloat16, buffer=nl.psum)
    weight_matmul = nl.ndarray((_MATMUL_K, _PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)

    weight = _load_down_weight_tile(
        down_weights_flat=down_weights_flat,
        down_scales_flat=down_scales_flat,
        dequant_indices=dequant_indices,
        expert_idx=expert_idx,
        i512_idx=0,
        d128_idx=d128_idx,
        n_I512=n_I512,
        D=D,
    )
    for q_idx in range(_I_LAYOUT_TILES_PER_I512):
        nisa.dma_copy(
            dst=weight_q,
            src=weight.ap(
                pattern=[[_MX_TILE, _PMAX], [_MX_PACK_WIDTH, _MATMUL_K]],
                offset=q_idx,
            ),
        )
        nisa.nc_transpose(dst=weight_transposed_psum, data=weight_q)
        nisa.tensor_copy(dst=weight_matmul, src=weight_transposed_psum)
        nisa.nc_matmul(
            dst=projection_psum,
            stationary=weight_matmul,
            moving=activation[:, q_idx, :],
            accumulate=q_idx != 0,
        )

    for i512_idx in nl.sequential_range(1, n_I512):
        weight = _load_down_weight_tile(
            down_weights_flat=down_weights_flat,
            down_scales_flat=down_scales_flat,
            dequant_indices=dequant_indices,
            expert_idx=expert_idx,
            i512_idx=i512_idx,
            d128_idx=d128_idx,
            n_I512=n_I512,
            D=D,
        )
        for q_idx in range(_I_LAYOUT_TILES_PER_I512):
            nisa.dma_copy(
                dst=weight_q,
                src=weight.ap(
                    pattern=[[_MX_TILE, _PMAX], [_MX_PACK_WIDTH, _MATMUL_K]],
                    offset=q_idx,
                ),
            )
            nisa.nc_transpose(dst=weight_transposed_psum, data=weight_q)
            nisa.tensor_copy(dst=weight_matmul, src=weight_transposed_psum)
            nisa.nc_matmul(
                dst=projection_psum,
                stationary=weight_matmul,
                moving=activation[:, i512_idx * _I_LAYOUT_TILES_PER_I512 + q_idx, :],
                accumulate=True,
            )

    projection_bf16 = nl.ndarray((_PMAX, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=projection_bf16, src=projection_psum)
    return projection_bf16


def _load_gate_up_weight_tile(
    gate_up_weights_flat: nl.NkiTensor,
    gate_up_scales_flat: nl.NkiTensor,
    dequant_indices: nl.NkiTensor,
    expert_idx: int,
    branch_idx: int,
    d512_idx: int,
    i_layout_idx: int,
    n_D512: int,
    I: int,
) -> nl.NkiTensor:
    """Load and dequantize one ``[128 output, 512 contract]`` gate/up tile."""
    i_offset = i_layout_idx * _PMAX
    weight_p_stride = 2 * n_D512 * I
    weight_offset = expert_idx * _MX_PACKED_TILE * weight_p_stride + branch_idx * n_D512 * I + d512_idx * I + i_offset
    scale_p_stride = 2 * n_D512 * I
    scale_offset = expert_idx * _MX_SCALE_ROWS * scale_p_stride + branch_idx * n_D512 * I + d512_idx * I + i_offset
    return _dequantize_mxfp4_tile(
        packed_src=gate_up_weights_flat.ap(
            pattern=[[1, _PMAX], [weight_p_stride, _MX_PACKED_TILE]],
            offset=weight_offset,
        ),
        scale_src=gate_up_scales_flat.ap(
            pattern=[[1, _PMAX], [scale_p_stride, _MX_SCALE_ROWS]],
            offset=scale_offset,
        ),
        dequant_indices=dequant_indices,
    )


def _load_down_weight_tile(
    down_weights_flat: nl.NkiTensor,
    down_scales_flat: nl.NkiTensor,
    dequant_indices: nl.NkiTensor,
    expert_idx: int,
    i512_idx: int,
    d128_idx: int,
    n_I512: int,
    D: int,
) -> nl.NkiTensor:
    """Load and dequantize one ``[128 output, 512 contract]`` down tile."""
    d_offset = d128_idx * _PMAX
    weight_p_stride = n_I512 * D
    weight_offset = expert_idx * _MX_PACKED_TILE * weight_p_stride + i512_idx * D + d_offset
    scale_p_stride = n_I512 * D
    scale_offset = expert_idx * _MX_SCALE_ROWS * scale_p_stride + i512_idx * D + d_offset
    return _dequantize_mxfp4_tile(
        packed_src=down_weights_flat.ap(
            pattern=[[1, _PMAX], [weight_p_stride, _MX_PACKED_TILE]],
            offset=weight_offset,
        ),
        scale_src=down_scales_flat.ap(
            pattern=[[1, _PMAX], [scale_p_stride, _MX_SCALE_ROWS]],
            offset=scale_offset,
        ),
        dequant_indices=dequant_indices,
    )


def _dequantize_mxfp4_tile(
    packed_src: nl.NkiTensor,
    scale_src: nl.NkiTensor,
    dequant_indices: nl.NkiTensor,
) -> nl.NkiTensor:
    """Dequantize a packed MXFP4 tile to BF16 entirely in SBUF."""
    packed = nl.ndarray((_PMAX, _MX_PACKED_TILE), dtype=nl.float4_e2m1fn_x4, buffer=nl.sbuf)
    scale = nl.ndarray((_PMAX, _MX_SCALE_ROWS), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.dma_copy(dst=packed, src=packed_src)
    nisa.dma_copy(dst=scale, src=scale_src)

    unpacked_swizzled = nl.ndarray((_PMAX, _MX_TILE), dtype=nl.float8_e5m2, buffer=nl.sbuf)
    _tensor_dequantize_e2m1(dst=unpacked_swizzled, src=packed.view(nl.uint8))
    unpacked = nl.ndarray((_PMAX, _MX_TILE), dtype=nl.float8_e5m2, buffer=nl.sbuf)
    nisa.nc_n_gather(dst=unpacked, data=unpacked_swizzled, indices=dequant_indices)

    scale_u32 = nl.ndarray((_PMAX, _MX_SCALE_ROWS), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=scale_u32, src=scale)
    nisa.tensor_scalar(dst=scale_u32, data=scale_u32, op0=nl.left_shift, operand0=23)
    scale_f32 = scale_u32.view(nl.float32)

    result = nl.ndarray((_PMAX, _MX_TILE), dtype=nl.bfloat16, buffer=nl.sbuf)
    unpacked_f32 = nl.ndarray((_PMAX, _MX_GROUP_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    scaled_f32 = nl.ndarray((_PMAX, _MX_GROUP_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    for group_idx in range(_MX_SCALE_ROWS):
        group_slice = nl.ds(group_idx * _MX_GROUP_SIZE, _MX_GROUP_SIZE)
        nisa.tensor_copy(dst=unpacked_f32, src=unpacked[:, group_slice])
        nisa.tensor_scalar(
            dst=scaled_f32,
            data=unpacked_f32,
            op0=nl.multiply,
            operand0=scale_f32[:, group_idx : group_idx + 1],
        )
        nisa.tensor_copy(dst=result[:, group_slice], src=scaled_f32)
    return result


def _tensor_dequantize_e2m1(dst: nl.NkiTensor, src: nl.NkiTensor):
    """Emit an unnamed E2M1-to-E5M2 TensorDequantize instruction.

    The experimental wrapper currently assigns every invocation the fixed name
    ``dequantize``. A kernel with multiple weight tiles therefore hits the MLIR
    tracer's duplicate-name check. This emits the same instruction without an
    optional operation name.
    """
    asm_bytes = (
        "7B_10_00_00_"
        "{events:NEURON_ISA_TBP_EVENTS:DATAPATH}_"
        "{srcs[0]:NEURON_ISA_TPB_TENSOR3D}_"
        "00_00_00_00_"
        f"09_09_{_PMAX:02X}_01_00_"
        "00_00_00_00_00_00_00_"
        "{dsts[0]:NEURON_ISA_TPB_TENSOR3D}_"
        "00_00_00_00"
    )
    get_backend().inline_asm_bytes(
        asm_bytes=asm_bytes,
        sync_type=sync_type.datapath,
        engine=nisa.engine.gpsimd,
        latency_ns=100,
        mem_reads=[tensor_ap_view(src.view(nl.uint32))],
        mem_writes=[tensor_ap_view(dst.view(nl.uint32))],
        scalars=[],
        callee_attrs="",
        simulate_fn=_make_simulate_fn(1, 0),
        name=None,
    )


def _build_dequant_indices() -> nl.NkiTensor:
    """Build the fixed TensorDequantize-output to low-nibble-first permutation."""
    desired_index = nl.ndarray((_PMAX, _MX_TILE), dtype=nl.uint32, buffer=nl.sbuf)
    packed_byte = nl.ndarray((_PMAX, _MX_TILE), dtype=nl.uint32, buffer=nl.sbuf)
    word = nl.ndarray((_PMAX, _MX_TILE), dtype=nl.uint32, buffer=nl.sbuf)
    nibble = nl.ndarray((_PMAX, _MX_TILE), dtype=nl.uint32, buffer=nl.sbuf)
    byte_lane = nl.ndarray((_PMAX, _MX_TILE), dtype=nl.uint32, buffer=nl.sbuf)

    nisa.iota(dst=desired_index, pattern=[[1, _MX_TILE]], offset=0, channel_multiplier=0)
    nisa.tensor_scalar(dst=packed_byte, data=desired_index, op0=nl.right_shift, operand0=1)
    nisa.tensor_scalar(dst=word, data=packed_byte, op0=nl.right_shift, operand0=2)
    nisa.tensor_scalar(dst=word, data=word, op0=nl.left_shift, operand0=3)
    nisa.tensor_scalar(dst=nibble, data=desired_index, op0=nl.bitwise_and, operand0=1)
    nisa.tensor_scalar(dst=nibble, data=nibble, op0=nl.left_shift, operand0=2)
    nisa.tensor_scalar(dst=byte_lane, data=packed_byte, op0=nl.bitwise_and, operand0=3)
    nisa.tensor_tensor(dst=desired_index, data1=word, data2=nibble, op=nl.add)
    nisa.tensor_tensor(dst=desired_index, data1=desired_index, data2=byte_lane, op=nl.add)
    return desired_index
