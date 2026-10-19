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

"""Kimi-K3 dense MLP with exact two-sided SiTU activation."""

import nki
import nki.isa as nisa
import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil

_P_MAX = 128
_F_MAX = 512
_GATE_UP_TILE = 512


@nki.jit
def kimi_k3_situ_mlp(
    hidden_states: nl.NkiTensor,
    gate_proj_weight: nl.NkiTensor,
    up_proj_weight: nl.NkiTensor,
    down_proj_weight: nl.NkiTensor,
    beta: float,
    linear_beta: float,
) -> nl.NkiTensor:
    """Compute the Kimi-K3 dense/shared MLP for decode and short prefill.

    The kernel is intended for LNC1 Trn3 execution with one decode token or
    prefill chunks of at most 128 tokens. It accepts checkpoint-natural weight
    layouts and preserves the model's BF16 projection boundaries.

    Dimensions:
        T: Token count, ``1 <= T <= 128``.
        H: Hidden width.
        I: Intermediate width.

    Args:
        hidden_states (nl.NkiTensor): BF16 input ``[T, H]``.
        gate_proj_weight (nl.NkiTensor): BF16 gate weights ``[I, H]``.
        up_proj_weight (nl.NkiTensor): BF16 up weights ``[I, H]``.
        down_proj_weight (nl.NkiTensor): BF16 down weights ``[H, I]``.
        beta (float): Positive gate-branch SiTU soft-clipping factor.
        linear_beta (float): Positive up-branch SiTU soft-clipping factor.

    Returns:
        output (nl.NkiTensor): BF16 MLP output ``[T, H]``.

    Notes:
        - Gate and up matmuls round to BF16 before the FP32 SiTU epilogue.
        - The SiTU result rounds to BF16 before the down matmul.
        - Partial H/I tiles are zero-padded, including TP shard widths such as
          ``H=224`` and ``I=192``.

    Pseudocode:
        hidden_tiles = transpose_and_pad(hidden_states)
        for each 512-wide intermediate tile:
            gate = bf16(hidden @ gate_weight.T)
            up = bf16(hidden @ up_weight.T)
            gate_term = beta * tanh(fp32(gate) / beta) * sigmoid(fp32(gate))
            up_term = linear_beta * tanh(fp32(up) / linear_beta)
            activation = bf16(gate_term * up_term)
        for each 512-wide hidden output tile:
            output_tile = bf16(activation @ down_weight.T)
        return output
    """
    _validate_inputs(
        hidden_states=hidden_states,
        gate_proj_weight=gate_proj_weight,
        up_proj_weight=up_proj_weight,
        down_proj_weight=down_proj_weight,
        beta=beta,
        linear_beta=linear_beta,
    )

    token_count, hidden_size = hidden_states.shape
    intermediate_size = gate_proj_weight.shape[0]
    hidden_tile_count = div_ceil(hidden_size, _P_MAX)
    intermediate_128_tile_count = div_ceil(intermediate_size, _P_MAX)
    intermediate_512_tile_count = div_ceil(intermediate_size, _GATE_UP_TILE)

    hidden_flat = hidden_states.reshape((token_count * hidden_size,))
    hidden_tiles = nl.ndarray(
        (_P_MAX, hidden_tile_count, token_count),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )
    nisa.memset(dst=hidden_tiles, value=0.0)
    for hidden_tile_idx in range(hidden_tile_count):
        hidden_offset = hidden_tile_idx * _P_MAX
        hidden_tile_size = min(_P_MAX, hidden_size - hidden_offset)
        nisa.dma_copy(
            dst=hidden_tiles[0:hidden_tile_size, hidden_tile_idx, 0:token_count],
            src=hidden_flat.ap(
                pattern=[[1, hidden_tile_size], [hidden_size, token_count]],
                offset=hidden_offset,
            ),
        )

    activation = nl.ndarray(
        (_P_MAX, intermediate_128_tile_count, token_count),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )
    nisa.memset(dst=activation, value=0.0)

    for intermediate_512_tile_idx in range(intermediate_512_tile_count):
        intermediate_offset = intermediate_512_tile_idx * _GATE_UP_TILE
        intermediate_tile_size = min(_GATE_UP_TILE, intermediate_size - intermediate_offset)

        gate_bf16 = _linear_projection_tile(
            hidden_tiles=hidden_tiles,
            weight=gate_proj_weight,
            intermediate_offset=intermediate_offset,
            intermediate_tile_size=intermediate_tile_size,
        )
        gate_f32 = nl.ndarray(
            (token_count, intermediate_tile_size),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        gate_sigmoid_f32 = nl.ndarray(
            (token_count, intermediate_tile_size),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        nisa.tensor_copy(dst=gate_f32, src=gate_bf16)
        nisa.activation(dst=gate_sigmoid_f32, op=nl.sigmoid, data=gate_f32)
        nisa.activation(
            dst=gate_f32,
            op=nl.tanh,
            data=gate_f32,
            scale=1.0 / beta,
        )
        nisa.scalar_tensor_tensor(
            dst=gate_f32,
            data=gate_f32,
            op0=nl.multiply,
            operand0=beta,
            op1=nl.multiply,
            operand1=gate_sigmoid_f32,
        )

        up_bf16 = _linear_projection_tile(
            hidden_tiles=hidden_tiles,
            weight=up_proj_weight,
            intermediate_offset=intermediate_offset,
            intermediate_tile_size=intermediate_tile_size,
        )
        up_f32 = nl.ndarray(
            (token_count, intermediate_tile_size),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        activation_ti = nl.ndarray(
            (token_count, intermediate_tile_size),
            dtype=nl.bfloat16,
            buffer=nl.sbuf,
        )
        nisa.tensor_copy(dst=up_f32, src=up_bf16)
        nisa.activation(
            dst=up_f32,
            op=nl.tanh,
            data=up_f32,
            scale=1.0 / linear_beta,
        )
        nisa.scalar_tensor_tensor(
            dst=activation_ti,
            data=up_f32,
            op0=nl.multiply,
            operand0=linear_beta,
            op1=nl.multiply,
            operand1=gate_f32,
        )

        intermediate_subtile_count = div_ceil(intermediate_tile_size, _P_MAX)
        activation_transpose = nl.ndarray(
            (_P_MAX, token_count),
            dtype=nl.bfloat16,
            buffer=nl.psum,
        )
        for intermediate_subtile_idx in range(intermediate_subtile_count):
            subtile_offset = intermediate_subtile_idx * _P_MAX
            subtile_size = min(_P_MAX, intermediate_tile_size - subtile_offset)
            activation_tile_idx = intermediate_512_tile_idx * (_GATE_UP_TILE // _P_MAX) + intermediate_subtile_idx
            nisa.nc_transpose(
                dst=activation_transpose[0:subtile_size, 0:token_count],
                data=activation_ti[0:token_count, subtile_offset : subtile_offset + subtile_size],
            )
            nisa.tensor_copy(
                dst=activation[0:subtile_size, activation_tile_idx, 0:token_count],
                src=activation_transpose[0:subtile_size, 0:token_count],
            )

    down_weight_flat = down_proj_weight.reshape((hidden_size * intermediate_size,))
    output = nl.ndarray(
        (token_count, hidden_size),
        dtype=nl.bfloat16,
        buffer=nl.shared_hbm,
    )
    hidden_output_tile_count = div_ceil(hidden_size, _F_MAX)
    for hidden_output_tile_idx in range(hidden_output_tile_count):
        hidden_output_offset = hidden_output_tile_idx * _F_MAX
        hidden_output_tile_size = min(_F_MAX, hidden_size - hidden_output_offset)
        output_psum = nl.ndarray(
            (token_count, hidden_output_tile_size),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        down_weight_matmul = nl.ndarray(
            (_P_MAX, hidden_output_tile_size),
            dtype=nl.bfloat16,
            buffer=nl.sbuf,
        )
        down_weight_natural = nl.ndarray(
            (_P_MAX, _P_MAX),
            dtype=nl.bfloat16,
            buffer=nl.sbuf,
        )
        down_weight_transpose = nl.ndarray(
            (_P_MAX, _P_MAX),
            dtype=nl.bfloat16,
            buffer=nl.psum,
        )
        hidden_output_subtile_count = div_ceil(hidden_output_tile_size, _P_MAX)

        for intermediate_128_tile_idx in range(intermediate_128_tile_count):
            intermediate_offset = intermediate_128_tile_idx * _P_MAX
            intermediate_tile_size = min(_P_MAX, intermediate_size - intermediate_offset)
            if intermediate_tile_size < _P_MAX:
                nisa.memset(dst=down_weight_matmul, value=0.0)
            for hidden_output_subtile_idx in range(hidden_output_subtile_count):
                hidden_subtile_offset = hidden_output_subtile_idx * _P_MAX
                hidden_subtile_size = min(_P_MAX, hidden_output_tile_size - hidden_subtile_offset)
                nisa.dma_copy(
                    dst=down_weight_natural[0:hidden_subtile_size, 0:intermediate_tile_size],
                    src=down_weight_flat.ap(
                        pattern=[
                            [intermediate_size, hidden_subtile_size],
                            [1, intermediate_tile_size],
                        ],
                        offset=(hidden_output_offset + hidden_subtile_offset) * intermediate_size + intermediate_offset,
                    ),
                )
                nisa.nc_transpose(
                    dst=down_weight_transpose[0:intermediate_tile_size, 0:hidden_subtile_size],
                    data=down_weight_natural[0:hidden_subtile_size, 0:intermediate_tile_size],
                )
                nisa.tensor_copy(
                    dst=down_weight_matmul[
                        0:intermediate_tile_size,
                        hidden_subtile_offset : hidden_subtile_offset + hidden_subtile_size,
                    ],
                    src=down_weight_transpose[0:intermediate_tile_size, 0:hidden_subtile_size],
                )
            nisa.nc_matmul(
                dst=output_psum,
                stationary=activation[:, intermediate_128_tile_idx, :],
                moving=down_weight_matmul,
                accumulate=intermediate_128_tile_idx != 0,
            )

        output_tile = nl.ndarray(
            (token_count, hidden_output_tile_size),
            dtype=nl.bfloat16,
            buffer=nl.sbuf,
        )
        nisa.tensor_copy(dst=output_tile, src=output_psum)
        nisa.dma_copy(
            dst=output[:, hidden_output_offset : hidden_output_offset + hidden_output_tile_size],
            src=output_tile,
        )

    return output


def _validate_inputs(
    hidden_states: nl.NkiTensor,
    gate_proj_weight: nl.NkiTensor,
    up_proj_weight: nl.NkiTensor,
    down_proj_weight: nl.NkiTensor,
    beta: float,
    linear_beta: float,
) -> None:
    """Validate the public tensor layouts and Trn3 LNC1 execution contract."""
    kernel_assert(
        nl.program_ndim() == 0 or (nl.program_ndim() == 1 and nl.num_programs(0) == 1),
        "kimi_k3_situ_mlp supports LNC1 only",
    )
    kernel_assert(nisa.get_nc_version() >= nisa.nc_version.gen4, "kimi_k3_situ_mlp requires Trn3 or newer")
    kernel_assert(len(hidden_states.shape) == 2, f"Expected hidden_states [T, H], got {hidden_states.shape}")
    kernel_assert(
        len(gate_proj_weight.shape) == 2,
        f"Expected gate_proj_weight [I, H], got {gate_proj_weight.shape}",
    )
    kernel_assert(
        len(up_proj_weight.shape) == 2,
        f"Expected up_proj_weight [I, H], got {up_proj_weight.shape}",
    )
    kernel_assert(
        len(down_proj_weight.shape) == 2,
        f"Expected down_proj_weight [H, I], got {down_proj_weight.shape}",
    )
    kernel_assert(hidden_states.dtype == nl.bfloat16, f"Expected BF16 hidden_states, got {hidden_states.dtype}")
    kernel_assert(gate_proj_weight.dtype == nl.bfloat16, f"Expected BF16 gate weights, got {gate_proj_weight.dtype}")
    kernel_assert(up_proj_weight.dtype == nl.bfloat16, f"Expected BF16 up weights, got {up_proj_weight.dtype}")
    kernel_assert(down_proj_weight.dtype == nl.bfloat16, f"Expected BF16 down weights, got {down_proj_weight.dtype}")

    token_count, hidden_size = hidden_states.shape
    intermediate_size = gate_proj_weight.shape[0]
    kernel_assert(
        token_count > 0 and token_count <= _P_MAX,
        f"Expected 1 <= T <= {_P_MAX}, got T={token_count}",
    )
    kernel_assert(hidden_size > 0, f"Expected H > 0, got H={hidden_size}")
    kernel_assert(intermediate_size > 0, f"Expected I > 0, got I={intermediate_size}")
    kernel_assert(
        gate_proj_weight.shape == (intermediate_size, hidden_size),
        f"Expected gate_proj_weight {(intermediate_size, hidden_size)}, got {gate_proj_weight.shape}",
    )
    kernel_assert(
        up_proj_weight.shape == (intermediate_size, hidden_size),
        f"Expected up_proj_weight {(intermediate_size, hidden_size)}, got {up_proj_weight.shape}",
    )
    kernel_assert(
        down_proj_weight.shape == (hidden_size, intermediate_size),
        f"Expected down_proj_weight {(hidden_size, intermediate_size)}, got {down_proj_weight.shape}",
    )
    kernel_assert(beta > 0.0, f"Expected beta > 0, got beta={beta}")
    kernel_assert(linear_beta > 0.0, f"Expected linear_beta > 0, got linear_beta={linear_beta}")


def _linear_projection_tile(
    hidden_tiles: nl.NkiTensor,
    weight: nl.NkiTensor,
    intermediate_offset: int,
    intermediate_tile_size: int,
) -> nl.NkiTensor:
    """Project one gate/up tile and round the accumulated result to BF16."""
    intermediate_size, hidden_size = weight.shape
    token_count = hidden_tiles.shape[2]
    hidden_tile_count = hidden_tiles.shape[1]
    weight_flat = weight.reshape((intermediate_size * hidden_size,))
    projection_psum = nl.ndarray(
        (token_count, intermediate_tile_size),
        dtype=nl.float32,
        buffer=nl.psum,
    )
    weight_matmul = nl.ndarray(
        (_P_MAX, intermediate_tile_size),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )
    weight_natural = nl.ndarray(
        (_P_MAX, _P_MAX),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )
    weight_transpose = nl.ndarray(
        (_P_MAX, _P_MAX),
        dtype=nl.bfloat16,
        buffer=nl.psum,
    )
    intermediate_subtile_count = div_ceil(intermediate_tile_size, _P_MAX)

    for hidden_tile_idx in range(hidden_tile_count):
        hidden_offset = hidden_tile_idx * _P_MAX
        hidden_tile_size = min(_P_MAX, hidden_size - hidden_offset)
        if hidden_tile_size < _P_MAX:
            nisa.memset(dst=weight_matmul, value=0.0)
        for intermediate_subtile_idx in range(intermediate_subtile_count):
            intermediate_subtile_offset = intermediate_subtile_idx * _P_MAX
            intermediate_subtile_size = min(
                _P_MAX,
                intermediate_tile_size - intermediate_subtile_offset,
            )
            nisa.dma_copy(
                dst=weight_natural[0:intermediate_subtile_size, 0:hidden_tile_size],
                src=weight_flat.ap(
                    pattern=[
                        [hidden_size, intermediate_subtile_size],
                        [1, hidden_tile_size],
                    ],
                    offset=(intermediate_offset + intermediate_subtile_offset) * hidden_size + hidden_offset,
                ),
            )
            nisa.nc_transpose(
                dst=weight_transpose[0:hidden_tile_size, 0:intermediate_subtile_size],
                data=weight_natural[0:intermediate_subtile_size, 0:hidden_tile_size],
            )
            nisa.tensor_copy(
                dst=weight_matmul[
                    0:hidden_tile_size,
                    intermediate_subtile_offset : intermediate_subtile_offset + intermediate_subtile_size,
                ],
                src=weight_transpose[0:hidden_tile_size, 0:intermediate_subtile_size],
            )
        nisa.nc_matmul(
            dst=projection_psum,
            stationary=hidden_tiles[:, hidden_tile_idx, :],
            moving=weight_matmul,
            accumulate=hidden_tile_idx != 0,
        )

    projection_bf16 = nl.ndarray(
        (token_count, intermediate_tile_size),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )
    nisa.tensor_copy(dst=projection_bf16, src=projection_psum)
    return projection_bf16
