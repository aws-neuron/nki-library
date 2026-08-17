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
"""RMSNorm TKG kernel using NeuroTile.

Implements ``X_norm = X * rsqrt(mean(X^2) + eps) * gamma`` with optional LNC-2
sharding on the BxS dimension and sendrecv exchange between cores.
"""

from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib_src.nkilib.experimental import neurotile as nt

_SHARDING_BXS_THRESHOLD = 18  # Minimum BxS to enable LNC-2 sharding
_BXS_TILE_F_BUDGET = 1024  # Max F-dim elements per BxS tile (SBUF-fit heuristic)
_PARTITION_DIM = 128  # SBUF partition dimension (H0 split)


def rmsnorm_tkg(
    hidden: nl.ndarray,
    gamma: nl.ndarray,
    eps: float,
    H_actual: Optional[int],
) -> nl.ndarray:
    """RMSNorm sub-kernel for token-generation (TKG) hidden states.

    Computes ``X_norm = X * rsqrt(mean(X^2) + eps) * gamma`` with optional
    LNC-2 sharding on BxS and sendrecv exchange so consumers see the full
    output on both cores. Intended for use inside a larger ``@nki.jit``
    kernel.

    Args:
        hidden (nl.ndarray): [B, S_tkg, H], input hidden states in HBM.
        gamma (nl.ndarray): [1, H], per-element scale weights in HBM.
        eps (float): epsilon for rsqrt numerical stability.
        H_actual (int | None): actual hidden dim when H is padded (e.g.,
            H=3072 but H_actual=2880). Used in mean computation:
            ``mean(x^2) = sum(x^2) / H_actual``. ``None`` means H is not
            padded and ``H`` is used directly.

    Returns:
        nl.ndarray: [H0, BxS, H1] result in SBUF, complete on both cores
        after the sendrecv exchange.

    Notes:
        - Sharding is enabled when ``lnc == 2``, ``BxS > 18`` and ``BxS``
          is even. Below that threshold the kernel runs unsharded on a
          single core.
        - ``H`` must be a multiple of ``H0 = 128``; ``H1 = H // 128``.
    """
    B = hidden.shape[0]
    S_tkg = hidden.shape[1]
    H = hidden.shape[2]
    H0 = _PARTITION_DIM
    H1 = H // H0
    BxS = B * S_tkg
    H_denom = H_actual if H_actual is not None else H

    # -- LNC setup ----------------------------------------------
    lnc = nl.num_programs(0)
    shard_id = nl.program_id(0)
    do_shard = lnc == 2 and BxS > _SHARDING_BXS_THRESHOLD and BxS % 2 == 0
    num_shards = lnc if do_shard else 1
    if not do_shard:
        shard_id = 0
    shard_size = BxS // num_shards
    bxs_tile = min(min(BxS, _BXS_TILE_F_BUDGET // H1), shard_size)

    # -- Hoisted loads ------------------------------------------

    # Gamma: [1, H] -> [H0, H1] via contiguous reshape, loaded in one DMA.
    # No tiling here -- a transformed NkiTensor is a valid dma_copy source.
    gamma_sb = nl.ndarray((H0, H1), dtype=gamma.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gamma_sb, src=gamma.reshape((H0, H1)))

    # All-ones [H0, H0] for partition-dim reduction via nc_matmul
    ones = nl.ndarray((H0, H0), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones, 1.0)

    # Epsilon [H0, 1] -- nisa.activation bias requires tensor, not scalar
    eps_sb = nl.ndarray((H0, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(eps_sb, eps)

    # -- Permuted view + sharded tiles --------------------------
    hidden = (
        hidden.flatten_dims(0, 1)  # [BxS, H]
        .reshape_dim(1, (H0, H1))  # [BxS, H0, H1]
        .permute((1, 0, 2))  # [H0, BxS, H1]
    )
    tile_size = (H0, bxs_tile, H1)
    shard_range = nt.block_range(
        rank=shard_id,
        num_shards=num_shards,
        total=nt.ceiling_div(BxS, bxs_tile),
    )
    x_tiles = nt.tiles(hidden, tile_size=tile_size)[:, shard_range, :]

    # Pre-allocate SBUF output: full BxS (both shards) for sendrecv exchange.
    full_buf = nl.ndarray(hidden.shape, dtype=hidden.dtype, buffer=nl.sbuf)
    out_tiles = nt.tiles(full_buf, tile_size=tile_size)[:, shard_range, :]

    for tile_idx in range(x_tiles.shape[1]):
        out_data = out_tiles[0, tile_idx, 0].data

        # Step 1: Load HBM -> SBUF output slot
        x_tiles[0, tile_idx, 0].load(dst=out_data)

        # Step 2: x^2 (3D intermediate, matching nkilib)
        x_sq = nl.ndarray(tile_size, dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(x_sq, op=nl.square, data=out_data)

        # Step 3: Reduce across H1 (axis=2 on 3D)
        reduced = nl.ndarray((H0, bxs_tile), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(reduced, nl.add, data=x_sq, axis=2)

        # Step 4: Reduce across H0 partitions
        full_reduced = nl.ndarray((H0, bxs_tile), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(full_reduced, stationary=ones, moving=reduced)

        # Step 5: rsqrt
        nisa.activation(reduced, op=nl.rsqrt, data=full_reduced, scale=1.0 / H_denom, bias=eps_sb)

        # Step 6: gamma * x (broadcast gamma via stride-0 view on the NkiTensor)
        gamma_bc = gamma_sb.expand_dim(1).broadcast(1, bxs_tile)
        gamma_x = nl.ndarray(tile_size, dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(gamma_x, out_data, gamma_bc, nl.multiply)

        # Step 7: (gamma * x) * inv_rms -> output slot (broadcast via stride-0 view)
        inv_rms_bc = reduced.expand_dim(2).broadcast(2, H1)
        nisa.tensor_tensor(out_data, gamma_x, inv_rms_bc, nl.multiply)

    # -- Exchange shards ----------------------------------------
    if do_shard:
        other_id = 1 - shard_id
        shard_tiles = nt.tiles(full_buf, tile_size=(H0, shard_size, H1))
        nisa.sendrecv(
            dst=shard_tiles[0, other_id, 0].data,
            src=shard_tiles[0, shard_id, 0].data,
            send_to_rank=other_id,
            recv_from_rank=other_id,
            pipe_id=0,
        )

    return full_buf


@nki.jit
def rmsnorm_tkg_kernel(
    hidden: nl.ndarray,
    gamma: nl.ndarray,
    eps: float,
    H_actual: Optional[int],
) -> nl.ndarray:
    """Standalone TKG RMSNorm kernel for independent testing and profiling.

    Wraps :func:`rmsnorm_tkg` and copies the SBUF result back to HBM. Use this
    as the entry point when running RMSNorm by itself; for fused use cases
    (e.g., fused into QKV projection), call :func:`rmsnorm_tkg` directly.

    TODO: Specify intended usage range (e.g., recommended H range,
    batch sizes where this kernel performs best).

    Dimensions:
        B: Batch size
        S_tkg: TKG sequence length (small for token generation)
        H: Hidden dimension (must be multiple of H0 = 128)
        H0: SBUF partition dimension (== 128)
        H1: H // H0
        BxS: B * S_tkg

    Args:
        hidden (nl.ndarray): [B, S_tkg, H] input hidden states in HBM.
        gamma (nl.ndarray): [1, H] per-element scale weights in HBM.
        eps (float): epsilon for rsqrt numerical stability.
        H_actual (int | None): actual hidden dim when H is padded; ``None``
            uses ``H`` directly.

    Returns:
        dst (nl.ndarray): [H0, BxS, H1] RMSNorm result in HBM.

    Notes:
        - LNC-2 sharding kicks in when ``BxS > 18`` and ``BxS`` is even.

    Pseudocode:
        x_norm_sbuf = rmsnorm_tkg(hidden, gamma, eps, H_actual)
        dst_hbm = allocate(x_norm_sbuf.shape, in=hbm)
        dma_copy(dst_hbm, x_norm_sbuf)
        return dst_hbm
    """
    result = rmsnorm_tkg(hidden, gamma, eps, H_actual)
    dst = nl.ndarray(result.shape, dtype=hidden.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst, result)
    return dst
