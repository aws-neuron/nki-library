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

"""Shape-tuned per-phase blocking for the MXFP8 MoE backward kernel.

This is empirical autotune DATA, not kernel source — the kernel accepts an MXFP8MOEBwdConfig
(built from its phase1_config..phase4_config plus phase3/4_transpose_mode) but carries no tuning
table itself. The test harness (and any caller that wants fast blocking) looks a shape up here and
passes the result in as an explicit config. The validated test harness supplies its own explicit
shape-derived config on a miss; it never relies on the kernel's one-tile defaults.

Keyed on the FULL run identity (TuningKey), because blocking that is optimal — or even legal — for
one (shape, sharding, affinity, dtype, feature) point is not for another: sharding/affinity change
each phase's (M,K,N); dtype and the spill/scale-packing/bias features change SBUF pressure and thus
the tile budget. A lookup misses unless every field matches an entry exactly.
"""

from typing import NamedTuple

import nki.language as nl
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_config import MatmulMxfp8KernelConfig
from nkilib_src.nkilib.experimental.mlp_mxfp8.common_utils import TILE_M
from nkilib_src.nkilib.experimental.moe.bwd.moe_bwd_parameters import AffinityOption, ShardOption
from nkilib_src.nkilib.experimental.moe_mxfp8.bwd.config import MXFP8MOEBwdConfig, QuantScheme, TransposeMode

MAX_DGT_LOAD_WIDTH = 1024
MAX_LOAD_TILES = 8


def derive_load_tiles(block_tiles, tile_size):
    """Choose the largest block divisor that fits one DGT load."""
    if block_tiles < 1 or tile_size < 1:
        raise ValueError(f"block_tiles and tile_size must be positive, got {block_tiles=} and {tile_size=}")

    max_load_tiles = min(MAX_LOAD_TILES, MAX_DGT_LOAD_WIDTH // tile_size)
    if max_load_tiles < 1:
        raise ValueError(f"tile_size={tile_size} exceeds the DGT load-width limit {MAX_DGT_LOAD_WIDTH}")

    for load_tiles in range(min(block_tiles, max_load_tiles), 0, -1):
        if block_tiles % load_tiles:
            continue
        if load_tiles == 1 and block_tiles > 1:
            raise ValueError(
                f"block_tiles={block_tiles} has no multi-tile divisor within the "
                f"DGT load-width limit {MAX_DGT_LOAD_WIDTH} for tile_size={tile_size}"
            )
        return load_tiles
    raise ValueError(f"no legal load tiles for {block_tiles=} and {tile_size=}")


def tune_block_and_load_tiles(block_tiles, tile_size):
    """Retune a requested block down to the nearest legal block/load pair."""
    for tuned_block_tiles in range(block_tiles, 0, -1):
        try:
            load_tiles = derive_load_tiles(tuned_block_tiles, tile_size)
        except ValueError:
            continue
        return tuned_block_tiles, load_tiles
    raise ValueError(f"no legal block/load pair for {block_tiles=} and {tile_size=}")


def _phase_config(**kwargs):
    block_m = kwargs["TILES_IN_BLOCK_M"]
    block_n = kwargs["TILES_IN_BLOCK_N"]
    tile_m = kwargs.get("tile_m", TILE_M)
    tile_n = kwargs["tile_n"]
    kwargs.setdefault("TILES_IN_LOAD_M", derive_load_tiles(block_m, tile_m))
    kwargs.setdefault("TILES_IN_LOAD_N", derive_load_tiles(block_n, tile_n))
    return MatmulMxfp8KernelConfig(quant_scheme=QuantScheme.WRAPX, **kwargs)


class TuningKey(NamedTuple):
    """Full identity a tuned per-phase blocking is valid for."""

    B: int
    H: int
    I_TP: int
    num_shards: int
    shard_option: object
    affinity_option: object
    compute_dtype: object
    spill_reload: bool
    use_scale_packing: bool
    bias: bool
    single_expert_dense: bool


# Qwen3-235B-A22B (H=4096, moe_inter=1536): TP1 I_TP=1536, TP2 768, TP4 384, TP8 192.
# B=4096 => single dense block over T=4096 tokens => per-block (M,K,N) == exp-one shapes.
# The transposed phases (P3 RHS / P4 LHS) build their [F,B] buffer with NC (nc_transpose on the
# PE). NC measured faster than DMA for the isolated single-block kernel: with DMA the [F,B]
# transpose piles onto the saturated DMA engine and the PE idles ~49% of wall-clock waiting for
# its operands before the first matmul; NC moves the transpose onto the (idle) PE so matmuls
# start immediately (N=1: 2.342ms DMA -> 2.190ms NC, -6.5%). DMA-transpose can still win
# END-TO-END when the transpose overlaps neighbor matmul-heavy kernels — re-measure per deployment.
#
# All entries below are the base identity: LNC2 (num_shards=2), SHARD_ON_FREE, AFFINITY_ON_I,
# bf16, and no spill_reload / scale_packing / bias. A run with any other identity misses the
# table and uses default blocking.
#
# Entries are MXFP8MOEBwdConfig — the exact struct the kernel already accepts, so no separate
# tuning type is needed. M/K/N on each phase_config are left 0: they follow from the run's shape
# and sharding, not the tuning; the validated test harness fills them per phase.
# DGT loads are capped at 1024 elements, so tile_n=512 uses two-tile loads and even N blocks.
_BASE_TUNING_KEY = {
    "num_shards": 2,
    "shard_option": ShardOption.SHARD_ON_FREE,
    "affinity_option": AffinityOption.AFFINITY_ON_I,
    "compute_dtype": nl.bfloat16,
    "spill_reload": False,
    "use_scale_packing": False,
    "bias": False,
    "single_expert_dense": False,
}
_DENSE_TUNING_KEY = {
    "num_shards": 2,
    "shard_option": ShardOption.SHARD_ON_FREE,
    "affinity_option": AffinityOption.AFFINITY_ON_I,
    "compute_dtype": nl.bfloat16,
    "spill_reload": False,
    "use_scale_packing": False,
    "bias": False,
    "single_expert_dense": True,
}
SHAPE_TUNED_CONFIGS = {
    TuningKey(B=4096, H=4096, I_TP=1536, **_BASE_TUNING_KEY): MXFP8MOEBwdConfig(  # TP1, single block (B=T=4096)
        # P2 M-blocking is 16 (not 32): 32 co-resident M-tiles overflow SBUF during QuantizeMx
        # (NCC_IBIR229) on this I_TP=1536 shape. 16 is the largest that fits without spill.
        phase1_config=_phase_config(
            M=0, K=0, N=0, TILES_IN_BLOCK_M=16, TILES_IN_BLOCK_N=2, TILES_IN_BLOCK_K=4, tile_n=512
        ),
        phase2_config=_phase_config(
            M=0, K=0, N=0, TILES_IN_BLOCK_M=16, TILES_IN_BLOCK_N=2, TILES_IN_BLOCK_K=6, tile_n=512
        ),
        phase3_config=_phase_config(
            M=0, K=0, N=0, TILES_IN_BLOCK_M=6, TILES_IN_BLOCK_N=4, TILES_IN_BLOCK_K=2, tile_n=512
        ),
        phase4_config=_phase_config(
            M=0, K=0, N=0, TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=2, TILES_IN_BLOCK_K=2, tile_n=512
        ),
        phase3_transpose_mode=TransposeMode.NC,
        phase4_transpose_mode=TransposeMode.NC,
    ),
    TuningKey(B=2048, H=4096, I_TP=1536, **_BASE_TUNING_KEY): MXFP8MOEBwdConfig(  # TP1, 2 blocks over T=4096
        # Per-block (M,K,N): P1 2048x4096x1536, P2 2048x3072x4096, P3 1536x2048x4096, P4 1536x2048x4096.
        # Mirrors the SBUF-safe single-block TP1 tiles (P2 M<=16 to avoid the QuantizeMx OOM);
        # P1/P2 M-blocking halves with B (16->8).
        phase1_config=_phase_config(
            M=0, K=0, N=0, TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=2, TILES_IN_BLOCK_K=4, tile_n=512
        ),
        phase2_config=_phase_config(
            M=0, K=0, N=0, TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=2, TILES_IN_BLOCK_K=6, tile_n=512
        ),
        phase3_config=_phase_config(
            M=0, K=0, N=0, TILES_IN_BLOCK_M=6, TILES_IN_BLOCK_N=4, TILES_IN_BLOCK_K=2, tile_n=512
        ),
        phase4_config=_phase_config(
            M=0, K=0, N=0, TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=2, TILES_IN_BLOCK_K=2, tile_n=512
        ),
        phase3_transpose_mode=TransposeMode.NC,
        phase4_transpose_mode=TransposeMode.NC,
    ),
    TuningKey(B=4096, H=4096, I_TP=768, **_BASE_TUNING_KEY): MXFP8MOEBwdConfig(  # TP2, single block (B=T=4096)
        phase1_config=_phase_config(
            M=0, K=0, N=0, TILES_IN_BLOCK_M=16, TILES_IN_BLOCK_N=3, TILES_IN_BLOCK_K=4, tile_n=256
        ),
        phase2_config=_phase_config(
            M=0, K=0, N=0, TILES_IN_BLOCK_M=16, TILES_IN_BLOCK_N=4, TILES_IN_BLOCK_K=3, tile_n=512
        ),
        phase3_config=_phase_config(
            M=0, K=0, N=0, TILES_IN_BLOCK_M=12, TILES_IN_BLOCK_N=2, TILES_IN_BLOCK_K=2, tile_n=512
        ),
        phase4_config=_phase_config(
            M=0, K=0, N=0, TILES_IN_BLOCK_M=6, TILES_IN_BLOCK_N=4, TILES_IN_BLOCK_K=4, tile_n=512
        ),
        phase3_transpose_mode=TransposeMode.NC,
        phase4_transpose_mode=TransposeMode.NC,
    ),
    TuningKey(B=2048, H=4096, I_TP=768, **_BASE_TUNING_KEY): MXFP8MOEBwdConfig(  # TP2, 2 blocks over T=4096
        # Per-block (M,K,N) matmul shapes map to the standalone autotune keys:
        #   P1 2048x4096x768, P2 2048x1536x4096, P3 1536x2048x4096, P4 768x2048x4096.
        # Tiles copied from that autotune result. Without this entry the block falls back to
        # the TILES_IN_BLOCK_*=1 heuristic (1 tile/block => SBUF nearly empty, operand re-reads).
        phase1_config=_phase_config(
            M=0, K=0, N=0, TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=3, TILES_IN_BLOCK_K=2, tile_n=256
        ),
        phase2_config=_phase_config(
            M=0, K=0, N=0, TILES_IN_BLOCK_M=16, TILES_IN_BLOCK_N=4, TILES_IN_BLOCK_K=3, tile_n=512
        ),
        phase3_config=_phase_config(
            M=0, K=0, N=0, TILES_IN_BLOCK_M=12, TILES_IN_BLOCK_N=4, TILES_IN_BLOCK_K=4, tile_n=512
        ),
        phase4_config=_phase_config(
            M=0, K=0, N=0, TILES_IN_BLOCK_M=6, TILES_IN_BLOCK_N=2, TILES_IN_BLOCK_K=4, tile_n=512
        ),
        phase3_transpose_mode=TransposeMode.NC,
        phase4_transpose_mode=TransposeMode.NC,
    ),
    TuningKey(B=1024, H=4096, I_TP=768, **_BASE_TUNING_KEY): MXFP8MOEBwdConfig(  # TP2, 4 blocks over T=4096
        # Per-block (M,K,N) matmul shapes map to the standalone autotune keys:
        #   P1 1024x4096x768, P2 1024x1536x4096, P3 1536x1024x4096, P4 768x1024x4096.
        phase1_config=_phase_config(
            M=0, K=0, N=0, TILES_IN_BLOCK_M=4, TILES_IN_BLOCK_N=1, TILES_IN_BLOCK_K=8, tile_n=768
        ),
        phase2_config=_phase_config(
            M=0, K=0, N=0, TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=4, TILES_IN_BLOCK_K=3, tile_n=512
        ),
        phase3_config=_phase_config(
            M=0, K=0, N=0, TILES_IN_BLOCK_M=12, TILES_IN_BLOCK_N=4, TILES_IN_BLOCK_K=2, tile_n=512
        ),
        phase4_config=_phase_config(
            M=0, K=0, N=0, TILES_IN_BLOCK_M=6, TILES_IN_BLOCK_N=4, TILES_IN_BLOCK_K=2, tile_n=512
        ),
        phase3_transpose_mode=TransposeMode.NC,
        phase4_transpose_mode=TransposeMode.NC,
    ),
    TuningKey(
        B=4096,
        H=4096,
        I_TP=1536,
        **_DENSE_TUNING_KEY,
    ): MXFP8MOEBwdConfig(  # TP1, one contiguous dense block
        phase1_config=_phase_config(
            M=0,
            K=0,
            N=0,
            TILES_IN_BLOCK_M=16,
            TILES_IN_BLOCK_N=2,
            TILES_IN_BLOCK_K=4,
            tile_n=512,
            spill_reload=False,
            enable_scale_packing=True,
        ),
        phase2_config=_phase_config(
            M=0,
            K=0,
            N=0,
            TILES_IN_BLOCK_M=16,
            TILES_IN_BLOCK_N=2,
            TILES_IN_BLOCK_K=6,
            tile_n=512,
            spill_reload=False,
            enable_scale_packing=True,
        ),
        # P3 computes the combined 3072x4096x4096 gate/up gradient.
        phase3_config=_phase_config(
            M=0,
            K=0,
            N=0,
            TILES_IN_BLOCK_M=12,
            TILES_IN_BLOCK_N=4,
            TILES_IN_BLOCK_K=2,
            tile_n=512,
            spill_reload=False,
            enable_scale_packing=True,
        ),
        phase4_config=_phase_config(
            M=0,
            K=0,
            N=0,
            TILES_IN_BLOCK_M=8,
            TILES_IN_BLOCK_N=2,
            TILES_IN_BLOCK_K=2,
            tile_n=512,
            enable_scale_packing=True,
        ),
        phase3_transpose_mode=TransposeMode.NC,
        phase4_transpose_mode=TransposeMode.NC,
    ),
    TuningKey(
        B=4096,
        H=4096,
        I_TP=768,
        **_DENSE_TUNING_KEY,
    ): MXFP8MOEBwdConfig(  # TP2, one contiguous dense block
        phase1_config=_phase_config(
            M=0,
            K=0,
            N=0,
            TILES_IN_BLOCK_M=16,
            TILES_IN_BLOCK_N=3,
            TILES_IN_BLOCK_K=4,
            tile_n=256,
            spill_reload=False,
        ),
        phase2_config=_phase_config(
            M=0,
            K=0,
            N=0,
            TILES_IN_BLOCK_M=16,
            TILES_IN_BLOCK_N=4,
            TILES_IN_BLOCK_K=3,
            tile_n=512,
            spill_reload=False,
        ),
        phase3_config=_phase_config(
            M=0,
            K=0,
            N=0,
            TILES_IN_BLOCK_M=12,
            TILES_IN_BLOCK_N=2,
            TILES_IN_BLOCK_K=2,
            tile_n=512,
            spill_reload=False,
        ),
        phase4_config=_phase_config(
            M=0, K=0, N=0, TILES_IN_BLOCK_M=6, TILES_IN_BLOCK_N=4, TILES_IN_BLOCK_K=4, tile_n=512
        ),
        phase3_transpose_mode=TransposeMode.NC,
        phase4_transpose_mode=TransposeMode.NC,
    ),
}


def get_shape_tuned_config(
    B,
    H,
    I_TP,
    num_shards,
    shard_option,
    affinity_option,
    compute_dtype,
    spill_reload=False,
    use_scale_packing=False,
    bias=False,
    single_expert_dense=False,
):
    """Return the tuned MXFP8MOEBwdConfig for the full run identity, or None on a miss.

    All fields must match a table entry exactly. Callers must provide an explicit
    shape-derived config when this returns None.
    """
    return SHAPE_TUNED_CONFIGS.get(
        TuningKey(
            B=B,
            H=H,
            I_TP=I_TP,
            num_shards=num_shards,
            shard_option=shard_option,
            affinity_option=affinity_option,
            compute_dtype=compute_dtype,
            spill_reload=spill_reload,
            use_scale_packing=use_scale_packing,
            bias=bias,
            single_expert_dense=single_expert_dense,
        )
    )
