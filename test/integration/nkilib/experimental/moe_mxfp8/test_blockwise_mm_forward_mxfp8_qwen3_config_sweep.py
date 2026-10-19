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

"""Blocking/tiling sweep for the MXFP8 MoE forward on the Qwen3-235B TP1/TP2 shapes.

This is a *tuning harness*, not a correctness gate: it enumerates a targeted grid
of per-GEMM blocking strategies for the Qwen3-235B per-expert forward shape
(H=4096, E=1, top_k=1) at both tensor-parallel degrees and runs every config
through the same three-metric MXFP8 validator so a numerically-wrong config can
never win. Run it on the shared fleet with perf collection to read the per-config
inference time and pick the fastest strategy.

There are ten sweep methods over the same blocking grids, differing only in TP
degree and how the operands are loaded/quantized:
  * ``tp1_blocking_sweep``                   : TP1 (I_TP=1536), DGT fast_dma_transpose,
    on-the-fly BF16 weight quantize.
  * ``tp2_blocking_sweep``                   : TP2 (I_TP=768),  DGT fast_dma_transpose.
  * ``tp1_1x32_blocking_sweep``              : TP1 (I_TP=1536), 1x32 PE-swizzle load.
  * ``tp2_1x32_blocking_sweep``              : TP2 (I_TP=768),  1x32 PE-swizzle load. (The
    flexible-K loader tiles K as 512 + a <512 remainder, so I_TP=768 works.)
  * ``tp1_kbyf_1x32_blocking_sweep``         : TP1 (I_TP=1536), raw BF16 weights in
    backward-natural K_BY_F orientation; activation via 1x32 PE-swizzle.
  * ``tp2_kbyf_1x32_blocking_sweep``         : TP2 (I_TP=768), backward-natural K_BY_F
    raw BF16 weights; activation via 1x32 PE-swizzle.
  * ``tp1_prequantized_blocking_sweep``      : TP1 (I_TP=1536), PRE-QUANTIZED MXFP8 x4
    weights (on-chip weight quantize + spill skipped); activation via WRAPX DGT.
  * ``tp2_prequantized_blocking_sweep``      : TP2 (I_TP=768),  pre-quantized (WRAPX).
  * ``tp1_prequantized_1x32_blocking_sweep`` : TP1 (I_TP=1536), pre-quantized MXFP8 x4
    weights swizzled with the 1x32 permutation; activation via 1x32 PE-swizzle.
  * ``tp2_prequantized_1x32_blocking_sweep`` : TP2 (I_TP=768),  pre-quantized 1x32.
``-k blocking_sweep`` runs all ten; ``-k 1x32_blocking_sweep`` runs the six 1x32
sweeps (regular + K_BY_F + pre-quantized); ``-k kbyf_1x32_blocking_sweep`` runs the
two backward-natural raw-BF16 sweeps; ``-k prequantized_blocking_sweep`` runs the
WRAPX pre-quantized sweeps; ``-k prequantized_1x32_blocking_sweep`` runs the 1x32
pre-quantized sweeps; ``-k "tp1_blocking_sweep and not 1x32 and not prequantized"``
runs only the DGT TP1 one.

Use ``--enable-perf-analysis`` (which writes ``neuron_test_output/qor_data_*.csv``
with per-config inference time / MFU) to pick the fastest strategy; do NOT pass
``--metric-output file`` without ``--output-directory`` (the collector errors).

Scope (per the current tuning target):
  * Fixed sequence length 8192, block_size in {2048, 4096} -> N in {4, 2} blocks,
    both >= num_shards(2) so every LNC2 core owns whole blocks.
  * Two TP degrees: TP1 (intermediate unsharded, I_TP=1536) and TP2 (I_TP=768).
  * Each GEMM phase is swept INDEPENDENTLY over its own block-tile counts, so a
    (gate/up, down) pair is a cross product, and each phase carries its own
    ``MatmulMxfp8KernelConfig``. The per-dimension sweep is
    ``TILES_IN_BLOCK_{M,N,K} = (total tiles) x [1, 1/2]`` with two exceptions:

    The M axis (block_size in both phases) keeps halving past total//2 down to a
    floor of 8 tiles, so B=4096 covers {32, 16, 8} rather than just {32, 16}:
    m=8 is the M block that wins at B=2048 and the one most likely to fit SBUF at
    B=4096. B=2048 (16 total) reaches the floor in one halving -> {16, 8}.

    The I_TP dimension occupies a DIFFERENT matmul axis in each phase:
      - gate/up: I_TP is the N axis (configurable tile_n). We ablate its tile size:
        TP2 (I_TP=768) tries tile_n in {256, 512}; TP1 (I_TP=1536) uses tile_n=512.
      - down: I_TP is the K axis, which the fwd kernel PINS to tile_k=512
        (kernel_assert), so TILES_IN_BLOCK_K is fixed at ceil(I_TP/512)
        (3 for TP1, 2 for TP2) and is NOT halved.
    tile_m (128) and tile_k (512) are pinned by the kernel; the other non-I_TP
    axis is H=4096 (gate/up K, down N), tiled at 512 -> 8 total tiles, swept {8, 4}.

  * TILES_IN_LOAD_{M,N} are reduced to a DGT-legal divisor of the (per-phase)
    block via the same rule the kernel's dropless impl uses.
  * SBUF pre-filter: every candidate is checked with the kernel's own
    ``_fits_sbuf`` on BOTH phases, and blocks that would overflow the state buffer
    are dropped instead of emitted as tests that can only fail at compile.
  * ``spill_reload`` is swept over {True, False}. It is per-phase on the matmul
    configs; this sweep sets both phases to the same value. It does not interact
    with the SBUF filter (spilled tiles land in ``private_hbm``), so both values
    multiply the same filtered candidate set.
  * ``reuse_spilled_weights`` is swept jointly with ``spill_reload`` as the pairs
    {(1,1), (1,0), (0,0)} -- config-level, and meaningless without spilling, so
    (0,1) is dropped. It also only matters when a core owns >1 block: each of the 2
    LNC2 cores takes (8192 // block_size) / 2 blocks, so B=2048 (N=4) reuses but
    B=4096 (N=2) does not (its reuse-on cells are dropped). Net grid growth is 1.5x
    at B=2048 and 1x at B=4096.

Test id encodes the spill flag and the per-phase tiling, e.g.
``tp2_B2048_sr1_ru1_gu_tn256_m8n3k4_l8x3_dn_m8n4k2_l8x2``:
  * ``tp``               : tp1 (I_TP=1536) or tp2 (I_TP=768).
  * ``B``                : block_size (tokens = 8192, fixed).
  * ``sr``               : spill_reload (0/1), applied to both phases.
  * ``ru``               : reuse_spilled_weights (0/1), applied to both phases.
  * ``gu_tn``            : gate/up tile_n (the I_TP-axis tile size).
  * ``gu_m/n/k``         : gate/up TILES_IN_BLOCK_M / _N / _K (M=B, N=I_TP, K=H).
  * ``gu_l<M>x<N>``      : gate/up TILES_IN_LOAD_M x TILES_IN_LOAD_N.
  * ``dn_*``             : same for the down phase (M=B, N=H, K=I_TP). down tile_n
                           is always 512, so it is omitted from the id.

Everything else (single_expert_dense, use_scale_packing, LNC2) is held fixed to the
Qwen3 deployment path so the only variables are the blocking/tiling, spill_reload,
the activation load mode (which the chosen sweep method fixes: DGT
fast_dma_transpose or 1x32 PE-swizzle), and the weight input form selected by the
method (forward-natural raw BF16, backward-natural K_BY_F raw BF16, or
pre-quantized MXFP8 x4).

Checkpoint policy for these benchmarks: only the gate/up activation is saved, in
the DIRECT (token-major [.., B, I_TP]) layout, so its store is a plain DMA with
no PE transpose; the scaled intermediate is not saved (the backward recomputes
it). This isolates the FFN + gate/up-store cost from the transposed-checkpoint
work, so the numbers are NOT directly comparable to a TRANSPOSED-checkpoint run.

NOTE: The SBUF filter bounds a block's static residency; it does not guarantee
compilation. block_size=4096 + fast_dma_transpose has a known partial-F-load-tile
compile failure for some tilings, so an occasional cell can still error at compile
rather than validate. Any such failure is a real finding (the filter's estimate
said it should fit), not expected noise -- unlike before the filter existed, when
roughly half the grid was expected to fail on state-buffer allocation.
"""

import functools
import math
from dataclasses import dataclass
from itertools import product
from typing import final

import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_config import (
    MatmulMxfp8KernelConfig,
    _fits_sbuf,
    _max_tiles_for_dgt,
)
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_constants import (
    BYTES_PER_DTYPE,
    PRECISION_FP32,
    SBUF_LIMIT_BYTES,
    TILE_SIZE_P_MAX_LOGICAL,
)
from nkilib_src.nkilib.experimental.mlp_mxfp8.common_utils import (
    L_TILE_K,
    MAX_TILES_IN_LOAD_M,
    TILE_M,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.fwd.blockwise_mm_forward_mxfp8 import (
    blockwise_mm_fwd_mxfp8,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.fwd.blockwise_mm_forward_mxfp8_torch import (
    blockwise_mm_fwd_mxfp8_torch_ref,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.moe_mxfp8_checkpoint_config import (
    CheckpointLayout,
    checkpoint_block_dims,
)

from test.integration.nkilib.experimental.moe_mxfp8.mxfp8_moe_fwd_test_utils import (
    build_mxfp8_moe_fwd_inputs,
)
from test.integration.nkilib.experimental.moe_mxfp8.mxfp8_moe_validated_common import (
    make_moe_validated_comparator,
)
from test.utils import common_dataclasses
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

bfloat16 = nl.bfloat16

# ============================================================================
# Fixed Qwen3-235B per-expert forward shape.
#   Gate+Up: [B, H] @ [H, I_TP]  -> M=B, K=H,    N=I_TP
#   Down:    [B, I_TP] @ [I_TP, H] -> M=B, K=I_TP,  N=H
# Sequence length is fixed; block_size and the per-phase tiling vary.
# ============================================================================
HIDDEN = 4096  # H
SEQ_LEN = 8192  # tokens (fixed)
EXPERT = 1
TOP_K = 1

# block_size in {2048, 4096} -> N = tokens // block_size in {4, 2}; both cores busy.
_BLOCK_SIZES = [2048, 4096]

# (label, I_TP): TP1 leaves the intermediate unsharded (moe_intermediate_size=1536);
# TP2 shards it in half (768).
_TP_CONFIGS = [("tp1", 1536), ("tp2", 768)]

# spill_reload is per-phase; this sweep applies one value to both phases. True first
# so each config's default lands next to its ablation in the report.
# reuse_spilled_weights carries the quantized weight spill buffers across blocks;
# meaningful only when spilling, so (spill off, reuse on) is dropped (3x grid, not 4x).
_SPILL_REUSE_OPTIONS = [(True, True), (True, False), (False, False)]


@dataclass
class BlockingParams:
    """Per-phase blocking overrides consumed by ``build_mxfp8_moe_fwd_inputs``."""

    gate_up: MatmulMxfp8KernelConfig
    down: MatmulMxfp8KernelConfig


def _halves(total):
    """Return ``[total, total // 2]`` deduped and floored at 1 (the ``[1, 1/2]`` rule)."""
    return sorted({max(1, total), max(1, total // 2)}, reverse=True)


# SBUF-fit parameters for the per-phase capacity filter below. These mirror
# ``generate_autotune_candidates`` exactly: the permissive mxfp8 k_bytes=2, an
# fp32 accumulator, and the 1.5 multiplier that accounts for the kernel reusing
# SBUF across loop iterations (the static estimate is conservative).
#
# Calibrated against a full 52-cell TP1 hardware run on trn3 B1: this predicts
# all 52 pass/fail outcomes exactly (27 fit, 25 overflow -- no false positives or
# negatives), where every observed failure was an NCC_IBIR229 / NCC_INLA001
# state-buffer allocation error. tile_k is pinned to L_TILE_K by the fwd kernel,
# so the tile_k-scaled limit collapses to a constant.
_SBUF_K_BYTES_MXFP8 = 2
_SBUF_OUT_BYTES = BYTES_PER_DTYPE[PRECISION_FP32]
_SBUF_LIMIT = SBUF_LIMIT_BYTES / (TILE_SIZE_P_MAX_LOGICAL / L_TILE_K) * 1.5


def _phase_fits_sbuf(tile_n, bm, bn, bk):
    """True if one GEMM phase's block fits SBUF, per the kernel's own estimator."""
    return _fits_sbuf(
        bm,
        bn,
        bk,
        TILE_M,
        tile_n,
        L_TILE_K,
        _SBUF_K_BYTES_MXFP8,
        _SBUF_OUT_BYTES,
        _SBUF_LIMIT,
    )


# Smallest TILES_IN_BLOCK_M worth sweeping (8 tiles = 1024 tokens per M block).
# The M chain halves down to this floor rather than stopping at total//2, so
# B4096 (32 total M tiles) also covers m=8 -- the M block that wins at B2048 and
# the one most likely to fit SBUF at B4096. B2048 (16 total) already reaches it.
_MIN_TILES_IN_BLOCK_M = 8


def _m_block_options(B):
    """TILES_IN_BLOCK_M options for a block size: halve from the total down to the floor.

    B=2048 -> [16, 8]; B=4096 -> [32, 16, 8]. Applies to both GEMM phases, whose
    M extent is block_size in each.
    """
    total = max(1, math.ceil(B / TILE_M))
    opts = {total}
    t = total
    while t // 2 >= _MIN_TILES_IN_BLOCK_M:
        t //= 2
        opts.add(t)
    return sorted(opts, reverse=True)


def _resolve_load(tiles_in_block, dgt_cap):
    """Reduce a full-block TILES_IN_LOAD to a DGT-legal divisor of the block.

    Mirrors the dropless impl's ``_resolve_tiles_in_load``: clamp to the block and
    the DGT transpose cap, then walk down to the largest divisor of the block.
    """
    load = min(tiles_in_block, dgt_cap)
    while load > 1 and tiles_in_block % load != 0:
        load -= 1
    return max(1, load)


def _itp_tile_n_variants(I_TP):
    """gate/up tile_n choices for the I_TP (N) axis, with the matching TILES_IN_BLOCK_N.

    TP1 (I_TP=1536) uses tile_n=512 (bn=3); TP2 (I_TP=768) ablates tile_n in
    {256, 512} (bn = I_TP // tile_n).
    """
    if I_TP == 1536:
        return [(512, 3)]
    return [(256, 3), (512, 2)]


def _phase_config(M, K, N, tile_n, bm, bn, bk, lm, ln) -> MatmulMxfp8KernelConfig:
    """Build a ``MatmulMxfp8KernelConfig`` for one GEMM phase from resolved tiles.

    Single source of truth for a per-phase config: used both to enumerate the sweep
    grid and to rebuild each config in the test from its serialized tile ints.
    """
    return MatmulMxfp8KernelConfig(
        M=M,
        K=K,
        N=N,
        tile_n=tile_n,
        TILES_IN_BLOCK_M=bm,
        TILES_IN_BLOCK_N=bn,
        TILES_IN_BLOCK_K=bk,
        TILES_IN_LOAD_M=lm,
        TILES_IN_LOAD_N=ln,
    )


def _gate_up_variants(B, H, I_TP):
    """Per-phase configs for gate/up (M=B, N=I_TP, K=H) over the [1, 1/2] grid.

    M (=ceil(B/tile_m)) and K (=ceil(H/tile_k)) are each swept {full, 1/2}; the
    I_TP/N axis ablates tile_n (and its fixed TILES_IN_BLOCK_N) per _itp_tile_n_variants.
    """
    m_dgt_cap = min(MAX_TILES_IN_LOAD_M, _max_tiles_for_dgt(TILE_M))
    variants = []
    for bm in _m_block_options(B):
        for bk in _halves(math.ceil(H / L_TILE_K)):
            for tile_n, bn in _itp_tile_n_variants(I_TP):
                if not _phase_fits_sbuf(tile_n, bm, bn, bk):
                    continue
                variants.append(
                    _phase_config(
                        B,
                        H,
                        I_TP,
                        tile_n=tile_n,
                        bm=bm,
                        bn=bn,
                        bk=bk,
                        lm=_resolve_load(bm, m_dgt_cap),
                        ln=_resolve_load(bn, _max_tiles_for_dgt(tile_n)),
                    )
                )
    return variants


def _down_variants(B, H, I_TP):
    """Per-phase configs for down (M=B, N=H, K=I_TP) over the [1, 1/2] grid.

    M (=ceil(B/tile_m)) and N (=ceil(H/tile_n)) are each swept {full, 1/2}. The
    I_TP/K axis is pinned to tile_k=512 by the kernel, so TILES_IN_BLOCK_K is fixed
    at ceil(I_TP/512) and not halved. down tile_n is always 512.
    """
    tile_n = 512
    bk = math.ceil(I_TP / L_TILE_K)
    m_dgt_cap = min(MAX_TILES_IN_LOAD_M, _max_tiles_for_dgt(TILE_M))
    n_dgt_cap = _max_tiles_for_dgt(tile_n)
    variants = []
    for bm in _m_block_options(B):
        for bn in _halves(math.ceil(H / tile_n)):
            if not _phase_fits_sbuf(tile_n, bm, bn, bk):
                continue
            variants.append(
                _phase_config(
                    B,
                    I_TP,
                    H,
                    tile_n=tile_n,
                    bm=bm,
                    bn=bn,
                    bk=bk,
                    lm=_resolve_load(bm, m_dgt_cap),
                    ln=_resolve_load(bn, n_dgt_cap),
                )
            )
    return variants


def _generate_sweep_params(tp, I_TP):
    """Enumerate one TP degree's per-GEMM [1, 1/2] tiling grid over the block sizes.

    gate/up and down are swept independently, so each param is the cross product of
    one gate/up variant and one down variant, taken at both ``spill_reload`` values.
    Spilling is orthogonal to the SBUF filter (spilled tiles go to private_hbm), so
    it multiplies the already-filtered blocking candidates rather than changing which
    ones survive.

    The param value is a flat dict of ints/bools (tp label, block_size, intermediate,
    spill_reload, and the effective per-phase tiles) so it serializes into
    ``--metric-output`` as one column per knob; the test rebuilds the two
    ``MatmulMxfp8KernelConfig``s from those ints via ``_phase_config``.
    """
    params = []
    for B in _BLOCK_SIZES:
        gate_up_variants = _gate_up_variants(B, HIDDEN, I_TP)
        down_variants = _down_variants(B, HIDDEN, I_TP)
        # Reuse needs >1 block per core; at B=4096 (N=2) each core owns one, so drop
        # the reuse-on cells there instead of emitting no-op duplicates.
        reuses_across_blocks = (SEQ_LEN // B) // 2 > 1
        spill_reuse_options = [(sr, ru) for sr, ru in _SPILL_REUSE_OPTIONS if reuses_across_blocks or not ru]
        for gu, dn in product(gate_up_variants, down_variants):
            for spill_reload, reuse_spilled_weights in spill_reuse_options:
                test_id = (
                    f"{tp}_B{B}_sr{int(spill_reload)}_ru{int(reuse_spilled_weights)}"
                    f"_gu_tn{gu.tile_n}_m{gu.TILES_IN_BLOCK_M}n{gu.TILES_IN_BLOCK_N}k{gu.TILES_IN_BLOCK_K}"
                    f"_l{gu.TILES_IN_LOAD_M}x{gu.TILES_IN_LOAD_N}"
                    f"_dn_m{dn.TILES_IN_BLOCK_M}n{dn.TILES_IN_BLOCK_N}k{dn.TILES_IN_BLOCK_K}"
                    f"_l{dn.TILES_IN_LOAD_M}x{dn.TILES_IN_LOAD_N}"
                )
                params.append(
                    pytest.param(
                        {
                            "tp": tp,
                            "intermediate": I_TP,
                            "block_size": B,
                            "spill_reload": spill_reload,
                            "reuse_spilled_weights": reuse_spilled_weights,
                            "gu_tile_n": gu.tile_n,
                            "gu_bm": gu.TILES_IN_BLOCK_M,
                            "gu_bn": gu.TILES_IN_BLOCK_N,
                            "gu_bk": gu.TILES_IN_BLOCK_K,
                            "gu_lm": gu.TILES_IN_LOAD_M,
                            "gu_ln": gu.TILES_IN_LOAD_N,
                            "dn_bm": dn.TILES_IN_BLOCK_M,
                            "dn_bn": dn.TILES_IN_BLOCK_N,
                            "dn_bk": dn.TILES_IN_BLOCK_K,
                            "dn_lm": dn.TILES_IN_LOAD_M,
                            "dn_ln": dn.TILES_IN_LOAD_N,
                        },
                        id=test_id,
                    )
                )
    return params


# One param list per TP degree so tp1 and tp2 run as separate test methods
# (each selectable/reportable on its own).
_TP1_SWEEP_PARAMS = _generate_sweep_params("tp1", 1536)
_TP2_SWEEP_PARAMS = _generate_sweep_params("tp2", 768)


# ============================================================================
# Test class
# ============================================================================


@pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN1, common_dataclasses.Platforms.TRN2])
@final
class TestMoeMxfp8FwdQwen3BlockingSweep:
    """Blocking/tiling sweep for the MXFP8 MoE forward on the Qwen3-235B TP1/TP2 shapes."""

    @staticmethod
    def _n_blocks(tokens, block_size):
        """Block count for the contiguous single-expert path (E=1, top_k=1)."""
        return math.ceil((tokens * TOP_K - (EXPERT - 1)) / block_size) + EXPERT - 1

    def _run(
        self,
        test_manager,
        platform_target,
        tp,
        intermediate,
        block_size,
        spill_reload,
        reuse_spilled_weights,
        gu_tile_n,
        gu_bm,
        gu_bn,
        gu_bk,
        gu_lm,
        gu_ln,
        dn_bm,
        dn_bn,
        dn_bk,
        dn_lm,
        dn_ln,
        use_1x32_pe_swizzle=False,
        prequantize_weights=False,
        weights_k_by_f=False,
    ):
        del tp  # part of the test id only; shape is fully determined by intermediate
        assert not (prequantize_weights and weights_k_by_f), (
            "K_BY_F selects raw BF16 weights and cannot be combined with pre-quantized MXFP8 weights"
        )
        T = SEQ_LEN
        H, I_TP = HIDDEN, intermediate
        # Benchmark checkpoint policy: save the gate/up activation in the DIRECT
        # (token-major [.., B, I_TP]) layout so the store is a plain DMA with no PE
        # transpose. (The scaled intermediate is no longer emitted at all.)
        save_gate_up_proj_act = True
        gate_up_proj_act_layout = CheckpointLayout.DIRECT

        # Rebuild the per-phase configs from the serialized (effective) tiles.
        # down tile_n is always 512 (H axis); gate/up tile_n is the swept I_TP-axis knob.
        blocking = BlockingParams(
            gate_up=_phase_config(block_size, H, I_TP, gu_tile_n, gu_bm, gu_bn, gu_bk, gu_lm, gu_ln),
            down=_phase_config(block_size, I_TP, H, 512, dn_bm, dn_bn, dn_bk, dn_lm, dn_ln),
        )

        build_kwargs = {
            "tokens": T,
            "hidden": H,
            "intermediate": I_TP,
            "expert": EXPERT,
            "block_size": block_size,
            "top_k": TOP_K,
            "blocking_params": blocking,
            # Swept knobs: applied to both GEMM phases by the kernel entry point.
            "spill_reload": spill_reload,
            "reuse_spilled_weights": reuse_spilled_weights,
            "use_scale_packing": True,
            # Activation load mode, chosen by the caller (one sweep method per mode):
            # DGT fast_dma_transpose (default) or the 1x32 PE-swizzle transpose.
            # K_BY_F raw weights require the PE-transpose family and therefore cannot
            # use fast_dma_transpose. Pre-quantized weights support either activation
            # load mode and are swizzled accordingly by the input builder.
            "fast_dma_transpose": not use_1x32_pe_swizzle and not weights_k_by_f,
            "use_1x32_pe_swizzle": use_1x32_pe_swizzle,
            "prequantize_weights": prequantize_weights,
            "weights_k_by_f": weights_k_by_f,
            "bias": False,
            "clamp_limits": None,
            "save_gate_up_proj_act": save_gate_up_proj_act,
            "gate_up_proj_act_layout": gate_up_proj_act_layout,
            "single_expert_dense": True,
        }

        if prequantize_weights:
            # The kernel gets the x4 weights; the golden gets the original fp32
            # standard-layout weights (returned alongside the inputs). Weight quant
            # is off the critical path here, so this sweep tunes the blocking for the
            # pre-quantized deployment path (weight spill/reuse are auto-bypassed).
            kernel_inputs, orig_gate_up_weight, orig_down_weight = build_mxfp8_moe_fwd_inputs(**build_kwargs)

            @functools.wraps(blockwise_mm_fwd_mxfp8_torch_ref)
            def _pq_torch_ref(**kwargs):
                kwargs["gate_up_proj_weight"] = torch.from_numpy(orig_gate_up_weight.astype(np.float32))
                kwargs["down_proj_weight"] = torch.from_numpy(orig_down_weight.astype(np.float32))
                return blockwise_mm_fwd_mxfp8_torch_ref(**kwargs)

            def input_gen(_):
                return kernel_inputs

            torch_ref = torch_ref_wrapper(_pq_torch_ref)
        else:

            def input_gen(_):
                return build_mxfp8_moe_fwd_inputs(**build_kwargs)

            torch_ref = torch_ref_wrapper(blockwise_mm_fwd_mxfp8_torch_ref)

        output_shapes = {
            "output_hidden_states": ((T, H), bfloat16),
        }
        N_blocks = self._n_blocks(T, block_size)
        if save_gate_up_proj_act:
            gate_up_ckpt_shape = (N_blocks, 2) + checkpoint_block_dims(gate_up_proj_act_layout, I_TP, block_size)
            output_shapes["gate_up_proj_act_checkpoint_T"] = (gate_up_ckpt_shape, bfloat16)

        def output_tensors(kernel_input):
            return {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in output_shapes.items()}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=blockwise_mm_fwd_mxfp8,
            torch_ref=torch_ref,
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

    @pytest.mark.parametrize("sweep_config", _TP1_SWEEP_PARAMS)
    def test_moe_mxfp8_fwd_qwen3_235B_tp1_blocking_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        sweep_config: dict,
    ):
        self._run(test_manager=test_manager, platform_target=platform_target, **sweep_config)

    @pytest.mark.parametrize("sweep_config", _TP2_SWEEP_PARAMS)
    def test_moe_mxfp8_fwd_qwen3_235B_tp2_blocking_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        sweep_config: dict,
    ):
        self._run(test_manager=test_manager, platform_target=platform_target, **sweep_config)

    # Same blocking grids as the DGT tp1/tp2 sweeps, but with the operand load mode
    # swapped to the 1x32 PE-swizzle transpose. Run in parallel with the DGT sweeps to
    # compare the two load paths on the same configs. Both TP degrees qualify: the
    # flexible-K 1x32 loader tiles K as 512 + a <512 remainder, so it handles TP2's
    # I_TP = 768 (down-GEMM K) as well as TP1's I_TP = 1536.
    @pytest.mark.parametrize("sweep_config", _TP1_SWEEP_PARAMS)
    def test_moe_mxfp8_fwd_qwen3_235B_tp1_1x32_blocking_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        sweep_config: dict,
    ):
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            use_1x32_pe_swizzle=True,
            **sweep_config,
        )

    @pytest.mark.parametrize("sweep_config", _TP2_SWEEP_PARAMS)
    def test_moe_mxfp8_fwd_qwen3_235B_tp2_1x32_blocking_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        sweep_config: dict,
    ):
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            use_1x32_pe_swizzle=True,
            **sweep_config,
        )

    # Newly enabled shared-forward/backward weight layout: raw BF16 weights arrive
    # in backward-natural K_BY_F orientation and forward consumes them directly,
    # avoiding a second model-weight copy. K_BY_F is incompatible with the fast DGT
    # path, so sweep it with the intended 1x32 activation load. Keeping the same TP1
    # and TP2 grids as the forward-natural 1x32 methods gives a direct performance
    # comparison and finds the best blocking config for each deployment shape.
    @pytest.mark.parametrize("sweep_config", _TP1_SWEEP_PARAMS)
    def test_moe_mxfp8_fwd_qwen3_235B_tp1_kbyf_1x32_blocking_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        sweep_config: dict,
    ):
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            weights_k_by_f=True,
            use_1x32_pe_swizzle=True,
            **sweep_config,
        )

    @pytest.mark.parametrize("sweep_config", _TP2_SWEEP_PARAMS)
    def test_moe_mxfp8_fwd_qwen3_235B_tp2_kbyf_1x32_blocking_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        sweep_config: dict,
    ):
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            weights_k_by_f=True,
            use_1x32_pe_swizzle=True,
            **sweep_config,
        )

    # Same blocking grids as the DGT tp1/tp2 sweeps, but with PRE-QUANTIZED weights:
    # the kernel receives forward-natural MXFP8 x4 weights + packed scales, so the
    # on-chip weight quantize + spill is skipped and the matmul consumes the x4 data
    # directly. This is the deployment/inference use case (weights quantized once,
    # offline). Weights are already WRAPX-swizzled, so the activation still loads via
    # the WRAPX (fast_dma_transpose) path. Run these to pick the best blocking for the
    # pre-quantized path and compare its inference time against the on-the-fly-quantize
    # DGT sweep on the identical grid. Select with `-k prequantized_blocking_sweep`.
    @pytest.mark.parametrize("sweep_config", _TP1_SWEEP_PARAMS)
    def test_moe_mxfp8_fwd_qwen3_235B_tp1_prequantized_blocking_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        sweep_config: dict,
    ):
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            prequantize_weights=True,
            **sweep_config,
        )

    @pytest.mark.parametrize("sweep_config", _TP2_SWEEP_PARAMS)
    def test_moe_mxfp8_fwd_qwen3_235B_tp2_prequantized_blocking_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        sweep_config: dict,
    ):
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            prequantize_weights=True,
            **sweep_config,
        )

    # PRE-QUANTIZED weights loaded via the 1x32 PE-swizzle path: the x4 weights are
    # swizzled with the 1x32 (contiguous-K) permutation to match the 1x32 activation
    # load, instead of the WRAPX permutation used by the DGT pre-quantized sweep above.
    # Compare against the regular (on-the-fly-quantize) 1x32 sweep on the identical
    # grid to isolate the weight-prequantization saving on the 1x32 load path. Select
    # with `-k prequantized_1x32_blocking_sweep`.
    @pytest.mark.parametrize("sweep_config", _TP1_SWEEP_PARAMS)
    def test_moe_mxfp8_fwd_qwen3_235B_tp1_prequantized_1x32_blocking_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        sweep_config: dict,
    ):
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            prequantize_weights=True,
            use_1x32_pe_swizzle=True,
            **sweep_config,
        )

    @pytest.mark.parametrize("sweep_config", _TP2_SWEEP_PARAMS)
    def test_moe_mxfp8_fwd_qwen3_235B_tp2_prequantized_1x32_blocking_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        sweep_config: dict,
    ):
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            prequantize_weights=True,
            use_1x32_pe_swizzle=True,
            **sweep_config,
        )
