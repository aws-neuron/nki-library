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

"""GpSIMD top-k kernel using the single-instruction nisa.topk primitive.

Computes torch.topk(inp, k, dim=-1, largest=True) over inp[BxS, vocab] by mapping each
vocab row onto the 16-partition "snake" layout nisa.topk consumes (one top-k per group
of 16 partitions, 8 rows per 128-partition tile) with a blocked contiguous load, then
de-snaking the K (value, snake-position) pairs, optionally sorting them descending, and
remapping snake positions back to vocab indices on-chip.
"""

from dataclasses import dataclass
from typing import Tuple

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil, get_verified_program_sharding_info

# Smallest finite bfloat16 value, used to pad snake slots that hold no real data.
BFLOAT16_MIN = -3.3895314e38  # most-negative bf16; padded slots never enter the top-k.
# Most-negative float32, used to pad the sort buffer so padding never wins.
FLOAT32_MIN = float(np.finfo(np.float32).min)

# nisa.topk runs one independent top-k per group of 16 partitions.
PARTS_PER_GROUP = 16
# Hardware max partition dimension (8 groups of 16).
PMAX = 128
GROUPS_PER_TILE = PMAX // PARTS_PER_GROUP  # 8


@dataclass(frozen=True, eq=True)
class GpsimdTopkConfig(nl.NKIObject):
    """Configuration for the GpSIMD nisa.topk kernel.

    Attributes:
        BxS: Combined batch*sequence dimension (number of rows).
        vocab_size: Length of the dimension reduced over (n for nisa.topk).
        k: Number of largest elements to return.
        sorted: Whether output must be in descending order.
        inp_dtype: Input/value data type (must be bfloat16).
        index_dtype: Output index data type (uint32).
        n_prgs: Number of logical cores (SPMD programs) over BxS.
        prg_id: Program id (set at config-build time for BxS == 1 path).
        per_lnc_BxS: Rows handled per logical core.
        out_shape: Logical output shape (mirrors input leading dims + k).
    """

    BxS: int
    vocab_size: int
    k: int
    sorted: bool
    inp_dtype: np.dtype
    index_dtype: np.dtype
    n_prgs: int
    prg_id: int
    per_lnc_BxS: int
    out_shape: tuple

    @property
    def topk_config(self):
        """Self-reference so the shared torch_ref (which reads .topk_config.k /
        .topk_config.sorted) works against this config without a separate object."""
        return self


def create_gpsimd_topk_config(
    inp_shape: Tuple,
    inp_dtype: np.dtype,
    k: int,
    sorted: bool = True,
    num_programs: int = 2,
) -> GpsimdTopkConfig:
    """Build a GpsimdTopkConfig from an input shape (2D or 3D) and parameters."""
    BxS = 1
    for d in inp_shape[:-1]:
        BxS *= d
    vocab_size = inp_shape[-1]
    out_shape = tuple(list(inp_shape[:-1]) + [k])

    n_prgs = num_programs
    prg_id = 0
    if BxS == 1:
        n_prgs = 1
        prg_id = 0

    per_lnc_BxS = div_ceil(BxS, n_prgs)

    return GpsimdTopkConfig(
        BxS=BxS,
        vocab_size=vocab_size,
        k=k,
        sorted=sorted,
        inp_dtype=inp_dtype,
        index_dtype=nl.uint32,
        n_prgs=n_prgs,
        prg_id=prg_id,
        per_lnc_BxS=per_lnc_BxS,
        out_shape=out_shape,
    )


def _invalid_desnake_col_runs(k: int, k_cols: int, k_pad: int) -> Tuple[Tuple[int, int], ...]:
    """Contiguous runs of de-snaked cv columns that hold UNWRITTEN topk output slots.

    nisa.topk writes only snake positions [0, k); positions [k, k_pad) are unwritten
    and carry garbage. The de-snake places snake position pos at cv column
    j = (pos % 16) * k_cols + (pos // 16), i.e. cv column j holds snake position
    pos = (j // k_cols) + 16 * (j % k_cols). This returns the contiguous (start, length)
    runs of cv columns j whose pos >= k, so the kernel can memset each run to
    FLOAT32_MIN (a legal full-partition free-dim write) before the descending sort.

    All arguments are compile-time constants, so this resolves fully at trace time.
    Computed in a module-level helper (NOT inline in the @nki.jit body, where the
    parser frontend rejects the list-building expression as an "unsupported expression").
    """
    runs = []
    in_run = False
    run_start = 0
    run_len = 0
    for col_idx in range(k_pad):
        col_quotient, col_remainder = divmod(col_idx, k_cols)
        pos = col_quotient + PARTS_PER_GROUP * col_remainder
        if pos >= k:
            if in_run:
                run_len += 1
            else:
                in_run = True
                run_start = col_idx
                run_len = 1
        elif in_run:
            runs.append((run_start, run_len))
            in_run = False
            run_len = 0
    if in_run:
        runs.append((run_start, run_len))
    return tuple(runs)


def _valid_desnake_col_runs(k: int, k_cols: int, k_pad: int) -> Tuple[Tuple[int, int, int], ...]:
    """Contiguous runs mapping VALID de-snaked cv columns to compacted output columns.

    The complement of _invalid_desnake_col_runs: cv column j holds snake-output
    position pos = (j // k_cols) + 16 * (j % k_cols), which is a real topk output iff
    pos < k. When k_pad > k (k % 16 != 0) the valid columns are NOT a contiguous prefix
    -- they are scattered among the k_pad columns, interleaved with the unwritten
    garbage columns. The unsorted fast path must COMPACT the valid columns to the
    contiguous output prefix [0, k). Each returned run is (src_col_start,
    dst_col_start, length): copy cv[:, src:src+length] -> out[:, dst:dst+length].

    For k % 16 == 0 (k_pad == k) the de-snake maps positions [0, k) onto cv columns
    [0, k), so the valid columns are already a contiguous prefix and this collapses to
    a SINGLE run (0, 0, k) (a full-width copy). Likewise for k_cols == 1 (k <= 16),
    where cv column j holds position j, so positions [0, k) occupy cv columns [0, k).

    All arguments are compile-time constants, so this resolves fully at trace time.
    Computed in a module-level helper (NOT inline in the @nki.jit body, where the
    parser frontend rejects the list-building expression as an "unsupported expression").
    """
    runs = []
    dst_col = 0
    in_run = False
    run_src_start = 0
    run_dst_start = 0
    run_len = 0
    for col_idx in range(k_pad):
        col_quotient, col_remainder = divmod(col_idx, k_cols)
        pos = col_quotient + PARTS_PER_GROUP * col_remainder
        if pos < k:
            if in_run:
                run_len += 1
            else:
                in_run = True
                run_src_start = col_idx
                run_dst_start = dst_col
                run_len = 1
            dst_col += 1
        elif in_run:
            runs.append((run_src_start, run_dst_start, run_len))
            in_run = False
            run_len = 0
    if in_run:
        runs.append((run_src_start, run_dst_start, run_len))
    return tuple(runs)


def _validate_gpsimd_topk(config: GpsimdTopkConfig) -> None:
    """Validate that the requested shape satisfies every nisa.topk constraint."""
    n = config.vocab_size
    k = config.k
    kernel_assert(config.inp_dtype == nl.bfloat16, "gpsimd_topk requires bfloat16 input")
    kernel_assert(8 <= n < 65536, f"gpsimd_topk requires 8 <= vocab ({n}) < 65536")
    kernel_assert(1 <= k < 32768, f"gpsimd_topk requires 1 <= k ({k}) < 32768")
    kernel_assert(k <= n, f"gpsimd_topk requires k ({k}) <= vocab ({n})")


@nki.jit
def gpsimd_topk(inp: nl.NkiTensor, config: GpsimdTopkConfig) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """Top-k over the last dimension using the GpSIMD nisa.topk instruction.

    Dimensions:
        BxS: number of rows (flattened batch*sequence)
        V:   vocab size (reduction dimension), 8 <= V < 65536
        k:   number of largest elements, 1 <= k <= V

    Args:
        inp: [BxS, V] bfloat16 input tensor in HBM.
        config: GpsimdTopkConfig describing the problem and sharding.

    Returns:
        (topk_values [BxS, k] bf16, topk_indices [BxS, k] uint32), both in HBM.
        Descending along the last dim when config.sorted is True (default);
        arbitrary order within each row when config.sorted is False.

    Notes:
        - Each vocab row is loaded into its 16-partition nisa.topk snake with a
          BLOCKED / contiguous DMA: partition p reads the contiguous HBM run
          inp[row, p*n_cols:(p+1)*n_cols] (free-stride 1), so snake position
          s = p + 16*c holds vocab index p*n_cols + c. The 16-partition snake
          LAYOUT is mandated by nisa.topk, but the ORDER of the placement is a
          free bijection (any order yields the same top-k value set). The blocked
          order is chosen so the load is contiguous; the alternative "snake
          position i == vocab index i" fill gives an identity remap but forces a
          transpose-on-load (.ap [[1,16],[16,n]], free-stride 16 -> non-contiguous
          per partition), which is avoided here. The returned snake-position
          indices are remapped back to vocab space on-chip before the index store.
        - 8 rows (groups of 16 partitions) are processed per nisa.topk call.
        - The hardware nisa.topk output order is not relied upon: the K values +
          paired snake-position indices are de-snaked and then sorted descending
          on-chip; the snake->vocab index remap is applied to the final k indices.
        - Two-phase structure: phase 1 runs nisa.topk per 8-row tile and
          de-snakes the K (value, index) pairs into per-row HBM buffers; phase 2
          runs the descending sort ONCE over up to 128 rows (one row per
          partition) instead of once per 8-row tile. max8 / nc_match_replace8 /
          nc_n_gather are per-partition free-dim ops, so widening the sort from
          8 to up to 128 partitions is free Vector-engine parallelism and removes
          the redundant per-tile sort passes (the measured HW bottleneck).
        - config.sorted gates only the phase-2 descending sort. When False the
          sort is skipped and the K results are compacted to [0, k) in arbitrary
          order; the value set and value<->index pairing are unchanged.

    Pseudocode:
        n_cols = ceil(V / 16)            # blocked snake free width
        for tile in range(ceil(BxS / 8)):          # 8 rows per 128-partition tile
            snake[p, c] = inp[row, p*n_cols + c]   # blocked contiguous load
            val_snake, idx_snake = nisa.topk(snake, n=16*n_cols)  # snake positions
            asc_val[row], asc_idx[row] = de_snake(val_snake, idx_snake)  # to HBM
        for row_tile in range(ceil(BxS / 128)):    # up to 128 rows in parallel
            v, s = reload(asc_val, asc_idx)
            if config.sorted:
                v, s = sort_descending(v, s)         # max8 / match_replace8 / gather
            else:
                v, s = compact_valid_columns(v, s)   # drop padding cols, keep [0, k)
            idx = (s % 16) * n_cols + (s // 16)      # snake position -> vocab index
            topk_values[row_tile], topk_indices[row_tile] = v[:, :k], idx[:, :k]
        return topk_values, topk_indices
    """
    _validate_gpsimd_topk(config)

    BxS = config.BxS
    vocab = config.vocab_size
    k = config.k
    index_dtype = config.index_dtype

    # Resolve runtime sharding (LNC). BxS == 1 collapses to a single program.
    shard_info = get_verified_program_sharding_info("gpsimd_topk", (0, 1), 2)
    if BxS > 1:
        n_prgs = shard_info[1]
        prg_id = shard_info[2]
    else:
        n_prgs = 1
        prg_id = 0

    per_lnc_BxS = div_ceil(BxS, n_prgs)

    topk_values = nl.ndarray((BxS, k), dtype=inp.dtype, buffer=nl.shared_hbm)
    topk_indices = nl.ndarray((BxS, k), dtype=index_dtype, buffer=nl.shared_hbm)

    # Snake free-dim sizes (BLOCKED layout: snake[p, c] = inp[row, p*n_cols + c]).
    n_cols = div_ceil(vocab, PARTS_PER_GROUP)  # src free dim; >= ceil(n/16)
    full_n = PARTS_PER_GROUP * n_cols  # FULL snake size; this is the n passed to nisa.topk
    # Blocked-layout valid/padding boundary: partitions 0..q_full-1 are fully valid
    # (n_cols real vocab elements each), partition q_full holds the first r_partial
    # valid elements, and the rest are padding. r_partial == 0 (=> q_full == 16) when
    # 16 | vocab. Padding only exists when vocab % 16 != 0 (has_pad).
    q_full = vocab // n_cols  # number of fully-valid partitions
    r_partial = vocab % n_cols  # valid elements in the partial partition q_full
    has_pad = full_n != vocab  # True iff some snake slots are padding (vocab % 16 != 0)

    inp_flat = inp.reshape((BxS * vocab,))

    # The hardware nisa.topk output snake layout differs from a simple ascending
    # snake (and from the simulator), so the topk output ORDER is treated as
    # unspecified: only the SET of K values + their paired snake-position indices is
    # trusted. Per tile, the K outputs are de-snaked into a contiguous [rows, k_pad]
    # tile (via a private-HBM round trip with one proven 2D strided .ap per group),
    # sorted into descending order on-chip with the proven max8 / nc_match_replace8 /
    # nc_n_gather sequence, and finally the snake-position -> vocab-index remap (see
    # the index-remap block below) is applied to the k indices. The remap is NOT an
    # identity here because the blocked load places vocab index p*n_cols+c at snake
    # position p+16*c (not at position == vocab index).
    #
    # De-snake all k_cols snake columns (a full [16, k_cols] block) with one proven
    # 2D strided .ap per group; the buffers are padded to k_pad = k_cols*16 (>= k)
    # so the partial last column needs no special-case 1D .ap (a 1D .ap store was
    # observed to drop to zero on hardware). Both HBM buffers are float32: a strided
    # .ap SBUF->HBM store was observed to zero out for the narrow dtypes (bf16
    # values, uint32 indices), so the DMA auto-casts up to float32 (indices < 65536
    # are exact and cast back to uint32 on the final store).
    k_cols = div_ceil(k, PARTS_PER_GROUP)
    k_pad = k_cols * PARTS_PER_GROUP  # >= k; padded snake-column width

    # --- Fast-vs-safe de-snake DMA selection (correctness gate) ----------------
    # CORRECTNESS BUG (low/odd batch): the phase-1 de-snake stores to private_hbm and
    # the phase-2 reload of the SAME region were issued with dge_mode.none (descriptors
    # pre-generated out-of-band, store taken off the GpSIMD engine) AND asc_val_hbm was
    # bf16. On a PARTIAL phase-1 tile -- one with par_dim < 128 (n_rows < 8), e.g. a
    # single-row par_dim==16 tile produced by an ODD per_lnc_BxS shard -- that combo let
    # the reload RACE the store and/or the narrow-dtype strided store silently zero on
    # hardware, corrupting both values and indices. It only manifests in the fully
    # overlapped device run; the simulator and the serialized device-dump/debugger build
    # never reproduce it (every per-op SBUF dump is correct yet the device output is
    # wrong). All shards have ONLY full 128-partition tiles iff per_lnc_BxS is a multiple
    # of GROUPS_PER_TILE AND no shard is clamped (BxS == n_prgs * per_lnc_BxS); that is
    # the case for every batched/GPT-OSS shape. fast_dma_safe is that compile-time
    # predicate: when True keep the original fast path (dge_mode.none + bf16 value
    # buffer, halving the value-bounce bytes); when False (any partial/single-row tile)
    # use the proven-safe path -- default DMA mode (SWDGE keeps the store ordered with
    # its top-k producer and the reload) and a float32 value buffer (4-byte elements
    # round-trip correctly under the strided store; the reload casts bf16->f32 anyway).
    fast_dma_safe = (per_lnc_BxS % GROUPS_PER_TILE == 0) and (BxS == n_prgs * per_lnc_BxS)
    # Fast path: dge_mode.none (descriptors generated off-GpSIMD, max load/compute
    # overlap). Safe path: dge_mode.unknown -> let the COMPILER pick the DGE mode. The
    # compiler-selected mode preserves the de-snake-store -> reload ordering on the SAME
    # private_hbm region (validated correct on hardware for all small/odd shapes); an
    # explicitly-forced SWDGE was observed to STILL drop values on the par_dim==16 case,
    # so do not pin it. HWDGE cannot generate the strided de-snake descriptor pattern
    # ([NCC_IBIR098]), so it is never used for the de-snake.
    desnake_dge = nisa.dge_mode.none if fast_dma_safe else nisa.dge_mode.unknown
    asc_val_dtype = nl.bfloat16 if fast_dma_safe else nl.float32
    asc_val_hbm = nl.ndarray((BxS, k_pad), dtype=asc_val_dtype, buffer=nl.private_hbm)
    asc_idx_hbm = nl.ndarray((BxS, k_pad), dtype=nl.float32, buffer=nl.private_hbm)
    asc_val_flat = asc_val_hbm.reshape((BxS * k_pad,))
    asc_idx_flat = asc_idx_hbm.reshape((BxS * k_pad,))

    lnc_row_start = prg_id * per_lnc_BxS
    n_tiles = div_ceil(per_lnc_BxS, GROUPS_PER_TILE)
    # This core's actual row range (clamped to BxS), used by both phases.
    lnc_row_end = min(lnc_row_start + per_lnc_BxS, BxS)

    # --- Coalesced + double-buffered phase-1 load (the load-cadence fix) ------
    # The per-tile blocked snake load was the phase-1 pacing item: 8 separate
    # HWDGE transfers, each ~1128-cyc issue interval, serialized one-per-tile on
    # the DMA queue while the ~100-cyc nisa.topk could not hide that latency.
    # Fix: COALESCE several tiles per load (fewer, larger HWDGE transfers so
    # load throughput, not per-tile latency, paces phase 1) while keeping
    # load<->compute OVERLAP via DOUBLE BUFFERING (a single whole-core DMA was
    # tried and regressed: it forced every topk to wait for the full load, the
    # opposite of overlap). When 16 | vocab and this core's n_tiles tiles are
    # ALL full (no per-tile clamping), each full tile is exactly 128*n_cols ==
    # 8*vocab contiguous HBM elements, and consecutive tiles are back-to-back, so
    # a chunk of CHUNK_TILES tiles is ONE contiguous HBM block loadable in a
    # single big HWDGE transfer into a [128, CHUNK_TILES, n_cols] SBUF buffer
    # (tile t-in-chunk at free offset t*n_cols). The blocked-layout offset for
    # (chunk-tile t, partition pp, col c) is
    #   base + t*8*vocab + (pp//16)*16*n_cols + (pp%16)*n_cols + c
    #     = base + t*(128*n_cols) + pp*n_cols + c      [vocab == 16*n_cols],
    # i.e. partition stride n_cols, tile stride 128*n_cols, free stride 1.
    # Two chunk buffers are alternated: while the GpSIMD top-k drains chunk c the
    # DMA prefetches chunk c+1, so loads run concurrently with compute and the
    # ~8 serialized per-tile completions become ceil(n_tiles/CHUNK_TILES) larger
    # overlapped transfers. Each tile's topk reads its slice
    # chunk_buf[:, t, :] (free-contiguous, identical layout to the old per-tile
    # src_snake), so the snake->vocab remap is unchanged.
    fast_coalesced = (not has_pad) and (per_lnc_BxS % GROUPS_PER_TILE == 0) and (lnc_row_start + per_lnc_BxS <= BxS)
    # Tiles coalesced into one DMA. 2 tiles == 128*2*n_cols bf16 (>=2KB/partition,
    # bandwidth-efficient) and ceil(n_tiles/2) chunks keeps the pipeline deep so
    # compute overlaps the next prefetch.
    CHUNK_TILES = 1
    n_chunks = div_ceil(n_tiles, CHUNK_TILES)
    # Engines to round-robin the per-chunk HWDGE loads across, so consecutive
    # loads land on DIFFERENT DGE sequencer queues and run concurrently instead
    # of serializing one-after-another on a single sync queue (this is the
    # "split across more DMA queues" lever; lets the FIRST topk start sooner).
    _load_engines = [nisa.engine.sync, nisa.engine.scalar]
    # N_PREFETCH_BUFS physically distinct buffers => up to N_PREFETCH_BUFS-1
    # chunks in flight (no anti-dependency forcing a later load to wait on an
    # earlier consumer's last read). 3 buffers keeps two loads in flight while
    # the GpSIMD top-k drains the third.
    N_PREFETCH_BUFS = min(3, n_chunks)
    chunk_bufs = []
    if fast_coalesced:
        for _b in range(N_PREFETCH_BUFS):
            chunk_bufs.append(nl.ndarray((PMAX, CHUNK_TILES, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf))
        # Prime the pipeline: prefetch the first N_PREFETCH_BUFS-1 chunks before the
        # tile loop starts (priming the multi-buffer pipeline), each a coalesced
        # HWDGE transfer (partition stride n_cols, tile stride 128*n_cols, free
        # stride 1) over its contiguous HBM block, round-robined across engines.
        # The last chunk may hold fewer than CHUNK_TILES tiles; clamp the tile count
        # so the DMA never over-reads past this core's region.
        n_prime = min(N_PREFETCH_BUFS - 1, n_chunks)
        for pc in range(n_prime):
            prime_first_tile = pc * CHUNK_TILES
            prime_row_start = lnc_row_start + prime_first_tile * GROUPS_PER_TILE
            prime_tiles = min(CHUNK_TILES, n_tiles - prime_first_tile)
            nisa.dma_copy(
                dst=chunk_bufs[pc % N_PREFETCH_BUFS][nl.ds(0, PMAX), nl.ds(0, prime_tiles), nl.ds(0, n_cols)],
                src=inp_flat.ap(
                    pattern=[[n_cols, PMAX], [PMAX * n_cols, prime_tiles], [1, n_cols]],
                    offset=prime_row_start * vocab,
                ),
                dge_mode=nisa.dge_mode.hwdge,
                engine=_load_engines[pc % 2],
            )

    # =====================================================================
    # PHASE 1 -- per-8-row tile: blocked load -> nisa.topk -> de-snake the
    # K (value, snake-index) pairs into the contiguous [BxS, k_pad] HBM
    # buffers asc_val_hbm / asc_idx_hbm. The expensive descending sort is
    # NOT done here: it runs ONCE in phase 2 over all this core's rows
    # (one row per partition, up to 128) instead of n_tiles serial sorts
    # that each used only n_rows (<= 8) of the 128 partitions. The sort is a
    # per-partition free-dim operation, so widening it from 8 to up to 128
    # partitions is free Vector-engine parallelism and removes the (n_tiles-1)
    # redundant serial sort passes -- the measured HW bottleneck.
    # =====================================================================
    for tile_idx in nl.sequential_range(n_tiles):
        tile_row_start = lnc_row_start + tile_idx * GROUPS_PER_TILE
        # Rows handled by this tile, clamped to this core's shard and to BxS.
        tile_row_end = min(tile_row_start + GROUPS_PER_TILE, lnc_row_end)
        n_rows = tile_row_end - tile_row_start
        if n_rows <= 0:
            continue

        par_dim = n_rows * PARTS_PER_GROUP  # multiple of 16

        # --- Blocked / contiguous load of each row into its 16-partition snake ---
        # snake[base + p, c] = inp[row, p*n_cols + c]: partition p reads the
        # CONTIGUOUS HBM run inp[row, p*n_cols:(p+1)*n_cols] (.ap partition-stride
        # n_cols, free-stride 1). Snake position s = p + 16*c therefore holds vocab
        # index p*n_cols + c; the returned indices are remapped back to vocab space
        # after the sort (see the index-remap block below). The snake LAYOUT is
        # required by nisa.topk; this blocked placement ORDER is a free choice (any
        # bijection gives the same top-k value set) picked for a contiguous load.
        if fast_coalesced:
            chunk_idx = tile_idx // CHUNK_TILES
            tile_in_chunk = tile_idx % CHUNK_TILES
            # Keep the pipeline (N_PREFETCH_BUFS-1 chunks deep) full: at the FIRST tile
            # of the current chunk, prefetch the chunk that is (N_PREFETCH_BUFS-1) ahead
            # into its rolling buffer slot, so its (coalesced) HWDGE load overlaps the
            # GpSIMD top-k of the chunks already resident. The destination slot differs
            # from every slot still being consumed, so there is no anti-dependency that
            # forces the load to wait. Loads round-robin across two DGE queues so two
            # are in flight concurrently.
            prefetch_chunk = chunk_idx + (N_PREFETCH_BUFS - 1)
            if tile_in_chunk == 0 and prefetch_chunk < n_chunks:
                pf_first_tile = prefetch_chunk * CHUNK_TILES
                pf_row_start = lnc_row_start + pf_first_tile * GROUPS_PER_TILE
                # Clamp to the tiles actually present (last chunk may be partial) so the
                # prefetch never over-reads past this core's region.
                pf_tiles = min(CHUNK_TILES, n_tiles - pf_first_tile)
                nisa.dma_copy(
                    dst=chunk_bufs[prefetch_chunk % N_PREFETCH_BUFS][
                        nl.ds(0, PMAX), nl.ds(0, pf_tiles), nl.ds(0, n_cols)
                    ],
                    src=inp_flat.ap(
                        pattern=[[n_cols, PMAX], [PMAX * n_cols, pf_tiles], [1, n_cols]],
                        offset=pf_row_start * vocab,
                    ),
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=_load_engines[prefetch_chunk % 2],
                )
            # This tile views its [128, n_cols] slice of the prefetched chunk buffer
            # (free-contiguous, identical to the old per-tile src_snake). par_dim == PMAX.
            src_snake = chunk_bufs[chunk_idx % N_PREFETCH_BUFS][nl.ds(0, PMAX), tile_in_chunk, nl.ds(0, n_cols)]
        elif not has_pad:
            src_snake = nl.ndarray((par_dim, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            # FAST PATH FALLBACK (16 | vocab but the coalesced precondition does not
            # hold, e.g. a partial last tile or a clamped shard). The whole-tile
            # blocked load is ONE contiguous HBM run: for partition pp = g*16 + p the
            # source offset is row*vocab + p*n_cols
            #   = (tile_row_start + pp//16)*vocab + (pp%16)*n_cols
            #   = tile_row_start*vocab + (pp//16)*16*n_cols + (pp%16)*n_cols   [vocab=16*n_cols]
            #   = tile_row_start*vocab + n_cols*(16*(pp//16) + pp%16)
            #   = tile_row_start*vocab + n_cols*pp,
            # i.e. partition stride n_cols, free stride 1 over the contiguous block
            # inp_flat[tile_row_start*vocab : tile_row_start*vocab + par_dim*n_cols].
            # Emitted as a SINGLE par_dim-partition DMA per tile. HWDGE on the Sync
            # engine keeps descriptor generation OFF GpSIMD so the load overlaps the
            # previous tile's top-k. No memset (every snake slot is a real vocab elt).
            nisa.dma_copy(
                dst=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
                src=inp_flat.ap(pattern=[[n_cols, par_dim], [1, n_cols]], offset=tile_row_start * vocab),
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
            )
        else:
            src_snake = nl.ndarray((par_dim, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            # vocab % 16 != 0: snake positions whose vocab index >= vocab are padding
            # (partition q_full's tail, plus any fully-padded higher partitions).
            # memset the whole tile to BFLOAT16_MIN so every padded slot is guaranteed
            # to lose the top-k; the valid contiguous prefix is then DMA'd over the top.
            nisa.memset(dst=src_snake, value=BFLOAT16_MIN)

            for g in nl.affine_range(n_rows):
                row = tile_row_start + g
                base = g * PARTS_PER_GROUP
                row_off = row * vocab
                # Bulk: the q_full fully-valid partitions in one contiguous DMA. Reads
                # inp_flat[row_off : row_off + q_full*n_cols]; q_full*n_cols <= vocab so
                # there is no over-read past this row.
                if q_full > 0:
                    nisa.dma_copy(
                        dst=src_snake[nl.ds(base, q_full), nl.ds(0, n_cols)],
                        src=inp_flat.ap(pattern=[[n_cols, q_full], [1, n_cols]], offset=row_off),
                    )
                # Tail: partition q_full holds the first r_partial valid elements (only
                # present when vocab % 16 != 0). One tiny contiguous DMA into a single
                # partition; reads inp_flat[row_off + q_full*n_cols : row_off + vocab].
                if r_partial != 0:
                    nisa.dma_copy(
                        dst=src_snake[nl.ds(base + q_full, 1), nl.ds(0, r_partial)],
                        src=inp_flat.ap(pattern=[[1, r_partial]], offset=row_off + q_full * n_cols),
                    )

        # --- Run the GpSIMD top-k over the FULL snake (n = full_n) ---
        # nisa.topk derives K from val_dst free dim, so val/idx must be [par_dim, k];
        # allocate the padded width k_pad so the partial last column is fully present
        # and de-snakeable with the 2D .ap below.
        # n = full_n (NOT vocab): with the blocked layout the valid vocab elements are
        # not a contiguous prefix of snake positions, so all full_n snake slots are
        # active. Padded slots hold BFLOAT16_MIN and never win the top-k.
        val_snake = nl.ndarray((par_dim, k_pad), dtype=nl.bfloat16, buffer=nl.sbuf)
        idx_snake = nl.ndarray((par_dim, k_pad), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.topk(val_dst=val_snake[:, nl.ds(0, k)], idx_dst=idx_snake[:, nl.ds(0, k)], src=src_snake, n=full_n)
        # When k_pad > k the topk leaves the unwritten snake positions [k, k_pad) with
        # garbage. Those slots live at val_snake[pos % 16, pos // 16] (column pos // 16 <
        # k_cols, partition pos % 16) -- a PARTITION-SUBRANGE that a phase-1 full-partition
        # memset cannot target (and a partition-subrange memset is rejected by the backend).
        # So the padding is NOT masked here; each phase-2 sort path masks it after the
        # de-snake reload, where the [n_srows, k_pad] layout makes it a legal full-partition
        # free-dim write: the full-width path memsets the invalid cv columns, the split path
        # masks + re-splits a full-width reload, and the unsorted path compacts only the valid
        # columns. (k % 16 == 0 -> k_pad == k -> no padding, so all gptoss configs skip it.)

        # --- De-snake the topk output set into a CONTIGUOUS [rows, k_pad] HBM tile ---
        # asc[row, p*k_cols + c] = val_snake[base + p, c]: partition p's k_cols outputs
        # land at HBM offset row*k_pad + p*k_cols, free-stride 1. Because k_pad ==
        # PARTS_PER_GROUP*k_cols, consecutive row-groups (16 partitions each) are exactly
        # k_pad apart, so the WHOLE par_dim-partition tile maps to ONE contiguous HBM run
        # [tile_row_start*k_pad : tile_row_start*k_pad + par_dim*k_cols] with partition
        # stride k_cols, free stride 1. Emitting it as a SINGLE par_dim-partition DMA per
        # tile (instead of n_rows per-row [16,k_cols] DMAs) cuts the de-snake DMA
        # instruction/descriptor count by ~8x -- the de-snake stores (lines 334/338) were
        # the top serial contributors in the HW profile (each Count=64, ~58.9k+39.5k
        # non-overlapped ns on the critical path). The flattened DMA packs all par_dim*k_cols
        # elements behind one descriptor stream. Row ORDER within a row is unspecified (the
        # phase-2 sort re-derives descending order; the per-slot value<->snake-position
        # pairing is preserved, so the snake->vocab remap is still correct). Unwritten
        # padding slots (k_pad > k) carry garbage here and are masked per-path after the
        # phase-2 reload (see the top-k call comment above).
        # The de-snake stores are STRIDED SBUF->HBM writes (.ap partition-stride k_cols).
        # HWDGE cannot generate descriptors for that pattern ([NCC_IBIR098]), so instead of
        # the default SWDGE (descriptors generated by the GpSIMD engine, which serializes
        # them with the GpSIMD top-k), use dge_mode.none: the Neuron Runtime pre-generates
        # the descriptors into HBM before execution, taking the store OFF the GpSIMD engine
        # so it overlaps the next tile's top-k.
        tile_out_off = tile_row_start * k_pad
        # De-snake store DMA mode is selected by fast_dma_safe (see desnake_dge above):
        # dge_mode.none for full-tile shards (off-GpSIMD, max overlap), SWDGE for shards
        # with any partial/single-row tile (keeps the store ordered with its producing
        # top-k so the phase-2 reload of the same private_hbm region cannot race it --
        # the low/odd-batch correctness fix).
        nisa.dma_copy(
            dst=asc_val_flat.ap(pattern=[[k_cols, par_dim], [1, k_cols]], offset=tile_out_off),
            src=val_snake[nl.ds(0, par_dim), nl.ds(0, k_cols)],
            dge_mode=desnake_dge,
        )
        nisa.dma_copy(
            dst=asc_idx_flat.ap(pattern=[[k_cols, par_dim], [1, k_cols]], offset=tile_out_off),
            src=idx_snake[nl.ds(0, par_dim), nl.ds(0, k_cols)],
            dge_mode=desnake_dge,
        )

    # =====================================================================
    # PHASE 2 -- batch-parallel descending sort. Each row's K (value, snake
    # index) pairs occupy one partition's free dim (k_pad wide). max8 /
    # nc_match_replace8 / nc_n_gather are per-partition free-dim operations,
    # so one invocation over up to 128 partitions sorts up to 128 rows at the
    # same wall-clock cost as one row. Replacing the n_tiles serial 8-row
    # sorts with ceil(per_lnc_BxS / 128) sorts removes the redundant passes
    # that dominated the measured HW profile. The per-partition math is
    # IDENTICAL to the old per-tile sort (independent rows, no cross-row
    # interaction), so correctness is unchanged.
    # =====================================================================
    n_pass = div_ceil(k, 8)
    log2_ppg = 4  # == log2(PARTS_PER_GROUP), PARTS_PER_GROUP == 16
    n_sort_tiles = div_ceil(per_lnc_BxS, PMAX)

    # --- Half-width split-sort + bitonic-merge gate ----------------------------
    # The descending sort is the measured HW bottleneck: a serial chain of
    # n_pass = ceil(k_pad/8) (=32 for k_pad=256) max8->match_replace8 passes, each
    # scanning the FULL k_pad-wide buffer. max8 / match_replace8 latency is
    # proportional to the per-partition scan width (Formula A, Vector engine), so
    # cutting both the pass count AND the width is a ~4x win on the sort. Phase 2
    # for this shape uses only 64 of 128 partitions (n_srows = per_lnc_BxS = 64),
    # so the idle 64 partitions are free parallelism: place each row's LOW half on
    # partition r and its HIGH half on partition r+n_srows, sort each half
    # descending in n_pass_h = ceil(HALF/8) (=16) passes over width HALF (=128) in
    # PARALLEL across all 2*n_srows partitions, then MERGE the two sorted halves
    # into one descending k_pad-row with a bitonic merge (log2(k_pad)=8 stages of
    # cheap select compare-exchanges that carry the paired indices). The merge
    # input is the "valley" [A_desc | reverse(B_desc)]; reverse(B) is one gather.
    # Requires k_pad a power of two, k_pad even, and 2*n_srows <= 128 so both
    # halves fit on distinct partitions.
    HALF = k_pad // 2
    use_split = (
        (k_pad >= 16)
        and (k_pad % 2 == 0)
        and ((k_pad & (k_pad - 1)) == 0)  # power of two -> valid bitonic length
        and (2 * per_lnc_BxS <= PMAX)
        # The merge writes a full k_pad-wide result into out_v/out_i (width
        # n_pass*8 == ceil(k/8)*8). For small k, k_pad = ceil(k/16)*16 can exceed
        # n_pass*8 (e.g. k<=8 -> k_pad=16 > n_pass*8=8), overrunning those buffers.
        # Only take the split path when the result fits; tiny k falls through to the
        # full-width sort below (which writes out_v only up to n_pass*8).
        and (k_pad <= n_pass * 8)
    )

    for sort_idx in nl.sequential_range(n_sort_tiles):
        sort_row_start = lnc_row_start + sort_idx * PMAX
        sort_row_end = min(sort_row_start + PMAX, lnc_row_end)
        n_srows = sort_row_end - sort_row_start
        if n_srows <= 0:
            continue

        out_v = nl.ndarray((PMAX, n_pass * 8), dtype=nl.float32, buffer=nl.sbuf)
        out_i = nl.ndarray((PMAX, n_pass * 8), dtype=nl.float32, buffer=nl.sbuf)

        # config.sorted gates ONLY the descending sort. Both paths share the reload
        # (mandatory), the snake->vocab index remap (below), and the store. When
        # sorted is False the K results are emitted in arbitrary order: reload, COMPACT
        # the valid de-snaked columns to the output prefix [0, k), and skip the sort.
        if config.sorted and use_split and (2 * n_srows <= PMAX):
            # ===== Half-width split-sort + bitonic merge ======================
            n_pass_h = div_ceil(HALF, 8)
            two = 2 * n_srows  # partitions used: A on [0:n_srows), B on [n_srows:2n)
            # Split-load: partition r holds row r's LOW half [0:HALF]; partition
            # r+n_srows holds row r's HIGH half [HALF:k_pad]. Use the DEFAULT DMA mode
            # (SWDGE), NOT dge_mode.none: the reload reads the SAME private_hbm region the
            # phase-1 de-snake just wrote, and dge_mode.none's out-of-band descriptors
            # dropped the store->reload ordering, letting this reload race the de-snake
            # store on small/partial tiles (corrupting odd-per_lnc shapes on device only).
            sv = nl.ndarray((PMAX, HALF), dtype=nl.float32, buffer=nl.sbuf)
            si = nl.ndarray((PMAX, HALF), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=sv[nl.ds(0, n_srows), :],
                src=asc_val_hbm[nl.ds(sort_row_start, n_srows), nl.ds(0, HALF)],
                dge_mode=desnake_dge,
            )
            nisa.dma_copy(
                dst=sv[nl.ds(n_srows, n_srows), :],
                src=asc_val_hbm[nl.ds(sort_row_start, n_srows), nl.ds(HALF, HALF)],
                dge_mode=desnake_dge,
            )
            nisa.dma_copy(
                dst=si[nl.ds(0, n_srows), :],
                src=asc_idx_hbm[nl.ds(sort_row_start, n_srows), nl.ds(0, HALF)],
                dge_mode=desnake_dge,
            )
            nisa.dma_copy(
                dst=si[nl.ds(n_srows, n_srows), :],
                src=asc_idx_hbm[nl.ds(sort_row_start, n_srows), nl.ds(HALF, HALF)],
                dge_mode=desnake_dge,
            )
            # Mask the de-snaked padding (unwritten snake positions [k, k_pad)) before the
            # half-sort. The phase-1 val_snake[:, k:k_pad] memset does NOT cover it: that
            # writes snake COLUMNS [k, k_pad), but the de-snake reads columns [0, k_cols) and
            # the unwritten slots live at column pos//16 (< k_cols), partition pos%16 -- a
            # partition-subrange the phase-1 full-partition memset cannot target. Left
            # unmasked, those slots carry garbage (often large positive) that wins max8 in the
            # half-sort and corrupts the merged top-k VALUES (observed for padded split k such
            # as 10/12/60 on device). The invalid columns cannot be masked in the split sv/si
            # halves directly: they land on the HIGH-half partitions only, and a
            # partition-subrange memset is rejected by the backend ([NCC_INLA001] "Invalid
            # access of N partitions starting at partition P"). Instead mask them the same way
            # as the full-width path: memset the invalid columns of the FULL-width reload
            # (a legal full-partition free-dim write on masked_v[0:n_srows, :]) and split from
            # that masked buffer via SBUF->SBUF copies. k % 16 == 0 (no padding) skips this;
            # all fast_dma_safe/GPT-OSS shapes have k_pad == k so this is a no-op there.
            if k_pad > k:
                masked_v = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=masked_v[nl.ds(0, n_srows), :],
                    src=asc_val_hbm[nl.ds(sort_row_start, n_srows), :],
                    dge_mode=desnake_dge,
                )
                _pad_runs = _invalid_desnake_col_runs(k, k_cols, k_pad)
                for _run_idx in range(len(_pad_runs)):
                    _col_start = _pad_runs[_run_idx][0]
                    _col_len = _pad_runs[_run_idx][1]
                    nisa.memset(dst=masked_v[:, nl.ds(_col_start, _col_len)], value=FLOAT32_MIN)
                # Re-split the masked full-width values into the two half-partition layout
                # (low half [0:HALF] -> partitions [0:n_srows); high half [HALF:k_pad] ->
                # [n_srows:2n)). The low half stays on the same partitions (tensor_copy); the
                # high half shifts partitions [0:n_srows) -> [n_srows:2n), which is a
                # partition move only DMA can express (tensor_copy is per-lane). si is left as
                # loaded: invalid columns carry garbage indices but their paired values are now
                # FLOAT32_MIN, so they lose the sort and are never gathered into the output.
                nisa.tensor_copy(dst=sv[nl.ds(0, n_srows), :], src=masked_v[nl.ds(0, n_srows), nl.ds(0, HALF)])
                nisa.dma_copy(dst=sv[nl.ds(n_srows, n_srows), :], src=masked_v[nl.ds(0, n_srows), nl.ds(HALF, HALF)])

            # Sort each partition's HALF values descending (16 passes, width HALF,
            # 2*n_srows partitions in parallel). hv = sorted values, hsnake =
            # gathered snake-positions (still float32) in the same descending order.
            hv = nl.ndarray((PMAX, HALF), dtype=nl.float32, buffer=nl.sbuf)
            hsnake = nl.ndarray((PMAX, HALF), dtype=nl.float32, buffer=nl.sbuf)
            for ps in nl.sequential_range(n_pass_h):
                cur = nl.ds(ps * 8, 8)
                pos = nl.ndarray((PMAX, 8), dtype=nl.uint32, buffer=nl.sbuf)
                nisa.max8(dst=hv[nl.ds(0, two), cur], src=sv[nl.ds(0, two), :])
                nisa.nc_match_replace8(
                    dst=sv[nl.ds(0, two), :],
                    dst_idx=pos[nl.ds(0, two), :],
                    data=sv[nl.ds(0, two), :],
                    vals=hv[nl.ds(0, two), cur],
                    imm=float("-inf"),
                )
                nisa.nc_n_gather(
                    dst=hsnake[nl.ds(0, two), cur], data=si[nl.ds(0, two), :], indices=pos[nl.ds(0, two), :]
                )

            # --- Co-locate the two sorted halves on one partition + build valley ---
            # mv[r, 0:HALF]   = A_desc      = hv[r, :]
            # mv[r, HALF:k]   = reverse(B_desc) = hv[r+n_srows, :] reversed
            # The reversed HIGH half makes [A_desc | B_asc] a valley (bitonic).
            # Bring partitions [n_srows:2n) down to [0:n_srows) via SBUF->SBUF DMA,
            # then reverse on the free dim with one nc_n_gather (reversed indices).
            mv = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            mi = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            # A half (already on partitions [0:n_srows)). The A-half COPY METHOD depends
            # on fast_dma_safe. hsnake is written column-by-column by the per-pass
            # nc_n_gather (GpSIMD engine) above. On a SINGLE-row sort (n_srows == 1, from
            # an odd per_lnc_BxS shard) a Vector/Scalar tensor_copy of the full [0:HALF]
            # width was observed on hardware to race the LAST gather pass (columns
            # [HALF-8:HALF]) and read stale/zero for that tail -> mv stayed correct but mi
            # LOST its last 8 indices (dropped-index bug, proven via device dump). The
            # B-half already uses dma_copy and never showed the tail zeroing, so the safe
            # path routes the A-half through the same fully-synchronized DMA. For
            # fast_dma_safe shards (GPT-OSS; n_srows >= 8) the tensor_copy is correct (no
            # tail zeroing was ever observed at n_srows >= 8) and is the cheaper original
            # path, so keep it there to avoid any perf change on the high-batch win.
            if fast_dma_safe:
                nisa.tensor_copy(dst=mv[nl.ds(0, n_srows), nl.ds(0, HALF)], src=hv[nl.ds(0, n_srows), :])
                nisa.tensor_copy(dst=mi[nl.ds(0, n_srows), nl.ds(0, HALF)], src=hsnake[nl.ds(0, n_srows), :])
            else:
                nisa.dma_copy(dst=mv[nl.ds(0, n_srows), nl.ds(0, HALF)], src=hv[nl.ds(0, n_srows), :])
                nisa.dma_copy(dst=mi[nl.ds(0, n_srows), nl.ds(0, HALF)], src=hsnake[nl.ds(0, n_srows), :])
            # B half: shift partitions [n_srows:2n) -> a contiguous [0:n_srows) scratch.
            bv = nl.ndarray((PMAX, HALF), dtype=nl.float32, buffer=nl.sbuf)
            bi = nl.ndarray((PMAX, HALF), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=bv[nl.ds(0, n_srows), :], src=hv[nl.ds(n_srows, n_srows), :])
            nisa.dma_copy(dst=bi[nl.ds(0, n_srows), :], src=hsnake[nl.ds(n_srows, n_srows), :])
            # Reversed-index constant [n_srows, HALF] = HALF-1, HALF-2, ..., 0.
            rev_idx = nl.ndarray((PMAX, HALF), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.iota(dst=rev_idx[nl.ds(0, n_srows), :], pattern=[[-1, HALF]], offset=HALF - 1)
            nisa.nc_n_gather(
                dst=mv[nl.ds(0, n_srows), nl.ds(HALF, HALF)],
                data=bv[nl.ds(0, n_srows), :],
                indices=rev_idx[nl.ds(0, n_srows), :],
            )
            nisa.nc_n_gather(
                dst=mi[nl.ds(0, n_srows), nl.ds(HALF, HALF)],
                data=bi[nl.ds(0, n_srows), :],
                indices=rev_idx[nl.ds(0, n_srows), :],
            )

            # --- Bitonic merge (descending) of the valley sequence -------------
            # For stride s in [HALF, HALF/2, ..., 1]: view free dim as
            # [k_pad/(2s)] blocks of [2, s]; compare lo=block[:,0,:] with
            # hi=block[:,1,:], keep the larger in lo (descending), carry indices
            # with the SAME predicate (select). PING-PONG between two physical
            # buffers each stage (read src strided blocks, write dst strided blocks
            # DIRECTLY) so no per-stage write-back copies are needed -- the value
            # max/min and the index selects write straight into the destination
            # buffer's strided views. log2(HALF)+1 = 8 stages for k_pad=256 (even),
            # so the final result lands back in mv/mi. Per stage: 1 pred + 2 value
            # (max,min) + 4 index (copy-on_false + predicated-copy-on_true x2) = 7
            # ops, down from 11 (the 4 write-backs are eliminated).
            mv2 = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            mi2 = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            # Order the valley writes before the merge reads them. mv[:, HALF:] / mi[:, HALF:]
            # are written by the reverse-gather (nc_n_gather, GpSIMD engine) just above; the
            # merge's first stage reads the full valley via a reshape view on a DIFFERENT
            # engine (Vector). On a small/partial tile (n_srows < GROUPS_PER_TILE, i.e. the
            # not-fast_dma_safe shards -- BxS <= 4, the decode regime) that cross-engine read
            # was observed on hardware to RACE the reverse-gather and read the high half as
            # stale zeros, so the whole B (high) half -- including the global max -- was
            # dropped by the first max/min (device-only; the simulator serializes and never
            # reproduced it). Route the valley through a synchronizing SBUF->SBUF DMA into the
            # merge's source buffers so stage 0 reads the settled valley. Full-tile shards
            # (fast_dma_safe; n_srows >= 8, GPT-OSS) never showed the race and keep the cheaper
            # in-place read.
            if not fast_dma_safe:
                mv_sync = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
                mi_sync = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=mv_sync[nl.ds(0, n_srows), :], src=mv[nl.ds(0, n_srows), :])
                nisa.dma_copy(dst=mi_sync[nl.ds(0, n_srows), :], src=mi[nl.ds(0, n_srows), :])
                mv, mi = mv_sync, mi_sync
            src_v, src_i, dst_v, dst_i = mv, mi, mv2, mi2
            s = HALF
            while s >= 1:
                ng = k_pad // (2 * s)
                sv3 = src_v.reshape((PMAX, ng, 2, s))
                si3 = src_i.reshape((PMAX, ng, 2, s))
                dv3 = dst_v.reshape((PMAX, ng, 2, s))
                di3 = dst_i.reshape((PMAX, ng, 2, s))
                lo_v = sv3[nl.ds(0, n_srows), :, 0, :]
                hi_v = sv3[nl.ds(0, n_srows), :, 1, :]
                lo_i = si3[nl.ds(0, n_srows), :, 0, :]
                hi_i = si3[nl.ds(0, n_srows), :, 1, :]
                d_lo_v = dv3[nl.ds(0, n_srows), :, 0, :]
                d_hi_v = dv3[nl.ds(0, n_srows), :, 1, :]
                d_lo_i = di3[nl.ds(0, n_srows), :, 0, :]
                d_hi_i = di3[nl.ds(0, n_srows), :, 1, :]
                # predicate m = (lo_v >= hi_v): keep lo when lo already larger.
                pred = nl.ndarray((PMAX, ng, s), dtype=nl.uint8, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=pred[nl.ds(0, n_srows), :, :], data1=lo_v, data2=hi_v, op=nl.greater_equal)
                # Values: lo' = max, hi' = min (write straight into dst strided blocks).
                nisa.tensor_tensor(dst=d_lo_v, data1=lo_v, data2=hi_v, op=nl.maximum)
                nisa.tensor_tensor(dst=d_hi_v, data1=lo_v, data2=hi_v, op=nl.minimum)
                # Indices follow the value winner: d_lo_i = pred?lo_i:hi_i,
                # d_hi_i = pred?hi_i:lo_i. tensor_copy_predicated only overwrites
                # where pred, so seed with the on_false operand first.
                nisa.tensor_copy(dst=d_lo_i, src=hi_i)
                nisa.tensor_copy_predicated(dst=d_lo_i, src=lo_i, predicate=pred[nl.ds(0, n_srows), :, :])
                nisa.tensor_copy(dst=d_hi_i, src=lo_i)
                nisa.tensor_copy_predicated(dst=d_hi_i, src=hi_i, predicate=pred[nl.ds(0, n_srows), :, :])
                src_v, dst_v = dst_v, src_v
                src_i, dst_i = dst_i, src_i
                s //= 2

            # After log2(HALF)+1 stages the result is in src_v / src_i (post-swap).
            nisa.tensor_copy(dst=out_v[nl.ds(0, n_srows), nl.ds(0, k_pad)], src=src_v[nl.ds(0, n_srows), :])
            nisa.tensor_copy(dst=out_i[nl.ds(0, n_srows), nl.ds(0, k_pad)], src=src_i[nl.ds(0, n_srows), :])
        elif config.sorted:
            # ===== Original full-width sort (small k / padding / generic) ======
            # --- Reload [n_srows, k_pad] and sort descending (max8 / match_replace8 / gather) ---
            # k_pad = k_cols*16 >= 8 already satisfies max8's >= 8 elements requirement.
            cv = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            ci = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            # Reload DMA mode follows desnake_dge: SWDGE for partial/odd shards keeps this
            # reload of the SAME private_hbm region ordered AFTER the phase-1 de-snake
            # store (dge_mode.none's out-of-band descriptors dropped that ordering and let
            # the reload race the store on small/partial tiles, reading stale/zeroed HBM
            # and corrupting values+indices on device only). Full-tile shards keep the
            # fast dge_mode.none path.
            nisa.dma_copy(
                dst=cv[nl.ds(0, n_srows), :],
                src=asc_val_hbm[nl.ds(sort_row_start, n_srows), :],
                dge_mode=desnake_dge,
            )
            nisa.dma_copy(
                dst=ci[nl.ds(0, n_srows), :],
                src=asc_idx_hbm[nl.ds(sort_row_start, n_srows), :],
                dge_mode=desnake_dge,
            )
            # Padding handling: nisa.topk only writes the K outputs into snake positions
            # [0, k); positions [k, k_pad) are NOT written and the de-snake carries their
            # (garbage) values into cv. Those garbage values can be large positives and
            # would spuriously win the descending sort, so they MUST be set to FLOAT32_MIN.
            # The de-snake places snake-output position pos at cv column j with
            #   j = (pos % 16) * k_cols + (pos // 16),   i.e. cv[row, j] holds position
            #   pos = (j // k_cols) + 16 * (j % k_cols).
            # The unwritten positions [k, k_pad) therefore map to a set of cv columns that
            # is the CONTIGUOUS tail [k, k_pad) only when k_cols == 1; for k_cols > 1 (e.g.
            # k=99 -> k_cols=7, k_pad=112) they are SCATTERED single columns (one per
            # partition >= k%16 in the partial last snake column). A single free-dim tail
            # memset (the old k_cols==1 path) misses those scattered columns and let
            # garbage (e.g. 4.59) outrank real values, corrupting k-non-multiple-of-16
            # shapes that are not a power-of-two k_pad (so they miss the split path too).
            # Enumerate the invalid cv columns at COMPILE TIME (k, k_cols, k_pad are
            # constants) and memset each contiguous run. Each memset is a legal
            # full-partition free-dim sub-range write (partition-sub-range memsets are
            # rejected by the backend, so the masking is done in cv's free dim, not by
            # partition). For k_cols == 1 this collapses to the single tail run [k, k_pad).
            # _pad_runs is a compile-time tuple of (col_start, col_len); the parser
            # frontend rejects tuple-unpacking in a for-target ("expecting simple
            # variable"), so iterate by index and read the two fields explicitly.
            _pad_runs = _invalid_desnake_col_runs(k, k_cols, k_pad)
            if k_pad > k:
                for _run_idx in range(len(_pad_runs)):
                    _col_start = _pad_runs[_run_idx][0]
                    _col_len = _pad_runs[_run_idx][1]
                    nisa.memset(dst=cv[:, nl.ds(_col_start, _col_len)], value=FLOAT32_MIN)
            for ps in nl.sequential_range(n_pass):
                cur = nl.ds(ps * 8, 8)
                pos = nl.ndarray((PMAX, 8), dtype=nl.uint32, buffer=nl.sbuf)
                # max8 returns the 8 largest of the remaining values (descending),
                # match_replace8 removes them (-inf) and reports their buffer positions.
                nisa.max8(dst=out_v[nl.ds(0, n_srows), cur], src=cv[nl.ds(0, n_srows), :])
                nisa.nc_match_replace8(
                    dst=cv[nl.ds(0, n_srows), :],
                    dst_idx=pos[nl.ds(0, n_srows), :],
                    data=cv[nl.ds(0, n_srows), :],
                    vals=out_v[nl.ds(0, n_srows), cur],
                    imm=float("-inf"),
                )
                # Gather the paired snake-position indices at those buffer positions
                # (remapped to vocab indices in the block below).
                nisa.nc_n_gather(
                    dst=out_i[nl.ds(0, n_srows), cur], data=ci[nl.ds(0, n_srows), :], indices=pos[nl.ds(0, n_srows), :]
                )
            # Order the gather writes before the shared index-remap reads them. out_i is
            # written ONLY by nc_n_gather (GpSIMD engine) above, one 8-wide slice per pass;
            # the remap's first op reads the whole out_i on a DIFFERENT engine (the
            # snake->vocab bit-ops force Vector). On a small/partial tile (n_srows <
            # GROUPS_PER_TILE, i.e. the not-fast_dma_safe shards -- BxS <= 4, the decode
            # regime) that cross-engine read RACES the last gather pass and reads the tail
            # columns as stale zeros -> snake position 0 -> vocab index 0 at the row tail
            # (values stay correct; only indices corrupt). Route out_i through a
            # synchronizing SBUF->SBUF DMA, matching the split-path valley fix above. The
            # split and unsorted paths write out_i with tensor_copy (same engine as the
            # remap read) so they need no sync; large-batch fast_dma_safe shards
            # (n_srows >= 8) never showed the race and keep the cheaper in-place read.
            if not fast_dma_safe:
                out_i_sync = nl.ndarray((PMAX, n_pass * 8), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=out_i_sync[nl.ds(0, n_srows), :], src=out_i[nl.ds(0, n_srows), :])
                out_i = out_i_sync
        else:
            # ===== Unsorted fast path (config.sorted == False) ================
            # Skip the descending sort entirely: the K results may be returned in any
            # order. Reload the de-snaked buffers (same reload + desnake_dge DMA mode as
            # the full-width sort above -- that mode keeps the reload ordered after the
            # phase-1 de-snake store, the low/odd-batch correctness fix), then COMPACT
            # the valid de-snaked columns into the output prefix [0, k).
            #
            # nisa.topk writes only snake positions [0, k); the de-snake scatters those K
            # valid columns among the k_pad cv columns (interleaved with unwritten garbage
            # columns when k % 16 != 0). _valid_desnake_col_runs enumerates the contiguous
            # (src, dst, length) runs that gather the valid columns to [0, k); for
            # k % 16 == 0 (or k_cols == 1) this is the single full-width run (0, 0, k).
            # Each tensor_copy is a full-partition free-dim sub-range write (SBUF->SBUF,
            # float32 buffers, so no narrow-dtype strided HBM store). Garbage columns are
            # never copied; the remap + store below read only out_v/out_i[:, 0:k].
            cv = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            ci = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=cv[nl.ds(0, n_srows), :],
                src=asc_val_hbm[nl.ds(sort_row_start, n_srows), :],
                dge_mode=desnake_dge,
            )
            nisa.dma_copy(
                dst=ci[nl.ds(0, n_srows), :],
                src=asc_idx_hbm[nl.ds(sort_row_start, n_srows), :],
                dge_mode=desnake_dge,
            )
            _valid_runs = _valid_desnake_col_runs(k, k_cols, k_pad)
            # _valid_runs is a compile-time tuple of (src_col, dst_col, length); the parser
            # frontend rejects tuple-unpacking in a for-target, so iterate by index and
            # read the three fields explicitly.
            for _run_idx in range(len(_valid_runs)):
                _src_col = _valid_runs[_run_idx][0]
                _dst_col = _valid_runs[_run_idx][1]
                _run_len = _valid_runs[_run_idx][2]
                nisa.tensor_copy(
                    dst=out_v[nl.ds(0, n_srows), nl.ds(_dst_col, _run_len)],
                    src=cv[nl.ds(0, n_srows), nl.ds(_src_col, _run_len)],
                )
                nisa.tensor_copy(
                    dst=out_i[nl.ds(0, n_srows), nl.ds(_dst_col, _run_len)],
                    src=ci[nl.ds(0, n_srows), nl.ds(_src_col, _run_len)],
                )

        # --- Blocked-layout snake-position -> vocab-index remap ---
        # out_i holds snake positions s as float32 (exact integers; s < full_n <= 2^24).
        # The blocked load placed vocab index p*n_cols + c at snake position s = p + 16*c,
        # so the inverse is v = (s % 16) * n_cols + (s // 16). PARTS_PER_GROUP == 16 is a
        # power of two, so the decomposition is EXACT via integer bit-ops (no float
        # floor / rounding):
        #     c = s // 16 == s >> 4   (right_shift by log2(16) = 4)
        #     p = s %  16 == s & 15   (bitwise_and with 16 - 1)
        # then v = p*n_cols + c. Bit-ops require integer dtype, so they run on uint32;
        # the recombine is done in float32 (exact since v <= vocab < 2^24) and cast back
        # to the uint32 index dtype on the store. Applied to only the k results per row.
        # log2(PARTS_PER_GROUP) as a plain literal: an int method (e.g. .bit_length())
        # inside @nki.jit is rejected by the parser frontend.
        s_u32 = nl.ndarray((PMAX, n_pass * 8), dtype=nl.uint32, buffer=nl.sbuf)
        c_u32 = nl.ndarray((PMAX, n_pass * 8), dtype=nl.uint32, buffer=nl.sbuf)
        p_u32 = nl.ndarray((PMAX, n_pass * 8), dtype=nl.uint32, buffer=nl.sbuf)
        c_f32 = nl.ndarray((PMAX, n_pass * 8), dtype=nl.float32, buffer=nl.sbuf)
        remapped_i = nl.ndarray((PMAX, n_pass * 8), dtype=nl.float32, buffer=nl.sbuf)
        # s (float32 exact integer) -> uint32 (exact for s < 2^24).
        nisa.tensor_copy(dst=s_u32[nl.ds(0, n_srows), :], src=out_i[nl.ds(0, n_srows), :])
        # c = s >> 4  (== s // 16).
        nisa.tensor_scalar(
            dst=c_u32[nl.ds(0, n_srows), :],
            data=s_u32[nl.ds(0, n_srows), :],
            op0=nl.right_shift,
            operand0=log2_ppg,
        )
        # p = s & 15  (== s % 16).
        nisa.tensor_scalar(
            dst=p_u32[nl.ds(0, n_srows), :],
            data=s_u32[nl.ds(0, n_srows), :],
            op0=nl.bitwise_and,
            operand0=PARTS_PER_GROUP - 1,
        )
        # Recombine in float32: v = p*n_cols + c.
        nisa.tensor_copy(dst=c_f32[nl.ds(0, n_srows), :], src=c_u32[nl.ds(0, n_srows), :])
        nisa.tensor_scalar(
            dst=remapped_i[nl.ds(0, n_srows), :],
            data=p_u32[nl.ds(0, n_srows), :],
            op0=nl.multiply,
            operand0=float(n_cols),
        )
        nisa.tensor_tensor(
            dst=remapped_i[nl.ds(0, n_srows), :],
            data1=remapped_i[nl.ds(0, n_srows), :],
            data2=c_f32[nl.ds(0, n_srows), :],
            op=nl.add,
        )

        nisa.dma_copy(dst=topk_values[nl.ds(sort_row_start, n_srows), :], src=out_v[nl.ds(0, n_srows), nl.ds(0, k)])
        nisa.dma_copy(
            dst=topk_indices[nl.ds(sort_row_start, n_srows), :], src=remapped_i[nl.ds(0, n_srows), nl.ds(0, k)]
        )

    return topk_values, topk_indices
