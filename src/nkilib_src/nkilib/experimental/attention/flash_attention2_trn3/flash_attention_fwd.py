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

"""FlashAttention-2 forward compute primitive for Trainium 3."""

import nki.isa as nisa
import nki.language as nl

from ....core.utils.allocator import sizeinbytes
from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.modular_allocator import ModularAllocator
from ... import neurotile as nt
from .config import SQ_TILE, SV_TILE, FlashAttentionGeometry, StepInfo
from .mixed_radix import flatten

# The reference before anything has been seen: a max-reduce ignores it, and the first rescale is
# exp(-huge) = 0.
_M_INIT = -3.0e38


class SoftmaxState(nl.NKIObject):
    """The carried online-softmax state of one Sq block: numerator, denominator and reference max.

    For every Sq row, between Sk blocks::
        running_out = sum(exp(s - running_max) * v)     running_sum = sum(exp(s - running_max))
    """

    def __init__(self, sq_tiles: int, D: int):
        """Allocate the state for a block of ``sq_tiles`` Sq tiles at head dimension ``D``."""
        self.running_out = nl.ndarray((SQ_TILE, sq_tiles, D), dtype=nl.float32, buffer=nl.sbuf)
        self.running_sum = nl.ndarray((SQ_TILE, sq_tiles), dtype=nl.float32, buffer=nl.sbuf)
        self.running_max = nl.ndarray((SQ_TILE, sq_tiles), dtype=nl.float32, buffer=nl.sbuf)

    def reset(self) -> None:
        """Initialize to the empty softmax: no numerator, no denominator, no reference seen."""
        nisa.memset(self.running_out, value=0.0)
        nisa.memset(self.running_sum, value=0.0)
        nisa.memset(self.running_max, value=_M_INIT)


class FlashAttentionFwd(nl.NKIObject):
    """FlashAttention-2 forward compute primitive.

    Streams the whole Sk range against a resident Q block, accumulating the online-softmax numerator
    and denominator in SBUF, and writes the normalized result into ``o_block``. The six stages are
    software-pipelined; :meth:`run` is the entire schedule.

    Scores stay fp32 from the matmul into the exponential; only the probabilities narrow to bf16, and
    only after the exponential has been taken.

    The engine assignment is the point of this variant. Two of its instructions exist only on
    NeuronCore-v4: ``activate2`` and ``exponential``.

    =================  ==============  ================================================
    stage              engine          instruction
    =================  ==============  ================================================
    ``_qk``            Tensor          ``nc_matmul``
                       Scalar          ``activate2``, scale and chain a reduce max
    ``_max_update``    GpSimd/Scalar   ``tensor_scalar`` fold, ``activation`` for the rescale factor
    ``_exp``           Vector          ``exponential``, subtracts a *positive* max
    ``_transpose``     Tensor/Vector   ``nc_transpose`` then ``tensor_copy`` to evict
    ``_pv``            Tensor          ``nc_matmul``
    ``_softmax_merge`` Vector/GpSimd   fold the point into the carried online-softmax state
    =================  ==============  ================================================

    The six stages are software-pipelined over a flat walk of ``(num_sk_blks, sq_tiles_per_blk,
    sk_tiles_per_blk)``.
    :meth:`FlashAttentionFwd.run` contains a fill (prologue), body, and drain (epilogue).
    Every intermediate is allocated once as a list of pipeline slots and fetched by
    grid index, with depths derived from the offsets.

    Both reduce chains -- the max in the Scalar engine's accumulator, the row sum in the Vector engine's
    -- live in registers that have no address, so the compiler cannot order around them. That is why a
    stage's lead is counted in whole Sk blocks: a fractional lead would land per-point work inside an
    open chain.
    """

    def __init__(self, scale: float, geometry: FlashAttentionGeometry):
        """Allocate every intermediate from the geometry alone.

        Args:
            scale: attention-score scale.
            geometry: tiling, pipeline and buffer geometry, resolved against the shape.
        """
        self.scale = scale
        self.geometry = geometry
        self.grid = geometry.grid
        self.buffers = {}
        self._alloc()

    # -- allocation ---------------------------------------------------------------------------

    def _alloc_sbuf(self, name: str, depth: int, shape: tuple, dtype) -> None:
        """Register ``depth`` SBUF slots under ``name``, at addresses this kernel chooses."""
        self.buffers[name] = self._sbuf.alloc_sbuf_tensor(shape, dtype, block_dim=[depth], num_free_tiles=[depth])

    def _alloc_psum(self, name: str, tile_size: tuple, dtype, placement: list) -> None:
        """Register one PSUM slot per ``(bank, byte_offset)`` entry of ``placement``."""
        used = sizeinbytes(dtype)
        for dim in range(1, len(tile_size)):
            used = used * tile_size[dim]

        bank_bytes = nl.tile_size.psum_bank_fmax_bytes
        num_banks = nl.tile_size.psum_num_banks
        slots = []
        for slot in range(len(placement)):
            bank = placement[slot][0]
            offset = placement[slot][1]
            kernel_assert(bank < num_banks, f"{name} slot {slot}: bank {bank} of {num_banks} out of range")
            kernel_assert(offset + used <= bank_bytes, f"{name} slot {slot}: {offset}+{used}B overflows a bank")
            slots.append(nl.ndarray(tile_size, dtype=dtype, buffer=nl.psum, address=(0, bank * bank_bytes + offset)))
        self.buffers[name] = slots

    def _alloc(self):
        """Allocate every intermediate this primitive uses."""
        cfg = self.geometry
        # PSUM map: (bank index, byte offset in that bank)
        o_place = [(0, 0)]  # MM2 output
        probs_place = [(1, 0), (2, 0)]  # nc_transpose output
        qk_place = [(3, 0), (4, 0), (5, 0), (6, 0), (7, 0)]  # MM1 output
        # SBUF map
        self._sbuf = ModularAllocator(initial_address=0)
        sv_tiles = cfg.sv_tiles_per_sk_tile

        self._alloc_sbuf("scores", cfg.scores_depth, (SQ_TILE, cfg.sk_tile), nl.float32)
        # block_max is this block's own max; current_max = max(block_max, running_max)
        self._alloc_sbuf("block_max", cfg.stats_depth, (SQ_TILE, 1), nl.float32)
        self._alloc_sbuf("current_max", cfg.stats_depth, (SQ_TILE, 1), nl.float32)
        self._alloc_sbuf("block_sum", cfg.stats_depth, (SQ_TILE, 1), nl.float32)
        self._alloc_sbuf("rescale", cfg.stats_depth, (SQ_TILE, 1), nl.float32)
        self._alloc_psum("block_out", (SQ_TILE, cfg.D), nl.float32, o_place)

        self._alloc_psum("scores_psum", (SQ_TILE, cfg.sk_tile), nl.float32, qk_place)
        # 2-D, not (Sq, V tile, SV_TILE): the exponential writes sk_valid elements, and once that is
        # not a whole number of V tiles only a flat free axis can express it as one rectangle.
        self._alloc_sbuf("probs", cfg.probs_depth, (SQ_TILE, cfg.sk_tile), nl.bfloat16)
        self._alloc_psum("probs_psum", (SV_TILE, sv_tiles, SQ_TILE), nl.bfloat16, probs_place)
        self._alloc_sbuf("probs_t", cfg.probs_t_depth, (SV_TILE, sv_tiles, SQ_TILE), nl.bfloat16)

        # Per Sq tile, in the normalize pass.
        self._alloc_sbuf("inv_sum", 2, (SQ_TILE, 1), nl.float32)

    def get_buffer(self, index: tuple, name: str):
        """Return the pipeline slot of intermediate ``name`` that grid index ``index`` maps to.

        Args:
            index: one entry per grid dimension, ``None`` where the operation does not depend on it.
            name: intermediate name, as registered by :meth:`_alloc`.
        """
        slots = self.buffers[name]
        return slots[flatten(index, self.grid) % len(slots)]

    # ================================================================================
    # =========================== Compute Stages =====================================
    # ================================================================================

    def _step_info(self, step: int) -> StepInfo:
        """This step's descriptor, for the Sq block :meth:`run` was handed."""
        return self.geometry.step_info(step, self._sq_tiles_valid)

    def _qk(self, step: int, q_block, k_stream, evict_engine: int = nisa.engine.scalar) -> None:
        """MM1 for one Sk tile, then evict it scaled onto ``evict_engine``'s max chain.
        Q @ K^t   ->  evict with scaling and reduce_max.

        Contracts over D, so Q arrives transposed and the scores land as ``[Sq, Sk]`` -- which is what
        makes the reference a free-axis reduction. ``activate2`` applies ``op(data * imm0 + imm1)``
        before reducing, so ``nl.copy`` with ``imm0=scale`` makes one instruction both the eviction and
        the scale. The max accumulates across the point's Sk tiles and is read out on the last.

        Loads K on the first step of an Sk block and cache it for the rest of the block.

        ``evict_engine`` picks the instruction:
        ``activate2`` is Scalar-only and ``tensor_scalar_reduce`` Vector-only.
        A point's whole chain must stay on one engine so the two accumulators never mix.
        """
        info = self._step_info(step)
        index = info.index
        sk_tile_idx = index[2]
        if index[1] == 0 and sk_tile_idx == 0:
            self._k_blk = k_stream.load(index[0], dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        if not info.exists:
            return

        scores_psum = self.get_buffer(index, "scores_psum")[:, : info.sk_valid]
        q_tile_col = q_block[:, index[1]]
        for d_tile_idx in range(self._k_blk.shape[0]):
            nisa.nc_matmul(
                scores_psum,
                stationary=q_tile_col[d_tile_idx, 0].data,
                moving=self._k_blk[d_tile_idx, sk_tile_idx].data,
                accumulate=(d_tile_idx > 0),
            )

        block_max = self.get_buffer((index[0], index[1], None), "block_max") if info.is_last_tile else None
        reduce_cmd = nisa.reduce_cmd.reset_reduce if info.is_first_tile else nisa.reduce_cmd.reduce
        scores = self.get_buffer(index, "scores")[:, : info.sk_valid]

        if evict_engine == nisa.engine.vector:
            nisa.tensor_scalar_reduce(
                scores,
                data=scores_psum,
                op0=nl.multiply,
                operand0=self.scale,
                reduce_op=nl.maximum,
                reduce_cmd=reduce_cmd,
                reduce_res=block_max,
            )
            return

        nisa.activate2(
            dst=scores,
            op=nl.copy,
            data=scores_psum,
            imm0=self.scale,
            imm1=0.0,
            op0=nl.multiply,
            op1=nl.add,
            reduce_op=nl.maximum,
            reduce_cmd=reduce_cmd,
            reduce_res=block_max,
        )

    def _max_update(self, step: int, state: SoftmaxState) -> None:
        """Fold the carried max into this point's, then form the rescale factor.
        ``current_max = max(running_max, block_max)``
        ``rescale = exp(running_max - current_max)``
        """
        info = self._step_info(step)
        if not info.exists or not info.is_first_tile:  # run once per (sk, sq) block
            return
        index = (info.index[0], info.index[1], None)
        running_max = state.running_max[:, index[1]]
        current_max = self.get_buffer(index, "current_max")
        nisa.tensor_scalar(
            dst=current_max,
            data=running_max,
            op0=nl.maximum,
            operand0=self.get_buffer(index, "block_max"),
            engine=nisa.engine.gpsimd,
        )
        nisa.activation(
            dst=self.get_buffer(index, "rescale"), op=nl.exp, data=current_max, scale=-1.0, bias=running_max
        )

    def _exp(self, step: int) -> None:
        """Exponentiate one Sk tile on the Vector engine, chaining the row sum.

        ``exponential`` subtracts ``max_value`` itself, so the reference is passed positive. The sum
        chains in the Vector engine's registers -- a different set from MM1's max chain on Scalar, so
        the two cannot disturb each other.
        """
        info = self._step_info(step)
        if not info.exists:
            return
        index = info.index
        blk = (index[0], index[1], None)

        nisa.exponential(
            dst=self.get_buffer(index, "probs")[:, : info.sk_valid],
            src=self.get_buffer(index, "scores")[:, : info.sk_valid],
            max_value=self.get_buffer(blk, "current_max"),
            reduce_cmd=nisa.reduce_cmd.reset_reduce if info.is_first_tile else nisa.reduce_cmd.reduce,
            reduce_res=self.get_buffer(blk, "block_sum") if info.is_last_tile else None,
        )

    def _transpose(self, step: int, evict_engine: int = nisa.engine.vector) -> None:
        """Transpose one Sk tile's P on the Tensor engine, evicting it back to SBUF."""
        info = self._step_info(step)
        if not info.exists:
            return
        index = info.index
        probs = self.get_buffer(index, "probs")
        probs_psum = self.get_buffer(index, "probs_psum")
        for sv_tile_idx in range(info.sv_tiles_valid):
            # Sk is the free axis here and the partition axis after the transpose, so a short final V
            # tile narrows the source's columns and the destination's partitions by the same amount.
            rows = SV_TILE
            if sv_tile_idx == info.sv_tiles_valid - 1:
                rows = info.sv_last_extent
            start = sv_tile_idx * SV_TILE
            nisa.nc_transpose(
                probs_psum[0:rows, sv_tile_idx, :],
                data=probs[:, start : start + rows],
                engine=nisa.engine.tensor,
            )
        # psum vectorization - evict once
        nisa.tensor_copy(
            dst=self.get_buffer(index, "probs_t")[:, : info.sv_tiles_valid, :],
            src=probs_psum[:, : info.sv_tiles_valid, :],
            engine=evict_engine,
        )

    def _pv(self, step: int, v_stream) -> None:
        """Accumulate ``P @ V`` for one Sk tile into the point's O accumulator.

        Loads V on the first step of an Sk block and holds the view for the rest of it. V arrives with
        one flat Sk axis, so the loaded block is re-tiled into (Sk tile, V tile) to match the walk.
        """
        info = self._step_info(step)
        index = info.index
        sk_tile_idx = index[2]
        if index[1] == 0 and sk_tile_idx == 0:  # once per sk_block
            loaded = v_stream.load(index[0], dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
            self._v_blk = nt.blocks(loaded, block_size=(self.geometry.sv_tiles_per_sk_tile, 1))
        if not info.exists:
            return

        block_out = self.get_buffer((index[0], index[1], None), "block_out")
        probs_t = self.get_buffer(index, "probs_t")
        v_sk_tile = self._v_blk[sk_tile_idx, 0]
        for sv_tile_idx in range(info.sv_tiles_valid):
            # for unaligned shapes we might compute smaller than tile size.
            rows = SV_TILE
            if sv_tile_idx == info.sv_tiles_valid - 1:
                rows = info.sv_last_extent
            nisa.nc_matmul(
                block_out,
                stationary=probs_t[0:rows, sv_tile_idx, :],
                moving=v_sk_tile[sv_tile_idx, 0].data[0:rows, :],
                accumulate=not (info.is_first_tile and sv_tile_idx == 0),
            )

    def _softmax_merge(self, step: int, state: SoftmaxState) -> None:
        """Fold this point's partial softmax into the carried state, on the point's last Sk tile.

        The second half of the online-softmax correction (:meth:`_max_update` is the first): the
        carried numerator and denominator are rescaled by ``exp(running_max - current_max)`` and the
        point's own ``block_out`` / ``block_sum`` added in.
        """
        info = self._step_info(step)
        if not info.exists or not info.is_last_tile:
            return
        index = (info.index[0], info.index[1], None)
        sq_tile_idx = index[1]
        rescale = self.get_buffer(index, "rescale")
        running_out = state.running_out[:, sq_tile_idx, :]
        running_sum = state.running_sum[:, sq_tile_idx]
        running_max = state.running_max[:, sq_tile_idx]

        # apply online softmax correction on output tile
        nisa.scalar_tensor_tensor(
            dst=running_out,
            data=running_out,
            op0=nl.multiply,
            operand0=rescale,
            op1=nl.add,
            operand1=self.get_buffer(index, "block_out"),
        )
        # Both on GpSimd, which is the idle engine: these are [SQ_TILE, 1] updates
        nisa.tensor_scalar(
            dst=running_sum,
            data=running_sum,
            op0=nl.multiply,
            operand0=rescale,
            op1=nl.add,
            operand1=self.get_buffer(index, "block_sum"),
            engine=nisa.engine.gpsimd,
        )
        nisa.tensor_copy(dst=running_max, src=self.get_buffer(index, "current_max"), engine=nisa.engine.gpsimd)

    def _normalize(self, index: tuple, o_block, state: SoftmaxState) -> None:
        """Write ``running_out / running_sum`` into ``o_block``, casting to its dtype."""
        sq_tile_idx = index[1]
        inv_sum = self.get_buffer(index, "inv_sum")
        nisa.reciprocal(dst=inv_sum, data=state.running_sum[:, sq_tile_idx])
        nisa.tensor_scalar(
            dst=o_block[sq_tile_idx, 0].data,
            data=state.running_out[:, sq_tile_idx, :],
            op0=nl.multiply,
            operand0=inv_sum,
        )

    # -- schedule -----------------------------------------------------------------------------

    def run(self, o_block, q_block, k_stream, v_stream) -> None:
        """Walk the flattened grid as a software pipeline.

        The schedule is the whole of this method: a fill, a steady-state body where every stage
        advances one step, and a drain. Because each stage is offset by a fixed number of steps, no
        call in the body needs a guard -- the fill and drain cover the ends.

        Args:
            o_block: SBUF ``(sq_tiles_per_blk, 1)`` grid of ``[SQ_TILE, D]`` -- written.
            q_block: SBUF ``(d_tiles, sq_tiles_per_blk)`` grid of ``[D_TILE, SQ_TILE]`` transposed Q.
            k_stream: stream of ``(d_tiles, sk_tiles_per_blk)`` grids of ``[D_TILE, sk_tile]``.
            v_stream: stream of ``(sk_blk / SV_TILE, 1)`` grids of ``[SV_TILE, D]``.

        Returns:
            None. The normalized result is written through ``o_block``.
        """
        cfg = self.geometry
        # unaligned shapes might result in fewer valid tiles
        self._sq_tiles_valid = q_block.shape[1]

        steps = cfg.num_steps
        qk_off = cfg.qk_offset
        kernel_assert(steps >= qk_off, f"walk of {steps} steps cannot fill a {qk_off}-step pipeline")

        # Per-call, compiler-placed
        state = SoftmaxState(sq_tiles=cfg.sq_tiles_per_blk, D=cfg.D)
        state.reset()

        # -- prologue: fill, deepest stage first ----------------------------------------
        # MM1 runs qk_off steps before MM2 starts, so the fill issues all of them here, in two loops
        # that differ only in the eviction engine. The first point evicts on Vector: nothing else is
        # running yet, so the two engines share the fill instead of queueing on Scalar.
        for step in range(cfg.sk_tiles_per_blk):
            self._qk(step, q_block, k_stream, evict_engine=nisa.engine.vector)
        # The remaining fill steps take the Scalar default, as the steady state does.
        for step in range(cfg.sk_tiles_per_blk, qk_off):
            self._qk(step, q_block, k_stream)
        for step in range(cfg.max_update_offset):
            self._max_update(step, state)
        for step in range(cfg.exp_offset):
            self._exp(step)
        for step in range(cfg.tp_offset):
            self._transpose(step)

        # -- body: every stage advances one step ----------------------------------------
        # Program order is forced for the steady state only.
        with nl.no_reorder():
            for i in range(steps - qk_off):
                self._qk(i + qk_off, q_block, k_stream)
                self._max_update(i + cfg.max_update_offset, state)
                self._exp(i + cfg.exp_offset)
                self._transpose(i + cfg.tp_offset)
                self._pv(i + cfg.pv_offset, v_stream)
                self._softmax_merge(i + cfg.pv_offset, state)

        # -- epilogue: the same body with MM1 dropped, stages falling out as steps run out -
        for i in range(steps - qk_off, steps):
            if i + cfg.max_update_offset < steps:
                self._max_update(i + cfg.max_update_offset, state)
            if i + cfg.exp_offset < steps:
                self._exp(i + cfg.exp_offset)
            if i + cfg.tp_offset < steps:
                # The drain evicts on Scalar, which is generally free (there is no QK^t)
                self._transpose(i + cfg.tp_offset, evict_engine=nisa.engine.scalar)
            self._pv(i + cfg.pv_offset, v_stream)
            self._softmax_merge(i + cfg.pv_offset, state)

        for sq_tile_idx in range(self._sq_tiles_valid):
            self._normalize((None, sq_tile_idx, None), o_block, state)
