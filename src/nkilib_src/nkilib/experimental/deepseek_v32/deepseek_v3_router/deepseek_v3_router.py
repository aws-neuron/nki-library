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

"""DeepSeek-V3.2 group-limited ``noaux_tc`` router: sigmoid scores, group pruning, top-k experts."""

import nki
import nki.isa as nisa
import nki.language as nl

from ....core.utils.entry_trace import trace_kernel_entry
from . import (
    MAX8_WIDTH,
    NEG_SENTINEL,
    P_MAX,
    DeepseekV3RouterShape,
    RouterOutputs,
    get_deepseek_v3_router_shape,
)


@nki.jit
def deepseek_v3_router(
    normed: nl.NkiTensor,  # [n_tokens, hidden], or [P_MAX, n_h_tiles, n_tokens] hidden-major SBUF
    router_weight: nl.NkiTensor,  # [n_experts, hidden] fp32, or [P_MAX, n_h_tiles, n_experts]
    e_score_correction_bias: nl.NkiTensor,  # [1, n_experts] fp32, selection bias
    n_group: int,
    topk_group: int,
    top_k: int,
    routed_scaling_factor: float,
    outputs: RouterOutputs = RouterOutputs.DENSE,
    output_in_sbuf: bool = False,
):
    """
    Computes the DeepSeek-V3.2 group-limited ``noaux_tc`` routing decision for one decode step:
    sigmoid expert scores, a correction bias applied for selection only, group-limited pruning to
    ``topk_group`` groups, then the ``top_k`` experts within the survivors, L1-normalized over their
    pre-bias scores and scaled.

    Both operands accept two layouts and the choice is auto-detected rather than flagged. A caller
    that just produced ``normed`` on chip passes its hidden-major SBUF tile straight in, and the
    router weight can be pre-transposed once at load time; both skip work described in Notes.

    Dimensions:
        n_tokens: Decode tokens in the batch. Must be <= P_MAX.
        hidden: Model hidden width, the contraction dimension. Must be a multiple of P_MAX.
        n_experts: Routed experts. Must be <= 512 and divide evenly into n_group.
        n_group: Expert groups. Must be <= MAX8_WIDTH.
        topk_group: Groups kept per token. Must be <= MAX8_WIDTH.
        top_k: Experts kept per token. Must be <= MAX8_WIDTH.

    Args:
        normed: ``[n_tokens, hidden]`` in HBM, loaded and transposed here, or
            ``[P_MAX, n_h_tiles, n_tokens]`` already hidden-major in SBUF, where both the load and
            the transpose are skipped. A bf16 SBUF tile is widened to fp32 on the way in, because
            ``nc_matmul`` forbids mixing fp32 with a narrower dtype and the router weight is fp32.
        router_weight: ``[n_experts, hidden]`` fp32 row-major, loaded contiguously and
            PE-transposed here, or ``[P_MAX, n_h_tiles, n_experts]`` pre-transposed in either
            memory space, which skips every weight transpose. Build the pre-transposed form once at
            load time as ``w.reshape(n_experts, hidden // P_MAX, P_MAX).transpose(2, 1, 0)``. It
            must be the 3-D form: pre-transposing to a flat ``[hidden, n_experts]`` would make each
            partition gather 1 KB runs and reintroduce exactly the degenerate DMA this layout
            exists to avoid.
        e_score_correction_bias: ``[1, n_experts]`` fp32. Added after the sigmoid, for selection
            only; the returned weights come from the pre-bias scores.
        n_group: Expert groups that group-limited routing scores and prunes.
        topk_group: Groups kept per token.
        top_k: Experts kept per token.
        routed_scaling_factor: Final scale applied to the normalized weights.
        outputs: Which tensors to return. See Returns.
        output_in_sbuf: When True the results are returned as the SBUF tiles they were computed in,
            with no HBM allocation and no store, so a fusing caller reads them in place. Only legal
            when this kernel is inlined into a larger one, since a top-level ``@nki.jit`` entry must
            return HBM tensors for the runtime to read.

    Returns:
        With ``RouterOutputs.DENSE``, ``expert_affinities [n_tokens, n_experts]`` fp32: the scaled
        top-k weights scattered to their expert columns, zero elsewhere.

        With ``RouterOutputs.TOPK``, ``(topk_weight, topk_idx)``, both ``[n_tokens, top_k]``, where
        the weights are fp32 and the indices are int32 global expert ids in descending selection
        order.

        With ``RouterOutputs.DENSE_AND_TOPK``, all three. Computing both costs only the per-top-k
        weight gather, since the matmul, sigmoid, group pruning and normalization are shared, and it
        avoids tracing the whole router twice.

    Notes:
        The router matmul runs in fp32 regardless of the input dtype. Its logits feed a discrete
        selection that is tie-sensitive, so bf16 rounding can flip which experts or groups are
        chosen relative to the fp32 reference. Everything downstream is fp32 already.

        Both operands are loaded in their natural layout and transposed on the PE rather than
        gathered into hidden-major by the DMA. Asking the DMA for that layout directly makes each
        destination partition gather stride-hidden elements, degenerating to one descriptor per
        element: the ``[256, 7168]`` fp32 weight measured 1123 us at 6.5 GB/s, 1.2% of trn3's
        528 GB/s, and was 90% of the whole kernel. ``dma_transpose`` cannot replace it either,
        since that path needs a 2-byte dtype.

        The logits matmul tiles the PE array's columns. See ``pe_column_tile`` on the shape object
        for why, and ``core/router_topk/router_topk.py`` for the same mechanism.

    Pseudocode:
        logits = normed @ router_weight.T                    # fp32
        scores = sigmoid(logits)
        for_choice = scores + e_score_correction_bias
        group_score[t, g] = sum(top2(for_choice[t, group g]))
        keep = top_k_indices(group_score, topk_group)
        masked = for_choice - NEG_SENTINEL * (group not in keep)
        idx = top_k_indices(masked, top_k)
        w = scores[idx] / (sum(scores[idx]) + 1e-20) * routed_scaling_factor
        return scatter(w, idx) and/or (w, idx)
    """

    trace_kernel_entry("deepseek_v3_router", locals())

    shapes = get_deepseek_v3_router_shape(normed, router_weight, e_score_correction_bias, n_group, topk_group, top_k)

    normed_sbuf = _load_normed_h_major(normed, shapes)
    weight_sbuf = _load_weight_h_major(router_weight, shapes)

    scores_sbuf = _router_scores(normed_sbuf, weight_sbuf, shapes)
    for_choice_sbuf = _add_selection_bias(scores_sbuf, e_score_correction_bias, shapes)

    group_score_sbuf = _group_scores(for_choice_sbuf, shapes)
    kept_group_idx_sbuf = _select_top_groups(group_score_sbuf, shapes)
    masked_sbuf = _mask_pruned_groups(for_choice_sbuf, kept_group_idx_sbuf, shapes)

    expert_idx_sbuf = _select_top_experts(masked_sbuf, shapes)
    affinity_sbuf = _normalized_affinity(scores_sbuf, expert_idx_sbuf, routed_scaling_factor, shapes)

    return _emit(affinity_sbuf, expert_idx_sbuf, outputs, output_in_sbuf, shapes)


def _load_normed_h_major(normed: nl.NkiTensor, shapes: DeepseekV3RouterShape) -> nl.NkiTensor:
    """
    Loads ``normed`` hidden-major so ``nc_matmul`` can contract the hidden dimension on partitions.

    A caller's SBUF tile is already in this layout, so only the fp32 widening remains, and nothing
    at all when that tile is fp32 too.
    """

    n_tokens, n_h_tiles = shapes.n_tokens, shapes.n_h_tiles
    normed_sbuf = nl.ndarray((P_MAX, n_h_tiles, n_tokens), dtype=nl.float32, buffer=nl.sbuf)

    if shapes.normed_h_major:
        for h_tile in range(n_h_tiles):
            nisa.tensor_copy(dst=normed_sbuf[:, h_tile, 0:n_tokens], src=normed[:, h_tile, 0:n_tokens])
        return normed_sbuf

    # Natural layout gives each partition one fully contiguous hidden row, then the transpose runs
    # on the Tensor engine, which this kernel otherwise leaves idle.
    rows_sbuf = nl.ndarray((P_MAX, shapes.hidden), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rows_sbuf[0:n_tokens, :], src=normed)
    for h_tile in range(n_h_tiles):
        tile_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(tile_psum[:, 0:n_tokens], rows_sbuf[0:n_tokens, h_tile * P_MAX : (h_tile + 1) * P_MAX])
        nisa.tensor_copy(dst=normed_sbuf[:, h_tile, 0:n_tokens], src=tile_psum[:, 0:n_tokens])
    return normed_sbuf


def _load_weight_h_major(router_weight: nl.NkiTensor, shapes: DeepseekV3RouterShape) -> nl.NkiTensor:
    """
    Loads ``router_weight`` hidden-major, skipping every transpose when it arrives pre-transposed.

    The pre-transposed path is one contiguous DMA and zero PE transposes, which matters because the
    weight is a static model weight: the row-major path re-derives the identical tile on every call
    of every layer.
    """

    n_experts, n_h_tiles = shapes.n_experts, shapes.n_h_tiles
    weight_sbuf = nl.ndarray((P_MAX, n_h_tiles, n_experts), dtype=nl.float32, buffer=nl.sbuf)

    if shapes.weight_h_major:
        for h_tile in range(n_h_tiles):
            nisa.dma_copy(dst=weight_sbuf[:, h_tile, :], src=router_weight[:, h_tile, :])
        return weight_sbuf

    rows_sbuf = nl.ndarray((P_MAX, shapes.hidden), dtype=nl.float32, buffer=nl.sbuf)
    for expert_tile in range(0, n_experts, P_MAX):
        tile_experts = min(P_MAX, n_experts - expert_tile)
        nisa.dma_copy(dst=rows_sbuf[0:tile_experts, :], src=router_weight[expert_tile : expert_tile + tile_experts, :])
        for h_tile in range(n_h_tiles):
            tile_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(
                tile_psum[:, 0:tile_experts],
                rows_sbuf[0:tile_experts, h_tile * P_MAX : (h_tile + 1) * P_MAX],
            )
            nisa.tensor_copy(
                dst=weight_sbuf[:, h_tile, expert_tile : expert_tile + tile_experts],
                src=tile_psum[:, 0:tile_experts],
            )
    return weight_sbuf


def _router_scores(normed_sbuf: nl.NkiTensor, weight_sbuf: nl.NkiTensor, shapes: DeepseekV3RouterShape) -> nl.NkiTensor:
    """
    Computes the router logits and their sigmoid.

    When PE column tiling applies, the hidden tiles are spread across disjoint column bands that
    execute in parallel, each writing its own partition range of a full-height PSUM tile. Because
    the partial sums land in partition ranges at multiples of the band width, the reduction is an
    ordinary partition-aligned add on Vector rather than a cross-partition shuffle.

    Computes ``scores = sigmoid(normed @ router_weight.T)``.
    """

    n_tokens, n_experts, n_h_tiles = shapes.n_tokens, shapes.n_experts, shapes.n_h_tiles
    scores_sbuf = nl.ndarray((n_tokens, n_experts), dtype=nl.float32, buffer=nl.sbuf)

    if not shapes.use_pe_column_tiling:
        logits_psum = nl.ndarray((n_tokens, n_experts), dtype=nl.float32, buffer=nl.psum)
        for h_tile in range(n_h_tiles):
            nisa.nc_matmul(
                dst=logits_psum,
                stationary=normed_sbuf[:, h_tile, :],
                moving=weight_sbuf[:, h_tile, :],
            )
        nisa.activation(dst=scores_sbuf, op=nl.sigmoid, data=logits_psum)
        return scores_sbuf

    column_tile, n_bands = shapes.pe_column_tile, shapes.n_pe_column_bands
    bands_psum = nl.ndarray((P_MAX, n_experts), dtype=nl.float32, buffer=nl.psum)
    for h_tile in range(n_h_tiles):
        band_offset = (h_tile % n_bands) * column_tile
        nisa.nc_matmul(
            dst=bands_psum[nl.ds(band_offset, n_tokens), :],
            stationary=normed_sbuf[:, h_tile, :],
            moving=weight_sbuf[:, h_tile, :],
            tile_position=(0, band_offset),
            tile_size=(P_MAX, column_tile),
        )

    # Band 0 evicts PSUM to SBUF and the rest accumulate onto it, so every band that received a
    # matmul is added exactly once.
    nisa.tensor_copy(dst=scores_sbuf, src=bands_psum[nl.ds(0, n_tokens), :])
    for band in range(1, min(n_h_tiles, n_bands)):
        nisa.tensor_tensor(
            dst=scores_sbuf,
            data1=scores_sbuf,
            data2=bands_psum[nl.ds(band * column_tile, n_tokens), :],
            op=nl.add,
        )
    nisa.activation(dst=scores_sbuf, op=nl.sigmoid, data=scores_sbuf)
    return scores_sbuf


def _add_selection_bias(
    scores_sbuf: nl.NkiTensor, e_score_correction_bias: nl.NkiTensor, shapes: DeepseekV3RouterShape
) -> nl.NkiTensor:
    """
    Adds the correction bias, which steers selection only and never reaches the returned weights.

    The bias is ``[1, n_experts]`` and has to reach every token's partition, so it is broadcast with
    a ones-vector matmul on the Tensor engine; there is no on-chip partition broadcast.
    """

    n_tokens, n_experts = shapes.n_tokens, shapes.n_experts

    bias_sbuf = nl.ndarray((1, n_experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bias_sbuf, src=e_score_correction_bias.reshape((1, n_experts)))

    ones_sbuf = nl.ndarray((1, n_tokens), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_sbuf, value=1.0)
    bias_psum = nl.ndarray((n_tokens, n_experts), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=bias_psum, stationary=ones_sbuf, moving=bias_sbuf, is_stationary_onezero=True)

    for_choice_sbuf = nl.ndarray((n_tokens, n_experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=for_choice_sbuf, data1=scores_sbuf, data2=bias_psum, op=nl.add)
    return for_choice_sbuf


def _group_scores(for_choice_sbuf: nl.NkiTensor, shapes: DeepseekV3RouterShape) -> nl.NkiTensor:
    """
    Scores each expert group by the sum of its two best experts.

    Returns ``[n_tokens, n_group]``.
    """

    n_tokens, n_group = shapes.n_tokens, shapes.n_group
    experts_per_group, slice_width = shapes.experts_per_group, shapes.group_slice_width

    group_score_sbuf = nl.ndarray((n_tokens, n_group), dtype=nl.float32, buffer=nl.sbuf)
    for group in range(n_group):
        group_start = group * experts_per_group
        group_columns = for_choice_sbuf[:, group_start : group_start + experts_per_group]

        if experts_per_group < MAX8_WIDTH:
            # Pad to the width max8 needs, with a sentinel low enough not to disturb the top two.
            padded_sbuf = nl.ndarray((n_tokens, slice_width), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=padded_sbuf, value=-NEG_SENTINEL)
            nisa.tensor_copy(dst=padded_sbuf[:, 0:experts_per_group], src=group_columns)
            group_columns = padded_sbuf

        group_top_sbuf = nl.ndarray((n_tokens, MAX8_WIDTH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.max8(dst=group_top_sbuf, src=group_columns)
        nisa.tensor_reduce(
            dst=group_score_sbuf[:, group : group + 1],
            op=nl.add,
            data=group_top_sbuf[:, 0:2],
            axis=1,
            keepdims=True,
        )
    return group_score_sbuf


def _select_top_groups(group_score_sbuf: nl.NkiTensor, shapes: DeepseekV3RouterShape) -> nl.NkiTensor:
    """
    Picks the ``topk_group`` best groups per token, returning their indices as fp32 for comparison.

    The indices come back as fp32 because the mask below compares them against an fp32 iota.
    """

    n_tokens = shapes.n_tokens

    top_vals_sbuf = nl.ndarray((n_tokens, MAX8_WIDTH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=top_vals_sbuf, src=group_score_sbuf)
    top_idx_sbuf = nl.ndarray((n_tokens, MAX8_WIDTH), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.nc_find_index8(dst=top_idx_sbuf, data=group_score_sbuf, vals=top_vals_sbuf)

    group_idx_sbuf = nl.ndarray((n_tokens, MAX8_WIDTH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=group_idx_sbuf, src=top_idx_sbuf)
    return group_idx_sbuf


def _mask_pruned_groups(
    for_choice_sbuf: nl.NkiTensor, kept_group_idx_sbuf: nl.NkiTensor, shapes: DeepseekV3RouterShape
) -> nl.NkiTensor:
    """
    Drives every expert outside the kept groups below any real score.

    Builds the keep mask by comparing a per-expert group iota against each kept group index, then
    subtracts the sentinel wherever the mask is zero.
    """

    n_tokens, n_experts = shapes.n_tokens, shapes.n_experts

    expert_group_sbuf = nl.ndarray((n_tokens, n_experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(
        dst=expert_group_sbuf,
        # [stride, count] per axis: step the group id once per group, hold it across the group.
        pattern=[[1, shapes.n_group], [0, shapes.experts_per_group]],
        offset=0,
        channel_multiplier=0,
    )

    keep_sbuf = nl.ndarray((n_tokens, n_experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=keep_sbuf, value=0.0)
    for kept in range(shapes.topk_group):
        equal_sbuf = nl.ndarray((n_tokens, n_experts), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=equal_sbuf,
            data=expert_group_sbuf,
            op0=nl.equal,
            operand0=kept_group_idx_sbuf[:, kept : kept + 1],
        )
        nisa.tensor_tensor(dst=keep_sbuf, data1=keep_sbuf, data2=equal_sbuf, op=nl.add)

    # (keep - 1) * NEG_SENTINEL is zero where kept and -NEG_SENTINEL where pruned.
    penalty_sbuf = nl.ndarray((n_tokens, n_experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=penalty_sbuf,
        data=keep_sbuf,
        op0=nl.subtract,
        operand0=1.0,
        op1=nl.multiply,
        operand1=NEG_SENTINEL,
    )
    masked_sbuf = nl.ndarray((n_tokens, n_experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=masked_sbuf, data1=for_choice_sbuf, data2=penalty_sbuf, op=nl.add)
    return masked_sbuf


def _select_top_experts(masked_sbuf: nl.NkiTensor, shapes: DeepseekV3RouterShape) -> nl.NkiTensor:
    """Picks the ``top_k`` best surviving experts per token, in descending selection order."""

    n_tokens = shapes.n_tokens

    top_vals_sbuf = nl.ndarray((n_tokens, MAX8_WIDTH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=top_vals_sbuf, src=masked_sbuf)
    top_idx_sbuf = nl.ndarray((n_tokens, MAX8_WIDTH), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.nc_find_index8(dst=top_idx_sbuf, data=masked_sbuf, vals=top_vals_sbuf)
    return top_idx_sbuf


def _normalized_affinity(
    scores_sbuf: nl.NkiTensor,
    expert_idx_sbuf: nl.NkiTensor,
    routed_scaling_factor: float,
    shapes: DeepseekV3RouterShape,
) -> nl.NkiTensor:
    """
    Scatters the selected experts' normalized weights into a dense affinity tensor.

    The weights come from the **pre-bias** scores, so the correction bias never leaves selection.

    Computes ``affinity = scatter(scores[idx] / (sum(scores[idx]) + 1e-20) * routed_scaling_factor)``.
    """

    n_tokens, n_experts = shapes.n_tokens, shapes.n_experts

    expert_iota_sbuf = nl.ndarray((n_tokens, n_experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=expert_iota_sbuf, pattern=[[1, n_experts]], offset=0, channel_multiplier=0)

    selected_idx_sbuf = nl.ndarray((n_tokens, MAX8_WIDTH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=selected_idx_sbuf, src=expert_idx_sbuf)

    onehot_sbuf = nl.ndarray((n_tokens, n_experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=onehot_sbuf, value=0.0)
    for selected in range(shapes.top_k):
        equal_sbuf = nl.ndarray((n_tokens, n_experts), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=equal_sbuf,
            data=expert_iota_sbuf,
            op0=nl.equal,
            operand0=selected_idx_sbuf[:, selected : selected + 1],
        )
        nisa.tensor_tensor(dst=onehot_sbuf, data1=onehot_sbuf, data2=equal_sbuf, op=nl.add)

    gathered_sbuf = nl.ndarray((n_tokens, n_experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=gathered_sbuf, data1=scores_sbuf, data2=onehot_sbuf, op=nl.multiply)

    # Summing over every expert is the L1 norm over the top-k, since the rest are exactly zero.
    weight_sum_sbuf = nl.ndarray((n_tokens, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=weight_sum_sbuf, op=nl.add, data=gathered_sbuf, axis=1, keepdims=True)
    nisa.tensor_scalar(dst=weight_sum_sbuf, data=weight_sum_sbuf, op0=nl.add, operand0=1e-20)
    nisa.reciprocal(dst=weight_sum_sbuf, data=weight_sum_sbuf)

    affinity_sbuf = nl.ndarray((n_tokens, n_experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=affinity_sbuf,
        data=gathered_sbuf,
        op0=nl.multiply,
        operand0=weight_sum_sbuf,
        op1=nl.multiply,
        operand1=float(routed_scaling_factor),
    )
    return affinity_sbuf


def _emit(
    affinity_sbuf: nl.NkiTensor,
    expert_idx_sbuf: nl.NkiTensor,
    outputs: RouterOutputs,
    output_in_sbuf: bool,
    shapes: DeepseekV3RouterShape,
):
    """
    Returns the requested tensors, either in place in SBUF or stored to HBM.

    The compact weights are read out with one ``nc_n_gather``: token identity stays on the partition
    axis and the expert id is the free index, which is exactly that instruction's contract, so one
    GpSimd op replaces a ``top_k``-deep one-hot, multiply and reduce unroll on Vector.
    """

    n_tokens, top_k = shapes.n_tokens, shapes.top_k

    def publish(tile: nl.NkiTensor, width: int, dtype: nl.DType) -> nl.NkiTensor:
        """Hands back the SBUF tile itself, or a shared-HBM copy of it."""
        if output_in_sbuf:
            return tile
        out = nl.ndarray((n_tokens, width), dtype=dtype, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=out, src=tile)
        return out

    if outputs == RouterOutputs.DENSE:
        return publish(affinity_sbuf, shapes.n_experts, nl.float32)

    topk_weight_sbuf = nl.ndarray((n_tokens, top_k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.nc_n_gather(dst=topk_weight_sbuf, data=affinity_sbuf, indices=expert_idx_sbuf[:, 0:top_k])

    # The ids are already in descending selection order and are all below n_experts <= 512, so the
    # narrowing to int32 is exact.
    topk_idx_sbuf = nl.ndarray((n_tokens, top_k), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=topk_idx_sbuf, src=expert_idx_sbuf[:, 0:top_k])

    topk_weight = publish(topk_weight_sbuf, top_k, nl.float32)
    topk_idx = publish(topk_idx_sbuf, top_k, nl.int32)

    if outputs == RouterOutputs.DENSE_AND_TOPK:
        return publish(affinity_sbuf, shapes.n_experts, nl.float32), topk_weight, topk_idx
    return topk_weight, topk_idx
