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

"""Public API for the DeepSeek-V3.2 group-limited ``noaux_tc`` router kernel."""

from dataclasses import dataclass
from enum import Enum

import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil

P_MAX = nl.tile_size.pmax
"""Maximum partition dimension of a tile."""

NEG_SENTINEL = 30000.0
"""
Finite stand-in for ``-inf``, subtracted from masked-out experts so they lose every selection.

Sized to stay fp32-safe under the subtraction while sitting far below any sigmoid score, which a
true infinity would not.
"""

MAX8_WIDTH = 8
"""Values that ``nisa.max8`` and ``nisa.nc_find_index8`` return, which bounds ``top_k``."""

_PE_COLUMN_TILE_32 = 32
"""Narrowest PE column band. The array slices its 128 columns at 32-column granularity."""

_PE_COLUMN_TILE_64 = 64
"""Middle PE column band, used when the token count exceeds one 32-wide band."""


class RouterOutputs(Enum):
    """
    Which tensors the router returns.

    One enum rather than the pair of booleans this replaces. The two were mutually exclusive and
    the source guarded the invalid combination with an assert; an enum makes it unrepresentable
    instead, so the guard is not needed.
    """

    DENSE = 0
    """``expert_affinities [n_tokens, n_experts]`` only."""

    TOPK = 1
    """``(topk_weight, topk_idx)``, both ``[n_tokens, top_k]``."""

    DENSE_AND_TOPK = 2
    """All three, for callers that need per-slot ids and the dense column slice."""


@dataclass
class DeepseekV3RouterShape(nl.NKIObject):
    """Shapes, routing widths and the derived PE tiling plan for one router call."""

    n_tokens: int
    """Decode tokens in the batch. Must be ``<= P_MAX``, one token per partition."""

    hidden: int
    """Model hidden width. The contraction dimension of the router projection."""

    n_experts: int
    """Routed experts. Must divide evenly into ``n_group``."""

    n_group: int
    """Expert groups that group-limited routing scores and prunes."""

    topk_group: int
    """Groups kept per token. Must be ``<= MAX8_WIDTH``."""

    top_k: int
    """Experts kept per token, across the surviving groups. Must be ``<= MAX8_WIDTH``."""

    normed_h_major: bool
    """
    Whether ``normed`` already arrives hidden-major in SBUF.

    True means a fusing caller handed over the ``[P_MAX, n_h_tiles, n_tokens]`` tile that
    ``nc_matmul`` wants, so both the load and the transpose are skipped.
    """

    weight_h_major: bool
    """
    Whether ``router_weight`` already arrives hidden-major as ``[P_MAX, n_h_tiles, n_experts]``.

    The router weight is a static model weight, so the row-major path re-derives the identical
    tile on every call of every layer. At the DeepSeek shape that measured about 44.5 us of PE
    plus 22.3 us of Vector eviction per call, and in the fused router plus MoE kernel it sits on
    the critical path rather than filling an idle engine.
    """

    dtype: nl.DType
    """Dtype of the incoming ``normed``. The router itself always computes in fp32."""

    @property
    def experts_per_group(self) -> int:
        """Experts in each group, which group-limited routing scores as a unit."""
        return self.n_experts // self.n_group

    @property
    def n_h_tiles(self) -> int:
        """``P_MAX``-tall tiles spanning the contraction dimension."""
        return div_ceil(self.hidden, P_MAX)

    @property
    def group_slice_width(self) -> int:
        """
        Columns fed to ``max8`` per group.

        ``max8`` needs at least ``MAX8_WIDTH`` elements per partition, so a configuration with
        narrow groups pads its slice; at the DeepSeek size of 32 experts per group the group's
        own columns are used directly.
        """
        return max(self.experts_per_group, MAX8_WIDTH)

    @property
    def pe_column_tile(self) -> int:
        """
        Width of one PE column band.

        The stationary operand is ``[hidden, n_tokens]``, so at decode token counts the systolic
        array is loaded only ``n_tokens`` of its 128 columns wide while the instruction's cost is
        set by streaming the moving operand's expert columns and is independent of ``n_tokens``.
        Rounding up to the array's 32-column granularity is what lets several hidden tiles occupy
        disjoint bands at once.
        """
        if self.n_tokens <= _PE_COLUMN_TILE_32:
            return _PE_COLUMN_TILE_32
        if self.n_tokens <= _PE_COLUMN_TILE_64:
            return _PE_COLUMN_TILE_64
        return P_MAX

    @property
    def n_pe_column_bands(self) -> int:
        """Independent PE column bands that execute in parallel."""
        return P_MAX // self.pe_column_tile

    @property
    def use_pe_column_tiling(self) -> bool:
        """
        Whether to spread the hidden tiles across parallel PE column bands.

        Only worth it with at least two bands and enough hidden tiles to fill them, otherwise the
        band-accumulate epilogue costs more than the parallelism buys.
        """
        return self.n_pe_column_bands > 1 and self.n_h_tiles >= self.n_pe_column_bands


def get_deepseek_v3_router_shape(
    normed: nl.NkiTensor,
    router_weight: nl.NkiTensor,
    e_score_correction_bias: nl.NkiTensor,
    n_group: int,
    topk_group: int,
    top_k: int,
) -> DeepseekV3RouterShape:
    """
    Derive the shape object from the kernel's own tensors, so that callers pass only the routing
    widths that no tensor carries.

    Both operands accept two layouts and the choice is auto-detected rather than flagged, so this
    is also where those two detections are centralized: ``normed`` by memory space, and
    ``router_weight`` by rank, where three dimensions mean hidden-major.
    """

    weight_h_major = len(router_weight.shape) == 3
    if weight_h_major:
        weight_partitions, weight_h_tiles, n_experts = router_weight.shape
        kernel_assert(
            weight_partitions == P_MAX,
            f"hidden-major router_weight partition dim {weight_partitions} must equal {P_MAX}",
        )
        weight_hidden = weight_h_tiles * P_MAX
    else:
        n_experts, weight_hidden = router_weight.shape

    # Auto-detected from the buffer the caller handed over, not from a flag, matching the
    # ``input_in_sbuf`` convention used across nkilib. Defaulting the attribute keeps this usable
    # from host-side code, such as a test computing output shapes from numpy inputs, where the
    # absence of a buffer correctly means "not an SBUF tile".
    normed_h_major = getattr(normed, "buffer", None) == nl.sbuf
    if normed_h_major:
        kernel_assert(
            len(normed.shape) == 3,
            f"hidden-major normed must be 3-D [P_MAX, n_h_tiles, n_tokens], got {normed.shape}",
        )
        normed_partitions, normed_h_tiles, n_tokens = normed.shape
        kernel_assert(
            normed_partitions == P_MAX,
            f"hidden-major normed partition dim {normed_partitions} must equal {P_MAX}",
        )
        hidden = normed_h_tiles * P_MAX
    else:
        n_tokens, hidden = normed.shape

    shapes = DeepseekV3RouterShape(
        n_tokens=n_tokens,
        hidden=hidden,
        n_experts=n_experts,
        n_group=n_group,
        topk_group=topk_group,
        top_k=top_k,
        normed_h_major=normed_h_major,
        weight_h_major=weight_h_major,
        dtype=normed.dtype,
    )

    kernel_assert(shapes.n_tokens <= P_MAX, f"n_tokens={shapes.n_tokens} must be <= {P_MAX}, one token per partition")
    kernel_assert(
        weight_hidden == shapes.hidden,
        f"router_weight hidden {weight_hidden} must equal normed hidden {shapes.hidden}",
    )
    kernel_assert(shapes.hidden % P_MAX == 0, f"hidden={shapes.hidden} must be a multiple of {P_MAX}")
    kernel_assert(shapes.n_experts <= 512, f"n_experts={shapes.n_experts} must be <= 512")
    kernel_assert(
        shapes.n_group <= MAX8_WIDTH,
        f"n_group={shapes.n_group} must be <= {MAX8_WIDTH} for max8-based group selection",
    )
    kernel_assert(
        shapes.n_experts % shapes.n_group == 0,
        f"n_experts={shapes.n_experts} must divide evenly into n_group={shapes.n_group}",
    )
    kernel_assert(
        shapes.top_k <= MAX8_WIDTH,
        f"top_k={shapes.top_k} must be <= {MAX8_WIDTH}, the width nc_find_index8 returns",
    )
    kernel_assert(
        shapes.topk_group <= MAX8_WIDTH,
        f"topk_group={shapes.topk_group} must be <= {MAX8_WIDTH}, the width max8 returns",
    )
    kernel_assert(
        shapes.topk_group <= shapes.n_group,
        f"topk_group={shapes.topk_group} cannot exceed n_group={shapes.n_group}",
    )
    kernel_assert(
        e_score_correction_bias.shape[-1] == shapes.n_experts,
        f"e_score_correction_bias last dim {e_score_correction_bias.shape[-1]} must equal n_experts={shapes.n_experts}",
    )
    return shapes


# Imported last: this module defines the constants, the enum and the shape class that
# ``deepseek_v3_router`` imports from the package, so the submodule import must follow them.
from .deepseek_v3_router import deepseek_v3_router  # noqa: E402
from .deepseek_v3_router_torch import deepseek_v3_router_torch_ref  # noqa: E402

__all__ = [
    "MAX8_WIDTH",
    "NEG_SENTINEL",
    "P_MAX",
    "DeepseekV3RouterShape",
    "RouterOutputs",
    "deepseek_v3_router",
    "deepseek_v3_router_torch_ref",
    "get_deepseek_v3_router_shape",
]
