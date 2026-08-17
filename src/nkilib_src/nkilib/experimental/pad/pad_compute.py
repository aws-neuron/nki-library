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

"""Tile-agnostic SBUF padding logic.

``pad_compute`` copies source data into a padded SBUF buffer and fills the
padding regions using a PadMode callback. It is independent of the tiling
strategy and can be used standalone for SBUF-to-SBUF padding.

Padding is filled inner-to-outer (W → H → D) so that each dimension's
padding automatically includes the already-filled padding of inner dimensions.

All tensor arguments are NkiTensors.
"""

import nki.isa as nisa
import nki.language as nl

from .pad_modes import PadMode
from .pad_params import PadParams


def _fill_dim(view: nl.NkiTensor, pad_mode: PadMode, params: PadParams, dim: int) -> None:
    """Fill the padding region along one spatial dimension in the SBUF tile.

    Copies data from the interior (non-padded) region of *view* into the
    before- and after-padding positions along *dim*, using the PadMode
    callback to determine the source index for each padded element.
    """
    axis = dim + 1
    pad_before = params.before(dim)
    pad_after = params.after(dim)
    interior_size = view.shape[axis] - pad_before - pad_after

    interior = view.slice(axis, pad_before, pad_before + interior_size)

    for i in range(pad_before):
        dst = view.select(axis, i)
        pad_mode.fill(dst, interior, dim, -(pad_before - i))

    for i in range(pad_after):
        dst = view.select(axis, pad_before + interior_size + i)
        pad_mode.fill(dst, interior, dim, i)


def pad_compute(src_sb: nl.NkiTensor, padded_sb: nl.NkiTensor, params: PadParams, pad_mode: PadMode) -> None:
    """Copy source data into padded buffer and fill all padding regions.

    Algorithm:
    1. Place the unpadded source data into the interior of the padded buffer.
    2. Fill padding inner-to-outer (W → H → D). Each pass sources from the
       interior region, which already includes padding from inner dimensions.
       This means H padding rows automatically carry correct W padding, and
       D padding slices carry correct H+W padding.

    This is the tile-agnostic core of the pad kernel. It can be called
    directly for SBUF-to-SBUF padding without the tiling/DMA machinery.
    """
    d_before, h_before, w_before = params.before(0), params.before(1), params.before(2)
    _, d_count, h_count, w_count = src_sb.shape

    # Build progressively narrower views for each dimension.
    d_interior_view = padded_sb.slice(1, d_before, d_before + d_count)
    dh_interior_view = d_interior_view.slice(2, h_before, h_before + h_count)
    dhw_interior_view = dh_interior_view.slice(3, w_before, w_before + w_count)

    # Copy source into the innermost interior region
    nisa.tensor_copy(dst=dhw_interior_view, src=src_sb)

    # Fill padding inner-to-outer: W → H → D
    _fill_dim(dh_interior_view, pad_mode, params, 2)  # W pad filling
    _fill_dim(d_interior_view, pad_mode, params, 1)  # H pad filling
    _fill_dim(padded_sb, pad_mode, params, 0)  # D pad filling
