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

"""Load primitive: HBM to SBUF TileStream."""

from typing import Optional

import nki.isa as nisa
import nki.language as nl
from nki.isa import oob_mode

from ....core.utils.kernel_assert import kernel_assert
from .. import tile_stream
from ..iter_order import RowMajor
from ..tile_stream import HBMStream, TileStream, get_logical_shape, tile_hbm


class Load(nl.NKIObject):
    """Load data from HBM to TileStream in SBUF.

    src is an HBMStream constructed by the user via tile_hbm().
    dst is a TileStream constructed by the user via tile_stream.tile().

    Supports Dynamic Gather Engine (DGE) for indexed loads via two mutually
    exclusive modes:

    **Scalar DGE** (`scalar_index` + `index_dim`):
        - Index is a scalar SBUF tensor (single value)
        - Same index value is broadcast to ALL partition rows
        - The indexed dimension is collapsed (removed from view)
        - Use case: "Load weights for expert X" where X is same for all rows

    **Vector DGE** (`vector_index` + `index_dim`) - REQUIRED:
        - Index is a tensor with shape (pdim_size, 1) in SBUF
        - Each partition row uses its OWN index value
        - `index_dim` specifies which dimension of the user's src view to index into
          (relative to the view passed to Load, before internal transformations)
        - The indexed dimension is NOT collapsed (unlike scalar DGE)
        - Use case: "Load scales where each row needs index = expert_idx * 16 + row"

    Note:
        `scalar_index` and `vector_index` are MUTUALLY EXCLUSIVE.
        You must specify at most one of them.
    """

    def __init__(
        self,
        dst: TileStream,
        src: HBMStream,
        scalar_index: Optional[nl.NkiTensor] = None,
        vector_index: Optional[nl.NkiTensor] = None,
        index_dim: Optional[int] = None,
        transpose: bool = False,
    ) -> None:
        """Initialize Load operation.

        Args:
            dst: Destination TileStream in SBUF
            src: Source HBMStream (from tile_hbm)
            scalar_index: (DGE) Scalar index tensor in SBUF - same value for all partition rows.
                          Mutually exclusive with vector_index.
            vector_index: (DGE) Vector index tensor in SBUF with shape (pdim_size, 1) -
                          one index per partition row. Mutually exclusive with scalar_index.
            index_dim: Dimension of user's src view to index into. REQUIRED for DGE operations.
                       This is the dimension index in the view passed to Load.
                       For scalar_index: dimension is collapsed. For vector_index: dimension remains.
            transpose: If True, use nisa.dma_transpose to swap partition and free dimensions
                       during the HBM-to-SBUF transfer. Caller must provide HBM tile shape
                       matching the HBM data layout (pre-transpose).
        """
        self._dst = dst
        self._transpose = transpose
        self._name = f"Load(dst={dst.get_name()})"

        # Validate transpose constraints
        if transpose:
            kernel_assert(
                scalar_index is None and vector_index is None,
                f"Load '{self._name}': transpose=True cannot be combined with DGE indexing",
            )

        # Validate mutual exclusivity of scalar_index and vector_index
        kernel_assert(
            not (scalar_index is not None and vector_index is not None),
            f"Load '{self._name}': scalar_index and vector_index are mutually exclusive. "
            "Use scalar_index for broadcast (same index for all rows), "
            "or vector_index for per-row indexing.",
        )

        # Validate scalar_index is in SBUF
        if scalar_index is not None:
            kernel_assert(
                scalar_index.buffer == nl.sbuf,
                f"Load '{self._name}': scalar_index must be in SBUF for DGE, got {scalar_index.buffer}",
            )
            kernel_assert(
                index_dim is not None,
                f"Load '{self._name}': index_dim is REQUIRED when using scalar_index",
            )

        # Validate vector_index is in SBUF and has correct shape
        if vector_index is not None:
            kernel_assert(
                vector_index.buffer == nl.sbuf,
                f"Load '{self._name}': vector_index must be in SBUF for DGE, got {vector_index.buffer}",
            )
            # Vector index should have pdim_size matching dst
            expected_pdim = dst.get_pdim_size()
            actual_pdim = vector_index.shape[0]
            kernel_assert(
                actual_pdim == expected_pdim,
                f"Load '{self._name}': vector_index partition dim size mismatch - "
                f"expected {expected_pdim} (dst pdim_size), got {actual_pdim}",
            )
            kernel_assert(
                index_dim is not None,
                f"Load '{self._name}': index_dim is REQUIRED when using vector_index. "
                "Specify which dimension of the source HBM tensor to index into.",
            )

        self._vector_index = vector_index
        self._index_dim = index_dim
        self._indexed_stride = None

        # For Vector DGE: store the stride of the indexed dimension for later lookup
        if vector_index is not None:
            src_tensor = src.get_container()
            self._indexed_stride = src_tensor.strides[index_dim]

        # For Scalar DGE: dynamic-select the indexed dim out of the src HBMStream's tensor.
        # This modifies the underlying view so get_tile() will emit ap() with scalar_offset.
        if scalar_index is not None:
            src_tensor = src.get_container()
            src_tensor = src_tensor.select(index_dim, scalar_index)
            # Rebuild HBMStream with the modified view
            # After select, ndim decreases by 1. Adjust tile_dims accordingly.
            old_tile_dims = src.get_tile_dims()
            new_tile_dims = []
            for td in old_tile_dims:
                if td < index_dim:
                    new_tile_dims.append(td)
                elif td > index_dim:
                    new_tile_dims.append(td - 1)
                # td == index_dim is removed
            new_tile_shape = []
            old_tile_shape = src.get_tile_shape()
            for i in range(len(old_tile_dims)):
                if old_tile_dims[i] != index_dim:
                    new_tile_shape.append(old_tile_shape[i])
            src = HBMStream(
                tensor=src_tensor,
                tile_shape=tuple(new_tile_shape),
                iter_order=src.get_iter_order(),
                tile_dims=tuple(new_tile_dims),
            )

        self._src = src

        # Validate tile count matches
        kernel_assert(
            dst.get_num_tiles() == self._src.get_num_tiles(),
            f"Load '{self._name}': tile count mismatch - dst={dst.get_num_tiles()}, src={self._src.get_num_tiles()}",
        )

    def _find_pattern_dim_by_stride(self, tile_view: nl.NkiTensor, target_stride: int) -> int:
        """Find which pattern dimension has the target stride.

        Args:
            tile_view: The tile view after HBMStream transformations
            target_stride: The stride of the dimension we want to find

        Returns:
            The pattern dimension index that has the target stride
        """
        for i in range(len(tile_view.strides)):
            if tile_view.strides[i] == target_stride:
                return i
        kernel_assert(
            False,
            f"Load '{self._name}': Could not find indexed dimension with stride {target_stride} in tile pattern. "
            "Ensure the indexed dimension is part of the tile (not a grid dimension).",
        )

    def execute(self) -> None:
        self._dst.reset_cur_tile()
        self._src.reset_cur_tile()

        for _ in range(self._src.get_num_tiles()):
            dst_tile = self._dst.get_tile()
            src_tile = self._src.get_tile()

            if self._transpose:
                nisa.dma_transpose(dst=dst_tile, src=src_tile)
            elif self._vector_index is not None:
                # Vector DGE: each partition row uses its own index
                physical_index_dim = self._find_pattern_dim_by_stride(src_tile, self._indexed_stride)

                ap_pattern, ap_offset = src_tile._get_pattern_and_offset()

                # Override the indexed dimension's size to match destination pdim_size
                pdim_size = self._dst.get_pdim_size()
                modified_pattern = []
                for i in range(len(ap_pattern)):
                    stride, size = ap_pattern[i]
                    if i == physical_index_dim:
                        modified_pattern.append((stride, pdim_size))
                    else:
                        modified_pattern.append((stride, size))

                nisa.dma_copy(
                    dst=dst_tile,
                    src=src_tile.ap(
                        pattern=modified_pattern,
                        offset=ap_offset,
                        vector_offset=self._vector_index,
                        indirect_dim=physical_index_dim,
                    ),
                    dge_mode=1,
                    oob_mode=oob_mode.skip,
                )
            else:
                # No DGE or Scalar DGE (scalar_offset already in nl.NkiTensor)
                nisa.dma_copy(dst=dst_tile, src=src_tile)

        self._dst.reset_cur_tile()
        self._src.reset_cur_tile()


def load(
    dst: nl.NkiTensor,
    src: nl.NkiTensor,
) -> None:
    """Compact load: HBM to SBUF. Whole tensor, no tiling.

    Wraps src in a single-tile HBMStream and dst in a single-tile TileStream.

    Args:
        dst: Destination tensor in SBUF (from alloc_logical or nl.NkiTensor)
        src: Source tensor in HBM (nl.NkiTensor or nl.NkiTensor)
    """
    logical_shape = get_logical_shape(dst)
    dst_ts = tile_stream.tile(dst, logical_shape, iter_order=RowMajor())

    src_view = src
    src_hbm = tile_hbm(src_view, tuple(src_view.shape), iter_order=RowMajor())

    Load(dst=dst_ts, src=src_hbm).execute()
