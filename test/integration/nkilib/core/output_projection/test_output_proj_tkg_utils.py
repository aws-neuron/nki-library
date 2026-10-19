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

"""Host-side unit tests for pure-Python TKG output-projection helpers.

Covers branches unreachable via runnable (non-sim-skipped) device shapes:

- ``_get_block_sizes_and_offsets`` non-zero-remainder path (last partial block).
- ``_calculate_head_folded_dimensions`` no-fold path (no valid partition size
  divides D) and the ``valid_size % D != 0`` continue arc.
"""

import nki.language as nl
import pytest
from nkilib_src.nkilib.core.output_projection.output_projection_tkg import (
    OutputProjectionTkgConfig,
    _budget_weight_blocks,
    _budget_weight_blocks_transpose,
    _get_block_sizes_and_offsets,
)
from nkilib_src.nkilib.core.output_projection.output_projection_tkg_mx_impl import (
    _calculate_head_folded_dimensions,
)
from nkilib_src.nkilib.core.utils.allocator import BufferManager
from nkilib_src.nkilib.core.utils.common_types import QuantizationType


@pytest.mark.fast
@pytest.mark.parametrize("transpose_out", [False, True])
def test_budget_weight_blocks_rejects_insufficient_sbuf(transpose_out):
    """Exhaust all candidates when even the smallest weight block cannot fit."""
    cfg = OutputProjectionTkgConfig(
        d_original_size=128,
        b_size=1,
        n_original_size=2,
        s_size=1,
        h_size=1024,
        num_prgs=1,
        prg_id=0,
        align=128,
        transpose_out=transpose_out,
        out_in_sb=False,
        is_quantized=False,
        quantization_type=QuantizationType.NONE,
        use_double_row=False,
        io_dtype=nl.bfloat16,
        n_size=2,
        h_sharded=1024,
        h_1_size=128,
        h_2_size=8,
    )
    sbm = BufferManager(sb_lower_bound=0, sb_upper_bound=128)
    if transpose_out:
        with pytest.raises(
            AssertionError, match="Not enough SBUF space for even 1 weight block in transpose path, free=128"
        ):
            _budget_weight_blocks_transpose(nl.bfloat16, cfg, sbm)
    else:
        with pytest.raises(
            AssertionError, match="Not enough SBUF space for even 1 weight block, free=128, out_sb=2048"
        ):
            _budget_weight_blocks(num_bxs_tiles=1, w_dtype=nl.bfloat16, cfg=cfg, sbm=sbm)


@pytest.mark.fast
def test_get_block_sizes_and_offsets_with_remainder():
    """h_sharded not divisible by block_size appends a smaller final block."""
    sizes, offsets = _get_block_sizes_and_offsets(3000, 2048)
    assert sizes == [2048, 952]
    assert offsets == [0, 2048]
    assert sum(sizes) == 3000


@pytest.mark.fast
def test_get_block_sizes_and_offsets_exact_multiple():
    """Exact multiple -> no remainder block (contrast arc)."""
    sizes, offsets = _get_block_sizes_and_offsets(4096, 2048)
    assert sizes == [2048, 2048]
    assert offsets == [0, 2048]


@pytest.mark.fast
@pytest.mark.parametrize("D", [3, 5, 6])
def test_calculate_head_folded_dimensions_no_valid_fold(D):
    """When no valid partition size (128/64/32) is divisible by D, FOLD_FACTOR
    stays 1 (loop runs to exhaustion; also exercises the ``valid_size % D != 0``
    continue arc for D that divides none of them)."""
    d_folded, n_folded, fold_factor = _calculate_head_folded_dimensions(N=8, D=D)
    assert fold_factor == 1
    assert d_folded == D
    assert n_folded == 8


@pytest.mark.fast
def test_calculate_head_folded_dimensions_folds_when_divisible():
    """D=32 folds by 4 (128 % 32 == 0) — contrast (break) arc."""
    d_folded, n_folded, fold_factor = _calculate_head_folded_dimensions(N=8, D=32)
    assert (d_folded, n_folded, fold_factor) == (128, 2, 4)
