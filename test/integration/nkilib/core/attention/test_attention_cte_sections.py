# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import nki.language as nl
import pytest
from nkilib_src.nkilib.core.attention.attention_cte import (
    _EXP_TILE_SZ,
    _K_TILE_SZ,
    AttnConfig,
    AttnTileParams,
    SectionParams,
    _compute_tile_parameters,
    _exp_tile_selection,
)


def _attention_config(
    *,
    is_prefix_caching: bool = False,
    seqlen_k_prior: int = 0,
    use_cp: bool = False,
    global_cp_deg: int = 1,
    causal_mask: bool = False,
) -> AttnConfig:
    return AttnConfig(
        seqlen_q=128,
        seqlen_k_active=4096,
        seqlen_k_prior=seqlen_k_prior,
        d=384,
        d_v=384,
        tp_q=False,
        tp_k=True,
        tp_out=False,
        is_prefix_caching=is_prefix_caching,
        causal_mask=causal_mask,
        use_swa=False,
        sliding_window=0,
        use_cp=use_cp,
        global_cp_deg=global_cp_deg,
        cp_strided_q_slicing=False,
        cp_striped_input=False,
        scale=1.0,
        cache_softmax=False,
        skip_output_normalization=False,
        dtype=nl.bfloat16,
        softmax_dtype=nl.float32,
        mm_out_dtype=nl.bfloat16,
        is_sequence_packed=False,
        has_kv_used_len=False,
        use_position_bias=False,
        bias_layout="dense",
        bias_prior_band_width=0,
        bias_active_offset=0,
        bias_active_band_width=0,
        bias_active_origin=0,
    )


@pytest.mark.fast
def test_flash_section_length_is_aligned_to_k_tiles():
    tile_params = _compute_tile_parameters(_attention_config(), is_seqlen_sharded=False)

    assert tile_params.section_len == 2560
    assert tile_params.section_len % _K_TILE_SZ == 0
    assert tile_params.num_k_tiles_per_section == 5


@pytest.mark.parametrize(
    "is_prefix_caching,seqlen_k_prior,use_cp,global_cp_deg,causal_mask",
    [
        (True, 777, False, 1, False),
        (False, 0, True, 2, True),
    ],
)
@pytest.mark.fast
def test_flash_section_alignment_is_preserved_for_prefix_and_cp(
    is_prefix_caching: bool,
    seqlen_k_prior: int,
    use_cp: bool,
    global_cp_deg: int,
    causal_mask: bool,
):
    config = _attention_config(
        is_prefix_caching=is_prefix_caching,
        seqlen_k_prior=seqlen_k_prior,
        use_cp=use_cp,
        global_cp_deg=global_cp_deg,
        causal_mask=causal_mask,
    )
    tile_params = _compute_tile_parameters(config, is_seqlen_sharded=False)

    assert tile_params.section_len == 2560
    assert tile_params.section_len % _K_TILE_SZ == 0


@pytest.mark.fast
def test_exp_selection_rejects_slots_past_section_allocation():
    config = _attention_config()
    tile_params = AttnTileParams(
        seqlen_k_active_updated=4096,
        is_causal=False,
        exp_inst_elems=_EXP_TILE_SZ,
        num_exp_insts_per_large_tile=4,
        num_k_tiles_per_section=6,
    )
    section = SectionParams()
    section.section_idx = 0
    section.section_offset = 0

    assert _exp_tile_selection(0, 1, 1, config, tile_params, section)[0]
    assert not _exp_tile_selection(0, 1, 2, config, tile_params, section)[0]
    assert not _exp_tile_selection(0, 1, 3, config, tile_params, section)[0]
