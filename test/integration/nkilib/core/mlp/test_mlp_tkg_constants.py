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

"""Host-side unit tests for pure-Python helpers in mlp_tkg_constants.

- ``MLPTKGConstants.calculate_constants`` runs its weight-rank dispatch and expert-K
  extraction on host; it calls ``nisa.get_nc_version()`` near the end which requires
  an active backend (raises RuntimeError on host), so the dispatch arcs are validated
  by catching that trailing RuntimeError (rank 6 asserts earlier).
- ``calculate_gate_up_tiles`` / ``calculate_down_tiles`` are fully host-executable and
  are driven with a real ``SbufManager`` plus a directly-constructed dims dataclass.
"""

from types import SimpleNamespace

import nki.language as nl
import pytest
from nkilib_src.nkilib.core.mlp.mlp_tkg.mlp_tkg_constants import (
    MLPTKGConstants,
    MLPTKGConstantsDimensionSizes,
    MLPTKGConstantsGateUpTileCounts,
)
from nkilib_src.nkilib.core.utils.allocator import SbufManager


class _QuantStub:
    def __init__(self, is_quant=False):
        self._is_quant = is_quant

    def is_quant(self):
        return self._is_quant


def _const_params(gate_shape, expert=False, batch_size=1, sequence_len=16, hidden_size=512):
    """params stub for calculate_constants (shard_on_h disabled -> pure code path)."""
    expert_params = None
    if expert:
        expert_params = SimpleNamespace(expert_index=SimpleNamespace(shape=(16, 8)))
    return SimpleNamespace(
        shard_on_h_disabled=True,
        batch_size=batch_size,
        sequence_len=sequence_len,
        hidden_size=hidden_size,
        gate_proj_weights_tensor=SimpleNamespace(shape=gate_shape),
        expert_params=expert_params,
        use_tkg_gate_up_proj_column_tiling=False,
    )


# ---------------------- calculate_constants: rank dispatch ----------------------
# get_nc_version() (near the end of the method) has no backend on host and raises
# RuntimeError; the rank-dispatch/K-extraction arcs execute before that point.


@pytest.mark.fast
def test_calculate_constants_rank2_dense():
    params = _const_params((512, 512))
    with pytest.raises(RuntimeError):
        MLPTKGConstants.calculate_constants(params)


@pytest.mark.fast
def test_calculate_constants_rank3_mx():
    params = _const_params((128, 1, 512))
    with pytest.raises(RuntimeError):
        MLPTKGConstants.calculate_constants(params)


@pytest.mark.fast
def test_calculate_constants_rank4_moe_with_expert_k():
    # rank 4 (MoE fused gate/up) plus expert_index present -> exercises K extraction.
    params = _const_params((2, 512, 2, 512), expert=True)
    with pytest.raises(RuntimeError):
        MLPTKGConstants.calculate_constants(params)


@pytest.mark.fast
def test_calculate_constants_rank5_mx_moe():
    params = _const_params((2, 128, 2, 1, 512))
    with pytest.raises(RuntimeError):
        MLPTKGConstants.calculate_constants(params)


@pytest.mark.fast
def test_calculate_constants_rank6_asserts():
    # Unsupported weight rank -> kernel_assert(False) before get_nc_version.
    params = _const_params((2, 2, 2, 2, 2, 2))
    with pytest.raises(AssertionError):
        MLPTKGConstants.calculate_constants(params)


# ------------------- calculate_gate_up_tiles / calculate_down_tiles -------------------


def _tile_params():
    return SimpleNamespace(
        up_proj_weights_tensor=SimpleNamespace(dtype=nl.bfloat16),
        down_proj_weights_tensor=SimpleNamespace(dtype=nl.bfloat16),
        quant_params=_QuantStub(is_quant=False),
        use_tkg_gate_up_proj_column_tiling=False,
        use_tkg_down_proj_column_tiling=False,
    )


def _dims(I, H_per_shard, max_I_shard_size, H1_shard, H0=128):
    return MLPTKGConstantsDimensionSizes(
        _pmax=128,
        _psum_fmax=512,
        _psum_bmax=8,
        _q_width=4,
        _q_height=8,
        T=16,
        H=H_per_shard,
        I=I,
        H0=H0,
        H1=H_per_shard // H0,
        I0=128,
        num_shards=1,
        shard_id=0,
        H_shard=H_per_shard,
        H1_shard=H1_shard,
        H1_offset=0,
        H_per_shard=H_per_shard,
        column_tiling_dim=128,
        column_tiling_factor=1,
        max_I_shard_size=max_I_shard_size,
        hidden_layout=None,  # ty: ignore[invalid-argument-type]
    )


@pytest.mark.fast
def test_gate_up_tiles_auto_alloc():
    sbm = SbufManager(0, 1_000_000, use_auto_alloc=True)
    dims = _dims(I=512, H_per_shard=1024, max_I_shard_size=1024, H1_shard=8)
    res = MLPTKGConstants.calculate_gate_up_tiles(_tile_params(), dims, sbm)
    assert res.num_allocated_w_tile == 2


@pytest.mark.fast
def test_gate_up_tiles_share_memory_ample():
    sbm = SbufManager(0, 10_000_000, use_auto_alloc=False)
    dims = _dims(I=512, H_per_shard=1024, max_I_shard_size=1024, H1_shard=8)
    res = MLPTKGConstants.calculate_gate_up_tiles(_tile_params(), dims, sbm, share_memory_scope=True)
    assert res.num_allocated_w_tile > 0


@pytest.mark.fast
def test_gate_up_tiles_share_memory_tight_reduces_htile():
    # Tiny free space forces the shared-scope HTile fallback to min_HTile (512).
    sbm = SbufManager(0, 5000, use_auto_alloc=False)
    dims = _dims(I=512, H_per_shard=1024, max_I_shard_size=512, H1_shard=8)
    res = MLPTKGConstants.calculate_gate_up_tiles(_tile_params(), dims, sbm, share_memory_scope=True)
    assert res.HTile == 512
    assert res.num_allocated_w_tile > 0


@pytest.mark.fast
def test_gate_up_tiles_owned_scope_reduces_htile_and_I():
    # Non-shared owned scope with large I and tight space exercises both the
    # HTile-reduction and the I-reduction while-loops.
    sbm = SbufManager(0, 40000, use_auto_alloc=False)
    dims = _dims(I=4096, H_per_shard=512, max_I_shard_size=4096, H1_shard=4)
    res = MLPTKGConstants.calculate_gate_up_tiles(_tile_params(), dims, sbm)
    assert res.num_allocated_w_tile > 0
    assert res.I_shard_size < 4096


@pytest.mark.fast
def test_down_tiles_auto_alloc():
    sbm = SbufManager(0, 1_000_000, use_auto_alloc=True)
    dims = _dims(I=512, H_per_shard=1024, max_I_shard_size=1024, H1_shard=8)
    gate_info = MLPTKGConstantsGateUpTileCounts(
        HTile=512,
        num_allocated_w_tile=2,
        last_accessed_addr=0,
        up_psum_base_bank=4,
        I_shard_size=512,
    )
    res = MLPTKGConstants.calculate_down_tiles(_tile_params(), dims, gate_info, sbm)
    assert res.num_allocated_w_tile == 2
    assert res.HTile == 1024


@pytest.mark.fast
def test_down_tiles_owned_scope_weight_base_offset():
    # Non-auto path: Gate/Up's last accessed address overlaps Down's tiles, so the
    # starting weight index is offset to avoid anti-dependencies.
    sbm = SbufManager(0, 10000, use_auto_alloc=False)
    dims = _dims(I=512, H_per_shard=512, max_I_shard_size=512, H1_shard=4)
    gate_info = MLPTKGConstantsGateUpTileCounts(
        HTile=512,
        num_allocated_w_tile=2,
        last_accessed_addr=2048,
        up_psum_base_bank=4,
        I_shard_size=512,
    )
    res = MLPTKGConstants.calculate_down_tiles(_tile_params(), dims, gate_info, sbm)
    assert res.weight_base_idx == 2
