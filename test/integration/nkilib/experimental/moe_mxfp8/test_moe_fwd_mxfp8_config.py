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

"""Tests for MXFP8 MoE forward pass configuration."""

import nki.language as nl
from nki.dtype import float8_e4m3fn_x4
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_config import (
    _AUTOTUNE_CACHE,
    MatmulMxfp8KernelConfig,
)
from nkilib_src.nkilib.experimental.mlp_mxfp8.common_utils import (
    L_TILE_K,
    TILE_M,
    TILE_N,
    build_tile_sizes,
    get_tile_sizes,
)
from nkilib_src.nkilib.experimental.moe.bwd.moe_bwd_parameters import (
    ActFnType,
    AffinityOption,
    ClampLimits,
    ShardOption,
    SkipMode,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.fwd.moe_fwd_mxfp8_config import (
    MXFP8MOEFwdConfig,
    auto_generate_moe_fwd_configs,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.moe_mxfp8_checkpoint_config import MXFP8MOECheckpointConfig


class TestMXFP8MOEFwdConfigDefaults:
    """Test default instantiation of MXFP8MOEFwdConfig."""

    def test_default_construction(self):
        config = MXFP8MOEFwdConfig()
        assert config.compute_dtype == nl.bfloat16
        assert config.fp8_x4_dtype == float8_e4m3fn_x4
        assert config.activation_type == ActFnType.SiLU
        assert config.shard_option == ShardOption.SHARD_ON_BLOCK
        assert config.affinity_option == AffinityOption.AFFINITY_ON_I
        assert config.is_tensor_update_accumulating
        assert not config.no_indirect_load
        assert not config.bias
        assert config.checkpoint_config is not None
        assert isinstance(config.checkpoint_config, MXFP8MOECheckpointConfig)
        assert config.checkpoint_config.save_gate_up_proj_act
        assert config.checkpoint_config.save_scaled_intermediate

    def test_post_init_creates_phase_configs(self):
        config = MXFP8MOEFwdConfig()
        assert config.gate_up_config is not None
        assert config.down_config is not None
        assert isinstance(config.gate_up_config, MatmulMxfp8KernelConfig)
        assert isinstance(config.down_config, MatmulMxfp8KernelConfig)

    def test_post_init_creates_clamp_limits(self):
        config = MXFP8MOEFwdConfig()
        assert config.clamp_limits is not None
        assert isinstance(config.clamp_limits, ClampLimits)
        assert config.clamp_limits.linear_clamp_upper_limit is None
        assert config.clamp_limits.linear_clamp_lower_limit is None
        assert config.clamp_limits.non_linear_clamp_upper_limit is None
        assert config.clamp_limits.non_linear_clamp_lower_limit is None

    def test_post_init_creates_skip_dma(self):
        config = MXFP8MOEFwdConfig()
        assert config.skip_dma is not None
        assert isinstance(config.skip_dma, SkipMode)
        assert not config.skip_dma.skip_token
        assert not config.skip_dma.skip_weight


class TestMXFP8MOEFwdConfigCustomValues:
    """Test custom instantiation of MXFP8MOEFwdConfig."""

    def test_custom_affinity_option(self):
        config = MXFP8MOEFwdConfig(affinity_option=AffinityOption.AFFINITY_ON_H)
        assert config.affinity_option == AffinityOption.AFFINITY_ON_H

    def test_custom_shard_option(self):
        config = MXFP8MOEFwdConfig(shard_option=ShardOption.SHARD_ON_FREE)
        assert config.shard_option == ShardOption.SHARD_ON_FREE

    def test_custom_accumulation_flag(self):
        config = MXFP8MOEFwdConfig(is_tensor_update_accumulating=False)
        assert not config.is_tensor_update_accumulating

    def test_custom_checkpoint_config(self):
        config = MXFP8MOEFwdConfig(
            checkpoint_config=MXFP8MOECheckpointConfig(
                save_gate_up_proj_act=False,
                save_scaled_intermediate=False,
            )
        )
        assert not config.checkpoint_config.save_gate_up_proj_act
        assert not config.checkpoint_config.save_scaled_intermediate

    def test_custom_no_indirect_load_flag(self):
        config = MXFP8MOEFwdConfig(no_indirect_load=True)
        assert config.no_indirect_load

    def test_custom_clamp_limits(self):
        limits = ClampLimits(
            linear_clamp_upper_limit=1.0,
            linear_clamp_lower_limit=-1.0,
            non_linear_clamp_upper_limit=2.0,
            non_linear_clamp_lower_limit=-2.0,
        )
        config = MXFP8MOEFwdConfig(clamp_limits=limits)
        assert config.clamp_limits.linear_clamp_upper_limit == 1.0
        assert config.clamp_limits.linear_clamp_lower_limit == -1.0
        assert config.clamp_limits.non_linear_clamp_upper_limit == 2.0
        assert config.clamp_limits.non_linear_clamp_lower_limit == -2.0

    def test_custom_skip_dma(self):
        skip = SkipMode(skip_token=True, skip_weight=False)
        config = MXFP8MOEFwdConfig(skip_dma=skip)
        assert config.skip_dma.skip_token
        assert not config.skip_dma.skip_weight

    def test_custom_bias(self):
        config = MXFP8MOEFwdConfig(bias=True)
        assert config.bias

    def test_custom_phase_config_not_overwritten(self):
        custom_gate_up = MatmulMxfp8KernelConfig(
            M=1024,
            K=512,
            N=2048,
            TILES_IN_BLOCK_M=4,
            TILES_IN_BLOCK_N=2,
            TILES_IN_BLOCK_K=1,
        )
        config = MXFP8MOEFwdConfig(gate_up_config=custom_gate_up)
        assert config.gate_up_config.M == 1024
        assert config.gate_up_config.K == 512
        assert config.gate_up_config.N == 2048
        assert config.gate_up_config.TILES_IN_BLOCK_M == 4
        assert config.gate_up_config.TILES_IN_BLOCK_N == 2
        assert config.gate_up_config.TILES_IN_BLOCK_K == 1
        # down_config should still get defaults
        assert config.down_config.M == 0


class TestMXFP8MOEFwdConfigAutotuneCache:
    """auto_generate_moe_fwd_configs consults the DGT autotune cache.

    The forward's GEMMs are unswizzled BF16, so a phase's cache key resolves to
    ``{M}x{K}x{N}_bf16_bf16_dgt`` (run_with_lnc2 is False, so the dims are not
    halved). A shape with a tuned DGT entry gets that entry's blocking; a shape
    with no entry falls through to the heuristic path.
    """

    # A gate/up shape whose key ({block_size}x{H}x{I_TP}_bf16_bf16_dgt) is a tuned
    # DGT entry. Expected values are read from the live cache so the test tracks
    # future re-tunes instead of hard-coding blocking numbers.
    _HIT_BLOCK_SIZE = 2048
    _HIT_H = 4096
    _HIT_I_TP = 1536
    _HIT_KEY = f"{_HIT_BLOCK_SIZE}x{_HIT_H}x{_HIT_I_TP}_bf16_bf16_dgt"

    def test_gate_up_config_hits_dgt_cache(self):
        if self._HIT_KEY not in _AUTOTUNE_CACHE:
            import pytest

            pytest.skip(f"autotune cache no longer carries {self._HIT_KEY}")
        expected = _AUTOTUNE_CACHE[self._HIT_KEY]

        config = MXFP8MOEFwdConfig()
        auto_generate_moe_fwd_configs(config, block_size=self._HIT_BLOCK_SIZE, H=self._HIT_H, I_TP=self._HIT_I_TP)

        gu = config.gate_up_config
        # Dims filled from the derived shape, load method resolves to dgt.
        assert (gu.M, gu.K, gu.N) == (self._HIT_BLOCK_SIZE, self._HIT_H, self._HIT_I_TP)
        assert not gu.lhs_is_swizzled
        assert not gu.rhs_is_swizzled
        assert not gu.run_with_lnc2
        # Blocking + spill_reload come from the cached DGT entry (the fields the
        # dropless impl actually consumes). tile_m/tile_k/tile_n are also copied
        # but the forward pins its own geometry, so they are inert.
        assert gu.TILES_IN_BLOCK_M == expected["TILES_IN_BLOCK_M"]
        assert gu.TILES_IN_BLOCK_N == expected["TILES_IN_BLOCK_N"]
        assert gu.TILES_IN_BLOCK_K == expected["TILES_IN_BLOCK_K"]
        assert gu.TILES_IN_LOAD_M == expected["TILES_IN_LOAD_M"]
        assert gu.TILES_IN_LOAD_N == expected["TILES_IN_LOAD_N"]
        if "spill_reload" in expected:
            assert gu.spill_reload == expected["spill_reload"]

    def test_cache_miss_falls_through_to_heuristic(self):
        # No DGT entry exists for these dims (gate/up 512x1024x256, down
        # 512x256x1024), so both phases take the heuristic fill path.
        block_size, H, I_TP = 512, 1024, 256
        assert f"{block_size}x{H}x{I_TP}_bf16_bf16_dgt" not in _AUTOTUNE_CACHE
        assert f"{block_size}x{I_TP}x{H}_bf16_bf16_dgt" not in _AUTOTUNE_CACHE

        config = MXFP8MOEFwdConfig()
        auto_generate_moe_fwd_configs(config, block_size=block_size, H=H, I_TP=I_TP)

        for phase in (config.gate_up_config, config.down_config):
            # Derived fields are populated with legal values by the heuristic.
            assert phase.TILES_IN_BLOCK_M is not None and phase.TILES_IN_BLOCK_M >= 1
            assert phase.TILES_IN_BLOCK_N is not None and phase.TILES_IN_BLOCK_N >= 1
            assert phase.TILES_IN_BLOCK_K is not None and phase.TILES_IN_BLOCK_K >= 1
            assert phase.TILES_IN_LOAD_M is not None and phase.TILES_IN_LOAD_M >= 1
            assert phase.TILES_IN_LOAD_N is not None and phase.TILES_IN_LOAD_N >= 1
            # TILES_IN_LOAD_* must divide the block (DGT-legal) so the dropless
            # impl's integer-divide load stays valid.
            assert phase.TILES_IN_BLOCK_M % phase.TILES_IN_LOAD_M == 0
            assert phase.TILES_IN_BLOCK_N % phase.TILES_IN_LOAD_N == 0


class TestBuildTileSizes:
    """build_tile_sizes matches get_tile_sizes and supports a non-default tile_n.

    The forward now feeds its per-phase tile_m/tile_k/tile_n through
    build_tile_sizes so a tuned tile_n (e.g. 768, which get_tile_sizes caps at
    TILE_N) can be used, while the default path stays byte-identical.
    """

    def test_defaults_match_get_tile_sizes(self):
        # With dims >= the tile sizes, get_tile_sizes yields the defaults; the
        # explicit builder must produce the identical dict.
        assert build_tile_sizes() == get_tile_sizes(L_TILE_K, TILE_M, TILE_N)
        assert build_tile_sizes(tile_m=TILE_M, l_tile_k=L_TILE_K, tile_n=TILE_N) == get_tile_sizes(2048, 1024, 4096)

    def test_non_default_tile_n_768(self):
        tiles = build_tile_sizes(tile_m=TILE_M, l_tile_k=L_TILE_K, tile_n=768)
        assert tiles["tile_n"] == 768
        # Derived N-dependent shapes scale with the requested tile_n; K/M shapes
        # are unchanged from the default.
        assert tiles["rhs_matmul_tile_physical"] == (tiles["matmul_tile_k_physical"], 768)
        assert tiles["rhs_load_tile"] == (L_TILE_K, 768)
        assert tiles["lhs_matmul_tile_physical"] == (tiles["matmul_tile_k_physical"], TILE_M)
        assert tiles["l_tile_k"] == L_TILE_K
