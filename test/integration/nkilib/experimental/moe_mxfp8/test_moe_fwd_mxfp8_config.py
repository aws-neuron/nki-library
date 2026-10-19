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

from inspect import signature

import nki.language as nl
import pytest
from nki.dtype import float8_e4m3fn_x4  # ty: ignore[unresolved-import]
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_config import (
    MatmulMxfp8KernelConfig,
)
from nkilib_src.nkilib.experimental.mlp_mxfp8._common_utils import (
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
from nkilib_src.nkilib.experimental.moe_mxfp8.config_helpers import ExecutionMode
from nkilib_src.nkilib.experimental.moe_mxfp8.fwd.blockwise_mm_forward_mxfp8 import (
    _validate_kernel_options,
    _weight_orientation,
    blockwise_mm_fwd_mxfp8,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.fwd.blockwise_mm_forward_mxfp8_torch import (
    blockwise_mm_fwd_mxfp8_torch_ref,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.fwd.config import (
    MXFP8MOEFwdConfig,
    _auto_generate_moe_fwd_configs,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.moe_mxfp8_checkpoint_config import MXFP8MOECheckpointConfig
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.common_dataclasses import (
    SwizzleMode,
    TensorDescriptor,
    TensorOrientation,
)


class TestMXFP8MOEFwdConfigDefaults:
    """Test default instantiation of MXFP8MOEFwdConfig."""

    def test_default_construction(self):
        config = MXFP8MOEFwdConfig()
        assert config.compute_dtype == nl.bfloat16
        assert config.fp8_x4_dtype == float8_e4m3fn_x4
        assert config.activation_type == ActFnType.SiLU
        assert config.shard_option == ShardOption.SHARD_ON_BLOCK
        assert config.affinity_option == AffinityOption.AFFINITY_ON_I
        assert not config.is_tensor_update_accumulating
        assert config.execution_mode.is_dense
        assert config.num_static_block is None
        assert config.reuse_spilled_weights
        assert config.use_1x32_pe_swizzle
        assert not config.bias
        assert config.checkpoint_config is not None
        assert isinstance(config.checkpoint_config, MXFP8MOECheckpointConfig)
        assert config.checkpoint_config.save_gate_up_proj_act
        # Weight layout descriptors stay None when unset (the entry resolves the default
        # per path: BF16 -> plain unswizzled BF16, pre-quantized -> 1x32/WRAPX by path).
        assert config.gate_up_weight_td is None
        assert config.down_weight_td is None
        assert not hasattr(config, "gate_up_proj_act_td")
        assert not hasattr(config, "gate_up_proj_act_layout")

    def test_phase_defaults_follow_cache_lookup(self):
        config = MXFP8MOEFwdConfig()
        assert config.gate_up_config is None
        assert config.down_config is None
        config.resolve_phase_defaults()
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
        config = MXFP8MOEFwdConfig(is_tensor_update_accumulating=True)
        assert config.is_tensor_update_accumulating

    def test_custom_num_static_block(self):
        config = MXFP8MOEFwdConfig(num_static_block=4)
        assert config.num_static_block == 4

    def test_custom_checkpoint_config(self):
        config = MXFP8MOEFwdConfig(checkpoint_config=MXFP8MOECheckpointConfig(save_gate_up_proj_act=False))
        assert not config.checkpoint_config.save_gate_up_proj_act

    def test_weight_tds_carry_swizzle_mode(self):
        config = MXFP8MOEFwdConfig(
            gate_up_weight_td=TensorDescriptor(orientation=TensorOrientation.K_BY_F),
        )
        # K_BY_F unswizzled BF16 selects PE swizzle (see TensorDescriptor.__post_init__).
        assert config.gate_up_weight_td is not None
        assert config.gate_up_weight_td.swizzle_mode == SwizzleMode.PE

    def test_unset_weight_tds_resolve_to_forward_natural(self):
        config = MXFP8MOEFwdConfig()
        assert config.gate_up_weight_td is None
        assert config.down_weight_td is None
        assert _weight_orientation(config.gate_up_weight_td) == TensorOrientation.F_BY_K
        assert _weight_orientation(config.down_weight_td) == TensorOrientation.F_BY_K

    @pytest.mark.parametrize("field", ["gate_up_weight_td", "down_weight_td"])
    def test_each_kbyf_weight_supports_routed_experts(self, field):
        config = MXFP8MOEFwdConfig(
            execution_mode=ExecutionMode.ROUTED_MOE,
            fast_dma_transpose=False,
            use_1x32_pe_swizzle=False,
            reuse_spilled_weights=False,
        )
        setattr(config, field, TensorDescriptor(orientation=TensorOrientation.K_BY_F))
        _validate_kernel_options(config=config)

    @pytest.mark.parametrize("field", ["gate_up_weight_td", "down_weight_td"])
    def test_fast_dma_policy_keeps_kbyf_weight_on_pe(self, field):
        config = MXFP8MOEFwdConfig(
            fast_dma_transpose=True,
            use_1x32_pe_swizzle=False,
        )
        setattr(config, field, TensorDescriptor(orientation=TensorOrientation.K_BY_F))
        _validate_kernel_options(config=config)
        assert getattr(config, field).swizzle_mode == SwizzleMode.PE

    def test_custom_single_expert_dense_flag(self):
        # Routed callers opt out; the dense-path load modes must go with it.
        config = MXFP8MOEFwdConfig(
            execution_mode=ExecutionMode.ROUTED_MOE,
            reuse_spilled_weights=False,
            use_1x32_pe_swizzle=False,
        )
        assert not config.execution_mode.is_dense
        assert not config.reuse_spilled_weights
        assert not config.use_1x32_pe_swizzle

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
        assert config.down_config is None
        config.resolve_phase_defaults()
        assert config.down_config.M == 0


class TestMXFP8MOEFwdDefaultResolution:
    def test_default_resolution_ignores_standalone_matmul_cache(self, monkeypatch):
        from nkilib_src.nkilib.experimental.matmul_mxfp8 import matmul_mxfp8_config

        class InvalidTuningCache(dict):
            def get(self, key, default=None):
                return {
                    "tile_m": 1,
                    "tile_k": 1,
                    "tile_n": 1,
                    "TILES_IN_BLOCK_M": 0,
                    "TILES_IN_BLOCK_N": 0,
                    "TILES_IN_BLOCK_K": 0,
                    "TILES_IN_LOAD_M": 0,
                    "TILES_IN_LOAD_N": 0,
                    "spill_reload": False,
                }

        monkeypatch.setattr(matmul_mxfp8_config, "_AUTOTUNE_CACHE", InvalidTuningCache())
        config = MXFP8MOEFwdConfig()
        _auto_generate_moe_fwd_configs(config, block_size=512, H=1024, I_TP=256)
        for phase in (config.gate_up_config, config.down_config):
            assert phase.tile_m == 128
            assert phase.tile_k == 512
            assert phase.spill_reload
            assert phase.TILES_IN_BLOCK_M is not None
            assert phase.TILES_IN_BLOCK_N is not None
            assert phase.TILES_IN_BLOCK_K is not None
            assert phase.TILES_IN_LOAD_M is not None
            assert phase.TILES_IN_LOAD_N is not None
            assert phase.TILES_IN_BLOCK_M >= 1
            assert phase.TILES_IN_BLOCK_N >= 1
            assert phase.TILES_IN_BLOCK_K >= 1
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


class TestMXFP8MOEFwdKernelDefaults:
    """The forward kernel entry defaults block_size to 4096 and makes config optional.

    A framework can call ``blockwise_mm_fwd_mxfp8`` with only the required tensors:
    ``block_size`` falls back to 4096 and ``config=None`` resolves to a default
    ``MXFP8MOEFwdConfig()`` inside the kernel.
    """

    def test_block_size_defaults_to_4096(self):
        assert signature(blockwise_mm_fwd_mxfp8).parameters["block_size"].default == 4096

    def test_config_is_optional(self):
        assert signature(blockwise_mm_fwd_mxfp8).parameters["config"].default is None

    def test_dynamic_args_preserve_existing_positional_order(self):
        expected = [
            "hidden_states",
            "expert_affinities_masked",
            "gate_up_proj_weight",
            "down_proj_weight",
            "token_position_to_id",
            "block_to_expert",
            "block_size",
            "config",
            "gate_up_weight_scales",
            "down_weight_scales",
            "num_valid_tokens",
        ]
        assert list(signature(blockwise_mm_fwd_mxfp8).parameters) == expected
        assert list(signature(blockwise_mm_fwd_mxfp8_torch_ref).parameters) == expected

    def test_config_none_resolves_to_default_config(self):
        # The kernel builds a default config when none is supplied, so a bare
        # MXFP8MOEFwdConfig() is a complete, valid configuration on its own.
        default = MXFP8MOEFwdConfig()
        default.resolve_phase_defaults()
        assert default.shard_option == ShardOption.SHARD_ON_BLOCK
        assert default.affinity_option == AffinityOption.AFFINITY_ON_I
        assert default.gate_up_config is not None
        assert default.down_config is not None
        # spill/reload and scale packing live on the phase configs, both on by default.
        for phase in (default.gate_up_config, default.down_config):
            assert phase.spill_reload
            assert phase.enable_scale_packing

    def test_torch_ref_mirrors_kernel_defaults(self):
        # The reference must resolve an omitted block_size / config identically to
        # the kernel, or a defaults-relying test would diverge from the golden.
        kernel = signature(blockwise_mm_fwd_mxfp8).parameters
        ref = signature(blockwise_mm_fwd_mxfp8_torch_ref).parameters
        assert ref["block_size"].default == kernel["block_size"].default == 4096
        assert ref["config"].default is None and kernel["config"].default is None
