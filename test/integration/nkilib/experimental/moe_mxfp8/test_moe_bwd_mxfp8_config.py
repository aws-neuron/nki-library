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

"""Tests for MXFP8 MoE backward pass configuration."""

import nki.language as nl
from nki.dtype import float8_e4m3fn_x4  # ty: ignore[unresolved-import]
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_config import MatmulMxfp8KernelConfig
from nkilib_src.nkilib.experimental.moe.bwd.moe_bwd_parameters import (
    ActFnType,
    AffinityOption,
    ClampLimits,
    ShardOption,
    SkipMode,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.bwd.blockwise_mm_backward_mxfp8 import _resolve_phase_configs
from nkilib_src.nkilib.experimental.moe_mxfp8.bwd.config import (
    MXFP8MOEBwdConfig,
    SwizzleMode,
    TransposeMode,
)
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.common_dataclasses import QuantScheme, TensorDescriptor

from test.integration.nkilib.experimental.moe_mxfp8.tuned_blocking import (
    MAX_DGT_LOAD_WIDTH,
    SHAPE_TUNED_CONFIGS,
    derive_load_tiles,
    tune_block_and_load_tiles,
)


class TestTensorDescriptorSwizzleMode:
    def test_uses_typed_tensor_local_mode(self):
        descriptor = TensorDescriptor(swizzle_mode=SwizzleMode.PE)

        assert descriptor.swizzle_mode == SwizzleMode.PE
        assert descriptor.uses_pe_swizzle

    def test_legacy_boolean_remains_supported(self):
        descriptor = TensorDescriptor(load_with_PE_swizzle=True)

        assert descriptor.load_with_PE_swizzle
        assert descriptor.uses_pe_swizzle


class TestMXFP8MOEBwdConfigDefaults:
    """Test default instantiation of MXFP8MOEBwdConfig."""

    def test_default_construction(self):
        config = MXFP8MOEBwdConfig()
        assert config.compute_dtype == nl.bfloat16
        assert config.fp8_x4_dtype == float8_e4m3fn_x4
        assert config.activation_type == ActFnType.SiLU
        assert config.shard_option == ShardOption.SHARD_ON_FREE
        assert config.affinity_option == AffinityOption.AFFINITY_ON_H
        assert config.accumulate_hidden_states_grad
        assert not config.skip_grad_initialization
        assert not config.single_expert_dense
        assert not config.fast_dma_transpose
        assert not config.bias

    def test_post_init_creates_phase_configs(self):
        config = MXFP8MOEBwdConfig()
        assert config.phase1_config is not None
        assert config.phase2_config is not None
        assert config.phase3_config is not None
        assert config.phase4_config is not None
        assert isinstance(config.phase1_config, MatmulMxfp8KernelConfig)
        assert isinstance(config.phase2_config, MatmulMxfp8KernelConfig)
        assert isinstance(config.phase3_config, MatmulMxfp8KernelConfig)
        assert isinstance(config.phase4_config, MatmulMxfp8KernelConfig)

    def test_post_init_phase_configs_have_default_blocking(self):
        config = MXFP8MOEBwdConfig()
        for phase_config in [
            config.phase1_config,
            config.phase2_config,
            config.phase3_config,
            config.phase4_config,
        ]:
            assert phase_config.M == 0
            assert phase_config.K == 0
            assert phase_config.N == 0
            assert phase_config.TILES_IN_BLOCK_M == 1
            assert phase_config.TILES_IN_BLOCK_N == 1
            assert phase_config.TILES_IN_BLOCK_K == 1

    def test_post_init_creates_clamp_limits(self):
        config = MXFP8MOEBwdConfig()
        assert config.clamp_limits is not None
        assert isinstance(config.clamp_limits, ClampLimits)
        assert config.clamp_limits.linear_clamp_upper_limit is None
        assert config.clamp_limits.linear_clamp_lower_limit is None
        assert config.clamp_limits.non_linear_clamp_upper_limit is None
        assert config.clamp_limits.non_linear_clamp_lower_limit is None

    def test_skip_dma_remains_none_after_post_init(self):
        config = MXFP8MOEBwdConfig()
        assert config.skip_dma is None


class TestMXFP8MOEBwdConfigCustomValues:
    """Test custom instantiation of MXFP8MOEBwdConfig."""

    def test_custom_activation_type(self):
        config = MXFP8MOEBwdConfig(activation_type=ActFnType.GELU)
        assert config.activation_type == ActFnType.GELU

    def test_custom_shard_option(self):
        config = MXFP8MOEBwdConfig(shard_option=ShardOption.SHARD_ON_HIDDEN)
        assert config.shard_option == ShardOption.SHARD_ON_HIDDEN

    def test_custom_affinity_option(self):
        config = MXFP8MOEBwdConfig(affinity_option=AffinityOption.AFFINITY_ON_I)
        assert config.affinity_option == AffinityOption.AFFINITY_ON_I

    def test_custom_accumulation_flags(self):
        config = MXFP8MOEBwdConfig(
            accumulate_hidden_states_grad=False,
            skip_grad_initialization=True,
        )
        assert not config.accumulate_hidden_states_grad
        assert config.skip_grad_initialization

    def test_custom_clamp_limits(self):
        limits = ClampLimits(
            linear_clamp_upper_limit=1.0,
            linear_clamp_lower_limit=-1.0,
            non_linear_clamp_upper_limit=2.0,
            non_linear_clamp_lower_limit=-2.0,
        )
        config = MXFP8MOEBwdConfig(clamp_limits=limits)
        assert config.clamp_limits.linear_clamp_upper_limit == 1.0
        assert config.clamp_limits.linear_clamp_lower_limit == -1.0
        assert config.clamp_limits.non_linear_clamp_upper_limit == 2.0
        assert config.clamp_limits.non_linear_clamp_lower_limit == -2.0

    def test_custom_skip_dma(self):
        skip = SkipMode(skip_token=True, skip_weight=False)
        config = MXFP8MOEBwdConfig(skip_dma=skip)
        assert config.skip_dma.skip_token
        assert not config.skip_dma.skip_weight

    def test_custom_bias(self):
        config = MXFP8MOEBwdConfig(bias=True)
        assert config.bias

    def test_custom_single_expert_dense(self):
        config = MXFP8MOEBwdConfig(single_expert_dense=True)
        assert config.single_expert_dense

    def test_custom_fast_dma_transpose(self):
        config = MXFP8MOEBwdConfig(fast_dma_transpose=True)
        assert config.fast_dma_transpose

    def test_custom_phase_config_not_overwritten(self):
        custom_phase1 = MatmulMxfp8KernelConfig(
            M=1024,
            K=512,
            N=2048,
            TILES_IN_BLOCK_M=4,
            TILES_IN_BLOCK_N=2,
            TILES_IN_BLOCK_K=1,
            quant_scheme=QuantScheme.WRAPX,
        )
        config = MXFP8MOEBwdConfig(phase1_config=custom_phase1)
        assert config.phase1_config.M == 1024
        assert config.phase1_config.K == 512
        assert config.phase1_config.N == 2048
        assert config.phase1_config.TILES_IN_BLOCK_M == 4
        assert config.phase1_config.TILES_IN_BLOCK_N == 2
        assert config.phase1_config.TILES_IN_BLOCK_K == 1
        # Other phases should still get defaults
        assert config.phase2_config.M == 0
        assert config.phase3_config.M == 0
        assert config.phase4_config.M == 0

    def test_partial_phase_configs_inherit_wrapper_defaults_only_when_omitted(self):
        phase1 = MatmulMxfp8KernelConfig(
            M=0,
            K=0,
            N=0,
            quant_scheme=QuantScheme.WRAPX,
            spill_reload=False,
            enable_scale_packing=False,
        )
        phase3 = MatmulMxfp8KernelConfig(
            M=0,
            K=0,
            N=0,
            quant_scheme=QuantScheme.WRAPX,
            spill_reload=False,
            enable_scale_packing=False,
        )
        config = MXFP8MOEBwdConfig(phase1_config=phase1, phase3_config=phase3)
        phase_shapes = (
            (256, 512, 128),
            (256, 512, 256),
            (256, 256, 256),
            (256, 256, 256),
        )

        _resolve_phase_configs(
            config=config,
            provided_phase_configs=(phase1, None, phase3, None),
            phase_shapes=phase_shapes,
            hidden_size=512,
            spill_reload=True,
            use_scale_packing=True,
            run_with_lnc2=True,
        )

        assert not config.phase1_config.spill_reload
        assert not config.phase1_config.enable_scale_packing
        assert not config.phase3_config.spill_reload
        assert not config.phase3_config.enable_scale_packing
        assert config.phase2_config.spill_reload
        assert config.phase2_config.enable_scale_packing
        assert config.phase4_config.spill_reload
        assert config.phase4_config.enable_scale_packing

        for phase_config, phase_shape in zip(
            (
                config.phase1_config,
                config.phase2_config,
                config.phase3_config,
                config.phase4_config,
            ),
            phase_shapes,
            strict=True,
        ):
            assert (phase_config.M, phase_config.K, phase_config.N) == phase_shape
            assert phase_config.run_with_lnc2
            assert phase_config.tile_m is not None
            assert phase_config.tile_k is not None
            assert phase_config.tile_n is not None

    def test_load_m_defaults_to_one(self):
        phase3 = MatmulMxfp8KernelConfig(
            M=0,
            K=0,
            N=0,
            TILES_IN_BLOCK_M=12,
            TILES_IN_BLOCK_N=4,
            TILES_IN_BLOCK_K=2,
            quant_scheme=QuantScheme.WRAPX,
        )
        config = MXFP8MOEBwdConfig(phase3_config=phase3)

        _resolve_phase_configs(
            config=config,
            provided_phase_configs=(None, None, phase3, None),
            phase_shapes=(
                (4096, 4096, 768),
                (4096, 3072, 2048),
                (3072, 4096, 2048),
                (2048, 4096, 1536),
            ),
            hidden_size=4096,
            spill_reload=False,
            use_scale_packing=False,
            run_with_lnc2=True,
        )

        assert config.phase3_config.TILES_IN_LOAD_M == 1


class TestMXFP8MOEBwdConfigValidation:
    def test_preserves_nested_configs_and_modes(self):
        config = MXFP8MOEBwdConfig(
            phase1_config=MatmulMxfp8KernelConfig(
                M=0,
                K=0,
                N=0,
                tile_n=256,
                TILES_IN_BLOCK_N=3,
                quant_scheme=QuantScheme.WRAPX,
                spill_reload=True,
                enable_scale_packing=False,
            ),
            phase3_transpose_mode=TransposeMode.DMA,
        )

        assert config.phase1_config.spill_reload
        assert not config.phase1_config.enable_scale_packing
        assert config.phase1_config.tile_n == 256
        assert config.phase1_config.TILES_IN_BLOCK_N == 3
        assert config.phase1_config.quant_scheme == QuantScheme.WRAPX
        assert config.phase3_transpose_mode == TransposeMode.DMA
        assert config.phase4_transpose_mode == TransposeMode.NC


class TestMXFP8MOEBwdTunedBlocking:
    def test_load_tuning_respects_dgt_width(self):
        assert derive_load_tiles(4, 512) == 2
        assert derive_load_tiles(3, 256) == 3

    def test_odd_blocks_are_retuned_to_multi_tile_loads(self):
        assert tune_block_and_load_tiles(3, 512) == (2, 2)
        assert tune_block_and_load_tiles(5, 512) == (4, 2)
        assert tune_block_and_load_tiles(7, 512) == (6, 2)

    def test_all_tuned_phases_have_explicit_legal_blocking_and_loads(self):
        for config in SHAPE_TUNED_CONFIGS.values():
            for phase_config in (
                config.phase1_config,
                config.phase2_config,
                config.phase3_config,
                config.phase4_config,
            ):
                block_m = phase_config.TILES_IN_BLOCK_M
                block_n = phase_config.TILES_IN_BLOCK_N
                block_k = phase_config.TILES_IN_BLOCK_K
                load_m = phase_config.TILES_IN_LOAD_M
                load_n = phase_config.TILES_IN_LOAD_N
                tile_n = phase_config.tile_n
                assert block_m is not None
                assert block_n is not None
                assert block_k is not None
                assert load_m is not None
                assert load_n is not None
                assert tile_n is not None
                assert min(block_m, block_n, block_k, load_m, load_n) >= 1
                assert block_m % load_m == 0
                assert block_n % load_n == 0
                if block_m > 1:
                    assert load_m > 1
                if block_n > 1:
                    assert load_n > 1
                assert load_n * tile_n <= MAX_DGT_LOAD_WIDTH
