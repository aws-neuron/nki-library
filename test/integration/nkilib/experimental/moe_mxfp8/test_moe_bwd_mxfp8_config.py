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
from nki.dtype import float8_e4m3fn_x4
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_config import MatmulMxfp8KernelConfig
from nkilib_src.nkilib.experimental.moe.bwd.moe_bwd_parameters import (
    ActFnType,
    AffinityOption,
    ClampLimits,
    ShardOption,
    SkipMode,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.bwd.moe_bwd_mxfp8_config import MXFP8MOEBwdConfig


class TestMXFP8MOEBwdConfigDefaults:
    """Test default instantiation of MXFP8MOEBwdConfig."""

    def test_default_construction(self):
        config = MXFP8MOEBwdConfig()
        assert config.compute_dtype == nl.bfloat16
        assert config.fp8_x4_dtype == float8_e4m3fn_x4
        assert config.activation_type == ActFnType.SiLU
        assert config.shard_option == ShardOption.SHARD_ON_FREE
        assert config.affinity_option == AffinityOption.AFFINITY_ON_H
        assert config.is_tensor_update_accumulating
        assert not config.skip_grad_initialization
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
            is_tensor_update_accumulating=False,
            skip_grad_initialization=True,
        )
        assert not config.is_tensor_update_accumulating
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

    def test_custom_phase_config_not_overwritten(self):
        custom_phase1 = MatmulMxfp8KernelConfig(
            M=1024,
            K=512,
            N=2048,
            TILES_IN_BLOCK_M=4,
            TILES_IN_BLOCK_N=2,
            TILES_IN_BLOCK_K=1,
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
