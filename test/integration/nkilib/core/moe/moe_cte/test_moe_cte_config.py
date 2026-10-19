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

"""Host-side unit tests for MoE CTE configuration/dispatch helpers.

These exercise pure-Python config dataclasses and dispatch enums that carry no
device component, so they run identically in simulation and on hardware and need
no kernel compilation. They cover branches that the device test suite leaves
untouched because every device config leaves the relevant flag at a single value
(e.g. out_p_offset always 0, force_lnc1 always False, a config always supplied).
"""

import pytest
from nkilib_src.nkilib.core.moe.moe_cte.bwmm_func import BWMMFunc
from nkilib_src.nkilib.core.moe.moe_cte.moe_cte import (
    MoECTEImplementation,
    MoECTEShardingSpecs,
    MoECTESpec,
    ShardOnBlockConfig,
    ShardOnIConfig,
)
from nkilib_src.nkilib.core.moe.moe_cte.moe_cte_mx_utils import ProjConfig as MXProjConfig
from nkilib_src.nkilib.core.moe.moe_cte.projection_mx_constants import (
    ProjConfig,
    pad_to_valid_qmx_partitions,
)

from test.utils.pytest_test_metadata import pytest_test_metadata


@pytest_test_metadata(name="MoE CTE Config", tags=["moe", "cte"])
@pytest.mark.fast
class TestMoeCteConfig:
    """Pure-Python config/dispatch coverage (no device)."""

    def test_pad_to_valid_qmx_partitions(self):
        # Exact and below a valid size round up to that size (loop hits `valid >= n`).
        assert pad_to_valid_qmx_partitions(1) == 32
        assert pad_to_valid_qmx_partitions(32) == 32
        assert pad_to_valid_qmx_partitions(33) == 64
        assert pad_to_valid_qmx_partitions(128) == 128
        # Larger than every valid size falls through the loop to the 128 default.
        assert pad_to_valid_qmx_partitions(200) == 128

    def test_bwmm_func_dispatch_variants(self):
        # Each variant not exercised by the device suite: confirm callable + flags.
        fn, is_jit, is_dynamic = BWMMFunc.SHARD_ON_BLOCK_V2.get_bwmm_func()
        assert callable(fn) and is_jit is True and is_dynamic is False
        fn, is_jit, is_dynamic = BWMMFunc.SHARD_ON_BLOCK_HW.get_bwmm_func()
        assert callable(fn) and is_jit is True and is_dynamic is True
        fn, is_jit, is_dynamic = BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING.get_bwmm_func()
        assert callable(fn) and is_jit is True and is_dynamic is False

    def test_bwmm_func_dispatch_unhandled_variant(self):
        # Variants with no dispatch mapping hit the defensive `assert False`.
        with pytest.raises(AssertionError):
            BWMMFunc.SHARD_ON_HIDDEN.get_bwmm_func()

    def test_sharding_specs_defaults_when_config_omitted(self):
        # Block implementation with no config -> default ShardOnBlockConfig.
        specs = MoECTEShardingSpecs(MoECTESpec(implementation=MoECTEImplementation.shard_on_block))
        assert isinstance(specs.shard_on_block, ShardOnBlockConfig)
        assert specs.shard_on_I is None

        # I implementation with no config -> default ShardOnIConfig.
        specs = MoECTEShardingSpecs(MoECTESpec(implementation=MoECTEImplementation.shard_on_i))
        assert isinstance(specs.shard_on_I, ShardOnIConfig)
        assert specs.shard_on_block is None

    def test_sharding_specs_implementation_without_config_slot(self):
        # An MX-I implementation matches neither the block nor the I-family branch,
        # so both config slots stay None (fall-through of the elif chain).
        specs = MoECTEShardingSpecs(MoECTESpec(implementation=MoECTEImplementation.shard_on_i_mx))
        assert specs.shard_on_block is None
        assert specs.shard_on_I is None

    def test_proj_config_out_p_offset_and_force_lnc1(self):
        # Default: out_p_offset == 0 and force_lnc1 False (both `if` arcs not taken).
        cfg = ProjConfig(H=512, I=2048, BxS=64, n_prgs=1, prg_id=0)
        assert cfg.out_p_offset == 0

        # Non-zero out_p_offset exercises the partition-offset validation block.
        cfg = ProjConfig(H=512, I=2048, BxS=64, n_prgs=1, prg_id=0, out_p_offset=32)
        assert cfg.out_p_offset == 32

        # force_lnc1 collapses the program grid to a single program.
        cfg = ProjConfig(H=512, I=2048, BxS=64, n_prgs=1, prg_id=0, force_lnc1=True)
        assert cfg.n_prgs == 1 and cfg.prg_id == 0

    def test_mx_proj_config_i_shard_force_lnc1(self):
        cfg = MXProjConfig(H=512, I=1024, BxS=64, n_prgs=1, prg_id=0, sharding_config="I", force_lnc1=True)
        assert cfg.n_prgs == 1 and cfg.prg_id == 0

    def test_mx_proj_config_h_shard_offset_no_force_lnc1(self):
        # H-sharding path with a non-zero partition offset and force_lnc1 left off.
        # This covers the offset-validation block (out_p_offset != 0) and the
        # force_lnc1==False fall-through to _check_shapes_H_sharded; the device
        # suite already exercises the force_lnc1==True side on the H path.
        cfg = MXProjConfig(
            H=512,
            I=1024,
            BxS=64,
            n_prgs=1,
            prg_id=0,
            sharding_config="H",
            out_p_offset=32,
            force_lnc1=False,
        )
        assert cfg.out_p_offset == 32

    def test_mx_proj_config_unsupported_sharding(self):
        # Neither "I" nor "H" hits the defensive kernel_assert in __post_init__.
        with pytest.raises(AssertionError):
            MXProjConfig(H=512, I=1024, BxS=64, n_prgs=1, prg_id=0, sharding_config="Z")
