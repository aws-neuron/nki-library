# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""High-rank attention block TKG tests requiring >4 NeuronCores.

These tests are split into a separate file so the pipeline routes only this
file to 48xl hosts (the detection is file-level based on @pytest.mark.high_rank).
"""

from typing import final

import pytest

from test.integration.nkilib.experimental.transformer.test_attention_block_tkg import (
    RANGE_ATTN_BLK_CFGS,
    _get_attention_block_metadata,
    _run_attention_block_test,
)

try:
    from test.integration.nkilib.experimental.transformer.test_attention_block_tkg_model_config import (
        attention_block_tkg_model_configs,
    )
except ImportError:
    attention_block_tkg_model_configs = {}

from test.integration.nkilib.experimental.transformer.test_attention_block_tkg_utils import (
    AttnBlkTestConfig,
)
from test.utils.common_dataclasses import ModelTestType, Platforms
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator


def _high_rank_cfgs(cfgs: list[AttnBlkTestConfig]) -> list[AttnBlkTestConfig]:
    """Filter configs that need more than 4 NeuronCores."""
    return [c for c in cfgs if c.is_high_rank()]


@pytest_test_metadata(name="Attention Block TKG High Rank", tags=["model"])
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN3, Platforms.TRN3_A0])
@pytest_marks(["attention", "tkg", "experimental", "mx"])
@final
@pytest.mark.high_rank
class TestRangeAttnBlkHighRank:
    """Attention block TKG tests requiring >4 NeuronCores (KVDP/CP > 4)."""

    # fmt: off
    @pytest.mark.parametrize("attn_blk_cfg",
        _high_rank_cfgs(RANGE_ATTN_BLK_CFGS),
        ids=lambda p: p.test_id(),
    # fmt: on
    )
    def test_attn_blk_megakernel(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        attn_blk_cfg: AttnBlkTestConfig,
    ):
        assert attn_blk_cfg.is_high_rank(), \
            f"Low-rank config (KVDP={attn_blk_cfg.KVDP}, CP={attn_blk_cfg.CP}) belongs in test_attention_block_tkg.py"
        if attn_blk_cfg.xfail_reason:
            pytest.xfail(attn_blk_cfg.xfail_reason)
        _run_attention_block_test(
            test_manager=test_manager,
            platform_target=platform_target,
            cfg=attn_blk_cfg,
        )


@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN3, Platforms.TRN3_A0])
@pytest_marks(["attention", "tkg", "experimental", "mx", "model"])
@final
@pytest.mark.high_rank
class TestAttnBlkModelHighRank:
    """Model regression tests for attention block TKG requiring >4 NeuronCores."""

    def _run_model_test(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        cfg: AttnBlkTestConfig,
    ):
        assert cfg.is_high_rank(), (
            f"Low-rank config (KVDP={cfg.KVDP}, CP={cfg.CP}) belongs in test_attention_block_tkg.py"
        )
        attn_blk_metadata_list = _get_attention_block_metadata()
        test_metadata_key = {
            "batch": cfg.batch,
            "q_heads": cfg.q_heads,
            "d_head": cfg.d_head,
            "H": cfg.H,
            "S_ctx": cfg.S_ctx,
            "S_tkg": cfg.S_tkg,
            "kv_quant": cfg.kv_quant,
            "KVDP": cfg.KVDP,
            "transposed_in": cfg.transposed_in,
            "DCP": cfg.CP,
        }
        collector.match_and_add_metadata_dimensions(test_metadata_key, attn_blk_metadata_list)
        _run_attention_block_test(
            test_manager=test_manager,
            platform_target=platform_target,
            cfg=cfg,
        )

    @pytest.mark.tier0
    @pytest.mark.parametrize(
        "cfg",
        _high_rank_cfgs(attention_block_tkg_model_configs.get(ModelTestType.TIER0, [])),
        ids=[c.test_id() for c in _high_rank_cfgs(attention_block_tkg_model_configs.get(ModelTestType.TIER0, []))],
    )
    def test_tier0(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        cfg: AttnBlkTestConfig,
    ):
        """TIER0: Critical model configs - highest priority for model validation."""
        self._run_model_test(test_manager, collector, platform_target, cfg)

    @pytest.mark.optimal
    @pytest.mark.parametrize(
        "cfg",
        _high_rank_cfgs(attention_block_tkg_model_configs.get(ModelTestType.OPTIMAL, [])),
        ids=[c.test_id() for c in _high_rank_cfgs(attention_block_tkg_model_configs.get(ModelTestType.OPTIMAL, []))],
    )
    def test_optimal(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        cfg: AttnBlkTestConfig,
    ):
        """OPTIMAL: Performance-optimized model configs."""
        self._run_model_test(test_manager, collector, platform_target, cfg)

    @pytest.mark.generality
    @pytest.mark.parametrize(
        "cfg",
        _high_rank_cfgs(attention_block_tkg_model_configs.get(ModelTestType.GENERALITY, [])),
        ids=[c.test_id() for c in _high_rank_cfgs(attention_block_tkg_model_configs.get(ModelTestType.GENERALITY, []))],
    )
    def test_generality(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        cfg: AttnBlkTestConfig,
    ):
        """GENERALITY: Broad coverage model configs."""
        self._run_model_test(test_manager, collector, platform_target, cfg)


KVDP_REPLICA_GROUP_CFGS = [cfg for cfg in RANGE_ATTN_BLK_CFGS if cfg.kvdp_replica_group is not None]


# Only Trn3 PDS supports collectives with this strided replica group.
# Trn2 and Trn3pd give a runtime error:
#   "failed to init a collective algorithm. reason: no_hier no_mesh"
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2, Platforms.TRN3, Platforms.TRN3_A0])
@pytest_marks(["attention", "tkg", "experimental"])
@final
@pytest.mark.high_rank
class TestKVDPReplicaGroup:
    """KVDP tests with explicit replica groups (requires trn3 PDS — trn2 CCOM cannot route these topologies)."""

    @pytest.mark.parametrize("attn_blk_cfg", KVDP_REPLICA_GROUP_CFGS, ids=lambda p: p.test_id())
    def test_kvdp_replica_group(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        attn_blk_cfg: AttnBlkTestConfig,
    ):
        _run_attention_block_test(
            test_manager=test_manager,
            platform_target=platform_target,
            cfg=attn_blk_cfg,
        )
