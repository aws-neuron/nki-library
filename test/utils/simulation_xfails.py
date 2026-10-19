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
"""Fast tests that fail under CPU simulation only (simulator numerical mismatch).

Listed by pytest nodeid. ``apply_xfail_simulation`` (in simulation_setup) marks each
as a non-strict ``xfail`` when the session runs in simulation mode, so they run in the
simulator but are not counted as failures; an XPASS signals the simulator now matches
and the entry can be removed. No effect in any other mode -- hardware runs are
unaffected. Centralised here (rather than as inline ``@pytest.mark.xfail_simulation``
marks) because many of these are generated / coverage-parametrized cases that cannot
carry an inline marker on a single failing parametrization.

Nodeids are relative to the ``test/`` rootdir (they begin ``integration/...``), matching
``item.nodeid`` as pytest collects it.
"""

SIMULATION_XFAIL_REASON: str = "simulator numerical mismatch"

SIMULATION_XFAIL_NODEIDS: frozenset[str] = frozenset(
    {
        "integration/nkilib/core/moe/moe_cte/test_moe_bwmm_mx_cte.py::TestMoeBwmmMxShardBlockKernel::test_moe_bwmm_mx_shard_block_prequant_hidden[trn3_a0-vnc-2_hid-512_tok-1024_inter-384_exp-16_blk-256_k-4_act-3_eas_mode-1_dt-bfloat16_wdt-float4_e2m1fn_x4_sk-1_bi-1_dyn-1_gcu-7.0_gcl-None_ucu-7.0_ucl--7.0_pack-1_paff-1_adt-bfloat16_ep-1_eprk-0]",
        "integration/nkilib/core/moe/moe_cte/test_moe_bwmm_mx_cte.py::TestMoeBwmmMxShardBlockKernel::test_moe_bwmm_mx_shard_block_prequant_hidden[trn3_a0-vnc-2_hid-512_tok-1024_inter-384_exp-16_blk-256_k-4_act-3_eas_mode-1_dt-bfloat16_wdt-float4_e2m1fn_x4_sk-3_bi-1_dyn-1_gcu-7.0_gcl-None_ucu-7.0_ucl--7.0_pack-1_paff-1_adt-bfloat16_ep-1_eprk-0]",
        "integration/nkilib/core/moe/moe_cte/test_moe_bwmm_mx_cte.py::TestMoeBwmmMxShardBlockKernel::test_moe_bwmm_mx_shard_block_prequant_hidden[trn3_a0-vnc-2_hid-512_tok-1024_inter-384_exp-16_blk-256_k-4_act-3_eas_mode-1_dt-bfloat16_wdt-float4_e2m1fn_x4_sk-3_bi-1_dyn-1_gcu-7.0_gcl-None_ucu-7.0_ucl--7.0_pack-1_paff-1_adt-float32_ep-1_eprk-0]",
        "integration/nkilib/core/moe/moe_cte/test_moe_bwmm_mx_cte.py::TestMoeBwmmMxShardBlockKernel::test_moe_bwmm_mx_shard_block_prequant_hidden[trn3_a0-vnc-2_hid-512_tok-300_inter-384_exp-16_blk-256_k-4_act-3_eas_mode-1_dt-bfloat16_wdt-float4_e2m1fn_x4_sk-3_bi-1_dyn-1_gcu-7.0_gcl-None_ucu-7.0_ucl--7.0_pack-1_paff-1_adt-bfloat16_ep-1_eprk-0]",
        "integration/nkilib/core/moe/moe_cte/test_moe_cte_unified.py::TestMoeCTEUnified::test_moe_cte_unified[trn3_a0-MoECTEImplementation.shard_on_block_mx-3072-1024-8-256-4-384-bfloat16-1-False-False-None-ActFnType.SiLU-ExpertAffinityScaleMode.POST_SCALE-None-None-None-None-False-float8_e4m3fn_x4-True-False-False-8-QuantizationType.STATIC_MX]",
        "integration/nkilib/core/moe/moe_cte/test_moe_cte_unified.py::TestMoeCTEUnified::test_moe_cte_unified[trn3_a0-MoECTEImplementation.shard_on_block_mx-3072-1024-8-256-4-384-bfloat16-1-False-False-None-ActFnType.Swish-ExpertAffinityScaleMode.POST_SCALE-7.0-None-7.0--7.0-False-float4_e2m1fn_x4-False-False-True-1-QuantizationType.MX]",
        "integration/nkilib/core/moe_block/test_moe_block_tkg.py::TestMoEBlockTkgKernel::test_moe_block_static_mx[trn3_a0-static_mx_10]",
        "integration/nkilib/experimental/matmul_mxfp8/test_matmul_mxfp8_generic_kernel.py::TestMatmulMxfp8GenericKernel::test_matmul_mxfp8_base_grid[trn3_a0-conf1]",
        "integration/nkilib/experimental/matmul_mxfp8/test_quantize_matmul_pipeline.py::TestQuantizeMatmulPipeline::test_lhs_prequantized_rhs_bf16[trn3_a0-False-unpacked-1024-3072-3072]",
        "integration/nkilib/experimental/matmul_mxfp8/test_quantize_matmul_pipeline.py::TestQuantizeMatmulPipeline::test_lhs_prequantized_rhs_bf16[trn3_a0-True-unpacked-1024-3072-3072]",
    }
)
