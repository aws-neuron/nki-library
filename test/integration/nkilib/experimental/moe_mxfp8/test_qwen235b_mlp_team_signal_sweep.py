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

"""Qwen3-235B expert MLP sweeps for the six requested Team Signal columns.

Each method sweeps explicit blocking candidates for one direction and worksheet
subcolumn. Run C/D/G/H with NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE unset. Run I/J
with it set to true in a separate process. The winning pytest node from each
sweep can be rerun alone to produce a publication OUTDIR.
"""

import functools
import os
from typing import final

import ml_dtypes
import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.experimental.mlp_mxfp8.mlp_bwd_mxfp8 import mlp_backward_mxfp8_nki
from nkilib_src.nkilib.experimental.mlp_mxfp8.mlp_fwd_mxfp8 import mlp_forward_mxfp8_nki
from nkilib_src.nkilib.experimental.moe_mxfp8.bwd.config import TransposeMode
from nkilib_src.nkilib.experimental.moe_mxfp8.config_helpers import ExecutionMode
from nkilib_src.nkilib.experimental.moe_mxfp8.fwd.config import MXFP8MOEFwdConfig
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.common_dataclasses import (
    QuantScheme,
    SwizzleMode,
    TensorDescriptor,
    TensorOrientation,
)

from test.integration.nkilib.experimental.mlp_mxfp8.qwen235b_mlp_tuned_configs import (
    build_bwd_candidate,
    build_fwd_candidate,
    bwd_config_as_dict,
    fwd_config_as_dict,
)
from test.integration.nkilib.experimental.moe_mxfp8.config_sweep_utils import (
    emit_cache_candidate,
    record_cache_candidate,
)
from test.integration.nkilib.experimental.moe_mxfp8.mxfp8_moe_bwd_test_utils import prequantize_moe_weights
from test.integration.nkilib.experimental.moe_mxfp8.mxfp8_moe_fwd_test_utils import prequantize_moe_weights_fwd
from test.integration.nkilib.experimental.moe_mxfp8.mxfp8_moe_validated_common import make_moe_validated_comparator
from test.integration.nkilib.experimental.moe_mxfp8.test_mlp_moe_based_mxfp8 import (
    _bwd_ref,
    _checkpoint,
    _fwd_ref,
    _inputs,
)
from test.utils import common_dataclasses
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

TOKENS, HIDDEN, INTERMEDIATE = 8192, 4096, 1536
BLOCK_SIZE = TOKENS // 2
NO_DMA_ENV = "NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE"
PRIMARY_SWIZZLE_FIELDS = (
    "output_grad_swizzle_mode",
    "down_weight_swizzle_mode",
    "d_gate_up_swizzle_mode",
    "gate_up_weight_swizzle_mode",
)
INTERNAL_SWIZZLE_FIELDS = (
    "d_gate_up_t_swizzle_mode",
    "hidden_states_t_swizzle_mode",
    "output_grad_t_swizzle_mode",
    "scaled_intermediate_t_swizzle_mode",
)

# Include the earlier measured winner for each cell and distinct nearby
# blocking/transpose alternatives. A winner is chosen from successful metrics,
# not inferred from these names.
FWD_CANDIDATES = {
    "C": ("baseline", "asymmetric", "full_k_gate", "gate_tn256"),
    "D": ("baseline", "asymmetric", "full_k_gate", "wide_m16"),
    "G": ("baseline", "wide_m8", "asymmetric", "full_k_gate", "gate_tn256"),
    "H": ("baseline", "asymmetric", "full_k_gate", "wide_m16"),
    "I": ("baseline", "wide_m8", "asymmetric", "full_k_gate", "gate_tn256"),
    "J": ("baseline", "asymmetric", "full_k_gate", "gate_tn256"),
}
BWD_CANDIDATES = {
    "C": ("baseline", "baseline_spill_reload", "phase1_alternate", "phase4_alternate"),
    "D": ("baseline", "baseline_spill_reload", "phase3_alternate", "phase4_alternate"),
    "G": ("baseline", "phase1_alternate", "phase3_alternate", "baseline_no_psum_copy_in"),
    "H": ("baseline", "phase1_alternate", "phase3_alternate", "baseline_no_psum_copy_in"),
    "I": ("baseline", "phase1_alternate", "phase3_alternate", "baseline_no_psum_copy_in"),
    "J": ("baseline", "phase1_alternate", "phase3_alternate", "baseline_no_psum_copy_in"),
}


def _env_guard(column: str) -> None:
    force_pe = os.environ.get(NO_DMA_ENV, "").lower() == "true"
    assert force_pe == (column in "IJ"), f"{column} requires {NO_DMA_ENV}={'true' if column in 'IJ' else 'unset'}"


def _record(test_manager: Orchestrator, direction: str, column: str, config_id: str, config) -> None:
    test_manager.collector.set_kernel_params(
        {
            "model": "qwen3_235b",
            "kernel": "mlp_mxfp8",
            "direction": direction,
            "table_column": column,
            "tokens": TOKENS,
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "block_size": BLOCK_SIZE,
            "num_blocks": 2,
            "quant_scheme": "wrapX" if column in "CD" else "1x32",
            "weight_input": "bf16" if column in "CGI" else "prequantized",
            "no_dma_transpose": column in "IJ",
            "operand_loader": "dma_transpose" if column in "CD" else "pe",
            "config_id": config_id,
            "config": fwd_config_as_dict(config) if direction == "forward" else bwd_config_as_dict(config),
        }
    )


def _fwd(
    test_manager: Orchestrator,
    platform_target,
    column: str,
    config_id: str,
    weight_orientation=TensorOrientation.F_BY_K,
) -> None:
    _env_guard(column)
    bf16_weights = column in "CGI"
    wrapx = column in "CD"
    scheme = QuantScheme.WRAPX if wrapx else QuantScheme._1x32
    tuned = build_fwd_candidate(config_id, scheme, pe_loads=not wrapx)
    config = MXFP8MOEFwdConfig(
        gate_up_config=tuned.gate_up_config,
        down_config=tuned.down_config,
        execution_mode=ExecutionMode.MLP,
        dma_transpose_enabled=column not in "IJ",
        use_1x32_pe_swizzle=not wrapx,
        reuse_spilled_weights=True,
        gate_up_weight_td=TensorDescriptor(
            orientation=weight_orientation if bf16_weights else TensorOrientation.K_BY_F,
            quant_scheme=scheme,
            swizzle_mode=SwizzleMode.DGT if wrapx else SwizzleMode.PE,
            disable_dma_transpose=column in "IJ",
        ),
        down_weight_td=TensorDescriptor(
            orientation=weight_orientation if bf16_weights else TensorOrientation.K_BY_F,
            quant_scheme=scheme,
            swizzle_mode=SwizzleMode.DGT if wrapx else SwizzleMode.PE,
            disable_dma_transpose=column in "IJ",
        ),
    )
    x, gate_up, down, _ = _inputs(TOKENS, HIDDEN, INTERMEDIATE)
    inputs = {"hidden_states": x, "config": config}
    if bf16_weights:
        inputs["gate_up_proj_weight"] = (
            np.transpose(gate_up, (2, 1, 0)).copy() if weight_orientation == TensorOrientation.F_BY_K else gate_up
        )
        inputs["down_proj_weight"] = down.T.copy() if weight_orientation == TensorOrientation.F_BY_K else down
    else:
        packed_gate, gate_scales, packed_down, down_scales = prequantize_moe_weights_fwd(
            gate_up[None], down[None], use_1x32=not wrapx
        )
        inputs.update(
            gate_up_proj_weight=packed_gate[0],
            down_proj_weight=packed_down[0],
            gate_up_weight_scales=gate_scales[0],
            down_weight_scales=down_scales[0],
        )

    @functools.wraps(_fwd_ref)
    def ref(**kwargs):
        kwargs["gate_up_proj_weight"] = torch.from_numpy(gate_up.astype(np.float32))
        kwargs["down_proj_weight"] = torch.from_numpy(down.astype(np.float32))
        return _fwd_ref(**kwargs)

    _record(test_manager, "forward", column, config_id, config)
    record_cache_candidate(test_manager, inputs)
    shapes = {
        "output_hidden_states": ((TOKENS, HIDDEN), nl.bfloat16),
        "gate_up_proj_act_checkpoint_T": ((TOKENS // BLOCK_SIZE, 2, BLOCK_SIZE, INTERMEDIATE), nl.bfloat16),
    }
    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=mlp_forward_mxfp8_nki,
        torch_ref=torch_ref_wrapper(ref),
        kernel_input_generator=lambda _: inputs,
        output_tensor_descriptor=lambda _: {
            name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in shapes.items()
        },
    )
    framework.run_test(
        test_config=None,
        compiler_args=common_dataclasses.CompilerArgs(logical_nc_config=2, platform_target=platform_target),
        custom_comparator=make_moe_validated_comparator(shapes),
    )
    emit_cache_candidate(test_manager)


def _bwd(test_manager: Orchestrator, platform_target, column: str, config_id: str) -> None:
    _env_guard(column)
    bf16_weights = column in "CGI"
    wrapx = column in "CD"
    scheme = QuantScheme.WRAPX if wrapx else QuantScheme._1x32
    use_scale_packing = not wrapx
    config = build_bwd_candidate(
        config_id,
        scheme,
        pe_loads=not wrapx,
        use_scale_packing=use_scale_packing,
    )
    config.execution_mode = ExecutionMode.MLP
    config.dma_transpose_enabled = column not in "IJ"
    config.fast_dma_transpose = config.dma_transpose_enabled
    config.pe_transpose_only = column in "IJ"
    for field in PRIMARY_SWIZZLE_FIELDS:
        setattr(config, field, SwizzleMode.DGT if wrapx else SwizzleMode.PE)
    for field in INTERNAL_SWIZZLE_FIELDS:
        setattr(config, field, SwizzleMode.PE)
    config.phase3_transpose_mode = TransposeMode.NC
    config.phase4_transpose_mode = TransposeMode.NC
    x, gate_up, down, dy = _inputs(TOKENS, HIDDEN, INTERMEDIATE)
    checkpoint = (
        _checkpoint(
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(gate_up.astype(np.float32)),
            BLOCK_SIZE,
        )
        .numpy()
        .astype(ml_dtypes.bfloat16)
    )
    inputs = {
        "hidden_states": x,
        "output_hidden_states_grad": dy,
        "gate_up_proj_act_checkpoint_T": checkpoint,
        "config": config,
    }
    if bf16_weights:
        inputs.update(gate_up_proj_weight=gate_up, down_proj_weight=down)
    else:
        packed_gate, gate_scales, packed_down, down_scales = prequantize_moe_weights(
            gate_up[None],
            down[None],
            quant_scheme=scheme,
        )
        inputs.update(
            gate_up_proj_weight=packed_gate[0],
            down_proj_weight=packed_down[0],
            gate_up_weight_scales=gate_scales[0],
            down_weight_scales=down_scales[0],
        )

    @functools.wraps(_bwd_ref)
    def ref(**kwargs):
        kwargs["gate_up_proj_weight"] = torch.from_numpy(gate_up.astype(np.float32))
        kwargs["down_proj_weight"] = torch.from_numpy(down.astype(np.float32))
        return _bwd_ref(**kwargs)

    _record(test_manager, "backward", column, config_id, config)
    record_cache_candidate(test_manager, inputs, backward=True)
    shapes = {
        "hidden_states_grad": ((TOKENS, HIDDEN), nl.bfloat16),
        "gate_up_proj_weight_grad": ((HIDDEN, 2, INTERMEDIATE), nl.bfloat16),
        "down_proj_weight_grad": ((INTERMEDIATE, HIDDEN), nl.bfloat16),
    }
    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=mlp_backward_mxfp8_nki,
        torch_ref=torch_ref_wrapper(ref),
        kernel_input_generator=lambda _: inputs,
        output_tensor_descriptor=lambda _: {
            name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in shapes.items()
        },
    )
    framework.run_test(
        test_config=None,
        compiler_args=common_dataclasses.CompilerArgs(logical_nc_config=2, platform_target=platform_target),
        custom_comparator=make_moe_validated_comparator(shapes),
    )
    emit_cache_candidate(test_manager)


@pytest_test_metadata(name="Qwen3-235B MXFP8 MLP Team Signal Sweep")
@pytest_marks(["mlp_mxfp8", "mxfp8", "model", "team_signal", "tuning"])
@pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN1, common_dataclasses.Platforms.TRN2])
@final
class TestQwen235BMlpTeamSignalSweep:
    @pytest.mark.parametrize("config_id", ["baseline", "wide_m8", "asymmetric", "full_k_gate"])
    def test_fwd_1x32_shared_bf16_inputs(self, test_manager: Orchestrator, platform_target, monkeypatch, config_id):
        from nkilib_src.nkilib.experimental.moe_mxfp8 import _best_config_cache

        # Each worker sweeps with isolated empty tables; the committed cache is preserved.
        monkeypatch.setattr(_best_config_cache, "_MOE_FWD_CONFIG_CACHE", {})
        monkeypatch.setattr(_best_config_cache, "_MOE_BWD_CONFIG_CACHE", {})
        column = "I" if os.environ.get(NO_DMA_ENV, "").lower() == "true" else "G"
        _fwd(test_manager, platform_target, column, config_id, TensorOrientation.K_BY_F)

    @pytest.mark.parametrize("config_id", FWD_CANDIDATES["C"])
    def test_fwd_wrapx_dma_bf16_inputs(self, test_manager: Orchestrator, platform_target, config_id):
        _fwd(test_manager, platform_target, "C", config_id)

    @pytest.mark.parametrize("config_id", FWD_CANDIDATES["D"])
    def test_fwd_wrapx_dma_prequant_weight(self, test_manager: Orchestrator, platform_target, config_id):
        _fwd(test_manager, platform_target, "D", config_id)

    @pytest.mark.parametrize("config_id", FWD_CANDIDATES["G"])
    def test_fwd_1x32_dma_bf16_inputs(self, test_manager: Orchestrator, platform_target, config_id):
        _fwd(test_manager, platform_target, "G", config_id)

    @pytest.mark.parametrize("config_id", FWD_CANDIDATES["H"])
    def test_fwd_1x32_dma_prequant_weight(self, test_manager: Orchestrator, platform_target, config_id):
        _fwd(test_manager, platform_target, "H", config_id)

    @pytest.mark.parametrize("config_id", FWD_CANDIDATES["I"])
    def test_fwd_1x32_no_dma_transpose_bf16_inputs(self, test_manager: Orchestrator, platform_target, config_id):
        _fwd(test_manager, platform_target, "I", config_id)

    @pytest.mark.parametrize("config_id", FWD_CANDIDATES["J"])
    def test_fwd_1x32_no_dma_transpose_prequant_weight(self, test_manager: Orchestrator, platform_target, config_id):
        _fwd(test_manager, platform_target, "J", config_id)

    @pytest.mark.parametrize("config_id", BWD_CANDIDATES["C"])
    def test_bwd_wrapx_dma_bf16_inputs(self, test_manager: Orchestrator, platform_target, config_id):
        _bwd(test_manager, platform_target, "C", config_id)

    @pytest.mark.parametrize("config_id", BWD_CANDIDATES["D"])
    def test_bwd_wrapx_dma_prequant_weight(self, test_manager: Orchestrator, platform_target, config_id):
        _bwd(test_manager, platform_target, "D", config_id)

    @pytest.mark.parametrize("config_id", BWD_CANDIDATES["G"])
    def test_bwd_1x32_dma_bf16_inputs(self, test_manager: Orchestrator, platform_target, config_id):
        _bwd(test_manager, platform_target, "G", config_id)

    @pytest.mark.parametrize("config_id", BWD_CANDIDATES["H"])
    def test_bwd_1x32_dma_prequant_weight(self, test_manager: Orchestrator, platform_target, config_id):
        _bwd(test_manager, platform_target, "H", config_id)

    @pytest.mark.parametrize("config_id", BWD_CANDIDATES["I"])
    def test_bwd_1x32_no_dma_transpose_bf16_inputs(self, test_manager: Orchestrator, platform_target, config_id):
        _bwd(test_manager, platform_target, "I", config_id)

    @pytest.mark.parametrize("config_id", BWD_CANDIDATES["J"])
    def test_bwd_1x32_no_dma_transpose_prequant_weight(self, test_manager: Orchestrator, platform_target, config_id):
        _bwd(test_manager, platform_target, "J", config_id)
