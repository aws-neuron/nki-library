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

"""Numerical tests for the static MLP path shared with MXFP8 MoE."""

import functools
from collections.abc import Callable
from typing import cast, final

import ml_dtypes
import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.core.utils.kernel_assert import kernel_assert
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_config import MatmulMxfp8KernelConfig
from nkilib_src.nkilib.experimental.mlp_mxfp8.mlp_bwd_mxfp8 import mlp_backward_mxfp8_nki
from nkilib_src.nkilib.experimental.mlp_mxfp8.mlp_fwd_mxfp8 import mlp_forward_mxfp8_nki
from nkilib_src.nkilib.experimental.moe_mxfp8 import _best_config_cache
from nkilib_src.nkilib.experimental.moe_mxfp8._config_helpers import (
    MoEInputShape,
    copy_config,
    make_moe_config_key,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.bwd import blockwise_mm_backward_mxfp8 as bwd_entry
from nkilib_src.nkilib.experimental.moe_mxfp8.bwd.config import MXFP8MOEBwdConfig, TransposeMode
from nkilib_src.nkilib.experimental.moe_mxfp8.config_helpers import ExecutionMode
from nkilib_src.nkilib.experimental.moe_mxfp8.fwd import blockwise_mm_forward_mxfp8 as fwd_entry
from nkilib_src.nkilib.experimental.moe_mxfp8.fwd.config import MXFP8MOEFwdConfig, _auto_generate_moe_fwd_configs
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.common_dataclasses import (
    QuantScheme,
    SwizzleMode,
    TensorDescriptor,
    TensorOrientation,
)

from test.integration.nkilib.experimental.mlp_mxfp8.mlp_perf_test_utils import _bwd_ref, _checkpoint, _fwd_ref, _inputs
from test.integration.nkilib.experimental.moe_mxfp8.config_sweep_utils import pack_cache_tables
from test.integration.nkilib.experimental.moe_mxfp8.mxfp8_moe_bwd_test_utils import prequantize_moe_weights
from test.integration.nkilib.experimental.moe_mxfp8.mxfp8_moe_fwd_test_utils import prequantize_moe_weights_fwd
from test.integration.nkilib.experimental.moe_mxfp8.mxfp8_moe_validated_common import (
    make_moe_validated_comparator,
)
from test.utils import common_dataclasses
from test.utils.coverage_parametrized_tests import FilterResult
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

_HIDDEN = 512
_INTERMEDIATE = 256
_TOKEN_COUNTS = [512, 1024, 1536, 2048]


def _install_test_cache(monkeypatch, config, inputs, backward, cache_case):
    """Exercise the reader inside compilation, including complete-hit rejection."""
    module = bwd_entry if backward else fwd_entry
    reader_name = "get_cached_moe_bwd_config" if backward else "get_cached_moe_fwd_config"
    cache_name = "_MOE_BWD_CONFIG_CACHE" if backward else "_MOE_FWD_CONFIG_CACHE"
    table = {}
    monkeypatch.setattr(_best_config_cache, cache_name, table)
    block_size = inputs["hidden_states"].shape[0] // 2
    shape = MoEInputShape(
        inputs["hidden_states"].shape,
        inputs["gate_up_proj_weight"].shape,
        inputs["down_proj_weight"].shape,
        block_size,
        weight_scale_shapes=(
            (inputs["gate_up_weight_scales"].shape, inputs["down_weight_scales"].shape)
            if "gate_up_weight_scales" in inputs
            else None
        ),
    )
    weight_dtype = nl.float8_e4m3fn_x4 if "gate_up_weight_scales" in inputs else nl.bfloat16
    key = make_moe_config_key(shape, weight_dtype, config, backward)
    if cache_case != "miss":
        cached = copy_config(config)
        if backward:
            bwd_entry._resolve_phase_configs(
                cached,
                (
                    (block_size, _HIDDEN, _INTERMEDIATE // 2),
                    (block_size, 2 * _INTERMEDIATE, _HIDDEN // 2),
                    (_HIDDEN // 2, block_size, 2 * _INTERMEDIATE),
                    (_INTERMEDIATE, block_size, _HIDDEN // 2),
                ),
                _HIDDEN,
                True,
            )
        else:
            _auto_generate_moe_fwd_configs(cached, block_size, _HIDDEN, _INTERMEDIATE)
        if cache_case == "conflict":
            cached.bias = True
            phase = cached.phase1_config if backward else cached.gate_up_config
            phase.TILES_IN_BLOCK_M = 999
        direction = "backward" if backward else "forward"
        tables = {"forward": {}, "backward": {}}
        tables[direction][key] = _best_config_cache.CachedConfig(cached, 0.0001)
        schemas, records, _, packed = pack_cache_tables(tables)
        monkeypatch.setattr(_best_config_cache, "_CONFIG_SCHEMAS", schemas)
        monkeypatch.setattr(_best_config_cache, "_CONFIG_RECORDS", records)
        table.update(packed[direction])
    reader = getattr(module, reader_name)
    expected_hit = cache_case == "hit"

    def checked_reader(key, config):
        result = reader(key, config)
        kernel_assert((result != None) == expected_hit, "Unexpected compiled config-cache lookup result")  # noqa: E711
        return result

    monkeypatch.setattr(module, reader_name, checked_reader)
    impl_name = "blockwise_mm_bwd_dropless_mxfp8" if backward else "blockwise_mm_fwd_dropless_mxfp8"
    impl = getattr(module, impl_name)

    def checked_impl(**kwargs):
        kernel_assert(kwargs["N"] == 2, "MLP must compute exactly two blocks")
        kernel_assert(
            kwargs["block_size"] == kwargs["T"] // 2,
            "MLP block size must be half the sequence length",
        )
        return impl(**kwargs)

    monkeypatch.setattr(module, impl_name, checked_impl)


_FWD_ABBREVS = {
    "tokens": "t",
    "weight_input": "w",
    "load_path": "lp",
    "weight_orientation": "wo",
    "spill_reload": "sr",
    "scale_packing": "sp",
    "reuse_spilled_weights": "rw",
}
_BWD_ABBREVS = {
    "tokens": "t",
    "weight_input": "w",
    "layout_path": "lp",
    "transpose_path": "tp",
    "spill_reload": "sr",
    "scale_packing": "sp",
}

_BWD_PRIMARY_SWIZZLE_FIELDS = (
    "output_grad_swizzle_mode",
    "down_weight_swizzle_mode",
    "d_gate_up_swizzle_mode",
    "gate_up_weight_swizzle_mode",
)
_BWD_INTERNAL_SWIZZLE_FIELDS = (
    "d_gate_up_t_swizzle_mode",
    "hidden_states_t_swizzle_mode",
    "output_grad_t_swizzle_mode",
    "scaled_intermediate_t_swizzle_mode",
)


def _phase_config(quant_scheme, spill_reload, scale_packing):
    return MatmulMxfp8KernelConfig(
        M=0,
        K=0,
        N=0,
        quant_scheme=quant_scheme,
        spill_reload=spill_reload,
        enable_scale_packing=scale_packing,
    )


def _fwd_filter(
    tokens,
    weight_input,
    load_path,
    weight_orientation,
    spill_reload,
    scale_packing,
    reuse_spilled_weights,
):
    del tokens, scale_packing
    if weight_input == "prequantized":
        if weight_orientation != TensorOrientation.K_BY_F:
            return FilterResult.REDUNDANT
        if load_path in ("wrapx_pe", "1x32_pe_no_dma"):
            return FilterResult.REDUNDANT
    elif load_path in ("wrapx_dgt", "wrapx_fast_dgt") and weight_orientation != TensorOrientation.F_BY_K:
        return FilterResult.REDUNDANT

    if load_path == "1x32_pe_no_dma" and weight_orientation != TensorOrientation.K_BY_F:
        return FilterResult.REDUNDANT
    if reuse_spilled_weights and not spill_reload:
        return FilterResult.REDUNDANT
    return FilterResult.VALID


def _make_fwd_config(load_path, weight_orientation, spill_reload, scale_packing, reuse_spilled_weights):
    is_1x32 = load_path.startswith("1x32")
    quant_scheme = QuantScheme._1x32 if is_1x32 else QuantScheme.WRAPX
    swizzle_mode = SwizzleMode.PE if load_path in ("wrapx_pe", "1x32_pe", "1x32_pe_no_dma") else SwizzleMode.DGT

    def weight_td():
        return TensorDescriptor(
            orientation=weight_orientation,
            quant_scheme=quant_scheme,
            swizzle_mode=swizzle_mode,
            disable_dma_transpose=load_path == "1x32_pe_no_dma",
        )

    return MXFP8MOEFwdConfig(
        gate_up_config=_phase_config(quant_scheme, spill_reload, scale_packing),
        down_config=_phase_config(quant_scheme, spill_reload, scale_packing),
        execution_mode=ExecutionMode.MLP,
        dma_transpose_enabled=load_path != "1x32_pe_no_dma",
        fast_dma_transpose=load_path == "wrapx_fast_dgt",
        use_1x32_pe_swizzle=is_1x32,
        reuse_spilled_weights=reuse_spilled_weights,
        gate_up_weight_td=weight_td(),
        down_weight_td=weight_td(),
    )


def _bwd_filter(
    tokens,
    weight_input,
    layout_path,
    transpose_path,
    spill_reload,
    scale_packing,
):
    del tokens, weight_input, spill_reload, scale_packing
    if layout_path.endswith("pe_only") and transpose_path != "nc":
        return FilterResult.REDUNDANT
    return FilterResult.VALID


def _bwd_quant_scheme(layout_path):
    return QuantScheme._1x32 if layout_path.startswith("1x32") else QuantScheme.WRAPX


def _make_bwd_config(layout_path, transpose_path, spill_reload, scale_packing):
    quant_scheme = _bwd_quant_scheme(layout_path)
    operand_path = layout_path.removeprefix("wrapx_").removeprefix("1x32_")
    phase_configs = [_phase_config(quant_scheme, spill_reload, scale_packing) for _ in range(4)]
    transpose_modes = {
        "nc": (TransposeMode.NC, TransposeMode.NC),
        "phase3_dma": (TransposeMode.DMA, TransposeMode.NC),
        "phase4_dma": (TransposeMode.NC, TransposeMode.DMA),
        "both_dma": (TransposeMode.DMA, TransposeMode.DMA),
    }
    phase3_transpose_mode, phase4_transpose_mode = transpose_modes[transpose_path]
    config = MXFP8MOEBwdConfig(
        phase1_config=phase_configs[0],
        phase2_config=phase_configs[1],
        phase3_config=phase_configs[2],
        phase4_config=phase_configs[3],
        execution_mode=ExecutionMode.MLP,
        fast_dma_transpose=operand_path == "fast_dgt",
        pe_transpose_only=operand_path == "pe_only",
        phase3_transpose_mode=phase3_transpose_mode,
        phase4_transpose_mode=phase4_transpose_mode,
    )
    primary_mode = SwizzleMode.PE if operand_path in ("pe", "pe_only") else SwizzleMode.DGT
    internal_mode = SwizzleMode.PE if operand_path in ("mixed", "pe", "pe_only") else SwizzleMode.DGT
    for field in _BWD_PRIMARY_SWIZZLE_FIELDS:
        setattr(config, field, primary_mode)
    for field in _BWD_INTERNAL_SWIZZLE_FIELDS:
        setattr(config, field, internal_mode)
    return config


@pytest_test_metadata(name="MLP MXFP8 MoE based")
@pytest_marks(["mlp_mxfp8", "mxfp8", "validated"])
@pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN1, common_dataclasses.Platforms.TRN2])
@final
class TestMlpMoeBasedMxfp8:
    """Exercise the public MLP entry points across the supported dense MoE paths."""

    def _run_forward(
        self,
        test_manager,
        platform_target,
        tokens,
        weight_input,
        load_path,
        weight_orientation,
        spill_reload,
        scale_packing,
        reuse_spilled_weights,
        cache_case="miss",
        monkeypatch=None,
    ):
        block_size = tokens // 2
        x, gate_up, down, _ = _inputs(tokens, _HIDDEN, _INTERMEDIATE)
        config = _make_fwd_config(
            load_path,
            weight_orientation,
            spill_reload,
            scale_packing,
            reuse_spilled_weights,
        )
        inputs = {
            "hidden_states": x,
            "config": config,
        }
        if weight_input == "prequantized":
            is_1x32 = load_path.startswith("1x32")
            packed_gate, gate_scales, packed_down, down_scales = prequantize_moe_weights_fwd(
                gate_up[None],
                down[None],
                use_1x32=is_1x32,
            )
            inputs.update(
                gate_up_proj_weight=packed_gate[0],
                down_proj_weight=packed_down[0],
                gate_up_weight_scales=gate_scales[0],
                down_weight_scales=down_scales[0],
            )
        elif weight_orientation == TensorOrientation.F_BY_K:
            inputs.update(
                gate_up_proj_weight=np.transpose(gate_up, (2, 1, 0)).copy(),
                down_proj_weight=down.T.copy(),
            )
        else:
            inputs.update(gate_up_proj_weight=gate_up, down_proj_weight=down)

        if monkeypatch is not None:
            _install_test_cache(monkeypatch, config, inputs, False, cache_case)

        @functools.wraps(_fwd_ref)
        def ref(**kwargs):
            kwargs["gate_up_proj_weight"] = torch.from_numpy(gate_up.astype(np.float32))
            kwargs["down_proj_weight"] = torch.from_numpy(down.astype(np.float32))
            return _fwd_ref(**kwargs)

        shapes = {
            "output_hidden_states": ((tokens, _HIDDEN), nl.bfloat16),
            "gate_up_proj_act_checkpoint_T": (
                (2, 2, block_size, _INTERMEDIATE),
                nl.bfloat16,
            ),
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

    def _run_backward(
        self,
        test_manager,
        platform_target,
        tokens,
        weight_input,
        layout_path,
        transpose_path,
        spill_reload,
        scale_packing,
        cache_case="miss",
        monkeypatch=None,
    ):
        block_size = tokens // 2
        x, gate_up, down, output_grad = _inputs(tokens, _HIDDEN, _INTERMEDIATE)
        checkpoint = (
            _checkpoint(
                torch.from_numpy(x.astype(np.float32)),
                torch.from_numpy(gate_up.astype(np.float32)),
                block_size,
            )
            .numpy()
            .astype(ml_dtypes.bfloat16)
        )
        inputs = {
            "hidden_states": x,
            "output_hidden_states_grad": output_grad,
            "gate_up_proj_act_checkpoint_T": checkpoint,
            "config": _make_bwd_config(
                layout_path,
                transpose_path,
                spill_reload,
                scale_packing,
            ),
        }
        if weight_input == "prequantized":
            packed_gate, gate_scales, packed_down, down_scales = prequantize_moe_weights(
                gate_up[None],
                down[None],
                quant_scheme=_bwd_quant_scheme(layout_path),
            )
            inputs.update(
                gate_up_proj_weight=packed_gate[0],
                down_proj_weight=packed_down[0],
                gate_up_weight_scales=gate_scales[0],
                down_weight_scales=down_scales[0],
            )
        else:
            inputs.update(gate_up_proj_weight=gate_up, down_proj_weight=down)

        if monkeypatch is not None:
            _install_test_cache(monkeypatch, inputs["config"], inputs, True, cache_case)

        @functools.wraps(_bwd_ref)
        def ref(**kwargs):
            kwargs["gate_up_proj_weight"] = torch.from_numpy(gate_up.astype(np.float32))
            kwargs["down_proj_weight"] = torch.from_numpy(down.astype(np.float32))
            return _bwd_ref(**kwargs)

        shapes = {
            "hidden_states_grad": ((tokens, _HIDDEN), nl.bfloat16),
            "gate_up_proj_weight_grad": ((_HIDDEN, 2, _INTERMEDIATE), nl.bfloat16),
            "down_proj_weight_grad": ((_INTERMEDIATE, _HIDDEN), nl.bfloat16),
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

    @pytest.mark.fast
    @pytest.mark.parametrize("cache_case", ["miss", "hit", "conflict"])
    @pytest.mark.parametrize("tokens", _TOKEN_COUNTS)
    def test_forward_fast(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        monkeypatch,
        cache_case,
        tokens,
    ):
        self._run_forward(
            test_manager,
            platform_target,
            tokens=tokens,
            weight_input="bf16",
            load_path="1x32_pe",
            weight_orientation=TensorOrientation.K_BY_F,
            spill_reload=True,
            scale_packing=True,
            reuse_spilled_weights=False,
            cache_case=cache_case,
            monkeypatch=monkeypatch,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("cache_case", ["miss", "hit", "conflict"])
    @pytest.mark.parametrize("tokens", _TOKEN_COUNTS)
    def test_backward_fast(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        monkeypatch,
        cache_case,
        tokens,
    ):
        self._run_backward(
            test_manager,
            platform_target,
            tokens=tokens,
            weight_input="bf16",
            layout_path="1x32_pe_only",
            transpose_path="nc",
            spill_reload=False,
            scale_packing=True,
            cache_case=cache_case,
            monkeypatch=monkeypatch,
        )

    @pytest.mark.coverage_parametrize(
        tokens=_TOKEN_COUNTS,
        weight_input=["bf16", "prequantized"],
        load_path=["wrapx_dgt", "wrapx_fast_dgt", "wrapx_pe", "1x32_pe", "1x32_pe_no_dma"],
        weight_orientation=[TensorOrientation.K_BY_F, TensorOrientation.F_BY_K],
        spill_reload=[False, True],
        scale_packing=[False, True],
        reuse_spilled_weights=[False, True],
        filter=_fwd_filter,
        coverage="pairs",
        abbrev=_FWD_ABBREVS,
        enable_automatic_boundary_tests=False,
        enable_invalid_combination_tests=False,
    )
    def test_forward_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        tokens,
        weight_input,
        load_path,
        weight_orientation,
        spill_reload,
        scale_packing,
        reuse_spilled_weights,
        is_negative_test_case,
    ):
        assert not is_negative_test_case
        self._run_forward(
            test_manager,
            platform_target,
            tokens,
            weight_input,
            load_path,
            weight_orientation,
            spill_reload,
            scale_packing,
            reuse_spilled_weights,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("intermediate", [3200, 6400])
    def test_backward_postprocess_shard_tail(self, test_manager, platform_target, intermediate):
        """Cover a padded 768-column matmul tile crossing a 2048-column chunk."""
        tokens, hidden = 512, 512
        block_size = tokens // 2
        x, gate_up, down, output_grad = _inputs(tokens, hidden, intermediate)
        config = _make_bwd_config("1x32_pe_only", "nc", False, False)
        config.phase1_config.tile_n = 768
        config.phase1_config.TILES_IN_BLOCK_N = 3
        checkpoint = (
            _checkpoint(
                torch.from_numpy(x.astype(np.float32)),
                torch.from_numpy(gate_up.astype(np.float32)),
                block_size,
            )
            .numpy()
            .astype(ml_dtypes.bfloat16)
        )
        inputs = {
            "hidden_states": x,
            "gate_up_proj_weight": gate_up,
            "down_proj_weight": down,
            "output_hidden_states_grad": output_grad,
            "gate_up_proj_act_checkpoint_T": checkpoint,
            "config": config,
        }
        shapes = {
            "hidden_states_grad": ((tokens, hidden), nl.bfloat16),
            "gate_up_proj_weight_grad": ((hidden, 2, intermediate), nl.bfloat16),
            "down_proj_weight_grad": ((intermediate, hidden), nl.bfloat16),
        }
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=mlp_backward_mxfp8_nki,
            torch_ref=torch_ref_wrapper(_bwd_ref),
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

    @pytest.mark.coverage_parametrize(
        tokens=_TOKEN_COUNTS,
        weight_input=["bf16", "prequantized"],
        layout_path=[
            "wrapx_dgt",
            "wrapx_fast_dgt",
            "wrapx_mixed",
            "wrapx_pe",
            "1x32_pe",
            "1x32_pe_only",
        ],
        transpose_path=["nc", "phase3_dma", "phase4_dma", "both_dma"],
        spill_reload=[False, True],
        scale_packing=[False, True],
        filter=_bwd_filter,
        coverage="pairs",
        abbrev=_BWD_ABBREVS,
        enable_automatic_boundary_tests=False,
        enable_invalid_combination_tests=False,
    )
    def test_backward_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: common_dataclasses.Platforms,
        tokens,
        weight_input,
        layout_path,
        transpose_path,
        spill_reload,
        scale_packing,
        is_negative_test_case,
    ):
        assert not is_negative_test_case
        self._run_backward(
            test_manager,
            platform_target,
            tokens,
            weight_input,
            layout_path,
            transpose_path,
            spill_reload,
            scale_packing,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("ratio", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("cache_case", ["miss", "hit"])
    def test_backward_feature_ratios(self, test_manager, platform_target, monkeypatch, ratio, cache_case):
        original = _phase_config

        def with_ratios(quant_scheme, spill_reload, scale_packing):
            phase = original(quant_scheme, spill_reload, scale_packing)
            phase.psum_copy_in_ratio = ratio
            phase.strided_copy_swizzle_1x32_ratio = ratio
            return phase

        monkeypatch.setattr(__name__ + "._phase_config", with_ratios)
        self._run_backward(
            test_manager,
            platform_target,
            tokens=1024,
            weight_input="prequantized",
            layout_path="1x32_pe_only",
            transpose_path="nc",
            spill_reload=False,
            scale_packing=True,
            cache_case=cache_case,
            monkeypatch=monkeypatch,
        )


@pytest_marks(["mlp_mxfp8", "mxfp8", "model", "validated"])
@pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN1, common_dataclasses.Platforms.TRN2])
@final
class TestMlpDefaultConfigCache:
    """Compare the real populated cache against defaults through the public API."""

    @pytest.mark.parametrize("direction", ["forward", "backward"])
    @pytest.mark.parametrize("weight_input", ["bf16", "prequantized"])
    @pytest.mark.parametrize("cache_case,caller_config", [("hit", "omitted"), ("miss", "omitted"), ("hit", "empty")])
    def test_public_mlp_default_config(
        self,
        test_manager: Orchestrator,
        platform_target,
        monkeypatch,
        direction,
        weight_input,
        cache_case,
        caller_config,
    ):
        tokens, hidden, intermediate = 8192, 4096, 1536
        block_size = tokens // 2
        backward = direction == "backward"
        config_type = MXFP8MOEBwdConfig if backward else MXFP8MOEFwdConfig
        module = bwd_entry if backward else fwd_entry
        reader_name = "get_cached_moe_bwd_config" if backward else "get_cached_moe_fwd_config"
        reader = getattr(module, reader_name)
        expected_hit = cache_case == "hit"
        if not expected_hit:
            monkeypatch.setattr(_best_config_cache, "_MOE_FWD_CONFIG_CACHE", {})
            monkeypatch.setattr(_best_config_cache, "_MOE_BWD_CONFIG_CACHE", {})

        def checked_reader(key, config):
            result = reader(key, config)
            kernel_assert(
                (result != None) == expected_hit,  # noqa: E711
                "Public MLP default config did not follow the expected cache path",
            )
            return result

        monkeypatch.setattr(module, reader_name, checked_reader)
        x, gate_up, down, dy = _inputs(tokens, hidden, intermediate)
        inputs = {
            "hidden_states": x,
            "gate_up_proj_weight": gate_up,
            "down_proj_weight": down,
        }
        if caller_config == "empty":
            inputs["config"] = config_type()
        if backward:
            inputs["output_hidden_states_grad"] = dy
            inputs["gate_up_proj_act_checkpoint_T"] = (
                _checkpoint(
                    torch.from_numpy(x.astype(np.float32)),
                    torch.from_numpy(gate_up.astype(np.float32)),
                    block_size,
                )
                .numpy()
                .astype(ml_dtypes.bfloat16)
            )
        if weight_input == "prequantized":
            if backward:
                packed_gate, gate_scales, packed_down, down_scales = prequantize_moe_weights(
                    gate_up[None], down[None], quant_scheme=QuantScheme._1x32
                )
            else:
                packed_gate, gate_scales, packed_down, down_scales = prequantize_moe_weights_fwd(
                    gate_up[None], down[None], use_1x32=True
                )
            inputs.update(
                gate_up_proj_weight=packed_gate[0],
                down_proj_weight=packed_down[0],
                gate_up_weight_scales=gate_scales[0],
                down_weight_scales=down_scales[0],
            )

        reference = cast(Callable[..., dict], _bwd_ref if backward else _fwd_ref)

        @functools.wraps(reference)
        def ref(**kwargs):
            kwargs["gate_up_proj_weight"] = torch.from_numpy(gate_up.astype(np.float32))
            kwargs["down_proj_weight"] = torch.from_numpy(down.astype(np.float32))
            return reference(**kwargs)

        if backward:
            shapes = {
                "hidden_states_grad": ((tokens, hidden), nl.bfloat16),
                "gate_up_proj_weight_grad": ((hidden, 2, intermediate), nl.bfloat16),
                "down_proj_weight_grad": ((intermediate, hidden), nl.bfloat16),
            }
        else:
            shapes = {
                "output_hidden_states": ((tokens, hidden), nl.bfloat16),
                "gate_up_proj_act_checkpoint_T": ((2, 2, block_size, intermediate), nl.bfloat16),
            }
        test_manager.collector.set_kernel_params(
            {
                "model": "qwen3_235b",
                "kernel": "mlp_mxfp8",
                "direction": direction,
                "tokens": tokens,
                "hidden": hidden,
                "intermediate": intermediate,
                "block_size": block_size,
                "num_blocks": 2,
                "weight_input": weight_input,
                "quant_scheme": "1x32",
                "cache_case": cache_case,
                "caller_config": caller_config,
                "dma_transpose_enabled": config_type().dma_transpose_enabled,
                "weight_orientation": "backward_natural",
            }
        )
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=mlp_backward_mxfp8_nki if backward else mlp_forward_mxfp8_nki,
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
