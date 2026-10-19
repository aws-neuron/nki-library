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

"""Qwen3-235B expert MLP blocking candidates for the Team Signal sweeps."""

from dataclasses import dataclass

from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_config import MatmulMxfp8KernelConfig
from nkilib_src.nkilib.experimental.moe_mxfp8.bwd.config import MXFP8MOEBwdConfig, TransposeMode
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.common_dataclasses import LoopOrder, QuantScheme


@dataclass(frozen=True)
class _PhaseSpec:
    tile_n: int
    block_m: int
    block_n: int
    block_k: int
    load_m: int
    load_n: int


@dataclass(frozen=True)
class _FwdSpec:
    gate_up: _PhaseSpec
    down: _PhaseSpec
    spill_reload: bool = True
    block_loop_order: LoopOrder = LoopOrder.MNK


@dataclass(frozen=True)
class FwdTunedConfig:
    gate_up_config: MatmulMxfp8KernelConfig
    down_config: MatmulMxfp8KernelConfig
    reuse_spilled_weights: bool = False


_FWD_BASE = _PhaseSpec(512, 16, 2, 3, 8, 2)
_FWD_WIDE_GATE_M8 = _PhaseSpec(512, 8, 3, 4, 8, 3)
_FWD_WIDE_GATE_M16 = _PhaseSpec(512, 16, 3, 4, 8, 3)
_FWD_FULL_K_GATE = _PhaseSpec(512, 8, 3, 8, 8, 3)
_FWD_TN256_GATE = _PhaseSpec(256, 8, 3, 4, 8, 3)
_FWD_WIDE_DOWN_M8 = _PhaseSpec(512, 8, 4, 3, 8, 4)
_FWD_WIDE_DOWN_M16 = _PhaseSpec(512, 16, 4, 3, 8, 4)

FWD_SWEEP_CONFIGS = {
    "baseline": _FwdSpec(_FWD_BASE, _FWD_BASE),
    "wide_m8": _FwdSpec(_FWD_WIDE_GATE_M8, _FWD_WIDE_DOWN_M8),
    "wide_m16": _FwdSpec(_FWD_WIDE_GATE_M16, _FWD_WIDE_DOWN_M16),
    "asymmetric": _FwdSpec(_FWD_WIDE_GATE_M8, _FWD_WIDE_DOWN_M16),
    "full_k_gate": _FwdSpec(_FWD_FULL_K_GATE, _FWD_WIDE_DOWN_M8),
    "gate_tn256": _FwdSpec(_FWD_TN256_GATE, _FWD_WIDE_DOWN_M8),
    "baseline_no_spill": _FwdSpec(_FWD_BASE, _FWD_BASE, spill_reload=False),
    "baseline_nmk": _FwdSpec(_FWD_BASE, _FWD_BASE, block_loop_order=LoopOrder.NMK),
}


_BWD_WRAPX = (
    _PhaseSpec(512, 16, 2, 4, 8, 2),
    _PhaseSpec(512, 16, 2, 6, 8, 2),
    _PhaseSpec(512, 12, 4, 2, 6, 2),
    _PhaseSpec(512, 8, 2, 2, 8, 2),
)
_BWD_1X32 = (
    _PhaseSpec(256, 8, 3, 8, 8, 3),
    _PhaseSpec(512, 8, 4, 3, 4, 4),
    _PhaseSpec(512, 24, 2, 2, 24, 2),
    _PhaseSpec(512, 16, 3, 2, 16, 3),
)

BWD_SWEEP_CONFIGS = (
    "baseline",
    "alternate_family",
    "phase1_alternate",
    "phase2_alternate",
    "phase3_alternate",
    "phase4_alternate",
    "baseline_nmk",
    "baseline_no_psum_copy_in",
    "baseline_spill_reload",
)


def _legal_dgt_load(block_tiles, tile_size):
    cap = 8 if tile_size == 128 else max(1, 1024 // tile_size)
    for load_tiles in range(min(block_tiles, cap), 0, -1):
        if block_tiles % load_tiles == 0:
            return load_tiles
    return 1


def _phase_config(
    shape,
    spec,
    *,
    quant_scheme,
    pe_loads,
    spill_reload,
    block_loop_order=LoopOrder.MNK,
    psum_copy_in_ratio=1 / 3,
    enable_scale_packing=True,
):
    M, K, N = shape
    load_m = spec.load_m if pe_loads else _legal_dgt_load(spec.block_m, 128)
    load_n = spec.load_n if pe_loads else _legal_dgt_load(spec.block_n, spec.tile_n)
    return MatmulMxfp8KernelConfig(
        M=M,
        K=K,
        N=N,
        tile_m=128,
        tile_k=512,
        tile_n=spec.tile_n,
        TILES_IN_BLOCK_M=spec.block_m,
        TILES_IN_BLOCK_N=spec.block_n,
        TILES_IN_BLOCK_K=spec.block_k,
        TILES_IN_LOAD_M=load_m,
        TILES_IN_LOAD_N=load_n,
        block_loop_order=block_loop_order,
        tile_loop_order=LoopOrder.MNK,
        spill_reload=spill_reload,
        enable_scale_packing=enable_scale_packing,
        psum_copy_in_ratio=psum_copy_in_ratio,
        quant_scheme=quant_scheme,
    )


def build_fwd_candidate(config_id, quant_scheme, pe_loads):
    spec = FWD_SWEEP_CONFIGS[config_id]
    return FwdTunedConfig(
        gate_up_config=_phase_config(
            (4096, 4096, 1536),
            spec.gate_up,
            quant_scheme=quant_scheme,
            pe_loads=pe_loads,
            spill_reload=spec.spill_reload,
            block_loop_order=spec.block_loop_order,
        ),
        down_config=_phase_config(
            (4096, 1536, 4096),
            spec.down,
            quant_scheme=quant_scheme,
            pe_loads=pe_loads,
            spill_reload=spec.spill_reload,
            block_loop_order=spec.block_loop_order,
        ),
    )


def _bwd_specs(config_id, quant_scheme):
    baseline = _BWD_1X32 if quant_scheme == QuantScheme._1x32 else _BWD_WRAPX
    alternate = _BWD_WRAPX if quant_scheme == QuantScheme._1x32 else _BWD_1X32
    if config_id == "alternate_family":
        return alternate
    if config_id.startswith("phase") and config_id.endswith("_alternate"):
        phase_index = int(config_id[5]) - 1
        return tuple(alternate[i] if i == phase_index else baseline[i] for i in range(4))
    return baseline


def build_bwd_candidate(config_id, quant_scheme, pe_loads, use_scale_packing):
    specs = _bwd_specs(config_id, quant_scheme)
    block_loop_order = LoopOrder.NMK if config_id == "baseline_nmk" else LoopOrder.MNK
    psum_copy_in_ratio = 0.0 if config_id == "baseline_no_psum_copy_in" else 1 / 3
    spill_reload = config_id == "baseline_spill_reload"
    shapes = (
        (4096, 4096, 768),
        (4096, 3072, 2048),
        (3072, 4096, 2048),
        (2048, 4096, 1536),
    )
    phases = tuple(
        _phase_config(
            shape,
            spec,
            quant_scheme=quant_scheme,
            pe_loads=pe_loads,
            spill_reload=spill_reload,
            block_loop_order=block_loop_order,
            psum_copy_in_ratio=psum_copy_in_ratio,
            enable_scale_packing=use_scale_packing,
        )
        for shape, spec in zip(shapes, specs, strict=True)
    )
    return MXFP8MOEBwdConfig(
        phase1_config=phases[0],
        phase2_config=phases[1],
        phase3_config=phases[2],
        phase4_config=phases[3],
        phase3_transpose_mode=TransposeMode.NC,
        phase4_transpose_mode=TransposeMode.NC,
    )


def phase_config_as_dict(config):
    return {
        "tile_m": config.tile_m,
        "tile_k": config.tile_k,
        "tile_n": config.tile_n,
        "block_m": config.TILES_IN_BLOCK_M,
        "block_n": config.TILES_IN_BLOCK_N,
        "block_k": config.TILES_IN_BLOCK_K,
        "load_m": config.TILES_IN_LOAD_M,
        "load_n": config.TILES_IN_LOAD_N,
        "block_loop_order": config.block_loop_order.value,
        "spill_reload": config.spill_reload,
        "enable_scale_packing": config.enable_scale_packing,
        "psum_copy_in_ratio": config.psum_copy_in_ratio,
    }


def fwd_config_as_dict(config):
    return {
        "gate_up": phase_config_as_dict(config.gate_up_config),
        "down": phase_config_as_dict(config.down_config),
        "reuse_spilled_weights": config.reuse_spilled_weights,
    }


def bwd_config_as_dict(config):
    return {
        "phase1": phase_config_as_dict(config.phase1_config),
        "phase2": phase_config_as_dict(config.phase2_config),
        "phase3": phase_config_as_dict(config.phase3_config),
        "phase4": phase_config_as_dict(config.phase4_config),
        "phase3_transpose_mode": config.phase3_transpose_mode.value,
        "phase4_transpose_mode": config.phase4_transpose_mode.value,
    }
