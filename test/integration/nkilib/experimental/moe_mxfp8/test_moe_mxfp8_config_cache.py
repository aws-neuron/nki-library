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

"""Whole-config cache hits must respect every caller preset."""

import json
import math
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import nki.language as nl
import numpy as np
import pytest
from nki.dtype import float8_e4m3fn_x4  # ty: ignore[unresolved-import]
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_config import MatmulMxfp8KernelConfig
from nkilib_src.nkilib.experimental.moe.bwd.moe_bwd_parameters import ClampLimits
from nkilib_src.nkilib.experimental.moe_mxfp8 import _best_config_cache
from nkilib_src.nkilib.experimental.moe_mxfp8._config_cache import (
    get_cached_moe_bwd_config,
    get_cached_moe_fwd_config,
)
from nkilib_src.nkilib.experimental.moe_mxfp8._config_codec import decode_config
from nkilib_src.nkilib.experimental.moe_mxfp8._config_helpers import (
    MoEInputShape,
    WeightOrientation,
    copy_config,
    make_moe_config_key,
    make_pair,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.bwd.blockwise_mm_backward_mxfp8 import _resolve_phase_configs
from nkilib_src.nkilib.experimental.moe_mxfp8.bwd.config import MXFP8MOEBwdConfig, TransposeMode
from nkilib_src.nkilib.experimental.moe_mxfp8.config_helpers import ExecutionMode
from nkilib_src.nkilib.experimental.moe_mxfp8.fwd.config import (
    MXFP8MOEFwdConfig,
    _auto_generate_moe_fwd_configs,
)
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.common_dataclasses import (
    QuantScheme,
    SwizzleMode,
    TensorDescriptor,
    TensorOrientation,
)

from test.integration.nkilib.experimental.moe_mxfp8.config_sweep_utils import (
    cache_lookup_config,
    config_literal,
    emit_cache_candidate,
    pack_cache_tables,
    read_cache_tables,
    read_config_literal,
    record_cache_candidate,
    select_cache_updates,
    write_cache_tables,
)
from test.utils.metrics_collector import MetricName, MetricsCollector

pytestmark = pytest.mark.fast


@pytest.fixture
def empty_cache(monkeypatch):
    monkeypatch.setattr(_best_config_cache, "_MOE_FWD_CONFIG_CACHE", {})
    monkeypatch.setattr(_best_config_cache, "_MOE_BWD_CONFIG_CACHE", {})
    monkeypatch.setattr(_best_config_cache, "_CONFIG_SCHEMAS", {})
    monkeypatch.setattr(_best_config_cache, "_CONFIG_RECORDS", ())


def _store_cached(cache, key, config, latency=0.0001):
    direction = "backward" if isinstance(config, MXFP8MOEBwdConfig) else "forward"
    tables = {"forward": {}, "backward": {}}
    tables[direction][key] = _best_config_cache.CachedConfig(config, latency)
    schemas, records, _, packed = pack_cache_tables(tables)
    _best_config_cache._CONFIG_SCHEMAS = schemas
    _best_config_cache._CONFIG_RECORDS = records
    cache[key] = packed[direction][key]


def _shape():
    return MoEInputShape((1024, 512), (1, 512, 2, 256), (1, 256, 512), 512)


def _case(backward):
    if backward:
        user = MXFP8MOEBwdConfig()
        reader = get_cached_moe_bwd_config
        cache = _best_config_cache._MOE_BWD_CONFIG_CACHE
        phase_names = ("phase1_config", "phase2_config", "phase3_config", "phase4_config")
    else:
        user = MXFP8MOEFwdConfig()
        reader = get_cached_moe_fwd_config
        cache = _best_config_cache._MOE_FWD_CONFIG_CACHE
        phase_names = ("gate_up_config", "down_config")
    return user, reader, cache, phase_names


def _cached(user, phase_names):
    cached = copy_config(user)
    cached.resolve_phase_defaults()
    for name in phase_names:
        setattr(
            cached,
            name,
            MatmulMxfp8KernelConfig(
                512,
                512,
                256,
                tile_m=128,
                tile_k=512,
                tile_n=256,
                TILES_IN_BLOCK_M=2,
                TILES_IN_BLOCK_N=2,
                TILES_IN_BLOCK_K=1,
                TILES_IN_LOAD_M=2,
                TILES_IN_LOAD_N=2,
                enable_scale_packing=True,
                quant_scheme=QuantScheme._1x32,
                spill_reload=False,
            ),
        )
    return cached


@pytest.mark.parametrize("backward", [False, True])
def test_empty_cache(empty_cache, backward):
    _, _, cache, _ = _case(backward)
    assert cache == {}


@pytest.mark.parametrize("backward", [False, True])
def test_miss_preserves_user_config(empty_cache, backward):
    user, reader, _, phases = _case(backward)
    user.num_static_block = 3
    key = make_moe_config_key(_shape(), nl.bfloat16, user, backward)
    assert reader(key, user) is None
    assert user.num_static_block == 3
    assert all(getattr(user, name) is None for name in phases)


@pytest.mark.parametrize("backward", [False, True])
def test_hit_returns_whole_independent_config(empty_cache, backward):
    user, reader, cache, phases = _case(backward)
    key = make_moe_config_key(_shape(), nl.bfloat16, user, backward)
    cached = _cached(user, phases)
    _store_cached(cache, key, cached)
    result = reader(key, user)
    assert result == cached
    assert result is not cached
    assert result.clamp_limits is not cached.clamp_limits
    for name in phases:
        assert getattr(result, name) is not getattr(cached, name)
    getattr(result, phases[0]).TILES_IN_BLOCK_M = 4
    assert getattr(cached, phases[0]).TILES_IN_BLOCK_M == 2
    assert getattr(reader(key, user), phases[0]).TILES_IN_BLOCK_M == 2
    assert getattr(user, phases[0]) is None


@pytest.mark.parametrize("backward", [False, True])
def test_compatible_phase_presets_accept_entire_hit(empty_cache, backward):
    user, reader, cache, phases = _case(backward)
    key = make_moe_config_key(_shape(), nl.bfloat16, user, backward)
    cached = _cached(user, phases)
    _store_cached(cache, key, cached)
    setattr(
        user, phases[0], MatmulMxfp8KernelConfig(0, 0, 0, enable_scale_packing=True, quant_scheme=QuantScheme._1x32)
    )
    result = reader(key, user)
    assert result is not None
    assert getattr(result, phases[-1]).TILES_IN_BLOCK_N == 2
    assert getattr(user, phases[0]).TILES_IN_BLOCK_M is None


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize(
    "field,value",
    [
        ("TILES_IN_BLOCK_M", 1),
        ("enable_scale_packing", False),
        ("spill_reload", True),
        ("psum_copy_in_ratio", 0.5),
        ("strided_copy_swizzle_1x32_ratio", 0.5),
    ],
)
def test_any_phase_conflict_rejects_whole_hit(empty_cache, backward, field, value):
    user, reader, cache, phases = _case(backward)
    key = make_moe_config_key(_shape(), nl.bfloat16, user, backward)
    _store_cached(cache, key, _cached(user, phases))
    preset = MatmulMxfp8KernelConfig(0, 0, 0, enable_scale_packing=True, quant_scheme=QuantScheme._1x32)
    setattr(preset, field, value)
    setattr(user, phases[0], preset)
    assert reader(key, user) is None
    assert getattr(preset, field) == value
    assert getattr(user, phases[-1]) is None


@pytest.mark.parametrize("backward", [False, True])
def test_nested_nonphase_conflict_rejects_hit(empty_cache, backward):
    user, reader, cache, phases = _case(backward)
    key = make_moe_config_key(_shape(), nl.bfloat16, user, backward)
    _store_cached(cache, key, _cached(user, phases))
    user.clamp_limits = ClampLimits(linear_clamp_upper_limit=1.0)
    assert reader(key, user) is None
    assert user.clamp_limits.linear_clamp_upper_limit == 1.0


@pytest.mark.parametrize("backward", [False, True])
def test_hit_cannot_enable_unrequested_clamping(empty_cache, backward):
    user, reader, cache, phases = _case(backward)
    key = make_moe_config_key(_shape(), nl.bfloat16, user, backward)
    cached = _cached(user, phases)
    cached.clamp_limits = ClampLimits(linear_clamp_upper_limit=1.0)
    _store_cached(cache, key, cached)
    assert reader(key, user) is None
    user.clamp_limits = ClampLimits(linear_clamp_upper_limit=1.0)
    assert reader(key, user) is not None


@pytest.mark.parametrize("backward", [False, True])
def test_mislabelled_cache_key_is_rejected(empty_cache, backward):
    user, reader, cache, phases = _case(backward)
    key = make_moe_config_key(_shape(), nl.bfloat16, user, backward)
    cached = _cached(user, phases)
    cached.dma_transpose_enabled = not key.dma_transpose_enabled
    _store_cached(cache, key, cached)
    assert reader(key, user) is None


@pytest.mark.parametrize("backward", [False, True])
def test_hit_keeps_default_packing_unless_user_overrides(empty_cache, backward):
    user, reader, cache, phases = _case(backward)
    key = make_moe_config_key(_shape(), nl.bfloat16, user, backward)
    cached = _cached(user, phases)
    getattr(cached, phases[0]).enable_scale_packing = False
    _store_cached(cache, key, cached)
    assert reader(key, user) is None
    setattr(
        user, phases[0], MatmulMxfp8KernelConfig(0, 0, 0, enable_scale_packing=False, quant_scheme=QuantScheme._1x32)
    )
    result = reader(key, user)
    assert result is not None
    assert not getattr(result, phases[0]).enable_scale_packing


@pytest.mark.parametrize("backward", [False, True])
def test_conflict_falls_back_without_using_other_cached_phases(empty_cache, backward):
    user, reader, cache, phases = _case(backward)
    key = make_moe_config_key(_shape(), nl.bfloat16, user, backward)
    _store_cached(cache, key, _cached(user, phases))
    setattr(
        user,
        phases[0],
        MatmulMxfp8KernelConfig(
            0,
            0,
            0,
            TILES_IN_BLOCK_M=1,
            enable_scale_packing=False,
            spill_reload=False,
        ),
    )
    assert reader(key, user) is None
    config = copy_config(user)
    if backward:
        _resolve_phase_configs(
            config,
            ((512, 512, 128), (512, 512, 256), (256, 512, 512), (256, 512, 256)),
            512,
            True,
        )
        assert config.phase4_config.TILES_IN_BLOCK_M == 1
    else:
        _auto_generate_moe_fwd_configs(config, 512, 512, 256)
        assert config.down_config.spill_reload
    assert not getattr(config, phases[0]).enable_scale_packing
    assert getattr(config, phases[0]).TILES_IN_BLOCK_M == 1
    assert getattr(user, phases[-1]) is None


@pytest.mark.parametrize("backward", [False, True])
def test_exact_key_and_pass_are_required(empty_cache, backward):
    user, reader, cache, phases = _case(backward)
    key = make_moe_config_key(_shape(), nl.bfloat16, user, backward)
    _store_cached(cache, key, _cached(user, phases))
    user.dma_transpose_enabled = False
    changed = make_moe_config_key(_shape(), nl.bfloat16, user, backward)
    assert reader(changed, user) is None
    other_reader = get_cached_moe_fwd_config if backward else get_cached_moe_bwd_config
    assert other_reader(key, user) is None


def test_six_keys_and_shared_execution_mode():
    config = MXFP8MOEFwdConfig(execution_mode=ExecutionMode.MLP)
    key = make_moe_config_key(_shape(), nl.bfloat16, config)
    assert len(fields(key)) == 6
    assert key.execution_mode == ExecutionMode.MLP
    assert key.quant_scheme == QuantScheme._1x32
    assert key.weight_orientation == (WeightOrientation.BWD_NATURAL, WeightOrientation.BWD_NATURAL)
    assert make_pair(nl.bfloat16) == (nl.bfloat16, nl.bfloat16)
    assert "single_expert_dense" not in {field.name for field in fields(config)}
    assert "mlp_mode" not in {field.name for field in fields(config)}


@pytest.mark.parametrize("factory", [MXFP8MOEFwdConfig, MXFP8MOEBwdConfig])
@pytest.mark.parametrize("dma", [False, True])
def test_dma_and_phase_defaults(factory, dma):
    config = factory(dma_transpose_enabled=dma)
    assert config.fast_dma_transpose == dma
    config.resolve_phase_defaults()
    phase = config.phase1_config if isinstance(config, MXFP8MOEBwdConfig) else config.gate_up_config
    assert phase.enable_scale_packing
    if isinstance(config, MXFP8MOEFwdConfig):
        assert config.reuse_spilled_weights


def test_explicit_packing_false_survives_default_resolution():
    phase = MatmulMxfp8KernelConfig(0, 0, 0, enable_scale_packing=False, quant_scheme=QuantScheme._1x32)
    config = MXFP8MOEFwdConfig(gate_up_config=phase)
    config.resolve_phase_defaults()
    assert config.gate_up_config is phase
    assert not config.gate_up_config.enable_scale_packing
    assert config.down_config.enable_scale_packing


@pytest.mark.parametrize("backward", [False, True])
def test_populated_entries_are_measured_complete_and_compatible(backward):
    _, reader, cache, phases = _case(backward)
    for key, entry in cache.items():
        cached = decode_config(entry[1], _best_config_cache._CONFIG_SCHEMAS, _best_config_cache._CONFIG_RECORDS)
        assert math.isfinite(entry[0]) and entry[0] > 0
        assert make_moe_config_key(key.input_shape, key.weight_dtype, cached, backward) == key
        user = cache_lookup_config(key, cached, backward)
        for name in phases:
            phase = getattr(cached, name)
            assert all(
                getattr(phase, field) is not None and getattr(phase, field) > 0
                for field in ("M", "K", "N", "tile_m", "tile_k", "tile_n")
            )
        assert make_moe_config_key(key.input_shape, key.weight_dtype, user, backward) == key
        result = reader(key, user)
        assert result == cached and result is not cached
        setattr(user, phases[0], copy_config(getattr(cached, phases[0])))
        assert reader(key, user) is not None
        last_phase = copy_config(getattr(user, phases[-1]))
        getattr(user, phases[0]).TILES_IN_BLOCK_M += 1
        assert reader(key, user) is None
        assert getattr(user, phases[-1]) == last_phase


def _measurement(backward, config, latency, status="SUCCESS"):
    key = make_moe_config_key(_shape(), nl.bfloat16, config, backward)
    return {
        "Status": status,
        "ActiveInferenceTime": latency,
        "cache_candidate": {
            "direction": "backward" if backward else "forward",
            "key": config_literal(key),
            "config": config_literal(config),
        },
    }


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("latency", [0.00008, 0.0001, 0.00012])
def test_cache_replaces_entire_entry_only_when_strictly_faster(empty_cache, backward, latency):
    user, _, _, phases = _case(backward)
    previous = _cached(user, phases)
    candidate = copy_config(previous)
    getattr(candidate, phases[0]).TILES_IN_BLOCK_M = 1
    key = make_moe_config_key(_shape(), nl.bfloat16, previous, backward)
    old_entry = _best_config_cache.CachedConfig(previous, 0.0001)
    tables = {"forward": {}, "backward": {}}
    direction = "backward" if backward else "forward"
    tables[direction][key] = old_entry
    result, updates = select_cache_updates(
        [_measurement(backward, candidate, latency)], tables["forward"], tables["backward"]
    )
    winner = result[direction][key]
    assert tables[direction][key] is old_entry
    if latency < old_entry.active_inference_time:
        assert winner.active_inference_time == latency
        assert winner.config == candidate and winner.config is not candidate
        assert len(updates) == 1
    else:
        assert winner is old_entry
        assert updates == []


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize(
    "latency,status",
    [(0, "SUCCESS"), (-1, "SUCCESS"), (float("nan"), "SUCCESS"), (float("inf"), "SUCCESS"), (0.00001, "FAILURE")],
)
def test_invalid_or_failed_measurement_cannot_populate_cache(empty_cache, backward, latency, status):
    user, _, _, phases = _case(backward)
    candidate = _cached(user, phases)
    tables, updates = select_cache_updates([_measurement(backward, candidate, latency, status)], {}, {})
    assert tables == {"forward": {}, "backward": {}}
    assert updates == []


@pytest.mark.parametrize("backward", [False, True])
def test_new_key_keeps_fastest_successful_whole_config(empty_cache, backward):
    user, _, _, phases = _case(backward)
    baseline = _cached(user, phases)
    faster = copy_config(baseline)
    getattr(faster, phases[-1]).spill_reload = True
    records = [
        _measurement(backward, baseline, 0.00012),
        _measurement(backward, faster, 0.00008),
        _measurement(backward, baseline, 0.0001),
    ]
    tables, _ = select_cache_updates(records, {}, {})
    key = make_moe_config_key(_shape(), nl.bfloat16, baseline, backward)
    winner = tables["backward" if backward else "forward"][key]
    assert winner.active_inference_time == 0.00008
    assert winner.config == faster
    assert config_literal(read_config_literal(config_literal(winner))) == config_literal(winner)


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("mode", [ExecutionMode.MLP, ExecutionMode.DENSE_MOE])
def test_sweep_recorder_uses_array_shapes_without_array_comparisons(empty_cache, monkeypatch, backward, mode):
    monkeypatch.setenv("NKILIB_MXFP8_CACHE_SWEEP", "1")
    monkeypatch.delenv("NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE", raising=False)
    prefix = () if mode == ExecutionMode.MLP else (1,)
    config_type = MXFP8MOEBwdConfig if backward else MXFP8MOEFwdConfig
    inputs = {
        "config": config_type(execution_mode=mode),
        "block_size": 512,
        "hidden_states": np.zeros((1024, 512), dtype=np.float32),
        "gate_up_proj_weight": np.zeros((*prefix, 256, 256), dtype=np.float32),
        "down_proj_weight": np.zeros((*prefix, 64, 512), dtype=np.float32),
        "gate_up_weight_scales": np.zeros((*prefix, 32, 256), dtype=np.uint8),
        "down_weight_scales": np.zeros((*prefix, 8, 512), dtype=np.uint8),
        "num_valid_tokens": np.array([1024], dtype=np.int32),
        "gate_up_proj_act_checkpoint_T": np.zeros((2, 2, 512, 256), dtype=np.float32),
    }
    collector = MetricsCollector()
    record_cache_candidate(SimpleNamespace(collector=collector), inputs, backward)
    candidate = collector.get_kernel_params()["cache_candidate"]
    key = read_config_literal(candidate["key"])
    assert key.input_shape.weight_scale_shapes == (
        (*prefix, 32, 256),
        (*prefix, 8, 512),
    )
    assert key.input_shape.num_valid_tokens_shape == (1,)
    measured = read_config_literal(candidate["config"])
    assert measured == inputs["config"]
    assert make_moe_config_key(key.input_shape, key.weight_dtype, measured, backward) == key


def test_sweep_emits_only_validated_positive_active_inference_time(empty_cache, monkeypatch, tmp_path):
    monkeypatch.setenv("NKILIB_MXFP8_CACHE_SWEEP", "1")
    collector = MetricsCollector(output_dir=str(tmp_path))
    collector.add_dimension({"Status": "SUCCESS", "InstanceType": ""})
    manager = SimpleNamespace(collector=collector)
    collector.record_metric(MetricName.ACTIVE_INFERENCE_TIME, 0.0001, "Seconds")
    emit_cache_candidate(manager)
    assert not (tmp_path / "metrics").exists()
    collector.record_metric(MetricName.VALIDATION_TIME, 0.1, "Seconds")
    emit_cache_candidate(manager)
    files = list((tmp_path / "metrics").glob("*.json"))
    assert len(files) == 1
    recorded = json.loads(files[0].read_text())
    assert recorded["ActiveInferenceTime"] == 0.0001 and recorded["Status"] == "SUCCESS"
    assert "InstanceType" not in recorded and collector.dimensions["InstanceType"] == ""
    collector.record_metric(MetricName.ACTIVE_INFERENCE_TIME, -1.0, "Seconds")
    with pytest.raises(ValueError, match="positive ActiveInferenceTime"):
        emit_cache_candidate(manager)


def test_cache_writer_round_trips_and_refuses_concurrent_changes(empty_cache, tmp_path):
    source_path = _best_config_cache.__file__
    original = Path(source_path).read_text()
    path = tmp_path / "best_config_cache.py"
    path.write_text(original)
    user, _, _, phases = _case(False)
    config = _cached(user, phases)
    tables, _ = select_cache_updates([_measurement(False, config, 0.0001)], {}, {})
    write_cache_tables(path, original, tables)
    source, read_back = read_cache_tables(path)
    assert read_back == tables
    assert "(0.0001," in source
    with pytest.raises(ValueError, match="changed during"):
        write_cache_tables(path, original, tables)


def test_cache_reader_rejects_unpacking(tmp_path):
    path = tmp_path / "best_config_cache.py"
    path.write_text("_MOE_FWD_CONFIG_CACHE: dict = {**{}}\n_MOE_BWD_CONFIG_CACHE: dict = {}\n")
    with pytest.raises(ValueError, match="unpacked mappings"):
        read_cache_tables(path)


def test_shared_records_reconstruct_independent_nested_configs(empty_cache):
    user, _, _, phases = _case(False)
    config = _cached(user, phases)
    config.down_config = config.gate_up_config
    key = make_moe_config_key(_shape(), nl.bfloat16, config)
    tables = {"forward": {key: _best_config_cache.CachedConfig(config, 0.0001)}, "backward": {}}
    schemas, records, _, packed = pack_cache_tables(tables)
    assert sum(name == "MatmulMxfp8KernelConfig" for name, _ in records) == 1
    decoded = decode_config(packed["forward"][key][1], schemas, records)
    assert decoded == config
    assert decoded.gate_up_config is not decoded.down_config
    decoded.gate_up_config.TILES_IN_BLOCK_M = 99
    assert decoded.down_config.TILES_IN_BLOCK_M == 2
    assert config.gate_up_config.TILES_IN_BLOCK_M == 2


@pytest.mark.parametrize("invalid", ["missing_field", "forward_reference"])
def test_compact_reader_rejects_incomplete_or_cyclic_records(empty_cache, tmp_path, invalid):
    original = Path(_best_config_cache.__file__).read_text()
    path = tmp_path / "best_config_cache.py"
    path.write_text(original)
    user, _, _, phases = _case(False)
    config = _cached(user, phases)
    tables, _ = select_cache_updates([_measurement(False, config, 0.0001)], {}, {})
    write_cache_tables(path, original, tables)
    source = path.read_text()
    if invalid == "missing_field":
        source = source.replace("'compute_dtype': 'bfloat16',", "", 1)
        expected = "every field exactly once"
    else:
        source = source.replace("('ref', 0)", "('ref', 999)", 1)
        expected = "earlier records"
    path.write_text(source)
    with pytest.raises(ValueError, match=expected):
        read_cache_tables(path)


def test_config_codec_rejects_tensor_data_and_executable_expressions(empty_cache):
    with pytest.raises(ValueError, match="data-free"):
        config_literal(object())
    with pytest.raises(ValueError, match="Unsupported"):
        read_config_literal("__import__('os').system('false')")


def test_backward_defaults_to_1x32_and_rejects_mutated_mixed_phases():
    config = MXFP8MOEBwdConfig()
    key = make_moe_config_key(_shape(), nl.bfloat16, config, True)
    assert key.quant_scheme == QuantScheme._1x32
    config.resolve_phase_defaults()
    assert all(getattr(config, f"phase{i}_config").quant_scheme == QuantScheme._1x32 for i in range(1, 5))
    config.phase4_config.quant_scheme = QuantScheme.WRAPX
    with pytest.raises(AssertionError, match="same quantization scheme"):
        config.resolve_phase_defaults()


@pytest.mark.parametrize(
    "field,tuned,fallback",
    [
        ("pe_transpose_only", True, False),
        ("phase3_transpose_mode", TransposeMode.DMA, TransposeMode.NC),
        ("phase4_transpose_mode", TransposeMode.DMA, TransposeMode.NC),
        ("output_grad_swizzle_mode", SwizzleMode.DGT, SwizzleMode.PE),
        ("down_weight_swizzle_mode", SwizzleMode.DGT, SwizzleMode.PE),
        ("d_gate_up_swizzle_mode", SwizzleMode.DGT, SwizzleMode.PE),
        ("gate_up_weight_swizzle_mode", SwizzleMode.DGT, SwizzleMode.PE),
        ("d_gate_up_t_swizzle_mode", SwizzleMode.DGT, SwizzleMode.PE),
        ("hidden_states_t_swizzle_mode", SwizzleMode.DGT, SwizzleMode.PE),
        ("output_grad_t_swizzle_mode", SwizzleMode.DGT, SwizzleMode.PE),
        ("scaled_intermediate_t_swizzle_mode", SwizzleMode.DGT, SwizzleMode.PE),
    ],
)
def test_unset_backward_tuning_accepts_winner_but_explicit_presets_constrain_it(empty_cache, field, tuned, fallback):
    user, reader, cache, phases = _case(True)
    assert getattr(user, field) is None
    cached = _cached(user, phases)
    setattr(cached, field, tuned)
    key = make_moe_config_key(_shape(), nl.bfloat16, user, True)
    _store_cached(cache, key, cached)
    assert getattr(reader(key, user), field) == tuned
    setattr(user, field, fallback)
    assert reader(key, user) is None
    user.resolve_phase_defaults()
    assert getattr(user, field) == fallback
    setattr(user, field, tuned)
    # Only the tuning preset changes; leave blocking unset for the next lookup.
    for name in phases:
        setattr(user, name, None)
    assert getattr(reader(key, user), field) == tuned


@pytest.mark.parametrize("backward", [False, True])
def test_populated_1x32_profiles_accept_ordinary_empty_user_configs(backward):
    factory = MXFP8MOEBwdConfig if backward else MXFP8MOEFwdConfig
    reader = get_cached_moe_bwd_config if backward else get_cached_moe_fwd_config
    table = _best_config_cache._MOE_BWD_CONFIG_CACHE if backward else _best_config_cache._MOE_FWD_CONFIG_CACHE
    checked = 0
    for key in table:
        if key.quant_scheme != QuantScheme._1x32 or key.execution_mode != ExecutionMode.MLP:
            continue
        # Config-free forward BF16 uses the shared backward-natural weight layout.
        if not backward and key.input_shape.weight_scale_shapes is None:
            if key.weight_orientation != (WeightOrientation.BWD_NATURAL,) * 2:
                continue
        user = factory(execution_mode=ExecutionMode.MLP, dma_transpose_enabled=key.dma_transpose_enabled)
        assert make_moe_config_key(key.input_shape, key.weight_dtype, user, backward) == key
        result = reader(key, user)
        assert result is not None
        assert all(
            getattr(user, name) is None
            for name in (
                ("phase1_config", "phase2_config", "phase3_config", "phase4_config")
                if backward
                else ("gate_up_config", "down_config")
            )
        )
        checked += 1
    assert checked >= (4 if backward else 2)


@pytest.mark.parametrize("use_1x32,scheme", [(True, QuantScheme._1x32), (False, QuantScheme.WRAPX)])
def test_forward_phase_defaults_match_resolved_loader_scheme(use_1x32, scheme):
    config = MXFP8MOEFwdConfig(use_1x32_pe_swizzle=use_1x32)
    config.resolve_phase_defaults()
    assert config.gate_up_config.quant_scheme == scheme
    assert config.down_config.quant_scheme == scheme


@pytest.mark.parametrize("backward", [False, True])
def test_packed_key_uses_physical_weight_layout_instead_of_descriptor_default(backward):
    factory = MXFP8MOEBwdConfig if backward else MXFP8MOEFwdConfig
    shape = MoEInputShape((1024, 512), (256, 256), (64, 512), 512, weight_scale_shapes=((32, 256), (8, 512)))
    config = factory(execution_mode=ExecutionMode.MLP)
    plain = make_moe_config_key(shape, float8_e4m3fn_x4, config, backward)
    config.gate_up_weight_td = TensorDescriptor(quant_scheme=QuantScheme._1x32)
    config.down_weight_td = TensorDescriptor(quant_scheme=QuantScheme._1x32, orientation=TensorOrientation.K_BY_F)
    assert make_moe_config_key(shape, float8_e4m3fn_x4, config, backward) == plain
    assert plain.weight_orientation == (WeightOrientation.BWD_NATURAL,) * 2


@pytest.mark.parametrize("backward", [False, True])
def test_custom_phase_ratios_survive_cache_round_trip(empty_cache, backward):
    user, reader, cache, phases = _case(backward)
    key = make_moe_config_key(_shape(), nl.bfloat16, user, backward)
    cached = _cached(user, phases)
    for name in phases:
        phase = getattr(cached, name)
        phase.psum_copy_in_ratio = 0.25
        phase.strided_copy_swizzle_1x32_ratio = 0.5
    _store_cached(cache, key, cached)
    setattr(user, phases[0], copy_config(getattr(cached, phases[0])))
    result = reader(key, user)
    assert result is not None
    for name in phases:
        phase = getattr(result, name)
        assert phase.psum_copy_in_ratio == 0.25
        assert phase.strided_copy_swizzle_1x32_ratio == 0.5
        assert phase is not getattr(cached, name)
