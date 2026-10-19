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

"""Host-side recording and faster-only updates for complete kernel configs."""

import ast
import math
import os
from dataclasses import fields, is_dataclass
from enum import Enum
from importlib import import_module
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

import nki.language as nl
from nkilib_src.nkilib.core.utils.kernel_assert import kernel_assert
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_config import MatmulMxfp8KernelConfig
from nkilib_src.nkilib.experimental.moe.bwd.moe_bwd_parameters import (
    ActFnType,
    AffinityOption,
    ClampLimits,
    ShardOption,
    SkipMode,
)
from nkilib_src.nkilib.experimental.moe_mxfp8 import _best_config_cache
from nkilib_src.nkilib.experimental.moe_mxfp8._best_config_cache import CachedConfig
from nkilib_src.nkilib.experimental.moe_mxfp8._config_cache import (
    get_cached_moe_bwd_config,
    get_cached_moe_fwd_config,
)
from nkilib_src.nkilib.experimental.moe_mxfp8._config_codec import _CONFIG_TYPES, decode_config
from nkilib_src.nkilib.experimental.moe_mxfp8._config_helpers import (
    MoEConfigKey,
    MoEInputShape,
    WeightOrientation,
    copy_config,
    input_shape_from_tensors,
    make_moe_config_key,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.bwd.blockwise_mm_backward_mxfp8 import _resolve_phase_configs
from nkilib_src.nkilib.experimental.moe_mxfp8.bwd.config import MXFP8MOEBwdConfig, TransposeMode
from nkilib_src.nkilib.experimental.moe_mxfp8.config_helpers import ExecutionMode
from nkilib_src.nkilib.experimental.moe_mxfp8.fwd.config import MXFP8MOEFwdConfig, _auto_generate_moe_fwd_configs
from nkilib_src.nkilib.experimental.moe_mxfp8.moe_mxfp8_checkpoint_config import MXFP8MOECheckpointConfig
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.common_dataclasses import (
    BlockDescriptor,
    LncShardingMode,
    LoopOrder,
    QuantScheme,
    SwizzleMode,
    TensorDescriptor,
    TensorOrientation,
)

from test.utils.metrics_collector import MetricName
from test.utils.metrics_emitter import MetricsEmitter, OutputMode

_RECORD_TYPES = {
    cls.__name__: cls
    for cls in (
        CachedConfig,
        MoEConfigKey,
        MoEInputShape,
        MXFP8MOEFwdConfig,
        MXFP8MOEBwdConfig,
        MatmulMxfp8KernelConfig,
        TensorDescriptor,
        BlockDescriptor,
        ClampLimits,
        SkipMode,
        MXFP8MOECheckpointConfig,
    )
}
_ENUM_TYPES = {
    cls.__name__: cls
    for cls in (
        ActFnType,
        AffinityOption,
        ExecutionMode,
        ShardOption,
        WeightOrientation,
        TransposeMode,
        LncShardingMode,
        LoopOrder,
        QuantScheme,
        SwizzleMode,
        TensorOrientation,
    )
}
_DTYPES = {
    name: getattr(nl, name)
    for name in (
        "bfloat16",
        "float16",
        "float32",
        "float8_e4m3fn",
        "float8_e4m3fn_x4",
        "float8_e5m2",
        "float8_e5m2_x4",
    )
}


def config_literal(value):
    """Serialize data-free records as explicit, reproducible constructors."""
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name}"
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    if isinstance(value, tuple):
        items = ", ".join(config_literal(item) for item in value)
        return "(" + items + ("," if len(value) == 1 else "") + ")"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{config_literal(key)}: {config_literal(item)}" for key, item in value.items()) + "}"
    for name, dtype in _DTYPES.items():
        if value is dtype:
            return f"nl.{name}"
    if is_dataclass(value) and type(value).__name__ in _RECORD_TYPES:
        args = ", ".join(
            f"{field.name}={config_literal(getattr(value, field.name))}" for field in fields(value) if field.init
        )
        return f"{type(value).__name__}({args})"
    raise ValueError(f"Cache configs must contain data-free records, got {type(value).__name__}")


def read_config_literal(source, constants=None, *, allow_positional=False):
    """Read only allowlisted constructors, enums, dtypes and literals."""

    def read(node):
        if isinstance(node, ast.Constant) and (node.value is None or isinstance(node.value, (bool, int, float, str))):
            return node.value
        if isinstance(node, ast.Tuple):
            return tuple(read(item) for item in node.elts)
        if isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                raise ValueError("Cache tables cannot contain unpacked mappings")
            pairs = [(read(key), read(value)) for key, value in zip(node.keys, node.values, strict=True)]
            result = dict(pairs)
            if len(result) != len(pairs):
                raise ValueError("Duplicate key in the cache literal")
            return result
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and constants is not None:
            if node.value.id in constants and isinstance(node.slice, ast.Constant) and type(node.slice.value) is int:
                index = node.slice.value
                values = constants[node.value.id]
                if 0 <= index < len(values):
                    return values[index]
                raise ValueError("Cache reference is outside its table")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            if type(node.operand.value) in (int, float):
                return -node.operand.value
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "nl" and node.attr in _DTYPES:
                return _DTYPES[node.attr]
            if node.value.id in _ENUM_TYPES:
                return _ENUM_TYPES[node.value.id][node.attr]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _RECORD_TYPES:
            cls = _RECORD_TYPES[node.func.id]
            names = tuple(field.name for field in fields(cls) if field.init)
            positional = names[: len(node.args)] if allow_positional else ()
            keywords = tuple(keyword.arg for keyword in node.keywords)
            supplied = positional + keywords
            if (
                (node.args and not allow_positional)
                or len(supplied) != len(names)
                or len(set(supplied)) != len(names)
                or set(supplied) != set(names)
            ):
                raise ValueError("Cache records must specify every field exactly once")
            args = {name: read(value) for name, value in zip(positional, node.args, strict=True)}
            args.update({keyword.arg: read(keyword.value) for keyword in node.keywords})
            return cls(**args)
        raise ValueError("Unsupported cache config expression")

    return read(ast.parse(source, mode="eval").body)


def cache_lookup_config(key, cached_config, backward=False):
    """Keep representation presets while leaving blocking for cache resolution."""
    config = copy_config(cached_config)
    phases = (
        ("phase1_config", "phase2_config", "phase3_config", "phase4_config")
        if backward
        else ("gate_up_config", "down_config")
    )
    for name in phases:
        phase = getattr(config, name)
        if not phase.enable_scale_packing or (backward and key.quant_scheme == QuantScheme.WRAPX):
            phase.M, phase.K, phase.N = 0, 0, 0
            phase.lnc_sharding = LncShardingMode.AUTO
            for field_name in (
                "tile_m",
                "tile_k",
                "tile_n",
                "TILES_IN_BLOCK_M",
                "TILES_IN_BLOCK_N",
                "TILES_IN_BLOCK_K",
                "TILES_IN_LOAD_M",
                "TILES_IN_LOAD_N",
            ):
                setattr(phase, field_name, None)
        else:
            setattr(config, name, None)
    return config


def record_cache_candidate(test_manager, inputs, backward=False):
    """Prepare exact configs for sweep recording or populated-cache validation."""
    sweep = os.environ.get("NKILIB_MXFP8_CACHE_SWEEP") == "1"
    validate = os.environ.get("NKILIB_MXFP8_CACHE_VALIDATE") == "1"
    if not sweep and not validate:
        return inputs
    if sweep and validate:
        raise ValueError("Choose either sweep recording or populated-cache validation")
    if sweep and (_best_config_cache._MOE_FWD_CONFIG_CACHE or _best_config_cache._MOE_BWD_CONFIG_CACHE):
        raise ValueError("Sweep runs require both config cache tables to be empty")
    config = copy_config(inputs["config"])
    force_pe = os.environ.get("NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE", "").lower() == "true"
    config.dma_transpose_enabled = not force_pe
    config.fast_dma_transpose = config.dma_transpose_enabled
    mlp = config.execution_mode == ExecutionMode.MLP

    def shape_fact(name, weight=False):
        tensor = inputs.get(name)
        if tensor is None:
            return None
        shape = (1, *tensor.shape) if mlp and weight else tensor.shape
        return SimpleNamespace(shape=shape)

    shape = input_shape_from_tensors(
        shape_fact("hidden_states"),
        shape_fact("gate_up_proj_weight", weight=True),
        shape_fact("down_proj_weight", weight=True),
        inputs["hidden_states"].shape[0] // 2 if mlp else inputs.get("block_size", 4096),
        config.execution_mode,
        shape_fact("token_position_to_id"),
        shape_fact("block_to_expert"),
        shape_fact("gate_up_weight_scales", weight=True),
        shape_fact("down_weight_scales", weight=True),
        shape_fact("num_valid_tokens"),
    )
    weight_dtype = tuple(_DTYPES[inputs[name].dtype.name] for name in ("gate_up_proj_weight", "down_proj_weight"))
    key = make_moe_config_key(shape, weight_dtype, config, backward)
    if validate:
        table = _best_config_cache._MOE_BWD_CONFIG_CACHE if backward else _best_config_cache._MOE_FWD_CONFIG_CACHE
        entry = table.get(key)
        if entry is None:
            raise ValueError("No measured cache entry matches the validation inputs")
        cached = decode_config(entry[1], _best_config_cache._CONFIG_SCHEMAS, _best_config_cache._CONFIG_RECORDS)
        config = cache_lookup_config(key, cached, backward)
        reader = get_cached_moe_bwd_config if backward else get_cached_moe_fwd_config
        if reader(key, config) is None:
            raise ValueError("The populated entry conflicts with its validation presets")
        direction = "bwd" if backward else "fwd"
        module = import_module(
            f"nkilib_src.nkilib.experimental.moe_mxfp8.{direction}.blockwise_mm_"
            f"{'backward' if backward else 'forward'}_mxfp8"
        )

        def checked_reader(key, user_config):
            result = reader(key, user_config)
            kernel_assert(result != None, "Populated-cache validation requires a compiled cache hit")  # noqa: E711
            return result

        setattr(module, f"get_cached_moe_{direction}_config", checked_reader)
        inputs["config"] = config
        params = dict(test_manager.collector.get_kernel_params())
        params["cache_validation"] = {"key": config_literal(key), "cached_seconds": entry[0]}
        test_manager.collector.set_kernel_params(params)
        return inputs
    H = inputs["hidden_states"].shape[1]
    B = shape.block_size
    if backward:
        I = inputs["gate_up_proj_act_checkpoint_T"].shape[-1]
        _resolve_phase_configs(
            config,
            (
                (B, H, I // 2),
                (B, 2 * I, H // 2),
                (H // 2, B, 2 * I if config.execution_mode.is_dense else I),
                (I, B, H // 2),
            ),
            H,
            True,
        )
    else:
        if config.execution_mode.is_dense and inputs["hidden_states"].shape[0] <= 2 * B:
            # With at most one block per core, this flag keeps its default and
            # does not change the number of weight loads.
            config.reuse_spilled_weights = True
        if inputs.get("down_weight_scales") is not None:
            I = shape.down_weight_shape[-2] * 4
        elif config.down_weight_td is None:
            I = shape.down_weight_shape[-2] if mlp else shape.down_weight_shape[-1]
        elif config.down_weight_td.orientation == TensorOrientation.F_BY_K:
            I = shape.down_weight_shape[-1]
        else:
            I = shape.down_weight_shape[-2]
        _auto_generate_moe_fwd_configs(config, B, H, I)
        config.gate_up_config.quant_scheme = key.quant_scheme
        config.down_config.quant_scheme = key.quant_scheme
    inputs["config"] = config
    params = dict(test_manager.collector.get_kernel_params())
    params["cache_candidate"] = {
        "direction": "backward" if backward else "forward",
        "key": config_literal(key),
        "config": config_literal(config),
    }
    test_manager.collector.set_kernel_params(params)
    return inputs


def emit_cache_candidate(test_manager):
    """Retain measured cache records after successful numerical validation."""
    if os.environ.get("NKILIB_MXFP8_CACHE_SWEEP") != "1" and os.environ.get("NKILIB_MXFP8_CACHE_VALIDATE") != "1":
        return
    collector = test_manager.collector
    latency = collector.get_metric(MetricName.ACTIVE_INFERENCE_TIME)
    validation_time = collector.get_metric(MetricName.VALIDATION_TIME)
    if collector.dimensions.get("Status") != "SUCCESS" or validation_time is None or validation_time <= 0:
        return
    if latency is None or not math.isfinite(latency) or latency <= 0:
        raise ValueError("Validated sweep candidates must have a positive ActiveInferenceTime")
    dimensions = collector.dimensions
    # Optional fleet metadata can be blank; EMF rejects empty dimensions.
    collector.dimensions = {name: value for name, value in dimensions.items() if value and value.strip()}
    try:
        MetricsEmitter(OutputMode.FILE).emit(collector)
    finally:
        collector.dimensions = dimensions


def select_cache_updates(records, forward_cache, backward_cache):
    """Accept validated measurements only when they beat the same cached key."""
    tables = {"forward": dict(forward_cache), "backward": dict(backward_cache)}
    updates = []
    for record in records:
        candidate = record.get("cache_candidate")
        latency = record.get("ActiveInferenceTime")
        if not candidate or record.get("Status") != "SUCCESS":
            continue
        if type(latency) not in (int, float) or not math.isfinite(latency) or latency <= 0:
            continue
        direction = candidate["direction"]
        if direction not in tables:
            raise ValueError(f"Unsupported cache pass: {direction}")
        key = read_config_literal(candidate["key"])
        config = read_config_literal(candidate["config"])
        backward = direction == "backward"
        expected_type = MXFP8MOEBwdConfig if backward else MXFP8MOEFwdConfig
        if not isinstance(key, MoEConfigKey) or not isinstance(config, expected_type):
            raise ValueError("Cache key or config has the wrong type for this pass")
        if make_moe_config_key(key.input_shape, key.weight_dtype, config, backward) != key:
            raise ValueError("Measured config does not match its cache key")
        phases = (
            (config.phase1_config, config.phase2_config, config.phase3_config, config.phase4_config)
            if backward
            else (config.gate_up_config, config.down_config)
        )
        required = (
            "M",
            "K",
            "N",
            "tile_m",
            "tile_k",
            "tile_n",
            "TILES_IN_BLOCK_M",
            "TILES_IN_BLOCK_N",
            "TILES_IN_BLOCK_K",
            "TILES_IN_LOAD_M",
            "TILES_IN_LOAD_N",
        )
        if any(
            phase is None or any(getattr(phase, name) is None or getattr(phase, name) <= 0 for name in required)
            for phase in phases
        ):
            raise ValueError("Measured cache configs must contain complete phase configs")
        table = tables[direction]
        previous = table.get(key)
        if previous is not None and (
            type(previous.active_inference_time) not in (int, float)
            or not math.isfinite(previous.active_inference_time)
            or previous.active_inference_time <= 0
        ):
            raise ValueError("Cached ActiveInferenceTime must be finite and positive")
        if previous is not None and latency >= previous.active_inference_time:
            continue
        table[key] = CachedConfig(copy_config(config), float(latency))
        updates.append(
            {
                "direction": direction,
                "key": candidate["key"],
                "previous": None if previous is None else previous.active_inference_time,
                "ActiveInferenceTime": float(latency),
                "test": record.get("TestName"),
            }
        )
    return tables, updates


def pack_cache_tables(tables):
    """Deduplicate records and store measured common fields once per type."""
    grouped = {}

    def collect(value):
        if is_dataclass(value):
            if type(value).__name__ not in _CONFIG_TYPES:
                raise ValueError("Cache configs must contain allowlisted data-free records")
            grouped.setdefault(type(value).__name__, []).append(value)
            for field in fields(value):
                if field.init:
                    collect(getattr(value, field.name))
        elif isinstance(value, tuple):
            for item in value:
                collect(item)
        else:
            config_literal(value)

    ordered = {
        direction: sorted(table.items(), key=lambda item: config_literal(item[0]))
        for direction, table in tables.items()
    }
    for entries in ordered.values():
        for _, entry in entries:
            collect(entry.config)
    raw_schemas = {}
    for name, instances in grouped.items():
        fixed, varying = {}, []
        for field in fields(instances[0]):
            if field.init:
                value = getattr(instances[0], field.name)
                if all(getattr(instance, field.name) == value for instance in instances):
                    fixed[field.name] = value
                else:
                    varying.append(field.name)
        raw_schemas[name] = fixed, tuple(varying)

    schemas, record_list, indexes = {}, [], {}

    def encode(value):
        if is_dataclass(value):
            name = type(value).__name__
            fixed, varying = raw_schemas[name]
            if name not in schemas:
                schemas[name] = ({}, varying)
                schemas[name] = ({field: encode(item) for field, item in fixed.items()}, varying)
            record = (name, tuple(encode(getattr(value, field)) for field in varying))
            if record not in indexes:
                indexes[record] = len(record_list)
                record_list.append(record)
            return "ref", indexes[record]
        if isinstance(value, tuple):
            if len(value) == 2 and value[0] == "ref":
                raise ValueError("Config tuples cannot use the reserved record reference tag")
            return tuple(encode(item) for item in value)
        return value

    packed, shapes = {}, []
    for direction, entries in ordered.items():
        packed[direction] = {}
        for key, entry in entries:
            if key.input_shape not in shapes:
                shapes.append(key.input_shape)
            packed[direction][key] = entry.active_inference_time, encode(entry.config)[1]
    records = tuple(record_list)
    for direction, table in packed.items():
        for key, (_, index) in table.items():
            if decode_config(index, schemas, records) != tables[direction][key].config:
                raise ValueError("Packed cache did not reconstruct the complete measured config")
    return schemas, records, tuple(shapes), packed


def _cache_nodes(source):
    names = (
        "_CONFIG_SCHEMAS",
        "_CONFIG_RECORDS",
        "_INPUT_SHAPES",
        "_MOE_FWD_CONFIG_CACHE",
        "_MOE_BWD_CONFIG_CACHE",
    )
    nodes = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in names:
            if node.target.id in nodes:
                raise ValueError(f"Expected one literal table named {node.target.id}")
            nodes[node.target.id] = node
    for name in names[-2:]:
        if name not in nodes or not isinstance(nodes[name].value, ast.Dict):
            raise ValueError(f"Expected one literal table named {name}")
        if any(key is None for key in nodes[name].value.keys):
            raise ValueError("Cache tables cannot contain unpacked mappings")
    if set(nodes) != set(names):
        raise ValueError("Expected compact cache schemas, records and input shapes")
    return nodes


def _validate_cache_records(schemas, records):
    def check_refs(value, index):
        if isinstance(value, tuple):
            if len(value) == 2 and value[0] == "ref":
                if type(value[1]) is not int or not 0 <= value[1] < index:
                    raise ValueError("Cache references must point to earlier records")
            else:
                for item in value:
                    check_refs(item, index)

    if not isinstance(schemas, dict) or not isinstance(records, tuple):
        raise ValueError("Invalid compact cache schemas or records")
    for name, schema in schemas.items():
        if name not in _CONFIG_TYPES or not isinstance(schema, tuple) or len(schema) != 2:
            raise ValueError("Unsupported compact config schema")
        fixed, varying = schema
        names = {field.name for field in fields(_CONFIG_TYPES[name]) if field.init}
        if (
            not isinstance(fixed, dict)
            or not isinstance(varying, tuple)
            or len(set(varying)) != len(varying)
            or set(fixed) & set(varying)
            or set(fixed) | set(varying) != names
        ):
            raise ValueError("Cache schemas must specify every field exactly once")
    for index, record in enumerate(records):
        if not isinstance(record, tuple) or len(record) != 2 or record[0] not in schemas:
            raise ValueError("Unsupported compact config record")
        fixed, varying = schemas[record[0]]
        values = record[1]
        if not isinstance(values, tuple) or len(values) != len(varying):
            raise ValueError("Cached record does not match its field schema")
        for value in (*fixed.values(), *values):
            check_refs(value, index)


def read_cache_tables(path):
    """Read and reconstruct compact literals without executing the cache module."""
    source = Path(path).read_text()
    nodes = _cache_nodes(source)

    def read(name, constants=None):
        return read_config_literal(ast.get_source_segment(source, nodes[name].value), constants, allow_positional=True)

    schemas, records = read("_CONFIG_SCHEMAS"), read("_CONFIG_RECORDS")
    _validate_cache_records(schemas, records)
    shapes = read("_INPUT_SHAPES")
    if not isinstance(shapes, tuple) or any(not isinstance(shape, MoEInputShape) for shape in shapes):
        raise ValueError("Cache shape profiles must contain exact MoEInputShape records")
    tables = {}
    for direction, name in (("forward", "_MOE_FWD_CONFIG_CACHE"), ("backward", "_MOE_BWD_CONFIG_CACHE")):
        table = {}
        for key, entry in read(name, {"_INPUT_SHAPES": shapes}).items():
            if (
                not isinstance(key, MoEConfigKey)
                or not isinstance(entry, tuple)
                or len(entry) != 2
                or type(entry[1]) is not int
                or not 0 <= entry[1] < len(records)
            ):
                raise ValueError("Invalid exact cache key or config reference")
            config = decode_config(entry[1], schemas, records)
            expected_type = MXFP8MOEBwdConfig if direction == "backward" else MXFP8MOEFwdConfig
            if not isinstance(config, expected_type):
                raise ValueError("Cache config has the wrong type for this pass")
            table[key] = CachedConfig(config, entry[0])
        tables[direction] = table
    return source, tables


def write_cache_tables(path, original_source, tables):
    """Write compact measured records atomically after checking reconstruction."""
    path = Path(path)
    tree = ast.parse(original_source)
    nodes = _cache_nodes(original_source)
    schemas, records, shapes, packed = pack_cache_tables(tables)
    literals = {
        "_CONFIG_SCHEMAS": config_literal(schemas),
        "_CONFIG_RECORDS": "(\n" + "".join(f"    {config_literal(record)},\n" for record in records) + ")",
        "_INPUT_SHAPES": "(\n"
        + "".join(
            "    MoEInputShape("
            + ", ".join(config_literal(getattr(shape, field.name)) for field in fields(shape) if field.init)
            + "),\n"
            for shape in shapes
        )
        + ")",
    }
    for direction, name in (("forward", "_MOE_FWD_CONFIG_CACHE"), ("backward", "_MOE_BWD_CONFIG_CACHE")):
        entries = []
        for key, entry in packed[direction].items():
            values = ", ".join(
                config_literal(getattr(key, field.name))
                for field in fields(key)
                if field.init and field.name != "input_shape"
            )
            entries.append(
                f"    MoEConfigKey(_INPUT_SHAPES[{shapes.index(key.input_shape)}], {values}): {config_literal(entry)},\n"
            )
        literals[name] = "{\n" + "".join(entries) + "}"
    used_names = {
        node.id
        for literal in literals.values()
        for node in ast.walk(ast.parse(literal, mode="eval"))
        if isinstance(node, ast.Name)
    }
    imports = {
        "..matmul_mxfp8.matmul_mxfp8_config": {"MatmulMxfp8KernelConfig"},
        "..moe.bwd.moe_bwd_parameters": {
            "ActFnType",
            "AffinityOption",
            "ClampLimits",
            "ShardOption",
            "SkipMode",
        },
        "..mxfp_utils.mxfp8_utils.common_dataclasses": {
            "BlockDescriptor",
            "LncShardingMode",
            "LoopOrder",
            "QuantScheme",
            "SwizzleMode",
            "TensorDescriptor",
            "TensorOrientation",
        },
        ".bwd.config": {"MXFP8MOEBwdConfig", "TransposeMode"},
        "._config_helpers": {"MoEConfigKey", "MoEInputShape", "WeightOrientation"},
        ".config_helpers": {"ExecutionMode"},
        ".fwd.config": {"MXFP8MOEFwdConfig"},
        ".moe_mxfp8_checkpoint_config": {"MXFP8MOECheckpointConfig"},
    }
    imported_names = {
        alias.asname or alias.name for node in tree.body if isinstance(node, ast.ImportFrom) for alias in node.names
    }
    extra_imports = []
    for module, names in imports.items():
        needed = sorted((used_names & names) - imported_names)
        if needed:
            extra_imports.append(f"from {module} import {', '.join(needed)}\n")
    lines = original_source.splitlines(keepends=True)
    replacements = [(node.value.lineno - 1, node.value.end_lineno, literals[name]) for name, node in nodes.items()]
    for start, end, literal in sorted(replacements, reverse=True):
        prefix = lines[start][: lines[start].index("=") + 1]
        lines[start:end] = [f"{prefix} {literal}\n"]
    last_import = max(node.end_lineno for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom)))
    lines[last_import:last_import] = extra_imports
    source = "".join(lines)
    ast.parse(source)
    if path.read_text() != original_source:
        raise ValueError("The cache changed during the update; read it again before retrying")
    with NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as temp:
        temp.write(source)
        temp_path = Path(temp.name)
    try:
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)
