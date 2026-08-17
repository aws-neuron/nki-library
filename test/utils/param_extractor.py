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
import hashlib
import json
import typing
from enum import Enum

# Maps variant parameter names → canonical name for metrics normalization.
# Only includes clear aliases of the same concept. Kernel-specific params
# (S_tkg, S_ctx, BxS) are intentionally excluded — they carry distinct semantics.
# Head count names normalize to n_* prefix (shorter, more common).
# Note: this only normalizes pytest param names that flow into OpenSearch,
# not kernel API keys like "num_q_heads" in kernel_input dicts.
CANONICAL_PARAM_NAMES: dict[str, str] = {
    # Batch dimension
    "batch_size": "batch",
    "bs": "batch",
    "B": "batch",
    # Sequence dimension
    "seq_len": "seqlen",
    "sequence_length": "seqlen",
    "S": "seqlen",
    # Hidden dimension
    "hidden_size": "hidden",
    "hidden_dim": "hidden",
    "H": "hidden",
    # Intermediate dimension
    "I": "intermediate",
    # Tokens dimension
    "T": "tokens",
    # Expert dimension
    "E": "expert",
    # Dtype aliases
    "dtype_str": "dtype",
    "quantization_type": "quant_type",
    "q_dtype": "quant_dtype",
    # Q heads — consolidate to n_q_heads
    "num_q_heads": "n_q_heads",
    "q_head": "n_q_heads",
    "q_heads": "n_q_heads",
    # KV heads — consolidate to n_kv_heads
    "num_kv_heads": "n_kv_heads",
    "nkv_heads": "n_kv_heads",
    # Generic heads
    "num_heads": "n_heads",
}


def _unwrap_kernel_func(kernel_func: typing.Callable) -> typing.Callable:
    """Unwrap NKI kernel objects and decorated functions to get the original function.

    NKI kernels are wrapped in GenericKernel objects. We need the original
    function to access type hints.
    """
    # Handle NKI GenericKernel objects - they have a 'func' attribute
    if hasattr(kernel_func, "func"):
        return _unwrap_kernel_func(kernel_func.func)

    # Handle decorated functions with __wrapped__
    if hasattr(kernel_func, "__wrapped__"):
        return _unwrap_kernel_func(kernel_func.__wrapped__)

    return kernel_func


def normalize_param_value(value):
    """Normalize a parameter value for JSON serialization.

    Converts Enums to their name, complex objects to strings.
    Returns None for None/empty/whitespace-only values.
    """
    if value is None:
        return None
    # Convert enums to their name (e.g., NormType.RMS_NORM -> "RMS_NORM")
    if isinstance(value, Enum):
        return value.name
    # Convert type objects to their name (e.g., nl.bfloat16 -> "bfloat16")
    if isinstance(value, type):
        return value.__name__
    # For lists/tuples: keep homogeneous arrays as native arrays so OpenSearch
    # maps them correctly. Only stringify mixed-type arrays to avoid mapping conflicts.
    if isinstance(value, (list, tuple)):
        types = {type(v) for v in value if v is not None}
        if len(types) <= 1:
            return list(value)
        return json.dumps(value)
    # Handle other non-serializable types (objects with __dict__)
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return str(value)
    # Handle empty strings or whitespace-only strings
    if isinstance(value, str) and not value.strip():
        return None
    return value


def extract_pytest_params(params: dict) -> dict:
    """Extract and serialize pytest parametrized values for metrics."""
    result = {}
    for key, value in params.items():
        if value is None:
            continue
        normalized = normalize_param_value(value)
        if normalized is not None:
            result[key] = normalized
    return result


def normalize_param_names(params: dict) -> dict:
    """Normalize parameter names to canonical forms for consistent metrics.

    Maps variant names (e.g., "batch_size", "seq_len") to their canonical
    equivalents ("batch", "seqlen") so OpenSearch always receives consistent
    field names regardless of which test file emitted the metric.

    Only renames keys that have a mapping in CANONICAL_PARAM_NAMES.
    If two source keys map to the same canonical name, the last one wins.
    """
    return {CANONICAL_PARAM_NAMES.get(k, k): v for k, v in params.items()}


def _stable_param_value(value):
    """Serialize a parameter value for stable hashing.

    Prioritizes stability over readability:
    - Enums use .value (integer) rather than .name (string) since values are
      part of the API contract and won't change if a member is renamed.
    - Numpy scalars are converted to Python native types.
    - Lists/tuples/dicts are recursively normalized.
    - Objects with __dict__ are decomposed into sorted attribute dicts.
    - Raises TypeError for unsupported types to force explicit handling.
    """
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, (list, tuple)):
        return [_stable_param_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _stable_param_value(v) for k, v in sorted(value.items())}
    # Handle sets by sorting for deterministic ordering
    if isinstance(value, (set, frozenset)):
        return sorted((_stable_param_value(v) for v in value), key=repr)
    # Handle numpy scalars (int64, float32, etc.)
    if hasattr(value, "item"):
        return value.item()
    # Decompose dataclasses and other objects into their attributes
    if hasattr(value, "__dict__"):
        return {k: _stable_param_value(v) for k, v in sorted(vars(value).items())}
    raise TypeError(
        f"Unsupported parameter type for hashing: {type(value).__name__}. Add explicit handling in _stable_param_value."
    )


def compute_params_hash(params: dict) -> str:
    """Compute a stable hash from raw pytest callspec params.

    Serializes all values (including None) in sorted key order to produce
    a deterministic fingerprint. This is used as part of the permutation
    identity key for regression tracking.

    Unlike extract_pytest_params, this preserves None values to avoid
    collisions between parametrizations that differ only by a None value.
    Uses _stable_param_value which prefers enum .value over .name for
    stability against member renames.
    """
    serialized = {}
    for key in sorted(params.keys()):
        serialized[key] = _stable_param_value(params[key])
    canonical = json.dumps(serialized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def derive_test_method_id(node) -> str:
    """Derive a stable test method identifier from a pytest node.

    Returns a string in the format module::class::method (or module::method
    if the test is not inside a class). This identifies the test code being
    executed, independent of parametrize values or pytest ID formatting.
    """
    module_name = node.module.__name__ if node.module else ""
    class_name = node.cls.__name__ if node.cls else ""
    method_name = node.originalname
    if class_name:
        return f"{module_name}::{class_name}::{method_name}"
    return f"{module_name}::{method_name}"
