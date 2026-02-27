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
import typing
from enum import Enum

from .ranged_test_harness import RangeTestCase


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
    # Handle other non-serializable types (objects with __dict__)
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return str(value)
    # Handle empty strings or whitespace-only strings
    if isinstance(value, str) and not value.strip():
        return None
    return value


def normalize_params_with_type_hints(params: dict, kernel_func: typing.Callable | None) -> dict:
    """Normalize param values using kernel function's type hints.

    Converts integer values to enum names when the corresponding kernel arg
    has an Enum type hint. This ensures consistent string representation
    regardless of whether params came from sweep tests (integers) or unit tests (enums).

    Uses suffix matching to handle cases where param name (e.g., "norm_type")
    doesn't exactly match kernel arg name (e.g., "fused_norm_type").

    If kernel_func is None, falls back to basic normalization via normalize_param_value.
    """
    if not params:
        return params

    # If no kernel func provided, just do basic normalization
    if not kernel_func:
        return {k: normalize_param_value(v) for k, v in params.items() if normalize_param_value(v) is not None}

    # Unwrap NKI kernel objects to get the original function
    unwrapped_func = _unwrap_kernel_func(kernel_func)

    try:
        hints = typing.get_type_hints(unwrapped_func)
    except Exception:
        # If we can't get type hints, fall back to basic normalization
        return {k: normalize_param_value(v) for k, v in params.items() if normalize_param_value(v) is not None}

    # Collect all enum and bool type hints from kernel function
    # Handle both direct types (bool) and Optional types (Optional[bool])
    enum_hints: dict[str, type] = {}
    bool_hints: set[str] = set()
    for arg_name, hint in hints.items():
        if isinstance(hint, type) and issubclass(hint, Enum):
            enum_hints[arg_name] = hint
        elif hint is bool:
            bool_hints.add(arg_name)
        # Handle Optional[bool] - check if it's typing.Optional[bool] or typing.Union[bool, None]
        elif hasattr(hint, "__origin__") and hint.__origin__ is typing.Union:
            args = getattr(hint, "__args__", ())
            if bool in args:
                bool_hints.add(arg_name)

    normalized = {}
    for key, value in params.items():
        # First apply basic normalization
        basic_normalized = normalize_param_value(value)
        if basic_normalized is None:
            continue

        # If basic normalization already converted enum to string, use it
        if isinstance(value, Enum):
            normalized[key] = basic_normalized
            continue

        # Handle non-integer values (strings, floats, lists, etc.) - pass through
        if not isinstance(value, int):
            normalized[key] = basic_normalized
            continue

        # Check for boolean type hint match (exact or suffix)
        is_bool_param = False
        # Get the ending part after first underscore for partial matching
        # e.g., 'fused_add' -> '_add' to match 'fused_residual_add'
        key_suffix = "_" + key.split("_")[-1] if "_" in key else None
        key_lower = key.lower()
        for arg_name in bool_hints:
            arg_lower = arg_name.lower()
            # Case-insensitive matching: exact match, suffix match, or ending suffix match
            if (
                arg_lower == key_lower
                or arg_lower.endswith("_" + key_lower)
                or key_lower.endswith("_" + arg_lower)
                or (
                    key_suffix
                    and arg_lower.endswith(key_suffix.lower())
                    and arg_lower.startswith(key.split("_")[0].lower())
                )
            ):
                is_bool_param = True
                break
        if is_bool_param:
            normalized[key] = bool(value)
            continue

        # Find matching enum hint by exact match or suffix match
        matched_hint = None
        for arg_name, hint in enum_hints.items():
            arg_lower = arg_name.lower()
            # Case-insensitive matching for enums too
            if arg_lower == key_lower or arg_lower.endswith("_" + key_lower) or key_lower.endswith("_" + arg_lower):
                matched_hint = hint
                break

        if matched_hint:
            try:
                normalized[key] = matched_hint(value).name
            except ValueError:
                normalized[key] = basic_normalized  # Keep original if conversion fails
        else:
            normalized[key] = basic_normalized

    return normalized


def extract_pytest_params(params: dict) -> dict:
    """Extract and serialize pytest parametrized values for metrics."""
    result = {}
    for key, value in params.items():
        if value is None:
            continue
        # Handle RangeTestCase objects by extracting tensor dimensions
        if isinstance(value, RangeTestCase):
            for tensor_name, dims in value.tensors.items():
                for dim_name, dim_value in dims.items():
                    normalized = normalize_param_value(dim_value)
                    if normalized is not None:
                        result[dim_name] = normalized
            result["test_type"] = value.test_type
            for k, v in value.additional_params.items():
                normalized = normalize_param_value(v)
                if normalized is not None:
                    result[k] = normalized
            continue
        normalized = normalize_param_value(value)
        if normalized is not None:
            result[key] = normalized
    return result
