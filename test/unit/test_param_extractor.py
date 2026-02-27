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
"""
Unit tests for param_extractor module.
"""

from enum import Enum
from typing import Optional
from unittest.mock import Mock

from ..utils.param_extractor import (
    _unwrap_kernel_func,
    extract_pytest_params,
    normalize_param_value,
    normalize_params_with_type_hints,
)
from ..utils.ranged_test_harness import RangeTestCase


class SampleEnum(Enum):
    """Sample enum for testing."""

    VALUE_A = 0
    VALUE_B = 1
    VALUE_C = 2


class AnotherEnum(Enum):
    """Another enum for testing suffix matching."""

    TYPE_X = 0
    TYPE_Y = 1


class TestNormalizeParamValue:
    """Tests for normalize_param_value function."""

    def test_none_returns_none(self):
        """None input returns None."""
        assert normalize_param_value(None) is None

    def test_enum_returns_name(self):
        """Enum values are converted to their name string."""
        assert normalize_param_value(SampleEnum.VALUE_A) == "VALUE_A"
        assert normalize_param_value(SampleEnum.VALUE_B) == "VALUE_B"

    def test_empty_string_returns_none(self):
        """Empty or whitespace-only strings return None."""
        assert normalize_param_value("") is None
        assert normalize_param_value("   ") is None
        assert normalize_param_value("\t\n") is None

    def test_non_empty_string_passes_through(self):
        """Non-empty strings pass through unchanged."""
        assert normalize_param_value("hello") == "hello"
        assert normalize_param_value("  hello  ") == "  hello  "

    def test_primitives_pass_through(self):
        """Primitive types pass through unchanged."""
        assert normalize_param_value(42) == 42
        assert normalize_param_value(3.14) == 3.14
        assert normalize_param_value(True) is True
        assert normalize_param_value(False) is False

    def test_object_with_dict_converted_to_string(self):
        """Objects with __dict__ are converted to string representation."""

        class CustomObj:
            def __init__(self):
                self.value = 123

            def __str__(self):
                return "CustomObj(123)"

        obj = CustomObj()
        assert normalize_param_value(obj) == "CustomObj(123)"

    def test_list_passes_through(self):
        """Lists pass through unchanged."""
        assert normalize_param_value([1, 2, 3]) == [1, 2, 3]


class TestUnwrapKernelFunc:
    """Tests for _unwrap_kernel_func function."""

    def test_plain_function_returns_unchanged(self):
        """Plain function returns unchanged."""

        def my_func():
            pass

        assert _unwrap_kernel_func(my_func) is my_func

    def test_unwraps_func_attribute(self):
        """Objects with 'func' attribute are unwrapped."""
        original = lambda: None
        mock_kernel = Mock()
        mock_kernel.func = original

        result = _unwrap_kernel_func(mock_kernel)
        assert result is original

    def test_unwraps_wrapped_attribute(self):
        """Decorated functions with __wrapped__ are unwrapped."""
        original = lambda: None
        decorated = lambda: None
        decorated.__wrapped__ = original

        assert _unwrap_kernel_func(decorated) is original

    def test_recursive_unwrap(self):
        """Nested wrappers are unwrapped recursively."""
        original = lambda: None

        # Create nested wrapper: GenericKernel(decorated(original))
        decorated = lambda: None
        decorated.__wrapped__ = original
        mock_kernel = Mock()
        mock_kernel.func = decorated

        result = _unwrap_kernel_func(mock_kernel)
        assert result is original


class TestNormalizeParamsWithTypeHints:
    """Tests for normalize_params_with_type_hints function."""

    def test_empty_params_returns_empty(self):
        """Empty params dict returns empty dict."""

        def kernel_func(x: int):
            pass

        assert normalize_params_with_type_hints({}, kernel_func) == {}

    def test_none_kernel_func_uses_basic_normalization(self):
        """When kernel_func is None, basic normalization is applied."""
        params = {"value": SampleEnum.VALUE_A, "count": 5}
        result = normalize_params_with_type_hints(params, None)
        assert result == {"value": "VALUE_A", "count": 5}

    def test_int_converted_to_enum_name(self):
        """Integer values are converted to enum names when type hint matches."""

        def kernel_func(mode: SampleEnum):
            pass

        params = {"mode": 1}
        result = normalize_params_with_type_hints(params, kernel_func)
        assert result == {"mode": "VALUE_B"}

    def test_int_converted_to_bool(self):
        """Integer values are converted to bool when type hint is bool."""

        def kernel_func(enabled: bool):
            pass

        params = {"enabled": 1}
        result = normalize_params_with_type_hints(params, kernel_func)
        assert result == {"enabled": True}

        params = {"enabled": 0}
        result = normalize_params_with_type_hints(params, kernel_func)
        assert result == {"enabled": False}

    def test_optional_bool_converted(self):
        """Integer values are converted to bool for Optional[bool] type hints."""

        def kernel_func(flag: Optional[bool]):
            pass

        params = {"flag": 1}
        result = normalize_params_with_type_hints(params, kernel_func)
        assert result == {"flag": True}

    def test_suffix_matching_for_enums(self):
        """Enum type hints match by suffix (e.g., 'type' matches 'norm_type')."""

        def kernel_func(norm_type: SampleEnum):
            pass

        params = {"type": 0}
        result = normalize_params_with_type_hints(params, kernel_func)
        assert result == {"type": "VALUE_A"}

    def test_suffix_matching_for_bools(self):
        """Bool type hints match by suffix."""

        def kernel_func(use_bias: bool):
            pass

        params = {"bias": 1}
        result = normalize_params_with_type_hints(params, kernel_func)
        assert result == {"bias": True}

    def test_enum_value_passes_through_as_name(self):
        """Enum values are converted to their name string."""

        def kernel_func(mode: SampleEnum):
            pass

        params = {"mode": SampleEnum.VALUE_C}
        result = normalize_params_with_type_hints(params, kernel_func)
        assert result == {"mode": "VALUE_C"}

    def test_non_integer_values_pass_through(self):
        """Non-integer values pass through unchanged."""

        def kernel_func(name: str, ratio: float):
            pass

        params = {"name": "test", "ratio": 0.5}
        result = normalize_params_with_type_hints(params, kernel_func)
        assert result == {"name": "test", "ratio": 0.5}

    def test_none_values_filtered_out(self):
        """None values are filtered from the result."""

        def kernel_func(x: int):
            pass

        params = {"x": None, "y": 5}
        result = normalize_params_with_type_hints(params, kernel_func)
        assert result == {"y": 5}

    def test_invalid_enum_value_keeps_original(self):
        """Invalid enum integer values keep the original value."""

        def kernel_func(mode: SampleEnum):
            pass

        params = {"mode": 999}  # Invalid enum value
        result = normalize_params_with_type_hints(params, kernel_func)
        assert result == {"mode": 999}

    def test_type_hints_unavailable_falls_back(self):
        """Falls back to basic normalization when type hints unavailable."""
        # Lambda functions don't have type hints accessible
        kernel_func = lambda x: x
        params = {"value": SampleEnum.VALUE_A, "count": 5}
        result = normalize_params_with_type_hints(params, kernel_func)
        assert result == {"value": "VALUE_A", "count": 5}


class TestExtractPytestParams:
    """Tests for extract_pytest_params function."""

    def test_empty_params_returns_empty(self):
        """Empty params dict returns empty dict."""
        assert extract_pytest_params({}) == {}

    def test_none_values_filtered(self):
        """None values are filtered out."""
        params = {"a": 1, "b": None, "c": 3}
        result = extract_pytest_params(params)
        assert result == {"a": 1, "c": 3}

    def test_enum_converted_to_name(self):
        """Enum values are converted to their name."""
        params = {"mode": SampleEnum.VALUE_A}
        result = extract_pytest_params(params)
        assert result == {"mode": "VALUE_A"}

    def test_range_test_case_extracted(self):
        """RangeTestCase objects have their tensors and params extracted."""
        # Create a mock for test_config_ref since it's required
        mock_config = Mock()

        test_case = RangeTestCase(
            tensors={"input": {"batch": 4, "seq_len": 128}},
            test_type="unit",
            additional_params={"use_bias": True},
            test_config_ref=mock_config,
        )
        params = {"case": test_case}
        result = extract_pytest_params(params)

        assert result["batch"] == 4
        assert result["seq_len"] == 128
        assert result["test_type"] == "unit"
        assert result["use_bias"] is True

    def test_mixed_params(self):
        """Mixed parameter types are handled correctly."""
        params = {
            "count": 42,
            "name": "test",
            "mode": SampleEnum.VALUE_B,
            "empty": "",
            "flag": True,
        }
        result = extract_pytest_params(params)

        assert result["count"] == 42
        assert result["name"] == "test"
        assert result["mode"] == "VALUE_B"
        assert result["flag"] is True
        assert "empty" not in result  # Empty string filtered
