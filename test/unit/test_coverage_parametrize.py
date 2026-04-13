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

"""Unit tests for the negative testing features in coverage_parametrized_tests."""

from enum import Enum
from test.utils.coverage_parametrized_tests import (
    BoundedRange,
    FilterResult,
    _format_param_value,
    generate_parametrized_test_case,
)

import ml_dtypes
import numpy as np


class QuantType(Enum):
    BF16 = 1
    FP16 = 2
    FP8 = 3


def test_bounded_range_integer_auto_boundaries():
    """Test automatic boundary generation for integers."""
    br = BoundedRange([1, 32, 128])
    boundaries = br.get_boundary_values()
    assert boundaries == [0, 129], f"Expected [0, 129], got {boundaries}"


def test_bounded_range_numpy_array_auto_boundaries():
    """Test automatic boundary generation for numpy arrays."""
    br = BoundedRange(np.array([1, 32, 128]))
    boundaries = br.get_boundary_values()
    assert boundaries == [0, 129], f"Expected [0, 129], got {boundaries}"


def test_bounded_range_explicit_boundaries():
    """Test explicit boundary values."""
    br = BoundedRange([128, 512], boundary_values=[513])
    assert br.get_boundary_values() == [513]


def test_bounded_range_disabled_boundaries():
    """Test disabling boundaries with empty list."""
    br = BoundedRange([64, 128], boundary_values=[])
    assert br.get_boundary_values() == []


def test_bounded_range_bool_no_auto_boundaries():
    """Test that booleans don't get automatic boundaries."""
    br = BoundedRange([True, False])
    assert br.get_boundary_values() == []


def test_bounded_range_enum_boundaries():
    """Test automatic boundary generation for enums (missing members)."""
    br = BoundedRange([QuantType.BF16, QuantType.FP8])
    boundaries = br.get_boundary_values()
    assert QuantType.FP16 in boundaries
    assert len(boundaries) == 1


def test_generate_parametrized_test_case_integration():
    """Test full integration with boundary and invalid combination tests."""

    def filter_func(a, b):
        if a + b > 10:
            return FilterResult.INVALID
        return FilterResult.VALID

    test_cases = generate_parametrized_test_case(
        params={"a": [1, 5, 10], "b": [1, 5]},
        coverage="singles",
        filter_func=filter_func,
        enable_automatic_boundary_tests=True,
        enable_invalid_combination_tests=True,
        n_tests_per_boundary_value=2,
    )

    valid_count = sum(1 for tc in test_cases if not tc.is_negative)
    boundary_count = sum(1 for tc in test_cases if tc.prefix == "boundary")
    invalid_count = sum(1 for tc in test_cases if tc.prefix == "invalid")

    assert valid_count > 0, "Should have valid tests"
    assert boundary_count > 0, "Should have boundary tests"
    assert invalid_count > 0, "Should have invalid combination tests"

    # Verify is_negative is correctly set for all test types
    for tc in test_cases:
        if tc.prefix == "":
            assert tc.is_negative is False, f"Valid test {tc.values} should have is_negative=False"
        else:
            assert tc.is_negative is True, f"Negative test {tc.values} should have is_negative=True"


def test_invalid_combination_tests_with_filter():
    """Test that invalid combination tests are generated when filter rejects combinations."""

    def my_filter(a, b, c):
        # Invalid: a=1 with b=True
        if a == 1 and b is True:
            return FilterResult.INVALID
        return FilterResult.VALID

    test_cases = generate_parametrized_test_case(
        params={
            "a": [1, 2, 3],
            "b": [True, False],
            "c": ["x", "y"],
        },
        coverage="pairs",
        filter_func=my_filter,
        enable_automatic_boundary_tests=False,
        enable_invalid_combination_tests=True,
    )

    invalid_tests = [tc for tc in test_cases if tc.prefix == "invalid"]
    valid_tests = [tc for tc in test_cases if tc.prefix == ""]
    boundary_tests = [tc for tc in test_cases if tc.prefix == "boundary"]

    assert len(valid_tests) > 0, "Should have valid tests"
    assert len(invalid_tests) > 0, "Should have invalid combination tests"
    assert len(boundary_tests) == 0, "Should have no boundary tests when disabled"

    # All invalid tests should fail the filter (my_filter returns INVALID)
    for tc in invalid_tests:
        a, b, c = tc.values
        assert my_filter(a, b, c) == FilterResult.INVALID, f"Invalid test {tc.values} should violate filter"

    # All valid tests should pass the filter
    for tc in valid_tests:
        a, b, c = tc.values
        assert my_filter(a, b, c) == FilterResult.VALID, f"Valid test {tc.values} should pass filter"


def test_redundant_combinations_excluded_from_both():
    """Test that REDUNDANT combinations are excluded from both valid and invalid tests."""

    def my_filter(a, b):
        # Invalid: a=1 with b=True
        if a == 1 and b is True:
            return FilterResult.INVALID
        # Redundant: a=3 (valid but we don't need to test it)
        if a == 3:
            return FilterResult.REDUNDANT
        return FilterResult.VALID

    test_cases = generate_parametrized_test_case(
        params={
            "a": [1, 2, 3],
            "b": [True, False],
        },
        coverage="full",
        filter_func=my_filter,
        enable_automatic_boundary_tests=False,
        enable_invalid_combination_tests=True,
    )

    invalid_tests = [tc for tc in test_cases if tc.prefix == "invalid"]
    valid_tests = [tc for tc in test_cases if tc.prefix == ""]

    # Valid tests should only have a=1 (with b=False) and a=2
    for tc in valid_tests:
        a, b = tc.values
        assert my_filter(a, b) == FilterResult.VALID, f"Valid test {tc.values} should be VALID"
        assert a != 3, f"Redundant a=3 should not appear in valid tests"

    # Invalid tests should only have a=1 with b=True
    for tc in invalid_tests:
        a, b = tc.values
        assert my_filter(a, b) == FilterResult.INVALID, f"Invalid test {tc.values} should be INVALID"
        assert a != 3, f"Redundant a=3 should not appear in invalid tests"


def test_generate_with_bounded_range_explicit():
    """Test using BoundedRange explicitly in params."""
    test_cases = generate_parametrized_test_case(
        params={
            "x": BoundedRange([1, 10], boundary_values=[0, 11]),
            "y": BoundedRange([True, False], boundary_values=[]),  # disabled
        },
        coverage="singles",
        filter_func=None,
        enable_automatic_boundary_tests=True,
        enable_invalid_combination_tests=False,
        n_tests_per_boundary_value=1,
    )

    boundary_tests = [tc for tc in test_cases if tc.prefix == "boundary"]
    # Should have boundary tests for x (0 and 11), but not for y
    assert len(boundary_tests) == 2, f"Expected 2 boundary tests, got {len(boundary_tests)}"


def test_numpy_array_params():
    """Test that numpy arrays as parameter values work correctly."""
    test_cases = generate_parametrized_test_case(
        params={
            "a": np.array([1, 2, 3]),
            "b": [10, 20],
        },
        coverage="pairs",
        filter_func=None,
        enable_automatic_boundary_tests=False,
        enable_invalid_combination_tests=False,
    )

    assert len(test_cases) > 0, "Should generate test cases"
    # Verify all values from numpy array appear
    a_values = {tc.values[0] for tc in test_cases}
    assert a_values == {1, 2, 3}, f"Expected {{1, 2, 3}}, got {a_values}"


if __name__ == "__main__":
    test_bounded_range_integer_auto_boundaries()
    print("✓ test_bounded_range_integer_auto_boundaries")

    test_bounded_range_explicit_boundaries()
    print("✓ test_bounded_range_explicit_boundaries")

    test_bounded_range_disabled_boundaries()
    print("✓ test_bounded_range_disabled_boundaries")

    test_bounded_range_bool_no_auto_boundaries()
    print("✓ test_bounded_range_bool_no_auto_boundaries")

    test_bounded_range_enum_boundaries()
    print("✓ test_bounded_range_enum_boundaries")

    test_generate_parametrized_test_case_integration()
    print("✓ test_generate_parametrized_test_case_integration")

    test_invalid_combination_tests_with_filter()
    print("✓ test_invalid_combination_tests_with_filter")

    test_redundant_combinations_excluded_from_both()
    print("✓ test_redundant_combinations_excluded_from_both")

    test_generate_with_bounded_range_explicit()
    print("✓ test_generate_with_bounded_range_explicit")

    test_numpy_array_params()
    print("✓ test_numpy_array_params")

    test_format_param_value_handles_types()
    print("✓ test_format_param_value_handles_types")

    print("\nAll tests passed!")


def test_format_param_value_handles_types():
    """Test that _format_param_value produces clean names for type objects."""
    # Type objects should use __name__ instead of str() which produces <class '...'>
    assert _format_param_value(np.float32) == "float32"
    assert _format_param_value(np.float16) == "float16"
    assert _format_param_value(ml_dtypes.bfloat16) == "bfloat16"

    # Regular values should use str()
    assert _format_param_value(42) == "42"
    assert _format_param_value("hello") == "hello"
    assert _format_param_value(3.14) == "3.14"
    assert _format_param_value(True) == "True"
