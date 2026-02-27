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

import inspect
import itertools
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from allpairspy import AllPairs

# Re-export assert_negative_test_case for convenience
from .ranged_test_harness import assert_negative_test_case as assert_negative_test_case


class FilterResult(Enum):
    """Result of a filter function evaluation.

    VALID: Combination is valid and should be tested in positive tests.
    INVALID: Combination is invalid and should be tested in negative tests.
    REDUNDANT: Combination is valid but redundant; skip in positive tests, don't use for negative tests.
    """

    VALID = "valid"
    INVALID = "invalid"
    REDUNDANT = "redundant"


@dataclass
class BoundedRange:
    """Wrapper for parameter values with optional boundary values for negative testing.

    If boundary_values is None, automatic boundary detection is applied:
    - For integers: generates min-1 and max+1
    - For enums: generates enum values not in the values list
    - For other types (bool, etc.): no automatic boundaries

    Set boundary_values=[] to explicitly disable boundary testing for a parameter.
    """

    values: List[Any]
    boundary_values: Optional[List[Any]] = None

    def get_boundary_values(self) -> List[Any]:
        """Return boundary values, computing automatically if not specified."""
        if self.boundary_values is not None:
            return self.boundary_values
        return _compute_automatic_boundaries(self.values)


def _compute_automatic_boundaries(values: List[Any]) -> List[Any]:
    """Compute automatic boundary values based on value types."""
    if len(values) == 0:
        return []

    first = values[0]

    # Integer boundaries: min-1, max+1
    if np.issubdtype(type(first), np.integer):
        min_val = min(values)
        max_val = max(values)
        return [min_val - 1, max_val + 1]

    # Enum boundaries: enum members not in values list
    if isinstance(first, Enum):
        enum_class = type(first)
        all_members = set(enum_class)
        covered = set(values)
        return list(all_members - covered)

    # Bool and other types: no automatic boundaries
    return []


@dataclass
class CoverageTestCase:
    """A test case with metadata for negative test handling."""

    values: tuple
    is_negative: bool = False
    prefix: str = ""  # "boundary", "invalid", or "" for valid tests


def _generate_boundary_tests(
    params: Dict[str, Any],
    valid_cases: List[tuple],
    n_tests_per_boundary_value: int = 3,
) -> List[CoverageTestCase]:
    """Generate boundary test cases by substituting boundary values into valid cases."""
    if not valid_cases:
        return []

    boundary_tests = []
    seen_values = (
        set()
    )  # Dedup: different base cases can produce same result when boundary value replaces the differing param

    for param_idx, param_spec in enumerate(params.values()):
        if not isinstance(param_spec, BoundedRange):
            continue

        boundary_values = param_spec.get_boundary_values()
        if not boundary_values:
            continue

        for boundary_val in boundary_values:
            # Randomly select n valid cases to modify
            selected = random.sample(valid_cases, min(n_tests_per_boundary_value, len(valid_cases)))
            for case in selected:
                # Create new case with boundary value substituted
                new_case = list(case)
                new_case[param_idx] = boundary_val
                new_values = tuple(new_case)

                if new_values in seen_values:
                    continue
                seen_values.add(new_values)

                boundary_tests.append(
                    CoverageTestCase(
                        values=new_values,
                        is_negative=True,
                        prefix="boundary",
                    )
                )

    return boundary_tests


def _generate_cases(
    params: Dict[str, Any],
    coverage: str,
    filter_func: Callable,
) -> List[tuple]:
    """Generate test cases for the given coverage strategy."""
    if coverage == "singles":
        return generate_singles(params, filter_func=filter_func)
    elif coverage == "pairs":
        return generate_pairs(params, filter_func=filter_func)
    elif coverage == "full":
        return generate_full(params, filter_func=filter_func)
    else:
        raise ValueError(f"Invalid coverage regime: {coverage}")


def _generate_valid_cases(
    params: Dict[str, Any],
    coverage: str,
    filter_func: Callable,
) -> List[tuple]:
    """Generate valid test cases for the given coverage strategy.

    Args:
        params: Parameter name to values mapping.
        coverage: Coverage strategy ("singles", "pairs", "full").
        filter_func: Function to filter valid combinations.

    Returns:
        List of test case tuples (only VALID results, not INVALID or REDUNDANT).
    """
    wrapped_filter = _make_valid_filter(filter_func, params)
    return _generate_cases(params, coverage, wrapped_filter)


def _generate_invalid_combination_tests(
    params: Dict[str, Any],
    filter_func: Callable,
    max_invalid_tests: int,
    max_sampling_attempts: int = 1000,
) -> List[CoverageTestCase]:
    """Generate test cases that violate the filter function.

    Only generates tests for combinations that return FilterResult.INVALID,
    not FilterResult.REDUNDANT (which are valid but skipped for efficiency).

    Uses random sampling instead of AllPairs to avoid exponential backtracking
    when INVALID combinations are rare in the parameter space.
    """
    if filter_func is None:
        return []

    param_names = list(params.keys())
    param_values = list(params.values())

    # Random sampling approach - much faster than AllPairs when INVALID is rare
    invalid_cases = []
    seen = set()
    attempts = 0

    while len(invalid_cases) < max_invalid_tests and attempts < max_sampling_attempts:
        attempts += 1
        # Generate random combination
        case = tuple(random.choice(v) for v in param_values)
        if case in seen:
            continue
        seen.add(case)

        # Check if it's INVALID
        case_dict = dict(zip(param_names, case))
        result = filter_func(**case_dict)
        if result == FilterResult.INVALID:
            invalid_cases.append(CoverageTestCase(values=case, is_negative=True, prefix="invalid"))

    return invalid_cases


def _extract_valid_values(params: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Extract valid values from params, unwrapping BoundedRange if present.

    Converts numpy arrays to Python lists for AllPairs compatibility.
    """
    result = {}
    for k, v in params.items():
        values = v.values if isinstance(v, BoundedRange) else v
        if isinstance(values, np.ndarray):
            values = values.tolist()
        elif not isinstance(values, list):
            raise TypeError(f"Parameter '{k}' must be a list or numpy array, got {type(values).__name__}")
        result[k] = values
    return result


def _wrap_with_bounded_range(params: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap plain lists with BoundedRange for automatic boundary detection."""
    result = {}
    for k, v in params.items():
        if isinstance(v, BoundedRange):
            result[k] = v
        else:
            result[k] = BoundedRange(values=v)
    return result


def _create_all_pairs(param_values: List[Any], filter_func=None) -> List[tuple]:
    """Create AllPairs combinations, handling None filter_func."""
    if filter_func is None:
        return list(AllPairs(param_values))
    return list(AllPairs(param_values, filter_func=filter_func))


def generate_singles(params: Dict[str, Any], filter_func=None):
    """Generate test cases ensuring each parameter value appears at least once (1-way coverage)."""
    param_names = list(params.keys())
    param_values = list(params.values())
    # 1. Zip initialization - minimal covering set without constraints
    max_len = max(len(v) for v in param_values)
    shuffled_values = param_values
    cycles = [itertools.cycle(v) for v in shuffled_values]
    singles = [[next(cycle) for cycle in cycles] for _ in range(max_len)]
    if filter_func is None:
        return singles
    # 2. Apply constraints
    filtered_singles = [case for case in singles if filter_func(case)]
    if len(params) < 2:
        return filtered_singles
    # 3. Track what we have covered vs what we need
    covered = set((k, v) for row in filtered_singles for k, v in zip(param_names, row))
    all_requirements = set((k, v) for k, values in params.items() for v in values)
    missing = all_requirements - covered

    # 4. If gaps exist, fill from AllPairs
    if missing:
        # Generate all valid pairs
        pairwise_pool = _create_all_pairs(param_values, filter_func)

        for row in pairwise_pool:
            row_coverage = set(zip(param_names, row))
            # Does this row cover any of our missing 1-way requirements?
            if row_coverage.intersection(missing):
                filtered_singles.append(list(row))
                missing -= row_coverage
            if not missing:
                break  # no more missing values

    return filtered_singles


def generate_pairs(params: Dict[str, Any], filter_func=None):
    """Generate test cases covering all parameter pairs (2-way coverage using AllPairs algorithm)."""
    if len(params) < 2:
        return generate_singles(params, filter_func)
    return _create_all_pairs(list(params.values()), filter_func)


def generate_full(params: Dict[str, Any], filter_func=None):
    """Generate complete cartesian product of all parameters (full coverage)."""
    param_values = list(params.values())
    full_product = itertools.product(*param_values)
    if filter_func is not None:
        full_product = filter(filter_func, full_product)
    return list(full_product)


def _make_partial_filter(filter_func, params: Dict[str, Any], accept_result: FilterResult):
    """Create a partial filter function that works with AllPairs algorithm.

    The AllPairs algorithm calls the filter on partial combinations during its
    internal processing (e.g., [a, b] before adding c). This wrapper returns True
    for partial combinations to allow exploration, and only evaluates the actual
    filter when all parameters are present.

    Handles two types of filter functions:
    1. Regular functions with named params: def filter(a, b, c) - checks by param names
    2. **kwargs functions: def filter(**kwargs) - checks by param count
    """
    param_names = list(params.keys())
    n_params = len(param_names)
    if filter_func is None:
        return None
    sig = inspect.signature(filter_func)

    # Check if function uses **kwargs (VAR_KEYWORD)
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    if has_var_keyword:
        # For **kwargs functions, we can't check by param names, so check by count
        def partial_filter(case):
            if len(case) < n_params:
                return True  # Allow partial combinations during generation
            case_params = dict(zip(param_names, case))
            return filter_func(**case_params) == accept_result
    else:
        # For regular functions, check required params by name
        filter_params = set(name for name in sig.parameters)
        required_params = set(name for name, param in sig.parameters.items() if param.default is param.empty)

        def partial_filter(case):
            case_params = dict(zip(param_names, case))
            if not required_params.issubset(case_params.keys()):
                return True  # Allow partial combinations during generation
            case_params = {k: case_params[k] if k in case_params else None for k in filter_params}
            return filter_func(**case_params) == accept_result

    return partial_filter


def _make_valid_filter(filter_func, params: Dict[str, Any]):
    """Create filter that accepts only VALID results."""
    return _make_partial_filter(filter_func, params, FilterResult.VALID)


def _make_invalid_filter(filter_func, params: Dict[str, Any]):
    """Create filter that accepts only INVALID results."""
    return _make_partial_filter(filter_func, params, FilterResult.INVALID)


def generate_parametrized_test_case(
    params: Dict[str, Any],
    coverage: str,
    filter_func: Callable,
    enable_automatic_boundary_tests: bool = True,
    enable_invalid_combination_tests: bool = True,
    n_tests_per_boundary_value: int = 3,
    max_invalid_tests: int = 30,
) -> List[CoverageTestCase]:
    """Generate test cases including valid, boundary, and invalid combination tests.

    Args:
        params: Parameter name to values mapping. Values can be lists or BoundedRange.
        coverage: Coverage strategy ("singles", "pairs", "full").
        filter_func: Function to filter valid combinations.
        enable_automatic_boundary_tests: If True, auto-wrap plain lists with BoundedRange.
        enable_invalid_combination_tests: If True, generate tests using negated filter.
        n_tests_per_boundary_value: Number of valid cases to modify per boundary value.
        max_invalid_tests: Maximum number of invalid combination tests to generate.

    Returns:
        List of CoverageTestCase objects with values and negative test metadata.
    """
    # Extract valid values (unwrap any BoundedRange)
    valid_params = _extract_valid_values(params)

    # Generate valid test cases
    valid_cases = _generate_valid_cases(valid_params, coverage, filter_func)

    # Convert to CoverageTestCase objects
    all_tests = [CoverageTestCase(values=tuple(c), is_negative=False, prefix="") for c in valid_cases]

    # Generate boundary tests
    params_with_bounds = _wrap_with_bounded_range(params) if enable_automatic_boundary_tests else params
    boundary_tests = _generate_boundary_tests(
        params_with_bounds,
        [tuple(c) for c in valid_cases],
        n_tests_per_boundary_value,
    )
    all_tests.extend(boundary_tests)

    # Generate invalid combination tests
    if enable_invalid_combination_tests and filter_func is not None:
        invalid_tests = _generate_invalid_combination_tests(
            valid_params,
            filter_func,
            max_invalid_tests=max_invalid_tests,
        )
        all_tests.extend(invalid_tests)

    random.shuffle(all_tests)
    return all_tests


def _format_param_value(val: Any) -> str:
    """Format a parameter value for use in test IDs.

    Handles special cases like type objects (np.float32, ml_dtypes.bfloat16)
    to produce cleaner test IDs without special characters like < > '.

    Args:
        val: Parameter value to format.

    Returns:
        String representation suitable for test IDs.
    """
    if isinstance(val, type):
        return val.__name__
    return str(val)


def extract_parametrize_args(
    params: Dict[str, Any],
    test_cases: List[CoverageTestCase],
) -> tuple[List[str], List[tuple], List[str]]:
    """Extract pytest.parametrize arguments from test cases.

    Args:
        params: Original parameter dict (for extracting param names).
        test_cases: List of CoverageTestCase objects.

    Returns:
        Tuple of (param_names, values_list, ids_list) for pytest.parametrize.
        param_names includes "is_negative_test_case" appended.
    """
    param_names = list(params.keys()) + ["is_negative_test_case"]
    values_list = []
    ids_list = []

    for tc in test_cases:
        values_list.append(tc.values + (tc.is_negative,))
        params_str = "-".join(_format_param_value(val) for val in tc.values)
        test_id = f"{tc.prefix}_{params_str}" if tc.prefix else params_str
        ids_list.append(test_id)

    return param_names, values_list, ids_list
