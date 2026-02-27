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
import collections
import contextvars
import logging
import math
import os
import random
import sys
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Optional, Union, final

from typing_extensions import override

# Context variable to track if we're in a negative test case
# This allows Orchestrator.execute() to automatically detect negative tests
_is_negative_test_context: contextvars.ContextVar[bool] = contextvars.ContextVar('is_negative_test', default=False)

developer_logging_format = '%(asctime)s %(levelname)s %(process)d [%(name)s]: %(message)s'
logging_datefmt = '%Y-%m-%dT%H:%M:%SZ'


def not_empty(list_like, name):
    assert list_like is not None and len(list_like) > 0, f"""{name} can't be empty"""


def find(list_like: Iterable[Any], comparator: Callable[[Any], bool]) -> Union[Any, None]:
    for v in list_like:
        if comparator(v):
            return v
    return None


@dataclass
class DimensionRangeConfig:
    name: str
    max: int = sys.maxsize
    min: int = 1
    max_is_binding: bool = True
    min_is_binding: bool = True
    multiple_of: Union[int, None] = None
    power_of: Union[int, None] = None

    def __post_init__(
        self,
    ) -> None:
        self.__validate()

    def __validate(self):
        assert (
            self.multiple_of is None or self.power_of is None
        ), "Either multiple_of or power_of can to be provided, not both at the same time"

    @override
    def __str__(self) -> str:
        return {
            "max": self.max,
            "min": self.min,
            "name": self.name,
            "max_is_binding": self.max_is_binding,
            "min_is_binding": self.min_is_binding,
            "multiple_of": self.multiple_of,
            "power_of": self.power_of,
        }.__str__()


@dataclass
class TensorConfig:
    dimensions: list[DimensionRangeConfig]


@dataclass
class TensorRangeConfig:
    tensor_configs: dict[str, TensorConfig]
    random_sample_size: int = 1
    monotonic_step_size: Union[int, None] = None
    monotonic_step_percent: Union[int, None] = None
    custom_generators: Union[list[object], None] = None

    def __post_init__(
        self,
    ) -> None:
        self.__validate()

    def __validate(self):
        assert (
            self.monotonic_step_percent is not None or self.monotonic_step_size is not None
        ), "either monotonic set size or percent has to be provided"

        if self.monotonic_step_percent is not None:
            assert (
                0 < self.monotonic_step_percent <= 100
            ), "invalid monotonic_step_percent provided, has to be between [0, 100)"


@dataclass
class RangeTestConfig:
    """Describes a single range test configuration (usually passed to RangeTestHarness) that consists of multiple tensors.
    The global_tensor_configs has to be provided, which describes an overall range that is supposed to be covered.

    Optionally, when greater control over tensor ranges is desired (e.g. varying step sizes for dimensions at specific ranges or performance targets), subdiving global range into smaller sub-ranges.
    All sub-ranges for every respective dimension/tensor combination *has* to add up to the global range for that tensor.

    global_tensor_configs: (required)
    additional_params: additional metadata associated with a test case, will be passed to test generator unchanged (for the time being)
    subrange_tensor_configs: (optional)
    """

    global_tensor_configs: TensorRangeConfig
    additional_params: dict[str, Any]
    subrange_tensor_configs: Union[list[TensorRangeConfig], None] = None

    def __post_init__(
        self,
    ) -> None:
        self.__validate()

    def __validate(self):
        assert self.global_tensor_configs is not None, "global tensor config has to be provided"
        self.__confirm_subranges_fill_global_range_with_no_gaps()

    def __confirm_subranges_fill_global_range_with_no_gaps(self):
        if self.subrange_tensor_configs is None:
            return

        tensor_dim_configs: dict[str, dict[str, list[DimensionRangeConfig]]] = defaultdict(
            lambda: defaultdict(lambda: list())
        )
        global_dim_map: dict[str, dict[str, DimensionRangeConfig]] = defaultdict(lambda: defaultdict())
        for c in self.subrange_tensor_configs:
            for tensor_name, tc in c.tensor_configs.items():
                for dc in tc.dimensions:
                    tensor_dim_configs[tensor_name][dc.name].append(dc)
                    global_dim = find(
                        self.global_tensor_configs.tensor_configs[tensor_name].dimensions,
                        lambda d: d.name == dc.name,
                    )
                    assert global_dim is not None
                    global_dim_map[tensor_name][dc.name] = global_dim

        for tensor_name, tc in tensor_dim_configs.items():
            for dim_name, dim_configs in tc.items():
                intervals = self.__merge_sub_dims(sorted(dim_configs, key=lambda el: el.min))
                global_dim = self.global_tensor_configs.tensor_configs[tensor_name].dimensions
                assert (
                    len(intervals)
                    == 1  # if all subranges are overlapping, we would have only a single resulting dimension range
                    and intervals[0].min == global_dim_map[tensor_name][dim_name].min
                    and intervals[0].max == global_dim_map[tensor_name][dim_name].max
                ), f"subranges don't cover entire global range for tensor '{tensor_name}' dimension '{dim_name}'. Wanted {global_dim_map[tensor_name][dim_name]} but got intervals of {self.__print_dimensions(intervals)}"

    def __print_dimensions(self, dims: list[DimensionRangeConfig]) -> str:
        result = ""

        for dim in dims:
            result += f"\t{dim},\n"

        return f"[\n{result}]\n"

    def __merge_sub_dims(self, sub_dim: list[DimensionRangeConfig]) -> list[DimensionRangeConfig]:
        """Given a list of dimension ranges, this method attempts to merge overlapping ranges to construct a larges contiguous unified range.

        By this logic, expect to recieve a list of size 1 when all ranges overlap and a list of greater length when there is at least 2 non-overlapping
        dimension ranges.

        Args:
            sub_dim: list of dimensions to merge based on thier ranges

        Returns:

        """
        result: deque[DimensionRangeConfig] = collections.deque()
        is_first_iter = True

        for dim in sub_dim:
            if is_first_iter:
                is_first_iter = False
                result.appendleft(dim)
            else:
                last_el = result.popleft()
                if dim.min <= last_el.max:
                    min_el = dim if dim.min < last_el.min else last_el
                    max_el = dim if dim.max > last_el.max else last_el
                    result.appendleft(
                        DimensionRangeConfig(
                            name=last_el.name,
                            min=min_el.min,
                            min_is_binding=min_el.min_is_binding,
                            max=max_el.max,
                            max_is_binding=max_el.max_is_binding,
                        )
                    )
                else:
                    result.appendleft(last_el)
                    result.appendleft(dim)

        return list(result)


@dataclass(frozen=True)
class RangeTestCase:
    test_type: str
    additional_params: dict[str, Any]
    tensors: dict[str, dict[str, int]]
    test_config_ref: TensorRangeConfig
    is_negative_test_case: bool = False

    def dedup_key(self) -> str:
        """Generate deduplication key based ONLY on tensor dimensions (not test_type).

        This enables cross-strategy deduplication while preserving test_type in display names.
        """
        key = ""
        for tensor_name, dim_configs in self.tensors.items():
            key += f"_{tensor_name}_"
            for name, value in sorted(dim_configs.items()):  # Sort for consistency
                key += f"_{name}-{value}"
        return key

    def unique_id(self) -> str:
        """Generate unique test ID including test_type for display purposes."""
        unique_id = ""

        for tensor_name, dim_configs in self.tensors.items():
            unique_id += f"_{tensor_name}_"
            for name, value in dim_configs.items():
                # convert Boolean and Enum to int to save character
                if isinstance(value, bool):
                    value = int(value)
                elif isinstance(value, Enum):
                    value = value.value

                unique_id += f"_{name}-{value}"
        pytest_test_name = ""
        if os.environ.__contains__("PYTEST_CURRENT_TEST"):
            pytest_test_name = os.environ["PYTEST_CURRENT_TEST"].split("::")[-1].split(" ")[0]
        return pytest_test_name + f"_{self.test_type}" + unique_id

    def is_test_expected_to_fail(self):
        return self.is_negative_test_case

    @override
    def __str__(self) -> str:
        return {
            "test_type": self.test_type,
            "additional_params": self.additional_params,
            "tensors": self.tensors,
            "is_negative_test_case": self.is_negative_test_case,
        }.__str__()


def assert_kernel_validation_exception(expected_validation_error: Optional[str], exception: Exception):
    assert exception is not None, "Expected to receive neuron assertion exception, but none was given"
    actual_exception_message = exception.__str__()

    assert "[NCC_INKI016] Kernel validation exception:" in actual_exception_message
    if expected_validation_error is not None:
        assert actual_exception_message.__contains__(expected_validation_error)


def is_in_negative_test_context() -> bool:
    """Check if we're currently inside an assert_negative_test_case context.

    This allows Orchestrator.execute() to automatically detect negative tests
    without requiring tests to explicitly pass is_negative_test to KernelArgs.
    """
    return _is_negative_test_context.get()


@contextmanager
def assert_negative_test_case(
    is_negative_test_case: bool,
    expected_validation_error: Optional[str] = None,
):
    token = _is_negative_test_context.set(is_negative_test_case)
    try:
        yield
    except Exception as e:
        if not is_negative_test_case:
            raise e
        assert_kernel_validation_exception(expected_validation_error, e)
    else:
        assert not is_negative_test_case, "Test case was expected to fail, but it hasn't!"
    finally:
        _is_negative_test_context.reset(token)


RANGE_TEST_CONFIG_ATTR_KEY = "range_test_config"
RANGE_TEST_FIXTURE_NAME = "range_test_options"
RANGE_TEST_LIMIT_NUM_TESTS_ATTR_KEY = "limit_num_of_test"
RANGE_TEST_RNG_SEED_ENV_KEY = "NEURON_PYTHONHASHSEED"


def set_test_func_attr(key: str, config: RangeTestConfig):
    def wrapper(func):
        setattr(func, key, config)
        return func

    return wrapper


def range_test_config(config: RangeTestConfig, limit_num_of_tests: Union[int, None] = None):
    """Function annotation used to set attribute in order to pass config required by RangeTestHarness

    Args:
        config:

    Returns:

    """
    f = set_test_func_attr(RANGE_TEST_CONFIG_ATTR_KEY, config)
    if limit_num_of_tests is not None and limit_num_of_tests > 0:
        setattr(f, RANGE_TEST_LIMIT_NUM_TESTS_ATTR_KEY, limit_num_of_tests)
    return f


class RangeGeneratorStrategy(ABC):
    def __init__(self, test_type: str) -> None:
        super().__init__()
        self.test_type: str = test_type

    @abstractmethod
    def generate_ranges(
        self, range_test_config: TensorRangeConfig, additional_params: dict[str, Any]
    ) -> list[RangeTestCase]:
        raise NotImplementedError()


@final
class RangeRandomGeneratorStrategy(RangeGeneratorStrategy):
    def __init__(self, num_of_samples: int) -> None:
        super().__init__("random")
        self.num_of_samples = num_of_samples

    def __generate_random_power_of(self, dim_config: DimensionRangeConfig) -> int:
        power_of = dim_config.power_of
        assert power_of is not None
        max_power = math.floor(math.log(dim_config.max, power_of))
        assert max_power is not None

        # bring power_of to a random power
        return int(math.pow(power_of, random.randint(1, max_power)))

    def __generate_random_multiple_of(self, dim_config: DimensionRangeConfig) -> int:
        assert dim_config.multiple_of is not None

        max_number_of_multiples = dim_config.max // dim_config.multiple_of
        min_number_of_multiples = (dim_config.min + dim_config.multiple_of - 1) // dim_config.multiple_of

        random_multiple_pos = random.randint(min_number_of_multiples, max_number_of_multiples)
        return random_multiple_pos * dim_config.multiple_of

    def __generate_random_in_range(self, dim_config: DimensionRangeConfig) -> int:
        return random.randint(dim_config.min, dim_config.max)

    def __generate_random(self, dim_config: DimensionRangeConfig) -> int:
        if dim_config.multiple_of is not None:
            return self.__generate_random_multiple_of(dim_config)
        elif dim_config.power_of is not None:
            return self.__generate_random_power_of(dim_config)
        else:
            return self.__generate_random_in_range(dim_config)

    @override
    def generate_ranges(
        self, range_test_config: TensorRangeConfig, additional_params: dict[str, Any]
    ) -> list[RangeTestCase]:
        test_cases = []
        for _ in range(range_test_config.random_sample_size):
            t_c = dict()
            for tensor_name, tensor_config in range_test_config.tensor_configs.items():
                dimension_configs = dict()
                for dimension in tensor_config.dimensions:
                    dimension_configs[dimension.name] = self.__generate_random(dimension)
                t_c[tensor_name] = dimension_configs

            test_cases.append(
                RangeTestCase(
                    additional_params=additional_params,
                    tensors=t_c,
                    test_type=self.test_type,
                    test_config_ref=range_test_config,
                )
            )

        return test_cases


class RangeMonotonicGeneratorStrategy(RangeGeneratorStrategy):
    def __init__(self, step_size: Union[int, None] = None, step_percent: Union[int, None] = None) -> None:
        super().__init__("monotonic")
        self.step_size: Union[int, None] = step_size
        self.step_percent: Union[int, None] = step_percent

    def __generate_monotonic_power_of(self, dim_config: DimensionRangeConfig) -> list[int]:
        configs = []
        assert dim_config.power_of is not None
        # if the user specifies a start that is not a power of power_of, we don't care
        start = dim_config.min
        while start <= dim_config.max:
            configs.append(start)
            start *= dim_config.power_of
        return configs

    def __generate_monotonic_in_range(self, dim_config: DimensionRangeConfig) -> list[int]:
        configs = []
        step_size = self.step_size
        if dim_config.multiple_of is not None:
            step_size = dim_config.multiple_of
        elif self.step_percent is not None:
            step_size = math.ceil(dim_config.max * (self.step_percent / 100))

        # in reality this will never happen because we have already checked it in test harness
        # lsp does not know it, so appease it by putting a check
        assert step_size is not None

        # Specifically handle the case of min=max so that we perform just 1 iteration.
        if dim_config.min == dim_config.max:
            configs.append(dim_config.min)
            return configs

        # Special case only seems to make sense for step_percent
        # I am not touching this logic with a 10 foot pole
        if dim_config.min == 1 and dim_config.min < dim_config.max:
            start = dim_config.min - 1 + step_size
        else:
            start = dim_config.min

        for c in range(start, dim_config.max + 1, step_size):
            if c <= dim_config.max:  # Safety check to ensure we don't exceed max
                configs.append(c)

        return configs

    def _generate_monotonic_configs(self, dimension: DimensionRangeConfig) -> list[int]:
        if dimension.power_of is not None:
            return self.__generate_monotonic_power_of(dimension)
        else:
            return self.__generate_monotonic_in_range(dimension)

    def __broadcast_dim_configs_to_size(
        self,
        tensor_configs: dict[str, dict[str, list[int]]],
        target_dim_config_length: int,
    ) -> dict[str, dict[str, list[int]]]:
        for tensor_name, dim_configs in tensor_configs.items():
            for dimension_name, c in dim_configs.items():
                expanded_dims = []
                num_of_time_to_duplicate = target_dim_config_length // len(c)

                for dim in c:
                    for _ in range(num_of_time_to_duplicate):
                        expanded_dims.append(dim)

                # last dimension config behaves as a residual value, which is going to be re-used more depending on config distribution
                for _ in range(target_dim_config_length - (num_of_time_to_duplicate * len(c))):
                    expanded_dims.append(c[len(c) - 1])

                tensor_configs[tensor_name][dimension_name] = expanded_dims

        return tensor_configs

    @override
    def generate_ranges(
        self, range_test_config: TensorRangeConfig, additional_params: dict[str, Any]
    ) -> list[RangeTestCase]:
        test_cases = []

        tensor_configs: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: list()))
        max_dim_config_length = -1

        for tensor_name, tensor_config in range_test_config.tensor_configs.items():
            for dimension in tensor_config.dimensions:
                dimension_configs = self._generate_monotonic_configs(dimension)

                tensor_configs[tensor_name][dimension.name] = dimension_configs
                max_dim_config_length = max(max_dim_config_length, len(dimension_configs))

        tensor_configs = self.__broadcast_dim_configs_to_size(tensor_configs, max_dim_config_length)

        for i in range(max_dim_config_length):
            curr_test_config = dict()
            for tensor_name, dim_configs in tensor_configs.items():
                curr_dim_config = dict()
                for dimension_name, dim_list in dim_configs.items():
                    curr_dim_config[dimension_name] = dim_list[i]
                curr_test_config[tensor_name] = curr_dim_config

            test_cases.append(
                RangeTestCase(
                    additional_params=additional_params,
                    tensors=curr_test_config,
                    test_type=self.test_type,
                    test_config_ref=range_test_config,
                )
            )

        return test_cases


@final
class RangeBoundaryGeneratorStrategy(RangeGeneratorStrategy):
    def __init__(self) -> None:
        super().__init__("boundary")

    def __generate_min_viable_tensor_configs(self, range_test_config: TensorRangeConfig):
        default_tensor_configs = dict()

        for tensor_name, tensor_config in range_test_config.tensor_configs.items():
            dimension_configs = dict()
            for dimension in tensor_config.dimensions:
                dimension_configs[dimension.name] = dimension.min

            default_tensor_configs[tensor_name] = dimension_configs

        return default_tensor_configs

    @override
    def generate_ranges(
        self, range_test_config: TensorRangeConfig, additional_params: dict[str, Any]
    ) -> list[RangeTestCase]:
        test_cases = []

        default_tensor_config = self.__generate_min_viable_tensor_configs(range_test_config)

        for tensor_name, tensor_config in range_test_config.tensor_configs.items():
            t_c = deepcopy(default_tensor_config)

            # process one dimension at a time by inserting boundary condition checks
            # all other dimenions within a test case are within valid range except for the one under test
            for dimension in tensor_config.dimensions:
                # beyond min boundary
                t_c[tensor_name][dimension.name] = dimension.min - 1
                test_cases.append(
                    RangeTestCase(
                        additional_params=additional_params,
                        tensors=deepcopy(t_c),
                        is_negative_test_case=dimension.min_is_binding,
                        test_type=self.test_type,
                        test_config_ref=range_test_config,
                    )
                )
                # beyond max boundary
                t_c[tensor_name][dimension.name] = dimension.max + 1
                test_cases.append(
                    RangeTestCase(
                        additional_params=additional_params,
                        tensors=deepcopy(t_c),
                        is_negative_test_case=dimension.max_is_binding,
                        test_type=self.test_type,
                        test_config_ref=range_test_config,
                    )
                )
                # reset tensor config to default to process other dimenions
                t_c[tensor_name] = deepcopy(default_tensor_config[tensor_name])

        return test_cases


@final
class RangeManualGeneratorStrategy(RangeGeneratorStrategy):
    def __init__(self, test_cases, test_type: str = "manual") -> None:
        super().__init__(test_type)
        self.test_cases = test_cases

    @override
    def generate_ranges(
        self, range_test_config: TensorRangeConfig, additional_params: dict[str, Any]
    ) -> list[RangeTestCase]:
        return [
            RangeTestCase(
                additional_params=additional_params,
                tensors=test_case,
                test_type=self.test_type,
                test_config_ref=range_test_config,
            )
            for test_case in self.test_cases
        ]


@final
class RangeProductConstraintMonotonicStrategy(RangeGeneratorStrategy):
    """Generator strategy for testing combined dimension constraints with product relationships.

    This strategy generates test cases for constraints like B * S <= 128 or N * D <= 4096,
    using monotonic stepping similar to RangeMonotonicGeneratorStrategy. It supports both
    fixed step sizes and percentage-based stepping.

    Args:
        fixed_dims: Dictionary mapping tensor names to their fixed dimension configs
                    (dict[str, dict[str, int]])
        product_dims: Tuple of two dimension names that have a product constraint
        product_limit: Maximum value for the product of the two dimensions
        step_size: Fixed step size for both dimensions (mutually exclusive with step_percent)
        step_percent: Percentage-based step size (1-100, mutually exclusive with step_size)
        dim_max: Optional dictionary specifying individual max values for each dimension
        dim_min: Optional dictionary specifying individual min values (floor) for each dimension (default: 1)
        constrained_tensors: Optional list of tensor names to apply constraint to (defaults to all)
    """

    def __init__(
        self,
        fixed_dims: dict[str, dict[str, int]],
        product_dims: tuple[str, str],
        product_limit: int,
        step_size: Union[int, None] = None,
        step_percent: Union[int, None] = None,
        dim_max: Union[dict[str, int], None] = None,
        dim_min: Union[dict[str, int], None] = None,
        constrained_tensors: Union[list[str], None] = None,
    ) -> None:
        super().__init__("product_monotonic")
        self.fixed_dims = fixed_dims
        self.product_dims = product_dims
        self.product_limit = product_limit
        self.step_size = step_size
        self.step_percent = step_percent
        self.dim_max = dim_max or {}
        self.dim_min = dim_min or {}
        self.constrained_tensors = constrained_tensors

        # Validate like the existing monotonic strategy
        assert (
            self.step_percent is not None or self.step_size is not None
        ), "either step size or step percent has to be provided"

        if self.step_percent is not None:
            assert 0 < self.step_percent <= 100, "invalid step_percent provided, has to be between (0, 100]"

    def _get_fixed_dims_for_tensor(self, tensor_name: str) -> dict[str, int]:
        """Get fixed dimensions for a specific tensor."""
        return self.fixed_dims.get(tensor_name, {})

    def _should_apply_constraint_to_tensor(self, tensor_name: str, tensor_config: TensorConfig) -> bool:
        """Determine if product constraint should apply to this tensor."""
        # If constrained_tensors is specified, only apply to those
        if self.constrained_tensors is not None:
            return tensor_name in self.constrained_tensors

        # Otherwise, check if tensor has both constrained dimensions
        dim_names = {dim.name for dim in tensor_config.dimensions}
        return self.product_dims[0] in dim_names and self.product_dims[1] in dim_names

    def _calculate_step_for_dimension(self, dim_name: str) -> int:
        """Calculate step size for a dimension, using percentage if specified."""
        if self.step_percent is not None:
            # Use the individual dimension max or the product limit as reference
            dim_max = self.dim_max.get(dim_name, self.product_limit)
            step = math.ceil(dim_max * (self.step_percent / 100))
        else:
            step = self.step_size

        # Ensure step is at least 1 and not None
        assert step is not None, "step_size or step_percent must be provided"
        return max(1, step)

    @override
    def generate_ranges(
        self, range_test_config: TensorRangeConfig, additional_params: dict[str, Any]
    ) -> list[RangeTestCase]:
        test_cases = []
        dim1_name, dim2_name = self.product_dims

        # Calculate steps for each dimension
        dim1_step = self._calculate_step_for_dimension(dim1_name)
        dim2_step = self._calculate_step_for_dimension(dim2_name)

        # Get max and min values for each dimension
        dim1_max = self.dim_max.get(dim1_name, self.product_limit)
        dim2_max = self.dim_max.get(dim2_name, self.product_limit)
        dim1_min = self.dim_min.get(dim1_name, 1)
        dim2_min = self.dim_min.get(dim2_name, 1)

        assert dim1_max <= self.product_limit
        assert dim2_max <= self.product_limit

        assert dim1_max >= dim1_min
        assert dim2_max >= dim2_min

        # Generate values for dim1 starting from dim1_min
        # Always include dim1_min, then add multiples of step size
        dim1_values = [dim1_min]
        if dim1_max > dim1_min:
            # Add stepped values starting from dim1_min + dim1_step
            current = dim1_min + dim1_step
            while current <= dim1_max:
                if current != dim1_min:  # Don't duplicate the starting value
                    dim1_values.append(current)
                current += dim1_step

        for dim1_val in dim1_values:
            # Calculate valid dim2 range given the constraint
            max_dim2_for_constraint = self.product_limit // dim1_val
            max_dim2_for_constraint = min(max_dim2_for_constraint, dim2_max)

            # Generate values for dim2 for this dim1 value, starting from dim2_min
            dim2_values = [dim2_min]
            if max_dim2_for_constraint > dim2_min:
                current = dim2_min + dim2_step
                while current <= max_dim2_for_constraint:
                    if current != dim2_min:  # Don't duplicate the starting value
                        dim2_values.append(current)
                    current += dim2_step

            for dim2_val in dim2_values:
                # Build test case for each tensor in the config
                t_c = dict()
                for (
                    tensor_name,
                    tensor_config,
                ) in range_test_config.tensor_configs.items():
                    # Check if we should apply constraint to this tensor
                    if self._should_apply_constraint_to_tensor(tensor_name, tensor_config):
                        # Get fixed dims for this tensor and add varying dimensions
                        tensor_fixed_dims = self._get_fixed_dims_for_tensor(tensor_name)
                        dimension_configs = tensor_fixed_dims.copy()
                        dimension_configs[dim1_name] = dim1_val
                        dimension_configs[dim2_name] = dim2_val
                        t_c[tensor_name] = dimension_configs
                    else:
                        # Tensor not constrained - use its own dimension configs from range_test_config
                        # Generate default values (could be enhanced to use other strategies)
                        dimension_configs = {}
                        for dim in tensor_config.dimensions:
                            dimension_configs[dim.name] = dim.min
                        t_c[tensor_name] = dimension_configs

                test_cases.append(
                    RangeTestCase(
                        additional_params=additional_params,
                        tensors=t_c,
                        test_type=self.test_type,
                        test_config_ref=range_test_config,
                    )
                )

        return test_cases


@final
class RangeTestHarness:
    """Test harness for generating parametrized test configurations.

    This class generates test cases based on range configurations but does NOT execute tests.
    Test execution is handled by pytest through individual test functions that receive
    range_test_options fixture and call Orchestrator.execute() directly.

    Usage with pytest:

    @range_test_config(config=RangeTestConfig(...))
    def test_my_kernel(range_test_options: RangeTestCase, test_manager: Orchestrator):
        # Use range_test_options.tensors, range_test_options.additional_params, etc.
        # to configure your kernel
        test_manager.execute(KernelArgs(...))
    """

    def __init__(
        self,
        range_test_config: RangeTestConfig,
        limit_num_of_test: Union[int, None] = None,
    ) -> None:
        self.logger = self.__setup_logger()
        test_cases = self.__generate_test_cases(range_test_config)
        self.range_test_config = range_test_config
        self.unique_test_cases = self.__dedupe_test_cases(test_cases)
        self.limit_num_of_test = limit_num_of_test

    def __setup_logger(self):
        logger = logging.getLogger("range-test-harness")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt=developer_logging_format, datefmt=logging_datefmt)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def get_unique_test_cases(self) -> tuple[list[str], list[RangeTestCase]]:
        """Returns tuple of (test_ids, test_cases) for pytest parametrization."""
        keys = [k for k in self.unique_test_cases.keys()]
        if self.limit_num_of_test is not None:
            keys = keys[: self.limit_num_of_test]
        return keys, [self.unique_test_cases[k] for k in keys]

    def __generate_test_cases(self, test_config: RangeTestConfig) -> list[RangeTestCase]:
        tensor_configs = [test_config.global_tensor_configs]

        if test_config.subrange_tensor_configs:
            tensor_configs.extend(test_config.subrange_tensor_configs)

        test_cases = []

        for tc in tensor_configs:
            generators = []
            if tc.custom_generators is not None and len(tc.custom_generators) > 0:
                generators.extend(tc.custom_generators)
            else:
                generators.extend(
                    [
                        RangeRandomGeneratorStrategy(
                            tc.random_sample_size,
                        ),
                        RangeMonotonicGeneratorStrategy(
                            tc.monotonic_step_size,
                            tc.monotonic_step_percent,
                        ),
                        RangeBoundaryGeneratorStrategy(),
                    ]
                )

            for generator in generators:
                test_cases.extend(generator.generate_ranges(tc, test_config.additional_params))

        return test_cases

    def __dedupe_test_cases(self, test_cases: list[RangeTestCase]) -> dict[str, RangeTestCase]:
        """Deduplicate test cases using dedup_key (dimensions only) while preserving unique_id for display."""
        dedup_map: set[str] = set()
        result: dict[str, RangeTestCase] = dict()

        for test_case in test_cases:
            dedup_key = test_case.dedup_key()
            if dedup_key not in dedup_map:
                dedup_map.add(dedup_key)
                result[test_case.unique_id()] = test_case  # Use unique_id for display

        return result
