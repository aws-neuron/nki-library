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

"""Unit Test Framework for NKI Kernels.

This framework standardizes test interfaces, enforces consistency, and simplifies unit test implementations.

Key Features:
    - Signature validation: kernel_entry ↔ torch_ref parameter consistency
    - Input validation: kernel_input keys match kernel_entry signature
    - Output validation: output_tensor_descriptor keys match torch_ref returns
    - Unused parameter detection (opt-in): catches forgotten pass-through in wrappers

Usage:
    Developers need to provide:
    1. Kernel input generator function
    2. Torch reference generator function
    3. Test configuration (parameters and dimensions)
    4. For SBUF I/O: a thin wrapper that handles HBM<->SBUF conversions

    The framework handles:
    - Test orchestration and execution
    - Input/output validation
    - Automatic parameter filtering
"""

import inspect
from inspect import signature
from typing import Callable, Optional

from nkilib_src.nkilib.core.utils.torch_ref_wrapper import torch_ref_wrapper  # noqa: F401

from .common_dataclasses import (
    CompilerArgs,
    InferenceArgs,
    KernelArgs,
    LazyGoldenGenerator,
    ValidationArgs,
)
from .coverage_parametrized_tests import assert_negative_test_case
from .golden_provider import CustomComparatorProducer, GoldenProducer, TorchRefProducer
from .metadata_loader import load_model_configs
from .metrics_collector import IMetricsCollector, MetricName
from .test_orchestrator import Orchestrator


class UnitTestFramework:
    """Framework for executing NKI kernel unit tests.

    Orchestrates the complete test flow:
    1. Generate kernel inputs from test configuration
    2. Execute kernel with generated inputs
    3. Validate outputs against golden reference
    """

    def __init__(
        self,
        test_manager: Orchestrator,
        kernel_entry: Callable,
        kernel_input_generator: Callable,
        torch_ref: Optional[Callable] = None,
        output_tensor_descriptor: Optional[Callable] = None,
        check_unused_params: bool = False,
        collector: Optional[IMetricsCollector] = None,
        trace_only: bool = False,
    ):
        """Initialize the test framework.

        Args:
            test_manager: Test orchestrator for execution
            kernel_entry: Kernel function under test
            kernel_input_generator: Function(test_config) -> dict of inputs
            torch_ref: Torch reference function (signature must match kernel_entry).
                Required unless trace_only=True.
            output_tensor_descriptor: Function(kernel_input) -> dict of output tensors.
                Required unless trace_only=True.
            check_unused_params: Check for unused parameters in kernel_entry
            collector: Optional metrics collector for model coverage tracking
            trace_only: If True, skip torch_ref validation and output comparison.
                Use for tests that only trace/compile the kernel and validate via
                kernel_assert inside the kernel itself.
        """
        self.trace_only = trace_only

        if not trace_only:
            if torch_ref is None:
                raise ValueError("torch_ref is required when trace_only=False")
            if output_tensor_descriptor is None:
                raise ValueError("output_tensor_descriptor is required when trace_only=False")
            validate_torch_ref_signature(kernel_entry, torch_ref)

        if check_unused_params:
            check_unused_parameters(kernel_entry)

        self.test_manager = test_manager
        self.kernel_entry = kernel_entry
        self.torch_ref = torch_ref
        self.kernel_input_generator = kernel_input_generator
        self.output_tensor_descriptor = output_tensor_descriptor
        self.collector = collector

    def run_test(
        self,
        test_config,
        compiler_args: CompilerArgs,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan_inf: bool = False,
        is_negative_test: bool = False,
        inference_args: Optional[InferenceArgs] = None,
        custom_validation_args: Optional[ValidationArgs] = None,
        custom_comparator: Optional[Callable] = None,
        metadata: Optional[dict] = None,
    ):
        """Execute a single test case.

        Args:
            test_config: Test configuration (can be None when using pytest.mark.parametrize)
            compiler_args: Compiler arguments
            rtol: Relative tolerance for validation
            atol: Absolute tolerance for validation
            equal_nan_inf: If True, matching NaN and matching infinity values are treated as equal
            is_negative_test: Whether this is a negative test case
            inference_args: Optional inference arguments (e.g., for determinism checking)
            custom_validation_args: Optional pre-built ValidationArgs that bypasses both
                torch_ref golden generation and the default comparison. Use this only when
                the golden cannot come from torch_ref at all.
            custom_comparator: Optional callable(golden_dict, output_tensors) -> dict mapping
                output names to CustomValidatorWithOutputTensorData. The framework runs
                torch_ref to produce golden_dict, then passes it to this function to build
                custom validators. This keeps golden generation in the framework while
                allowing custom comparison logic (e.g., cosine similarity, scaled tolerances).
                Mutually exclusive with custom_validation_args.
            metadata: Optional dict with 'config_name' (str for load_model_configs) and 'key' (test dimensions dict)
        """
        if custom_validation_args is not None and custom_comparator is not None:
            raise ValueError("custom_validation_args and custom_comparator are mutually exclusive")

        if self.collector is not None and metadata is not None:
            metadata_list = load_model_configs(metadata["config_name"])
            self.collector.match_and_add_metadata_dimensions(metadata["key"], metadata_list)

        with assert_negative_test_case(is_negative_test):
            # Generate kernel inputs
            kernel_input = self.kernel_input_generator(test_config)

            # Validate and filter inputs using shared helpers
            validate_input_keys(kernel_input, self.kernel_entry)
            filtered_kernel_input = filter_kernel_input(kernel_input, self.kernel_entry)

            if self.trace_only:
                # Trace-only mode: no golden comparison, just trace/compile the kernel.
                # Build output_ndarray from .must_alias_input keys so the compiler knows
                # the kernel's output names (required for compilation).
                output_ndarray = {}
                for k, v in kernel_input.items():
                    if k.endswith(".must_alias_input"):
                        base_key = k.rsplit(".must_alias_input", 1)[0]
                        output_ndarray[base_key] = v
                validation_args = ValidationArgs(
                    golden_output=LazyGoldenGenerator(output_ndarray=output_ndarray, lazy_golden_generator=None),
                )
                kernel_args = KernelArgs(
                    kernel_func=self.kernel_entry,
                    compiler_input=compiler_args,
                    kernel_input=filtered_kernel_input,
                    validation_args=validation_args,
                )
                if inference_args is not None:
                    kernel_args.inference_args = inference_args
                self.test_manager.execute(kernel_args)
                return

            ref_input = filter_ref_input(kernel_input, self.torch_ref)

            # Generate output tensors
            output_tensors = self.output_tensor_descriptor(kernel_input)

            # Defer torch_ref computation to validation time via lazy golden generator.
            # This avoids running torch_ref in compile-only/trace-only modes (orchestrator
            # returns early and .golden is never accessed), and in normal mode it runs
            # after the kernel compile+infer, right before output data comparison.
            def compute_ref():
                # Timed as GoldenComputationTime — the actual reference compute. The
                # producer may serve the golden without invoking this, so the metric is
                # present only when a compute actually happens.
                with self.test_manager.collector.timer(MetricName.GOLDEN_COMPUTATION_TIME):
                    ref_result = self.torch_ref(**ref_input)
                _validate_key_sets(
                    expected=set(ref_result.keys()),
                    actual=set(output_tensors.keys()),
                    msg_header="Output tensor mismatch:",
                    expected_label="torch_ref returns but output_tensor_descriptor doesn't provide",
                    actual_label="output_tensor_descriptor provides but torch_ref doesn't return",
                )
                return ref_result

            # Golden production is a pluggable stage (GoldenProducer): the producer yields
            # the golden the validator consumes, for both the plain and custom_comparator
            # paths. run_test composes the producer with the validator; how the producer
            # obtains the golden is its own concern.
            producer: GoldenProducer = TorchRefProducer(
                compute_ref,
                self.torch_ref,
                ref_input,
                output_tensors,
                self.test_manager.collector,
                self.test_manager.torch_ref_cache_path,
            )
            if custom_comparator is not None:
                producer = CustomComparatorProducer(producer, custom_comparator, output_tensors)

            lazy_golden = LazyGoldenGenerator(
                lazy_golden_generator=producer.produce,
                output_ndarray=producer.output_spec(),
            )

            # Execute test
            validation_args = custom_validation_args or ValidationArgs(
                golden_output=lazy_golden,
                relative_accuracy=rtol,
                absolute_accuracy=atol,
                equal_nan_inf=equal_nan_inf,
            )
            kernel_args = KernelArgs(
                kernel_func=self.kernel_entry,
                compiler_input=compiler_args,
                kernel_input=filtered_kernel_input,
                validation_args=validation_args,
            )
            if inference_args is not None:
                kernel_args.inference_args = inference_args

            self.test_manager.execute(kernel_args)


# --- Shared Validation Helpers ---


def validate_input_keys(kernel_input: dict, kernel_entry: Callable) -> None:
    """Validate kernel_input keys against kernel_entry signature.

    Checks for extra keys not in the signature and missing required parameters.
    Handles .must_alias_input suffix transparently.
    """
    sig = signature(kernel_entry)
    kernel_params = set(sig.parameters.keys())

    extra_keys = []
    for k in kernel_input.keys():
        base_key = k.rsplit(".must_alias_input", 1)[0] if k.endswith(".must_alias_input") else k
        if base_key not in kernel_params:
            extra_keys.append(k)
    if extra_keys:
        raise ValueError(
            f"kernel_input has keys {extra_keys} that don't match kernel_entry signature. "
            f"Expected parameters: {sorted(kernel_params)}"
        )

    missing_required = []
    for param_name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty:
            if param_name not in kernel_input and f"{param_name}.must_alias_input" not in kernel_input:
                missing_required.append(param_name)
    if missing_required:
        raise ValueError(f"kernel_input missing required parameters: {missing_required}")


def filter_kernel_input(kernel_input: dict, kernel_entry: Callable) -> dict:
    """Filter kernel_input to only parameters accepted by kernel_entry.

    Handles .must_alias_input suffix transparently.
    """
    kernel_params = set(signature(kernel_entry).parameters.keys())
    filtered = {}
    for k, v in kernel_input.items():
        if k in kernel_params:
            filtered[k] = v
        elif k.endswith(".must_alias_input"):
            base_key = k.rsplit(".must_alias_input", 1)[0]
            if base_key in kernel_params:
                filtered[k] = v
    return filtered


def filter_ref_input(kernel_input: dict, torch_ref: Callable) -> dict:
    """Filter kernel_input to only parameters accepted by torch_ref.

    Handles .must_alias_input suffix: strips suffix and copies arrays to avoid aliasing.
    """
    ref_params = set(signature(torch_ref).parameters.keys())
    ref_input = {}
    for k, v in kernel_input.items():
        if k in ref_params:
            ref_input[k] = v
        elif k.endswith(".must_alias_input"):
            base_key = k.rsplit(".must_alias_input", 1)[0]
            if base_key in ref_params:
                ref_input[base_key] = v.copy() if hasattr(v, "copy") else v
    return ref_input


# --- Helper Functions ---


def _validate_key_sets(expected: set, actual: set, msg_header: str, expected_label: str, actual_label: str) -> None:
    """Raise ValueError if two key sets don't match, with a descriptive diff message."""
    if expected != actual:
        missing = expected - actual
        extra = actual - expected
        msg = msg_header
        if missing:
            msg += f"\n  {expected_label}: {sorted(missing)}"
        if extra:
            msg += f"\n  {actual_label}: {sorted(extra)}"
        raise ValueError(msg)


def validate_torch_ref_signature(kernel_entry: Callable, torch_ref: Callable) -> None:
    """Validate torch reference signature matches kernel signature."""
    kernel_params = set(signature(kernel_entry).parameters.keys())
    ref_params = set(signature(torch_ref).parameters.keys())
    _validate_key_sets(
        expected=kernel_params,
        actual=ref_params,
        msg_header="Torch ref signature mismatch with kernel:",
        expected_label="Missing in torch_ref",
        actual_label="Extra in torch_ref",
    )


def check_unused_parameters(func: Callable) -> None:
    """Raise error if any parameter in func's signature appears unused in the function body.

    This helps catch bugs where a wrapper accepts a parameter but forgets to pass it through.

    Raises:
        ValueError: If a parameter appears only in the signature (likely unused).
    """
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return  # Can't get source, skip check

    sig = signature(func)
    unused = []
    for param_name in sig.parameters:
        # Count occurrences - if only 1, it's just in the signature
        if source.count(param_name) == 1:
            unused.append(param_name)

    if unused:
        raise ValueError(
            f"Parameters {unused} may be unused in {func.__name__}. "
            "Ensure all parameters are forwarded to the underlying function."
        )
