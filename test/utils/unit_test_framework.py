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
from test.utils.common_dataclasses import (
    CompilerArgs,
    InferenceArgs,
    KernelArgs,
    LazyGoldenGenerator,
    ValidationArgs,
)
from test.utils.coverage_parametrized_tests import assert_negative_test_case
from test.utils.metadata_loader import load_model_configs
from test.utils.metrics_collector import IMetricsCollector
from test.utils.test_orchestrator import Orchestrator
from typing import Callable, Optional

import numpy as np


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
        torch_ref: Callable,
        kernel_input_generator: Callable,
        output_tensor_descriptor: Callable,
        check_unused_params: bool = False,
        collector: Optional[IMetricsCollector] = None,
    ):
        """Initialize the test framework.

        Args:
            test_manager: Test orchestrator for execution
            kernel_entry: Kernel function under test
            torch_ref: Torch reference function (signature must match kernel_entry)
            kernel_input_generator: Function(test_config) -> dict of inputs
            output_tensor_descriptor: Function(kernel_input) -> dict of output tensors
            check_unused_params: Check for unused parameters in kernel_entry
            collector: Optional metrics collector for model coverage tracking
        """
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
        is_negative_test: bool = False,
        inference_args: Optional[InferenceArgs] = None,
        custom_validation_args: Optional[ValidationArgs] = None,
        metadata: Optional[dict] = None,
    ):
        """Execute a single test case.

        Args:
            test_config: Test configuration (can be None when using pytest.mark.parametrize)
            compiler_args: Compiler arguments
            rtol: Relative tolerance for validation
            atol: Absolute tolerance for validation
            is_negative_test: Whether this is a negative test case
            inference_args: Optional inference arguments (e.g., for determinism checking)
            metadata: Optional dict with 'config_name' (str for load_model_configs) and 'key' (test dimensions dict)
        """
        if self.collector is not None and metadata is not None:
            metadata_list = load_model_configs(metadata["config_name"])
            self.collector.match_and_add_metadata_dimensions(metadata["key"], metadata_list)

        with assert_negative_test_case(is_negative_test):
            # Generate kernel inputs
            kernel_input = self.kernel_input_generator(test_config)

            # Validate kernel_input against kernel_entry signature
            sig = signature(self.kernel_entry)
            kernel_params = set(sig.parameters.keys())

            # Check for extra keys in kernel_input that don't match kernel_entry
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

            # Check for required parameters missing from kernel_input
            missing_required = []
            for param_name, param in sig.parameters.items():
                if param.default is inspect.Parameter.empty:  # required param
                    if param_name not in kernel_input and f"{param_name}.must_alias_input" not in kernel_input:
                        missing_required.append(param_name)
            if missing_required:
                raise ValueError(f"kernel_input missing required parameters: {missing_required}")

            # Filter parameters for kernel (handle .must_alias_input suffix)
            filtered_kernel_input = {}
            for k, v in kernel_input.items():
                if k in kernel_params:
                    filtered_kernel_input[k] = v
                elif k.endswith(".must_alias_input"):
                    base_key = k.rsplit(".must_alias_input", 1)[0]
                    if base_key in kernel_params:
                        filtered_kernel_input[k] = v

            # Filter parameters for torch_ref (handle .must_alias_input suffix)
            ref_sig = signature(self.torch_ref)
            ref_params = set(ref_sig.parameters.keys())
            ref_input = {}
            for k, v in kernel_input.items():
                if k in ref_params:
                    ref_input[k] = v
                elif k.endswith(".must_alias_input"):
                    base_key = k.rsplit(".must_alias_input", 1)[0]
                    if base_key in ref_params:
                        ref_input[base_key] = v.copy() if hasattr(v, "copy") else v

            # Generate output tensors
            output_tensors = self.output_tensor_descriptor(kernel_input)

            # Defer torch_ref computation to validation time via lazy golden generator.
            # This avoids running torch_ref in compile-only/trace-only modes (orchestrator
            # returns early and .golden is never accessed), and in normal mode it runs
            # after the kernel compile+infer, right before output data comparison.
            def compute_ref():
                ref_result = self.torch_ref(**ref_input)
                _validate_key_sets(
                    expected=set(ref_result.keys()),
                    actual=set(output_tensors.keys()),
                    msg_header="Output tensor mismatch:",
                    expected_label="torch_ref returns but output_tensor_descriptor doesn't provide",
                    actual_label="output_tensor_descriptor provides but torch_ref doesn't return",
                )
                return ref_result

            lazy_golden = LazyGoldenGenerator(
                lazy_golden_generator=compute_ref,
                output_ndarray=output_tensors,
            )

            # Execute test
            validation_args = custom_validation_args or ValidationArgs(
                golden_output=lazy_golden,
                relative_accuracy=rtol,
                absolute_accuracy=atol,
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


def torch_ref_wrapper(torch_ref_func: Callable) -> Callable:
    """Wrap a torch reference function to handle numpy<->torch conversion.

    Converts numpy arrays to torch tensors (float16->float32 for CPU compatibility),
    calls the torch reference, and converts results back to numpy.

    Args:
        torch_ref_func: Torch reference function that takes torch tensors as kwargs

    Returns:
        Wrapped function that takes numpy arrays and returns numpy arrays

    Example:
        @torch_ref_wrapper
        def my_kernel_torch_ref(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return torch.matmul(input, weight)

        # Can now call with numpy arrays:
        result = my_kernel_torch_ref(input=np_array, weight=np_weight)
    """
    import functools

    import torch

    @functools.wraps(torch_ref_func)
    def wrapped(**kwargs):
        # Convert numpy arrays to torch tensors (float16/bfloat16->float32 for CPU)
        torch_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                dtype_str = str(value.dtype)
                # Handle MX packed x4 types: pass as numpy for torch ref to handle
                if 'x4' in dtype_str:
                    torch_kwargs[key] = value
                    continue
                # Handle uint32 by converting to int32 (torch doesn't support uint32)
                if value.dtype == np.uint32:
                    value = value.astype(np.int32)
                # Handle bfloat16 by converting to float32 (torch.from_numpy doesn't support bfloat16)
                elif 'bfloat16' in dtype_str:
                    value = value.astype(np.float32)
                # Handle float8 types by converting to float32 (torch.from_numpy doesn't support float8)
                elif 'float8' in dtype_str:
                    value = value.astype(np.float32)
                tensor = torch.from_numpy(value)
                # Convert float16 to float32 for CPU compatibility
                if tensor.dtype == torch.float16:
                    tensor = tensor.float()
                torch_kwargs[key] = tensor
            else:
                torch_kwargs[key] = value

        # Call torch reference
        result = torch_ref_func(**torch_kwargs)

        # Convert result back to numpy
        if isinstance(result, torch.Tensor):
            if result.dtype == torch.bfloat16:
                result = result.float()
            return {"out": result.numpy()}
        elif isinstance(result, dict):
            out = {}
            for k, v in result.items():
                if isinstance(v, torch.Tensor):
                    if v.dtype == torch.bfloat16:
                        v = v.float()
                    out[k] = v.numpy()
                else:
                    out[k] = v
            return out
        else:
            return result

    return wrapped
