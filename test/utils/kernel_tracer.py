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
from dataclasses import dataclass
from typing import Any, List

import numpy as np
from neuronxcc.nki_standalone import (
    NKI_IR_VERSION,
    _compile_nki_ir_to_tensorizer_ir,
    _write_tensorizer_ir,
    compile_nki_ir_kernel_to_neff,
)

from .common_dataclasses import (
    CompilerArgs,
    CustomValidatorWithOutputTensorData,
    GoldenTensorDict,
    KernelArgs,
    PerRankLazyInputGenerator,
    SeparationPassMode,
    TraceMode,
    ValidationArgs,
    normalize_golden_output,
)

DEFAULT_COMPILER_DEBUG_FLAGS = [
    "--internal-backend-options=--print-format=condensed",
    "--internal-compiler-debug-mode=all",
]

DUMP_AFTER_LOWERING_FLAGS = [
    "--internal-backend-options='--print-format=condensed --print-after=translate_nki_ast_to_bir,lower_klir_kernel'",
]

DEFAULT_COMPILER_FLAGS = [
    "--verbose=info",
    "--pipeline",
    "compile",
    "SaveTemps",
]


def __construct_additional_arguments__(
    compiler_args: CompilerArgs, validation_args: ValidationArgs | None = None
) -> str:
    additional_args = compiler_args.additional_cmd_args.copy()

    if compiler_args.enable_debugging:
        additional_args.extend(DEFAULT_COMPILER_DEBUG_FLAGS)

    if compiler_args.enable_birsim:
        atol = validation_args.absolute_accuracy if validation_args else ValidationArgs.absolute_accuracy
        rtol = validation_args.relative_accuracy if validation_args else ValidationArgs.relative_accuracy
        rtol_percent = 100 * rtol
        birsim_flags = [
            f"--internal-backend-options='--enable-birsim=True --enable-birsim-at-begin=False --enable-birsim-after-all=False --enable-birsim-at-end=True --birsim-output-tolerance {rtol_percent},{atol}'"
        ]
        additional_args.extend(birsim_flags)

    if compiler_args.separation_pass_mode != SeparationPassMode.NONE:
        additional_args.append(
            f"--internal-enable-separate-load-and-compute={compiler_args.separation_pass_mode.value}"
        )

    if compiler_args.dump_after_lowering:
        additional_args.extend(DUMP_AFTER_LOWERING_FLAGS)

    additional_args.extend(DEFAULT_COMPILER_FLAGS)

    return " ".join(additional_args)


def is_tensor(maybe_tensor: Any) -> bool:
    return hasattr(maybe_tensor, "shape")  # and not isinstance(type(maybe_tensor), type)


@dataclass
class TensorStub:
    """
    Duck type for constructing kernels inside the compiler
    """

    shape: List[int]
    dtype: 'np.dtype'
    name: str


def construct_kernel_IO_from_kernel_args(kernel_under_test: KernelArgs):
    # Get kernel inputs resolved for rank 0 (for compilation)
    kernel_input_dict = {}
    if kernel_under_test.kernel_input is not None:
        if isinstance(kernel_under_test.kernel_input, PerRankLazyInputGenerator):
            kernel_input_dict = kernel_under_test.kernel_input.for_rank(0)
        else:
            kernel_input_dict = kernel_under_test.kernel_input

    kernel_outputs: list = []
    if kernel_under_test.validation_args is not None:
        golden_output: GoldenTensorDict = normalize_golden_output(kernel_under_test.validation_args.golden_output)

        for (
            output_key,
            output_value,
        ) in golden_output.items():
            if is_tensor(output_value):
                assert isinstance(output_value, np.ndarray)

                tensor_ir = TensorStub(shape=output_value.shape, dtype=output_value.dtype, name=output_key)
                kernel_outputs.append(tensor_ir)
            else:
                assert isinstance(output_value, CustomValidatorWithOutputTensorData)

                output_ndarray = output_value.output_ndarray
                tensor_ir = TensorStub(shape=output_ndarray.shape, dtype=output_ndarray.dtype, name=output_key)
                kernel_outputs.append(tensor_ir)

    return kernel_input_dict, kernel_outputs


def trace_kernel_only(kernel_under_test: KernelArgs, output_directory: str):
    kernel_input_dict, kernel_outputs = construct_kernel_IO_from_kernel_args(kernel_under_test)

    ir_module, _ = _compile_nki_ir_to_tensorizer_ir(
        kernel_under_test.kernel_func,
        str(kernel_under_test.compiler_input.platform_target.get_compile_target()),
        kernel_under_test.compiler_input.logical_nc_config,
        kernel_input_dict,
        kernel_outputs,
        NKI_IR_VERSION.beta2,
    )

    _write_tensorizer_ir(ir_module, output_directory)


def compile_kernel_to_neff(kernel_under_test: KernelArgs, output_directory: str):
    # Prepare to call compiler utility to construct the NEFF from kernels
    additional_compiler_args = __construct_additional_arguments__(
        kernel_under_test.compiler_input, kernel_under_test.validation_args
    )

    kernel_input_dict, kernel_outputs = construct_kernel_IO_from_kernel_args(kernel_under_test)

    compile_nki_ir_kernel_to_neff(
        kernel_under_test.kernel_func,
        kernel_input_dict,
        kernel_outputs,
        str(kernel_under_test.compiler_input.platform_target.get_compile_target()),
        kernel_under_test.compiler_input.logical_nc_config,
        output_directory,
        NKI_IR_VERSION.beta2,
        additional_compiler_args,
    )


def trace_kernel(
    kernel_under_test: KernelArgs,
    mode: TraceMode,
    output_directory,
):
    if mode == TraceMode.TraceOnly:
        trace_kernel_only(kernel_under_test, output_directory)
    elif mode in (TraceMode.CompileOnly, TraceMode.CompileAndInfer):
        compile_kernel_to_neff(kernel_under_test, output_directory)
    else:  # Simulator
        assert (
            False
        ), f"Simulator mode not yet supported. Please use {TraceMode.CompileOnly} or {TraceMode.CompileAndInfer}"
        # kwargs.update(kernel_input)
        # return simulate_kernel(kernel_func, **kwargs)
