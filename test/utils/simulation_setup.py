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
NkiCpuSimulator integration - all simulator-dependent code in one place.

This module is only imported when simulation mode is active. If NkiCpuSimulator
is not available, imports will fail with a clear error message.
"""

import builtins
import logging
import os
import sys

import numpy as np

# NkiCpuSimulator is optional - if not installed, simulation mode will fail gracefully.
# This is expected when running on systems without the simulator package.
try:
    import nki_cpu_simulator.nki.builtin as sim_builtin
    import nki_cpu_simulator.nki.compiler as sim_ncc
    import nki_cpu_simulator.nki.isa as sim_nisa
    import nki_cpu_simulator.nki.isa.constants as sim_nisa_constants
    import nki_cpu_simulator.nki.language as sim_nl
    import nki_cpu_simulator.nki.language.dtypes as sim_dtypes
    import nki_cpu_simulator.nki.tensor as sim_ntensor
    import nki_cpu_simulator.nki.typing as sim_nt
    from nki_cpu_simulator import nki as sim_nki

    SIMULATOR_AVAILABLE = True
except ImportError as e:
    SIMULATOR_AVAILABLE = False
    _IMPORT_ERROR = e

from .common_dataclasses import GoldenTensorDict, normalize_golden_output
from .simulation_constants import SIMULATION_RUN_ALL_ENV_VAR

# Patterns to identify tests with large shapes that are slow on CPU simulation
_LARGE_SHAPE_PATTERNS = ("4096", "5120", "8192", "16384", "32768", "36864")


def setup_simulation_mode():
    """Setup simulation mode: alias nki modules and configure environment."""
    if not SIMULATOR_AVAILABLE:
        raise ImportError(f"NkiCpuSimulator not available: {_IMPORT_ERROR}")

    # Limit BLAS threading in xdist workers to avoid contention
    if "PYTEST_XDIST_WORKER" in os.environ:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    # Alias nki modules to use NkiCpuSimulator
    builtins.builtin = sim_builtin
    sys.modules["nki"] = sim_nki
    sys.modules["nki.language"] = sim_nl
    sys.modules["nki.isa"] = sim_nisa
    sys.modules["nki.isa.constants"] = sim_nisa_constants
    sys.modules["nki.typing"] = sim_nt
    sys.modules["nki.compiler"] = sim_ncc
    sys.modules["nki.dtype"] = sim_dtypes
    sys.modules["nki.tensor"] = sim_ntensor


def skip_slow_simulation_tests(items, skip_marker):
    """Mark tests with large tensor shapes for skipping. Set NKILIB_SIMULATION_RUN_ALL=1 to run all."""
    if os.environ.get(SIMULATION_RUN_ALL_ENV_VAR) == "1":
        return
    for item in items:
        if item.get_closest_marker("slow_simulation") or any(pattern in item.name for pattern in _LARGE_SHAPE_PATTERNS):
            item.add_marker(skip_marker)


def simulate_kernel(kernel_func, kernel_input: dict, lnc_count: int) -> list:
    """Execute kernel using NkiCpuSimulator."""
    import nki

    # Strip ".must_alias_input" suffix from parameter names - this suffix is added
    # for the graph compiler but NkiCpuSimulator expects original param names
    cleaned_input = {k.removesuffix(".must_alias_input"): v for k, v in kernel_input.items()}
    result = nki.cpu_simulate(kernel_func)[lnc_count](**cleaned_input)

    if result is None:
        return []
    elif isinstance(result, (list, tuple)):
        return list(result)
    else:
        return [result]


def run_simulator_inference(kernel_under_test) -> dict[str, np.ndarray]:
    """Run kernel using NkiCpuSimulator and return outputs.

    Returns dict mapping output names to numpy arrays, ready for dumping/validation.
    """
    logging.info("Running kernel via NkiCpuSimulator")

    os.environ["NKI_NC_VERSION"] = kernel_under_test.compiler_input.platform_target.get_nc_gen()
    kernel_outputs = simulate_kernel(
        kernel_under_test.kernel_func,
        kernel_under_test.kernel_input or {},
        kernel_under_test.compiler_input.logical_nc_config,
    )

    if kernel_under_test.validation_args is None:
        raise ValueError("Simulation mode requires validation_args to verify outputs")

    golden_output = kernel_under_test.validation_args.golden_output
    golden_tensors: GoldenTensorDict = normalize_golden_output(golden_output)
    output_names: list[str] = list(golden_tensors.keys())

    if len(kernel_outputs) != len(output_names):
        raise RuntimeError(
            f"Kernel returned {len(kernel_outputs)} outputs but expected {len(output_names)}. "
            f"Got: {[type(o).__name__ for o in kernel_outputs]}, Expected names: {output_names}"
        )

    # Convert kernel outputs to numpy arrays with dtypes matching golden tensors
    return {
        name: np.asarray(output, dtype=golden.dtype)
        for name, output, golden in zip(output_names, kernel_outputs, golden_tensors.values())
    }
