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
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TextIO

import nki.isa as nisa
import numpy.typing as npt
from typing_extensions import override

from .metrics_emitter import IMetricsEmitter

# Directory name for inference artifacts
INF_ARTIFACT_DIR_NAME = "infer_result"

# Test type constants for parametrized test filtering
# MODEL_TEST_TYPE is used to identify model-derived test configs for weekly regression runs
MODEL_TEST_TYPE = "MODEL_WIP"


class TraceMode(Enum):
    CompileAndInfer = "compile_and_infer"
    CompileOnly = "compile_only"
    TraceOnly = "trace_only"
    Simulator = "simulation"

    @staticmethod
    def create(mode: str):
        return TraceMode(mode.lower().replace("-", "_"))


class SeparationPassMode(Enum):
    NONE = "none"
    DIRECT = "direct"
    INDIRECT = "indirect"


class CustomValidator(ABC):
    """Custom validator for test output."""

    def __init__(self, logfile: TextIO | None = None):
        self.logfile: TextIO | None = logfile
        self.logger: logging.Logger = logging.getLogger(__name__)

    def _print_with_log(self, message: str):
        """Helper method for printing inside a validator for output in logger and logfile."""
        self.logger.info(message)
        if self.logfile is not None:
            print(message, file=self.logfile)

    @abstractmethod
    def validate(self, inference_output: npt.NDArray[Any]) -> bool:
        """Given the inference_output, return true if it is correct and false otherwise"""
        pass


@dataclass
class CustomValidatorWithOutputTensorData:
    """Custom validator augmented with tensor to specify output shape and dtype."""

    validator: type[CustomValidator]
    output_ndarray: npt.NDArray[Any]

    @property
    def dtype(self):
        return self.output_ndarray.dtype


@dataclass
class LazyGoldenGenerator:
    """Custom validator augmented with tensor to specify output shape and dtype."""

    # required in order to let compiler know what to expect as output tensors
    output_ndarray: dict[str, npt.NDArray[Any]]

    # could be None to disable validation of outputs
    lazy_golden_generator: Callable[[], dict[str, npt.NDArray[Any]]] | None = None

    __cached_golden__: dict[str, npt.NDArray[Any]] | None = field(init=False, default=None)

    @property
    def golden(self) -> dict[str, npt.NDArray[Any]] | None:
        if self.lazy_golden_generator is None:
            return None

        if self.__cached_golden__ is None:
            self.__cached_golden__ = self.lazy_golden_generator()

        return self.__cached_golden__


@dataclass
class PerRankLazyGenerator:
    """Base class for per-rank lazy generators with caching."""

    generator: Callable[[int], dict[str, Any]]
    __cache__: dict[int, dict[str, Any]] = field(init=False, default_factory=dict)

    def for_rank(self, rank_id: int) -> dict[str, Any]:
        if rank_id not in self.__cache__:
            try:
                self.__cache__[rank_id] = self.generator(rank_id=rank_id)
            except TypeError as e:
                # Only catch TypeError from wrong parameter name, not from inside the generator
                if "unexpected keyword argument" in str(e) and "rank_id" in str(e):
                    location = ""
                    if hasattr(self.generator, "__code__"):
                        code = self.generator.__code__
                        location = f" at {code.co_filename}:{code.co_firstlineno}"
                    raise TypeError(f"Generator{location} must use 'rank_id' as parameter name") from e
                raise
        return self.__cache__[rank_id]


class PerRankLazyInputGenerator(PerRankLazyGenerator):
    """Generates inputs per rank for collectives tests.

    Example:
        def create_inputs(rank_id):
            return {"x_in": data[rank_id], "G": 8}

        kernel_input=PerRankLazyInputGenerator(create_inputs)
    """


class PerRankLazyGoldenGenerator(PerRankLazyGenerator):
    """Generates golden values per rank for collectives tests.

    Example:
        def create_golden(rank_id):
            return {"out": expected[rank_id]}

        golden_output=PerRankLazyGoldenGenerator(create_golden)
    """


# Type aliases for golden output handling:
# - GoldenTensorDict: The normalized form - a dict mapping output names to tensors or validators
# - GoldenOutputType: All possible input forms that can be normalized to GoldenTensorDict
GoldenTensorDict = dict[str, npt.NDArray[Any] | CustomValidatorWithOutputTensorData]
GoldenOutputType = LazyGoldenGenerator | PerRankLazyGoldenGenerator | GoldenTensorDict


def normalize_golden_output(golden_output: GoldenOutputType, rank_id: int = 0) -> GoldenTensorDict:
    """Normalize golden_output to a dict of output name -> tensor.

    Handles all golden output types consistently:
    - LazyGoldenGenerator: returns output_ndarray (shape/dtype placeholders)
    - PerRankLazyGoldenGenerator: returns tensors for specified rank
    - Plain dict: returns as-is
    """
    if isinstance(golden_output, PerRankLazyGoldenGenerator):
        return golden_output.for_rank(rank_id)
    elif isinstance(golden_output, LazyGoldenGenerator):
        return golden_output.output_ndarray
    else:
        return golden_output


@dataclass
class ValidationArgs:
    """
    Attributes:
        relative_accuracy: relative accuracy needs to be between 0 and 1
    """

    golden_output: LazyGoldenGenerator | PerRankLazyGoldenGenerator | dict[str, CustomValidatorWithOutputTensorData]

    relative_accuracy: float = 1e-05
    absolute_accuracy: float = 1e-08

    accuracy_buffer_percent: int | None = 0

    def __post_init__(self):
        error_message: str = f"ValidationArgs only supports LazyGoldenGenerator, PerRankLazyGoldenGenerator, or dict[str, CustomValidatorWithOutputTensorData], but got {self.golden_output}"
        if isinstance(self.golden_output, (LazyGoldenGenerator, PerRankLazyGoldenGenerator)):
            pass
        elif isinstance(self.golden_output, dict):
            for key, value in self.golden_output.items():
                assert isinstance(key, str) and isinstance(value, CustomValidatorWithOutputTensorData), error_message
        else:
            assert False, error_message


class Platforms(Enum):
    TRN1 = "trn1"
    TRN2 = "trn2"
    TRN3 = "trn3"
    TRN3_A0 = "trn3_a0"

    @override
    def __str__(self) -> str:
        return self.value

    def is_trn3(self) -> bool:
        return self in (Platforms.TRN3, Platforms.TRN3_A0)

    def get_compile_target(self) -> str:
        if self == Platforms.TRN3_A0:
            return Platforms.TRN3.value
        else:
            return self.value

    def get_nc_gen(self) -> str:
        gen_map = {
            Platforms.TRN1: nisa.nc_version.gen2,
            Platforms.TRN2: nisa.nc_version.gen3,
            Platforms.TRN3: nisa.nc_version.gen4,
            Platforms.TRN3_A0: nisa.nc_version.gen4,
        }

        return gen_map[self].name


@dataclass
class TargetHost:
    """Represents a target host for test execution."""

    ssh_host: str
    host_type: Platforms


class CompilerArgs:
    logical_nc_config: int
    platform_target: Platforms
    additional_cmd_args: list[str]
    enable_debugging: bool
    enable_birsim: bool
    dump_after_lowering: bool
    separation_pass_mode: SeparationPassMode

    def __init__(
        self,
        logical_nc_config: int | None = None,
        platform_target: Platforms = Platforms.TRN2,
        additional_cmd_args: list[str] = [],
        enable_debugging: bool = False,
        enable_birsim: bool = False,
        dump_after_lowering: bool = False,
        separation_pass_mode: SeparationPassMode = SeparationPassMode.NONE,
    ):
        self.platform_target = platform_target
        self.additional_cmd_args = additional_cmd_args
        self.enable_debugging = enable_debugging
        self.enable_birsim = enable_birsim
        self.dump_after_lowering = dump_after_lowering
        self.separation_pass_mode = separation_pass_mode
        if os.environ.get("NKILIB_ENABLE_SEPARATION_ANALYSIS"):
            self.separation_pass_mode = SeparationPassMode.INDIRECT
        if logical_nc_config is None:
            self.logical_nc_config = self.__get_logical_nc_config_for_platform__()
        else:
            self.logical_nc_config = logical_nc_config

    def __get_logical_nc_config_for_platform__(self) -> int:
        platform_to_logical_nc_config: dict[Platforms, int] = {
            Platforms.TRN1: 1,
            Platforms.TRN2: 2,
            Platforms.TRN3: 2,
            Platforms.TRN3_A0: 2,
        }

        return platform_to_logical_nc_config[self.platform_target]


@dataclass
class InferenceArgs:
    """Configuration for inference behavior during test execution."""

    profile_all_runs: bool = False  # Profile all executions or just the last (controls --profile-nth-exec)
    profile_all_ranks: bool = False  # Profile all ranks or just rank 0 (controls --collectives-profile-id)
    enable_determinism_check: bool = False
    num_runs: Optional[int] = None
    collective_ranks: int = (
        1  # Number of collective ranks (each rank is a logical NeuronCore). Default 1 = no collectives
    )
    env_vars: Optional[dict[str, str]] = None

    def __post_init__(self):
        """Set default num_runs based on enable_determinism_check"""
        if self.num_runs is None:
            # Default: 2 for determinism check, 1 otherwise
            self.num_runs = 2 if self.enable_determinism_check else 1

        # Validate collective_ranks - NRT only supports 1, 2, 4, 8, 16 or multiples of 32
        valid_ranks = {1, 2, 4, 8, 16}
        if self.collective_ranks not in valid_ranks and self.collective_ranks % 32 != 0:
            raise ValueError(
                f"Unsupported collective_ranks={self.collective_ranks}. "
                f"Supported values are 1, 2, 4, 8, 16 or multiples of 32."
            )


# Pre-configured InferenceArgs for TKG kernels with determinism checking
TKG_INFERENCE_ARGS = InferenceArgs(enable_determinism_check=True, num_runs=10)


@dataclass
class KernelArgs:
    kernel_func: Callable  # pyright: ignore[reportMissingTypeArgument]
    compiler_input: CompilerArgs = field(default_factory=CompilerArgs)
    # Kernel inputs: either a dict (same for all ranks) or PerRankLazyInputGenerator (per-rank)
    kernel_input: dict[str, Any] | PerRankLazyInputGenerator | None = None
    validation_args: ValidationArgs | None = None
    inference_args: InferenceArgs = field(default_factory=InferenceArgs)
    emitter: IMetricsEmitter | None = None


@dataclass
class NeuronDeviceInfo:
    neuron_device: int
    bdf: str
    cpu_affinity: str
    numa_node: str
    connected_to: list[int] | None
    nc_count: int
    memory_size: int
    neuroncore_ids: list[int]
    neuron_processes: list[dict[str, Any]]
    logical_neuroncore_config: int = (
        2  # future version of compiler and runtime simply default to lnc 2 without explicitly specifying it
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NeuronDeviceInfo":
        """Create NeuronDeviceInfo from dictionary (parsed JSON)."""
        return cls(
            neuron_device=data["neuron_device"],
            bdf=data["bdf"],
            cpu_affinity=data["cpu_affinity"],
            numa_node=data["numa_node"],
            connected_to=data.get("connected_to"),
            nc_count=data["nc_count"],
            logical_neuroncore_config=data.get("logical_neuroncore_config", cls.logical_neuroncore_config),
            memory_size=data["memory_size"],
            neuroncore_ids=data["neuroncore_ids"],
            neuron_processes=data.get("neuron_processes", []),
        )

    def get_memory_size_gb(self) -> float:
        """Get memory size in GB."""
        return self.memory_size / (1024**3)

    def is_in_use(self) -> bool:
        """Check if any processes are using this device."""
        return len(self.neuron_processes) > 0

    def get_core_range(self) -> str:
        """Get core IDs as a range string (e.g., '0-3')."""
        if not self.neuroncore_ids:
            return ""
        min_id = min(self.neuroncore_ids)
        max_id = max(self.neuroncore_ids)
        if max_id - min_id + 1 == len(self.neuroncore_ids):
            return f"{min_id}-{max_id}"
        return ",".join(str(id) for id in self.neuroncore_ids)
