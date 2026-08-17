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
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from types import CodeType
from typing import Any, Callable, Mapping, Optional, Protocol, TextIO, runtime_checkable

from typing_extensions import override

# NOTE: use a sys.version_info guard rather than try/except ImportError. mypy
# (run with --python-version 3.10) cannot resolve a try/except import of
# enum.StrEnum and falls back to typeshed, making members resolve to plain str.
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    # Python 3.10 fallback: create a StrEnum-like base that uses value-based
    # equality and hashing, making members safe for cross-module dict lookups.
    class StrEnum(str, Enum):  # type: ignore[no-redef]
        def __str__(self) -> str:
            return self.value

        def __eq__(self, other) -> bool:
            if isinstance(other, str):
                return str.__eq__(self.value, other)
            return NotImplemented

        def __hash__(self) -> int:
            return hash(self.value)


import numpy.typing as npt

from .metrics_collector import IMetricsCollector

# Directory name for inference artifacts
INF_ARTIFACT_DIR_NAME = "infer_result"

# Directory name for profiler parquet output
PROFILER_DB_DIR_NAME = "profiler_db"

# The compiled NEFF and separated NEFF derived from the single compile version separated pass.
# Hoisting writes a new file rather than modifying NEFF_NAME in place, so each NEFF stays
# paired with the NTFF of the inference that ran against it.
NEFF_NAME = "file.neff"
SEPARATED_NEFF_NAME = "file-separated.neff"

# Test type constants for parametrized test filtering
# MODEL_TEST_TYPE is used to identify model-derived test configs for weekly regression runs
MODEL_TEST_TYPE = "MODEL_WIP"


# Env var pytest-xdist sets in each worker process (value is the worker id, e.g. "gw0");
# absent in the controller/master process.
PYTEST_XDIST_WORKER_ENV = "PYTEST_XDIST_WORKER"


def is_xdist_worker() -> bool:
    """True if running inside a pytest-xdist worker process (False in the controller, or
    when not running under xdist at all)."""
    return PYTEST_XDIST_WORKER_ENV in os.environ


def resolve_probe_worker_count(num_hosts: int, max_probe_workers: int | None = None) -> int:
    """Thread-pool size for probing ``num_hosts`` hosts concurrently over SSH.

    Bounds the pool by the run's parallelism (``max_probe_workers`` — the xdist
    ``--maxprocesses`` cap threaded in from config), falling back to the CPU count when it is
    unset or non-positive, and never exceeds ``num_hosts`` nor drops below 1."""
    if not max_probe_workers or max_probe_workers <= 0:
        max_probe_workers = os.cpu_count() or 1
    return max(1, min(num_hosts, max_probe_workers))


class ModelTestType(Enum):
    """Classification for model test configs.

    Model config files use a dict keyed by ModelTestType:
        model_configs = {
            ModelTestType.BROAD: [ [param1, ...], ... ],
            ModelTestType.GENERALITY: [ [param1, ...], ... ],
        }
    """

    GENERALITY = "GENERALITY"
    OPTIMAL = "OPTIMAL"
    BROAD = "BROAD"
    TIER0 = "TIER0"

    @property
    def test_id_prefix(self) -> str:
        """Return the prefix used in pytest test IDs, e.g. 'BROAD'."""
        return self.value

    @property
    def pytest_mark(self) -> str:
        """Return the pytest marker name for this tier, e.g. 'tier0'."""
        return self.value.lower()


def is_model_test_type(test_type: str) -> bool:
    """Check if a test_type string represents any model test type."""
    model_prefixes = tuple(t.value for t in ModelTestType)
    return test_type.startswith(model_prefixes)


def get_test_tier(node) -> ModelTestType | None:
    """Return the ModelTestType for a pytest node based on its tier marks.

    Args:
        node: A pytest Item (or any object with get_closest_marker).

    Returns:
        The matching ModelTestType, or None if no tier mark is present.

    Raises:
        ValueError: If the node has more than one tier mark.
    """
    matched = [tier for tier in ModelTestType if node.get_closest_marker(tier.pytest_mark)]
    if len(matched) > 1:
        names = [t.pytest_mark for t in matched]
        raise ValueError(f"Test has multiple tier marks: {names}. Expected at most one.")
    return matched[0] if matched else None


def _iter_model_configs(configs):
    """Iterate over model configs yielding (ModelTestType, params, supported_platforms) triples.

    Each entry in the config list can be:
      - A plain params object (list, dataclass, etc.) — runs on all platforms.
      - A (params, supported_platforms) tuple — runs only on the specified platforms.

    Supports both dict format {ModelTestType: [configs...]} and legacy flat list format.
    """
    if isinstance(configs, dict):
        for model_type, entries in configs.items():
            for entry in entries:
                params, platforms = _unpack_platform_config(entry)
                yield model_type, params, platforms
    else:
        for entry in configs:
            params, platforms = _unpack_platform_config(entry)
            yield ModelTestType.BROAD, params, platforms


def _unpack_platform_config(entry):
    """Unpack a config entry that may carry platform restrictions.

    Returns (params, supported_platforms). supported_platforms is None if unrestricted.
    A platform-restricted entry is a (params, set_of_platforms) tuple.
    """
    if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], set):
        return entry[0], entry[1]
    return entry, None


def prepare_model_parametrize(configs, id_formatter=None):
    """Produce (params_list, ids_list) for pytest.mark.parametrize from model configs.

    Args:
        configs: Dict {ModelTestType: [configs...]} or legacy flat list.
                 Entries may be plain params or (params, supported_platforms) tuples.
        id_formatter: Optional callable(params) -> str for the param portion of the ID.
                      Defaults to joining str(p) with '-'.

    Returns:
        (params_list, ids_list) tuple ready for @pytest.mark.parametrize(..., params_list, ids=ids_list).
        When an entry has platform restrictions, it is wrapped in pytest.param with platform marks
        so that pytest -m filtering works at collection time.
    """
    import pytest

    if id_formatter is None:

        def id_formatter(params):
            return "-".join(str(p.value) if hasattr(p, "value") else str(p) for p in params)

    params_list = []
    ids_list = []
    for model_type, params, supported_platforms in _iter_model_configs(configs):
        if supported_platforms is not None:
            excluded = set(Platforms) - supported_platforms
            marks = [pytest.mark.platforms(exclude=list(excluded))]
            params_list.append(pytest.param(*params, marks=marks))
        else:
            params_list.append(params)
        ids_list.append(f"{model_type.test_id_prefix}_{id_formatter(params)}")
    return params_list, ids_list


def unpack_model_config(entry):
    """Unpack a model config entry that may be a (ModelTestType, params) tuple or raw params.

    Returns (ModelTestType, params). Raw params default to ModelTestType.BROAD.
    """
    if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], ModelTestType):
        return entry[0], entry[1]
    return ModelTestType.BROAD, entry


class TraceMode(Enum):
    CompileAndInfer = "compile_and_infer"
    CompileAndInferAndSeparate = "compile_and_infer_and_separate"
    CompileOnly = "compile_only"
    TraceOnly = "trace_only"
    Simulator = "simulation"
    Debugger = "debugger"
    NkiStandaloneTracer = "nki_standalone_tracer"
    NkiStandaloneParser = "nki_standalone_parser"

    @staticmethod
    def create(mode: str):
        return TraceMode(mode.lower().replace("-", "_"))


class HostProvisioningMode(StrEnum):
    """How a run obtains the hosts its tests execute on."""

    LOCAL = "local"
    """tests run on this machine's own Neuron chip (no --target-* given)."""
    STATIC_HOSTS = "static_hosts"
    """a fixed host list via --target-host; the test process drives them
       over SSH locally (no controller-side store bootstrap)."""
    STATIC_FILE = "static_file"
    """a fixed, possibly-heterogeneous host list via --target-host-file;
       a remote run whose membership the workers initialize from the file."""
    PLUGIN_PROVISIONED = "plugin_provisioned"
    """hosts supplied by a provisioning plug-in, which reports it via
       HostProvisioningResult.plugin_provisioned (see maybe_setup_shared_fleet)."""


@dataclass(frozen=True)
class HostProvisioningResult:
    """What a host-provisioning plug-in did for this run, returned to the controller."""

    plugin_provisioned: bool
    """A provisioning plug-in claimed this run's host pool (supplied a remote, store-backed
    set of hosts). Drives HostProvisioningMode.PLUGIN_PROVISIONED."""
    recoverable: bool = False
    """The host pool can regain hosts mid-run (a background refresher re-resolves it), so a
    failed claim should wait for a host rather than fail immediately."""


class NKICompilationMode(Enum):
    parser = "parser"
    tracer = "tracer"


class SeparationPassMode(Enum):
    NONE = "none"
    DIRECT = "direct"
    INDIRECT = "indirect"
    INDIRECT_SERIALIZE = "indirect-serialize"  # Indirect separation with ordered edges between all moved loads


class UploadProfileMode(Enum):
    ALWAYS = "always"
    ON_FAIL_ONLY = "on-fail-only"

    @staticmethod
    def from_str(value: str) -> "UploadProfileMode":
        return UploadProfileMode(value.lower())


class CleanupPreserve(Enum):
    """Artifact types to preserve when using --force-local-cleanup."""

    METRICS = "metrics"


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
    computation_time: float | None = field(init=False, default=None)

    @property
    def golden(self) -> dict[str, npt.NDArray[Any]] | None:
        if self.lazy_golden_generator is None:
            return None

        if self.__cached_golden__ is None:
            start = time.perf_counter()
            self.__cached_golden__ = self.lazy_golden_generator()
            self.computation_time = time.perf_counter() - start

        return self.__cached_golden__


class PerRankGenerator(Protocol):
    """A callable that produces a per-rank input mapping, keyed by a named ``rank_id``."""

    def __call__(self, *, rank_id: int) -> dict[str, Any]: ...


@dataclass
class PerRankLazyGenerator:
    """Base class for per-rank lazy generators with caching."""

    generator: PerRankGenerator
    __cache__: dict[int, dict[str, Any]] = field(init=False, default_factory=dict)
    computation_times: dict[int, float] = field(init=False, default_factory=dict)

    def for_rank(self, rank_id: int) -> dict[str, Any]:
        if rank_id not in self.__cache__:
            start = time.perf_counter()
            try:
                self.__cache__[rank_id] = self.generator(rank_id=rank_id)
            except TypeError as e:
                # Only catch TypeError from wrong parameter name, not from inside the generator
                if "unexpected keyword argument" in str(e) and "rank_id" in str(e):
                    code = getattr(self.generator, "__code__", None)
                    location = f" at {code.co_filename}:{code.co_firstlineno}" if isinstance(code, CodeType) else ""
                    raise TypeError(f"Generator{location} must use 'rank_id' as parameter name") from e
                raise
            self.computation_times[rank_id] = time.perf_counter() - start
        return self.__cache__[rank_id]


@dataclass
class PerRankLazyInputGenerator(PerRankLazyGenerator):
    """Generates inputs per rank for collectives tests.

    Example:
        def create_inputs(rank_id):
            return {"x_in": data[rank_id], "G": 8}

        kernel_input=PerRankLazyInputGenerator(create_inputs)
    """

    # Set by the collective harness to the rank-0 input mapping, for callers that
    # need a representative shape/dtype without materialising every rank. Defaults
    # to empty so the harness can construct first and assign it right after.
    base_input: dict[str, Any] = field(default_factory=dict)
    input_artifacts_directory: str | None = None


@dataclass
class PerRankLazyGoldenGenerator(PerRankLazyGenerator):
    """Lazy golden generator for multi-rank collective tests.

    Wraps a callable that runs torch refs across all ranks via SimDistRunner.
    Only created in hardware (compile-and-infer) mode. In compile-only mode,
    the framework passes validation_args=None instead, skipping golden gen.

    output_keys: List of output tensor names returned by the torch ref.
    Required so the framework knows output names without calling the generator,
    enabling compile-only mode to build output descriptors for the NEFF.
    """

    output_keys: list = field(default_factory=list)

    def for_rank(self, rank_id: int) -> dict:
        return super().for_rank(rank_id)


# Type aliases for golden output handling:
# - GoldenTensorMapping: The normalized form - output names mapped to tensors or validators.
#   A read-only Mapping rather than a dict, because Mapping is covariant in its value type
#   and so also accepts a narrower dict (e.g. one holding only validators).
# - GoldenOutputType: All possible input forms that can be normalized to GoldenTensorMapping
GoldenTensorMapping = Mapping[str, npt.NDArray[Any] | CustomValidatorWithOutputTensorData]
GoldenOutputType = LazyGoldenGenerator | PerRankLazyGoldenGenerator | GoldenTensorMapping


def normalize_golden_output(golden_output: GoldenOutputType, rank_id: int = 0) -> GoldenTensorMapping:
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
    equal_nan_inf: bool = False

    def __post_init__(self):
        error_message: str = f"ValidationArgs only supports LazyGoldenGenerator, PerRankLazyGoldenGenerator, or dict[str, CustomValidatorWithOutputTensorData], but got {type(self.golden_output)}"
        if isinstance(self.golden_output, (LazyGoldenGenerator, PerRankLazyGoldenGenerator)):
            pass
        elif isinstance(self.golden_output, dict):
            for key, value in self.golden_output.items():
                assert isinstance(key, str) and isinstance(value, CustomValidatorWithOutputTensorData), error_message
        else:
            raise AssertionError(error_message)


class Platforms(StrEnum):
    """Target hardware platforms for NKI kernel compilation and execution.

    Uses StrEnum so that members use string-based equality and hashing. This is
    critical because the same module can be loaded under two paths (test.utils.*
    and nkilib_testing.*) when the pytest11 entry point is active — StrEnum
    ensures cross-module dict lookups and set operations work correctly.
    """

    TRN1 = "trn1"
    TRN2 = "trn2"
    TRN3 = "trn3"
    TRN3_A0 = "trn3_a0"
    TRN3_PDS = "trn3_pds"
    TRN3_PDS_A0 = "trn3_pds_a0"

    @override
    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_str_safe(cls, value: str) -> "Platforms | None":
        """Parse a platform string, returning None (with a warning) for an
        unrecognized value. Use ``Platforms(value)`` for strict."""
        try:
            return cls(value)
        except ValueError:
            logging.getLogger(__name__).warning("Unknown platform %r; treating as None", value)
            return None

    def is_trn3(self) -> bool:
        return self in (Platforms.TRN3, Platforms.TRN3_A0, Platforms.TRN3_PDS, Platforms.TRN3_PDS_A0)

    def get_compile_target(self) -> str:
        if self in (Platforms.TRN3_A0, Platforms.TRN3_PDS_A0):
            return "trn3pre"
        elif self == Platforms.TRN3_PDS:
            return Platforms.TRN3.value
        else:
            return self.value

    def get_nc_gen(self) -> str:
        # Imported lazily: nki is not installed in every environment that imports this module.
        try:
            import nki.isa as nisa
        except ModuleNotFoundError as exc:
            raise ImportError("nki.isa is required for get_nc_gen() but is not installed") from exc

        gen_map = {
            Platforms.TRN1: nisa.nc_version.gen2,
            Platforms.TRN2: nisa.nc_version.gen3,
            Platforms.TRN3: nisa.nc_version.gen4,
            Platforms.TRN3_A0: nisa.nc_version.gen4,
            Platforms.TRN3_PDS: nisa.nc_version.gen4,
            Platforms.TRN3_PDS_A0: nisa.nc_version.gen4,
        }

        return gen_map[self].name


@runtime_checkable
class PlatformAware(Protocol):
    """Protocol for test config objects that declare platform restrictions.

    Any dataclass or object with a ``supported_platforms`` attribute satisfying
    this signature will be recognized by ``pytest_collection_modifyitems`` and
    used to restrict which platform marks are applied to the test item.

    When ``supported_platforms`` is ``None``, the test runs on all platforms.
    Otherwise, it should be a set of :class:`Platforms` values the test supports.
    """

    supported_platforms: set[Platforms] | None


@dataclass
class TargetHost:
    """Represents a target host for test execution."""

    ssh_host: str
    host_type: Platforms


@dataclass(frozen=True)
class ResolvedHost:
    """A single provisioned host.

    ``ssh_host`` is the SSH alias the test harness connects to (e.g. the host's
    public IP). ``host_type`` is the platform that this host supports.
    ``num_physical_cores`` is the host's total physical-core capacity, probed by the
    source when it resolves the host (0 if unknown / not probed); it drives capacity-based
    routing in the host-state store — a host with fewer cores than a request needs is
    ineligible for it.
    """

    ssh_host: str
    host_type: Platforms
    num_physical_cores: int = 0


class InstanceSize(StrEnum):
    """A fleet host's instance size.

    ``UNKNOWN`` covers a size not enumerated here (or an absent one)."""

    X_48XLARGE = "48xlarge"
    X_3XLARGE = "3xlarge"
    UNKNOWN = "unknown"

    @override
    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value: str | None) -> "InstanceSize":
        """Parse an instance-size string, mapping an unrecognized/absent value to ``UNKNOWN``
        (with a warning) so a new size is flagged for a real core-count assignment rather than
        silently mis-weighted."""
        try:
            return cls(value)
        except ValueError:
            logging.getLogger(__name__).warning(
                "Unknown instance size %r; treating as UNKNOWN (1 core). Add it to InstanceSize "
                "with its real core count so the capacity gate weights it correctly",
                value,
            )
            return cls.UNKNOWN

    def get_core_count(self) -> int:
        """Nominal physical cores for this size. UNKOWN is treated as 1 (assumed to have at least 1 core)."""
        return {
            InstanceSize.X_48XLARGE: 128,
            InstanceSize.X_3XLARGE: 8,
            InstanceSize.UNKNOWN: 1,
        }[self]


class CompilerArgs:
    logical_nc_config: int
    platform_target: Platforms
    additional_cmd_args: list[str]
    enable_debugging: bool
    enable_birsim: bool
    dump_after_lowering: bool
    separation_pass_mode: SeparationPassMode
    enable_device_dump: bool  # Inserts a device_print after each ISA instruction to dump intermediate tensors. WARNING: Local debug only. High disk/perf impact.
    # Per-test NKI compiler frontend override. None => use the session/CLI default
    # (--nki-compilation-mode, default parser). Set to NKICompilationMode.tracer to
    # pin the tracer frontend for a test regardless of how the suite is invoked.
    nki_compilation_mode: "NKICompilationMode | None"

    def __init__(
        self,
        platform_target: Platforms,
        logical_nc_config: int | None = None,
        additional_cmd_args: list[str] | None = None,
        enable_debugging: bool = False,
        enable_birsim: bool = False,
        dump_after_lowering: bool = False,
        separation_pass_mode: SeparationPassMode = SeparationPassMode.NONE,
        enable_device_dump: bool = False,
        nki_compilation_mode: "NKICompilationMode | None" = None,
    ):
        self.platform_target = platform_target
        self.additional_cmd_args = additional_cmd_args if additional_cmd_args is not None else []
        self.enable_debugging = enable_debugging
        self.enable_birsim = enable_birsim
        self.dump_after_lowering = dump_after_lowering
        self.separation_pass_mode = separation_pass_mode
        self.enable_device_dump = enable_device_dump
        self.nki_compilation_mode = nki_compilation_mode
        if os.environ.get("NKILIB_ENABLE_SEPARATION_ANALYSIS"):
            sep_mode = os.environ.get("NKILIB_ENABLE_SEPARATION_ANALYSIS")
            if sep_mode == "serialize":
                self.separation_pass_mode = SeparationPassMode.INDIRECT_SERIALIZE
            else:
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
            Platforms.TRN3_PDS: 2,
            Platforms.TRN3_PDS_A0: 2,
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
        self.num_runs = self.resolved_num_runs

        # Validate collective_ranks - NRT only supports 1, 2, 4, 8, 16 or multiples of 32
        valid_ranks = {1, 2, 4, 8, 16}
        if self.collective_ranks not in valid_ranks and self.collective_ranks % 32 != 0:
            raise ValueError(
                f"Unsupported collective_ranks={self.collective_ranks}. "
                f"Supported values are 1, 2, 4, 8, 16 or multiples of 32."
            )

    @property
    def resolved_num_runs(self) -> int:
        """Number of executions: 2 when checking determinism, 1 otherwise, unless set explicitly."""
        if self.num_runs is None:
            return 2 if self.enable_determinism_check else 1
        return self.num_runs


# Pre-configured InferenceArgs for TKG kernels with determinism checking
TKG_INFERENCE_ARGS = InferenceArgs(enable_determinism_check=True, num_runs=10)


class NamedCallable(Protocol):
    """A callable that also carries a name, as every plain function does.

    Used where the harness reports which entry point it is working on: a bare
    callable type promises no name, but a function does.
    """

    __name__: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass
class KernelArgs:
    kernel_func: NamedCallable
    compiler_input: CompilerArgs
    # Kernel inputs: either a dict (same for all ranks) or PerRankLazyInputGenerator (per-rank)
    kernel_input: dict[str, Any] | PerRankLazyInputGenerator | None = None
    validation_args: ValidationArgs | None = None
    inference_args: InferenceArgs = field(default_factory=InferenceArgs)
    collector: IMetricsCollector | None = None


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
    instance_type: str = ""
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
            instance_type=data.get("instance_type", ""),
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
