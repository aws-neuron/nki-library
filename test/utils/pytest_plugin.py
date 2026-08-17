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
"""NKI Library Testing — pytest plugin.

Auto-registered via pytest11 entry point. Provides shared CLI options, fixtures,
and hooks for NKI kernel testing.

Fixtures:
    platform_target, trace_mode, output_directory, metric_output_mode,
    collector, emitter, host_manager, perf_analysis_enabled, hw_profile_enabled,
    test_manager

Hooks:
    pytest_addoption, pytest_configure, pytest_generate_tests,
    pytest_collection_modifyitems, pytest_sessionstart, pytest_collection_finish
"""

from __future__ import annotations

import argparse
import functools
import getpass
import logging
import os
import random
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pytest
from _pytest.config import Config
from _pytest.mark import Mark
from _pytest.python import Metafunc
from _pytest.stash import Stash, StashKey

from .common_dataclasses import (
    HostProvisioningMode,
    HostProvisioningResult,
    NKICompilationMode,
    PlatformAware,
    Platforms,
    ResolvedHost,
    TargetHost,
    TraceMode,
    UploadProfileMode,
    get_test_tier,
)
from .coverage_parametrized_tests import extract_parametrize_args, generate_parametrized_test_case
from .feature_flag_helper import (
    derive_pytest_test_id,
    get_feature_flag,
    get_platform_targets,
    resolve_base_output_directory,
    resolve_ssh_config_path,
)
from .host_management import HostManager, detect_local_neuron_devices
from .host_state import HostStateStore
from .metrics_collector import IMetricsCollector, MetricsCollector, NoopMetricsCollector
from .metrics_emitter import IMetricsEmitter, MetricsEmitter, NoopMetricsEmitter, OutputMode, SessionContext
from .param_extractor import compute_params_hash, derive_test_method_id, extract_pytest_params, normalize_param_names
from .pytest_test_metadata import discover_pytest_test_metadata_marks, resolve_file_kernel_name
from .s3_utils import S3ArtifactUploadConfig
from .simulation_setup import setup_simulation_mode
from .suite_chunking import get_chunked_tests
from .test_orchestrator import Orchestrator

_RNG_SEED_ENV_KEY = "NEURON_PYTHONHASHSEED"

logging.getLogger("paramiko.transport").setLevel(logging.WARNING)


@runtime_checkable
class _CallSpec(Protocol):
    """The parameter mapping of one parametrized case."""

    params: dict[str, Any]


@runtime_checkable
class _ParametrizedItem(Protocol):
    """A collected test produced by parametrization, which carries its case's parameters.

    Non-parametrized tests do not have this attribute at all, so presence of the
    attribute is exactly the "is parametrized" test.
    """

    callspec: _CallSpec


class _StashHolder(Protocol):
    """The only thing session-state writers need from a pytest config: its stash."""

    @property
    def stash(self) -> Stash: ...


class _MarkerSource(Protocol):
    """The only thing marker inspection needs from a collected test: its resolved markers."""

    def iter_markers(self, name: str | None = None) -> Iterator[Mark]: ...


# ─── Session state on ``config`` ───
# All per-session state we attach to pytest's ``config`` object lives here as typed ``StashKey``s.
SESSION_TRACE_MODE_KEY: StashKey[TraceMode] = StashKey()
"""Resolved once from --test-mode + host mode; the trace mode the session runs and reports."""
HOST_PROVISIONING_MODE_KEY: StashKey[HostProvisioningMode] = StashKey()
"""How the run obtains hosts. Lazily CLI-derived; a provisioning plug-in overrides it to
PLUGIN_PROVISIONED (see apply_host_provisioning_result)."""
HOST_RECOVERABLE_KEY: StashKey[bool] = StashKey()
"""Whether the host pool can regain hosts mid-run (only ever set on the plug-in path)."""
TESTRUN_UID_KEY: StashKey[str] = StashKey()
"""Controller-generated session id, propagated to xdist workers via --testrunuid."""
QOR_SESSION_ID_KEY: StashKey[str] = StashKey()
"""Timestamp id grouping this run's QoR CSV output (controller-set, bridged to workers)."""
SESSION_CONTEXT_KEY: StashKey[SessionContext] = StashKey()
"""Immutable session-level metric dimensions (target, trace mode, run type, …)."""
RELEVANT_TEST_DIRS_KEY: StashKey[set[str] | None] = StashKey()
"""When --run-relevant-tests is active, the dir set to filter collection to (None = all)."""
CONTROLLER_SETUP_COLLECTOR_KEY: StashKey[IMetricsCollector] = StashKey()
"""Controller-side metrics collector for work done during session setup, before workers spawn.
The controller has no per-test emit cycle, so setup-phase metrics are recorded here."""


# ─── Helper functions ───


def _single_compile_separation_requested() -> bool:
    """Whether this run requested the single compile DMA-hoisting separation flow.

    The request is an env var rather than a pytest flag. When the implementing module is
    unavailable, the import fails and the request degrades to False.
    """
    try:
        from .single_compile_separation_private import separation_requested
    except ImportError:
        return False
    return separation_requested()


def resolve_session_trace_mode(config: Config) -> TraceMode:
    """Resolve session trace mode from CLI flags. Result is cached on config."""
    if SESSION_TRACE_MODE_KEY in config.stash:
        return config.stash[SESSION_TRACE_MODE_KEY]

    test_mode = get_feature_flag(config, "test_mode")
    if _single_compile_separation_requested():
        if test_mode:
            raise pytest.UsageError("The single compile separation pass flow cannot be combined with --test-mode")
        mode = TraceMode.CompileAndInferAndSeparate
    elif test_mode:
        mode = TraceMode.create(test_mode)
    elif not is_local_run(config):
        mode = TraceMode.CompileAndInfer
    else:
        neuron_installation_path = get_feature_flag(config, "neuron_tools_bin_path")
        if detect_local_neuron_devices(neuron_installation_path):
            mode = TraceMode.CompileAndInfer
            logging.info("Local Neuron devices detected, using CompileAndInfer mode")
        else:
            mode = TraceMode.CompileOnly

    config.stash[SESSION_TRACE_MODE_KEY] = mode
    return mode


def is_simulation_mode(config: Config) -> bool:
    """Check if simulation mode is active."""
    return resolve_session_trace_mode(config) == TraceMode.Simulator


def is_debugger_mode(config: Config) -> bool:
    """Check if debugger mode is active."""
    return resolve_session_trace_mode(config) == TraceMode.Debugger


def apply_host_provisioning_result(config: _StashHolder, result: HostProvisioningResult) -> None:
    """Decompose a plug-in's :class:`HostProvisioningResult` onto ``config`` — the single place
    config learns what a plug-in did. Folds ``plugin_provisioned`` into the provisioning mode
    (so there is one representation of it, not a separate flag) and records recoverability.
    A non-provisioning result is a no-op: the mode stays lazily CLI-derived and the run keeps
    its non-recoverable default."""
    if result.plugin_provisioned:
        config.stash[HOST_PROVISIONING_MODE_KEY] = HostProvisioningMode.PLUGIN_PROVISIONED
        config.stash[HOST_RECOVERABLE_KEY] = result.recoverable


def resolve_host_provisioning_mode(config: Config) -> HostProvisioningMode:
    """Classify how this run obtains its hosts, once (cached on config). This is the
    single source of truth that the run-mode predicates below project from, so the
    classification logic isn't duplicated across conftest helpers and make_host_manager.

    Derived from the public CLI signals.

    Precedence among the CLI signals: a static host file wins over a static host list, which
    wins over local. A plug-in claim (the override) supersedes all of them, matching the
    controller layering (a provisioning plug-in claims the run before static hosts)."""
    if HOST_PROVISIONING_MODE_KEY in config.stash:
        return config.stash[HOST_PROVISIONING_MODE_KEY]

    if get_feature_flag(config, "target_host_file"):
        mode = HostProvisioningMode.STATIC_FILE
    elif get_feature_flag(config, "target_host", []):
        mode = HostProvisioningMode.STATIC_HOSTS
    else:
        mode = HostProvisioningMode.LOCAL

    config.stash[HOST_PROVISIONING_MODE_KEY] = mode
    return mode


def is_local_run(config: Config) -> bool:
    """True when tests run on this machine's own chip (no remote hosts of any kind)."""
    return resolve_host_provisioning_mode(config) is HostProvisioningMode.LOCAL


def is_remote_bootstrap_run(config: Config) -> bool:
    """True when the controller bootstraps host_state.json before workers run: the
    static-file path, or a plug-in-provisioned pool. NOT the --target-host list (that
    drives a fixed set over SSH from a local-style session with no controller-side store
    setup), and NOT plain local. All bootstrap paths imply CompileAndInfer."""
    return resolve_host_provisioning_mode(config) in (
        HostProvisioningMode.STATIC_FILE,
        HostProvisioningMode.PLUGIN_PROVISIONED,
    )


def is_recoverable_run(config: Config) -> bool:
    """True when the host pool can gain or regain hosts mid-run, so a failed claim should
    wait rather than fail immediately. Set by the provisioning plug-in (which runs the
    background re-resolver) via apply_host_provisioning_result; only a plug-in-provisioned
    run is ever recoverable, so recoverability is stored only on that path (default False)."""
    return config.stash.get(HOST_RECOVERABLE_KEY, False)


@functools.lru_cache(maxsize=None)
def resolve_current_user(default: str | None = None) -> str | None:
    """Return the current OS user, falling back to ``default`` if unresolved.

    Result is cached per ``default`` value.
    """
    try:
        return getpass.getuser()
    except Exception:
        return default


@functools.lru_cache(maxsize=None)
def resolve_git_short_sha(default: str | None = None) -> str | None:
    """Return the short SHA of HEAD, falling back to ``default`` if unresolved.

    Result is cached per ``default`` value.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        return sha or default
    except Exception:
        return default


def get_pytest_mark_names(node: _MarkerSource) -> list[str]:
    """Return all resolved pytest marker names in deterministic order."""
    return sorted({marker.name for marker in node.iter_markers()})


def make_collector(
    request: pytest.FixtureRequest,
    metric_output_mode: OutputMode | None,
    namespace: str = "NeuronCompiler",
) -> IMetricsCollector:
    """Create a metrics collector. Returns NoopMetricsCollector when metrics are disabled."""
    if metric_output_mode is None:
        return NoopMetricsCollector()

    collector = MetricsCollector()
    collector.set_namespace(namespace)

    if hasattr(request.node, "callspec"):
        raw_params = request.node.callspec.params
        params = extract_pytest_params(raw_params)
        params = normalize_param_names(params)
        collector.set_kernel_params(params)
        collector.add_dimension({"TestParamsHash": compute_params_hash(raw_params)})
    else:
        collector.add_dimension({"TestParamsHash": compute_params_hash({})})

    # Emit stable test method identifier: module::class::method
    collector.add_dimension({"TestMethodId": derive_test_method_id(request.node)})

    collector.set_pytest_marks(get_pytest_mark_names(request.node))

    # Set TestName from nodeid (same derivation as orchestrator)
    collector.set_test_name(derive_pytest_test_id())

    # Set TestTier from pytest tier marks (tier0, optimal, generality, broad)
    tier = get_test_tier(request.node)
    if tier is not None:
        collector.add_dimension({"TestTier": tier.value})

    labeled_kernel = resolve_file_kernel_name(request.path)
    if labeled_kernel:
        collector.add_dimension({"KernelName": labeled_kernel})

    return collector


def make_emitter(
    metric_output_mode: OutputMode | None,
) -> IMetricsEmitter:
    """Create a basic metrics emitter. Returns NoopMetricsEmitter when metrics are disabled."""
    if metric_output_mode is None:
        return NoopMetricsEmitter()
    return MetricsEmitter(output_mode=metric_output_mode)


def resolve_testrun_uid(config: Config) -> str:
    """Session id identical across the controller and all xdist workers.

    Workers return the uid xdist injects via ``workerinput``. The controller
    seeds xdist's ``--testrunuid`` option before NodeManager starts (our
    sessionstart hook runs before xdist's trylast one), so xdist adopts our
    uid and propagates that same value to every worker. A random uid is only
    ever generated for that controller seed or in explicit non-xdist mode;
    any other xdist state raises rather than risk a controller/worker
    mismatch.
    """
    workerinput = getattr(config, "workerinput", None)
    if workerinput is not None:
        uid = workerinput.get("testrunuid")
        if uid is None:
            raise RuntimeError("xdist worker is missing 'testrunuid' in workerinput")
        return uid
    dist = getattr(config.option, "dist", "no")
    if dist != "no":
        # xdist controller. If NodeManager exists its uid is authoritative;
        # otherwise seed the --testrunuid option NodeManager will adopt.
        dsession = config.pluginmanager.getplugin("dsession")
        nodemanager = getattr(dsession, "nodemanager", None) if dsession else None
        if nodemanager is not None:
            return nodemanager.testrunuid
        uid = getattr(config.option, "testrunuid", None)
        if uid is None:
            import uuid

            uid = uuid.uuid4().hex
            config.option.testrunuid = uid
        return uid
    # Non-xdist: stable per-session id cached on config.
    if config.stash.get(TESTRUN_UID_KEY, None) is None:
        import uuid

        config.stash[TESTRUN_UID_KEY] = uuid.uuid4().hex
    return config.stash[TESTRUN_UID_KEY]


def make_host_manager(
    config: Config,
    *,
    target_hosts: list[TargetHost] | None = None,
    s3_config: S3ArtifactUploadConfig | None = None,
) -> HostManager:
    """Create a HostManager from shared CLI options.

    Local/remote and recoverability are derived from the run's host-provisioning mode
    (see resolve_host_provisioning_mode), so callers don't thread those booleans in.

    Args:
        target_hosts: Override host list (default: built from --target-host CLI). A
            plug-in-provisioned or static-file caller passes its own (possibly filtered)
            list; membership for those still flows through the host-state store separately.
        s3_config: Override S3 config (default: built from --artifact-upload-s3-* CLI).
    """
    neuron_installation_path: str = get_feature_flag(config, "neuron_tools_bin_path")
    ssh_config_path: str = resolve_ssh_config_path(config)

    if s3_config is None:
        s3_config = S3ArtifactUploadConfig(
            bucket=get_feature_flag(config, "artifact_upload_s3_bucket"),
            prefix=get_feature_flag(config, "artifact_upload_s3_prefix"),
            profile=get_feature_flag(config, "aws_profile"),
            region=get_feature_flag(config, "artifact_upload_s3_region"),
        )

    if target_hosts is None:
        target_hosts_cli: list[str] = get_feature_flag(config, "target_host", [])
        if target_hosts_cli:
            platforms = get_platform_targets(config)
            assert len(platforms) == 1, f"--target-host requires a single --platform-target, got {platforms}"
            target_hosts = [TargetHost(ssh_host=ip, host_type=platforms[0]) for ip in target_hosts_cli]
        else:
            target_hosts = []

    state_store = HostStateStore(resolve_base_output_directory(config))
    if target_hosts:
        bootstrap = [ResolvedHost(ssh_host=th.ssh_host, host_type=th.host_type) for th in target_hosts]
        state_store.initialize(bootstrap)

    # A local host is needed only for a local run that actually executes on the Neuron device.
    runs_on_hardware = resolve_session_trace_mode(config) in (
        TraceMode.CompileAndInfer,
        TraceMode.CompileAndInferAndSeparate,
        TraceMode.Debugger,
    )

    hm = HostManager(
        state_store=state_store,
        neuron_installation_path=neuron_installation_path,
        ssh_config_path=ssh_config_path,
        testrun_uid=resolve_testrun_uid(config),
        needs_local_host=is_local_run(config) and runs_on_hardware,
        s3_config=s3_config,
        transport=get_feature_flag(config, "transport", "paramiko"),
        host_rotation_patience_seconds=get_feature_flag(config, "ssh_host_rotation_patience_seconds"),
        hosts_recoverable=is_recoverable_run(config),
    )

    # Static --target-host hosts enter the store with unknown (0) capacity, which makes them
    # ineligible for capacity-based routing. Probe their real core counts once here (bounded
    # by the xdist worker cap) and persist them. The plug-in/fleet path instead probes at
    # resolution time, so it needs no probe here. Local runs bypass the store entirely.
    if target_hosts and not is_local_run(config):
        hm.probe_and_record_capacity(
            [th.ssh_host for th in target_hosts],
            max_probe_workers=config.getoption("maxprocesses", default=None),
        )

    return hm


def make_test_manager(
    config: Config,
    trace_mode: TraceMode,
    host_manager: HostManager,
    collector: IMetricsCollector,
    perf_analysis_enabled: bool = False,
    hw_profile_enabled: bool = True,
    kernel_name: str | None = None,
) -> Orchestrator:
    """Create a standard Orchestrator from shared config."""
    return Orchestrator(
        config,
        trace_mode,
        host_manager,
        collector,
        perf_analysis_enabled=perf_analysis_enabled,
        hw_profile_enabled=hw_profile_enabled,
        kernel_name=kernel_name,
        nki_compilation_mode=NKICompilationMode[get_feature_flag(config, "nki_compilation_mode")],
    )


# ─── Shared CLI options ───


def pytest_addoption(parser):
    group = parser.getgroup("nkilib", "NKI Library Testing")

    group.addoption(
        "--target-host",
        default=[],
        nargs="+",
        help="Hostname(s) of MLA accelerator hosts to execute tests on remotely",
    )
    group.addoption(
        "--target-host-file",
        action="store",
        default=None,
        help="Path to JSON file containing MLA accelerator host definitions.  This is used in place of --target-host if the set of hosts are not homogeneous with respect to host type.",
    )
    group.addoption(
        "--output-directory",
        default="neuron_test_output",
        help="Base directory for artifacts produced by test cases",
    )
    group.addoption(
        "--neuron-tools-bin-path",
        default="/opt/aws/neuron/bin",
        help="Path to directory containing neuron tools (neuron-explorer, neuron-ls, etc.) on remote hosts",
    )
    group.addoption(
        "--ssh-config-path",
        help="Path to SSH config file for remote connections (default: ~/.ssh/config)",
    )
    group.addoption(
        "--skip-remote-cleanup",
        action="store_true",
        default=False,
        help="Skip cleanup of remote directories after test execution",
    )
    group.addoption(
        "--debug-kernels",
        action="store_true",
        default=False,
        help="Dump additional debug output inside test directory",
    )
    group.addoption(
        "--metric-output",
        nargs="?",
        const="file",
        default="file",
        choices=["file", "stdout", "stderr"],
        help="Enable metrics collection: 'file' (default), 'stdout', or 'stderr'",
    )
    group.addoption(
        "--test-mode",
        action="store",
        choices=[
            "trace-only",
            "compile-only",
            "compile-and-infer",
            "simulation",
            "debugger",
        ],
        help="Override default trace mode (markers take precedence)",
    )
    group.addoption(
        "--nki-compilation-mode",
        action="store",
        default=NKICompilationMode.tracer.value,
        choices=[m.value for m in NKICompilationMode],
        help="NKI compiler frontend to use",
    )
    group.addoption(
        "--debugger-interactive",
        action="store_true",
        default=False,
        help="Enable interactive mode for nki.debug",
    )
    group.addoption(
        "--debugger-core-id",
        action="store",
        type=int,
        default=0,
        help="NeuronCore ID to debug (default: 0)",
    )
    group.addoption(
        "--debugger-replay",
        action="store_true",
        default=False,
        help="Replay nki.debug() against device dumps from a previous run",
    )
    group.addoption(
        "--platform-target",
        action="store",
        default=None,
        help="Target instance family for test execution. "
        "Single value or comma-separated list (e.g., trn2,trn3_a0). Auto-detected if omitted.",
    )
    group.addoption(
        "--validation-histograms",
        action="store_true",
        default=False,
        help="Dump full report with histograms during validation",
    )
    group.addoption(
        "--enable-perf-analysis",
        action="store_true",
        default=False,
        help="Enable performance analysis with perf sim and detailed profiled JSON",
    )
    group.addoption(
        "--enable-hw-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable HW profile capture (NTFF -> MBU/MFU/cycles). Default True. Use --no-enable-hw-profile to disable.",
    )
    group.addoption(
        "--force-local-cleanup",
        action="store_true",
        default=False,
        help="Automatically cleanup test output directory regardless of test outcome",
    )
    group.addoption(
        "--force-local-cleanup-keep",
        nargs="+",
        choices=["metrics"],
        default=[],
        help="Artifact types to preserve when using --force-local-cleanup",
    )
    group.addoption(
        "--enable-dge-notifs",
        action="store_true",
        default=False,
        help="Enable DGE Notifications during profiling",
    )
    group.addoption(
        "--ucode-lib-path",
        action="store",
        default=None,
        help="Path to a custom uCode library (.so). Uploaded to the remote host "
        "alongside the NEFF; NEURON_RT_UCODE_LIB_PATH is set for the profile run.",
    )
    group.addoption(
        "--artifact-upload-s3-bucket",
        action="store",
        default=None,
        help="S3 bucket for artifact file transfer to remote hosts",
    )
    group.addoption(
        "--artifact-upload-s3-prefix",
        action="store",
        default="artifacts_tmp",
        help="S3 prefix for artifact file transfer (default: artifacts_tmp)",
    )
    group.addoption(
        "--artifact-upload-s3-region",
        action="store",
        default=None,
        help=(
            "Region of the artifact-transfer S3 bucket. Optional; if unset it is "
            "resolved from the bucket. Passed as --region to the remote 'aws s3 cp' "
            "so it signs for the bucket's region rather than the host's (required "
            "for hosts in Local Zones with no S3 endpoint)."
        ),
    )
    group.addoption(
        "--aws-profile",
        action="store",
        default=None,
        help="AWS profile name for S3 authentication",
    )
    group.addoption(
        "-U",
        "--upload-profile-to-explorer",
        nargs="?",
        const=UploadProfileMode.ALWAYS.value,
        default=None,
        choices=[m.value for m in UploadProfileMode],
        help="Upload profiling artifacts to Neuron Explorer. 'always' (default when flag provided), 'on-fail-only' (upload only on test failure)",
    )

    group.addoption(
        "--skip-model-tests",
        action="store_true",
        default=False,
        help="Exclude model tests from collection",
    )
    group.addoption(
        "--neuronx-cc-jobs",
        action="store",
        default="auto",
        metavar="N",
        help="Thread count passed to neuronx-cc via --jobs. "
        "'auto' (default) divides CPU count by the number of xdist workers so "
        "concurrent compilations don't thrash. '0' leaves neuronx-cc to choose.",
    )
    group.addoption(
        "--s3-neuronx-cc-cache-path",
        action="store",
        default="",
        help="S3 URI for BIR-to-NEFF compilation cache (e.g. s3://my-bucket/neff-cache). "
        "When set, reuses cached NEFFs when neuronxcc version and BIR are unchanged.",
    )
    group.addoption(
        "--s3-torch-ref-cache-path",
        action="store",
        default="",
        help="S3 URI for the torch-reference (golden) output cache (e.g. s3://my-bucket/torch-ref-cache). "
        "When set, reuses cached goldens when the reference's transitive source and inputs are unchanged. "
        "Off by default.",
    )
    group.addoption(
        "--transport",
        action="store",
        default=None,
        choices=["ssh", "paramiko"],
        help="Remote-host transport: 'paramiko' (default; persistent fabric2/paramiko connection "
        "per worker) or 'ssh' (native ssh subprocess per command, multiplexed via ControlMaster).",
    )

    group.addoption(
        "--skip-core-reset",
        action="store_true",
        default=False,
        help="Opportunistically skip neuron core reset behavior",
    )

    group.addoption(
        "--ssh-host-rotation-patience-seconds",
        default=None,
        action="store",
        type=int,
        help="Seconds threshold controlling acceptable wait inference queue wait time. Negative value disable patience"
        + " mechanic entirely",
    )

    coverage_group = parser.getgroup("coverage-parametrize")
    coverage_group.addoption(
        "--coverage",
        action="store",
        default="singles",
        choices=["singles", "pairs", "full"],
        help="Default parameter coverage regime for coverage_parametrize tests",
    )
    coverage_group.addoption(
        "--skip-coverage-parametrize",
        action="store_true",
        help="Exclude coverage_parametrize tests from collection",
    )

    chunking_group = parser.getgroup("suite-chunking")
    chunking_group.addoption(
        "--suite-chunk-total-count",
        action="store",
        default=None,
        help="Total number of suite chunks. Used when splitting a single test suite into multiple chunks to be executed concurrently.",
    )
    chunking_group.addoption(
        "--suite-chunk-number",
        action="store",
        default=None,
        help="Desired index of the current suite chunk. Has to be in range of 1 <= chunk_num <= total_chunk_count. To be used with --suite-chunk-total-count",
    )


# ─── Shared fixtures ───


@pytest.fixture(autouse=True)
def _neuron_tag_workflow_item(request: pytest.FixtureRequest):
    """Set NEURON_TAG_WORKFLOW_ITEM to the pytest nodeid for the test's
    duration. The remote-host pass-through in host_management.py forwards
    this to the workload process, where the hardware monitor agent reads
    it from the env at /dev/neuron0 open time.
    """
    os.environ["NEURON_TAG_WORKFLOW_ITEM"] = request.node.nodeid.rsplit("::", 1)[-1]
    try:
        yield
    finally:
        os.environ.pop("NEURON_TAG_WORKFLOW_ITEM", None)


@pytest.fixture(scope="session")
def output_directory(request: pytest.FixtureRequest) -> str:
    output_dir_path = resolve_base_output_directory(request.config)
    Path(output_dir_path).mkdir(exist_ok=True)
    return output_dir_path


@pytest.fixture(scope="session")
def session_trace_mode(request: pytest.FixtureRequest) -> TraceMode:
    """Session-wide trace mode from CLI flags. Individual tests may override via markers."""
    return resolve_session_trace_mode(request.config)


@pytest.fixture
def trace_mode(request: pytest.FixtureRequest, session_trace_mode: TraceMode) -> TraceMode:
    """Per-test trace mode. Markers override the session default."""
    for mode in TraceMode:
        if request.node.get_closest_marker(mode.value) is not None:
            return mode
    return session_trace_mode


@pytest.fixture
def metric_output_mode(request: pytest.FixtureRequest) -> OutputMode | None:
    metric_output: str | None = get_feature_flag(request.config, "metric_output")
    valid_values = {None, OutputMode.FILE.value, OutputMode.STDOUT.value, OutputMode.STDERR.value}
    assert metric_output in valid_values, (
        f"Invalid --metric-output value: '{metric_output}'. "
        f"Valid options: '{OutputMode.FILE.value}', '{OutputMode.STDOUT.value}', '{OutputMode.STDERR.value}'"
    )
    return OutputMode(metric_output) if metric_output else None


@pytest.fixture
def collector(request: pytest.FixtureRequest, metric_output_mode: OutputMode | None) -> IMetricsCollector:
    """Create metrics collector (Noop when metrics are disabled)."""
    return make_collector(request, metric_output_mode)


@pytest.fixture
def emitter(metric_output_mode: OutputMode | None) -> IMetricsEmitter:
    """Create metrics emitter (Noop when metrics are disabled)."""
    return make_emitter(metric_output_mode)


@pytest.fixture(scope="session")
def host_manager(request: pytest.FixtureRequest) -> HostManager:
    """Host manager using --target-host and --artifact-upload-s3-* CLI options. No host-file support."""
    return make_host_manager(request.config)


@pytest.fixture
def perf_analysis_enabled(request: pytest.FixtureRequest) -> bool:
    return get_feature_flag(request.config, "enable_perf_analysis", False)


@pytest.fixture
def hw_profile_enabled(request: pytest.FixtureRequest) -> bool:
    """Whether HW profile capture runs. Gated by --enable-hw-profile."""
    return get_feature_flag(request.config, "enable_hw_profile", True)


@pytest.fixture
def test_manager(
    request: pytest.FixtureRequest,
    trace_mode: TraceMode,
    host_manager: HostManager,
    collector: IMetricsCollector,
    perf_analysis_enabled: bool,
    hw_profile_enabled: bool,
) -> Orchestrator:
    """Standard kernel test orchestrator."""
    kernel_name = resolve_file_kernel_name(request.path)
    return make_test_manager(
        request.config,
        trace_mode,
        host_manager,
        collector,
        perf_analysis_enabled=perf_analysis_enabled,
        hw_profile_enabled=hw_profile_enabled,
        kernel_name=kernel_name,
    )


@pytest.fixture
def platform_target(request: pytest.FixtureRequest) -> Platforms:
    """Injected via indirect parametrization from pytest_generate_tests."""
    return request.param


# ─── Shared hooks ───


def _set_neuron_tag_defaults(config: Config) -> None:
    """Populate known NEURON_TAG_* the orchestrator didn't set.

    Both the pipeline and local --shared-fleet runs go through pytest, so
    setting defaults here is the only place they're guaranteed to land in
    both flows. setdefault means orchestrator-provided values always win.

    Runs only on the xdist controller — workers inherit env from the
    controller via fork, so they don't need to re-resolve the same values.
    """
    if hasattr(config, "workerinput"):
        return  # xdist worker; env was inherited from the controller

    os.environ.setdefault("NEURON_TAG_TEAM", "nkilib")
    os.environ.setdefault("NEURON_TAG_REQUESTER", "User")
    os.environ.setdefault("NEURON_TAG_WORKFLOW_NAME", f"local/{resolve_current_user(default='unknown')}")
    os.environ.setdefault("NEURON_TAG_WORKFLOW_ID", "local")
    os.environ.setdefault("NEURON_TAG_WORKFLOW_VERSION", f"git/{resolve_git_short_sha(default='unknown')}")


def pytest_configure(config: Config):
    """Auto-discover marks, set up simulation mode, register platform markers."""
    _set_neuron_tag_defaults(config)

    # Discover marks from @pytest_test_metadata decorators
    # Use config.rootpath so discovery works whether the plugin is loaded from
    # the source tree or from the installed nkilib_testing wheel.
    test_root = config.rootpath / "test"
    if test_root.is_dir():
        discovered_marks = discover_pytest_test_metadata_marks(test_root)
        for mark_name, description in discovered_marks.items():
            config.addinivalue_line("markers", f"{mark_name}: {description}")

    if is_simulation_mode(config):
        setup_simulation_mode()

    for p in Platforms:
        config.addinivalue_line(
            "markers",
            f"{p.value}: Dynamically applied to tests targeting the {p.value} platform",
        )

    # Apply platform-target as pytest markexpr for correct test collection
    platforms = get_platform_targets(config)
    platform_expr = "(" + " or ".join(p.value for p in platforms) + ")"
    marker_expr: str | None = config.option.markexpr
    if marker_expr:
        config.option.markexpr = f"{marker_expr} and {platform_expr}"
    else:
        config.option.markexpr = platform_expr


def pytest_sessionstart(session):
    """Seed random generators for deterministic test collection across xdist workers."""
    if _RNG_SEED_ENV_KEY in os.environ:
        seed = int(os.environ[_RNG_SEED_ENV_KEY])
        session._original_random_state = random.getstate()
        session._original_numpy_state = np.random.get_state()
        random.seed(seed)
        np.random.seed(seed)


def pytest_collection_finish(session):
    """Restore random generators after collection and validate debugger mode."""
    if is_debugger_mode(session.config) and len(session.items) != 1:
        raise pytest.UsageError(
            f"Debugger mode requires exactly one test. Got {len(session.items)}. Use -k to select a single test."
        )
    if hasattr(session, "_original_random_state"):
        random.setstate(session._original_random_state)
    if hasattr(session, "_original_numpy_state"):
        np.random.set_state(session._original_numpy_state)


@pytest.hookimpl(hookwrapper=True)
def pytest_collection_modifyitems(config: Config, items: list[pytest.Item]):
    """Apply platform marks, skip slow simulation tests, and chunk the eligible suite.

    Implemented as a hookwrapper so the two phases straddle pytest's builtin
    deselect_by_mark (a plain hookimpl that runs during the yield):

      * Pre-yield: add platform marks so deselect_by_mark can honour the
        platform markexpr set in pytest_configure, plus coverage_parametrize
        and simulation-skip handling.
      * Post-yield: `items` now reflects -m/-k deselection, so it is the truly
        eligible set. Chunk it here and fire pytest_deselected for the dropped
        tests so the collected/deselected counts (and --collect-only output,
        which the terminal reporter emits later) stay accurate.
    """

    # Deselect coverage_parametrize tests when --skip-coverage-parametrize is set
    if get_feature_flag(config, "skip_coverage_parametrize", default_value=False):
        items[:] = [item for item in items if not item.get_closest_marker("coverage_parametrize")]

    for item in items:
        platforms_marker = item.get_closest_marker("platforms")
        excluded = set(platforms_marker.kwargs.get("exclude") or []) if platforms_marker else set()

        if isinstance(item, _ParametrizedItem):
            for param_val in item.callspec.params.values():
                if isinstance(param_val, PlatformAware) and param_val.supported_platforms is not None:
                    excluded |= set(Platforms) - param_val.supported_platforms

        supported = set(Platforms) - excluded

        if isinstance(item, _ParametrizedItem) and "platform_target" in item.callspec.params:
            supported &= {item.callspec.params["platform_target"]}

        for p in supported:
            item.add_marker(pytest.mark.__getattr__(p.value))

    if is_simulation_mode(config):
        from .simulation_setup import skip_slow_simulation_tests

        # Skip tests that carry an explicit TraceMode marker override (e.g.
        # trace_only, compile_only, compile_and_infer).  The trace_mode fixture
        # would honour these markers and switch away from Simulator mode,
        # causing the orchestrator to take a non-simulation code-path.
        non_sim_modes = [m for m in TraceMode if m != TraceMode.Simulator]
        for item in items:
            for mode in non_sim_modes:
                if item.get_closest_marker(mode.value):
                    item.add_marker(
                        pytest.mark.skip(reason=f"test requests {mode.value} mode, incompatible with simulation")
                    )
                    break

        skip_marker = pytest.mark.skip(reason="Skipping slow simulation test (see test/simulation.md)")
        skip_slow_simulation_tests(items, skip_marker)

        skip_incompatible = pytest.mark.skip(reason="Known simulator incompatibility (skip_simulation)")
        for item in items:
            if item.get_closest_marker("skip_simulation"):
                item.add_marker(skip_incompatible)

    # Let builtin deselect_by_mark (and -k) run; afterwards `items` is the eligible set.
    yield

    # Break the eligible suite into chunks, if requested.
    current_chunk_number: int | None = get_feature_flag(config, "suite_chunk_number", default_value=None)
    total_chunk_count: int | None = get_feature_flag(config, "suite_chunk_total_count", default_value=None)
    if current_chunk_number is not None and total_chunk_count is not None and items:
        logging.info(
            f"Test suite chunking detected! Current chunk is {current_chunk_number} out of {total_chunk_count}"
        )
        kept = get_chunked_tests(
            current_chunk_number=int(current_chunk_number),
            total_chunk_count=int(total_chunk_count),
            tests=items,
        )
        kept_ids = {id(item) for item in kept}
        dropped = [item for item in items if id(item) not in kept_ids]
        if dropped:
            config.hook.pytest_deselected(items=dropped)
        items[:] = kept


# =========================
# COVERAGE GENERATORS - @pytest.mark.coverage_parametrize
# =========================
"""
Coverage Parametrize Feature
============================

The coverage_parametrize marker provides intelligent test case generation with configurable
coverage strategies. It generates parameter combinations based on coverage requirements
while supporting filtering and validation.

Usage:
    @pytest.mark.coverage_parametrize(
        param1=[value1, value2, ...],
        param2=[value1, value2, ...],
        coverage="singles|pairs|full",  # Optional: overrides CLI default
        filter=filter_function          # Optional: constraint function
    )

Coverage Strategies:
    - "singles": Each parameter value appears at least once (1-way coverage)
    - "pairs": All parameter pairs are covered (2-way coverage using AllPairs)
    - "full": Complete cartesian product of all parameters

Filter Functions:
    - Must accept parameter names as keyword arguments
    - Return True to include the combination, False to exclude
    - Specifying default values for filter arguments helps create smaller covering sets
    - Example: def filter_func(param1, param2=None): return param1 < param2

Limitations:
    - All parameter values must be hashable (strings, numbers, tuples, etc.)
    - Filter functions with default values work better with AllPairs algorithm
    - Large parameter spaces with "full" coverage can generate many test cases

CLI Options:
    --coverage {singles,pairs,full}  Set default coverage strategy
"""


def pytest_generate_tests(metafunc: Metafunc):
    """Parametrize platform_target and coverage_parametrize tests."""
    # Always parametrize platform_target so test IDs are stable regardless of how many platforms are requested
    platforms = get_platform_targets(metafunc.config)
    if "platform_target" in metafunc.fixturenames:
        metafunc.parametrize("platform_target", platforms, indirect=True, ids=[str(p) for p in platforms])

    # Handle coverage_parametrize (independent of platform parametrization)
    coverage_marker = metafunc.definition.get_closest_marker("coverage_parametrize")
    skip_coverage = get_feature_flag(metafunc.config, "skip_coverage_parametrize", default_value=False)
    if not coverage_marker or skip_coverage:
        return

    # When -m selects fast tests, skip coverage generation for non-fast tests.
    # Coverage-parametrized tests never carry the fast mark, so they'll always be
    # deselected by -m fast. Skipping here avoids expensive AllPairs / combinatorial
    # expansion for ~100+ test functions that would be thrown away during deselection.
    mark_expr = metafunc.config.option.markexpr or ""
    if "fast" in mark_expr and "not fast" not in mark_expr:
        if not metafunc.definition.get_closest_marker("fast"):
            return

    if _RNG_SEED_ENV_KEY in os.environ:
        random.seed(int(os.environ[_RNG_SEED_ENV_KEY]))

    params = dict(coverage_marker.kwargs)
    assert params, "No parameters defined for coverage_parametrize"
    coverage_override = params.pop("coverage", None)
    filter_func = params.pop("filter", None)
    enable_automatic_boundary_tests = params.pop("enable_automatic_boundary_tests", True)
    enable_invalid_combination_tests = params.pop("enable_invalid_combination_tests", True)
    n_tests_per_boundary_value = params.pop("n_tests_per_boundary_value", 3)
    max_invalid_tests = params.pop("max_invalid_tests", 30)
    abbrev = params.pop("abbrev", None)

    coverage = coverage_override if coverage_override is not None else get_feature_flag(metafunc.config, "coverage")

    test_cases = generate_parametrized_test_case(
        params=params,
        coverage=coverage,
        filter_func=filter_func,
        enable_automatic_boundary_tests=enable_automatic_boundary_tests,
        enable_invalid_combination_tests=enable_invalid_combination_tests,
        n_tests_per_boundary_value=n_tests_per_boundary_value,
        max_invalid_tests=max_invalid_tests,
    )

    param_names, values_list, ids_list = extract_parametrize_args(
        params, test_cases, abbrev=abbrev, test_func_name=metafunc.function.__name__
    )
    metafunc.parametrize(param_names, values_list, ids=ids_list)
