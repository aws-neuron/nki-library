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
import argparse
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np

# Set consistent hash seed for xdist workers to ensure identical test collection
if "PYTEST_XDIST_WORKER" in os.environ and "NEURON_PYTHONHASHSEED" not in os.environ:
    os.environ["NEURON_PYTHONHASHSEED"] = "1234"

import uuid

import pytest
import pytest_html.extras

from .utils.artifact_manager import (
    UploadOutcome,
    should_upload_artifacts,
    upload_test_artifacts_to_s3,
    validate_s3_credentials,
)
from .utils.common_dataclasses import Platforms, TargetHost, TraceMode
from .utils.composite_emitter import CompositeEmitter
from .utils.coverage_parametrized_tests import (
    extract_parametrize_args,
    generate_parametrized_test_case,
)
from .utils.feature_flag_helper import (
    construct_test_output_directory_name,
    get_feature_flag,
    resolve_base_output_directory,
)
from .utils.host_management import HostManager
from .utils.metrics_collector import IMetricsCollector, MetricsCollector, NoopMetricsCollector
from .utils.metrics_emitter import IMetricsEmitter, MetricsEmitter, NoopMetricsEmitter, OutputMode
from .utils.pytest_test_metadata import discover_pytest_test_metadata_marks
from .utils.qor_collector import collect_qor_from_test_dir
from .utils.ranged_test_harness import (
    RANGE_TEST_CONFIG_ATTR_KEY,
    RANGE_TEST_FIXTURE_NAME,
    RANGE_TEST_LIMIT_NUM_TESTS_ATTR_KEY,
    RANGE_TEST_RNG_SEED_ENV_KEY,
    RangeTestConfig,
    RangeTestHarness,
)
from .utils.s3_utils import S3ArtifactUploadConfig, prefetch_and_cache_credentials
from .utils.simulation_constants import SIMULATION_MODE_ENV_VAR
from .utils.sqs_emitter import SQSEmitter
from .utils.test_orchestrator import Orchestrator


# Simulation mode setup - must run before test collection imports nki
def _init_simulation_mode() -> bool:
    """Check if simulation mode requested and initialize if so. Returns True if active."""
    # Check env var first (set by xdist master for workers, since workers don't inherit argv).
    # If not "1", fall back to parsing --test-mode from CLI args.
    is_simulation_enabled = os.environ.get(SIMULATION_MODE_ENV_VAR) == "1"
    if not is_simulation_enabled:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--test-mode", default=None)
        args, _ = parser.parse_known_args()
        is_simulation_enabled = args.test_mode == "simulation"

    if not is_simulation_enabled:
        return False

    try:
        from .utils.simulation_setup import setup_simulation_mode

        setup_simulation_mode()
        return True
    except ImportError as e:
        raise ImportError(
            f"Simulation mode requires NkiCpuSimulator package which is not installed: {e}\n"
            "Use --test-mode=compile-only or --test-mode=compile-and-infer instead."
        ) from e


_SIMULATION_MODE_ACTIVE = _init_simulation_mode()


def pytest_addoption(parser):
    parser.addoption(
        "--target-host",
        default=[],
        nargs="+",
        help="Hostname(s) of MLA accelerator hosts to execute tests on remotely",
    )
    parser.addoption(
        "--target-host-file",
        action="store",
        default=None,
        help="Path to JSON file containing MLA accelerator host definitions.  This is used in place of --target-host if the set of hosts are not homogeneous with respect to host type.",
    )
    parser.addoption(
        "--output-directory",
        default="neuron_test_output",
        help="Base directory for artifacts produced by test cases",
    )
    parser.addoption(
        "--neuron-tools-bin-path",
        default="/opt/aws/neuron/bin",
        help="Path to directory containing all neuron tools like neuron-profile, neuron-ls etc. on the remote hosts",
    )
    parser.addoption(
        "--ssh-config-path",
        help="Path to SSH config file to use for remote connections (default: ~/.ssh/config)",
    )
    parser.addoption(
        "--skip-remote-cleanup",
        action="store_true",
        default=False,
        help="Skip cleanup of remote directories after test execution (useful for debugging)",
    )
    parser.addoption(
        "--force-local-cleanup",
        action="store_true",
        default=False,
        help="Automatically cleanup test output directory regardless of test outcome (useful for space-constrained devices)",
    )
    parser.addoption(
        "--debug-kernels",
        action="store_true",
        default=False,
        help="Dump additional debug output inside test directory to aid in kernel debugging",
    )
    parser.addoption(
        "--metric-output",
        nargs="?",
        const="file",
        choices=["file", "stdout", "stderr"],
        help="Enable metrics collection. 'file' (default) writes to JSON files in test artifacts, 'stderr' writes to stderr, 'stdout' writes to stdout",
    )
    parser.addoption(
        "--metrics-namespace",
        action="store",
        default="NeuronCompiler",
        help="CloudWatch namespace for metrics. Default: NeuronCompiler",
    )
    parser.addoption(
        "--test-mode",
        action="store",
        choices=["trace-only", "compile-only", "compile-and-infer", "simulation"],
        help="Override default trace mode (markers take precedence over this flag)",
    )
    parser.addoption(
        "--sqs-queue-url",
        action="store",
        default=None,
        help="SQS Standard queue URL for metrics ingestion (enables OpenSearch storage)",
    )
    parser.addoption(
        "--run-id",
        action="store",
        default=None,
        help="Pipeline run ID (auto-generated if not provided)",
    )

    parser.addini(
        "artifacts_dir",
        type="string",
        help="Path where supporting artifacts are going to be stored e.g. test reports, log files etc.",
    )
    parser.addoption(
        "--artifact-upload-s3-bucket",
        action="store",
        default=None,
        help="S3 bucket for artifact file transfer to remote hosts",
    )
    parser.addoption(
        "--artifact-upload-s3-prefix",
        action="store",
        default="artifacts_tmp",
        help="S3 prefix for artifact file transfer (default: artifacts_tmp)",
    )
    parser.addoption(
        "--aws-profile",
        action="store",
        default=None,
        help="AWS profile name for S3 authentication (used for both artifact upload and test output upload)",
    )
    parser.addoption(
        "--test-output-s3-bucket",
        action="store",
        default="",
        help="S3 bucket name to upload test output artifacts (e.g., my-bucket-name)",
    )

    parser.addoption(
        "--test-output-s3-prefix",
        action="store",
        default="",
        help="S3 prefix/path within bucket for test output artifacts (e.g., test-artifacts/)",
    )

    parser.addoption(
        "--upload-test-outcomes",
        action="store",
        default="",
        choices=[""] + [e.value for e in UploadOutcome],
        help=f"Which test outcomes to upload to S3: {', '.join(repr(e.value) for e in UploadOutcome)}. Requires --test-output-s3-bucket",
    )
    parser.addoption(
        "--platform-target",
        action="store",
        default="trn2",
        help="Target instance family for test execution (e.g., trn2, trn3)",
    )
    parser.addoption(
        "--validation-histograms",
        action="store_true",
        default=False,
        help="Dump full report with histograms during validation",
    )

    group = parser.getgroup("coverage-parametrize")

    group.addoption(
        "--coverage",
        action="store",
        default="singles",
        choices=["singles", "pairs", "full"],
        help="Default parameter coverage regime for unspecified tests",
    )


@pytest.fixture(scope="session")
def output_directory(request: pytest.FixtureRequest) -> str:
    output_dir_path = resolve_base_output_directory(request.config)
    Path(output_dir_path).mkdir(exist_ok=True)
    return output_dir_path


@pytest.fixture(scope="session")
def artifacts_output_directory(request: pytest.FixtureRequest) -> str:
    artifacts_dir_path = str(request.config.getini("artifacts_dir"))
    Path(artifacts_dir_path).mkdir(exist_ok=True, parents=True)
    return artifacts_dir_path


@pytest.fixture(scope="session")
def test_worker_id() -> str:
    return os.environ.get("PYTEST_XDIST_WORKER", default="gw_master")


# Logging from xdist is very tricky, as workers can only write to stderr and ignore most of the logging configuration.
@pytest.fixture(scope="session", autouse=True)
def setup_logging(request: pytest.FixtureRequest, test_worker_id: str, artifacts_output_directory: str):
    log_dir: str = os.path.join(artifacts_output_directory, "log")
    Path(log_dir).mkdir(exist_ok=True)
    log_file: str = os.path.join(log_dir, f"pytest_{test_worker_id}.txt")

    # need to set up logger even for the 'default' thread.
    log_level: str = str(request.config.getini("log_level"))
    log_format = str(request.config.getini("log_format")).replace("__worker_id__", test_worker_id)
    log_date_format: str = str(request.config.getini("log_date_format"))

    # Create file handler to output logs into corresponding worker file
    file_handler = logging.FileHandler(log_file, mode="w", delay=True)
    file_handler.setFormatter(
        logging.Formatter(
            fmt=log_format,
            datefmt=log_date_format,
            style="%",
        )
    )

    # Create stream handler to output logs on console
    # This is a workaround for a known limitation:
    # https://pytest-xdist.readthedocs.io/en/latest/known-limitations.html
    console_handler = logging.StreamHandler(sys.stderr)  # pytest only prints error logs
    console_handler.setFormatter(
        logging.Formatter(
            fmt=log_format,
            datefmt=log_date_format,
            style="%",
        )
    )
    logging.basicConfig(
        handlers=[file_handler, console_handler],
        level=log_level,
        # critical here, as otherwise config doesn't update as the result of this call.
        force=True,
    )
    logging.info(f'Logging format in worker: {log_format=} {log_file=} {log_level=}')


@pytest.fixture(scope="session")
def host_manager(request: pytest.FixtureRequest) -> HostManager:
    target_host_file: str | None = get_feature_flag(request.config, "target_host_file")
    target_hosts_cli: list[str] = get_feature_flag(request.config, "target_host", list())
    platform_target_str: str = get_feature_flag(request.config, "platform_target")
    neuron_installation_path: str = get_feature_flag(request.config, "neuron_tools_bin_path")

    ssh_config_path: str = os.path.expanduser(get_feature_flag(request.config, "ssh_config_path", "~/.ssh/config"))

    # Build S3 config from CLI options
    s3_config = S3ArtifactUploadConfig(
        bucket=get_feature_flag(request.config, "artifact_upload_s3_bucket"),
        prefix=get_feature_flag(request.config, "artifact_upload_s3_prefix"),
        profile=get_feature_flag(request.config, "aws_profile"),
    )

    # Build target hosts list
    platform_target = Platforms(platform_target_str)
    target_hosts: list[TargetHost] = []
    if target_host_file:
        # Load hosts from JSON file (includes host types, defaults to platform_target if missing)
        with open(target_host_file, "r") as f:
            data = json.load(f)
        for i, host in enumerate(data["sharedFleet"]):
            # sshHost takes precedence over publicIp.
            # TODO: Support for publicIp can be removed once we stop supporting static host lists in the pipeline.
            if "sshHost" in host:
                ssh_host = host["sshHost"]
            elif "publicIp" in host:
                ssh_host = host["publicIp"]
            else:
                raise ValueError(
                    f"Host entry {i} in {target_host_file} is missing required 'sshHost' or 'publicIp' field"
                )
            host_type = Platforms(host["hostType"]) if "hostType" in host else platform_target
            target_hosts.append(
                TargetHost(
                    ssh_host=ssh_host,
                    host_type=host_type,
                )
            )
    elif target_hosts_cli:
        # Use CLI hosts with platform_target as host type
        for host_ip in target_hosts_cli:
            target_hosts.append(TargetHost(ssh_host=host_ip, host_type=platform_target))

    host_manager = HostManager(
        base_host_info_path=resolve_base_output_directory(request.config),
        neuron_installation_path=neuron_installation_path,
        target_hosts=target_hosts,
        ssh_config_path=ssh_config_path,
        default_platform_target=platform_target,
        s3_config=s3_config,
    )

    host_manager.initialize_host_stats()

    return host_manager


@pytest.fixture(scope="session")
def default_trace_mode(host_manager: HostManager) -> TraceMode:
    is_local_trn = False  # TODO: implement when host_manager supports
    is_remote_trn = not host_manager.is_local

    if is_local_trn or is_remote_trn:
        return TraceMode.CompileAndInfer

    return TraceMode.Simulator


@pytest.fixture
def trace_mode(request: pytest.FixtureRequest, default_trace_mode: TraceMode) -> TraceMode:
    # Check markers first (highest priority)
    for mode in TraceMode:
        if request.node.get_closest_marker(mode.value) is not None:
            return mode

    # Fall back to CLI flag
    cli_trace_mode = get_feature_flag(request.config, "test_mode")
    if cli_trace_mode:
        return TraceMode.create(cli_trace_mode)

    # Default fallback
    logging.warning(f"Defaulting trace mode for {request.function.__name__} to {default_trace_mode}")
    return default_trace_mode


@pytest.fixture
def metric_output_mode(request: pytest.FixtureRequest) -> OutputMode | None:
    metric_output: str | None = get_feature_flag(request.config, "metric_output")

    # Validate metric_output value
    valid_values = {None, OutputMode.FILE.value, OutputMode.STDOUT.value, OutputMode.STDERR.value}
    assert metric_output in valid_values, (
        f"Invalid --metric-output value: '{metric_output}'. "
        f"Valid options: '{OutputMode.FILE.value}', '{OutputMode.STDOUT.value}', '{OutputMode.STDERR.value}' (no value defaults to '{OutputMode.FILE.value}')."
    )

    return OutputMode(metric_output) if metric_output else None


@pytest.fixture
def collector(request: pytest.FixtureRequest, metric_output_mode: OutputMode | None) -> IMetricsCollector:
    """
    Create metrics collector for in-memory metric storage.

    The collector shared across test and orchestrator.

    Automatically captures pytest parametrized values as kernel params for metrics.
    """
    if metric_output_mode is None:
        return NoopMetricsCollector()
    else:
        namespace = get_feature_flag(request.config, "metrics_namespace", default_value="NeuronCompiler")
        collector = MetricsCollector()
        collector.set_namespace(namespace)

        # Auto-capture pytest parametrized params
        if hasattr(request.node, 'callspec'):
            params = _extract_pytest_params(request.node.callspec.params)
            collector.set_kernel_params(params)

        return collector


def _extract_pytest_params(callspec_params: dict) -> dict:
    """Extract and serialize pytest parametrized values for metrics."""
    params = {}
    for key, value in callspec_params.items():
        # Skip None values
        if value is None:
            continue
        # Convert enums to their name
        if hasattr(value, 'name'):
            params[key] = value.name
        # Handle other non-serializable types
        elif hasattr(value, '__dict__'):
            params[key] = str(value)
        else:
            params[key] = value
    return params


@pytest.fixture
def emitter(
    request: pytest.FixtureRequest, metric_output_mode: OutputMode | None, collector: IMetricsCollector
) -> IMetricsEmitter:
    """
    Create metrics emitter with configuration from CLI flags or environment variables.
    Output modes:
    - Not set: Metrics disabled
    - "file": Enabled with file output to test artifact directory
    - "stdout": Enabled with stdout output for log ingestion

    If --sqs-queue-url is provided, also sends metrics to SQS for OpenSearch ingestion.
    """

    if metric_output_mode is None:
        return NoopMetricsEmitter()

    emitters: list[IMetricsEmitter] = [MetricsEmitter(collector=collector, output_mode=metric_output_mode)]

    # SQS emitter (controls its own enable/disable via queue_url)
    sqs_queue_url = get_feature_flag(request.config, "sqs_queue_url")
    run_id = get_feature_flag(request.config, "run_id") or os.environ.get("KERNEL_PERF_RUN_ID") or str(uuid.uuid4())
    emitters.append(SQSEmitter(collector=collector, queue_url=sqs_queue_url, run_id=run_id))
    return CompositeEmitter(collector=collector, emitters=emitters)


@pytest.fixture
def test_manager(
    request: pytest.FixtureRequest,
    trace_mode: TraceMode,
    host_manager: HostManager,
    emitter: IMetricsEmitter,
) -> Orchestrator:
    return Orchestrator(request.config, trace_mode, host_manager, emitter)


@pytest.fixture
def platform_target(request: pytest.FixtureRequest) -> Platforms:
    """Return the target instance family specified via --target-instance-family CLI option."""
    platform_str = get_feature_flag(request.config, "platform_target")
    return Platforms(platform_str)


def _pytest_configure(config):
    """Auto-discover and register pytest markers from @pytest_test_metadata decorators."""
    test_root = Path(__file__).parent
    discovered_marks = discover_pytest_test_metadata_marks(test_root)

    for mark_name, description in discovered_marks.items():
        config.addinivalue_line("markers", f"{mark_name}: {description}")


def _pytest_generate_tests(metafunc):
    """Generate parameterized tests for range test fixtures.

    This hook integrates the RangeTestHarness with pytest's parametrization system.
    When a test function includes the 'range_test_options' fixture and is decorated
    with @range_test_config, this will automatically generate test cases based on
    the dimension ranges specified in the configuration.
    """
    if RANGE_TEST_FIXTURE_NAME in metafunc.fixturenames:
        assert hasattr(
            metafunc.function, RANGE_TEST_CONFIG_ATTR_KEY
        ), f"Unable to generate ranged tests, '{metafunc.function.__name__}' is missing '{RANGE_TEST_CONFIG_ATTR_KEY}' attribute"

        range_test_config: RangeTestConfig = getattr(metafunc.function, RANGE_TEST_CONFIG_ATTR_KEY, None)
        assert isinstance(range_test_config, RangeTestConfig)

        additional_args = {}

        # limit the number of executed tests
        if hasattr(metafunc.function, RANGE_TEST_LIMIT_NUM_TESTS_ATTR_KEY):
            limit_num_of_tests: int = getattr(metafunc.function, RANGE_TEST_LIMIT_NUM_TESTS_ATTR_KEY)
            additional_args[RANGE_TEST_LIMIT_NUM_TESTS_ATTR_KEY] = limit_num_of_tests

        # seed python's rng across all distributed workers to the same value, so that we end up collecting the same test ids
        # this is specifically used for pytest-xdist
        old_random_state = None
        if os.environ.__contains__(RANGE_TEST_RNG_SEED_ENV_KEY):
            logging.info(f"{RANGE_TEST_RNG_SEED_ENV_KEY} is set to {os.environ.get(RANGE_TEST_RNG_SEED_ENV_KEY)}")
            old_random_state = random.getstate()  # save original seed before setting it to worker-shared seed
            random.seed(int(os.environ[RANGE_TEST_RNG_SEED_ENV_KEY]))

        try:
            harness = RangeTestHarness(range_test_config, **additional_args)
            ids, test_configs = harness.get_unique_test_cases()
        finally:
            if old_random_state is not None:
                # reset seed to what it was previously
                random.setstate(old_random_state)

        metafunc.parametrize(RANGE_TEST_FIXTURE_NAME, test_configs, ids=ids)


# Store S3 config for use in makereport hook (set in fixture before test runs)
_s3_upload_config: dict[str, tuple[S3ArtifactUploadConfig, str]] = {}


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture test outcome and add S3 link to HTML report."""
    outcome = yield
    rep = outcome.get_result()

    # Store the report on the item for access in fixture teardown
    setattr(item, f"rep_{rep.when}", rep)

    # Add S3 link to report during "call" phase (when test outcome is known)
    if rep.when == "call":
        extra = getattr(rep, "extra", [])

        # Check if S3 upload is configured for this test
        if item.nodeid in _s3_upload_config:
            s3_config, upload_test_outcomes = _s3_upload_config[item.nodeid]
            test_outcome = "passed" if rep.passed else "failed"

            if should_upload_artifacts(test_outcome, upload_test_outcomes):
                # Get the test output directory
                test_dir_path = construct_test_output_directory_name()
                output_directory = resolve_base_output_directory(item.config)
                test_dir_full_path = os.path.join(output_directory, test_dir_path)

                s3_result = upload_test_artifacts_to_s3(test_dir_full_path, s3_config, item.name)
                if s3_result.s3_url:
                    extra.append(pytest_html.extras.url(s3_result.s3_url, name="S3 Artifacts"))
                # Store for potential use elsewhere
                item._s3_upload_result = s3_result

        rep.extra = extra


@pytest.fixture(autouse=True)
def run_after_every_test(
    request: pytest.FixtureRequest,
):
    # Store S3 config BEFORE test runs so makereport hook can use it
    upload_test_outcomes = get_feature_flag(request.config, "upload_test_outcomes")
    if upload_test_outcomes:
        s3_config = S3ArtifactUploadConfig(
            bucket=get_feature_flag(request.config, "test_output_s3_bucket"),
            prefix=get_feature_flag(request.config, "test_output_s3_prefix"),
            profile=get_feature_flag(request.config, "aws_profile"),
        )
        _s3_upload_config[request.node.nodeid] = (s3_config, upload_test_outcomes)

    try:
        # immediately yield as there is no setup needed
        yield
    finally:
        # regardless of the test outcome, we want to have a chance to clean up
        # below code is executed right after test has finished running
        test_dir_path = construct_test_output_directory_name()
        output_directory = resolve_base_output_directory(request.config)
        test_dir_full_path = os.path.join(output_directory, test_dir_path)

        # Collect QoR data BEFORE cleanup (only if test dir exists)
        if os.path.isdir(test_dir_full_path):
            # Get session ID from master or worker
            if hasattr(request.config, "_qor_session_id"):
                session_id = request.config._qor_session_id
            elif hasattr(request.config, "workerinput"):
                session_id = request.config.workerinput.get("qor_session_id")
            else:
                session_id = None
            collect_qor_from_test_dir(test_dir_full_path, output_directory, session_id)

        # Cleanup
        force_cleanup: bool = get_feature_flag(request.config, "force_local_cleanup")
        if force_cleanup:
            shutil.rmtree(test_dir_full_path, ignore_errors=True)


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


# =========================
# PYTEST CONFIGURE HOOK
# This runs before xdist spawns workers, so credential pre-fetch happens once
# and workers inherit the credentials via environment variables
# =========================
def pytest_configure(config):
    _pytest_configure(config)  # old flow

    config.addinivalue_line(
        "markers",
        "coverage_parametrize(**params, coverage=None, filter=None): "
        "Intelligent test parametrization with configurable coverage strategies (singles/pairs/full) and constraint filtering",
    )

    config.addinivalue_line(
        "markers",
        "fast: Mark test as a fast-running test using manual test vectors only",
    )

    config.addinivalue_line(
        "markers",
        "slow_simulation: Mark test as slow for CPU simulation (large tensor shapes)",
    )

    # Pre-fetch AWS credentials before xdist workers spawn to avoid Isengard rate limiting
    # This only runs in the main process; workers will inherit the env vars
    # Workers have 'workerinput' attribute set on config, master does not
    if not hasattr(config, "workerinput"):
        # Generate QoR session ID for this test run
        from datetime import datetime, timezone

        config._qor_session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        artifact_bucket = get_feature_flag(config, "artifact_upload_s3_bucket")
        test_output_bucket = get_feature_flag(config, "test_output_s3_bucket")
        sqs_queue_url = get_feature_flag(config, "sqs_queue_url")
        # Prefetch credentials for S3 or SQS operations
        if artifact_bucket or test_output_bucket or sqs_queue_url:
            profile = get_feature_flag(config, "aws_profile")
            prefetch_and_cache_credentials(profile)

    # Validate S3 credentials for test output upload
    # This runs after pre-fetch so it uses the cached credentials
    # Validate S3 credentials before tests run
    upload_outcomes = get_feature_flag(config, "upload_test_outcomes")
    if upload_outcomes:
        s3_config = S3ArtifactUploadConfig(
            bucket=get_feature_flag(config, "test_output_s3_bucket"),
            prefix=get_feature_flag(config, "test_output_s3_prefix"),
            profile=get_feature_flag(config, "aws_profile"),
        )
        validate_s3_credentials(s3_config)


def pytest_collection_modifyitems(config, items):
    """Skip tests with large shapes when running in simulation mode."""
    if _SIMULATION_MODE_ACTIVE:
        from .utils.simulation_setup import skip_slow_simulation_tests

        skip_marker = pytest.mark.skip(reason="Skipping slow simulation test (see test/simulation.md)")
        skip_slow_simulation_tests(items, skip_marker)


def pytest_configure_node(node):
    """Pass QoR session ID from master to workers (xdist hook)."""
    if hasattr(node.config, "_qor_session_id"):
        node.workerinput["qor_session_id"] = node.config._qor_session_id


def pytest_sessionfinish(session, exitstatus):
    """End of test session - emit run_complete and log QoR CSV path (master only)."""
    # Only run on master (not xdist workers)
    if hasattr(session.config, "workerinput"):
        return

    # Emit run_complete and targeted model configs to SQS
    sqs_queue_url = get_feature_flag(session.config, "sqs_queue_url")
    if sqs_queue_url:
        run_id = get_feature_flag(session.config, "run_id") or os.environ.get("KERNEL_PERF_RUN_ID") or str(uuid.uuid4())

        # Get test stats from terminal reporter
        terminalreporter = session.config.pluginmanager.get_plugin("terminalreporter")
        # Count unique test IDs (terminal reporter includes setup/call/teardown as separate reports)
        passed_ids = {report.nodeid for report in terminalreporter.stats.get("passed", [])}
        failed_ids = {report.nodeid for report in terminalreporter.stats.get("failed", [])}
        passed = len(passed_ids)
        failed = len(failed_ids)

        emitter = SQSEmitter(collector=MetricsCollector(), queue_url=sqs_queue_url, run_id=run_id)

        # Only emit run_complete if KERNEL_NAME is set (CI pipeline runs)
        # This ensures only automated pipeline runs emit completion signals
        kernel_name = os.environ.get("KERNEL_NAME")
        if kernel_name:
            emitter.emit_run_complete(kernel_name=kernel_name, tests_passed=passed, tests_total=passed + failed)
    # Log QoR CSV file path
    if hasattr(session.config, "_qor_session_id"):
        output_dir = resolve_base_output_directory(session.config)
        filepath = os.path.join(output_dir, f"qor_data_{session.config._qor_session_id}.csv")
        if os.path.exists(filepath):
            print(f"\nQoR data collected to {filepath}")


# =========================
# PYTEST SESSION HOOKS - Random seeding for deterministic test collection
# =========================
def pytest_sessionstart(session):
    """Seed random generators before collection for deterministic test parametrization.

    This ensures that random.sample() calls in test class bodies (which execute at import time
    during collection) produce consistent results across xdist workers.
    """
    if RANGE_TEST_RNG_SEED_ENV_KEY in os.environ:
        seed = int(os.environ[RANGE_TEST_RNG_SEED_ENV_KEY])
        session._original_random_state = random.getstate()
        session._original_numpy_state = np.random.get_state()
        random.seed(seed)
        np.random.seed(seed)


def pytest_collection_finish(session):
    """Restore random generators after collection to allow true randomness during test execution.

    This ensures that host_management.py core allocation shuffling gets true randomness,
    not deterministic behavior from the collection seed.
    """
    if hasattr(session, '_original_random_state'):
        random.setstate(session._original_random_state)
    if hasattr(session, '_original_numpy_state'):
        np.random.set_state(session._original_numpy_state)


# =========================
# PYTEST HOOK
# =========================
def pytest_generate_tests(metafunc):
    # Check for deprecated range tests first (old flow)
    if RANGE_TEST_FIXTURE_NAME in metafunc.fixturenames:
        return _pytest_generate_tests(metafunc)

    # Handle coverage_parametrize
    marker = metafunc.definition.get_closest_marker("coverage_parametrize")
    if not marker:
        return

    if RANGE_TEST_RNG_SEED_ENV_KEY in os.environ:
        random.seed(int(os.environ[RANGE_TEST_RNG_SEED_ENV_KEY]))
    # Parameters defined by the test
    params = marker.kwargs.copy()
    assert params, "No parameters defined for coverage_parametrize"
    coverage_override = params.pop("coverage", None)
    filter_func = params.pop("filter", None)
    enable_automatic_boundary_tests = params.pop("enable_automatic_boundary_tests", True)
    enable_invalid_combination_tests = params.pop("enable_invalid_combination_tests", True)
    n_tests_per_boundary_value = params.pop("n_tests_per_boundary_value", 3)
    max_invalid_tests = params.pop("max_invalid_tests", 30)

    # Coverage resolution: per-test override > CLI default
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

    param_names, values_list, ids_list = extract_parametrize_args(params, test_cases)
    metafunc.parametrize(param_names, values_list, ids=ids_list)
