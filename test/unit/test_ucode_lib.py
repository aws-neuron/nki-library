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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from test.utils.common_dataclasses import (
    CompilerArgs,
    InferenceArgs,
    KernelArgs,
    NKICompilationMode,
    Platforms,
    TraceMode,
)
from test.utils.test_orchestrator import Orchestrator


def _make_orchestrator(tmp_path, ucode_lib_path):
    feature_flags = {
        "neuron_tools_bin_path": "",
        "neuronx_cc_jobs": "0",
        "ucode_lib_path": ucode_lib_path,
    }

    def get_feature_flag(config, key, default_value=None):
        return feature_flags.get(key, default_value)

    with (
        patch(
            "test.utils.test_orchestrator.feature_flag_helper.get_feature_flag",
            side_effect=get_feature_flag,
        ),
        patch(
            "test.utils.test_orchestrator.feature_flag_helper.resolve_base_output_directory",
            return_value=str(tmp_path),
        ),
    ):
        return Orchestrator(
            config=MagicMock(),  # ty: ignore — test double for Config
            trace_mode=TraceMode.CompileOnly,
            host_manager=MagicMock(),  # ty: ignore — test double for HostManager
            collector=MagicMock(),
            nki_compilation_mode=NKICompilationMode.tracer,
        )


def test_existing_file_is_accepted(tmp_path):
    ucode_lib = tmp_path / "libcustom_ucode.so"
    ucode_lib.write_bytes(b"ucode")

    orchestrator = _make_orchestrator(tmp_path, str(ucode_lib))

    assert orchestrator.ucode_lib_path == str(ucode_lib)


def test_missing_file_raises(tmp_path):
    missing_ucode_lib = tmp_path / "missing.so"

    with pytest.raises(ValueError, match=r"--ucode-lib-path does not exist:"):
        _make_orchestrator(tmp_path, str(missing_ucode_lib))


def test_stages_lib_into_artifacts_dir(tmp_path):
    ucode_lib = tmp_path / "libcustom_ucode.so"
    ucode_lib.write_bytes(b"ucode")
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    def no_host_assignments():
        yield from ()

    orchestrator = object.__new__(Orchestrator)
    orchestrator.trace_mode = TraceMode.CompileAndInfer
    orchestrator.ucode_lib_path = str(ucode_lib)
    orchestrator.collector = MagicMock()
    orchestrator.fs_config = SimpleNamespace(
        artifacts_output_directory_path=str(artifacts_dir),
        host_manager=SimpleNamespace(
            get_host_assignment_with_retry=lambda **kwargs: no_host_assignments(),
        ),
    )
    kernel_under_test = SimpleNamespace(
        compiler_input=SimpleNamespace(
            platform_target=Platforms.TRN3,
            logical_nc_config=1,
        ),
        inference_args=SimpleNamespace(collective_ranks=1),
    )

    orchestrator._run_inference(kernel_under_test, {})

    assert (artifacts_dir / ucode_lib.name).read_bytes() == b"ucode"


def test_env_var_points_at_uploaded_lib(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    ucode_lib = tmp_path / "libcustom_ucode.so"
    ucode_lib.write_bytes(b"ucode")

    orchestrator = object.__new__(Orchestrator)
    orchestrator.trace_mode = TraceMode.CompileAndInfer
    orchestrator.ucode_lib_path = str(ucode_lib)
    orchestrator.enable_kernel_debugging = False
    orchestrator.enable_dge_notifs = False
    orchestrator.perf_analysis_enabled = False
    orchestrator.hw_profile_enabled = True
    orchestrator.profiler_binary_path = "neuron-explorer"
    orchestrator.explorer_binary_path = "neuron-explorer"
    orchestrator.skip_core_reset = False
    orchestrator.collector = SimpleNamespace(metrics_enabled=False)
    orchestrator.fs_config = SimpleNamespace(artifacts_output_directory_path=str(artifacts_dir))

    kernel_under_test = KernelArgs(
        kernel_func=lambda: None,
        compiler_input=CompilerArgs(
            platform_target=Platforms.TRN3,
            logical_nc_config=1,
        ),
        inference_args=InferenceArgs(),
    )
    execution_host = MagicMock()

    with patch("test.utils.test_orchestrator.ProfilerCommands") as profiler_commands:
        profiler_commands.return_value.get_hardware_command.return_value = "capture"
        profiler_commands.return_value.get_post_lock_command.return_value = ""

        orchestrator.__run_profiler_on_host__(
            execution_host,
            kernel_under_test,
            {},
            MagicMock(),
        )

    env_vars = profiler_commands.call_args.kwargs["env_vars"]
    assert env_vars["NEURON_RT_UCODE_LIB_PATH"] == f"./{ucode_lib.name}"


def test_reuses_per_rank_input_manifest_with_absolute_paths(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "input-0.bin").write_bytes(b"0")
    (source / "input-1.bin").write_bytes(b"1")
    (source / "2rank_inputs.txt").write_text("x input-0.bin\nx input-1.bin\n")

    result = Orchestrator.__reuse_per_rank_inputs__(
        str(output),
        str(source),
        2,
    )

    assert result == {"--multi-input": "2rank_inputs.txt"}
    assert (output / "2rank_inputs.txt").read_text().splitlines() == [
        f"x {source / 'input-0.bin'}",
        f"x {source / 'input-1.bin'}",
    ]
