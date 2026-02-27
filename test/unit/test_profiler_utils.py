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
Unit tests for profiler command generation.
"""

from ..utils.profiler_utils import (
    ProfilerCommands,
    extract_and_filter_output_files,
)


def test_no_warmup_profile_last():
    """No warmup, profile last -> single ntff file, with --profile-nth-exec."""
    cmds = ProfilerCommands(
        num_runs=1,
        profile_all_runs=False,
        profiler_binary_path="neuron-profile",
        kernel_input_args="--arg1 val1",
        metrics_enabled=False,
    )

    assert "neuron-profile capture" in cmds.capture_cmd
    assert "--num-exec=1 --profile-nth-exec=1" in cmds.capture_cmd
    assert cmds.expected_ntff_files == ["profile.ntff"]
    assert cmds.expected_profiler_view_json_files == []
    assert cmds.expected_show_session_json_files == ["show_session_0.json"]
    assert "SHOW-SESSION RUN 0 (profile.ntff)" in cmds.show_cmd
    assert "show-session -s profile.ntff --show-errors -j > show_session_0.json" in cmds.show_cmd


def test_no_warmup_profile_all():
    """No warmup, profile all -> same as profile last (single execution)."""
    cmds = ProfilerCommands(
        num_runs=1,
        profile_all_runs=True,
        profiler_binary_path="neuron-profile",
        kernel_input_args="--arg1 val1",
        metrics_enabled=False,
    )

    assert "--num-exec=1" in cmds.capture_cmd
    assert "--profile-nth-exec" not in cmds.capture_cmd
    assert cmds.expected_ntff_files == ["profile.ntff"]
    assert "SHOW-SESSION RUN 0 (profile.ntff)" in cmds.show_cmd


def test_warmup_profile_last():
    """With warmup, profile last -> only final ntff file."""
    cmds = ProfilerCommands(
        num_runs=3,
        profile_all_runs=False,
        profiler_binary_path="neuron-profile",
        kernel_input_args="--arg1 val1",
        metrics_enabled=False,
    )

    assert "--num-exec=3 --profile-nth-exec=3" in cmds.capture_cmd
    assert cmds.expected_ntff_files == ["profile_exec_3.ntff"]
    assert cmds.expected_show_session_json_files == ["show_session_0.json"]
    assert "SHOW-SESSION RUN 0 (profile_exec_3.ntff)" in cmds.show_cmd
    assert "show-session -s profile_exec_3.ntff --show-errors -j > show_session_0.json" in cmds.show_cmd


def test_warmup_profile_all():
    """With warmup, profile all -> multiple ntff files."""
    cmds = ProfilerCommands(
        num_runs=3,
        profile_all_runs=True,
        profiler_binary_path="neuron-profile",
        kernel_input_args="--arg1 val1",
        metrics_enabled=False,
    )

    assert "--num-exec=3" in cmds.capture_cmd
    assert "--profile-nth-exec" not in cmds.capture_cmd
    assert cmds.expected_ntff_files == ["profile.ntff", "profile_exec_2.ntff", "profile_exec_3.ntff"]
    assert cmds.expected_show_session_json_files == [
        "show_session_0.json",
        "show_session_1.json",
        "show_session_2.json",
    ]
    # Verify headers for each run
    assert "SHOW-SESSION RUN 0 (profile.ntff)" in cmds.show_cmd
    assert "SHOW-SESSION RUN 1 (profile_exec_2.ntff)" in cmds.show_cmd
    assert "SHOW-SESSION RUN 2 (profile_exec_3.ntff)" in cmds.show_cmd
    assert "show-session -s profile.ntff --show-errors -j > show_session_0.json" in cmds.show_cmd
    assert "show-session -s profile_exec_2.ntff --show-errors -j > show_session_1.json" in cmds.show_cmd
    assert "show-session -s profile_exec_3.ntff --show-errors -j > show_session_2.json" in cmds.show_cmd


def test_metrics_enabled_generates_json_commands():
    """With metrics enabled, JSON generation commands should be created."""
    cmds = ProfilerCommands(
        num_runs=3,
        profile_all_runs=True,
        profiler_binary_path="neuron-profile",
        kernel_input_args="--arg1 val1",
        metrics_enabled=True,
    )

    assert cmds.expected_profiler_view_json_files == ["ntff_0.json", "ntff_1.json", "ntff_2.json"]
    assert "view -n file.neff -s profile.ntff --output-format=summary-json > ntff_0.json" in cmds.json_generation_cmd
    assert (
        "view -n file.neff -s profile_exec_2.ntff --output-format=summary-json > ntff_1.json"
        in cmds.json_generation_cmd
    )
    assert (
        "view -n file.neff -s profile_exec_3.ntff --output-format=summary-json > ntff_2.json"
        in cmds.json_generation_cmd
    )


def test_metrics_disabled_no_json_commands():
    """With metrics disabled, no JSON generation commands."""
    cmds = ProfilerCommands(
        num_runs=3,
        profile_all_runs=True,
        profiler_binary_path="neuron-profile",
        kernel_input_args="--arg1 val1",
        metrics_enabled=False,
    )

    assert cmds.expected_profiler_view_json_files == []
    assert cmds.json_generation_cmd == ""


def test_profile_all_runs_single_warmup():
    """Profile all runs with single warmup -> two ntff files."""
    cmds = ProfilerCommands(
        num_runs=2,
        profile_all_runs=True,
        profiler_binary_path="neuron-profile",
        kernel_input_args="--arg1 val1",
        metrics_enabled=False,
    )

    assert "--num-exec=2" in cmds.capture_cmd
    assert "--profile-nth-exec" not in cmds.capture_cmd
    assert cmds.expected_ntff_files == ["profile.ntff", "profile_exec_2.ntff"]


def test_profile_all_runs_many_warmups():
    """Profile all runs with many warmups -> many ntff files."""
    cmds = ProfilerCommands(
        num_runs=6,
        profile_all_runs=True,
        profiler_binary_path="neuron-profile",
        kernel_input_args="--arg1 val1",
        metrics_enabled=False,
    )

    assert "--num-exec=6" in cmds.capture_cmd
    assert "--profile-nth-exec" not in cmds.capture_cmd
    expected = [
        "profile.ntff",
        "profile_exec_2.ntff",
        "profile_exec_3.ntff",
        "profile_exec_4.ntff",
        "profile_exec_5.ntff",
        "profile_exec_6.ntff",
    ]
    assert cmds.expected_ntff_files == expected


def test_profile_all_runs_show_commands():
    """Profile all runs -> show commands generated for each ntff."""
    cmds = ProfilerCommands(
        num_runs=4,
        profile_all_runs=True,
        profiler_binary_path="neuron-profile",
        kernel_input_args="--arg1 val1",
        metrics_enabled=False,
    )

    # Should have 4 total executions (3 warmup + 1 actual)
    assert "--num-exec=4" in cmds.capture_cmd

    # Should have show commands for all 4 ntff files
    assert cmds.show_cmd.count("show-session") == 4
    assert "show-session -s profile.ntff --show-errors -j > show_session_0.json" in cmds.show_cmd
    assert "show-session -s profile_exec_2.ntff --show-errors -j > show_session_1.json" in cmds.show_cmd
    assert "show-session -s profile_exec_3.ntff --show-errors -j > show_session_2.json" in cmds.show_cmd
    assert "show-session -s profile_exec_4.ntff --show-errors -j > show_session_3.json" in cmds.show_cmd
    assert cmds.expected_show_session_json_files == [
        "show_session_0.json",
        "show_session_1.json",
        "show_session_2.json",
        "show_session_3.json",
    ]


def test_profile_all_runs_json_generation():
    """Profile all runs with metrics -> JSON files for each ntff."""
    cmds = ProfilerCommands(
        num_runs=3,
        profile_all_runs=True,
        profiler_binary_path="neuron-profile",
        kernel_input_args="--arg1 val1",
        metrics_enabled=True,
    )

    # Should have 3 ntff files -> 3 json files
    assert len(cmds.expected_ntff_files) == 3
    assert len(cmds.expected_profiler_view_json_files) == 3

    # Each ntff should have corresponding view command
    json_cmds = [cmd for cmd in cmds.json_generation_cmd.split(" && ") if cmd]
    assert len(json_cmds) == 3


def test_profile_all_false_with_warmup_only_profiles_last():
    """Profile all=False with warmup -> only last execution profiled."""
    cmds = ProfilerCommands(
        num_runs=4,
        profile_all_runs=False,
        profiler_binary_path="neuron-profile",
        kernel_input_args="--arg1 val1",
        metrics_enabled=False,
    )

    # Should run 4 times but only profile the last one
    assert "--num-exec=4" in cmds.capture_cmd
    assert "--profile-nth-exec=4" in cmds.capture_cmd
    assert cmds.expected_ntff_files == ["profile_exec_4.ntff"]
    assert cmds.expected_show_session_json_files == ["show_session_0.json"]

    # Only one show command
    assert cmds.show_cmd.count("show-session") == 1


def test_get_output_files_single_output_single_execution():
    """Test extraction with single output, single execution."""
    stdout = 'saved output "out" as "out"'

    # Test download_all=True
    result = extract_and_filter_output_files(stdout, download_all=True)
    assert result == ["out"]

    # Test download_all=False (should be same for single execution)
    result = extract_and_filter_output_files(stdout, download_all=False)
    assert result == ["out"]


def test_get_output_files_single_output_multiple_executions_download_all():
    """Test extraction with single output, multiple executions, download_all=True."""
    stdout = '''
    saved output "out" as "out"
    saved output "out" as "out.2"
    saved output "out" as "out.3"
    '''

    result = extract_and_filter_output_files(stdout, download_all=True)
    assert result == ["out", "out.2", "out.3"]


def test_get_output_files_single_output_multiple_executions_last_only():
    """Test extraction with single output, multiple executions, download_all=False."""
    stdout = '''
    saved output "out" as "out"
    saved output "out" as "out.2"
    saved output "out" as "out.3"
    '''

    result = extract_and_filter_output_files(stdout, download_all=False)
    assert result == ["out.3"]


def test_get_output_files_multiple_outputs_download_all():
    """Test extraction with multiple outputs (e.g., out + k_out), download_all=True."""
    stdout = '''
    saved output "out" as "out"
    saved output "k_out" as "k_out"
    saved output "out" as "out.2"
    saved output "k_out" as "k_out.2"
    saved output "out" as "out.3"
    saved output "k_out" as "k_out.3"
    '''

    result = extract_and_filter_output_files(stdout, download_all=True)
    assert result == ["out", "k_out", "out.2", "k_out.2", "out.3", "k_out.3"]


def test_get_output_files_multiple_outputs_last_only():
    """Test extraction with multiple outputs (e.g., out + k_out), download_all=False.

    This is the critical test case - we need to keep BOTH outputs from the last execution,
    not just the very last file.
    """
    stdout = '''
    saved output "out" as "out"
    saved output "k_out" as "k_out"
    saved output "out" as "out.2"
    saved output "k_out" as "k_out.2"
    saved output "out" as "out.3"
    saved output "k_out" as "k_out.3"
    '''

    result = extract_and_filter_output_files(stdout, download_all=False)
    assert result == ["out.3", "k_out.3"]


def test_get_output_files_empty_stdout():
    """Test extraction with empty stdout."""
    stdout = ""

    result = extract_and_filter_output_files(stdout, download_all=True)
    assert result == []

    result = extract_and_filter_output_files(stdout, download_all=False)
    assert result == []


def test_get_output_files_no_matches():
    """Test extraction with stdout that doesn't match the pattern."""
    stdout = "some random output without saved output lines"

    result = extract_and_filter_output_files(stdout, download_all=True)
    assert result == []

    result = extract_and_filter_output_files(stdout, download_all=False)
    assert result == []


def test_get_output_files_collectives_download_all():
    """Test extraction with collectives output format (output_worker_N/), download_all=True."""
    stdout = '''
    saved output "out" as "output_worker_0/out"
    saved output "out" as "output_worker_1/out"
    saved output "out" as "output_worker_0/out.2"
    saved output "out" as "output_worker_1/out.2"
    '''

    result = extract_and_filter_output_files(stdout, download_all=True)
    assert result == [
        "output_worker_0/out",
        "output_worker_1/out",
        "output_worker_0/out.2",
        "output_worker_1/out.2",
    ]


def test_get_output_files_collectives_last_only():
    """Test extraction with collectives output format (output_worker_N/), download_all=False."""
    stdout = '''
    saved output "out" as "output_worker_0/out"
    saved output "out" as "output_worker_1/out"
    saved output "out" as "output_worker_0/out.2"
    saved output "out" as "output_worker_1/out.2"
    saved output "out" as "output_worker_0/out.3"
    saved output "out" as "output_worker_1/out.3"
    '''

    # With 2 workers, we have 2 outputs per execution, so keep last 2
    result = extract_and_filter_output_files(stdout, download_all=False)
    assert result == ["output_worker_0/out.3", "output_worker_1/out.3"]


def test_timeout_included_in_capture_cmd():
    """Test that timeout is included in capture command."""
    cmds = ProfilerCommands(
        num_runs=1,
        profile_all_runs=False,
        profiler_binary_path="neuron-profile",
        kernel_input_args="--arg1 val1",
        metrics_enabled=False,
    )

    assert "timeout 55 neuron-profile capture" in cmds.capture_cmd


def test_collectives_profile_all_ranks():
    """Test collectives with profile_all_ranks=True profiles all ranks."""
    ranks = 2
    cmds = ProfilerCommands(
        num_runs=2,
        profile_all_runs=True,
        profiler_binary_path="neuron-profile",
        kernel_input_args=f"--multi-input {ranks}rank_inputs.txt",
        metrics_enabled=False,
        collective_ranks=ranks,
        profile_all_ranks=True,
    )

    assert f"--collectives-workers-per-node={ranks}" in cmds.capture_cmd
    assert "--collectives-profile-id=all" in cmds.capture_cmd
    assert f"--multi-input {ranks}rank_inputs.txt" in cmds.capture_cmd
    # All 2 ranks × 2 runs = 4 ntff files
    expected = [
        "profile_rank_0.ntff",
        "profile_rank_0_exec_2.ntff",
        "profile_rank_1.ntff",
        "profile_rank_1_exec_2.ntff",
    ]
    assert cmds.expected_ntff_files == expected


def test_collectives_profile_rank0_only():
    """Test collectives with profile_all_ranks=False profiles only rank 0."""
    ranks = 4
    cmds = ProfilerCommands(
        num_runs=3,
        profile_all_runs=False,
        profiler_binary_path="neuron-profile",
        kernel_input_args=f"--multi-input {ranks}rank_inputs.txt",
        metrics_enabled=False,
        collective_ranks=ranks,
        profile_all_ranks=False,
    )

    assert f"--collectives-workers-per-node={ranks}" in cmds.capture_cmd
    assert "--collectives-profile-id=0" in cmds.capture_cmd
    assert f"--multi-input {ranks}rank_inputs.txt" in cmds.capture_cmd
    # Only rank 0, last run
    assert cmds.expected_ntff_files == ["profile_rank_0_exec_3.ntff"]


def test_collectives_all_runs_rank0_only():
    """Test collectives with profile_all_runs=True but profile_all_ranks=False."""
    ranks = 8
    cmds = ProfilerCommands(
        num_runs=2,
        profile_all_runs=True,
        profiler_binary_path="neuron-profile",
        kernel_input_args=f"--multi-input {ranks}rank_inputs.txt",
        metrics_enabled=False,
        collective_ranks=ranks,
        profile_all_ranks=False,
    )

    assert "--collectives-profile-id=0" in cmds.capture_cmd
    # All runs but only rank 0
    expected = ["profile_rank_0.ntff", "profile_rank_0_exec_2.ntff"]
    assert cmds.expected_ntff_files == expected


def test_collectives_single_run():
    """Test collectives with single run - no _exec_N suffix."""
    ranks = 16
    cmds = ProfilerCommands(
        num_runs=1,
        profile_all_runs=False,
        profiler_binary_path="neuron-profile",
        kernel_input_args=f"--multi-input {ranks}rank_inputs.txt",
        metrics_enabled=False,
        collective_ranks=ranks,
    )

    assert "--collectives-profile-id=0" in cmds.capture_cmd
    # Single run, rank 0 only - no _exec_N suffix
    assert cmds.expected_ntff_files == ["profile_rank_0.ntff"]


def test_setting_env_vars():
    """Test that environment variables are set correctly in env_vars_cmd."""
    cmds0 = ProfilerCommands(
        num_runs=1,
        profile_all_runs=False,
        profiler_binary_path="neuron-profile",
        kernel_input_args="--arg1 val1",
        metrics_enabled=False,
        env_vars={"VAR1": "value1", "VAR2": "value2"},
    )

    cmds1 = ProfilerCommands(
        num_runs=1,
        profile_all_runs=False,
        profiler_binary_path="neuron-profile",
        kernel_input_args="--arg1 val1",
        metrics_enabled=False,
    )

    assert cmds0.env_vars_cmd == "export VAR1=value1 VAR2=value2"
    assert cmds1.env_vars_cmd == ""


def test_perf_analysis_uses_output_file_flag():
    """Perf analysis detailed JSON uses --output-file= to avoid clobbering summary ntff.json."""
    cmds = ProfilerCommands(
        num_runs=1,
        profile_all_runs=False,
        profiler_binary_path="neuron-profile",
        kernel_input_args="--arg1 val1",
        metrics_enabled=True,
        perf_analysis_enabled=True,
    )

    # Summary ntff.json should still be generated via redirect
    assert "> ntff.json" in cmds.json_generation_cmd

    # Detailed command should use --output-file= instead of mv, so ntff.json is not clobbered
    assert "--output-file=ntff_detailed.json" in cmds.detailed_json_generation_cmd
    assert "mv ntff.json" not in cmds.detailed_json_generation_cmd
    assert cmds.expected_detailed_json_files == ["ntff_detailed.json"]
