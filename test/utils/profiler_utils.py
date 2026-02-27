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
Utility functions for neuron-profile command generation.
"""

import re
from dataclasses import dataclass
from typing import Optional

from .host_management import SshHost

NEURON_RT_ENABLE_DGE_NOTIFICATIONS: str = "NEURON_RT_ENABLE_DGE_NOTIFICATIONS"


def extract_and_filter_output_files(stdout: str, download_all: bool) -> list[str]:
    """
    Extract output files from neuron-profile stdout and filter based on profiling configuration.

    Args:
        stdout: stdout from neuron-profile execution
        download_all: If True, return all outputs. If False, return only last execution.

    Returns:
        List of output filenames to copy from remote host
    """
    # Extract output files from neuron-profile stdout
    output_files = re.findall(r'saved output "[^"]+" as "([^"]+)"', stdout)

    # Filter output files based on download_all flag.
    # When download_all=False, neuron-profile still generates all outputs,
    # but we only want to copy outputs from the last execution for validation.
    # For kernels with multiple outputs (e.g., out + k_out), we need to keep
    # ALL outputs from the last execution, not just the very last file.
    if not download_all and len(output_files) > 0:
        # Determine number of outputs per execution by counting unique base names
        # Example: ['out', 'k_out', 'out.2', 'k_out.2'] has 2 unique outputs
        base_names = set()
        for filename in output_files:
            # Strip execution number suffix (e.g., "out.2" -> "out")
            base_name = re.sub(r'\.\d+$', '', filename)
            base_names.add(base_name)

        num_outputs_per_execution = len(base_names)

        # Keep only the last N files, where N = number of kernel outputs
        # Example: with 2 outputs and 3 executions, keep last 2 files
        # ['out', 'k_out', 'out.2', 'k_out.2', 'out.3', 'k_out.3'] -> ['out.3', 'k_out.3']
        if num_outputs_per_execution > 0:
            output_files = output_files[-num_outputs_per_execution:]

    return output_files


@dataclass
class ProfilerCommands:
    """Container for all neuron-profile commands and expected files."""

    capture_cmd: str
    show_cmd: str
    json_generation_cmd: str
    env_vars_cmd: str
    expected_ntff_files: list[str]
    expected_profiler_view_json_files: list[str]
    expected_show_session_json_files: list[str]

    def __init__(
        self,
        num_runs: int,
        profile_all_runs: bool,
        profiler_binary_path: str,
        kernel_input_args: str,
        metrics_enabled: bool,
        collective_ranks: int = 1,
        profile_all_ranks: bool = False,
        env_vars: Optional[dict[str, str]] = None,
        perf_analysis_enabled: bool = False,
    ):
        """
        Build all neuron-profile commands for capture, show-session, and JSON generation.

        Args:
            num_runs: Total number of kernel executions
            profile_all_runs: Whether to profile all executions or just the last (controls --profile-nth-exec)
            profiler_binary_path: Path to neuron-profile binary
            kernel_input_args: Formatted kernel input arguments for profiler
                For single input: "input inp-input-000.bin weights inp-weights-000.bin"
                For per-rank inputs: "--multi-input 4rank_inputs.txt"
            metrics_enabled: Whether metrics collection is enabled
            collective_ranks: Number of collective ranks (each rank is a logical NeuronCore). Default 1 = no collectives
            profile_all_ranks: Whether to profile all ranks or just rank 0 (controls --collectives-profile-id)
            env_vars: Optional environment variables to set
        """
        self.collective_ranks = collective_ranks
        self.expected_ntff_files = self._generate_expected_ntff_files(num_runs, profile_all_runs, profile_all_ranks)
        profiler_exec_args = self._generate_profiler_exec_args(num_runs, profile_all_runs)
        collective_args = self._generate_collective_args(collective_ranks, profile_all_ranks)

        # Timeout should be less than lock timeout to avoid hanging past lock expiration
        timeout_seconds = SshHost.DEFAULT_CORE_LOCK_TIMEOUT_MINUTES * 60 - 5

        # Build capture command
        capture_cmd_parts = [
            "TIMEFORMAT='NEURON_PROFILE_CAPTURE_TIME: %R'; time",
            f"timeout {timeout_seconds}",
            profiler_binary_path,
            "capture",
            "--save-output",
            "--neff file.neff",
            profiler_exec_args,
            collective_args,
            kernel_input_args,
        ]
        self.capture_cmd = f"({' '.join(capture_cmd_parts)})"

        # Build show-session commands for each ntff file
        # Use -j flag to output JSON to stdout, redirect to file, stderr (logs) goes to console
        # Add header to log output to identify which run is being processed
        self.expected_show_session_json_files = []
        show_cmd_parts = []
        for i, ntff_file in enumerate(self.expected_ntff_files):
            json_file = f"show_session_{i}.json"
            self.expected_show_session_json_files.append(json_file)
            header = f"echo '===== SHOW-SESSION RUN {i} ({ntff_file}) ====='"
            show_parts = [
                f"TIMEFORMAT='NEURON_PROFILE_SHOW_TIME_RUN_{i}: %R'; time",
                profiler_binary_path,
                "show-session",
                f"-s {ntff_file}",
                "--show-errors",
                "-j",
                f"> {json_file}",
            ]
            show_cmd_parts.append(f"({header} && {' '.join(show_parts)})")
        self.show_cmd = " && ".join(show_cmd_parts)

        # Build JSON generation commands if metrics enabled (neuron-profile view).
        # Uses summary-json format which only outputs summary metrics
        self.expected_profiler_view_json_files = []
        json_cmd_parts = []
        if metrics_enabled:
            for i, ntff_file in enumerate(self.expected_ntff_files):
                # For compatibility with metrics parsing:
                # - Single ntff file -> ntff.json
                # - Multiple ntff files -> ntff_0.json, ntff_1.json, etc.
                if len(self.expected_ntff_files) == 1:
                    json_file = "ntff.json"
                else:
                    json_file = f"ntff_{i}.json"
                self.expected_profiler_view_json_files.append(json_file)
                json_parts = [
                    f"TIMEFORMAT='PROFILE_JSON_GENERATION_TIME_RUN_{i}: %R'; time",
                    profiler_binary_path,
                    "view",
                    "-n file.neff",
                    f"-s {ntff_file}",
                    "--output-format=summary-json",
                    f"> {json_file}",
                ]
                json_cmd_parts.append(f"({' '.join(json_parts)})")
        self.json_generation_cmd = " && ".join(json_cmd_parts)

        # Build detailed JSON generation commands for perf analysis (neuron-profile view --output-format=json)
        # Use --output-file= to write directly to the target filename, avoiding clobbering
        # the summary ntff.json produced by the metrics step.
        self.expected_detailed_json_files = []
        detailed_json_cmd_parts = []
        if perf_analysis_enabled:
            for i, ntff_file in enumerate(self.expected_ntff_files):
                if len(self.expected_ntff_files) == 1:
                    json_file = "ntff_detailed.json"
                else:
                    json_file = f"ntff_detailed_{i}.json"
                self.expected_detailed_json_files.append(json_file)
                json_parts = [
                    f"TIMEFORMAT='PROFILE_DETAILED_JSON_GENERATION_TIME_RUN_{i}: %R'; time",
                    profiler_binary_path,
                    "view",
                    "-n file.neff",
                    f"-s {ntff_file}",
                    "--output-format=json",
                    f"--output-file={json_file}",
                ]
                detailed_json_cmd_parts.append(f"({' '.join(json_parts)})")
        self.detailed_json_generation_cmd = " && ".join(detailed_json_cmd_parts)

        # Build environment variables command if env_vars is not None
        if env_vars:
            env_vars_str = " ".join([f"{key}={value}" for key, value in env_vars.items()])
            self.env_vars_cmd = f"export {env_vars_str}"
        else:
            self.env_vars_cmd = ""

    def get_complete_command(self) -> str:
        """
        Build complete profiler command by joining non-empty command parts.
        Filters out empty commands (e.g., json_generation_cmd when metrics disabled).

        Returns:
            Complete command string ready for shell execution
        """
        return " && ".join(
            filter(
                None,
                [
                    self.env_vars_cmd,
                    self.capture_cmd,
                    self.show_cmd,
                    self.json_generation_cmd,
                    self.detailed_json_generation_cmd,
                ],
            )
        )

    def _generate_expected_ntff_files(
        self, num_runs: int, profile_all_runs: bool, profile_all_ranks: bool
    ) -> list[str]:
        """
        Generate list of expected ntff file names based on configuration.

        neuron-profile naming behavior:
        - --num-exec=1: creates profile.ntff
        - --num-exec=N (N>1): creates profile.ntff for exec 1, profile_exec_2.ntff, ..., profile_exec_N.ntff
        - --num-exec=N --profile-nth-exec=M: creates profile_exec_M.ntff (or profile.ntff if N=1)
        - For collectives with --collectives-profile-id=all: creates profile_rank_N.ntff for each rank
        - For collectives with --collectives-profile-id=0: creates profile_rank_0.ntff (rank 0 only)

        Args:
            num_runs: Total number of kernel executions
            profile_all_runs: Whether to profile all executions or just the last (controls --profile-nth-exec)
            profile_all_ranks: Whether to profile all ranks or just rank 0 (controls --collectives-profile-id)

        Returns:
            List of expected ntff file names
        """
        # Determine base names: for collectives, one per rank or just rank 0; otherwise just "profile"
        if self.collective_ranks > 1:
            if profile_all_ranks:
                bases = [f"profile_rank_{rank}" for rank in range(self.collective_ranks)]
            else:
                bases = ["profile_rank_0"]
        else:
            bases = ["profile"]

        ntff_files = []
        for base in bases:
            if profile_all_runs:
                ntff_files.append(f"{base}.ntff")
                for exec_num in range(2, num_runs + 1):
                    ntff_files.append(f"{base}_exec_{exec_num}.ntff")
            else:
                if num_runs == 1:
                    ntff_files.append(f"{base}.ntff")
                else:
                    ntff_files.append(f"{base}_exec_{num_runs}.ntff")

        return ntff_files

    def _generate_profiler_exec_args(self, num_runs: int, profile_all_runs: bool) -> str:
        """
        Generate neuron-profile execution arguments.

        Args:
            num_runs: Total number of kernel executions
            profile_all_runs: Whether to profile all executions or just the last

        Returns:
            String of neuron-profile arguments (e.g., "--num-exec=3 --profile-nth-exec=3")
        """

        if profile_all_runs:
            # Profile all executions - no --profile-nth-exec needed
            return f"--num-exec={num_runs}"
        else:
            # Profile only the last execution
            return f"--num-exec={num_runs} --profile-nth-exec={num_runs} --enable-dge-notifs"

    def _generate_collective_args(self, collective_ranks: int, profile_all_ranks: bool) -> str:
        """
        Generate neuron-profile collective arguments.

        Args:
            collective_ranks: Number of collective ranks
            profile_all_ranks: Whether to profile all ranks or just rank 0 (controls --collectives-profile-id)

        Returns:
            String of neuron-profile arguments for collectives
        """
        if collective_ranks <= 1:
            return ""

        profile_id = "all" if profile_all_ranks else "0"
        # --collectives-workers-per-node: number of workers on the current node
        # --collectives-profile-id: profile all ranks or just rank 0
        return f"--collectives-workers-per-node={collective_ranks} --collectives-profile-id={profile_id}"
