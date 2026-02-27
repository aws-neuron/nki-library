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

"""Utilities for determinism checking across multiple inference runs."""

import logging
import pathlib

import torch

from .common_dataclasses import KernelArgs
from .exceptions import ValidationException
from .metrics_collector import IMetricsCollector, MetricName
from .output_validator import load_output_tensor_as_bytes
from .tensor_histogram import TensorHistogram


class DeterminismChecker:
    """Checker for determinism across multiple inference runs."""

    def __init__(
        self,
        kernel_under_test: KernelArgs,
        artifact_path: str,
        num_runs: int,
        collector: IMetricsCollector,
        logfile_path: str | None = None,
        rank_id: int | None = None,
    ):
        """
        Initialize determinism checker.

        Args:
            kernel_under_test: Kernel configuration
            artifact_path: Path to directory containing all output files from all runs
            num_runs: Number of runs to compare
            collector: Metrics collector for timing
            logfile_path: Optional path to log file for output
            rank_id: Optional rank ID for collectives (for logging)
        """
        self.kernel_under_test = kernel_under_test
        self.artifact_path = artifact_path
        self.num_runs = num_runs
        self.collector = collector
        self.logfile_path = logfile_path
        self.rank_prefix = f"[rank {rank_id}] " if rank_id is not None else ""

    def check(self) -> None:
        """
        Run determinism check by comparing outputs locally from multiple inference runs.
        Downloads all output files and performs byte-by-byte comparison.

        Validates configuration and times the check.

        Raises:
            ValidationException: If outputs don't match across runs (determinism check failed)
        """

        # Validate configuration before starting
        assert self.num_runs >= 1, f"num_runs must be at least 1, but got num_runs={self.num_runs}"
        # Validate we have enough runs to compare
        if self.num_runs < 2:
            raise ValidationException(
                f"Determinism check requires at least 2 runs, but num_runs={self.num_runs}. "
                "Set num_runs >= 2 when enable_determinism_check=True"
            )

        self._log(f"{self.rank_prefix}Running determinism check: comparing {self.num_runs} runs")

        with self.collector.timer(MetricName.DETERMINISM_CHECK_TIME):
            # Compare outputs sequentially
            reference_outputs = self._load_outputs_for_run(0)
            for run_idx in range(1, self.num_runs):
                run_outputs = self._load_outputs_for_run(run_idx)

                # Compare this run against reference (raises on mismatch)
                self._compare_two_runs(reference_outputs, run_outputs, run_idx)

        self._log(f"{self.rank_prefix}Determinism check PASSED ({self.num_runs} runs)")

    def _log(self, message: str, level: int = logging.INFO) -> None:
        """Log message to both logging and optional log file."""
        logging.log(level, message)
        if self.logfile_path:
            with open(self.logfile_path, "a") as f:
                f.write(message + "\n")

    def _load_outputs_for_run(self, run_idx: int) -> dict[str, torch.Tensor]:
        """Load all output tensors for a specific run."""
        artifact_dir = pathlib.Path(self.artifact_path)

        # Determine which files belong to this run based on neuron-profile naming convention:
        # Run 0: 'out', 'k_out'
        # Run 1: 'out.2', 'k_out.2'
        # Run 2: 'out.3', 'k_out.3'
        if run_idx == 0:
            # For first run, find base files by looking for files with .2 suffix
            # this tells us which output tensor names exist
            base_files = {f.stem for f in artifact_dir.glob("*.2") if f.is_file()}
            return {
                name: torch.from_numpy(load_output_tensor_as_bytes(artifact_dir / name))
                for name in base_files
                if (artifact_dir / name).exists()
            }
        else:
            # For subsequent runs, find files with .N suffix where N = run_idx + 1
            suffix = f".{run_idx + 1}"
            return {
                f.name[: -len(suffix)]: torch.from_numpy(load_output_tensor_as_bytes(f))
                for f in artifact_dir.glob(f"*{suffix}")
                if f.is_file()
            }

    def _compare_two_runs(
        self,
        reference_outputs: dict[str, torch.Tensor],
        run_outputs: dict[str, torch.Tensor],
        run_idx: int,
    ) -> None:
        """
        Compare two runs for determinism. Raises ValidationException immediately on mismatch.

        Args:
            reference_outputs: Outputs from reference run (run 0)
            run_outputs: Outputs from current run
            run_idx: Index of current run (1-based for user-facing messages)
        """
        run_tensor_names = set(run_outputs.keys())
        ref_tensor_names = set(reference_outputs.keys())

        # Check if tensor names match
        if run_tensor_names != ref_tensor_names:
            missing_tensors = ref_tensor_names - run_tensor_names
            extra_tensors = run_tensor_names - ref_tensor_names

            parts = [f"  Run {run_idx} has different output tensors"]
            if missing_tensors:
                parts.append(f"    Missing: {', '.join(sorted(missing_tensors))}")
            if extra_tensors:
                parts.append(f"    Extra: {', '.join(sorted(extra_tensors))}")

            error_msg = "✗ Determinism check FAILED!\n" + "\n".join(parts)
            raise ValidationException(error_msg)

        # Compare tensor contents byte-by-byte
        for tensor_name in ref_tensor_names:
            if not torch.equal(reference_outputs[tensor_name], run_outputs[tensor_name]):
                ref_tensor = reference_outputs[tensor_name]
                run_tensor = run_outputs[tensor_name]

                if ref_tensor.numel() != run_tensor.numel():
                    error_msg = (
                        f"✗ Determinism check FAILED!\n"
                        f"  Tensor: '{tensor_name}'\n"
                        f"    Size mismatch: reference has {ref_tensor.numel()} bytes, "
                        f"Run {run_idx} has {run_tensor.numel()} bytes"
                    )
                    self._log(error_msg, logging.ERROR)
                    raise ValidationException(error_msg)

                # Find first differing byte using torch
                diff_mask = ref_tensor != run_tensor
                diff_idx = torch.argmax(diff_mask.to(torch.int8)).item()
                num_mismatches = diff_mask.sum().item()

                error_msg = (
                    f"✗ Determinism check FAILED!\n"
                    f"  Tensor: '{tensor_name}'\n"
                    f"    Total mismatches: {num_mismatches} / {ref_tensor.numel()} ({100*num_mismatches/ref_tensor.numel():.2f}%)\n"
                    f"    First mismatch at byte offset {diff_idx}\n"
                    f"    Reference: 0x{ref_tensor[diff_idx]:02x} ({ref_tensor[diff_idx]}), "
                    f"Run {run_idx}: 0x{run_tensor[diff_idx]:02x} ({run_tensor[diff_idx]})"
                )
                self._log(error_msg, logging.ERROR)

                # Print 2D mismatch map
                close = ~diff_mask
                histogram = TensorHistogram()
                self._log("\n2D Mismatch Map (. = match, ! = mismatch):")
                histogram.print_2d_mismatch_map(close)

                raise ValidationException(error_msg)
