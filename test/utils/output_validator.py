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
import math
import os
import pathlib
from contextlib import nullcontext
from test.integration.nkilib.utils.comparators import get_largest_abs_diff, maxAllClose
from typing import Any, Union

import numpy as np
import numpy.typing as npt

from .common_dataclasses import (
    CustomValidatorWithOutputTensorData,
    KernelArgs,
    LazyGoldenGenerator,
    PerRankLazyGoldenGenerator,
)
from .metrics_collector import IMetricsCollector, MetricName
from .tensor_histogram import TensorHistogram


def load_output_tensor_as_bytes(filepath: Union[str, pathlib.Path]) -> npt.NDArray[np.uint8]:
    """
    Load an output tensor file as raw bytes.

    Used for byte-level comparison and determinism checking.

    Args:
        filepath: Path to the output tensor file

    Returns:
        Numpy array of uint8 bytes
    """
    return np.fromfile(filepath, dtype=np.uint8)


class OutputValidator:
    def __init__(
        self,
        kernels_args: KernelArgs,
        output_file_list: list[str],
    ):
        self.kernels_args: KernelArgs = kernels_args
        self.output_file_list: list[str] = output_file_list
        self.LOGGER: logging.Logger = logging.getLogger(__name__)

    def validate(self, logfile_path: str | None = None, enable_histograms: bool = False):
        """Validate outputs for all ranks."""
        assert self.kernels_args.emitter
        metrics_collector: IMetricsCollector = self.kernels_args.emitter.get_collector()

        params = self.kernels_args.validation_args

        if params is None or (
            isinstance(params.golden_output, LazyGoldenGenerator) and params.golden_output.golden is None
        ):
            metrics_collector.record_metric(MetricName.IS_VALIDATION_SKIPPED, 1, "None")
            self.LOGGER.warning(
                "Validation is disabled because no validation arguments were passed or golden output is missing"
            )
            return

        collective_ranks = self.kernels_args.inference_args.collective_ranks
        is_per_rank_golden = isinstance(params.golden_output, PerRankLazyGoldenGenerator)

        # For single golden with collectives, only validate rank 0
        ranks_to_validate = range(collective_ranks) if is_per_rank_golden else range(1)

        all_failed_keys: list[str] = []

        # Calculate and record AccuracyHw, write results to logfile
        with open(logfile_path, "a") if logfile_path else nullcontext() as logfile:
            for rank_id in ranks_to_validate:
                golden_output, dtype_source = self._get_golden_for_rank_and_dtype_source(
                    params, rank_id, metrics_collector
                )

                if collective_ranks > 1:
                    self.LOGGER.info(f"Validating rank {rank_id}")
                    if logfile:
                        print(f"\n{'='*60}\nValidating rank {rank_id}\n{'='*60}", file=logfile)
                    rank_dir = f"output_worker_{rank_id}/"
                    rank_files = [f for f in self.output_file_list if rank_dir in f]
                    assert rank_files, f"No output files found for rank {rank_id}"
                else:
                    rank_files = self.output_file_list

                with metrics_collector.timer(MetricName.VALIDATION_OUTPUT_LOAD_TIME):
                    actual_outputs = self.__read_all_output_tensors__(rank_files, golden_output, dtype_source)
                with metrics_collector.timer(MetricName.VALIDATION_COMPARE_TIME):
                    failed_keys = self._compare_outputs(
                        actual_outputs,
                        golden_output,
                        params,
                        metrics_collector,
                        logfile,
                        enable_histograms,
                        rank_id=rank_id if collective_ranks > 1 else None,
                        logfile_path=logfile_path,
                    )
                if collective_ranks > 1:
                    all_failed_keys.extend([f"rank{rank_id}:{k}" for k in failed_keys])
                else:
                    all_failed_keys.extend(failed_keys)

        assert len(all_failed_keys) == 0, f"Validation failed for {all_failed_keys}"
        self.LOGGER.info(f"Validation comparison results written to: {logfile_path}")

    def _get_golden_for_rank_and_dtype_source(
        self,
        params,
        rank_id: int,
        metrics_collector: IMetricsCollector,
    ) -> tuple[
        dict[str, CustomValidatorWithOutputTensorData | npt.NDArray[Any]],
        dict[str, npt.NDArray[Any]] | None,
    ]:
        """Get golden output and dtype source for a specific rank.

        Returns:
            Tuple of (golden_output, dtype_source). dtype_source is used to determine
            dtype when loading output tensors, and may be None if not available.
        """
        if isinstance(params.golden_output, PerRankLazyGoldenGenerator):
            return params.golden_output.for_rank(rank_id), None
        elif isinstance(params.golden_output, LazyGoldenGenerator):
            assert params.golden_output.golden
            with metrics_collector.timer(MetricName.GOLDEN_COMPUTATION_TIME):
                return params.golden_output.golden, params.golden_output.output_ndarray
        elif isinstance(params.golden_output, dict):
            return params.golden_output, None
        else:
            assert False, f"Unknown golden generator/validator found"

    def _compare_outputs(
        self,
        actual_outputs: dict[str, npt.NDArray[Any]],
        golden_output: dict[str, CustomValidatorWithOutputTensorData | npt.NDArray[Any]],
        params,
        metrics_collector: IMetricsCollector,
        logfile,
        enable_histograms: bool,
        rank_id: int | None = None,
        logfile_path: str | None = None,
    ) -> list[str]:
        """Compare actual outputs against golden, return list of failed keys."""
        # Initialize visualizer
        visualizer = TensorHistogram()
        failed_keys = []
        rank_prefix = f"rank{rank_id}:" if rank_id is not None else ""

        for output_key, expected_value in golden_output.items():
            assert (
                output_key in actual_outputs
            ), f"{rank_prefix}{output_key} was not emitted by neuron-profile capture. Double check the names of golden outputs and variable names of what {self.kernels_args.kernel_func.__name__} returns "

            if isinstance(expected_value, CustomValidatorWithOutputTensorData):
                expected_value_validator = expected_value.validator(logfile)
                comparison_passed = expected_value_validator.validate(actual_outputs[output_key])
                if not comparison_passed:
                    failed_keys.append(output_key)
            else:
                # Convert both arrays to float32 to avoid NumPy dtype promotion issues
                actual_output = actual_outputs[output_key].astype(np.float32)
                expected_output = expected_value.astype(np.float32)

                largest_abs_diff = get_largest_abs_diff(actual_output, expected_output, atol=params.absolute_accuracy)

                # Record -1 if accuracy metric is invalid
                if math.isfinite(largest_abs_diff):
                    metrics_collector.record_metric(MetricName.ACCURACY_HW, largest_abs_diff, "None")
                else:
                    metrics_collector.record_metric(MetricName.ACCURACY_HW, -1.0, "None")

                # Perform comparison
                self.LOGGER.info(f"Results for {rank_prefix}{output_key}:")
                if logfile:
                    print(f"Results for {rank_prefix}{output_key}:", file=logfile)
                comparison_passed = maxAllClose(
                    actual_output,
                    expected_output,
                    params.relative_accuracy,
                    params.absolute_accuracy,
                    verbose=1,
                    logfile=logfile,
                )

                # Print visualization report (always if enabled, regardless of pass/fail)
                visualizer.print_full_comparison_report(
                    actual=actual_output,
                    expected=expected_output,
                    name=f"{rank_prefix}{output_key}",
                    atol=params.absolute_accuracy,
                    rtol=params.relative_accuracy,
                    passed=comparison_passed,
                    logfile=logfile,
                    enable_histograms=enable_histograms,
                )

                # Save golden output to file for debug if comparison failed
                if not comparison_passed:
                    failed_keys.append(output_key)
                    if logfile_path:
                        golden_dir = os.path.dirname(logfile_path)
                        golden_path = os.path.join(golden_dir, f"golden-{rank_prefix}{output_key}.bin")
                        expected_output.tofile(golden_path)
                        self.LOGGER.info(f"Golden output saved to: {golden_path}")

        return failed_keys

    def __read_all_output_tensors__(
        self,
        output_file_list: list[str],
        golden_output: dict[str, CustomValidatorWithOutputTensorData | npt.NDArray[Any]],
        dtype_source: dict[str, npt.NDArray[Any]] | None = None,
    ):
        result: dict[str, npt.NDArray[Any]] = {}

        for output_file_path in output_file_list:
            # neuron-profiler names files the same as output variable name
            file_name_without_extension = os.path.splitext(os.path.basename(output_file_path))[0]

            if file_name_without_extension not in golden_output:
                continue

            golden_value = golden_output[file_name_without_extension]

            if isinstance(golden_value, CustomValidatorWithOutputTensorData):
                result[file_name_without_extension] = load_output_tensor_as_bytes(output_file_path)
            else:
                # Use dtype from dtype_source if available, otherwise fall back to golden_value
                # Shape always comes from golden_value since output_ndarray may have placeholder shape
                dtype_to_use = (
                    dtype_source.get(file_name_without_extension, golden_value).dtype
                    if dtype_source
                    else golden_value.dtype
                )
                result[file_name_without_extension] = np.fromfile(
                    file=output_file_path,
                    dtype=dtype_to_use,
                ).reshape(golden_value.shape)

        return result
