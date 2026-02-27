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
"""Tensor comparison visualization utilities with histograms and statistics."""

import logging
import re
from typing import Any, Optional, TextIO, Union

import plotext as plt
import torch


class BoxTable:
    """Helper class for creating box-drawing character tables."""

    def __init__(self, col_widths: list[int]):
        """
        Initialize table with specified column widths.

        Args:
            col_widths: List of column widths in characters
        """
        self.col_widths = col_widths
        self.rows: list[str] = []

    def add_header(self, headers: list[str]):
        """Add table header with top border."""
        # Top border
        top = "┌" + "┬".join("─" * (w + 2) for w in self.col_widths) + "┐"
        self.rows.append(top)

        # Header row
        self.add_row(headers)

        # Header separator
        sep = "├" + "┼".join("─" * (w + 2) for w in self.col_widths) + "┤"
        self.rows.append(sep)

    def add_row(self, values: list[str]):
        """Add a data row to the table."""
        cells = []
        for val, width in zip(values, self.col_widths):
            # Truncate or pad to fit column width
            cell = val[:width].ljust(width)
            cells.append(f" {cell} ")
        row = "│" + "│".join(cells) + "│"
        self.rows.append(row)

    def add_separator(self):
        """Add a separator line between rows."""
        sep = "├" + "┼".join("─" * (w + 2) for w in self.col_widths) + "┤"
        self.rows.append(sep)

    def add_footer(self):
        """Add bottom border."""
        bottom = "└" + "┴".join("─" * (w + 2) for w in self.col_widths) + "┘"
        self.rows.append(bottom)

    def render(self) -> str:
        """Render the table as a string."""
        return "\n".join(self.rows)


class TensorHistogram:
    """Visualize tensor comparison results with histograms and statistics."""

    # Y-axis scaling factor for histogram display (adds headroom above max count)
    HISTOGRAM_Y_SCALE_FACTOR = 1.15

    def __init__(self, histogram_width: int = 120, histogram_height: int = 20, num_bins: int = 100):
        """
        Initialize the tensor histogram visualizer.

        Args:
            histogram_width: Maximum width of histogram bars in characters
            histogram_height: Height of histogram in characters
            num_bins: Number of bins for histograms
        """
        self.histogram_width = histogram_width
        self.histogram_height = histogram_height
        self.num_bins = num_bins
        self.logger = logging.getLogger(__name__)

    def _print_and_log(self, text: str, logfile: Optional[TextIO] = None):
        """Log to logger and optionally to a log file."""
        ansi_escape = re.compile(r'\033\[[0-9;]*m')
        clean_text = ansi_escape.sub('', text)
        self.logger.info(clean_text)
        if logfile:
            # Strip ANSI color codes when writing to file
            logfile.write(clean_text + "\n")

    def _to_torch(self, data: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Convert input to torch tensor if it isn't already."""
        if isinstance(data, torch.Tensor):
            return data
        return torch.from_numpy(data)

    def print_histogram(
        self, data: Union[torch.Tensor, Any], title: str, logfile: Optional[TextIO] = None, show_mean: bool = True
    ):
        """
        Print a histogram of the data distribution.

        Args:
            data: Input array to visualize (torch.Tensor or numpy array)
            title: Title for the histogram
            logfile: Optional file handle to write output to
            show_mean: Whether to show a vertical line at the mean value
        """
        data = self._to_torch(data)
        flat_data = data.flatten()
        valid_data = flat_data[torch.isfinite(flat_data)]

        if len(valid_data) == 0:
            self._print_and_log(f"\n{title}", logfile)
            self._print_and_log("  No valid data to display", logfile)
            return

        # If values are to close to max, plotext bug results in a bin above max_bin so we need to clamp values to max
        threshold = 1e-10
        data_max = valid_data.max()
        valid_data = torch.where(valid_data > data_max - threshold, data_max, valid_data)

        plt.clf()

        # Create histogram and get the counts to determine y-axis range
        valid_data_np = valid_data.cpu().numpy()
        counts, _ = torch.histogram(valid_data.float(), bins=self.num_bins)
        max_count = torch.max(counts).item()

        plt.hist(valid_data_np.tolist(), bins=self.num_bins)
        plt.plotsize(self.histogram_width, self.histogram_height)

        # Set y-axis limit with configurable scaling factor
        plt.ylim(0, max_count * self.HISTOGRAM_Y_SCALE_FACTOR)

        # Add vertical line at mean or zero and set title
        if show_mean:
            mean_val = torch.mean(valid_data.float()).item()
            plt.vline(mean_val, "red")
        else:
            # For difference plots, show line at zero
            plt.vline(0, "red")

        plt.title(title)

        plot_str = plt.build()
        self._print_and_log(f"\n{plot_str}", logfile)

    def print_comparison_histogram(
        self,
        actual: Union[torch.Tensor, Any],
        expected: Union[torch.Tensor, Any],
        title: str,
        logfile: Optional[TextIO] = None,
    ):
        """
        Print separate histograms comparing actual and expected distributions.

        Args:
            actual: Actual output array (torch.Tensor or numpy array)
            expected: Expected output array (torch.Tensor or numpy array)
            title: Title for the histogram
            logfile: Optional file handle to write output to
        """
        actual = self._to_torch(actual)
        expected = self._to_torch(expected)

        actual_flat = actual.flatten()
        expected_flat = expected.flatten()
        actual_valid = actual_flat[torch.isfinite(actual_flat)]
        expected_valid = expected_flat[torch.isfinite(expected_flat)]

        if len(actual_valid) == 0 and len(expected_valid) == 0:
            self._print_and_log(f"\n{title}", logfile)
            self._print_and_log("  No valid data to display", logfile)
            return

        self._print_and_log(f"\n{title}", logfile)

        self.print_histogram(actual, "ACTUAL", logfile)
        self.print_histogram(expected, "EXPECTED", logfile)

    def print_comparison_stats(
        self,
        actual: Union[torch.Tensor, Any],
        expected: Union[torch.Tensor, Any],
        atol: float,
        rtol: float,
        logfile: Optional[TextIO] = None,
    ) -> torch.Tensor:
        """
        Print a comprehensive statistics table comparing actual and expected values.
        Uses the same matching logic as maxAllClose in comparators.py.

        Args:
            actual: Actual output array (torch.Tensor or numpy array)
            expected: Expected output array (torch.Tensor or numpy array)
            atol: Absolute tolerance
            rtol: Relative tolerance
            logfile: Optional file handle to write output to

        Returns:
            Boolean tensor indicating which elements match within tolerance
        """
        actual = self._to_torch(actual)
        expected = self._to_torch(expected)

        # Calculate differences with the same logic as maxAllClose
        diff = actual.float() - expected.float()
        abs_diff = torch.abs(diff)

        # Replace matching infinities (same sign) with 0.0 difference
        matching_inf = torch.isinf(actual) & torch.isinf(expected) & (torch.sign(actual) == torch.sign(expected))
        abs_diff = torch.where(matching_inf, torch.tensor(0.0), abs_diff)

        # Filter out NaN/Inf for statistics
        actual_valid = actual[torch.isfinite(actual)]
        expected_valid = expected[torch.isfinite(expected)]
        diff_valid = diff[torch.isfinite(diff)]
        abs_diff_valid = abs_diff[torch.isfinite(abs_diff)]

        # Calculate brange using maxAllClose "max" mode logic: single scalar value
        expected_finite = expected[torch.isfinite(expected)]
        brange = torch.max(torch.abs(expected_finite)).item() if len(expected_finite) > 0 else 1.0

        # Calculate relative difference
        rel_diff_max = (
            (torch.max(abs_diff_valid).item() - atol) / brange if len(abs_diff_valid) > 0 and brange > 0 else 0.0
        )

        # Calculate matching elements using maxAllClose logic with scalar brange
        close = abs_diff <= atol + rtol * brange

        num_matching = torch.sum(close).item()
        num_mismatched = torch.sum(~close).item()
        total_elements = actual.numel()
        match_percentage = (num_matching / total_elements * 100) if total_elements > 0 else 0.0

        # Calculate NaN and Inf counts
        actual_nan_count = torch.sum(torch.isnan(actual)).item()
        expected_nan_count = torch.sum(torch.isnan(expected)).item()
        actual_inf_count = torch.sum(torch.isinf(actual)).item()
        expected_inf_count = torch.sum(torch.isinf(expected)).item()
        matching_nan_count = torch.sum(torch.isnan(actual) & torch.isnan(expected)).item()
        matching_inf_count = matching_inf.sum().item()

        # Create table with BoxTable helper
        table = BoxTable([19, 19, 19, 19])
        table.add_header(["Metric", "Actual", "Expected", "Difference"])

        # Shape and dtype
        table.add_row(["Shape", str(tuple(actual.shape)), str(tuple(expected.shape)), "-"])
        table.add_row(["Dtype", str(actual.dtype), str(expected.dtype), "-"])
        table.add_row(["Total Elements", f"{total_elements:,}", f"{total_elements:,}", "-"])
        table.add_row(
            ["NaN Count", f"{actual_nan_count:,}", f"{expected_nan_count:,}", f"{matching_nan_count:,} matching"]
        )
        table.add_row(
            ["Inf Count", f"{actual_inf_count:,}", f"{expected_inf_count:,}", f"{matching_inf_count:,} matching"]
        )
        table.add_separator()

        # Statistics
        if len(actual_valid) > 0 and len(expected_valid) > 0:
            actual_min = torch.min(actual_valid).item()
            expected_min = torch.min(expected_valid).item()
            actual_max = torch.max(actual_valid).item()
            expected_max = torch.max(expected_valid).item()
            actual_mean = torch.mean(actual_valid.float()).item()
            expected_mean = torch.mean(expected_valid.float()).item()
            actual_std = torch.std(actual_valid.float()).item()
            expected_std = torch.std(expected_valid.float()).item()
            actual_median = torch.median(actual_valid.float()).item()
            expected_median = torch.median(expected_valid.float()).item()

            table.add_row(
                [
                    "Min",
                    f"{actual_min:.6g}",
                    f"{expected_min:.6g}",
                    f"{torch.min(diff_valid).item():.6g}" if len(diff_valid) > 0 else "N/A",
                ]
            )
            table.add_row(
                [
                    "Max",
                    f"{actual_max:.6g}",
                    f"{expected_max:.6g}",
                    f"{torch.max(diff_valid).item():.6g}" if len(diff_valid) > 0 else "N/A",
                ]
            )
            table.add_row(
                [
                    "Mean",
                    f"{actual_mean:.6g}",
                    f"{expected_mean:.6g}",
                    f"{torch.mean(diff_valid).item():.6g}" if len(diff_valid) > 0 else "N/A",
                ]
            )
            table.add_row(
                [
                    "Std Dev",
                    f"{actual_std:.6g}",
                    f"{expected_std:.6g}",
                    f"{torch.std(diff_valid).item():.6g}" if len(diff_valid) > 0 else "N/A",
                ]
            )
            table.add_row(
                [
                    "Median",
                    f"{actual_median:.6g}",
                    f"{expected_median:.6g}",
                    f"{torch.median(diff_valid).item():.6g}" if len(diff_valid) > 0 else "N/A",
                ]
            )
        else:
            table.add_row(["Statistics", "N/A (no valid data)", "N/A (no valid data)", "N/A"])

        table.add_separator()

        # Difference statistics
        if len(abs_diff_valid) > 0:
            table.add_row(["Abs Diff Max", "", "", f"{torch.max(abs_diff_valid).item():.6g}"])
            table.add_row(["Abs Diff Mean", "", "", f"{torch.mean(abs_diff_valid).item():.6g}"])
            table.add_row(["Rel Diff Max", "", "", f"{rel_diff_max * 100:.4g}%"])
        else:
            table.add_row(["Abs Diff Max", "", "", "N/A"])

        table.add_separator()

        # Matching statistics
        table.add_row(["Elements Matching", "", "", f"{match_percentage:.2f}%"])
        table.add_row(["Elements Mismatched", "", "", f"{num_mismatched:,}"])

        table.add_footer()

        # Print table
        self._print_and_log("\n" + table.render(), logfile)

        # Print tolerance info
        tolerance_info = f"Tolerance: atol={atol:.2e}, rtol={rtol:.2e} ({rtol*100:.4g}%)"
        self._print_and_log(tolerance_info, logfile)

        return close

    def print_full_comparison_report(
        self,
        actual: Union[torch.Tensor, Any],
        expected: Union[torch.Tensor, Any],
        name: str,
        atol: float,
        rtol: float,
        passed: bool,
        logfile: Optional[TextIO] = None,
        enable_histograms: bool = False,
    ):
        """
        Print a complete comparison report with histograms and statistics.

        Args:
            actual: Actual output array (torch.Tensor or numpy array)
            expected: Expected output array (torch.Tensor or numpy array)
            name: Name of the output being compared
            atol: Absolute tolerance
            rtol: Relative tolerance
            passed: Whether the comparison passed
            logfile: Optional file handle to write output to
            enable_histograms: Whether to enable histogram visualization
        """
        status = "PASSED" if passed else "FAILED"
        self._print_and_log(f"COMPARISON: {name} - {status}", logfile)
        if not enable_histograms:
            return

        actual = self._to_torch(actual)
        expected = self._to_torch(expected)

        self.print_comparison_histogram(actual, expected, "ACTUAL vs EXPECTED DISTRIBUTION", logfile)

        diff = actual.float() - expected.float()
        self.print_histogram(diff, "DIFFERENCE (ACTUAL - EXPECTED) DISTRIBUTION", logfile, show_mean=False)

        close = self.print_comparison_stats(actual, expected, atol, rtol, logfile)

        # Show 2D mismatch map if comparison failed
        if not passed:
            self.print_2d_mismatch_map(close, logfile)

    def print_2d_mismatch_map(
        self,
        close: torch.Tensor,
        logfile: Optional[TextIO] = None,
        max_width: int = 128,
        max_height: int = 64,
    ):
        """
        Print a 2D map showing matches (.) and mismatches (!) between tensors.

        Args:
            close: Pre-computed boolean tensor of matches
            logfile: Optional file handle to write output to
            max_width: Maximum width of the visualization
            max_height: Maximum height of the visualization
        """
        # Flatten to 2D if needed
        if close.dim() > 2:
            close = close.view(-1, close.shape[-1])
        elif close.dim() == 1:
            close = close.unsqueeze(0)

        h, w = close.shape

        def calc_zone_size(width):
            """Calculate zone size (8 or 4) that evenly divides width."""
            return 8 if width % 8 == 0 else 4

        # Compress representation if tensor is too large
        if h > max_height or w > max_width:
            block_h = max(1, h // max_height)
            block_w = max(1, w // max_width)

            # Only use block sizes if they are powers of 2
            def is_power_of_2(n):
                return n > 0 and (n & (n - 1)) == 0

            if not is_power_of_2(block_h):
                block_h = 1 << (block_h - 1).bit_length()  # Next power of 2
            if not is_power_of_2(block_w):
                block_w = 1 << (block_w - 1).bit_length()  # Next power of 2

            display_h = (h + block_h - 1) // block_h
            display_w = (w + block_w - 1) // block_w

            zone_size = calc_zone_size(display_w)

            # Build header with full position numbers at zone boundaries
            col_header = "     "  # Space for row numbers + border
            j = 0
            while j < display_w:
                if j % zone_size == 0:
                    # Show full position number at zone boundary
                    block_pos = j * block_w
                    pos_str = str(block_pos)
                    col_header += pos_str
                    # Skip ahead by the length of the position string
                    j += len(pos_str)
                else:
                    col_header += " "
                    j += 1
                # Add space for zone separator in header
                if j < display_w and j % zone_size == 0:
                    col_header += " "  # Space for zone separator

            # Calculate the actual width of the map (including borders and separators)
            map_width = 4 + display_w + (display_w // zone_size)  # row label + data + separators + right border
            title = f"2D MISMATCH MAP ({h}x{w} -> {display_h}x{display_w} blocks, {block_h}x{block_w} each)"
            centered_title = title.center(map_width)

            rows = [f"\n{centered_title}"]
            rows.append(col_header)

            # Create top border with zone separators
            border_line = "    ┌"
            for j in range(display_w):
                border_line += "─"
                if j < display_w - 1 and (j + 1) % zone_size == 0:
                    border_line += "┬"
            border_line += "┐"
            rows.append(border_line)

            for i in range(display_h):
                row = f"{i:3d} │"  # Row number with solid vertical border
                for j in range(display_w):
                    start_h = i * block_h
                    end_h = min((i + 1) * block_h, h)
                    start_w = j * block_w
                    end_w = min((j + 1) * block_w, w)

                    block_total = (end_h - start_h) * (end_w - start_w)
                    block_matches = torch.sum(close[start_h:end_h, start_w:end_w]).item()
                    block_mismatches = block_total - block_matches

                    if block_mismatches == 0:
                        row += "."
                    elif block_mismatches == block_total:
                        row += "!"
                    elif block_mismatches <= 9:
                        row += str(block_mismatches)
                    elif block_mismatches <= 35:
                        row += chr(ord('a') + block_mismatches - 10)
                    elif block_mismatches <= 61:
                        row += chr(ord('A') + block_mismatches - 36)
                    else:
                        row += "#"

                    # Add zone separator
                    if j < display_w - 1 and (j + 1) % zone_size == 0:
                        row += "│"
                row += "│"  # Right border
                rows.append(row)

            # Create bottom border with zone separators
            border_line = "    └"
            for j in range(display_w):
                border_line += "─"
                if j < display_w - 1 and (j + 1) % zone_size == 0:
                    border_line += "┴"
            border_line += "┘"
            rows.append(border_line)
            rows.append(
                "Legend of mismatch count per block: . = 0, ! = all mismatch, a-z = 10-35, A-Z = 36-61, # = 62+"
            )
            rows.append("Check 'python test/utils/zoom_diff.py --help' for more details to zoom in mismatch values")
            self._print_and_log("\n".join(rows), logfile)
        else:
            zone_size = calc_zone_size(w)

            # Build header with exact positions accounting for zone separators
            col_header = "     "  # Space for row numbers + border
            for j in range(0, w, max(1, w // 20)):  # Show every 20th position
                pos_str = f"{j:3d}"
                col_header += pos_str
                # Add padding to align with zone separators
                remaining_chars = max(1, w // 20) - len(pos_str)
                col_header += " " * remaining_chars

            # Calculate the actual width of the map (including borders and separators)
            map_width = 4 + w + (w // zone_size)  # row label + data + separators + right border
            title = f"2D MISMATCH MAP ({h}x{w})"
            centered_title = title.center(map_width)

            rows = [f"\n{centered_title}"]
            rows.append(col_header)

            # Create top border with zone separators
            border_line = "    ┌"
            for j in range(w):
                border_line += "─"
                if j < w - 1 and (j + 1) % zone_size == 0:
                    border_line += "┬"
            border_line += "┐"
            rows.append(border_line)

            for i in range(h):
                row_label = f"{i:3d} │" if i % max(1, h // 20) == 0 else "    │"

                # Build row with zone separators
                row_with_zones = row_label
                for j in range(w):
                    row_with_zones += "." if close[i, j] else "x"
                    if j < w - 1 and (j + 1) % zone_size == 0:
                        row_with_zones += "│"
                row_with_zones += "│"
                rows.append(row_with_zones)

            # Create bottom border with zone separators
            border_line = "    └"
            for j in range(w):
                border_line += "─"
                if j < w - 1 and (j + 1) % zone_size == 0:
                    border_line += "┴"
            border_line += "┘"
            rows.append(border_line)
            rows.append("Legend: . = match, x = mismatch")
            self._print_and_log("\n".join(rows), logfile)
