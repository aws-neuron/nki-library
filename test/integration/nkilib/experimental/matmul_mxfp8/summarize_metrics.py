#!/usr/bin/env python3
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

"""Generate summary CSV from metrics_summary.csv.

Groups shapes into "Core Attention" (Attn in name) and "Full model ex core attention"
(everything else), then computes average Speedup vs BF16 and MFU for each config.

Each cell is "Speedup vs BF16 (MFU%)" where speedup = BF16_time / kernel_time.

Usage:
  python summarize_metrics.py metrics_summary.csv [-o summary.csv]
"""

import argparse
import csv
import os

# Config definitions: (config_name, match_criteria)
# match_criteria is a dict of CSV column -> value
CONFIGS = [
    (
        "MXFP8 Pre-quantized Swizzled",
        {"lhs_dtype": "MXFP8", "lhs_is_swizzled": "True", "enable_scale_packing": "True", "spill_reload": "False"},
    ),
    (
        "BF16 Swizzled (no scale pack)",
        {"lhs_dtype": "BFLOAT16", "lhs_is_swizzled": "True", "enable_scale_packing": "False", "spill_reload": "False"},
    ),
    (
        "BF16 Swizzled (scale pack)",
        {"lhs_dtype": "BFLOAT16", "lhs_is_swizzled": "True", "enable_scale_packing": "True", "spill_reload": "False"},
    ),
    (
        "BF16 Swizzled (spill reload)",
        {"lhs_dtype": "BFLOAT16", "lhs_is_swizzled": "True", "enable_scale_packing": "False", "spill_reload": "True"},
    ),
    (
        "BF16 Swizzled (scale+spill)",
        {"lhs_dtype": "BFLOAT16", "lhs_is_swizzled": "True", "enable_scale_packing": "True", "spill_reload": "True"},
    ),
    ("BF16 Unswizzled", {"lhs_dtype": "BFLOAT16", "lhs_is_swizzled": "False"}),
]

LAYER_GROUPS = [
    ("Full model ex core attention", lambda shape: "Attn" not in shape),
    ("Core Attention", lambda shape: "Attn" in shape),
]


def parse_metric(s):
    """Parse '134.76 µs' or '60.79%' or '3.00x' to float, return None if N/A or negative."""
    if not s or s == "N/A":
        return None
    s = s.replace(" µs", "").replace("%", "").replace("x", "").replace(",", "").strip()
    try:
        v = float(s)
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


def matches_config(row, criteria):
    """Return True if row matches all config criteria."""
    return all(row.get(k) == v for k, v in criteria.items())


def fmt_cell(speedup, mfu):
    """Format speedup and MFU into a summary cell string."""
    if speedup is None:
        return "N/A"
    mfu_str = f"{mfu:.2f}%" if mfu is not None else "N/A"
    return f"{speedup:.2f}x ({mfu_str})"


def main():
    """Generate config x layer-group summary table from metrics CSV."""
    parser = argparse.ArgumentParser(description="Summarize metrics CSV into config x layer-group table.")
    parser.add_argument("csv_path", help="Path to metrics_summary.csv")
    parser.add_argument("-o", "--output", default=None, help="Output CSV path (default: summary.csv next to input)")
    args = parser.parse_args()

    output_path = args.output or os.path.join(os.path.dirname(os.path.abspath(args.csv_path)), "summary.csv")

    with open(args.csv_path, "r") as f:
        rows = list(csv.DictReader(f))

    # For each (config, layer_group), collect inference times and MFUs
    # We compute: avg speedup = avg(BF16_time / kernel_time), avg MFU
    results = {}

    for config_name, criteria in CONFIGS:
        config_rows = [row for row in rows if matches_config(row, criteria)]
        if not config_rows:
            for group_name, _ in LAYER_GROUPS:
                results[(config_name, group_name)] = (None, None, None, None)
            continue

        for group_name, group_filter in LAYER_GROUPS:
            group_rows = [row for row in config_rows if group_filter(row["Shape"])]

            speedups = []
            active_speedups = []
            mfus = []

            for row in group_rows:
                sp = parse_metric(row.get("Speedup vs BF16"))
                asp = parse_metric(row.get("Active Speedup vs BF16"))
                mfu = parse_metric(row.get("MFU (%)"))

                if sp is not None:
                    speedups.append(sp)
                if asp is not None:
                    active_speedups.append(asp)
                if mfu is not None:
                    mfus.append(mfu)

            avg_sp = sum(speedups) / len(speedups) if speedups else None
            avg_asp = sum(active_speedups) / len(active_speedups) if active_speedups else None
            avg_mfu = sum(mfus) / len(mfus) if mfus else None

            results[(config_name, group_name)] = (avg_sp, avg_asp, avg_mfu, len(group_rows))

    # Also compute BF16 baseline average MFU per layer group (the "1x" reference column)
    bf16_baseline = {}
    for group_name, group_filter in LAYER_GROUPS:
        # Use any config's rows — BF16 baseline is the same for all configs with same shape
        group_rows = [row for row in rows if group_filter(row["Shape"])]
        bf16_mfus = []
        for row in group_rows:
            mfu = parse_metric(row.get("BF16 MFU (%)"))
            if mfu is not None:
                bf16_mfus.append(mfu)
        bf16_baseline[group_name] = sum(bf16_mfus) / len(bf16_mfus) if bf16_mfus else None

    # Print and write summary
    print("\n=== Summary: Speedup vs BF16 (MFU%) ===\n")

    # Build table
    header = ["Model", "Layers", "BF16 Baseline (MFU%)"]
    for config_name, _ in CONFIGS:
        header.append(f"{config_name} - Full Kernel")
        header.append(f"{config_name} - Active (excl neff switch)")

    summary_rows = []
    for group_name, _ in LAYER_GROUPS:
        row = {
            "Model": "Qwen3-8B-TP4",
            "Layers": group_name,
            "BF16 Baseline (MFU%)": f"1x ({bf16_baseline[group_name]:.2f}%)" if bf16_baseline[group_name] else "N/A",
        }
        for config_name, _ in CONFIGS:
            avg_sp, avg_asp, avg_mfu, count = results[(config_name, group_name)]
            row[f"{config_name} - Full Kernel"] = fmt_cell(avg_sp, avg_mfu)
            row[f"{config_name} - Active (excl neff switch)"] = fmt_cell(avg_asp, avg_mfu)
        summary_rows.append(row)

    # Print table
    col_widths = {col: len(col) for col in header}
    for row in summary_rows:
        for col in header:
            col_widths[col] = max(col_widths[col], len(str(row.get(col, ""))))

    def print_row(vals):
        print(" | ".join(str(vals.get(col, "")).ljust(col_widths[col]) for col in header))

    def print_sep():
        print("-+-".join("-" * col_widths[col] for col in header))

    print_sep()
    print_row({col: col for col in header})
    print_sep()
    for row in summary_rows:
        print_row(row)
    print_sep()

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSummary CSV written to: {output_path}")


if __name__ == "__main__":
    main()
