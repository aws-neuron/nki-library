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

"""Generate ablation summary tables for scale packing and spill reload impact.

Compares BF16 swizzled configs (tests 2-5) using Full model ex core attention averages.

Ablation 1 - Scale Packing Impact (no spill reload):
  test 2 (no scale pack, no spill) vs test 3 (scale pack, no spill)

Ablation 2 - Spill Reload Impact (no scale packing):
  test 2 (no scale pack, no spill) vs test 4 (no scale pack, spill)

Ablation 3 - Combined Impact:
  test 2 (baseline) vs test 5 (scale pack + spill)

Usage:
  python ablation_summary.py metrics_summary.csv [-o ablation_summary.csv]
"""

import argparse
import csv
import os

CONFIGS = {
    "BF16 Swizzled (no scale pack, no spill)": {
        "lhs_dtype": "BFLOAT16",
        "lhs_is_swizzled": "True",
        "enable_scale_packing": "False",
        "spill_reload": "False",
    },
    "BF16 Swizzled (scale pack, no spill)": {
        "lhs_dtype": "BFLOAT16",
        "lhs_is_swizzled": "True",
        "enable_scale_packing": "True",
        "spill_reload": "False",
    },
    "BF16 Swizzled (no scale pack, spill)": {
        "lhs_dtype": "BFLOAT16",
        "lhs_is_swizzled": "True",
        "enable_scale_packing": "False",
        "spill_reload": "True",
    },
    "BF16 Swizzled (scale pack + spill)": {
        "lhs_dtype": "BFLOAT16",
        "lhs_is_swizzled": "True",
        "enable_scale_packing": "True",
        "spill_reload": "True",
    },
}


def parse_metric(s):
    if not s or s == "N/A":
        return None
    s = s.replace(" µs", "").replace("%", "").replace("x", "").replace(",", "").strip()
    try:
        v = float(s)
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


def matches(row, criteria):
    """Return True if row matches all key-value criteria."""
    return all(row.get(k) == v for k, v in criteria.items())


def avg_metrics(rows):
    """Return (avg_speedup, avg_active_speedup, avg_mfu, avg_inference_time_us) for non-attention rows."""
    non_attn = [row for row in rows if "Attn" not in row["Shape"]]
    speedups, active_speedups, mfus, times = [], [], [], []
    for row in non_attn:
        sp = parse_metric(row.get("Speedup vs BF16"))
        asp = parse_metric(row.get("Active Speedup vs BF16"))
        mfu = parse_metric(row.get("MFU (%)"))
        t = parse_metric(row.get("InferenceTime (µs)"))
        if sp:
            speedups.append(sp)
        if asp:
            active_speedups.append(asp)
        if mfu:
            mfus.append(mfu)
        if t:
            times.append(t)
    return (
        sum(speedups) / len(speedups) if speedups else None,
        sum(active_speedups) / len(active_speedups) if active_speedups else None,
        sum(mfus) / len(mfus) if mfus else None,
        sum(times) / len(times) if times else None,
    )


def fmt(val, suffix=""):
    """Format a float value with suffix, or return N/A."""
    return f"{val:.2f}{suffix}" if val is not None else "N/A"


def delta(a, b):
    """Return percentage change from a to b."""
    if a is None or b is None or a == 0:
        return "N/A"
    pct = (b - a) / a * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def main():
    """Generate ablation summary CSV comparing BF16 swizzled configs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    output_path = args.output or os.path.join(os.path.dirname(os.path.abspath(args.csv_path)), "ablation_summary.csv")

    with open(args.csv_path, "r") as f:
        rows = list(csv.DictReader(f))

    # Compute per-config averages (non-attention only)
    stats = {}
    for name, criteria in CONFIGS.items():
        config_rows = [row for row in rows if matches(row, criteria)]
        stats[name] = avg_metrics(config_rows)

    baseline_name = "BF16 Swizzled (no scale pack, no spill)"

    # Build ablation table rows
    ablation_rows = []
    header = [
        "Comparison",
        "Baseline Config",
        "Test Config",
        "Baseline Speedup vs BF16",
        "Test Speedup vs BF16",
        "Δ Speedup",
        "Baseline Active Speedup",
        "Test Active Speedup",
        "Δ Active Speedup",
        "Baseline MFU (%)",
        "Test MFU (%)",
        "Δ MFU",
        "Baseline Avg Time (µs)",
        "Test Avg Time (µs)",
        "Δ Time",
    ]

    comparisons = [
        ("Scale Packing Impact", baseline_name, "BF16 Swizzled (scale pack, no spill)"),
        ("Spill Reload Impact", baseline_name, "BF16 Swizzled (no scale pack, spill)"),
        ("Scale Pack + Spill Combined", baseline_name, "BF16 Swizzled (scale pack + spill)"),
        ("Spill on top of Scale Pack", "BF16 Swizzled (scale pack, no spill)", "BF16 Swizzled (scale pack + spill)"),
    ]

    for comp_name, base_name, test_name in comparisons:
        b = stats[base_name]
        t = stats[test_name]
        ablation_rows.append(
            {
                "Comparison": comp_name,
                "Baseline Config": base_name,
                "Test Config": test_name,
                "Baseline Speedup vs BF16": fmt(b[0], "x"),
                "Test Speedup vs BF16": fmt(t[0], "x"),
                "Δ Speedup": delta(b[0], t[0]),
                "Baseline Active Speedup": fmt(b[1], "x"),
                "Test Active Speedup": fmt(t[1], "x"),
                "Δ Active Speedup": delta(b[1], t[1]),
                "Baseline MFU (%)": fmt(b[2], "%"),
                "Test MFU (%)": fmt(t[2], "%"),
                "Δ MFU": delta(b[2], t[2]),
                "Baseline Avg Time (µs)": fmt(b[3]),
                "Test Avg Time (µs)": fmt(t[3]),
                "Δ Time": delta(b[3], t[3]),
            }
        )

    # Print
    print("\n=== Ablation: Scale Packing & Spill Reload (Full model ex core attention) ===\n")

    col_w = {col: len(col) for col in header}
    for row in ablation_rows:
        for col in header:
            col_w[col] = max(col_w[col], len(str(row.get(col, ""))))

    sep = "-+-".join("-" * col_w[col] for col in header)

    def pr(vals):
        print(" | ".join(str(vals.get(col, "")).ljust(col_w[col]) for col in header))

    print(sep)
    pr({col: col for col in header})
    print(sep)
    for row in ablation_rows:
        pr(row)
    print(sep)

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(ablation_rows)

    print(f"\nAblation CSV written to: {output_path}")


if __name__ == "__main__":
    main()
