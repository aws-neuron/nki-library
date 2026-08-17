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

"""Process metrics from JSON files and generate CSV with profile uploads."""

import argparse
import csv
import glob
import json
import os
import subprocess
import uuid
from datetime import datetime, timezone


def upload_profile(
    test_dir, test_name, uploader=None, namespace="global", upload_cmd="neuron_explorer_profile_upload", timeout=300
):
    """Upload NEFF + NTFF and return profile URL."""
    neff_path = os.path.join(test_dir, "file.neff")
    ntff_path = os.path.join(test_dir, "infer_result", "profile.ntff")

    if not os.path.exists(neff_path):
        print(f"  ⚠ No file.neff in {test_dir}")
        return None
    if not os.path.exists(ntff_path):
        print(f"  ⚠ No profile.ntff in {test_dir}")
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uid = uuid.uuid4().hex[:8]
    unique_name = f"{test_name}_{timestamp}_{short_uid}"

    cmd = [
        upload_cmd,
        "-F",
        f"neff=@{neff_path}",
        "-F",
        f"ntff=@{ntff_path}",
        "-F",
        f"name={unique_name}",
        "--skip-version-check",
    ]
    if uploader:
        cmd.extend(["-F", f"uploader={uploader}"])
    if namespace and namespace != "global":
        cmd.extend(["-F", f"namespace={namespace}"])

    print(f"  Uploading {test_name}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        output = result.stdout + "\n" + result.stderr

        for line in output.splitlines():
            if "View profile at:" in line:
                url = line.split("View profile at:")[-1].strip()
                print(f"    ✓ {url}")
                return url

        print("    ⚠ No URL found in output")
        return None
    except Exception as e:
        print(f"    ✗ Upload failed: {e}")
        return None


def find_metrics_json(test_dir):
    """Find the metrics JSON file in test_dir/metrics/."""
    metrics_dir = os.path.join(test_dir, "metrics")
    if not os.path.isdir(metrics_dir):
        return None

    json_files = glob.glob(os.path.join(metrics_dir, "*.json"))
    return json_files[0] if json_files else None


def format_metric(col, value):
    """Format metric values."""
    if value == "" or value == "N/A" or value is None:
        return value
    try:
        v = float(value)
    except (ValueError, TypeError):
        return value

    if col == "MbuEstimatedPercent":
        return f"{v:.2f}%"
    elif col == "ProfilerMFU":
        return f"{v:.2f}%"
    elif col == "InferenceTime":
        return f"{v * 1_000_000:.2f} µs"
    return str(value)


def main(
    base_dir,
    bf16_cache_path=None,
    upload_profiles=False,
    profile_uploader=None,
    profile_namespace="global",
    upload_cmd="neuron_explorer_profile_upload",
):
    """Process metrics from test output directories and generate summary CSV."""
    base_dir = os.path.abspath(base_dir)
    output_csv = os.path.join(base_dir, "metrics_summary.csv")

    # Load BF16 baseline cache
    bf16_cache = {}
    if bf16_cache_path and os.path.exists(bf16_cache_path):
        with open(bf16_cache_path, "r") as f:
            bf16_cache = json.load(f)
        print(f"  Loaded BF16 cache with {len(bf16_cache)} entries from {bf16_cache_path}")

    print(f"  base_dir   = {base_dir}")
    print(f"  output_csv = {output_csv}\n")

    # Find all test directories
    test_dirs = []
    for entry in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path) and entry.startswith("out-"):
            test_dirs.append((entry, full_path))

    print(f"  Found {len(test_dirs)} test directories\n")

    # Collect all parameter keys from all tests
    all_param_keys = set()
    for _test_name, test_dir in test_dirs:
        metrics_json = find_metrics_json(test_dir)
        if metrics_json:
            with open(metrics_json, "r") as f:
                data = json.load(f)
                # Collect parameter keys (excluding metrics and metadata)
                exclude_keys = {
                    "_aws",
                    "TestName",
                    "KernelName",
                    "Target",
                    "LNCCores",
                    "Status",
                    "IsSuccessful",
                    "shape_name",
                    "M",
                    "K",
                    "N",
                    "TpbSgCyclesSum",
                    "InferenceTime",
                    "InferenceTimeTotal",
                    "MbuEstimatedPercent",
                    "ProfilerMFU",
                    "BF16BaselineInferenceTime",
                    "BF16BaselineActiveInferenceTime",
                    "BF16BaselineMFU",
                    "BF16BaselineMBU",
                    "ActiveInferenceTime",
                    "ActiveInferenceTimeOutliers",
                    "ActiveInferenceTimeQCD",
                    "ActiveInferenceTimeSamples",
                    "ArtifactParseTime",
                    "BirToNeffTime",
                    "DeterminismCheckTime",
                    "InputDumpTime",
                    "KernelAPI",
                    "MlirToBirTime",
                    "FrontendTraceTime",
                    "SftpDownloadTime",
                    "SftpUploadTime",
                    "SimulationTime",
                    "TraceMode",
                }
                for key in data.keys():
                    if (
                        key not in exclude_keys
                        and not key.startswith("Compilation")
                        and not key.startswith("Host")
                        and not key.startswith("File")
                        and not key.startswith("Core")
                        and not key.startswith("Golden")
                        and not key.startswith("Validation")
                        and not key.startswith("TPB")
                        and not key.startswith("Profile")
                        and not key.startswith("Neuron")
                        and not key.startswith("Cycle")
                        and not key.startswith("Accuracy")
                        and not key.startswith("Elapsed")
                    ):
                        all_param_keys.add(key)

    all_param_keys = sorted(all_param_keys)
    print(f"  Found {len(all_param_keys)} parameter keys: {all_param_keys}\n")

    # Process each test directory
    output_rows = []
    for test_name, test_dir in test_dirs:
        metrics_json = find_metrics_json(test_dir)
        if not metrics_json:
            print(f"  ⚠ No metrics JSON in {test_dir}")
            continue

        with open(metrics_json, "r") as f:
            data = json.load(f)

        # Extract core fields
        shape_name = data.get("shape_name", "N/A")

        # Split shape_name into model and shape
        if " - " in shape_name:
            model, shape = shape_name.split(" - ", 1)
        else:
            model = "N/A"
            shape = shape_name

        M = data.get("M", "N/A")
        N = data.get("N", "N/A")
        K = data.get("K", "N/A")

        # Extract metrics
        inference_time = data.get("InferenceTime", "N/A")
        active_inference_time = data.get("ActiveInferenceTime", "N/A")
        mbu_percent = data.get("MbuEstimatedPercent", "N/A")
        profiler_mfu = data.get("ProfilerMFU", "N/A")

        # Extract parameters
        params = {key: data.get(key, "N/A") for key in all_param_keys}

        # Extract BF16 baseline metrics from cache
        bf16_key = f"{M}x{K}x{N}"
        bf16 = bf16_cache.get(bf16_key, {})
        bf16_inference_time = bf16.get("inference_time") if bf16.get("inference_time", -1) > 0 else None
        bf16_active_inference_time = (
            bf16.get("active_inference_time") if bf16.get("active_inference_time", -1) > 0 else None
        )
        bf16_mfu = bf16.get("mfu_percent") if bf16.get("mfu_percent", -1) > 0 else None

        # Compute speedup vs BF16 baseline
        speedup = "N/A"
        active_speedup = "N/A"
        try:
            if (
                bf16_inference_time is not None
                and bf16_inference_time > 0
                and inference_time != "N/A"
                and float(inference_time) > 0
            ):
                speedup = f"{float(bf16_inference_time) / float(inference_time):.2f}x"
            if (
                bf16_active_inference_time is not None
                and bf16_active_inference_time > 0
                and active_inference_time != "N/A"
                and float(active_inference_time) > 0
            ):
                active_speedup = f"{float(bf16_active_inference_time) / float(active_inference_time):.2f}x"
        except (ValueError, TypeError):
            pass

        # Upload profile if requested
        profile_url = ""
        if upload_profiles:
            profile_url = (
                upload_profile(
                    test_dir, test_name, uploader=profile_uploader, namespace=profile_namespace, upload_cmd=upload_cmd
                )
                or ""
            )

        row = {
            "Model": model,
            "Shape": shape,
            "M": M,
            "N": N,
            "K": K,
            "InferenceTime (µs)": format_metric("InferenceTime", inference_time),
            "ActiveInferenceTime (µs)": format_metric("InferenceTime", active_inference_time),
            "MBU (%)": format_metric("MbuEstimatedPercent", mbu_percent),
            "MFU (%)": format_metric("ProfilerMFU", profiler_mfu),
            "BF16 InferenceTime (µs)": format_metric("InferenceTime", bf16_inference_time)
            if bf16_inference_time and bf16_inference_time > 0
            else "N/A",
            "BF16 ActiveInferenceTime (µs)": format_metric("InferenceTime", bf16_active_inference_time)
            if bf16_active_inference_time and bf16_active_inference_time > 0
            else "N/A",
            "BF16 MFU (%)": format_metric("ProfilerMFU", bf16_mfu) if bf16_mfu and bf16_mfu > 0 else "N/A",
            "Speedup vs BF16": speedup,
            "Active Speedup vs BF16": active_speedup,
            "Profile URL": profile_url,
            **params,
        }
        output_rows.append(row)

    # Define header
    header = [
        "Model",
        "Shape",
        "M",
        "N",
        "K",
        "InferenceTime (µs)",
        "ActiveInferenceTime (µs)",
        "MBU (%)",
        "MFU (%)",
        "BF16 InferenceTime (µs)",
        "BF16 ActiveInferenceTime (µs)",
        "BF16 MFU (%)",
        "Speedup vs BF16",
        "Active Speedup vs BF16",
        "Profile URL",
    ] + all_param_keys

    # Print to console
    col_widths = {col: len(col) for col in header}
    for row in output_rows:
        for col in header:
            val = str(row.get(col, ""))
            if col == "Profile URL":
                val = val[:40]
            col_widths[col] = max(col_widths[col], len(val))

    def print_row(values):
        parts = []
        for col in header:
            val = str(values.get(col, ""))
            if col == "Profile URL" and len(val) > 40:
                val = val[:37] + "..."
            parts.append(val.ljust(col_widths[col]))
        print(" | ".join(parts))

    def print_sep():
        print("-+-".join("-" * col_widths[col] for col in header))

    print("\n")
    print_sep()
    print_row({col: col for col in header})
    print_sep()
    for row in output_rows:
        print_row(row)
    print_sep()

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    print(f"\nCSV written to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process metrics from JSON files and generate CSV with optional profile uploads."
    )
    parser.add_argument("base_dir", help="Path to the base directory containing out-* test subdirectories.")
    parser.add_argument(
        "--cache", default=None, help="Path to BF16 baseline JSON cache file (bf16_baseline_cache.json)."
    )
    parser.add_argument(
        "--upload-profiles",
        action="store_true",
        default=False,
        help="Upload NEFF/NTFF to NeuronExplorer and add profile URLs to CSV.",
    )
    parser.add_argument(
        "--profile-uploader", default=None, help="Custom uploader name for neuron_explorer_profile_upload."
    )
    parser.add_argument(
        "--profile-namespace", default="global", help="Namespace for uploaded profiles (default: global)."
    )
    parser.add_argument(
        "--upload-cmd",
        default="neuron_explorer_profile_upload",
        help="Path to neuron_explorer_profile_upload script (default: neuron_explorer_profile_upload).",
    )

    args = parser.parse_args()
    main(
        args.base_dir,
        bf16_cache_path=args.cache,
        upload_profiles=args.upload_profiles,
        profile_uploader=args.profile_uploader,
        profile_namespace=args.profile_namespace,
        upload_cmd=args.upload_cmd,
    )
