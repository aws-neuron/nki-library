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

"""Add BF16 torch.matmul baseline to a JSON cache file.

For each unique (M, K, N) shape in the input CSV that is not already in the
JSON cache, compiles a BF16 matmul (in a subprocess to avoid XLA core-lifetime
issues), profiles it, and writes the results (inference_time,
active_inference_time, mfu_percent, mbu_percent) to the cache.

Usage:
  python add_bf16_baseline_v2.py --csv metrics_summary.csv [--cache bf16_baseline_cache.json]
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(SCRIPT_DIR, "bf16_baseline_cache.json")
WORK_DIR = os.path.join(SCRIPT_DIR, "bf16_baseline_work")

# Opcodes that are setup/teardown — everything else is considered work.
# Copied from test/utils/metrics_collector.py MetricsCollector._SETUP_OPCODES
_SETUP_OPCODES = {
    "NOP",
    "SET_ORDERING_MODE",
    "EVENT_SEMAPHORE",
    "EVENT_SEMAPHORE_RANGE_CLEAR",
    "NOTIFY",
    "COMPARE_BRANCH",
    "DRAIN",
    "WRITE",
    "TENSOR_LOAD",
}


def _compute_active_inference_time(profiler_data: dict) -> float:
    """Compute active inference time excluding setup/teardown opcodes, including DMA.

    Copied from test/utils/metrics_collector.py MetricsCollector._compute_active_inference_time
    """
    min_ts, max_ts = float("inf"), 0

    for inst in profiler_data.get("instruction", []):
        if inst.get("opcode") not in _SETUP_OPCODES:
            ts = inst.get("timestamp", 0)
            dur = inst.get("duration", 0)
            min_ts = min(min_ts, ts)
            max_ts = max(max_ts, ts + dur)

    for dma in profiler_data.get("dma", []):
        if dma.get("semaphore_id") == "-1":
            continue
        ts = dma.get("timestamp", 0)
        dur = dma.get("duration", 0)
        min_ts = min(min_ts, ts)
        max_ts = max(max_ts, ts + dur)

    if min_ts == float("inf"):
        return -1.0
    return (max_ts - min_ts) * 1e-9  # ns → seconds


def load_cache(cache_path):
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache, cache_path):
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"Cache written to {cache_path} ({len(cache)} entries)")


def get_shapes_from_csv(csv_path):
    """Return list of (shape_name, M, K, N) from CSV, deduplicated by MxKxN key."""
    seen = {}
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            M, K, N = int(row["M"]), int(row["K"]), int(row["N"])
            key = f"{M}x{K}x{N}"
            if key not in seen:
                shape_name = row.get("Shape", "") or row.get("shape_name", "")
                model = row.get("Model", "")
                desc = f"{model} - {shape_name}" if model and model != "N/A" else shape_name
                seen[key] = (desc, M, K, N)
    return list(seen.values())


def missing_shapes(shapes, cache):
    """Return shapes not already in cache."""
    return [(desc, M, K, N) for desc, M, K, N in shapes if f"{M}x{K}x{N}" not in cache]


# ============ STEP 1: Compile BF16 matmuls ============

_COMPILE_SCRIPT = '''
import os
cache_dir = "{cache_dir}"
os.environ.setdefault("NEURON_RT_ENABLE_OCP_SATURATION", "1")
os.environ.setdefault("NEURON_RT_ENABLE_OCP", "1")
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn3")
os.environ["NEURON_CC_FLAGS"] = " --target=trn3"
os.environ["NEURON_COMPILE_CACHE_URL"] = cache_dir
os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
os.environ["NEURON_RT_NUM_CORES"] = "1"

import glob, torch, torch_xla.core.xla_model as xm
import numpy as np

M, K, N = {M}, {K}, {N}
A_np = np.random.randn(M, K).astype(np.float32)
B_np = np.random.randn(N, K).astype(np.float32)
device = xm.xla_device()
A_xla = torch.from_numpy(A_np).to(torch.bfloat16).to(device)
B_xla = torch.from_numpy(B_np).to(torch.bfloat16).to(device)
C = torch.matmul(A_xla, B_xla.T)
xm.mark_step()

neff_files = sorted(glob.glob(os.path.join(cache_dir, "**/*.neff"), recursive=True),
                    key=os.path.getmtime, reverse=True)
if neff_files:
    print(f"NEFF: {{neff_files[0]}}")
else:
    print("ERROR: No NEFF found"); import sys; sys.exit(1)
'''


def _compile_shape(key, desc, M, K, N):
    """Compile a single BF16 matmul in a subprocess. Returns NEFF path or None."""
    work = os.path.join(WORK_DIR, key)
    cache_dir = os.path.join(work, "neuron_cache")
    os.makedirs(cache_dir, exist_ok=True)

    script_path = os.path.join(work, "_compile.py")
    with open(script_path, "w") as f:
        f.write(_COMPILE_SCRIPT.format(cache_dir=cache_dir, M=M, K=K, N=N))

    r = subprocess.run([sys.executable, script_path], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"  COMPILE FAILED: {r.stderr[:300]}")
        return None

    for line in r.stdout.splitlines():
        if line.startswith("NEFF:"):
            return line.split("NEFF:")[1].strip()
    return None


def _profile_neff(key, neff_path):
    """Profile a NEFF. Returns dict with metrics or None on failure."""
    profile_dir = os.path.join(WORK_DIR, key, "profile")
    os.makedirs(profile_dir, exist_ok=True)

    ntff_path = os.path.join(profile_dir, "profile.ntff")
    detailed_json_path = os.path.join(profile_dir, "ntff_detailed.json")
    env = {**os.environ, "NEURON_RT_VISIBLE_CORES": "0", "NEURON_RT_NUM_CORES": "1"}

    # Capture
    r = subprocess.run(
        ["neuron-explorer", "capture", "-n", neff_path, "-s", ntff_path],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    if r.returncode != 0 or not os.path.exists(ntff_path):
        print(f"  Capture failed: {r.stderr[:200]}")
        return None

    for _ in range(30):
        if os.path.exists(ntff_path) and os.path.getsize(ntff_path) > 0:
            break
        time.sleep(1)

    # Summary JSON
    r = subprocess.run(
        ["neuron-explorer", "view", "-n", neff_path, "-s", ntff_path, "--output-format=summary-json"],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    if r.returncode != 0 or not r.stdout.strip():
        print(f"  Summary view failed: {r.stderr[:200]}")
        return None

    try:
        summary = next(iter(json.loads(r.stdout).values()), {})
    except Exception as e:
        print(f"  Summary parse failed: {e}")
        return None

    infer_sec = summary.get("total_time", -1)
    mbu_raw = summary.get("mbu_estimated_percent", -1)
    mfu_raw = summary.get("mfu_estimated_percent", -1)

    # Detailed JSON (for active inference time)
    subprocess.run(
        [
            "neuron-explorer",
            "view",
            "-n",
            neff_path,
            "-s",
            ntff_path,
            "--output-format=json",
            f"--output-file={detailed_json_path}",
        ],
        capture_output=True,
        timeout=300,
        env=env,
    )

    active_infer_sec = -1.0
    if os.path.exists(detailed_json_path):
        try:
            with open(detailed_json_path, "r") as f:
                active_infer_sec = _compute_active_inference_time(json.load(f))
        except Exception as e:
            print(f"  Active inference time parse failed: {e}")

    return {
        "inference_time": float(infer_sec) if infer_sec and infer_sec > 0 else -1.0,
        "active_inference_time": float(active_infer_sec),
        "mfu_percent": float(mfu_raw * 100) if mfu_raw is not None and mfu_raw >= 0 else -1.0,
        "mbu_percent": float(mbu_raw * 100) if mbu_raw is not None and mbu_raw >= 0 else -1.0,
    }


def run_baselines(csv_path, cache_path):
    """Compile and profile each shape in one pass."""
    cache = load_cache(cache_path)
    shapes = missing_shapes(get_shapes_from_csv(csv_path), cache)
    if not shapes:
        print("All shapes already in cache, nothing to do.")
        return

    os.makedirs(WORK_DIR, exist_ok=True)

    for desc, M, K, N in shapes:
        key = f"{M}x{K}x{N}"

        # Compile
        print(f"Compiling BF16 matmul {key} ({desc})...")
        neff_path = _compile_shape(key, desc, M, K, N)
        if not neff_path:
            continue

        # Profile
        print(f"Profiling {key} ({desc})...")
        metrics = _profile_neff(key, neff_path)
        if not metrics:
            time.sleep(10)
            continue

        cache[key] = {
            "shape_name": desc,
            "M": M,
            "K": K,
            "N": N,
            "num_lnc": 2,
            **metrics,
        }
        print(
            f"  InferenceTime={metrics['inference_time']:.6f}s  "
            f"ActiveInferenceTime={metrics['active_inference_time']:.6f}s  "
            f"MFU={metrics['mfu_percent']:.2f}%"
        )

        save_cache(cache, cache_path)
        time.sleep(5)

    save_cache(cache, cache_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BF16 baseline: compile and profile matmuls, write to JSON cache.")
    parser.add_argument("--csv", required=True, help="Input CSV with M, N, K columns (e.g. metrics_summary.csv)")
    parser.add_argument("--cache", default=DEFAULT_CACHE, help=f"JSON cache file (default: {DEFAULT_CACHE})")
    args = parser.parse_args()

    run_baselines(args.csv, args.cache)
