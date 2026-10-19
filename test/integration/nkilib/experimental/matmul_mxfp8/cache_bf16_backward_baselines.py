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

"""Measure BF16 baselines for the FUSED matmul backward pass.

The backward kernel (matmul_mxfp8_backward) produces both gradients from one
forward shape (M, K, N), where the forward pass is Y = X @ W^T with X=[M, K],
W=[N, K], Y=[M, N]:

    dX = dY @ W               logical matmul (M, N, K)
    dW = dY^T @ X             logical matmul (N, M, K)

WHY THIS IS A SEPARATE MEASUREMENT, not a sum of cached forward entries:

bf16_baseline_cache.json holds standalone `torch.matmul` measurements, each
compiled and profiled in isolation. A real BF16 backward pass is not two isolated
matmuls. The two phases share the dY operand, are compiled and scheduled as one
graph, and pay a different mix of DMA, transpose and layout costs than either
matmul does alone. Adding two isolated numbers produces a figure no BF16
implementation actually achieves, so a speedup computed against it would be
unfounded -- it could flatter the kernel or penalise it, with no way to tell
which. This script instead compiles ONE graph containing both gradient
computations and profiles that, which is the number the fused kernel must beat.

The cache (bf16_backward_baseline_cache.json) ships EMPTY. Until it is populated
on hardware, BackwardTestConfig.get_bf16_baseline returns None and the perf suite
reports raw runtime and MFU with no speedup column.

Structure and profiling logic deliberately mirror cache_bf16_baselines.py (same
compile-in-a-subprocess pattern, same neuron-explorer capture/view calls, same
active-inference-time computation) so the two baselines stay comparable and there
is one shape of code to maintain.

Usage:
  # Report which shapes are cached and which are missing (no hardware needed)
  python3 cache_bf16_backward_baselines.py --check

  # Measure the missing shapes on a TRN3 host and write them to the cache
  python3 cache_bf16_backward_baselines.py --measure

  # Restrict to particular TP degrees (default: 1 and 2)
  python3 cache_bf16_backward_baselines.py --measure --tp 1
"""

import argparse
import json
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(SCRIPT_DIR, "bf16_backward_baseline_cache.json")
DEFAULT_MODEL_CONFIG = os.path.join(SCRIPT_DIR, "qwen3_235B-A22B.json")
WORK_DIR = os.path.join(SCRIPT_DIR, "bf16_backward_baseline_work")

# Opcodes that are setup/teardown -- everything else is considered work.
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
    return (max_ts - min_ts) * 1e-9  # ns -> seconds


def load_json(path):
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache, cache_path):
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"Cache written to {cache_path} ({len(cache)} entries)")


def projection_forward_shapes(model_config_path, TP):
    """Forward (name, M, K, N) shapes for the projections the matmul kernel owns.

    Mirrors model_config_reader.generate_attention_shapes, restricted to the
    fused-QKV and O projections. Kept as a local computation so this script runs
    standalone on a bare TRN3 host with no package imports available.
    """
    cfg = load_json(model_config_path)
    S = cfg["max_seq_len"]
    H = cfg["dim"]
    Q = cfg["n_heads"]
    KV = cfg["n_kv_heads"]
    h = cfg["head_dim"]

    # Fused QKV: [S, H] @ [H, (Q*h + 2*KV*h)/TP]
    qkv_out = (Q * h + 2 * KV * h) // TP
    # O projection: [S, Q*h/TP] @ [Q*h/TP, H]. Note Q*h is 2*H for this model,
    # so O-proj K is 8192/TP.
    attn_out = Q * h // TP

    return [
        ("FWD-FuseQKV", S, H, qkv_out),
        ("FWD-OProj", S, attn_out, H),
    ]


def collect(model_config_path, tps):
    """Build the list of (label, forward (M, K, N)) to measure."""
    entries = []
    for TP in tps:
        for name, M, K, N in projection_forward_shapes(model_config_path, TP):
            entries.append((f"Qwen3-235B-TP{TP} - {name}", (M, K, N)))
    return entries


def key_of(shape):
    return "x".join(str(d) for d in shape)


# ============ STEP 1: Compile the BF16 backward pass ============
#
# Both gradients live in ONE graph, so the profile covers the whole backward pass
# rather than two separately-scheduled matmuls. dY is a single tensor feeding both,
# which is exactly the operand sharing a summed baseline would fail to capture.
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

# Forward shape: X=[M, K], W=[N, K], Y=[M, N] for Y = X @ W^T
M, K, N = {M}, {K}, {N}
X_np = np.random.randn(M, K).astype(np.float32)
W_np = np.random.randn(N, K).astype(np.float32)
dY_np = np.random.randn(M, N).astype(np.float32)

device = xm.xla_device()
X = torch.from_numpy(X_np).to(torch.bfloat16).to(device)
W = torch.from_numpy(W_np).to(torch.bfloat16).to(device)
dY = torch.from_numpy(dY_np).to(torch.bfloat16).to(device)

# Both gradients in one graph, sharing dY -- this is the whole backward pass.
dX = torch.matmul(dY, W)             # [M, N] @ [N, K] -> [M, K]
dW = torch.matmul(dY.T, X)           # [N, M] @ [M, K] -> [N, K]
xm.mark_step()

neff_files = sorted(glob.glob(os.path.join(cache_dir, "**/*.neff"), recursive=True),
                    key=os.path.getmtime, reverse=True)
if neff_files:
    print(f"NEFF: {{neff_files[0]}}")
else:
    print("ERROR: No NEFF found"); import sys; sys.exit(1)
'''


def _compile_shape(key, M, K, N):
    """Compile a single BF16 backward pass in a subprocess. Returns NEFF path or None."""
    work = os.path.join(WORK_DIR, key)
    cache_dir = os.path.join(work, "neuron_cache")
    os.makedirs(cache_dir, exist_ok=True)

    script_path = os.path.join(work, "_compile.py")
    with open(script_path, "w") as f:
        f.write(_COMPILE_SCRIPT.format(cache_dir=cache_dir, M=M, K=K, N=N))

    r = subprocess.run([sys.executable, script_path], capture_output=True, text=True, timeout=900)
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


def report(entries, cache):
    """Print per-entry cache status. Returns the entries with no cached baseline."""
    missing = []
    n_ready = 0
    for label, fwd in entries:
        entry = cache.get(key_of(fwd))
        if entry and entry.get("inference_time", -1.0) > 0:
            n_ready += 1
            status = f"READY    BF16 backward = {entry['inference_time'] * 1e6:9.1f}us"
        else:
            status = "MISSING"
            missing.append((label, fwd))
        print(f"  {label:36s} fwd={key_of(fwd):18s} {status}")

    print(f"\n{n_ready}/{len(entries)} shapes have a measured BF16 backward baseline.")
    if missing:
        print(f"{len(missing)} shape(s) need measuring on a TRN3 host (--measure).")
    return missing


def measure(missing, cache, cache_path):
    """Compile and profile a real BF16 backward pass for each missing shape."""
    if not missing:
        print("Nothing to measure.")
        return

    os.makedirs(WORK_DIR, exist_ok=True)

    for label, fwd in missing:
        M, K, N = fwd
        key = key_of(fwd)

        print(f"Compiling BF16 backward {key} ({label})...")
        neff_path = _compile_shape(key, M, K, N)
        if not neff_path:
            continue

        print(f"Profiling {key} ({label})...")
        metrics = _profile_neff(key, neff_path)
        if not metrics:
            time.sleep(10)
            continue

        cache[key] = {
            "shape_name": f"{label} [BF16 backward]",
            "M": M,
            "K": K,
            "N": N,
            "input_grad_shape": key_of((M, N, K)),
            "weight_grad_shape": key_of((N, M, K)),
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
    p = argparse.ArgumentParser(description="Measured BF16 baselines for the fused matmul backward pass.")
    p.add_argument("--cache", default=DEFAULT_CACHE, help="Backward baseline cache JSON.")
    p.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG, help="Model config JSON.")
    p.add_argument("--tp", type=int, action="append", help="TP degree (repeatable). Default: 1, 2.")
    p.add_argument("--check", action="store_true", help="Report cache status (default action).")
    p.add_argument("--measure", action="store_true", help="Measure missing shapes. Requires a TRN3 host.")
    args = p.parse_args()

    tps = args.tp or [1, 2]
    cache = load_json(args.cache)
    print(f"Backward baseline cache: {args.cache} ({len(cache)} entries)\n")

    entries = collect(args.model_config, tps)
    missing = report(entries, cache)

    if args.measure:
        measure(missing, cache, args.cache)
