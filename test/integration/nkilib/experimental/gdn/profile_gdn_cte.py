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
"""Accuracy + device-metric comparison of the two gdn_cte prefill algorithms.

This is the script that produces the comparison tables in
``src/nkilib_src/nkilib/experimental/gdn/README.md``. It is NOT a pytest test (the tests
live in ``test_gdn_cte.py``); it exists because the tables need two things pytest does not
report: the max relative error against the CPU reference per shape, and profiler metrics
from a trace captured with the REAL inputs.

DGE NOTIFICATIONS ARE OFF, deliberately: worth ~4% of every number here.
`capture --enable-dge-notifs` (equivalently `NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1`) notifies
on every dynamic DMA packet, and these kernels issue ~100k per launch -- A/B on the same
NEFF, schur at S=12288 measured 3459 us without and 3613 us with, ~4.5% (both figures predate
the instruction-count optimization pass; the ratio is what carries over). The ratio between the
algorithms is unaffected, but do not compare an instrumented figure against a clean one.

Phases, so a slow batch of compiles does not have to serialize behind profiling:
  ``run``      compile to a NEFF, execute on device, score against the torch reference.
  ``profile``  capture a trace of the already-compiled NEFF and read the metrics.
  ``both``     (default) run then profile, per configuration.
  ``table``    only re-emit the markdown table from result JSONs already on disk.

Usage:
    NEURON_RT_VISIBLE_CORES=<cores> python profile_gdn_cte.py \\
        --shapes "1,1,1,8192 1,2,4,12288" --algorithms schur,neumann

    # The compute dtype is the dtype of `query`, so precision is a sweep axis:
    NEURON_RT_VISIBLE_CORES=<cores> python profile_gdn_cte.py \\
        --shapes 1,2,4,12288 --dtypes bfloat16,float32

    # Compile many configurations concurrently, then profile serially on a quiet device:
    # parallel profiling makes the runs contend for HBM bandwidth and inflates total_time.
    for cfg in ...; do python profile_gdn_cte.py --phase run --shapes $cfg & done; wait
    python profile_gdn_cte.py --phase profile --shapes "..."

    # LNC=2 shards value heads across a 2-program grid.
    NEURON_LOGICAL_NC_CONFIG=2 NEURON_RT_VISIBLE_CORES=32-47 python profile_gdn_cte.py \\
        --shapes 1,2,4,12288 --algorithms schur --lnc 2

Requirements: a Trainium host with a free NeuronCore, ``neuron-explorer`` on PATH, and
``neuronx-cc`` importable/executable from the same environment as ``nki``.
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

# The kernels are imported from the source tree, not an installed wheel, so this script
# works in a checkout without an editable install.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

import neuron_dtypes as dt  # noqa: E402

# `nki.framework.compiled` ships an empty type stub (it is not public API), so the checker
# sees no members; the NEFF path this script needs requires `artifacts_dir`, which the
# public `nki.jit` does not accept.
from nki.compiler.frontend import TracerFrontend  # noqa: E402  # ty: ignore[unresolved-import]
from nki.framework.compiled import CompileKernel  # noqa: E402  # ty: ignore[unresolved-import]
from nkilib_src.nkilib.experimental.gdn.gdn_cte import gdn_cte  # noqa: E402
from nkilib_src.nkilib.experimental.gdn.gdn_cte_utils import (  # noqa: E402
    ALGORITHM_NEUMANN,
    ALGORITHM_SCHUR,
    ALGORITHMS,
    build_mask_pack,
)

HEAD_DIM = 128
# The kernel's tensor inputs, in signature order. Also the ifmap NAMES: the NEFF takes its
# input names from the kernel's parameter names. Both `mask_pack` and `recurrent_state` are
# optional, so which of them a NEFF actually has depends on the algorithm -- see
# `input_names`.
INPUT_ORDER = ["query", "key", "value", "g_log", "beta", "mask_pack", "recurrent_state"]
# The one input the kernels alias to an output, and the suffix the NEFF gives it.
ALIASED_INPUT = "recurrent_state"
_ALIAS_SUFFIX = ".must_alias_input"
# Shapes the README table covers: every device shape in test_gdn_cte.py.
# The last four have chunk counts that are NOT multiples of the default group size, so
# they exercise the residual group (5, 7, 17 and 65 schur chunks -> residual 1, 3, 1, 1).
DEFAULT_SHAPES = (
    "1,1,1,128 1,1,1,256 1,2,4,256 32,1,1,128 1,2,4,768 32,1,1,2304 1,1,1,8192 1,2,4,12288 "
    "1,1,1,640 1,2,4,896 32,1,1,2176 1,1,1,8320"
)


def input_names(algorithm):
    """The kernel inputs THIS algorithm takes, in positional order.

    Args:
        algorithm (str): "schur" or "neumann".

    Returns:
        list: Input names, which are also the NEFF's input names.

    Notes:
        The neumann path takes neither `recurrent_state` (it always starts from a zero
        state) nor `mask_pack` (it builds its constants on device), so its NEFF has neither
        input and the ifmap must not offer them.
    """
    if algorithm == ALGORITHM_NEUMANN:
        return [name for name in INPUT_ORDER if name not in (ALIASED_INPUT, "mask_pack")]
    return list(INPUT_ORDER)


def make_inputs(batch, n_k_heads, n_v_heads, seqlen, dtype="bfloat16", algorithm="schur"):
    """Build the kernel inputs, in the SAME distribution (and seed) as test_gdn_cte.py.

    Args:
        batch (int): Batch size.
        n_k_heads (int): Key/query head count.
        n_v_heads (int): Value head count, a multiple of n_k_heads.
        seqlen (int): Sequence length.
        dtype (str): Dtype of q/k/v, which is the kernel's COMPUTE DTYPE.
        algorithm (str): Which algorithm the inputs are for -- it decides whether the
            key/query heads are duplicated and whether a state is supplied.

    Returns:
        dict: Input name -> numpy array, ready to pass to the kernel positionally.

    Notes:
        Matching the test's distribution is what makes the errors reported here the same
        quantity the tests bound: q/k L2-normalized (the kernels apply the query scale
        themselves), gates negative, beta in (0, 1), and a zero initial state.

        THE NEUMANN PATH GETS ITS KEY/QUERY HEADS DUPLICATED HERE, on the host, and gets no
        `recurrent_state`: that kernel has a flat [B, S, D] layout with no head sharing and
        always starts from zero. Doing the duplication in the input builder keeps it OUT of
        everything measured -- it happens before the NEFF is compiled, let alone captured --
        while letting one shape list cover both algorithms.
    """
    rng = np.random.RandomState(42)

    def l2_normed(shape):
        arr = rng.randn(*shape).astype(np.float32)
        return arr / np.sqrt((arr * arr).sum(axis=-1, keepdims=True) + 1e-6)

    # TOKEN-major (B, Hk, S, Dk): the natural projection layout, and the one the kernels
    # take. They fold the permute their matmuls need into the load DMA.
    query = l2_normed((batch, n_k_heads, seqlen, HEAD_DIM))
    key = l2_normed((batch, n_k_heads, seqlen, HEAD_DIM))
    if algorithm == ALGORITHM_NEUMANN and n_k_heads != n_v_heads:
        repeats = n_v_heads // n_k_heads
        query = np.repeat(query, repeats, axis=1)
        key = np.repeat(key, repeats, axis=1)

    inputs = {
        "query": dt.static_cast(query, dtype),
        "key": dt.static_cast(key, dtype),
        "value": dt.static_cast(rng.randn(batch, n_v_heads, seqlen, HEAD_DIM).astype(np.float32) * 0.5, dtype),
        "g_log": -np.abs(rng.randn(batch, n_v_heads, seqlen).astype(np.float32)) * 0.05,
        "beta": 1.0 / (1.0 + np.exp(-rng.randn(batch, n_v_heads, seqlen).astype(np.float32))),
    }
    if algorithm != ALGORITHM_NEUMANN:
        # Required by schur, and omitted for neumann (which builds its constants on device).
        inputs["mask_pack"] = build_mask_pack()
        # Supplied, so the aliased mutable-in/out path is what gets measured.
        inputs[ALIASED_INPUT] = np.zeros((batch, n_v_heads, HEAD_DIM, HEAD_DIM), dtype=np.float32)
    return inputs


def query_scale():
    """The 1/sqrt(Dk) attention scale the kernels apply on device."""
    return float(1.0 / np.sqrt(np.float32(HEAD_DIM)))


def max_rel_err(got, want):
    """Max absolute deviation normalized by the reference's peak magnitude.

    Args:
        got (np.ndarray): Device output.
        want (np.ndarray): CPU reference output.

    Returns:
        float: max|got - want| / max|want|.

    Notes:
        Normalizing by the peak rather than per element is deliberate: `out` contains
        near-zero entries whose per-element relative error is meaningless but whose
        absolute error is irrelevant to the consumer of the tensor.
    """
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    peak = np.abs(want).max()
    return float(np.abs(got - want).max() / (peak if peak else 1.0))


def config_tag(algorithm, batch, n_k_heads, n_v_heads, seqlen, dtype):
    """Stable per-configuration name for the artifacts directory and result JSON."""
    return f"{algorithm}_b{batch}_kh{n_k_heads}_vh{n_v_heads}_s{seqlen}_{dtype}"


def run_phase(kernel_inputs, algorithm, artifacts_dir, lnc=1):
    """Compile to a NEFF, execute on device, and score against the CPU reference.

    Args:
        kernel_inputs (dict): From `make_inputs`.
        algorithm (str): "schur" or "neumann".
        artifacts_dir (str): Where the compiler writes kernel.neff.

    Returns:
        dict: {"out_rel_err": float, "state_rel_err": float}.

    Notes:
        When a state is supplied the kernel mutates it in place, so the reference is run on
        a COPY of the inputs -- reading them back after the device call would compare the
        result against itself.
    """
    import torch
    from nkilib_src.nkilib.experimental.gdn.gdn_cte_torch import gdn_cte_torch_ref

    scale = query_scale()
    names = input_names(algorithm)
    # Only the five required tensors are positional. `mask_pack` and `recurrent_state` sit
    # after the trace-time knobs in the signature, so passing either positionally would land
    # on `scale`; both are keywords here, and absent entirely on the neumann path.
    positional = ["query", "key", "value", "g_log", "beta"]
    optional = [name for name in names if name not in positional]
    as_tensor = lambda name: torch.from_numpy(np.asarray(kernel_inputs[name], dtype=np.float32).copy())  # noqa: E731

    # SNAPSHOT THE REFERENCE INPUTS BEFORE THE KERNEL RUNS. `recurrent_state` is an aliased
    # mutable in/out, so the kernel overwrites it in place; copying it afterwards would hand
    # the reference the FINAL state as its initial state. That failure is invisible in bf16 --
    # it sits under the operand-rounding floor -- and shows up as a ~10x error in fp32.
    ref_args = [as_tensor(name) for name in positional]
    ref_mask_pack = as_tensor("mask_pack") if "mask_pack" in optional else None
    ref_state_in = as_tensor(ALIASED_INPUT) if ALIASED_INPUT in optional else None

    # `kernel[lnc]` is the SPMD launch: `gdn_cte_schur` splits value heads across the grid,
    # so lnc=2 needs NEURON_LOGICAL_NC_CONFIG=2 in the environment to match.
    # THE TRACER FRONTEND, pinned: it is what `--nki-compilation-mode` defaults to in the pytest
    # plugin, so this script measures the build the test suite validates.
    kernel = CompileKernel(func=gdn_cte.func, artifacts_dir=artifacts_dir, _frontend_cls=TracerFrontend)
    if lnc > 1:
        kernel = kernel[lnc]
    kwargs = {"scale": scale, "algorithm": algorithm}
    kwargs.update({name: kernel_inputs[name] for name in optional})
    out, state_out = kernel(*[kernel_inputs[name] for name in positional], **kwargs)
    ref_out, ref_state = gdn_cte_torch_ref(
        ref_args[0],
        ref_args[1],
        ref_args[2],
        ref_args[3],
        ref_args[4],
        scale=scale,
        algorithm=algorithm,
        mask_pack=ref_mask_pack,
        recurrent_state=ref_state_in,
    )
    return {
        "out_rel_err": max_rel_err(out, ref_out.numpy()),
        "state_rel_err": max_rel_err(state_out, ref_state.numpy()),
    }


def profile_phase(kernel_inputs, artifacts_dir, algorithm):
    """Capture a trace of the compiled NEFF with real inputs and read its metrics.

    Args:
        kernel_inputs (dict): From `make_inputs`. Written out as ifmap .npy files.
        artifacts_dir (str): Directory holding kernel.neff from the run phase.
        algorithm (str): Which algorithm was compiled -- it decides the input list.

    Returns:
        dict: total_time_us, total_active_time_us, mfu_percent, mbu_percent, plus
            per-engine active times.

    Notes:
        Three executions are captured and only the third is traced, so the numbers are
        steady state rather than a cold first run.
    """
    neff = os.path.join(artifacts_dir, "kernel.neff")
    if not os.path.exists(neff):
        raise FileNotFoundError(f"{neff} missing -- run --phase run first")

    ifmap = []
    for name in input_names(algorithm):
        array = np.ascontiguousarray(kernel_inputs[name])
        if array.dtype not in (np.float32, np.float16):
            array = array.view(np.float16)  # see the module docstring
        path = os.path.join(artifacts_dir, f"in_{name}.npy")
        np.save(path, array)
        # An input the kernel ALIASES to an output is named "<name>.must_alias_input" in
        # the NEFF, even though the MLIR still calls it `<name>`. Passing the plain name
        # fails with "unable to find input tensor alias ... for output <name>".
        neff_name = f"{name}{_ALIAS_SUFFIX}" if name == ALIASED_INPUT else name
        ifmap += [neff_name, path]

    # capture writes <session stem>_exec_<n>.ntff; clear stale traces so the newest is
    # unambiguous.
    for stale in glob.glob(os.path.join(artifacts_dir, "profile*.ntff")):
        os.remove(stale)
    subprocess.run(
        [
            "neuron-explorer",
            "capture",
            "-n",
            neff,
            "-s",
            os.path.join(artifacts_dir, "profile.ntff"),
            "--num-exec=3",
            "--profile-nth-exec=3",
            *ifmap,
        ],
        check=True,
        cwd=artifacts_dir,
    )
    trace = sorted(glob.glob(os.path.join(artifacts_dir, "profile*.ntff")))[-1]
    view = subprocess.run(
        ["neuron-explorer", "view", "-n", neff, "-s", trace, "--output-format", "summary-json"],
        check=True,
        capture_output=True,
        text=True,
        cwd=artifacts_dir,
    )
    # The summary is keyed by one opaque execution id; the last stdout line is the JSON.
    summary = json.loads(view.stdout.strip().split("\n")[-1])
    metrics = next(iter(summary.values()))
    result = {
        "total_time_us": metrics["total_time"] * 1e6,
        "total_active_time_us": metrics["total_active_time"] * 1e6,
        "mfu_percent": metrics["mfu_estimated_percent"] * 100,
        "mbu_percent": metrics["mbu_estimated_percent"] * 100,
        "dma_active_us": metrics["dma_active_time"] * 1e6,
    }
    for engine in ("tensor", "vector", "scalar", "gpsimd"):
        result[f"{engine}_active_us"] = metrics[f"{engine}_engine_active_time"] * 1e6
    return result


def emit_table(results):
    """Print the markdown comparison table the README carries.

    One row per configuration with both algorithms side by side: `total_time`, MFU and MBU
    for each, plus the neumann/schur time ratio. Accuracy is checked in the `run` phase
    (which prints the per-configuration error) and is not a column here.

    Args:
        results (list): Result dicts, one per (shape, dtype, algorithm).
    """

    def cell(entry, name, fmt, suffix=""):
        """One table cell, or "n/a" if that phase was not run for this configuration."""
        value = None if entry is None else entry.get(name)
        return format(value, fmt) + suffix if value is not None else "n/a"

    by_config = {}
    for entry in results:
        key = (entry["batch"], entry["n_k_heads"], entry["n_v_heads"], entry["seqlen"], entry["dtype"])
        by_config.setdefault(key, {})[entry["algorithm"]] = entry

    print("| B | Hk | Hv | S | dtype | schur | neumann | speedup | schur MFU | neumann MFU | schur MBU | neumann MBU |")
    print("|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for key in sorted(by_config, key=lambda k: (k[3], k[0], k[2], k[4])):
        schur = by_config[key].get(ALGORITHM_SCHUR)
        neumann = by_config[key].get(ALGORITHM_NEUMANN)
        schur_us = None if schur is None else schur.get("total_time_us")
        neumann_us = None if neumann is None else neumann.get("total_time_us")
        speedup = f"{neumann_us / schur_us:.2f}x" if schur_us and neumann_us else "n/a"
        print(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {key[4]} | "
            f"{cell(schur, 'total_time_us', '.1f', ' us')} | "
            f"{cell(neumann, 'total_time_us', '.1f', ' us')} | {speedup} | "
            f"{cell(schur, 'mfu_percent', '.2f', '%')} | {cell(neumann, 'mfu_percent', '.2f', '%')} | "
            f"{cell(schur, 'mbu_percent', '.2f', '%')} | {cell(neumann, 'mbu_percent', '.2f', '%')} |"
        )


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument(
        "--shapes",
        default=DEFAULT_SHAPES,
        help='Space-separated "batch,n_k_heads,n_v_heads,seqlen" tuples.',
    )
    parser.add_argument(
        "--algorithms",
        default=",".join(ALGORITHMS),
        help="Comma-separated subset of the gdn_cte algorithms.",
    )
    parser.add_argument(
        "--dtypes",
        default="bfloat16",
        help="Comma-separated q/k/v dtypes, i.e. the kernels' compute dtype.",
    )
    parser.add_argument("--phase", default="both", choices=["run", "profile", "both", "table"])
    parser.add_argument(
        "--lnc", type=int, default=1, help="SPMD grid size (LNC degree). Set NEURON_LOGICAL_NC_CONFIG to match."
    )
    parser.add_argument(
        "--artifacts",
        default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "gdn_cte_profile"),
        help="Directory for NEFFs, traces and result JSONs.",
    )
    args = parser.parse_args()

    shapes = [tuple(int(part) for part in shape.split(",")) for shape in args.shapes.split()]
    algorithms = args.algorithms.split(",")
    dtypes = args.dtypes.split(",")
    os.makedirs(args.artifacts, exist_ok=True)

    results = []
    for shape in shapes:
        for dtype in dtypes:
            for algorithm in algorithms:
                batch, n_k_heads, n_v_heads, seqlen = shape
                tag = config_tag(algorithm, batch, n_k_heads, n_v_heads, seqlen, dtype)
                artifacts_dir = os.path.join(args.artifacts, tag)
                os.makedirs(artifacts_dir, exist_ok=True)
                result_path = os.path.join(args.artifacts, f"result_{tag}.json")
                # Results accumulate across phases (and across separate invocations), so a
                # `profile` pass keeps the accuracy numbers a previous `run` pass measured.
                result = {}
                if os.path.exists(result_path):
                    with open(result_path) as handle:
                        result = json.load(handle)
                result.update(
                    {
                        "tag": tag,
                        "algorithm": algorithm,
                        "batch": batch,
                        "n_k_heads": n_k_heads,
                        "n_v_heads": n_v_heads,
                        "seqlen": seqlen,
                        "dtype": dtype,
                        "lnc": args.lnc,
                    }
                )

                if args.phase in ("run", "both", "profile"):
                    kernel_inputs = make_inputs(batch, n_k_heads, n_v_heads, seqlen, dtype, algorithm)
                if args.phase in ("run", "both"):
                    print(f"[{tag}] compiling and running on device", flush=True)
                    result.update(run_phase(kernel_inputs, algorithm, artifacts_dir, lnc=args.lnc))
                    print(
                        f"[{tag}] out_rel_err {result['out_rel_err']:.3e} state_rel_err {result['state_rel_err']:.3e}",
                        flush=True,
                    )
                if args.phase in ("profile", "both"):
                    print(f"[{tag}] capturing a profile", flush=True)
                    result.update(profile_phase(kernel_inputs, artifacts_dir, algorithm))
                    print(
                        f"[{tag}] total_time {result['total_time_us']:.1f} us "
                        f"active {result['total_active_time_us']:.1f} us "
                        f"mfu {result['mfu_percent']:.2f}% mbu {result['mbu_percent']:.2f}%",
                        flush=True,
                    )

                if args.phase != "table":
                    with open(result_path, "w") as handle:
                        json.dump(result, handle, indent=1, sort_keys=True)
                results.append(result)

    emit_table(results)


if __name__ == "__main__":
    main()
