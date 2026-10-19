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

"""The pinned DMA-order schedule of the gpt-oss-120b c128 giga-kernel (GK) decode test.

The GK decode kernel is compiled with a DMA-order constraint file
(``--dma-order-constraint-file``). The compiler identifies each constrained op by its BIR
instruction name, which ends in the kernel-call scope ``_inst__I-<K>-0``; ``K`` is the index
of the kernel call in the OUTER graph, so it differs between the vllm-neuron serve graph and
this test harness's graph. The compiler SKIPS a constraint whose name it cannot resolve, and
reports neither an error nor a mismatch count, so a wrong ``K`` (or a wrong op name) compiles
clean with less of the schedule applied. The test therefore rewrites the scope
(`materialize_schedule`) and checks the compile log afterwards (`assert_schedule_bound`).

The order itself is owned by vllm-neuron
(``test/vllm_neuron/model/gpt_oss/mxfp4/gk/gpt_oss_120b_gk_decode_i37srvfe_h38_qkvh1sel_gate2_dma_order.json``),
derived against the SERVE graph. This test's graph is a subset of that graph, so the file is
migrated here by `derive_test_schedule` -- the inverse of vllm-neuron's
``derive_serve_schedule.py`` -- and the result is checked in beside this module as
`GK_SCHEDULE`. Re-derive, never hand-edit::

    python -m test.integration.nkilib.experimental.gpt_oss.gk_dma_schedule \\
        --serve ../vllm-neuron/test/vllm_neuron/model/gpt_oss/mxfp4/gk/\\
gpt_oss_120b_gk_decode_i37srvfe_h38_qkvh1sel_gate2_dma_order.json

THREE transforms, each the reverse of one serve-side transform, each checked by an assertion:

1. **Collective ordinals shift by -1.** The serve graph fuses the token embedding into the
   kernel, which adds ONE ReduceScatter at ordinal 0; ``AG-<n>``/``RS-<n>`` share one numbering
   sequence, so every collective of the serve order is one higher than in this harness's graph
   (``VLLM_NEURON_MEGA_FUSE_EMBED`` is off by default here). A wrong shift does NOT fail the
   bind check: it binds names that EXIST and orders the WRONG collectives. The i37 order the
   previous revision of this test ran on was derived on this graph, so it is the reference the
   shifted sequence is checked against.

2. **Renames 36 kv-scale loads.** The serve runner passes the V quant scale as a ``(1, 1)``
   scalar, which the kernel broadcasts (``_kv_quant_scale_bcast_load_v``); this harness passes
   a replicated ``(128, 1)`` tile, which the kernel copies (``_kv_quant_scale_load_v``).

3. **Keeps the 7 device-mask prologue ops.** Both graphs build the attention masks on device
   by default, so the serve file's mask DMAs bind here as they are. (The i37 order lacked them
   because it was derived on the ``VLLM_NEURON_MEGA_HOST_MASK=1`` arm, which is why
   ``derive_serve_schedule`` had to put them back.) `HOST_MASK` is therefore NOT compatible with
   this schedule: the 7 ops would float free, and the two extra kernel inputs would move ``K``.

The kernel flags the order was tuned with (``ROUTERW_DMA_ENGINE=sync``, ``H0_RESID1``,
``QMX_BATCH_XPOSE``, ``GU_RS_FLOWGATE``, ``MOE_OUT_FP8``, ``GU_IOUTER=2``,
``GU_PREFETCH_HSPLIT``) are `SCHEDULE_KERNEL_FLAGS`; the test seeds them before the kernel
modules are imported, exactly as ``vllm_neuron/__init__.py`` does when a constraint file is set.
The runtime environment a served decode runs with is `SERVED_RUNTIME_ENV`; the test seeds it
into the worker environment, so the in-test profile is the deployment's runtime.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import tempfile
from collections import Counter
from collections.abc import MutableMapping
from pathlib import Path

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent

# The order as this harness's graph needs it: the serve order migrated by `derive_test_schedule`.
# 3299 ops, ``constraint_deps_mode: none``, ``sbuf_alloc_limit: 0.75``, no ``allocate: true``.
GK_SCHEDULE = str(_HERE / "gpt_oss_120b_gk_decode_h38_qkvh1sel_gate2_test_dma_order.json")

# The order this test ran on before: nkilib's own i37 publication (2938 ops; derived on THIS
# graph, with host masks). It is the collective-ordinal reference for `derive_test_schedule`.
# steering_private is not synced to the test hosts, so nothing at test time reads it.
I37_SCHEDULE = str(
    _HERE.parents[4] / "steering_private" / "gptoss_dma_json_rev15_nki07" / "json" / "gk_hostmask_order_i37.json"
)

# The kernel flags the order was derived with. The kernel modules read them at IMPORT time, so
# they must be in the environment before the giga-kernel package is imported. Every value here
# is also what vllm-neuron seeds for a served GK decode with a constraint file.
SCHEDULE_KERNEL_FLAGS = {
    "VLLM_NEURON_MEGA_ROUTERW_DMA_ENGINE": "sync",
    "VLLM_NEURON_MEGA_H0_RESID1": "1",
    "VLLM_NEURON_MEGA_QMX_BATCH_XPOSE": "1",
    "VLLM_NEURON_MEGA_GU_RS_FLOWGATE": "1",
    "VLLM_NEURON_MEGA_MOE_OUT_FP8": "1",
    "VLLM_NEURON_MEGA_GU_IOUTER": "2",
    "VLLM_NEURON_MEGA_GU_PREFETCH_HSPLIT": "1",
}

# The runtime environment of a served GK decode, replayed on the test's workers so the device
# profile measures the deployment's runtime. Both values are vllm-neuron's: neuron_worker.py sets
# the collective DMA QoS level for a GK decode server, and its GK tests set DGE_SKIP_NOTIF
# (packed-FP8 KV writes flood benign skip notifications). Update them together with vllm-neuron.
# DGE notifications are pinned off: the harness enables them for instruction-level DGE
# visibility in the profile, which a deployment does not run with (~6 us more Full-layer
# attention per layer).
SERVED_RUNTIME_ENV = {
    "NEURON_RT_DBG_CC_DMA_QOS_SERVICE_LEVEL": "3",
    "NEURON_RT_DBG_VECTOR_DGE_SKIP_NOTIF": "1",
    "NEURON_RT_ENABLE_DGE_NOTIFICATIONS": "0",
}

# ``MEGA_DMA_ORDER_SCHEDULE`` selects the order: unset/"default" (GK_SCHEDULE), "none" (no
# constraint file at all), or a path. ``MEGA_DMA_ORDER_CONSTRAINTS_FILE`` (a path) is the
# previous name and still works, with the same meaning as a path here.
SCHEDULE_ENV = "MEGA_DMA_ORDER_SCHEDULE"
LEGACY_SCHEDULE_FILE_ENV = "MEGA_DMA_ORDER_CONSTRAINTS_FILE"
_SCHEDULE_ALIASES = {"default": GK_SCHEDULE}

# The transforms, as constants so the assertions and the docstring agree.
FUSED_EMBED_ORDINAL_SHIFT = -1
KV_SCALE_RENAME = ("_kv_quant_scale_bcast_load_v", "_kv_quant_scale_load_v")
EXPECTED_KV_SCALE_RENAMES = 36  # one per layer
DEVICE_MASK_OPS = (
    "U-pro_mask_swap_pos_ids_load_sp10240",
    "U-pro_megamask_swa_pos_ids_load",
    "U-pro_megamask_swa_start_pos_load",
    "U-pro_active_mask_block_kv_swap_0_bo0_sp0",
    "U-pro_mask_swap_banded_store_sp10240_f0",
    "U-megamask_swa_active_mask_block_kv_0_bo0_sp0",
    "U-pro_megamask_swa_mask_out_store_sp0_bo0",
)

# The floor only separates a bound schedule from a skipped one; the full order binds ~6500
# (a constraint may match several instructions).
MIN_BOUND_CONSTRAINTS = 2000

_COLLECTIVE_RE = re.compile(r"^(AG|RS)-(\d+)(_inst__.*)?$")
_SCOPE_SUFFIX_RE = re.compile(r"_inst__[A-Za-z0-9-]+$")
_SCOPE_RE = re.compile(r"inst__I-(\d+)-0")
# The constraint name at the end of an OrderConstraintsPass "found instruction matched" line.
_MATCHED_NAME_RE = re.compile(r"((?:U-|AG-|RS-)[A-Za-z0-9_.-]+_inst__I-\d+-\d+)\s*$")


# ------------------------------------------------------------------------------- selection
def schedule_source() -> str | None:
    """Return the schedule file the test should apply, or None for no constraint file."""
    want = os.environ.get(SCHEDULE_ENV, "").strip()
    legacy = os.environ.get(LEGACY_SCHEDULE_FILE_ENV, "").strip()
    if want and legacy:
        raise ValueError(f"Set only one of {SCHEDULE_ENV} and {LEGACY_SCHEDULE_FILE_ENV}")
    want = want or legacy or "default"
    if want.lower() == "none":
        return None
    path = _SCHEDULE_ALIASES.get(want, want)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{SCHEDULE_ENV}={want!r}: no schedule file at {path}")
    return os.path.abspath(path)


def seed_schedule_kernel_flags(env: MutableMapping[str, str] | None = None) -> dict[str, str]:
    """Put `SCHEDULE_KERNEL_FLAGS` into ``env`` (default ``os.environ``) where unset; return what was added."""
    target: MutableMapping[str, str] = os.environ if env is None else env
    added = {}
    for name, value in SCHEDULE_KERNEL_FLAGS.items():
        if name not in target:
            target[name] = value
            added[name] = value
    return added


def seed_served_runtime_env(env: MutableMapping[str, str]) -> dict[str, str]:
    """Put `SERVED_RUNTIME_ENV` into the worker env ``env`` where unset; return what was added."""
    added = {}
    for name, value in SERVED_RUNTIME_ENV.items():
        if name not in env:
            env[name] = value
            added[name] = value
    return added


def host_mask_requested() -> bool:
    return os.environ.get("VLLM_NEURON_MEGA_HOST_MASK", "0") not in ("", "0", "false", "False")


# ------------------------------------------------------------------------------- names
def op_base(instr_name: str) -> str:
    """Return an op name without its ``_inst__I-<K>-0`` scope suffix."""
    return _SCOPE_SUFFIX_RE.sub("", instr_name)


def shift_collectives(instr_name: str, shift: int) -> str:
    """Return ``instr_name`` with an ``AG``/``RS`` ordinal moved by ``shift``. Others pass through."""
    m = _COLLECTIVE_RE.match(instr_name)
    if not m:
        return instr_name
    kind, ordinal, suffix = m.group(1), int(m.group(2)), m.group(3) or ""
    new = ordinal + shift
    assert new >= 0, f"{instr_name}: ordinal {ordinal}{shift:+d} is negative"
    return f"{kind}-{new}{suffix}"


def collective_sequence(ops: list[dict]) -> list[tuple[str, int]]:
    """Return the ``(kind, ordinal)`` pairs of the collectives, in file order."""
    out = []
    for o in ops:
        m = _COLLECTIVE_RE.match(o["instr_name"])
        if m:
            out.append((m.group(1), int(m.group(2))))
    return out


def schedule_scope(ops: list[dict]) -> str:
    """Return the single ``inst__I-<K>-0`` scope the ops carry."""
    scopes = Counter(m.group(1) for o in ops for m in _SCOPE_RE.finditer(o["instr_name"]))
    assert len(scopes) == 1, f"the schedule holds {len(scopes)} scopes ({dict(scopes)}); exactly one is needed"
    return next(iter(scopes))


# ------------------------------------------------------------------------------- derivation
def derive_test_schedule(serve: str, reference: str, out_path: str) -> dict:
    """Write this harness's schedule from the serve-graph schedule ``serve``; return the figures.

    ``reference`` is an order derived on THIS graph (the i37 publication); only its collective
    sequence is used, to prove the ordinal shift.
    """
    data = json.load(open(serve))
    ops = data["constraint_ops"]
    n_serve = len(ops)

    n_alloc = sum(1 for o in ops if o.get("allocate") is True)
    assert n_alloc == 0, (
        f"{serve} carries {n_alloc} ops with allocate:true; every i37-line order is built without "
        f"them. Re-check which file vllm-neuron ships."
    )

    renamed = 0
    for o in ops:
        o["instr_name"] = shift_collectives(o["instr_name"], FUSED_EMBED_ORDINAL_SHIFT)
        if KV_SCALE_RENAME[0] in o["instr_name"]:
            o["instr_name"] = o["instr_name"].replace(*KV_SCALE_RENAME)
            renamed += 1
    assert renamed == EXPECTED_KV_SCALE_RENAMES, (
        f"Renamed {renamed} kv-scale loads, expected {EXPECTED_KV_SCALE_RENAMES} (one per layer). "
        f"Check the name attention_block_tkg.quantize_to_fp8 gives the (pmax, 1) scale load now."
    )

    ref_ops = json.load(open(reference))["constraint_ops"]
    got, want = sorted(collective_sequence(ops)), sorted(collective_sequence(ref_ops))
    assert got == want, (
        f"The shifted collectives do not match {os.path.basename(reference)}.\n"
        f"  migrated : {len(got)} collectives, first 6 {got[:6]}\n"
        f"  reference: {len(want)} collectives, first 6 {want[:6]}\n"
        f"  A wrong shift still binds and orders the WRONG collectives. Check whether the serve "
        f"graph still adds exactly one collective at ordinal 0 for the fused embedding."
    )

    names = [op_base(o["instr_name"]) for o in ops]
    missing = sorted(set(DEVICE_MASK_OPS) - set(names))
    assert not missing, (
        f"{serve} does not name the device-mask prologue ops {missing}. This harness builds the "
        f"masks on device, so those DMAs must be in the order."
    )
    dups = [n for n, c in Counter(names).items() if c > 1]
    assert not dups, f"duplicate op names after migration: {dups[:5]}"

    scope = schedule_scope(ops)
    data["constraint_ops"] = ops
    with open(out_path, "w") as fh:
        json.dump(data, fh)
    return {
        "serve": serve,
        "serve_ops": n_serve,
        "written": out_path,
        "written_ops": len(ops),
        "collectives_checked": len(got),
        "kv_scale_renames": renamed,
        "device_mask_ops_kept": len(DEVICE_MASK_OPS),
        "ordinal_shift": FUSED_EMBED_ORDINAL_SHIFT,
        "scope": scope,
        "knobs": {k: v for k, v in data.items() if k != "constraint_ops"},
    }


# ------------------------------------------------------------------------------- apply
def materialize_schedule(k: int, *, source: str | None = None, out_dir: str | None = None) -> str:
    """Write a copy of the schedule whose scope is ``K`` and return its path."""
    source = source or schedule_source()
    assert source is not None, "no schedule selected"
    data = json.load(open(source))
    ops = data["constraint_ops"]
    k_src = schedule_scope(ops)
    old, new = f"inst__I-{k_src}-0", f"inst__I-{k}-0"
    for o in ops:
        o["instr_name"] = o["instr_name"].replace(old, new)
        if "dbg" in o:
            o["dbg"] = o["dbg"].replace(old, new)
    out_dir = out_dir or tempfile.mkdtemp(prefix="gk_sched_")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{Path(source).stem}_K{k}.json")
    with open(out, "w") as fh:
        json.dump(data, fh)
    logger.info(
        "[GK-SCHED] schedule written: %s (scope I-%s -> I-%s on %d ops, deps_mode=%s)",
        out,
        k_src,
        k,
        len(ops),
        data.get("constraint_deps_mode"),
    )
    return out


def compiler_args_for_schedule(schedule_path: str) -> list[str]:
    """Return the neuronx-cc arguments that apply ``schedule_path``.

    The file is passed by PATH through ``--internal-backend-options``: the inline
    ``--internal-dma-order-constraints`` flag carries the JSON content as one argv element and a
    realistic schedule (500 KB) exceeds Linux MAX_ARG_STRLEN. SB allocation is paired with the
    order, as on the derive path.
    """
    return [
        f"--internal-backend-options=--dma-order-constraint-file={os.path.abspath(schedule_path)}",
        "--internal-bir-enable-sb-allocation",
    ]


# ------------------------------------------------------------------------------- compile-log check
def compile_log_path(artifacts_root: str) -> str:
    """Return the neuronx-cc log of a test's artifact directory."""
    return os.path.join(artifacts_root, "artifacts", "log-neuron-cc.txt")


def inspect_compile_log(log_path: str, schedule_path: str | None = None) -> dict:
    """Return what the constraint pass did, read from ``log_path``.

    With ``schedule_path``, also list the applied op names the pass never matched: the compiler
    skips such a name without a word, and a bind count alone cannot tell a skipped op from one
    that matched several instructions.
    """
    bound = 0
    cycles = 0
    removed = 0
    scopes: Counter[str] = Counter()
    matched: set[str] = set()
    with open(log_path, errors="ignore") as fh:
        for line in fh:
            if "found instruction matched to constraint" in line:
                bound += 1
                for m in _MATCHED_NAME_RE.finditer(line):
                    matched.add(m.group(1))
            if "order_constraints: dependency cycle" in line:
                cycles += 1
            m = re.search(r"Removed ordered from (\d+) insts", line)
            if m:
                removed = max(removed, int(m.group(1)))
            for m in _SCOPE_RE.finditer(line):
                scopes[m.group(1)] += 1
    graph_k = int(scopes.most_common(1)[0][0]) if scopes else None
    unbound: list[str] = []
    if schedule_path is not None:
        names = [o["instr_name"] for o in json.load(open(schedule_path))["constraint_ops"]]
        unbound = [n for n in names if n not in matched]
    return {
        "bound": bound,
        "cycles": cycles,
        "removed_edges": removed,
        "graph_k": graph_k,
        "unbound": unbound,
        "log": log_path,
    }


def assert_schedule_bound(
    log_path: str, requested_k: int, *, schedule_path: str | None = None, min_bound: int = MIN_BOUND_CONSTRAINTS
) -> dict:
    """Fail unless every op of the schedule bound to the compiled graph with no dependency cycle.

    A stale ``K`` makes the compiler skip every constraint without an error, so the compile log
    is the only witness that the NEFF carries the order.
    """
    report = inspect_compile_log(log_path, schedule_path)
    bound, graph_k = report["bound"], report["graph_k"]
    if bound == 0:
        hint = (
            f"The graph's GK kernel scope is K={graph_k}; re-run with MEGA_DMA_ORDER_K={graph_k}."
            if graph_k is not None and graph_k != requested_k
            else "Check that the compile reached the constraint pass and that the log is this run's."
        )
        raise AssertionError(
            f"[GK-SCHED] The DMA-order schedule bound 0 constraints, so this NEFF carries NO "
            f"schedule.\n  requested K : {requested_k}\n  graph K     : {graph_k}\n"
            f"  log         : {log_path}\n  fix         : {hint}"
        )
    assert bound >= min_bound, (
        f"[GK-SCHED] Only {bound} constraints bound, below the floor of {min_bound}: the schedule is "
        f"partly resolved, so this NEFF is not the scheduled one.\n  requested K : {requested_k}\n"
        f"  graph K     : {graph_k}\n  log         : {log_path}"
    )
    assert report["cycles"] == 0, (
        f"[GK-SCHED] {report['cycles']} 'order_constraints: dependency cycle' lines: the order does "
        f"not fit this graph. log: {log_path}"
    )
    unbound = report["unbound"]
    assert not unbound, (
        f"[GK-SCHED] {len(unbound)} of the schedule's ops matched no instruction, so the NEFF carries "
        f"only part of the order. The kernel renamed or dropped these DMAs, or the test runs an arm "
        f"the order was not derived for (host masks, bf16 KV, shared layer inputs). First 10: "
        f"{[op_base(n) for n in unbound[:10]]}\n  log: {log_path}"
    )
    logger.info(
        "[GK-SCHED] schedule bound: %d constraints, K=%d, %d edges removed, %d cycles, 0 unbound (log: %s)",
        bound,
        requested_k,
        report["removed_edges"],
        report["cycles"],
        log_path,
    )
    return report


def main() -> None:
    """Re-derive `GK_SCHEDULE` from the vllm-neuron serve schedule. See the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serve", required=True, help="vllm-neuron's serve-graph order JSON")
    parser.add_argument("--reference", default=I37_SCHEDULE, help="an order derived on this graph (collective check)")
    parser.add_argument("--out", default=GK_SCHEDULE, help="where to write the migrated schedule")
    args = parser.parse_args()
    report = derive_test_schedule(args.serve, args.reference, args.out)
    for key, value in report.items():
        print(f"  {key:22s}: {value}")


if __name__ == "__main__":
    main()
