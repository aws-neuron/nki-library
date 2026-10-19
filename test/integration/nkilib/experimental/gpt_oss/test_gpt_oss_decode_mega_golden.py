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

# ty: ignore — vendored GPT-OSS test; golden/optional-arg idioms trip ty's type mapping
"""Multi-layer GPT-OSS decode "mega kernel" vs the HF golden, on hardware.

Stacks ``num_layers`` fused DP x TP attention + EP MoE decoder layers
(``gpt_oss_decode_mega_kernel``) and validates the whole stack directly against
the vendored HF-bit-validated whole-layer oracle ``gptoss_mxfp4_decode_layer``,
called ``num_layers`` times threading the hidden state (exactly
``GptOssModel.forward_decode``).

Identical to the real GPT-OSS-20B decode structure:
  * 32 experts, top-k 4  -> EP32, DP4 x TP8 (32 ranks, 1 expert/rank) by default.
  * GQA: 64 Q-heads, 8 KV-heads (num_kv_groups=8). TP is env-tunable via
    ``VLLM_NEURON_MEGA_TP`` (default 8); the per-rank shard is derived by dividing
    the fixed 64 Q / 8 KV counts by TP. At TP8, rank r owns Q-heads [8r,8r+8) AND
    KV head r (clean 1:1); at TP4 a rank owns 16 Q + 2 KV heads, at TP1 all 64 + 8.
    RANKS/EP are held fixed, so lowering TP raises DP (DP = RANKS // TP).
  * head_dim 64; hidden 2880 (kernel-padded to 3072, hidden_actual=2880);
    intermediate 2880 (padded to 3072); MXFP4 experts.
  * SWA-alternating: even layers slide (window 128), odd layers full attention.
  * Block (paged) KV cache, per layer.

Layer-chaining: each fused layer's output ``[B/(DP*TP), H]`` is the next layer's
input (same SP layout), so the kernel just loops the layer body threading ``x``;
per-layer weights + KV caches are passed as lists. The golden ref mirrors this by
looping the whole-layer oracle, threading the running hidden state and using each
layer's own weights, KV cache, and SWA setting.

The per-layer attention head-split (full 64Q/8KV == sum of per-rank (8Q + own KV
head) partials, O-proj bias once) and the MXFP4->dense adapter are CPU-proven; the
single-layer variant of this exact composition is HW-validated in
test_gpt_oss_decode_layer_dp_tp_ep_golden.py. This test adds the multi-layer
chaining, the GQA-8 sharding, real 20B dims, and SWA alternation.

MXFP4 experts -> TRN3 only.

Set ``VLLM_NEURON_MEGA_FULL_120B=1`` for the 36-layer DP16 x TP8 x EP128
profiling configuration. Combine it with ``VLLM_NEURON_MEGA_SKIP_GOLDEN=1`` and
``VLLM_NEURON_MEGA_SHARE_LAYER_INPUTS=1`` to reuse one copy of invariant
weights and KV caches across all layers. Set
``VLLM_NEURON_MEGA_INPUT_CACHE_DIR`` to persist generated per-rank inputs as
raw files; subsequent runs memory-map and hard-link them instead of regenerating
or rewriting them. Change ``VLLM_NEURON_MEGA_INPUT_CACHE_TAG`` to force a miss.
Set ``VLLM_NEURON_MEGA_FP8_KV=1`` to use the packed OCP-E4M3 KV-cache path.
Set ``VLLM_NEURON_MEGA_TP=<n>`` (n | 64 and n | 8, i.e. 1/2/4/8) to change the TP
shard; the golden validates at any TP (the whole-layer oracle is TP-invariant), and
the multi-KV-head write path is exercised for TP<8.
Set ``VLLM_NEURON_MEGA_GIGA=<names>`` (comma-separated; default ``c128``) to select
which vendored giga-kernel variant(s) to exercise, e.g. ``c512`` or ``c128,c512``.
Set ``VLLM_NEURON_MEGA_X_FIRST_FULL=1`` to exercise the layer-0 AllGather-skip
path: the kernel is given the already-gathered ``x_first_full`` input and takes
its ``x_full is not None`` branch instead of the redundant dma_copy + AllGather.
The golden ref ignores it, so the run validates the skip path against the
AllGather path (they must match bit-for-bit).
The 36-layer c128 configuration compiles with the pinned DMA order of the served
GK decode (``gk_dma_schedule.GK_SCHEDULE``); ``MEGA_DMA_ORDER_SCHEDULE=none``
drops it, ``=<path>`` applies another order, and ``MEGA_DMA_ORDER_K`` overrides
the kernel-call scope the order is rewritten to. With a schedule applied the
workers run the served runtime environment (``gk_dma_schedule.SERVED_RUNTIME_ENV``),
overridable per key through ``MEGA_WORKER_ENV``. See ``gk_dma_schedule.py``.
"""

import inspect
import logging
import os
import sys

import nki.language as nl
import numpy as np
import pytest
import torch
from nki.collectives import ReplicaGroup

from test.integration.nkilib.core.moe_block.test_moe_block_tkg import generate_inputs
from test.integration.nkilib.experimental.transformer.test_attention_block_tkg import (
    generate_kernel_inputs,
)
from test.integration.nkilib.experimental.transformer.test_attention_block_tkg_utils import (
    AttnBlkTestConfig,
)
from test.utils.common_dataclasses import (
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    InferenceArgs,
    NKICompilationMode,
    Platforms,
    TraceMode,
)
from test.utils.mx_utils import maybe_pack64_gate_up_scale
from test.utils.persistent_input_cache import PersistentPerRankInputCache, build_input_cache_key
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_collective_framework import CollectiveUnitTestFramework

from . import gk_dma_schedule as _gk_sched
from .gptoss_mxfp4_golden import GptOssConfig, gptoss_mxfp4_decode_layer
from .gptoss_mxfp4_golden.kernels import attention_decode as _golden_attention_decode
from .mxfp4_dense_adapter import (
    packed_mx_to_dense_down,
    packed_mx_to_dense_gate_up,
    unpack_gate_up_bias,
)

# The GIGA kernel (gpt_oss_decode_mega_kernel) is maintained as one or more
# self-contained, vendored packages under src/.../c*_giga_kernel/ (generated by
# scripts/vendor_giga_kernel.py). Each is a fully isolated package (relative imports,
# no shared module names), so multiple variants can coexist in one process. The test
# selects which variant(s) to run via VLLM_NEURON_MEGA_GIGA (default: c128, matching
# mainline; e.g. "c512" or "c128,c512"). The kernel entry is loaded lazily per variant
# so a WIP/broken variant never breaks collection of the others; the mega decode
# harness + golden below are shared verbatim across all variants (identical entry
# signature -- the variants differ only in internal context tiling).
_GIGA_VARIANTS = {
    "c128": "nkilib_src.nkilib.models.gpt_oss.c128_giga_kernel.experimental.gpt_oss.gpt_oss_decode_layer_dp_tp_ep",
    "c512": "nkilib_src.nkilib.models.gpt_oss.c512_giga_kernel.experimental.gpt_oss.gpt_oss_decode_layer_dp_tp_ep",
}
# Comma-separated selection; unknown names fail loudly at parametrize time (below).
_GIGA_SEL = [v.strip() for v in os.environ.get("VLLM_NEURON_MEGA_GIGA", "c128").split(",") if v.strip()]


def _load_mega_kernel(variant: str):
    """Lazily import the ``gpt_oss_decode_mega_kernel`` entry for a giga-kernel variant."""
    import importlib

    return importlib.import_module(_GIGA_VARIANTS[variant]).gpt_oss_decode_mega_kernel


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "0") not in ("", "0", "false", "False")


# Set to skip the (expensive) golden reference — the fused multi-layer golden runs
# the whole-layer HF oracle once per rank (32 ranks, sequential) at real 20B dims,
# which dominates wall time (~40 min at 8 layers). When set, the kernel still
# compiles, runs on device, and profiles; only the golden generation + numerical
# validation are skipped (via a zero-returning stub ref, so the harness output
# descriptors stay correct). Use for perf/profiling runs, NOT correctness runs.
_SKIP_GOLDEN = _env_enabled("VLLM_NEURON_MEGA_SKIP_GOLDEN")
_FULL_120B_PROFILE = _env_enabled("VLLM_NEURON_MEGA_FULL_120B")
_SHARE_LAYER_INPUTS = _env_enabled("VLLM_NEURON_MEGA_SHARE_LAYER_INPUTS")
_INPUT_CACHE_DIR = os.environ.get("VLLM_NEURON_MEGA_INPUT_CACHE_DIR", "").strip()
_INPUT_CACHE_TAG = os.environ.get("VLLM_NEURON_MEGA_INPUT_CACHE_TAG", "default")
_REUSE_INPUT_ARTIFACTS = os.environ.get("VLLM_NEURON_MEGA_REUSE_INPUT_ARTIFACTS", "").strip()

# Runtime env vars forwarded to every worker process (the harness passes only
# InferenceArgs.env_vars through; ambient env does not reach the workers).
# Format: "NAME=VALUE;;NAME2=VALUE2".
# Optional: ship a locally-built libnrt.so with the job and point the 128 workers at it via
# LD_LIBRARY_PATH. This is PROCESS-SCOPED -- the harness uploads the artifacts dir to a
# PID-unique /tmp path and LD_LIBRARY_PATH is per-process env, so /opt/aws/neuron/lib and every
# other user's job on the shared host are untouched (no dpkg, no shared-state mutation).
# Used to test the 128-peer mesh all_to_all descriptor-uniformity patch.
_EXTRA_LIB_DIR = os.environ.get("MEGA_EXTRA_LIB_DIR", "")

# Entries are ";;"-separated, so a VALUE that itself needs ";" (e.g.
# NEURON_RT_CC_ALG_TYPES_PER_STREAM="0:ALG;1:ALG") cannot be written literally -- it gets
# truncated at the first ";;"-split. Write "@" in the value and it is translated to ";" here.
_WORKER_ENV = {
    k: v.replace("@", ";")
    for k, v in (kv.split("=", 1) for kv in os.environ.get("MEGA_WORKER_ENV", "").split(";;") if "=" in kv)
}

# MEGA_EXTRA_LIB_DIR only STAGES the .so into the artifacts dir; nothing pointed the workers at
# it, so a locally built libnrt.so was uploaded and then ignored. Prepend the remote artifacts
# dir (cwd of the capture command) to LD_LIBRARY_PATH so the staged lib actually loads.
if _EXTRA_LIB_DIR:
    # LD_LIBRARY_PATH alone is NOT enough: /opt/aws/neuron/bin/neuron-explorer is linked with
    # RUNPATH=/opt/aws/neuron/lib, and RUNPATH is searched BEFORE LD_LIBRARY_PATH for a binary's
    # direct DT_NEEDED entries -- so the stock libnrt.so.1 always won and every "patched runtime"
    # run in this workspace silently used the stock library. LD_PRELOAD takes precedence over
    # both, so force the staged copy in explicitly.
    _WORKER_ENV["LD_LIBRARY_PATH"] = "$PWD:/opt/aws/neuron/lib"
    _WORKER_ENV["LD_PRELOAD"] = "$PWD/libnrt.so.1"

# Packed FP8 KV uses the OCP E4M3 dtype so it can coexist with the OCP MXFP4
# expert weights in one NEFF. K is packed as two FP8 rows per storage row;
# V remains in the ordinary FP8 block layout.
_FP8_KV = _env_enabled("VLLM_NEURON_MEGA_FP8_KV")
_KV_QUANT_DTYPE = nl.float8_e4m3fn

# KV CACHE UPDATE (on by default; VLLM_NEURON_MEGA_KV_UPDATE=0 to disable): the kernel writes
# each decode token's K/V back into the paged cache in-place and returns the caches it was given
# -- (x, K_cache, V_cache) for the stacked interface, or (x, *K_cache_i, *V_cache_i) for the
# per-buffer one (VLLM_NEURON_MEGA_KV_INDIVIDUAL). The harness aliases those onto the matching
# input buffers (create_inputs uses *.must_alias_input keys) and the golden compares the
# post-update caches. Each layer must own a distinct cache (its own K_cache[li] slice, or its own
# buffer), so it is INCOMPATIBLE with SHARE_LAYER_INPUTS.
#
# PROFILING REPRESENTATIVENESS: a KV-update profile is only representative with weight sharing
# DISABLED (VLLM_NEURON_MEGA_SHARE_LAYER_INPUTS unset -> layer_input_count == num_layers). With
# sharing on, all layers reuse one weight/cache tensor, so the compiler skips per-layer weight
# re-load DMA that a real decode pays; that also makes the caches unstackable (see above). To
# A/B the KV-write cost against a read-only baseline, run BOTH with sharing off so the only
# difference is the write -- otherwise the DMA delta is dominated by the layer_input_count 1<->36
# weight-reload change, not the KV update.
_KV_UPDATE = os.environ.get("VLLM_NEURON_MEGA_KV_UPDATE", "1") not in (
    "",
    "0",
    "false",
    "False",
)

# FIRST-LAYER ALLGATHER SKIP (VLLM_NEURON_MEGA_X_FIRST_FULL=1): layer 0's decode input is
# TP-replicated, so the per-layer AllGather[TP] that rebuilds [T_attn, H] from this rank's
# SP shard is redundant for it. When set, create_inputs passes the kernel an explicit
# ``x_first_full`` = the already-gathered [T_attn, H] layer-0 input, and the kernel takes
# the ``x_full is not None`` branch for layer 0 (skips the dma_copy + AllGather).
#
# CORRECTNESS: ``x_first_full`` is built to equal exactly what the AllGather would
# reconstruct -- all TP ranks in a replica share one X_full, so gather(x_shard) == X_full.
# The golden ref ignores ``x_first_full`` (it regenerates X_full from the same seed), so a
# correct skip path must match the golden bit-for-bit; a broken branch (wrong reshape,
# stale/ignored input) diverges. Layer-0-only, so it composes with any num_layers / KV_UPDATE.
_X_FIRST_FULL = _env_enabled("VLLM_NEURON_MEGA_X_FIRST_FULL")

# SHARE LAYER WEIGHTS (VLLM_NEURON_MEGA_SHARE_LAYER_WEIGHTS=1): a HARNESS-ONLY speed knob for
# profiling runs. Instead of generating 36 independent weight sets, generate ONE per layer PARITY
# (sliding vs full) and repeat it, which is what dominates input-generation wall time at L=36.
# The kernel still sees leading dim == num_layers, so layer_input_count and every per-layer weight
# re-load DMA are UNCHANGED -- unlike SHARE_LAYER_INPUTS, which collapses the leading dim to 1 and
# therefore changes what is measured. Profiling only: the golden would still pass, but distinct
# per-layer weights are the stronger correctness test.
_SHARE_LAYER_WEIGHTS = _env_enabled("VLLM_NEURON_MEGA_SHARE_LAYER_WEIGHTS")

# PACK64 gate/up MX scale (kernel default ON; VLLM_NEURON_MEGA_SCALE_PACK64=0 to disable).
# The kernel reads the scale in the packed [.., 2, 64, H/512, I/4] HBM layout, so the harness
# must remap what it hands the kernel. Applied ONLY to the kernel's input dict -- the torch
# golden keeps the ORIGINAL unpacked scales, so kernel-vs-golden agreement IS the proof that
# the packed layout + 4-per-quadrant SBUF placement + partition-offset AP reproduce the
# unpacked math. Nothing extra had to be written to validate the pack.
# The env read itself lives in test/utils/mx_utils.py so it cannot drift from the kernel's.

# INDIVIDUAL KV BUFFERS (VLLM_NEURON_MEGA_KV_INDIVIDUAL=1): pass the per-buffer KV
# caches to the kernel as separate K_cache_i/V_cache_i tensors (the HMA interface)
# with the stacked K_cache/V_cache set to None, instead of one stacked tensor. This
# validates the kernel's backward-compatible individual-buffer interface: routing
# mirrors the stacked leading-dim logic by the count of buffers provided, so passing
# num_layers per-layer buffers reproduces the per-layer golden exactly.
#
# Composes with KV_UPDATE: the kernel writes each layer's decode K/V in place and returns
# the buffers it was given, so every per-buffer tensor is aliased onto its input and the
# golden validates each post-update buffer individually (see create_inputs/output_keys).
_KV_INDIVIDUAL = _env_enabled("VLLM_NEURON_MEGA_KV_INDIVIDUAL")


def _indiv_kv_keys(d):
    """The per-buffer KV keys present in ``d`` (``K_cache_<i>`` / ``V_cache_<i>``).

    Matches the kernel's per-buffer parameter names only -- ``K_cache``/``V_cache`` (no
    index) are the stacked form and are deliberately excluded.
    """
    return [
        k for k in list(d) if (k.startswith("K_cache_") or k.startswith("V_cache_")) and k.rsplit("_", 1)[-1].isdigit()
    ]


def _mega_output_keys(n_kv_buffers):
    """Validated kernel outputs: ``out``, plus the post-update KV caches under KV_UPDATE.

    Under KV_UPDATE the kernel returns the caches it was given, so the validated names follow
    the interface: the two stacked tensors, or one name per provided per-buffer tensor.
    """
    if not _KV_UPDATE:
        return ["out"]
    if _KV_INDIVIDUAL:
        return ["out"] + [f"K_cache_{j}" for j in range(n_kv_buffers)] + [f"V_cache_{j}" for j in range(n_kv_buffers)]
    return ["out", "K_cache", "V_cache"]


def _attn_seed(dp_replica, li):
    """Per-(replica, layer) attention-input seed.

    Distinct per layer normally. Under SHARE_LAYER_WEIGHTS collapse to one seed per layer
    PARITY (sliding vs full) so the memoized ``aki`` is reused across all like-parity
    layers -- 2 builds instead of 36. Shapes and stacking are unchanged; only the
    generated VALUES repeat, and the two SWA/full geometries are still distinct (they
    must be: different mask and block table).
    """
    if _SHARE_LAYER_WEIGHTS:
        return 1000 * dp_replica + (0 if _is_sliding(li) else 31)
    return 1000 * dp_replica + 31 * li


# FUSED EMBEDDING (VLLM_NEURON_MEGA_FUSE_EMBED=1): validate the kernel's in-NEFF
# embedding prologue. create_inputs builds a per-rank vocab-shard table +
# local token indices whose gather + TP-reduce reconstructs layer 0's X_full
# EXACTLY, then zeros x_shard so the fused prologue is the ONLY source of layer
# 0's input. Each token is owned by exactly one rank at a distinct local index;
# out-of-shard tokens get -1 (kernel oob_mode.skip -> 0 row), so the TP reduce
# sums one real row + zeros == X_full[t]. The golden ref still builds its hidden
# from the same X_full and ignores embed_weight/embed_ids, so a broken prologue
# (wrong rows, zeros, bad offsets) drops the cosine below the gate. Mutually
# exclusive with X_FIRST_FULL (both set layer 0's input).
_FUSE_EMBED = _env_enabled("VLLM_NEURON_MEGA_FUSE_EMBED")
# VLLM_NEURON_MEGA_FUSE_ROPE=1: validate the kernel's ON-CHIP RoPE. create_inputs
# replaces the external cos/sin with position-derived tables (cos(pos*inv_freq)*conc)
# and passes inv_freq + concentration, so the golden uses those exact tables while
# the kernel recomputes them on-chip from pos_ids + inv_freq (via _rope). A broken
# on-chip RoPE then diverges from the golden. inv_freq/conc here are test values;
# the kernel is agnostic to their specifics.
_FUSE_ROPE = _env_enabled("VLLM_NEURON_MEGA_FUSE_ROPE")
_FUSE_ROPE_CONC = 1.3  # test RoPE concentration (exercises the scale multiply)
# Diagnostic: with _FUSE_ROPE, still override cos/sin with the position-derived
# tables but do NOT pass inv_freq, so the kernel uses those (numpy-accurate) cos/sin
# as inputs (no on-chip RoPE). Comparing this to the full _FUSE_ROPE run isolates
# the on-chip _rope precision from the cos/sin VALUES.
_FUSE_ROPE_INPUT_ONLY = _env_enabled("VLLM_NEURON_MEGA_FUSE_ROPE_INPUT_ONLY")
# VLLM_NEURON_MEGA_FUSE_EMBED_HIDDEN=1 selects the HIDDEN-dim sharding (design
# option (b)): each rank holds [vocab, H/TP] (its H-slice of every vocab row) and
# looks up GLOBAL token ids, so there is NO cross-rank sum. One SBUF all_gather
# along the hidden dim rebuilds the full [T_attn, H]; the pre-attention all_reduce
# of the vocab-dim path (option (a)) is gone. Unset -> vocab-dim sharding.
_FUSE_EMBED_HIDDEN = _env_enabled("VLLM_NEURON_MEGA_FUSE_EMBED_HIDDEN")

# VLLM_NEURON_MEGA_HOST_MASK=1 builds the full + SWA attention masks on the HOST
# (via the byte-exact gen_mask_tkg_hbm torch reference) and passes them as kernel
# inputs, so the kernel LOADS them instead of running gen_mask_tkg_hbm on-chip.
# This makes the device profile representative of a real deployment, where the
# runner builds the mask on the host. Unset -> the kernel's on-chip hoist runs
# gen_mask_tkg_hbm (twice) in the NEFF.
_HOST_MASK = _env_enabled("VLLM_NEURON_MEGA_HOST_MASK")

# PINNED DMA ORDER. The 36-layer c128 configuration compiles with the DMA-order schedule the
# served GK decode runs on (gk_dma_schedule.GK_SCHEDULE, migrated from vllm-neuron's serve-graph
# order); MEGA_DMA_ORDER_SCHEDULE=none runs the compiler's own order, =<path> another file. The
# order binds by instruction name inside the kernel-call scope ``_inst__I-<K>-0`` and the compiler
# skips any name it cannot resolve WITHOUT an error, so the test rewrites the scope to
# _SCHEDULE_K (MEGA_DMA_ORDER_K to override) and reads the compile log afterwards.
#
# The order was derived with seven kernel flags on; the kernel modules read them at IMPORT time,
# so they are seeded here, before _load_mega_kernel, whenever a schedule is selected. vllm-neuron
# seeds the same flags in the same way when a served decode applies a constraint file, so every
# test in this module then runs the kernel configuration a deployment runs.
#
# The workers then also run the served runtime environment (gk_dma_schedule.SERVED_RUNTIME_ENV:
# collective DMA QoS level 3, DGE_SKIP_NOTIF, no DGE notifications), so the device profile
# measures the deployment's runtime instead of the harness default. MEGA_WORKER_ENV overrides any
# of the three, e.g. NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1 for instruction-level DGE in the profile.
_SCHEDULE_SOURCE = _gk_sched.schedule_source()
if _SCHEDULE_SOURCE is not None:
    _gk_sched.seed_schedule_kernel_flags()
    _gk_sched.seed_served_runtime_env(_WORKER_ENV)
# The kernel-call scope of this harness's 36-layer graph. Any change to the kernel's input list
# moves it; the bind check then names the right value.
_SCHEDULE_K = int(os.environ.get("MEGA_DMA_ORDER_K", "35"))


def _schedule_for(giga_variant: str, num_layers: int) -> str | None:
    """Return the schedule file to compile ``(giga_variant, num_layers)`` with, or None.

    The default order targets the 36-layer c128 graph only: its op names are per layer
    (``U-L35_...``) and its collective ordinals count every collective of a 36-layer NEFF, so on
    a shorter stack it would bind a prefix of the names against a different numbering. A file
    named explicitly (a path in MEGA_DMA_ORDER_SCHEDULE) is applied to whatever is run.
    """
    if _SCHEDULE_SOURCE is None:
        return None
    if _SCHEDULE_SOURCE != os.path.abspath(_gk_sched.GK_SCHEDULE):
        return _SCHEDULE_SOURCE
    if giga_variant != "c128" or num_layers != 36:
        return None
    # The order names the on-device mask prologue and the FP8 KV scale loads, and expects each
    # layer to load its own weights; on another graph those names float free with no error.
    bad = [
        name
        for name, hit in (
            ("VLLM_NEURON_MEGA_HOST_MASK=1 (the order names the 7 on-device mask DMAs)", _HOST_MASK),
            ("VLLM_NEURON_MEGA_FP8_KV unset (the order names the 36 FP8 KV scale loads)", not _FP8_KV),
            ("VLLM_NEURON_MEGA_SHARE_LAYER_INPUTS=1 (the order names every layer's weight loads)", _SHARE_LAYER_INPUTS),
        )
        if hit
    ]
    if bad:
        pytest.fail(
            "The default DMA-order schedule does not fit this configuration: "
            + "; ".join(bad)
            + ". Run with MEGA_DMA_ORDER_SCHEDULE=none to let the compiler order the DMAs."
        )
    return _SCHEDULE_SOURCE


def _build_host_masks(
    pos_full,
    pos_swa,
    start_swa,
    abt_full,
    abt_swa,
    active_mask,
    block_len,
    q_head,
    s_tkg,
    b_attn,
    d_head,
    fp8_packed,
    kv_heads,
):
    """Build the full-attention + SWA masks on the host, byte-identical to the
    kernel's on-chip ``gen_mask_tkg_hbm`` hoist. Mirrors that hoist's arguments
    (see gpt_oss_decode_layer_dp_tp_ep): the mask layout is chosen by
    ``is_qk_swapped`` per prior length, and the SWA mask uses the window start.
    Returns (mask_full, mask_swa) as uint8 numpy arrays in the kernel HBM layout.
    """
    from nkilib.models.gpt_oss.c128_giga_kernel.core.attention.attention_tkg_utils import is_qk_swapped
    from nkilib.models.gpt_oss.c128_giga_kernel.core.attention.gen_mask_tkg_torch import (
        gen_mask_tkg_hbm_torch_ref,
    )

    s_prior_full = abt_full.shape[-1] * block_len
    s_prior_swa = abt_swa.shape[-1] * block_len
    is_2byte = not fp8_packed  # fp8 KV is 1 byte; bf16/fp16 KV is 2 bytes

    def _swapped(sp):
        return is_qk_swapped(
            bs=b_attn,
            q_head=q_head,
            d_head=d_head,
            s_active=s_tkg,
            curr_sprior=sp,
            lnc=1,
            p_max=_P_MAX,
            block_len=block_len,
            is_2byte_kv=is_2byte,
            fp8_packed=fp8_packed,
            fuse_rope=False,
            kv_heads=kv_heads,
        )

    n = b_attn * s_tkg
    pf = torch.from_numpy(np.ascontiguousarray(pos_full).astype(np.float32)).reshape(1, n)
    ps = torch.from_numpy(np.ascontiguousarray(pos_swa).astype(np.float32)).reshape(1, n)
    ss = torch.from_numpy(np.ascontiguousarray(start_swa).astype(np.float32)).reshape(1, n)
    am = torch.from_numpy(np.ascontiguousarray(active_mask)).float()
    mf = gen_mask_tkg_hbm_torch_ref[1](
        pos_ids_hbm=pf,
        bs=b_attn,
        q_head=q_head,
        s_active=s_tkg,
        s_prior=s_prior_full,
        start_pos_hbm=None,
        block_len=block_len,
        active_mask=am,
        transposed_out=_swapped(s_prior_full),
    )
    ms = gen_mask_tkg_hbm_torch_ref[1](
        pos_ids_hbm=ps,
        bs=b_attn,
        q_head=q_head,
        s_active=s_tkg,
        s_prior=s_prior_swa,
        start_pos_hbm=ss,
        block_len=block_len,
        active_mask=am,
        transposed_out=_swapped(s_prior_swa),
    )
    return (
        np.ascontiguousarray(mf.numpy().astype(np.uint8)),
        np.ascontiguousarray(ms.numpy().astype(np.uint8)),
    )


# ── Real GPT-OSS-20B decode structure (kernel-padded dims). ──
# TP is env-tunable (VLLM_NEURON_MEGA_TP). The model head counts are fixed (64 Q / 8 KV,
# GQA); the per-rank shard is DERIVED by dividing them by TP, so TP1/2/4/8 all work:
#   TP8 -> 8 Q / 1 KV per rank (the clean 1:1 case);  TP4 -> 16 Q / 2 KV;  TP1 -> 64 Q / 8 KV.
# At TP<8 a rank owns >1 KV head, which flips the per-rank KV tensors (W_qkv K/V slices,
# K/V cache, block table, update index) from single-head to multi-head — see
# generate_array_inputs and _localize_update_idx.
_TP = int(os.environ.get("VLLM_NEURON_MEGA_TP", "8"))
_H = 3072  # hidden, padded from 2880 to a multiple of 512 for the MX kernel
_H_ACTUAL = 2880  # true 20B hidden (RMSNorm divisor)
_I = 3072  # intermediate, padded from 2880
_D_HEAD = 64
_FULL_Q_HEADS = 64  # full-model Q-heads (fixed by the model)
_FULL_KV_HEADS = 8  # full-model KV-heads / GQA groups (fixed by the model)
assert _FULL_Q_HEADS % _TP == 0 and _FULL_KV_HEADS % _TP == 0, (
    f"TP={_TP} must divide both Q-heads ({_FULL_Q_HEADS}) and KV-heads ({_FULL_KV_HEADS})"
)
_Q_HEADS = _FULL_Q_HEADS // _TP  # Q-heads per rank (TP8->8, TP4->16, TP1->64)
_KV_HEADS = _FULL_KV_HEADS // _TP  # KV-heads per rank (TP8->1, TP4->2, TP1->8)
# The profiling default is 10k. Override this for smaller correctness runs.
_S_CTX = int(os.environ.get("VLLM_NEURON_MEGA_S_CTX", "10240"))
_S_TKG = 1
_TOP_K = 4
_EPS = 1e-5
_SWIGLU_ALPHA = 1.702
_GATE_CLAMP_UPPER = 7.0
_UP_CLAMP_UPPER = 8.0
_UP_CLAMP_LOWER = -6.0
_BLOCK_LEN = 128  # block (page) size; matches the production decode configuration
_SLIDING_WINDOW = 128  # real 20B SWA window (even layers)
_P_MAX = 128  # attention kernel partition-tile size (S_ctx must be a multiple)
_MX = nl.float4_e2m1fn_x4
_DTYPE = nl.float16

# The default is GPT-OSS-20B. Full-scale profiling uses the GPT-OSS-120B
# topology previously exercised by the optimization commits.
_N_EXPERTS = 128 if _FULL_120B_PROFILE else 32
_RANKS = 128 if _FULL_120B_PROFILE else 32
# DP replicas fill the rank grid once TP is fixed: TP8 -> DP4 (20B) / DP16 (120B);
# TP4 -> DP8 (20B) / DP32 (120B); TP1 -> DP32 / DP128. RANKS/N_EXPERTS (EP == 1 expert/rank)
# are held fixed, so a TP change is absorbed entirely by DP.
_DP = _RANKS // _TP
_MOE_SEED = 1234  # global np.random seed set right before generate_inputs so its
# gen_moe_mx_weights (which runs before generate_inputs' own
# np.random.seed(42)) is reproducible across kernel + ref.

# (num_layers, logical_nc_config)
_UNKNOWN_GIGA = [v for v in _GIGA_SEL if v not in _GIGA_VARIANTS]
assert not _UNKNOWN_GIGA, f"Unknown VLLM_NEURON_MEGA_GIGA variant(s) {_UNKNOWN_GIGA}; valid: {sorted(_GIGA_VARIANTS)}"

# (giga_variant, num_layers, logical_nc_config) — cross the selected giga variant(s) with
# the layer configs so each variant lands in its own test id (e.g. ...[c512-L1-lnc1]).
LAYER_PARAM_NAMES = "giga_variant, num_layers, logical_nc_config"
_LAYER_CFGS = [(36, 1)] if _FULL_120B_PROFILE else [(1, 1), (2, 1), (4, 1), (8, 1)]
LAYER_PARAMS = [(g, nlyr, lnc) for g in _GIGA_SEL for (nlyr, lnc) in _LAYER_CFGS]
_ABBREVS = {"giga_variant": "giga", "num_layers": "L", "logical_nc_config": "lnc"}


def _is_sliding(layer_idx: int) -> bool:
    """Real GPT-OSS: even layers slide, odd layers are full attention."""
    return layer_idx % 2 == 0


def _num_swa_blocks(sliding_window, block_len, num_full_blocks):
    """Number of KV blocks a sliding layer's TRIMMED block table needs.

    Mirrors the production model runner ``_compute_swa_num_blocks``: the window
    needs ``sliding_window // block_len + 1`` blocks, rounded UP to a multiple of
    ``_P_MAX // block_len`` (so the trimmed S_ctx stays a multiple of the
    attention kernel's P_MAX tile), then clipped to the full-context block count.
    At block_len 128 / window 128 this is 2 blocks (trimmed S_ctx = 256) vs the
    full 80 blocks (S_ctx 10240).
    """
    min_blocks = sliding_window // block_len + 1
    blocks_per_pmax = max(1, _P_MAX // block_len)
    rounded = (min_blocks + blocks_per_pmax - 1) // blocks_per_pmax * blocks_per_pmax
    return int(min(rounded, num_full_blocks))


def _trim_block_table_and_positions(btab, pos_ids, sliding_window, block_len):
    """Trim a FULL per-rank block table + absolute positions into the SWA window
    frame, exactly like the production ``_compute_swa_decode_tensors``.

    Args:
        btab: full block table, either ``[B, num_full_blocks]`` (kv_heads==1 form) or
            ``[B, kv_heads, num_full_blocks]`` (kv_heads>1); LOCAL block indices, -1
            sentinels for inactive blocks. The window (start_block) is head-independent
            (derived from positions), so every head is trimmed with the SAME window.
        pos_ids: absolute positions ``[B, S_tkg]`` (== per-seq cache_len for S_tkg=1).
        sliding_window: SWA window (>0).
        block_len: block (page) size.

    Returns:
        (trimmed_btab ``[B, num_swa]``, pos_ids_swa ``[B, S_tkg]``), where the
        trimmed table gathers the window-relevant blocks
        ``[start_block, start_block + num_swa)`` per seq and ``pos_ids_swa`` is
        ``pos_ids - start_block * block_len`` (position in the trimmed frame).
        Reading the trimmed table with the shifted positions yields bit-identical
        attention to reading the full table with absolute positions (proven), but
        touches only ``num_swa`` blocks of HBM.
    """
    num_full = btab.shape[-1]
    num_swa = _num_swa_blocks(sliding_window, block_len, num_full)
    # Per-seq window start block from the active position (min over S_tkg == the
    # single decode token here). clamp>=0 for short sequences. Head-independent.
    pos = pos_ids.astype(np.int64)
    min_pos = pos.min(axis=1)  # [B]
    start_block = np.clip(min_pos // block_len - num_swa + 1, 0, None)  # [B]
    base_gather = start_block[:, None] + np.arange(num_swa, dtype=np.int64)[None, :]  # [B, num_swa]
    if btab.ndim == 3:
        # [B, kv_heads, num_full] -> trim the last axis per head with the SAME window.
        gather_idx = np.broadcast_to(base_gather[:, None, :], (btab.shape[0], btab.shape[1], num_swa))
        trimmed = np.take_along_axis(btab, gather_idx.astype(btab.dtype), axis=2)
    else:
        trimmed = np.take_along_axis(btab, base_gather.astype(btab.dtype), axis=1)
    pos_swa = (pos - (start_block * block_len)[:, None]).astype(np.float32)
    return np.ascontiguousarray(trimmed.astype(np.int32)), np.ascontiguousarray(pos_swa)


def _localize_update_idx(upd, block_len, full_kv_heads, local_kv_heads, local_head):
    """Map a GLOBAL block-KV update slot to this rank's LOCAL merged-pool slot for one head.

    ``generate_kernel_inputs`` builds ``kv_cache_update_idx`` against the flattened
    ``num_blocks * full_kv_heads`` pool: the global slot is
    ``(base * full_kv_heads + global_head) * block_len + offset``. A rank owns a contiguous
    span of ``local_kv_heads`` heads; its own cache ``[nb, local_kv_heads, block_len, d]`` is
    squeezed HEAD-INNER by attention_block_tkg into a merged ``[nb * local_kv_heads, block_len, d]``
    pool where ``merged_block = base * local_kv_heads + local_head``, and the in-kernel scatter
    (_update_block_cache_vectorized) writes the index DIRECTLY into that merged pool. So the local
    write slot for this rank's ``local_head`` is
    ``(base * local_kv_heads + local_head) * block_len + offset``.

    IMPORTANT: this is NOT the same localization the block TABLE (read path) gets. The read table
    stays head-independent ``base`` (attention_tkg adds the head from the folded batch position);
    only the WRITE index carries the local-head factor. At TP8 (local_kv_heads=1, local_head=0)
    this collapses to ``base*block_len + offset``, matching the previous single-head behavior.
    Getting it wrong addresses out of bounds -> every scatter skipped/clamped (write never runs
    in-bounds; fp8 gather-modify-scatter desyncs ranks -> collective hang). The uint32 sentinel
    (0xFFFFFFFF, intentionally-padded tokens) is preserved so those tokens still skip.

    Args:
        upd: global block-KV update index (uint32), any shape; value =
            (base*full_kv_heads+global_head)*block_len + offset, or 0xFFFFFFFF for padding.
        block_len: block (page) size.
        full_kv_heads: full-model KV-head count folded into the global pool (GQA fan-out).
        local_kv_heads: KV heads this rank owns (== _KV_HEADS).
        local_head: this head's index within the rank's span (0..local_kv_heads-1).

    Returns:
        uint32 array (same shape) of local merged-pool slots
        ((base*local_kv_heads+local_head)*block_len+offset), sentinels preserved.

    Example:
        >>> # TP8: 8 KV heads, 1 per rank -> base*block_len+offset
        >>> _localize_update_idx(np.array([(5*8+3)*128 + 7], np.uint32), 128, 8, 1, 0)
        array([647], dtype=uint32)  # base=5 -> 5*128+7
    """
    u = np.asarray(upd).astype(np.int64)
    sentinel = u >= int(np.iinfo(np.uint32).max)  # 0xFFFFFFFF padding
    base = (u // block_len) // full_kv_heads
    offset = u % block_len
    local = (base * local_kv_heads + local_head) * block_len + offset
    return np.where(sentinel, np.int64(0xFFFFFFFF), local).astype(np.uint32)


def _attn_cfg_full(batch, sliding_window):
    """ONE coherent full-model (64Q/8KV GQA) attention set for a DP replica.

    All TP ranks in a replica share the KV heads / cache / tokens / cos / sin /
    mask; each rank slices its own 8-Q-head + KV-head-r shard. sliding_window>0
    for even (SWA) layers, 0 for odd (full) layers.
    """
    return AttnBlkTestConfig(
        batch=batch,
        q_heads=_FULL_Q_HEADS,
        kv_heads=_FULL_KV_HEADS,
        d_head=_D_HEAD,
        H=_H,
        H_actual=_H_ACTUAL,
        S_ctx=_S_CTX,
        S_max_ctx=_S_CTX,
        S_tkg=_S_TKG,
        lnc=1,
        block_len=_BLOCK_LEN,
        rmsnorm_X=True,
        test_sink=True,
        use_pos_id=True,
        sliding_window=sliding_window,
        update_cache=_KV_UPDATE,
        dtype=_DTYPE,
        kv_quant=_FP8_KV,
        kv_quant_dtype=_KV_QUANT_DTYPE,
        # PACKED FP8 for both read-only profiling AND KV UPDATE. The packed in-kernel WRITE
        # (_update_block_cache_vectorized -> _k_update_fp8_packed parity-split load-modify-store)
        # was previously believed to hang the inter-rank collective watchdog, so KV-update forced
        # UNPACKED. That hang was actually a FAULTY (out-of-bounds) kv_cache_update_idx desyncing
        # ranks (now fixed by _localize_update_idx); it manifested in BOTH packed and unpacked
        # layouts. With the index localized, the packed write is exercised in-bounds. The golden is
        # packed-aware (unpacks the 5D K on read, repacks the 6D stacked K on write), so packed KV
        # update is validated end-to-end here.
        fp8_packed=_FP8_KV,
    )


def _np_dtype():
    """numpy dtype matching the kernel stream dtype (_DTYPE = nl.float16)."""
    return np.float16


def _zero_attn_hidden_padding(aki):
    """Zero the padded hidden columns [H_actual:H] of the O-projection weight, in
    place, so attention writes 0 (not garbage) into the padded output columns.

    Real GPT-OSS 20B has hidden 2880; the LNC=1 MXFP4 kernel needs H%512==0 so we
    pad to 3072. ``generate_kernel_inputs`` already zeros the padded INPUT columns
    (X[:, H_actual:]=0) but leaves W_out's padded output columns [:, H_actual:]
    random. Those feed the residual stream and the next layer's MoE, and the
    kernel's MXFP4 path can overflow them to +/-inf, which then mismatches the
    dense fp32 golden (finite garbage) as a finite-vs-inf error even though the
    real 2880 columns agree. Zeroing W_out's padded output columns makes both the
    kernel and the (locally-regenerated) golden emit 0 there, so the full-tensor
    comparison passes and we validate the true 2880-wide result. Called on the
    SAME regenerated ``aki`` in both create_inputs and the golden ref, so they stay
    bit-identical.
    """
    Wout = np.asarray(aki["W_out"])  # [q_heads*d_head, H]
    if Wout.shape[1] > _H_ACTUAL:
        Wout[:, _H_ACTUAL:] = 0
    return aki


# Memoized attention-input generation. The full 64Q/8KV attention set (aki) is a
# pure function of (seed, sliding_window, B_attn): generate_array_inputs and the
# golden ref both build it as _zero_attn_hidden_padding(generate_kernel_inputs(
# _attn_cfg_full(B_attn, sliding), seed=seed)), with seed = 1000*dp_replica +
# 31*layer. So only DP*layer_input_count distinct akis exist, yet the harness
# regenerates them per rank in the cross-rank consistency check, the eager ref-input
# build, AND the golden compute. generate_kernel_inputs is the single largest golden
# cost (~1-2 s/call at these dims). Memoizing on the key collapses that to one build
# per distinct aki. The cached aki is treated as read-only: every consumer copies
# before mutating (slice / np.ascontiguousarray / _to_t / .clone()), and
# _zero_attn_hidden_padding is idempotent, so sharing one instance across ranks is
# bit-safe. The memo is a module global so a parent process's fills are inherited
# copy-on-write by ranks the SimDistRunner forks (see make_golden_ref's priming).
_ATTN_AKI_MEMO = {}


def _gen_attn_inputs(seed, sliding, batch):
    """Return the (memoized) full-model attention input set for this (seed, sliding,
    batch). Byte-identical to the previous inline
    ``_zero_attn_hidden_padding(generate_kernel_inputs(_attn_cfg_full(batch, sliding),
    seed=seed))`` — it just caches the result so repeated (seed, sliding) across
    ranks reuse one build."""
    key = (seed, sliding, batch)
    aki = _ATTN_AKI_MEMO.get(key)
    if aki is None:
        aki = _zero_attn_hidden_padding(generate_kernel_inputs(_attn_cfg_full(batch, sliding), seed=seed))
        _ATTN_AKI_MEMO[key] = aki
    return aki


def _zero_moe_hidden_padding(mg):
    """Zero the padded hidden-output columns [H_actual:H] of the MoE down projection
    (packed weight, its scale, and the down bias), in place. The down projection's
    last axis IS the hidden output H (packed down weight [E,128,n_I512,H], scale
    [E,16,n_I512,H], bias [E,H]); its padded columns are random and would land
    random values in the layer's padded output columns (differently for the kernel's
    MXFP4 path vs the dense golden). Zeroing them makes both emit 0 there so the
    full-tensor comparison validates only the real 2880-wide result. Applied to the
    same weight set create_inputs slices from AND the ref regenerates, keeping them
    identical."""
    for key, hax in (("expert_down_weights", -1), ("expert_down_weights_scale", -1), ("expert_down_bias", -1)):
        a = np.asarray(mg[key])
        if a.shape[hax] > _H_ACTUAL:
            a[..., _H_ACTUAL:] = 0
    return mg


def _regen_moe_global():
    """Return the global MXFP4->dense MoE weight set (identical across layers AND
    ranks: ``generate_inputs`` seeds internally, so every layer's ``create_inputs``
    MoE slice comes from the same weights). Generation (~33s at H=I=3072, 32 experts)
    + dense dequant (~6s) dominate the golden cost and are recomputed identically by
    all 32 ranks (each in its own process), so the result is DISK-MEMOIZED: the first
    rank builds + pickles it, the rest load it. Cost becomes O(1) across ranks+layers
    instead of O(ranks * layers)."""
    import os
    import pickle
    import tempfile

    cache = os.path.join(tempfile.gettempdir(), f"gptoss_mega_moe_dense_H{_H}_I{_I}_E{_N_EXPERTS}.pkl")
    if os.path.exists(cache):
        try:
            with open(cache, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass  # corrupt/partial -> rebuild
    # CRITICAL determinism: generate_inputs calls gen_moe_mx_weights (which draws
    # from GLOBAL np.random) BEFORE its own np.random.seed(42), so the MXFP4 expert
    # weights depend on the global RNG state at call time. create_inputs (kernel side)
    # and this ref must seed identically right before the call so both get the SAME
    # weights, else the golden compares against different experts (cosine ~0.16).
    np.random.seed(_MOE_SEED)
    mg = generate_inputs(
        batch=_RANKS,
        seqlen=1,
        hidden=_H,
        hidden_actual=_H_ACTUAL,
        intermediate=_I,
        num_global_experts=_N_EXPERTS,
        num_local_experts=_N_EXPERTS,
        top_k=_TOP_K,
        router_fn=None,
        hidden_act_fn=None,
        expert_affinities_scaling_mode=None,
        moe_weight_dtype=_MX,
        input_dtype=_DTYPE,
        has_bias=True,
        has_clamp=True,
        router_act_first=False,
        norm_topk_prob=False,
        skip_router_logits=True,
        router_mm_dtype=nl.float16,
        is_all_expert=True,
    )
    mg["expert_gate_up_bias"][:, :, 1, ...] += 1.0  # GPT-OSS +1 on up bias
    _zero_moe_hidden_padding(mg)  # zero padded H output cols
    dense = {
        "gu": packed_mx_to_dense_gate_up(mg["expert_gate_up_weights"], mg["expert_gate_up_weights_scale"], _H, _I),
        "dn": packed_mx_to_dense_down(mg["expert_down_weights"], mg["expert_down_weights_scale"], _H, _I),
        "gub": unpack_gate_up_bias(mg["expert_gate_up_bias"], _H, _I),
        "dnb": _to_t(mg["expert_down_bias"]),
        "router_w": _to_t(mg["router_weights"]).t().contiguous(),
        "router_b": _to_t(mg["router_bias"]).reshape(-1),
        "gamma": _to_t(mg["gamma"]).reshape(_H),
    }
    try:
        tmp = cache + f".{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(dense, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache)
    except Exception:
        pass  # cache is best-effort
    return dense


def _to_t(a):
    if a is None:
        return None
    if isinstance(a, torch.Tensor):
        return a.float()
    s = str(a.dtype)
    if "bfloat16" in s or "float8" in s or "float16" in s:
        return torch.from_numpy(a.astype(np.float32))
    if a.dtype == np.uint32:
        return torch.from_numpy(a.astype(np.int64))
    return torch.from_numpy(a.copy())


def make_golden_ref(tp_group, ep_group, tp_size, ep_size, num_layers):
    """Golden-backed distributed ref: loop the whole-layer oracle num_layers times.

    Threads the hidden state layer-to-layer (== GptOssModel.forward_decode); each
    layer rebuilds its full 64Q/8KV attention weights + KV cache via AllGather[TP]
    and its full expert set via AllGather[EP], then calls gptoss_mxfp4_decode_layer
    with that layer's own SWA setting. Returns this rank's SP token shard.
    """

    # Per-DP-replica ``out_full`` cache (see ref's memo note). Keyed by dp_replica.
    _outfull_memo = {}

    def _compute_out_full(
        dp_replica,
        B_attn,
        S_tkg,
        num_layers,
        top_k,
        eps,
        gate_clamp_upper_limit,
        cos,
        sin,
        pos_ids,
        pos_ids_swa,
        active_blocks_table,
        active_blocks_table_swa,
    ):
        """Run the num_layers whole-layer oracle stack for a whole DP replica and
        return the full ``out_full`` [B_attn, H]. This is the exact per-rank layer
        loop the ref used to run inline — factored out so it runs ONCE per replica
        instead of once per TP rank. The shared attention control (cos/sin/pos/block
        tables) is TP-rank-invariant, so passing the caller's copies is exact."""
        H = _H
        I = _I
        d_head = _D_HEAD

        # Shared attention control. Full (odd) layers use the absolute positions +
        # full block table; sliding (even) layers use the SAME shared trimmed table
        # + window-frame positions the kernel uses (passed as kernel inputs), so the
        # golden reads the identical physical blocks / logical span the kernel does.
        cos_g = _to_t(cos).permute(1, 2, 0).reshape(B_attn * S_tkg, d_head // 2)
        sin_g = _to_t(sin).permute(1, 2, 0).reshape(B_attn * S_tkg, d_head // 2)
        pos_flat = _to_t(pos_ids).reshape(-1).long()  # absolute (full layers)
        pos_flat_swa = _to_t(pos_ids_swa).reshape(-1).long()  # window frame (sliding layers)
        # Shared full / trimmed block tables. At TP8 these arrive as the kv_heads==1 form [B, nb]
        # holding LOGICAL base indices; at TP<8 they are [B, kv_heads, nb] holding MERGED indices
        # (base*kv_heads + head) for the kernel's head-inner-squeezed cache. The full oracle
        # (attention_decode) gathers a [nb, 8, bl, d] cache with a 2-D [B, nb] table of LOGICAL base
        # indices, so recover base from head 0's value: base*kv_heads // kv_heads == base (the
        # torch.where guards -1 sentinels, which floor-divide to -1 anyway).
        _abt_ro = _to_t(active_blocks_table)
        _abt_ro_swa = _to_t(active_blocks_table_swa)
        if _abt_ro.dim() == 3:
            _kvh = _abt_ro.shape[1]
            _abt_ro = torch.where(_abt_ro[:, 0, :] < 0, _abt_ro[:, 0, :], _abt_ro[:, 0, :] // _kvh)
            _abt_ro_swa = torch.where(_abt_ro_swa[:, 0, :] < 0, _abt_ro_swa[:, 0, :], _abt_ro_swa[:, 0, :] // _kvh)
        btab_full_shared = _abt_ro.reshape(B_attn, -1).int()
        btab_swa_shared = _abt_ro_swa.reshape(B_attn, -1).int()
        # KV UPDATE: the golden oracle writes this token's K/V in-place into the bound cache at
        # slot_mapping. Use the SAME physical slot the kernel writes (kv_cache_update_idx, which
        # is physical_block*block_len + offset), and DON'T clone, so the reference cache carries
        # the update. Without KV update the write is discarded (slot=0, cache cloned) as before.
        # Collected post-update per-layer full caches [nb, kv_heads, block_len, d] (kv_heads=8).
        # Per-layer ingredients for ref's per-rank (_KV_HEADS-head) cache-update reproduction.
        kv_upd_ingredients = {} if _KV_UPDATE else None
        slot = torch.zeros(B_attn * S_tkg, dtype=torch.long)  # read-only path only (write discarded)

        # Full MoE dense weight set (identical for every layer; built + dequantized
        # ONCE — the expensive generate_inputs (~33s) and dequant (~6s) are hoisted
        # out of the per-layer loop so cost is O(1) in num_layers, not O(L)).
        moe_dense = _regen_moe_global()

        # Running hidden state: replica's B_attn pre-norm tokens (regenerated at
        # layer 0 from the same seed create_inputs used; identical on every TP rank).
        hidden = None
        for li in range(num_layers):
            sliding = _SLIDING_WINDOW if _is_sliding(li) else 0
            aki = _gen_attn_inputs(_attn_seed(dp_replica, li), sliding, B_attn)
            if li == 0:
                hidden = _to_t(aki["X"]).reshape(B_attn, H)

            # Full 64Q/8KV attention weights + KV cache (no reassembly needed).
            Wqkv = _to_t(aki["W_qkv"])
            Wout = _to_t(aki["W_out"])
            sink_g = _to_t(aki["sink"]).reshape(-1)
            Kc, Vc = _to_t(aki["K_cache"]), _to_t(aki["V_cache"])  # [nb, 8, block_len, d]
            iln_w = _to_t(aki["rmsnorm_X_gamma"]).reshape(H)
            if _FP8_KV:
                k_scale_scalar = float(_to_t(aki["k_scale"]).float().flatten()[0])
                v_scale_scalar = float(_to_t(aki["v_scale"]).float().flatten()[0])
                # Packed K is [nb, kv, block_len//2, d, 2]. Restore the logical
                # block axis before feeding the dense golden.
                if Kc.dim() == 5:
                    nb, kv_heads, half_block, d_head, packed = Kc.shape
                    Kc = Kc.permute(0, 1, 2, 4, 3).reshape(nb, kv_heads, half_block * packed, d_head)
                Kc = Kc.float() / k_scale_scalar
                Vc = Vc.float() / v_scale_scalar
                # Input generation divides W_out by v_scale for the raw-FP8
                # kernel path. Undo that transform for the dense golden.
                Wout = Wout.float() * v_scale_scalar
            bqkv = torch.zeros(Wqkv.shape[1])  # test_bias=False -> no bias
            bout = torch.zeros(H)

            # SWA block-table trimming (mirror the kernel exactly; see the mega
            # kernel docstring). Sliding (even) layers read the SAME shared trimmed
            # table + window-frame positions the kernel reads, so the golden gathers
            # the identical window blocks of this layer's own KV cache; full (odd)
            # layers read the full table + absolute positions. Because the oracle is
            # trim-invariant (the stored K carries absolute-position RoPE, so
            # relabelling positions into the trimmed frame is exact), this matches
            # the pre-trim golden numerically while faithfully modelling the reduced
            # KV span. The full/trimmed tables and positions are layer-invariant here
            # (decode reuses layer 0's), so the shared kernel inputs are reused.
            if _is_sliding(li):
                layer_btab = btab_swa_shared
                layer_pos = pos_flat_swa
                sw = _SLIDING_WINDOW
            else:
                layer_btab = btab_full_shared
                layer_pos = pos_flat
                sw = None

            # This layer's full expert set (identical across layers; pre-dequantized).
            all_gu, all_dn = moe_dense["gu"], moe_dense["dn"]
            all_gub, all_dnb = moe_dense["gub"], moe_dense["dnb"]
            router_w, router_b = moe_dense["router_w"], moe_dense["router_b"]
            paln_w = moe_dense["gamma"]

            cfg = GptOssConfig(
                hidden_size=H,
                intermediate_size=I,
                num_attention_heads=_FULL_Q_HEADS,
                num_key_value_heads=_FULL_KV_HEADS,
                head_dim=d_head,
                num_local_experts=ep_size,
                num_experts_per_tok=top_k,
                rms_norm_eps=eps,
                swiglu_limit=gate_clamp_upper_limit,
                swiglu_alpha=_SWIGLU_ALPHA,
                torch_dtype=torch.float32,
                sliding_window=(_SLIDING_WINDOW if sw is not None else None),
            )
            out_full = gptoss_mxfp4_decode_layer(
                cfg,
                hidden_states=hidden,
                positions=layer_pos,
                cos=cos_g,
                sin=sin_g,
                input_layernorm_weight=iln_w,
                qkv_proj_weight=Wqkv,
                qkv_proj_bias=bqkv,
                o_proj_weight=Wout,
                o_proj_bias=bout,
                sinks=sink_g,
                post_attention_layernorm_weight=paln_w,
                router_weight=router_w,
                router_bias=router_b,
                gate_up_weight=all_gu,
                gate_up_bias=all_gub,
                down_weight=all_dn,
                down_bias=all_dnb,
                k_cache=Kc.clone(),
                v_cache=Vc.clone(),
                block_table=layer_btab,
                slot_mapping=slot,
                sliding_window=sw,
            ).to(torch.float32)  # [B_attn, H]

            if _KV_UPDATE:
                # Stash the per-layer FULL-model ingredients ref needs to reproduce THIS rank's
                # cache write with the validated attention_decode primitive (ref slices its own
                # _KV_HEADS heads and runs num_kv_heads=_KV_HEADS; _write_kv_cache writes all heads
                # at the per-token slot, matching the kernel per head).
                # Positions: full (odd) layers use absolute; sliding (even) use the window frame.
                # The KV-write oracle (_golden_attention_decode) does a RAW X @ W_qkv with NO input
                # RMSNorm, but the kernel's attention_block_tkg has rmsnorm_X_enabled=True (normalizes
                # before QKV). So pre-apply this layer's input RMSNorm to the threaded hidden here and
                # stash the NORMED state, so the oracle's K/V matches the kernel's written K/V. Formula
                # matches moe_block_tkg's RMSNorm (fp32 accum, gamma scale).
                _xin = hidden.to(torch.float32)
                _var = _xin.pow(2).mean(-1, keepdim=True)
                _xnormed = (_xin * torch.rsqrt(_var + eps)) * iln_w.to(torch.float32)
                kv_upd_ingredients[li] = {
                    # THREADED, INPUT-NORMED hidden == the actual attention-QKV input for THIS layer
                    # (== the kernel's post-AllGather + input-RMSNorm state). MUST derive from `hidden`
                    # (the residual stream from the previous layer), NOT aki["X"]: the kernel computes
                    # each layer's K/V from the threaded state, so layer>=1's cache write depends on it.
                    # (For li==0 hidden==aki["X"].) `hidden` is still this layer's input here — it's
                    # rebound to out_full only after the stash. Cast back to the compute dtype.
                    "X": _xnormed.to(hidden.dtype).reshape(B_attn, H),
                    # Wqkv/Wout here are already in DEQUANT space for fp8 (Wout *= v_scale above),
                    # so attention_decode's fp16 write produces dequantized new K/V; ref re-quantizes.
                    "Wqkv": Wqkv,
                    "sink": sink_g,
                    "Wout": Wout,
                    "bout": torch.zeros(H),
                    # K0/V0 are the kernel's RAW cache: fp16 [nb,8,bl,d] normally, or packed-fp8
                    # (K [nb,8,bl//2,d,2], V [nb,8,bl,d]) when _FP8_KV. Kept in kernel storage form.
                    "K0": np.ascontiguousarray(aki["K_cache"]),
                    "V0": np.ascontiguousarray(aki["V_cache"]),
                    # NOTE: this layer's own GLOBAL kv_cache_update_idx is intentionally NOT stashed;
                    # the write slot is the SHARED, LOCALIZED layer-0 index (kv_cache_update_idx param)
                    # the kernel uses for every layer. See the slot_r assignment in ref().
                    "btab": layer_btab,
                    "pos": layer_pos,
                    "cos": cos_g,
                    "sin": sin_g,
                    "sw": sw,
                    "k_scale": (k_scale_scalar if _FP8_KV else None),
                    "v_scale": (v_scale_scalar if _FP8_KV else None),
                }

            # Thread the FULL layer output as the next layer's input (identical on
            # every TP rank in the replica). No collectives needed.
            hidden = out_full

        if _KV_UPDATE:
            return out_full, kv_upd_ingredients
        return out_full

    def ref(
        x_shard,
        input_layernorm_weight,
        W_qkv,
        bias_qkv,
        W_out,
        bias_out,
        sink,
        K_cache,
        V_cache,
        post_attn_layernorm_weight,
        router_weights,
        router_bias,
        expert_gate_up_weights,
        expert_down_weights,
        expert_gate_up_weights_scale,
        expert_down_weights_scale,
        expert_gate_up_bias,
        expert_down_bias,
        swa_start_pos_ids,
        cos,
        sin,
        active_blocks_table,
        active_blocks_table_swa,
        attention_mask,
        kv_cache_update_idx,
        pos_ids,
        pos_ids_swa,
        rank_id,
        tp_replica_group,
        ep_replica_group,
        num_layers,
        tp_size,
        ep_size,
        B_attn,
        S_tkg,
        top_k,
        eps,
        hidden_actual,
        softmax_scale,
        gate_clamp_upper_limit,
        up_clamp_upper_limit,
        up_clamp_lower_limit,
        k_scale=None,
        v_scale=None,
        # Optional kernel params (ignored by this golden ref): the signature
        # validator requires the ref's param set to match the kernel's exactly.
        # This standalone golden uses per-layer stacked caches (not the HMA
        # shared-buffer layout), so kv_cache_update_idx_swa stays None, and
        # layer 0 gets its hidden from x_shard, so x_first_full stays None.
        kv_cache_update_idx_swa=None,
        x_first_full=None,
        # HOST GEN-MASK: pre-generated masks (ignored by this golden ref, which
        # generates the causal/SWA band itself). Present so the ref param set
        # matches the kernel's exactly (the signature validator requires it).
        pregen_mask_full=None,
        pregen_mask_swa=None,
        # FUSED EMBEDDING: ignored by this golden ref. It rebuilds the layer-0
        # hidden from the same X_full the embedding table encodes, so the kernel's
        # in-NEFF reconstruction is validated end-to-end through the layer output.
        # embed_x_shard_ids recovers the SP shard in the hidden-dim (option b) path.
        embed_weight=None,
        embed_ids=None,
        embed_x_shard_ids=None,
        # FUSED ROPE: ignored by this golden ref. The ref uses the (position-derived)
        # cos/sin the test also feeds the kernel to reconstruct on-chip, so the
        # on-chip RoPE is validated through the layer output.
        inv_freq=None,
        rope_concentration=1.0,
        # HMA individual KV-cache buffers (kernel's alternate interface). This ref
        # regenerates each layer's cache from seeds, so they are unused here; present
        # only so the ref's parameter set matches the kernel's (signature validator).
        K_cache_0=None,
        K_cache_1=None,
        K_cache_2=None,
        K_cache_3=None,
        K_cache_4=None,
        K_cache_5=None,
        K_cache_6=None,
        K_cache_7=None,
        K_cache_8=None,
        K_cache_9=None,
        K_cache_10=None,
        K_cache_11=None,
        K_cache_12=None,
        K_cache_13=None,
        K_cache_14=None,
        K_cache_15=None,
        K_cache_16=None,
        K_cache_17=None,
        V_cache_0=None,
        V_cache_1=None,
        V_cache_2=None,
        V_cache_3=None,
        V_cache_4=None,
        V_cache_5=None,
        V_cache_6=None,
        V_cache_7=None,
        V_cache_8=None,
        V_cache_9=None,
        V_cache_10=None,
        V_cache_11=None,
        V_cache_12=None,
        V_cache_13=None,
        V_cache_14=None,
        V_cache_15=None,
        V_cache_16=None,
        V_cache_17=None,
    ):
        # NOTE: the golden ref uses NO torch.distributed collectives. The
        # SimDistRunner golden harness caps a distributed ref at MAX_COLLECTIVE_PASSES
        # (=10) total collectives, which a multi-layer ref (>=~12/layer) blows past
        # -> deadlock. Instead each rank REGENERATES the full per-(layer, DP-replica)
        # inputs locally from the same deterministic seed create_inputs used
        # (seed = 1000*dp_replica + 31*li for attention; the shared global MoE set),
        # so it needs no gathers. The full 64Q/8KV weights, KV cache, and 32-expert
        # set are reconstructed directly; the whole-layer oracle then runs the full
        # layer over the replica's B_attn tokens and we slice this rank's token.
        dtype = x_shard.dtype
        T_shard, H = x_shard.shape
        rid = int(rank_id[0, 0])
        tp_rank = rid % tp_size
        dp_replica = rid // tp_size

        # KV form, mirroring the kernel's own detection: stacked K_cache, or the per-buffer
        # K_cache_i set with the stacked one None. Under KV_UPDATE the post-update caches are
        # returned in whichever form the kernel was given, so shape/dtype come from that form's
        # reference tensor (the stacked tensor, or buffer 0).
        _use_indiv_kv = K_cache is None
        _kv_ref_k = K_cache_0 if _use_indiv_kv else K_cache
        _kv_ref_v = V_cache_0 if _use_indiv_kv else V_cache

        # The whole-layer forward output ``out_full`` [B_attn, H] depends only on
        # (dp_replica, num_layers): every TP rank in a replica regenerates identical
        # inputs (attention seed = 1000*dp_replica + 31*li, shared MoE set) and
        # computes the identical ``out_full`` — each rank only differs in which token
        # row it slices at the end. So compute it ONCE per replica and memoize; the
        # other 7 TP ranks of the replica reuse it. Verified bit-identical across TP
        # ranks and across separate forked processes. The memo is a closure dict so a
        # parent process's fills (see ``prime`` below) are inherited copy-on-write by
        # the ranks SimDistRunner forks, avoiding recompute in every child.
        memo = _outfull_memo.get(dp_replica)
        if memo is None:
            memo = _compute_out_full(
                dp_replica,
                B_attn,
                S_tkg,
                num_layers,
                top_k,
                eps,
                gate_clamp_upper_limit,
                cos,
                sin,
                pos_ids,
                pos_ids_swa,
                active_blocks_table,
                active_blocks_table_swa,
            )
            _outfull_memo[dp_replica] = memo
        # _compute_out_full returns out_full, plus {li: ingredients} under KV_UPDATE.
        if _KV_UPDATE:
            out_full, upd_k = memo
        else:
            out_full = memo

        # This rank's own token: its TP position within the replica.
        out = out_full[tp_rank * T_shard : (tp_rank + 1) * T_shard, :].contiguous()
        result = {"out": out.numpy().astype(dtype)}

        if _KV_UPDATE:
            # Per-rank post-update cache, reproduced with the SAME validated attention_decode
            # primitive the kernel mirrors, run for THIS rank's _KV_HEADS KV heads. attention_decode
            # natively handles a multi-head cache [nb, kv_heads, bl, d] (its _write_kv_cache writes
            # ALL heads at the per-token slot), matching the kernel's per-rank cache; stacked over
            # layers -> [L, nb, _KV_HEADS, bl, d], exactly the K_cache input layout. Uses upd_k[li]
            # ingredients (full-model tensors); W_qkv/W_out/sink/cache are sliced to this rank here
            # (mirrors generate_array_inputs). At TP8 (_KV_HEADS==1) this is the original single-head
            # path.
            qd_r = _Q_HEADS * _D_HEAD
            q_all = _FULL_Q_HEADS * _D_HEAD
            kv_r = _KV_HEADS * _D_HEAD  # W_qkv K/V columns for this rank's KV heads
            kbase, vbase = q_all, q_all + _FULL_KV_HEADS * _D_HEAD
            k_layers, v_layers = [], []
            # Write slot: the kernel's shared kv_cache_update_idx is the MERGED per-head write index
            # [B, _KV_HEADS, S_tkg] (values (base*_KV_HEADS+h)*bl+offset — see _localize_update_idx).
            # attention_decode's _write_kv_cache instead wants the head-independent LOGICAL slot
            # base*bl+offset (it derives block/pos from it and writes ALL kv heads there). Recover it
            # from head 0: base = (merged0//bl)//_KV_HEADS, then base*bl + merged0%bl, preserving the
            # uint32 sentinel. Shared across layers (the kernel uses ONE index for every layer).
            _kv_idx_np = np.asarray(kv_cache_update_idx)
            _merged0 = _kv_idx_np[:, 0, :] if _kv_idx_np.ndim == 3 else _kv_idx_np
            _m = _merged0.astype(np.int64).reshape(-1)
            _sent = _m >= int(np.iinfo(np.uint32).max)
            _logical = ((_m // _BLOCK_LEN) // _KV_HEADS) * _BLOCK_LEN + (_m % _BLOCK_LEN)
            slot_r = torch.from_numpy(np.ascontiguousarray(np.where(_sent, np.int64(0xFFFFFFFF), _logical))).long()
            for li in range(num_layers):
                ing = upd_k[li]
                Wqkv_f = ing["Wqkv"]  # [H, 64D + 2*8D]
                Wqkv_r = torch.cat(
                    [
                        Wqkv_f[:, tp_rank * qd_r : (tp_rank + 1) * qd_r],
                        Wqkv_f[:, kbase + tp_rank * kv_r : kbase + (tp_rank + 1) * kv_r],
                        Wqkv_f[:, vbase + tp_rank * kv_r : vbase + (tp_rank + 1) * kv_r],
                    ],
                    dim=1,
                ).contiguous()
                Wout_r = ing["Wout"][tp_rank * qd_r : (tp_rank + 1) * qd_r, :].contiguous()
                sink_r = ing["sink"].reshape(-1)[tp_rank * _Q_HEADS : (tp_rank + 1) * _Q_HEADS].contiguous()
                # This rank's KV heads, in kernel storage form.
                _hs = slice(tp_rank * _KV_HEADS, (tp_rank + 1) * _KV_HEADS)
                K0_r = ing["K0"][:, _hs]  # fp16 [nb,kv,bl,d] OR packed-fp8 [nb,kv,bl//2,d,2]
                V0_r = ing["V0"][:, _hs]  # fp16/fp8 [nb,kv,bl,d]
                if _FP8_KV:
                    # Unpack K [nb,kv,bl//2,d,2]->[nb,kv,bl,d] and DEQUANT both to the float space the
                    # write compute uses (Wout already carries v_scale, so the written K/V come out
                    # dequantized). attention_decode's _write_kv_cache casts to the passed cache dtype,
                    # so give it float caches, then re-quantize+pack below to the kernel's stored form.
                    ksc, vsc = ing["k_scale"], ing["v_scale"]
                    # ml_dtypes.float8_e4m3fn arrays can't go through torch.from_numpy; cast to
                    # float32 in numpy first (dequant is exact for fp8-representable values).
                    Kt = torch.from_numpy(np.ascontiguousarray(K0_r).astype(np.float32))
                    if Kt.dim() == 5:  # [nb,kv,bl//2,d,2] -> [nb,kv,bl,d]
                        nb_, _kv, hb, dd, pk = Kt.shape
                        Kt = Kt.permute(0, 1, 2, 4, 3).reshape(nb_, _kv, hb * pk, dd)
                    Kc_r = Kt / ksc
                    Vc_r = torch.from_numpy(np.ascontiguousarray(V0_r).astype(np.float32)) / vsc
                else:
                    Kc_r = torch.from_numpy(np.ascontiguousarray(K0_r))  # fp16 [nb,kv,bl,d]
                    Vc_r = torch.from_numpy(np.ascontiguousarray(V0_r))
                # block_table for the write oracle: this rank's LOCAL base indices [B, nb] (shared
                # across heads; head 0's table, base being head-independent). It only affects the
                # (discarded) attention output — the cache WRITE uses slot_r — but must be in-bounds
                # so the prior-K/V gather doesn't fault (attention_decode clamps -1 sentinels to 0).
                abt_r = ing["btab"].int()
                _golden_attention_decode(
                    X=ing["X"].reshape(B_attn, S_tkg, H),
                    W_qkv=Wqkv_r,
                    bias_qkv=torch.zeros(Wqkv_r.shape[1]),
                    num_q_heads=_Q_HEADS,
                    num_kv_heads=_KV_HEADS,
                    head_dim=_D_HEAD,
                    cos=ing["cos"],
                    sin=ing["sin"],
                    K_cache=Kc_r,
                    V_cache=Vc_r,
                    block_table=abt_r,
                    slot_mapping=slot_r,
                    pos_ids=ing["pos"].reshape(-1).long(),
                    sliding_window=ing["sw"],
                    sink=sink_r,
                    softmax_scale=softmax_scale,
                    W_out=Wout_r,
                    bias_out=ing["bout"],
                    update_cache=True,
                )  # Kc_r/Vc_r mutated in place (dequant space for fp8, native for fp16)
                if _FP8_KV:
                    # Re-quantize to fp8 (mirror attention_block_tkg_torch._quantize_kv_to_fp8).
                    # PACKED fp8 KV update: the cache is the packed layout, stacked
                    # [L,nb,kv,bl//2,d,2] or per-buffer [nb,kv,bl//2,d,2], so repack this layer's
                    # written K [nb,kv,bl,d] below (interleave adjacent pairs into the trailing dim).
                    ksc, vsc = ing["k_scale"], ing["v_scale"]
                    fp8_max = 448.0  # float8_e4m3fn max positive
                    Kq = torch.clamp(Kc_r * ksc, -fp8_max, fp8_max)  # [nb,kv,bl,d]
                    Vq = torch.clamp(Vc_r * vsc, -fp8_max, fp8_max)
                    # Packed iff the reference cache carries the trailing pair axis: stacked adds a
                    # leading layer axis (6D) over the per-buffer form (5D).
                    if _kv_ref_k.ndim == (5 if _use_indiv_kv else 6):
                        nb_, _kv, bl_, dd = Kq.shape
                        Kq = Kq.reshape(nb_, _kv, bl_ // 2, 2, dd).permute(0, 1, 2, 4, 3)
                    k_layers.append(Kq.numpy())
                    v_layers.append(Vq.numpy())
                else:
                    k_layers.append(Kc_r.numpy())
                    v_layers.append(Vc_r.numpy())
            if _use_indiv_kv:
                # Per-buffer interface: each layer's post-update cache is its OWN buffer (no
                # leading layer axis), matching the K_cache_i/V_cache_i the kernel aliases back.
                for _j in range(len(k_layers)):
                    result[f"K_cache_{_j}"] = k_layers[_j].astype(_kv_ref_k.dtype)
                    result[f"V_cache_{_j}"] = v_layers[_j].astype(_kv_ref_v.dtype)
            else:
                result["K_cache"] = np.stack(k_layers, axis=0).astype(K_cache.dtype)  # [L, nb, kv, bl(/2..), d(,2)]
                result["V_cache"] = np.stack(v_layers, axis=0).astype(V_cache.dtype)
        return result

    def prime(create_inputs_fn):
        """Populate ``_outfull_memo`` in the CURRENT (parent) process, before the
        SimDistRunner forks per-rank workers. Runs the whole-layer stack once per DP
        replica (rank ``dp*tp_size``); the forked ranks then inherit these results
        copy-on-write and every rank's ref() is a memo hit (just a token slice). This
        turns the golden compute from ranks full forwards into DP. Without this
        priming, each forked rank starts with an empty memo and would recompute its
        replica, so the sharing MUST happen in the parent. Uses torch threads<=4
        (verified bit-identical to the threads=1 result; forked ranks pin threads=1)."""
        import torch as _torch

        _saved = _torch.get_num_threads()
        _torch.set_num_threads(min(4, _saved))
        try:
            for dp in range(ep_size // tp_size):
                rank = dp * tp_size
                raw = create_inputs_fn(rank_id=rank)
                # Strip any ".must_alias_input" suffix (KV_UPDATE renames K_cache/V_cache) so the
                # ref's plain K_cache/V_cache params are populated, mirroring filter_ref_input.
                raw = {k.rsplit(".must_alias_input", 1)[0]: v for k, v in raw.items()}
                kwargs = {k: raw[k] for k in _REF_PARAM_NAMES if k in raw}
                ref(**kwargs)  # fills _outfull_memo[dp]
        finally:
            _torch.set_num_threads(_saved)

    ref.prime = prime
    return ref


# Parameter names of make_golden_ref's ``ref`` — used to filter create_inputs'
# (superset) output down to the ref signature when priming (see ref.prime). The
# optional k_scale/v_scale are included so the FP8-KV path passes them through.
_REF_PARAM_NAMES = (
    "x_shard",
    "input_layernorm_weight",
    "W_qkv",
    "bias_qkv",
    "W_out",
    "bias_out",
    "sink",
    "K_cache",
    "V_cache",
    "post_attn_layernorm_weight",
    "router_weights",
    "router_bias",
    "expert_gate_up_weights",
    "expert_down_weights",
    "expert_gate_up_weights_scale",
    "expert_down_weights_scale",
    "expert_gate_up_bias",
    "expert_down_bias",
    "swa_start_pos_ids",
    "cos",
    "sin",
    "active_blocks_table",
    "active_blocks_table_swa",
    "attention_mask",
    "kv_cache_update_idx",
    "pos_ids",
    "pos_ids_swa",
    "rank_id",
    "tp_replica_group",
    "ep_replica_group",
    "num_layers",
    "tp_size",
    "ep_size",
    "B_attn",
    "S_tkg",
    "top_k",
    "eps",
    "hidden_actual",
    "softmax_scale",
    "gate_clamp_upper_limit",
    "up_clamp_upper_limit",
    "up_clamp_lower_limit",
    "k_scale",
    "v_scale",
    # Per-buffer KV interface (VLLM_NEURON_MEGA_KV_INDIVIDUAL): the ref needs these to see
    # which form it was given, and under KV_UPDATE to emit each post-update buffer.
    *(f"K_cache_{_j}" for _j in range(18)),
    *(f"V_cache_{_j}" for _j in range(18)),
)


def make_stub_ref():
    """Zero-returning stand-in for the golden ref, used when VLLM_NEURON_MEGA_SKIP_GOLDEN
    is set. Same signature/output shape as make_golden_ref's ``ref`` so the harness's
    output descriptors are correct, but does NO weight regeneration / oracle math —
    the ~40-min golden pass becomes instant. Validation against this stub is
    meaningless (all zeros); use only for on-device perf/profiling runs.
    """

    def ref(
        x_shard,
        input_layernorm_weight,
        W_qkv,
        bias_qkv,
        W_out,
        bias_out,
        sink,
        K_cache,
        V_cache,
        post_attn_layernorm_weight,
        router_weights,
        router_bias,
        expert_gate_up_weights,
        expert_down_weights,
        expert_gate_up_weights_scale,
        expert_down_weights_scale,
        expert_gate_up_bias,
        expert_down_bias,
        swa_start_pos_ids,
        cos,
        sin,
        active_blocks_table,
        active_blocks_table_swa,
        attention_mask,
        kv_cache_update_idx,
        pos_ids,
        pos_ids_swa,
        rank_id,
        tp_replica_group,
        ep_replica_group,
        num_layers,
        tp_size,
        ep_size,
        B_attn,
        S_tkg,
        top_k,
        eps,
        hidden_actual,
        softmax_scale,
        gate_clamp_upper_limit,
        up_clamp_upper_limit,
        up_clamp_lower_limit,
        k_scale=None,
        v_scale=None,
        # Optional kernel params (ignored by the stub ref): the signature
        # validator requires the ref's param set to match the kernel's exactly.
        # kv_cache_update_idx_swa = SWA disjoint-slot write index (KV_UPDATE);
        # x_first_full = layer-0 pre-gathered hidden (AllGather-skip);
        # pregen_mask_full / pregen_mask_swa = host-built attention masks;
        # embed_weight / embed_ids / embed_x_shard_ids = fused-embedding inputs. All
        # default None in the kernel and are unused by this zero-returning stub.
        kv_cache_update_idx_swa=None,
        x_first_full=None,
        pregen_mask_full=None,
        pregen_mask_swa=None,
        embed_weight=None,
        embed_ids=None,
        embed_x_shard_ids=None,
        inv_freq=None,
        rope_concentration=1.0,
        # HMA individual KV-cache buffers (kernel's alternate interface); unused by
        # this zero-returning stub, present only to match the kernel's parameter set.
        K_cache_0=None,
        K_cache_1=None,
        K_cache_2=None,
        K_cache_3=None,
        K_cache_4=None,
        K_cache_5=None,
        K_cache_6=None,
        K_cache_7=None,
        K_cache_8=None,
        K_cache_9=None,
        K_cache_10=None,
        K_cache_11=None,
        K_cache_12=None,
        K_cache_13=None,
        K_cache_14=None,
        K_cache_15=None,
        K_cache_16=None,
        K_cache_17=None,
        V_cache_0=None,
        V_cache_1=None,
        V_cache_2=None,
        V_cache_3=None,
        V_cache_4=None,
        V_cache_5=None,
        V_cache_6=None,
        V_cache_7=None,
        V_cache_8=None,
        V_cache_9=None,
        V_cache_10=None,
        V_cache_11=None,
        V_cache_12=None,
        V_cache_13=None,
        V_cache_14=None,
        V_cache_15=None,
        V_cache_16=None,
        V_cache_17=None,
    ):
        T_shard, H = x_shard.shape
        res = {"out": np.zeros((T_shard, H), dtype=x_shard.dtype)}
        if _KV_UPDATE:
            # output_keys includes the post-update caches under KV_UPDATE; return matching-shape
            # zeros (validation is meaningless under skip-golden anyway). Mirror whichever KV
            # interface the kernel was given, since that is what output_keys names.
            if K_cache is None:
                _k_bufs = [
                    K_cache_0, K_cache_1, K_cache_2, K_cache_3, K_cache_4, K_cache_5,
                    K_cache_6, K_cache_7, K_cache_8, K_cache_9, K_cache_10, K_cache_11,
                    K_cache_12, K_cache_13, K_cache_14, K_cache_15, K_cache_16, K_cache_17,
                ]  # fmt: skip
                _v_bufs = [
                    V_cache_0, V_cache_1, V_cache_2, V_cache_3, V_cache_4, V_cache_5,
                    V_cache_6, V_cache_7, V_cache_8, V_cache_9, V_cache_10, V_cache_11,
                    V_cache_12, V_cache_13, V_cache_14, V_cache_15, V_cache_16, V_cache_17,
                ]  # fmt: skip
                for _j, (_k, _v) in enumerate(zip(_k_bufs, _v_bufs, strict=True)):
                    if _k is None:
                        break
                    res[f"K_cache_{_j}"] = np.zeros_like(np.asarray(_k))
                    res[f"V_cache_{_j}"] = np.zeros_like(np.asarray(_v))
            else:
                res["K_cache"] = np.zeros_like(np.asarray(K_cache))
                res["V_cache"] = np.zeros_like(np.asarray(V_cache))
        return res

    return ref


# Per-rank acceptance on the real 2880 hidden cols. Cosine is the PRIMARY gate
# (direction of the token's output vector); rel-L2 is a loose magnitude sanity
# bound. Most ranks land cos>=0.999 / rel<=0.05 (MXFP4 quant noise). A few tokens
# (~3/32 at 2 layers) sit near a router top-k boundary and flip to a different
# expert set in the MXFP4 kernel vs the dense fp32 golden -> still cos~0.94 (same
# direction) but rel-L2 up to ~0.38 for that one token. That is a real, expected
# MXFP4-vs-dense routing effect at the quant boundary (not a kernel bug), so the
# rel-L2 bound is set to admit it while cosine still guarantees the output is not
# garbage/orthogonal. Matches the repo's MX-MoE cosine-first validation convention.
_COS_MIN = 0.90  # cosine-similarity floor (primary correctness gate)
_REL_L2_MAX = 0.45  # relative-L2 ceiling (loose; admits single-token router flips)


def _make_cosine_validator(golden_out, name="?"):
    """Per-rank validator: compare only the real (unpadded) hidden columns via
    cosine similarity + relative-L2, not strict elementwise allclose.

    Rationale (repo MX-MoE convention): the fused kernel is MXFP4 with in-matmul
    mxfp8 activation quant; the dense fp32 golden does not model that. Across
    stacked layers, a token whose layer-1 output lands near a router top-k boundary
    can pick a different expert set in the kernel vs the golden, changing many of
    that token's output columns — a real MXFP4-boundary effect, not a kernel bug.
    Strict per-element rel-diff explodes on those tokens (and on near-zero outputs),
    so we validate direction (cosine) + aggregate magnitude (rel-L2) on the real
    2880 columns, exactly as the repo's MX MoE / FP8-KV tests do.
    """
    g = np.asarray(golden_out)
    _gdtype = g.dtype
    _gshape = g.shape

    def _cos_rel(x, y):
        x = np.asarray(x).reshape(-1).astype(np.float64)
        y = np.asarray(y).reshape(-1).astype(np.float64)
        n = min(x.shape[0], y.shape[0], _H_ACTUAL)  # real (unpadded) hidden cols
        x, y = x[:n], y[:n]
        cos = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
        rel = float(np.linalg.norm(x - y) / (np.linalg.norm(y) + 1e-12))
        return cos, rel

    class _Cos(CustomValidator):
        def validate(self, actual):
            # The framework hands `actual` as a raw byte view; reinterpret it as the
            # golden's dtype/shape (same pattern as make_cosine_similarity_validator).
            av = np.asarray(actual)
            try:
                av = av.view(_gdtype).reshape(_gshape)
            except (ValueError, TypeError):
                pass
            a_full = np.asarray(av)
            b_full = g

            # For the [T_shard, H] "out" tensor, validate EVERY token (not just token 0 --
            # the old flattened check only saw the first _H_ACTUAL cols = token 0).
            if a_full.ndim == 2 and b_full.ndim == 2 and b_full.shape[0] > 1 and b_full.shape[1] >= _H_ACTUAL:
                T = b_full.shape[0]
                ok = True
                for t in range(T):
                    cos, rel = _cos_rel(a_full[t], b_full[t])
                    tok_ok = (cos >= _COS_MIN) and (rel <= _REL_L2_MAX)
                    ok = ok and tok_ok
                    self._print_with_log(
                        f"[mega cosine-validate:{name}] token={t} cosine={cos:.6f} (min {_COS_MIN}) "
                        f"relative_L2={rel:.4f} (max {_REL_L2_MAX}) -> {'PASS' if tok_ok else 'FAIL'}"
                    )
                return ok

            # Fallback (K/V caches, or single-token out): original aggregate check.
            cos, rel = _cos_rel(a_full, b_full)
            ok = (cos >= _COS_MIN) and (rel <= _REL_L2_MAX)
            self._print_with_log(
                f"[mega cosine-validate:{name}] cosine={cos:.6f} (min {_COS_MIN}) "
                f"relative_L2={rel:.4f} (max {_REL_L2_MAX}) -> {'PASS' if ok else 'FAIL'}"
            )
            return ok

    return CustomValidatorWithOutputTensorData(validator=_Cos, output_ndarray=g)


def _cosine_comparator(rank_id, golden_dict):
    """run_test custom_comparator: wrap each output key's golden in the cosine
    validator so the framework validates via cosine+relL2 on the real 2880 cols."""
    return {k: _make_cosine_validator(v, name=k) for k, v in golden_dict.items()}


def _make_passthrough_validator(golden_out):
    """Accept-all validator (VLLM_NEURON_MEGA_SKIP_GOLDEN): validate() always True.
    Paired with the zero stub ref so device run + profiling proceed without a real
    golden. Does NOT check correctness."""
    g = np.asarray(golden_out)

    class _Pass(CustomValidator):
        def validate(self, actual):
            self._print_with_log("[mega skip-golden] validation SKIPPED (perf/profile run)")
            return True

    return CustomValidatorWithOutputTensorData(validator=_Pass, output_ndarray=g)


def _passthrough_comparator(rank_id, golden_dict):
    return {k: _make_passthrough_validator(v) for k, v in golden_dict.items()}


@pytest_test_metadata(
    name="GPT-OSS-120B multi-layer mega decode profile"
    if _FULL_120B_PROFILE
    else "GPT-OSS-20B multi-layer mega decode vs golden"
)
@pytest_marks(["collectives"])
@pytest.mark.high_rank
@pytest.mark.skip_simulation
class TestGptOssMegaDecodeGolden:
    """Multi-layer GPT-OSS decode with DP/TP/EP collectives, GQA, SWA, and block KV."""

    # Not marked @pytest.mark.fast: the multi-layer mega decode exceeds the release
    # precommit's 3000 MB compile memory-limit. Runs on shared fleet instead.
    @pytest_parametrize(LAYER_PARAM_NAMES, LAYER_PARAMS, abbrevs=_ABBREVS)
    def test_mega_decode_vs_golden(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        giga_variant: str,
        num_layers: int,
        logical_nc_config: int,
    ):
        if not platform_target.is_trn3():
            pytest.skip("MXFP4 (float4_e2m1fn_x4) is only supported on TRN3.")
        if _FULL_120B_PROFILE and not _SKIP_GOLDEN:
            pytest.fail("VLLM_NEURON_MEGA_FULL_120B requires VLLM_NEURON_MEGA_SKIP_GOLDEN=1")
        if _SHARE_LAYER_INPUTS and not _SKIP_GOLDEN:
            pytest.fail("VLLM_NEURON_MEGA_SHARE_LAYER_INPUTS is only valid with skip-golden profiling")
        if _REUSE_INPUT_ARTIFACTS and not _SKIP_GOLDEN:
            pytest.fail("VLLM_NEURON_MEGA_REUSE_INPUT_ARTIFACTS is only valid with skip-golden profiling")
        if _FUSE_EMBED and _X_FIRST_FULL:
            pytest.fail(
                "VLLM_NEURON_MEGA_FUSE_EMBED and VLLM_NEURON_MEGA_X_FIRST_FULL are "
                "mutually exclusive; both set layer 0's input."
            )
        if _KV_INDIVIDUAL and num_layers > 18:
            pytest.fail("VLLM_NEURON_MEGA_KV_INDIVIDUAL supports at most 18 buffers (num_layers <= 18)")

        tp_subgroups = [list(range(d * _TP, (d + 1) * _TP)) for d in range(_DP)]
        tp_group = ReplicaGroup(tp_subgroups)
        ep_group = ReplicaGroup([list(range(_RANKS))])
        # Tokens per rank. NOTE: raising this does NOT reduce per-layer latency -- T_ep = T_shard *
        # RANKS, so the EP collective payloads grow linearly with it (RS-EP 20.3 -> 32.9 -> 58.2 us
        # modelled for T_shard 1/2/4) while matmul efficiency improves only sublinearly. It lowers
        # cost PER TOKEN (27.9 -> 20.3 -> 16.5 us of CC per token), i.e. it is a THROUGHPUT lever,
        # not a latency one. Left env-tunable for throughput studies.
        T_shard = int(os.environ.get("VLLM_NEURON_MEGA_T_SHARD", "1"))
        B_attn = T_shard * _TP  # 8 tokens/DP replica
        softmax_scale = float(_D_HEAD**-0.5)
        layer_input_count = 1 if _SHARE_LAYER_INPUTS else num_layers

        input_cache = None
        if _INPUT_CACHE_DIR:
            source_files = [
                source
                for source in (
                    __file__,
                    inspect.getsourcefile(generate_inputs),
                    inspect.getsourcefile(generate_kernel_inputs),
                )
                if source is not None
            ]
            cache_key = build_input_cache_key(
                {
                    "tag": _INPUT_CACHE_TAG,
                    "num_layers": num_layers,
                    "layer_input_count": layer_input_count,
                    "logical_nc_config": logical_nc_config,
                    "dp": _DP,
                    "tp": _TP,
                    "ranks": _RANKS,
                    "experts": _N_EXPERTS,
                    "hidden": _H,
                    "hidden_actual": _H_ACTUAL,
                    "intermediate": _I,
                    "head_dim": _D_HEAD,
                    "q_heads_per_rank": _Q_HEADS,
                    "kv_heads_per_rank": _KV_HEADS,
                    "context": _S_CTX,
                    "block_len": _BLOCK_LEN,
                    "sliding_window": _SLIDING_WINDOW,
                    "moe_seed": _MOE_SEED,
                    # T_shard sets B_attn, so it changes every per-rank input shape. Omitting it
                    # made a T_shard=2 run silently load the T_shard=1 cache and die at trace with
                    # "cannot reshape tensor of size 8 (shape (8,1)) into shape (1,16)".
                    "t_shard": T_shard,
                    "fp8_kv": _FP8_KV,
                    "kv_quant_dtype": str(_KV_QUANT_DTYPE) if _FP8_KV else None,
                    "fuse_embed": _FUSE_EMBED,
                    "fuse_embed_hidden": _FUSE_EMBED_HIDDEN,
                    "host_mask": _HOST_MASK,
                    "fuse_rope": _FUSE_ROPE,
                },
                source_files,
            )
            input_cache = PersistentPerRankInputCache(_INPUT_CACHE_DIR, cache_key)
            logging.info("GPT-OSS mega input cache namespace: %s", input_cache.directory)

        # Per-layer global MXFP4 MoE weight sets (E=_N_EXPERTS, sliced per rank).
        # generate_inputs seeds internally (np.random.seed(42)), so the MoE weights
        # are identical across layers; that is fine for a correctness test (the
        # golden ref uses the same per-layer weights). The default correctness mode
        # still gives layers distinct attention inputs; shared-input profiling
        # intentionally reuses layer 0's tensors while preserving all 36 executions.
        # MoE weights are identical across layers (generate_inputs is seeded); build
        # ONCE and reuse. Keep this lazy so a complete persistent-cache hit never
        # generates the large global expert set.
        mg_shared = None

        def get_moe_inputs():
            nonlocal mg_shared
            if mg_shared is None:
                # Seed immediately before generation so this matches
                # _regen_moe_global. MXFP4 generation draws from the global RNG
                # before generate_inputs applies its internal seed.
                np.random.seed(_MOE_SEED)
                mg_shared = generate_inputs(
                    # MUST match the golden's generate_inputs(batch=_RANKS) call: generate_inputs
                    # draws inputs["inp"] (size batch*seqlen*hidden) from the same internal
                    # default_rng(42) BEFORE router_weights/expert weights, so a T-dependent batch
                    # (B_g = T_shard*_RANKS) drifts the RNG and yields DIFFERENT MoE weights than the
                    # golden at T_shard>1 -> kernel routes/computes with mismatched weights -> spurious
                    # accuracy failure. `inp` is unused on the kernel side (hidden comes from attention),
                    # so batch only needs to align the RNG with the golden. See _regen_moe_global / L458.
                    batch=_RANKS,
                    seqlen=1,
                    hidden=_H,
                    hidden_actual=_H_ACTUAL,
                    intermediate=_I,
                    num_global_experts=_N_EXPERTS,
                    num_local_experts=_N_EXPERTS,
                    top_k=_TOP_K,
                    router_fn=None,
                    hidden_act_fn=None,
                    expert_affinities_scaling_mode=None,
                    moe_weight_dtype=_MX,
                    input_dtype=_DTYPE,
                    has_bias=True,
                    has_clamp=True,
                    router_act_first=False,
                    norm_topk_prob=False,
                    skip_router_logits=True,
                    router_mm_dtype=nl.float16,
                    is_all_expert=True,
                )
                mg_shared["expert_gate_up_bias"][:, :, 1, ...] += 1.0
                _zero_moe_hidden_padding(mg_shared)
            return mg_shared

        def generate_array_inputs(rank_id: int):
            nonlocal mg_shared
            dp_replica = rank_id // _TP
            tp_rank = rank_id % _TP
            e = rank_id  # this rank's global expert index (EP)
            qd_r = _Q_HEADS * _D_HEAD
            q_all = _FULL_Q_HEADS * _D_HEAD
            kv_r = _KV_HEADS * _D_HEAD  # W_qkv K/V columns for this rank's KV heads (TP8 -> _D_HEAD)
            moe_inputs = get_moe_inputs()

            # Per-layer lists.
            (iln, wqkv, bqkv, wout, bout, sink, kcache, vcache, paln, rw, rb, egu, edn, egus, edns, egub, edb, swa) = (
                [] for _ in range(18)
            )
            for li in range(layer_input_count):
                sliding = _SLIDING_WINDOW if _is_sliding(li) else 0
                # Seed per (layer, replica) so layers/replicas are distinct but all
                # TP ranks in a replica share the SAME full attention set. Memoized on
                # (seed, sliding, B_attn) so the 8 TP ranks of a replica reuse one build.
                aki = _gen_attn_inputs(_attn_seed(dp_replica, li), sliding, B_attn)

                X_full = aki["X"].reshape(B_attn, _H)
                if li == 0:
                    x_shard = X_full[tp_rank * T_shard : (tp_rank + 1) * T_shard].copy()
                    # Layer-0 pre-gathered [T_attn, H] input for the AllGather-skip path
                    # (VLLM_NEURON_MEGA_X_FIRST_FULL). This is exactly what AllGather[TP]
                    # of x_shard reconstructs: every TP rank in a replica shares this X_full,
                    # and T_attn == B_attn (S_tkg=1) == TP * T_shard.
                    x_first_full_arr = np.ascontiguousarray(X_full.astype(x_shard.dtype)) if _X_FIRST_FULL else None
                    embed_x_shard_ids_r = None
                    if _FUSE_EMBED and _FUSE_EMBED_HIDDEN:
                        # OPTION (b): shard the embedding by HIDDEN dim. Rank r holds
                        # [vocab, H/TP] (its H-slice of every vocab row) and looks up
                        # GLOBAL token ids (no masking, no cross-rank sum). The SBUF
                        # all_gather[dim=1] concatenates the tp_size slices in rank
                        # order to rebuild X_full[t] exactly. embed_x_shard_ids gives
                        # this rank's token rows for the SP-shard slice.
                        assert _H % _TP == 0
                        h_slice = _H // _TP
                        embed_vocab = 2 * B_attn + 2
                        _ids = (np.arange(B_attn) * 2 + 1).astype(np.int32)  # distinct
                        embed_weight_r = np.zeros((embed_vocab, h_slice), dtype=x_shard.dtype)
                        for _t in range(B_attn):
                            embed_weight_r[_ids[_t]] = X_full[_t, tp_rank * h_slice : (tp_rank + 1) * h_slice]
                        embed_weight_r = np.ascontiguousarray(embed_weight_r)
                        embed_ids_r = np.ascontiguousarray(_ids)  # SAME on all ranks
                        embed_x_shard_ids_r = np.ascontiguousarray(
                            (tp_rank * T_shard + np.arange(T_shard)).astype(np.int32)
                        )
                        x_shard = np.zeros((T_shard, _H), dtype=x_shard.dtype)
                    elif _FUSE_EMBED:
                        # OPTION (a): shard the embedding by VOCAB dim. Per-rank table
                        # + local indices whose gather + TP-reduce rebuilds X_full.
                        # owner(t) = t % TP, local(t) = (t // TP) * 3 + 1 (a distinct,
                        # nonzero index per rank, to exercise the gather offset).
                        embed_vocab = 4 * T_shard + 4  # per-rank shard rows (> max local)
                        _owner = np.arange(B_attn) % _TP
                        _local = (np.arange(B_attn) // _TP) * 3 + 1
                        embed_weight_r = np.zeros((embed_vocab, _H), dtype=x_shard.dtype)
                        embed_ids_r = np.full((B_attn,), -1, dtype=np.int32)
                        for _t in range(B_attn):
                            if _owner[_t] == tp_rank:
                                embed_weight_r[_local[_t]] = X_full[_t]
                                embed_ids_r[_t] = _local[_t]
                        embed_weight_r = np.ascontiguousarray(embed_weight_r)
                        embed_ids_r = np.ascontiguousarray(embed_ids_r)
                        # Zero x_shard: the fused prologue must be the ONLY source of
                        # layer 0's input. A prologue that wrongly reused x_shard would
                        # then corrupt residual add #1 and fail the cosine gate.
                        x_shard = np.zeros((T_shard, _H), dtype=x_shard.dtype)

                # W_qkv: this rank's Q-head cols (_Q_HEADS*D) + its KV-head span K/V cols
                # (_KV_HEADS*D each). At TP8 that is 8 Q-heads + 1 KV head; at TP4, 16 + 2.
                Wqkv_full = aki["W_qkv"]  # [H, 64*D + 2*(8*D)]
                kbase = q_all
                vbase = q_all + _FULL_KV_HEADS * _D_HEAD
                Wqkv_r = np.ascontiguousarray(
                    np.concatenate(
                        [
                            Wqkv_full[:, tp_rank * qd_r : (tp_rank + 1) * qd_r],
                            Wqkv_full[:, kbase + tp_rank * kv_r : kbase + (tp_rank + 1) * kv_r],
                            Wqkv_full[:, vbase + tp_rank * kv_r : vbase + (tp_rank + 1) * kv_r],
                        ],
                        axis=1,
                    )
                )
                Wout_r = np.ascontiguousarray(aki["W_out"][tp_rank * qd_r : (tp_rank + 1) * qd_r, :])
                sink_r = np.ascontiguousarray(aki["sink"][tp_rank * _Q_HEADS : (tp_rank + 1) * _Q_HEADS, :])
                # This rank's _KV_HEADS KV heads. Slice only the KV-head axis (dim 1) so both packed
                # and ordinary layouts are accepted; result [nb, _KV_HEADS, bl, d(,2)].
                Kc_full, Vc_full = aki["K_cache"], aki["V_cache"]
                _kh = slice(tp_rank * _KV_HEADS, (tp_rank + 1) * _KV_HEADS)
                Kc_r = np.ascontiguousarray(Kc_full[:, _kh])
                Vc_r = np.ascontiguousarray(Vc_full[:, _kh])
                # active_blocks_table: [B, 8, num_blocks] holds GLOBAL block indices into the
                # flattened num_blocks*_FULL_KV_HEADS pool (value = base*8 + head), base =
                # global // _FULL_KV_HEADS. attention_block_tkg squeezes this rank's cache
                # HEAD-INNER to [nb*_KV_HEADS, bl, d] (merged_block = base*_KV_HEADS + local_head) and
                # the READ gathers with the table value used DIRECTLY as the physical block index
                # (only scaled by the block-len resize_factor, never by kv_heads). So the table must
                # carry the MERGED block index base*_KV_HEADS + local_head — NOT the head-independent
                # base (that only works at TP8 where _KV_HEADS==1 -> base*1+0==base). Feed [B, nb] when
                # _KV_HEADS==1, else the [B, _KV_HEADS, nb] form attention_block_tkg asserts.
                abt_full = aki["active_blocks_table"]  # [B, 8, num_blocks]
                if _KV_HEADS == 1:
                    abt_head = abt_full[:, tp_rank, :]  # [B, num_blocks], values base*8+tp_rank
                    abt_r = np.ascontiguousarray(np.where(abt_head < 0, abt_head, abt_head // _FULL_KV_HEADS))
                else:
                    abt_r = np.ascontiguousarray(
                        np.stack(
                            [
                                np.where(
                                    abt_full[:, tp_rank * _KV_HEADS + h, :] < 0,
                                    abt_full[:, tp_rank * _KV_HEADS + h, :],
                                    (abt_full[:, tp_rank * _KV_HEADS + h, :] // _FULL_KV_HEADS) * _KV_HEADS + h,
                                )
                                for h in range(_KV_HEADS)
                            ],
                            axis=1,
                        )
                    )  # [B, _KV_HEADS, nb], merged block indices (base*_KV_HEADS + local_head)
                mask_full = aki["attention_mask"]  # [S_tkg-ish, B, 64, S_tkg] head-broadcast
                mask_r = np.ascontiguousarray(mask_full[:, :, tp_rank * _Q_HEADS : (tp_rank + 1) * _Q_HEADS, :])
                # kv_cache_update_idx: aki gives [B, 8, S_tkg] GLOBAL slots (base*8+head)*bl+offset.
                # KV UPDATE: build this rank's MERGED per-head write index [B, _KV_HEADS, S_tkg]. The
                # kernel squeezes its cache HEAD-INNER to [nb*_KV_HEADS, bl, d] and scatters the index
                # directly, so each local head h needs (base*_KV_HEADS + h)*bl + offset — NOT the
                # head-independent base the block table uses (see _localize_update_idx). A wrong slot
                # addresses out of bounds -> scatter skipped/clamped and fp8 gather-modify-scatter
                # desyncs ranks -> collective hang. Read-only (update off) keeps a 3D [B, 1, S_tkg]
                # slice (unused by the read path, but preserves the validated trace shape).
                upd_full = aki["kv_cache_update_idx"]
                if upd_full.ndim == 3:
                    if _KV_UPDATE:
                        upd_r = np.stack(
                            [
                                _localize_update_idx(
                                    upd_full[:, tp_rank * _KV_HEADS + h, :],
                                    _BLOCK_LEN,
                                    _FULL_KV_HEADS,
                                    _KV_HEADS,
                                    h,
                                )
                                for h in range(_KV_HEADS)
                            ],
                            axis=1,
                        )  # [B, _KV_HEADS, S_tkg]
                        # The block-KV write path wants the kv_heads==1 form [B, S_tkg] when a rank
                        # owns a single head; kv_heads>1 keeps [B, _KV_HEADS, S_tkg] (kernel folds it
                        # to [B*_KV_HEADS, S_tkg]).
                        if _KV_HEADS == 1:
                            upd_r = upd_r[:, 0, :]
                        upd_r = np.ascontiguousarray(upd_r)
                    else:
                        upd_r = np.ascontiguousarray(upd_full[:, tp_rank * _KV_HEADS : tp_rank * _KV_HEADS + 1, :])
                else:
                    upd_r = upd_full
                # bias_qkv/bias_out are None here (test_bias=False). Can't stack None,
                # and the kernel expects a real tensor per layer -> pass zeros (zero
                # bias == no bias; the ref also uses zeros, so it's exact).
                qkv_cols = _Q_HEADS * _D_HEAD + 2 * _KV_HEADS * _D_HEAD
                bqkv_r = np.zeros((1, qkv_cols), dtype=np.float32).astype(_np_dtype())
                bout_r = np.zeros((1, _H), dtype=np.float32).astype(_np_dtype())

                iln.append(aki["rmsnorm_X_gamma"])
                wqkv.append(Wqkv_r)
                bqkv.append(bqkv_r)
                wout.append(Wout_r)
                bout.append(bout_r)
                sink.append(sink_r)
                kcache.append(Kc_r)
                vcache.append(Vc_r)
                mg = moe_inputs
                paln.append(mg["gamma"])
                rw.append(mg["router_weights"])
                rb.append(mg["router_bias"])
                egu.append(mg["expert_gate_up_weights"][e : e + 1])
                edn.append(mg["expert_down_weights"][e : e + 1])
                egus.append(mg["expert_gate_up_weights_scale"][e : e + 1])
                edns.append(mg["expert_down_weights_scale"][e : e + 1])
                egub.append(mg["expert_gate_up_bias"][e : e + 1])
                edb.append(mg["expert_down_bias"][e : e + 1])
                # Shared (layer-invariant) attention control taken from layer 0.
                if li == 0:
                    cos0, sin0 = aki["cos"], aki["sin"]
                    abt0, mask0, upd0, pos0 = abt_r, mask_r, upd_r, aki["pos_ids"]
                    if _FP8_KV:
                        k_scale0 = np.ascontiguousarray(aki["k_scale"])
                        v_scale0 = np.ascontiguousarray(aki["v_scale"])
                    inv_freq_arr = None
                    if _FUSE_ROPE:
                        # Override cos/sin with position-derived tables and pass
                        # inv_freq + concentration; the kernel recomputes them on-chip.
                        half_d = _D_HEAD // 2
                        inv_freq_np = 1.0 / (10000.0 ** (np.arange(0, half_d, dtype=np.float32) / half_d))
                        _pos = np.asarray(pos0, dtype=np.float32).reshape(-1)  # [T]
                        _freqs = _pos[:, None] * inv_freq_np[None, :]  # [T, half_d]
                        _cos = np.cos(_freqs) * _FUSE_ROPE_CONC  # [T, half_d]
                        _sin = np.sin(_freqs) * _FUSE_ROPE_CONC
                        # kernel/golden cos/sin layout: [half_d, B_attn, S_tkg]
                        cos0 = np.ascontiguousarray(
                            _cos.reshape(B_attn, _S_TKG, half_d).transpose(2, 0, 1).astype(cos0.dtype)
                        )
                        sin0 = np.ascontiguousarray(
                            _sin.reshape(B_attn, _S_TKG, half_d).transpose(2, 0, 1).astype(sin0.dtype)
                        )
                        inv_freq_arr = np.ascontiguousarray(inv_freq_np.reshape(half_d, 1).astype(np.float32))

            # SWA block-table trimming (matches production decode; see the mega
            # kernel docstring). Sliding (even) layers read only the window's blocks:
            # trim the shared full table + shift the shared positions into the window
            # frame. Both the kernel and the golden ref use these SAME shared trimmed
            # tensors for sliding layers, so they read identical physical blocks of
            # each layer's own KV cache -> bit-consistent, while touching only
            # ``num_swa`` blocks (2 at block_len 128 / window 128) instead of nbf (80).
            abt0_swa, pos0_swa = _trim_block_table_and_positions(
                np.asarray(abt0), np.asarray(pos0).astype(np.int64), _SLIDING_WINDOW, _BLOCK_LEN
            )
            # Per-layer SWA window start, expressed in the frame each layer uses.
            # This remains a small num_layers stack even when large tensors share
            # one leading-axis entry, because sliding and full layers differ.
            #  * sliding (even) layer -> trimmed frame: clamp(pos_swa - window + 1, 0)
            #  * full   (odd)  layer -> zeros (start 0 == full causal; ignored, the
            #    full path passes swa_start_pos_ids=0 as before).
            _start_swa = np.clip(pos0_swa - _SLIDING_WINDOW + 1, 0, None).astype(np.float32)
            _zero_start = np.zeros((B_attn, _S_TKG), dtype=np.float32)
            swa = [np.ascontiguousarray(_start_swa if _is_sliding(li) else _zero_start) for li in range(num_layers)]

            def _stack(lst):
                return np.ascontiguousarray(np.stack(lst, axis=0))

            arrays = {
                "x_shard": x_shard,
                # Per-layer weights STACKED on a leading layer axis (NKI rejects
                # list kernel args; HBM integer-index removes dim 0 -> layer i).
                "input_layernorm_weight": _stack(iln),
                "W_qkv": _stack(wqkv),
                "bias_qkv": _stack(bqkv),
                "W_out": _stack(wout),
                "bias_out": _stack(bout),
                "sink": _stack(sink),
                "K_cache": _stack(kcache),
                "V_cache": _stack(vcache),
                "post_attn_layernorm_weight": _stack(paln),
                "router_weights": _stack(rw),
                "router_bias": _stack(rb),
                "expert_gate_up_weights": _stack(egu),
                "expert_down_weights": _stack(edn),
                "expert_gate_up_weights_scale": maybe_pack64_gate_up_scale(_stack(egus)),
                "expert_down_weights_scale": _stack(edns),
                "expert_gate_up_bias": _stack(egub),
                "expert_down_bias": _stack(edb),
                "swa_start_pos_ids": _stack(swa),
                "cos": cos0,
                "sin": sin0,
                "active_blocks_table": abt0,
                "attention_mask": mask0,
                "active_blocks_table_swa": abt0_swa,
                "kv_cache_update_idx": upd0,
                "pos_ids": pos0,
                "pos_ids_swa": pos0_swa,
                "rank_id": np.array([[rank_id]], dtype=np.uint32),
            }
            if _FP8_KV:
                arrays["k_scale"] = k_scale0
                arrays["v_scale"] = v_scale0
            if _X_FIRST_FULL:
                # Feed the pre-gathered layer-0 input; the kernel skips its AllGather.
                arrays["x_first_full"] = x_first_full_arr
            if _FUSE_EMBED:
                # Feed the embedding table + indices; the kernel builds layer 0's
                # input in-NEFF. Option (a): vocab-shard -> reduce_scatter + all_gather.
                # Option (b): hidden-shard -> one SBUF all_gather[dim=1] + the
                # embed_x_shard_ids slice.
                arrays["embed_weight"] = embed_weight_r
                arrays["embed_ids"] = embed_ids_r
                if embed_x_shard_ids_r is not None:
                    arrays["embed_x_shard_ids"] = embed_x_shard_ids_r
            if _HOST_MASK:
                # Build the full + SWA masks on the HOST and pass them, so the
                # kernel loads them and runs NO on-chip gen_mask_tkg_hbm (matches a
                # real deployment where the runner builds the mask). Uses the SAME
                # shared layer-0 control (pos, window-frame pos, swa start, tables,
                # active mask) the kernel's hoist would use.
                mask_full_h, mask_swa_h = _build_host_masks(
                    pos_full=pos0,
                    pos_swa=pos0_swa,
                    start_swa=_start_swa,
                    abt_full=abt0,
                    abt_swa=abt0_swa,
                    active_mask=mask0,
                    block_len=_BLOCK_LEN,
                    q_head=_Q_HEADS,
                    s_tkg=_S_TKG,
                    b_attn=B_attn,
                    d_head=_D_HEAD,
                    fp8_packed=_FP8_KV,
                    kv_heads=_KV_HEADS,
                )
                arrays["pregen_mask_full"] = mask_full_h
                arrays["pregen_mask_swa"] = mask_swa_h
            if _FUSE_ROPE and not _FUSE_ROPE_INPUT_ONLY:
                # Feed inv_freq; the kernel computes cos/sin on-chip (rope fused).
                # In INPUT_ONLY diagnostic mode, skip inv_freq so the kernel uses the
                # numpy-accurate position-derived cos/sin tables as plain inputs (no
                # on-chip _rope) -> isolates on-chip RoPE precision from the values.
                arrays["inv_freq"] = inv_freq_arr
            if _KV_INDIVIDUAL:
                # Pass the per-buffer caches as SEPARATE kernel tensors (HMA interface)
                # with the stacked K_cache/V_cache set to None. len(kcache) == num_layers
                # here (SHARE_LAYER_INPUTS off), so the kernel routes per-layer (buffer i
                # -> layer i), reproducing the per-layer golden exactly.
                arrays["K_cache"] = None
                arrays["V_cache"] = None
                for _j in range(len(kcache)):
                    arrays[f"K_cache_{_j}"] = kcache[_j]
                    arrays[f"V_cache_{_j}"] = vcache[_j]
            if rank_id == _RANKS - 1:
                mg_shared = None
            return arrays

        def create_inputs(rank_id: int):
            if input_cache is None:
                arrays = generate_array_inputs(rank_id)
            else:
                arrays = input_cache.get_or_create(rank_id, lambda: generate_array_inputs(rank_id))

            result = {
                **arrays,
                "tp_replica_group": tp_group,
                "ep_replica_group": ep_group,
                "num_layers": num_layers,
                "tp_size": _TP,
                "ep_size": _RANKS,
                "B_attn": B_attn,
                "S_tkg": _S_TKG,
                "top_k": _TOP_K,
                "eps": _EPS,
                "hidden_actual": _H_ACTUAL,
                "softmax_scale": softmax_scale,
                "gate_clamp_upper_limit": _GATE_CLAMP_UPPER,
                "up_clamp_upper_limit": _UP_CLAMP_UPPER,
                "up_clamp_lower_limit": _UP_CLAMP_LOWER,
            }
            if _FUSE_ROPE:
                result["rope_concentration"] = float(_FUSE_ROPE_CONC)
            if _FP8_KV:
                # The kernel receives an explicit softmax scale, so it must
                # include the inverse K quantization scale. Read the scale from
                # `arrays` so cache hits follow the exact same path as misses.
                k_scale_scalar = float(np.asarray(arrays["k_scale"]).reshape(-1)[0])
                result["softmax_scale"] = softmax_scale / k_scale_scalar
            if _KV_UPDATE:
                # Aliasing: the kernel writes K/V in-place and returns them; the ".must_alias_input"
                # suffix tells the harness to alias those outputs onto these input buffers (the
                # kernel is called with the suffix stripped). The golden compares the post-update
                # caches, so they become validated outputs (output_keys below).
                if _KV_INDIVIDUAL:
                    # Per-buffer interface: the kernel returns the buffers it was given
                    # (x, *K_cache_i, *V_cache_i), so each per-buffer tensor gets its own alias.
                    # Validation matches outputs to goldens BY NAME, so alias order is irrelevant.
                    # The stacked K_cache/V_cache entries stay (as None) -- they are what selects
                    # this interface in the kernel, and both kernel and ref take them positionally.
                    for _key in _indiv_kv_keys(result):
                        result[f"{_key}.must_alias_input"] = result.pop(_key)
                else:
                    result["K_cache.must_alias_input"] = result.pop("K_cache")
                    result["V_cache.must_alias_input"] = result.pop("V_cache")
            return result

        # Perf/profiling mode: skip the expensive golden (stub ref = instant) and
        # accept-all validation, so the kernel still compiles + runs on device +
        # profiles without the ~40-min golden pass. Correctness is NOT checked here.
        if _SKIP_GOLDEN:
            torch_ref = make_stub_ref()
            comparator = _passthrough_comparator
        else:
            torch_ref = make_golden_ref(tp_group, ep_group, _TP, _RANKS, num_layers)
            comparator = _cosine_comparator
            # Prime the per-replica out_full cache in THIS (parent) process so the
            # ranks SimDistRunner forks inherit it copy-on-write (DP forwards total
            # instead of one per rank). Skip in compile-only/trace-only, where the
            # framework never generates the golden — priming would waste the compute.
            _is_compile_only = getattr(test_manager, "trace_mode", None) in (TraceMode.CompileOnly, TraceMode.TraceOnly)
            if not _is_compile_only:
                torch_ref.prime(create_inputs)

        # Build compiler args. MEGA_EXTRA_CC_ARGS is whitespace-split (space-free
        # flags only).
        _extra_cc_args = ["--enable-ocp-compliant-scale-computation"] + os.environ.get("MEGA_EXTRA_CC_ARGS", "").split()
        # The pinned DMA order, rewritten to this graph's kernel-call scope. The file goes to the
        # backend by PATH (gk_dma_schedule.compiler_args_for_schedule): inline, the JSON content
        # would be one argv element and blow MAX_ARG_STRLEN.
        _schedule_src = _schedule_for(giga_variant, num_layers)
        _schedule = None
        if _schedule_src is not None:
            _schedule = _gk_sched.materialize_schedule(_SCHEDULE_K, source=_schedule_src)
            _extra_cc_args += _gk_sched.compiler_args_for_schedule(_schedule)
            logging.info("[GK-SCHED] applying %s as %s (K=%d)", _schedule_src, _schedule, _SCHEDULE_K)

        # MEGA_BACKEND_OPTS: pass an arbitrary backend-options string (may
        # contain spaces, e.g. multiple --enable-narwhal-m2-feature flags) as ONE
        # argv element. MEGA_EXTRA_CC_ARGS can't carry it -- it's whitespace-split
        # and would shatter the space-separated backend flags.
        _backend_opts = os.environ.get("MEGA_BACKEND_OPTS", "").strip()
        if _backend_opts:
            _extra_cc_args.append(f"--internal-backend-options={_backend_opts}")

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_load_mega_kernel(giga_variant),
            torch_ref=torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=_RANKS,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                logical_nc_config=logical_nc_config,
                platform_target=platform_target,
                nki_compilation_mode=NKICompilationMode.tracer,
                additional_cmd_args=_extra_cc_args,
            ),
            output_keys=_mega_output_keys(layer_input_count),
            rtol=7e-2,
            atol=1e-2,
            custom_comparator=comparator,  # cosine+relL2 (or accept-all if skip-golden)
            inference_args=InferenceArgs(
                collective_ranks=_RANKS,
                enable_determinism_check=False,
                profile_all_ranks=_env_enabled("VLLM_NEURON_MEGA_PROFILE_ALL_RANKS"),
                env_vars=_WORKER_ENV or None,
            ),
            profile_only=_SKIP_GOLDEN,
            input_artifacts_directory=_REUSE_INPUT_ARTIFACTS or None,
        )

        # A stale scope compiles clean with NO order applied, so the compile log is the only
        # witness that the NEFF this run measured carries the schedule.
        if _schedule is not None:
            _gk_sched.assert_schedule_bound(
                _gk_sched.compile_log_path(test_manager.get_test_artifact_output_path()),
                _SCHEDULE_K,
                schedule_path=_schedule,
            )

    @pytest_parametrize(LAYER_PARAM_NAMES, [(g, 2, 1) for g in _GIGA_SEL], abbrevs=_ABBREVS)
    def test_mega_decode_individual_kv_vs_golden(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        giga_variant: str,
        num_layers: int,
        logical_nc_config: int,
        monkeypatch,
    ):
        """Golden accuracy for the per-buffer (non-stacked) KV interface, as deployed.

        Pins the configuration the vLLM DI deployment actually runs: the per-buffer HMA
        caches (stacked ``K_cache=None``, ``K_cache_0..`` set) + PACKED FP8 KV + in-place
        ``KV_UPDATE``. The default (stacked, bf16) cases drive none of that, so without this
        node the shipped path has no standing coverage.

        What each pin buys:
        - ``KV_INDIVIDUAL``: exercises the kernel's ``K_cache is None`` detection and the
          per-buffer routing.
        - ``FP8_KV``: the deployed cache format. A per-buffer packed K is 5D
          ``[nb, kv, block_len//2, d, 2]`` (one axis fewer than the 6D stacked form), so the
          golden's packed-repack detection is per-buffer-aware; getting it wrong compares
          packed against unpacked and the caches come out garbage rather than exact.
        - ``KV_UPDATE`` (already default-on): each layer writes its decode token's K/V into
          its own buffer and the kernel returns those buffers, so each is aliased onto its
          input and the golden validates every post-update buffer BY NAME -- the in-place
          write-back the stacked case checks, on the per-buffer interface.

        Restricted to L-2 (individual supports <=18 buffers; L-2 covers a SWA + a full layer).

        Each knob is set two ways: ``setenv`` because the KERNEL reads these flags from the
        env at trace time (and it feeds any re-imported worker process), ``setattr`` because
        they are import-time module constants this test module already bound, which
        ``create_inputs``/the golden ref read in-process. ``monkeypatch`` auto-reverts both,
        so nothing leaks to the stacked cases in a shared process.
        """
        mod = sys.modules[__name__]
        monkeypatch.setenv("VLLM_NEURON_MEGA_KV_INDIVIDUAL", "1")
        monkeypatch.setattr(mod, "_KV_INDIVIDUAL", True)
        monkeypatch.setenv("VLLM_NEURON_MEGA_FP8_KV", "1")
        monkeypatch.setattr(mod, "_FP8_KV", True)
        self.test_mega_decode_vs_golden(
            test_manager=test_manager,
            platform_target=platform_target,
            giga_variant=giga_variant,
            num_layers=num_layers,
            logical_nc_config=logical_nc_config,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-x"])
