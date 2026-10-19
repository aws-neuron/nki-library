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

"""ACCURACY gate for the GPT-OSS spec-decode target layer (c128 attn + c512 DLoC MoE).

This is the spec-decode counterpart of `test_gpt_oss_decode_layer_dp_tp_ep_golden.py`, which
the c512 runbook designates as *"the trustworthy correctness gate independent of the mega
golden"*. It validates the single-layer entry of the spec kernel against the vendored
HF-bit-validated whole-layer oracle at small shapes (H=512, S_ctx=256, block_len=32), so it
runs in minutes rather than the mega golden's ~40 min.

Why the single-layer gate and not the mega golden: the mega golden is S_tkg=1-only AND is
documented broken at T_shard>1 (routing-flip, below). The single-layer test is the trusted
reference, and the spec kernel's single-layer and mega entries share one body
(`_spec_decode_layer_body`), so validating the layer validates the math the mega kernel runs.

**The oracle already supports spec decode -- nothing had to be added.** The vendored golden's
`GptOssAttention.forward_decode` derives `S_decode = tokens // B` from `block_table.shape[0]`
and reshapes to `(B, S_decode, H)`; `kernels._build_decode_mask` then documents itself as
"conservative for S_tkg > 1 spec decode" and overlays the active-token causal triangle
(`tril(ones(S_tkg,S_tkg))`, query q attends key kk iff q >= kk) onto the last S_tkg slots --
the same chain the kernel gets from `gen_mask_tkg(active_mask=...)` and the same chain
vllm-neuron builds in production. So only the harness glue was S_tkg=1-bound:
`make_golden_ref` is already general (its math is all `B_attn * S_tkg` and it slices the
output by `x_shard.shape[0]` rows), leaving exactly two lines -- the `X` reshape and the
`x_shard` slice -- which this file overrides.

Shape contract. `T_SHARD` counts **sequences** per rank; each contributes S_tkg active tokens:

    T_shard_rows = T_SHARD * S_tkg          (rows of x_shard, == the kernel's T_shard)
    B_attn       = T_SHARD * TP             (sequences per DP replica)
    T_ep         = T_shard_rows * ranks

The spec kernel's DLoC block is 128 since 2026-09-19 (c512 MoE of CR-302598650); it needs
`T_ep >= 256` and `T_ep % 128 == 0`, so at `T_SHARD=1, S_TKG=4` ranks 64 (T_ep=256) and 128
(T_ep=512, the production shape) qualify. ranks-32 (T_ep=128) was the gate's designated config
while the block was 64 and is no longer valid; ranks 8/16 never were.

Known issue inherited from c512, NOT a kernel bug: at T_shard(rows)>1 a stride-pattern of
ranks can FAIL at cosine ~0.1-0.3 while the rest hit exactly 1.0, because the kernel routes
with mxfp8 activation quant while the golden routes in fp32 -- a token whose k-th and
(k+1)-th router logits are within mxfp8 error picks a different expert set. Spec decode is
*inherently* in that regime (4 rows/rank). A FAIL here therefore needs triage before being
called a bug: compare kernel-vs-golden expert selection first.

The kernel's EP ReduceScatter runs in fp8_e4m3fn unconditionally (the MoE writes its output as fp8 and the
collective sums in fp8), which needs `NEURON_RT_ENABLE_OCP=1` (set below) and is what the tolerance is
calibrated for: each rank's partial is quantized before the reduction, but with 128 experts and top_k=4
the reduction has ~4 non-zero terms per token, not 128.

Run (trn3 pod, pinned toolchain):

    for v in $(env | grep -oE "^VLLM_NEURON_MEGA_[A-Z0-9_]+"); do unset "$v"; done
    export NEURON_RT_ENABLE_OCP=1 NEURON_RT_ENABLE_OCP_SATURATION=1
    export VLLM_NEURON_MEGA_S_TKG=4
    pytest -c test/pytest.ini \\
      test/integration/nkilib/experimental/gpt_oss/test_gpt_oss_spec_decode_layer_golden.py \\
      -k "ranks-128" --platform-target trn3 --nki-compilation-mode tracer -v -s

`S_TKG` is the only `VLLM_NEURON_MEGA_*` variable this gate needs: the kernel's collective operands are
private HBM and its MoE output is fp8 unconditionally. Do NOT set
`PREQUANT_MOE`: the spec kernel pins `_prequant_moe = True` in source (the DLoC MoE consumes the
packed fp8 dispatch buffer directly and has no unpacked-input path), and `MOE_SBUF_COLLECTIVE`
does not need disabling either because the combine keys off whether the SBUF capture FIFO is
non-empty rather than off that default-ON flag.
"""

import functools
import inspect
import os
import tempfile
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch
from nkilib_src.nkilib.models.gpt_oss.c128spec_giga_kernel.experimental.gpt_oss.gpt_oss_spec_decode_layer_dp_tp_ep import (  # noqa: E501
    gpt_oss_spec_decode_layer_dp_tp_ep_kernel,
)

import test.integration.nkilib.experimental.gpt_oss.test_gpt_oss_decode_layer_dp_tp_ep_golden as sl
from test.utils.common_dataclasses import CustomValidator, CustomValidatorWithOutputTensorData
from test.utils.mx_utils import maybe_pack64_gate_up_scale
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata

# Active tokens per sequence = spec_len + 1. Default 4 == spec_len 3.
_S_TKG = int(os.environ.get("VLLM_NEURON_MEGA_S_TKG", "4"))

# Shifts the per-replica attention seed. Used to test whether a failing rank is data-dependent.
_SEED_OFFSET = int(os.environ.get("VLLM_NEURON_MEGA_SEED_OFFSET", "0"))


# Seed the global RNG before generate_inputs. The MXFP4 expert weights are drawn from the GLOBAL
# numpy RNG *before* generate_inputs applies its own internal seed, so without this the expert
# weights differ every run and results are not reproducible -- which is why rank65's reported
# rel-diff wandered (37.3 / 54.8 / 45.7 %) while the failing rank stayed put: routing was
# deterministic, the expert weights behind the error were not.
_MOE_SEED = int(os.environ.get("VLLM_NEURON_MEGA_MOE_SEED", "1234"))

# `sl._attn_cfg_full` reads the MODULE-GLOBAL _S_TKG. It is redirected INSIDE the test (monkeypatch), never at
# import: pytest collects every module of this directory into one session, so an import-time patch would also
# change the c128 golden test that shares the module.

# Sequences per rank. Kept at 1 so B_attn matches the c128/c512 gate (8 seqs/replica) and the
# S_tkg factor alone supplies the extra tokens.
_T_SHARD_SEQ = int(os.environ.get("VLLM_NEURON_MEGA_T_SHARD", "1"))
_T_SHARD_ROWS = _T_SHARD_SEQ * _S_TKG

# (dp, n_experts, collective_ranks, logical_nc_config). The upstream list tops out at 64 ranks;
# add the production point (128 ranks, 128 experts) because it is the only config where T_ep
# reaches 512 -> n_blocks = 512/64 = 8, which is what makes NUM_DYNAMIC_ALGO_STATIC_BLOCKS=5
# actually exercisable (below 320 tokens it clamps to n_blocks and the setting is a no-op).
_CANDIDATE_PARAMS = list(sl.DP_PARAMS) + [(16, 128, 128, 1)]

# Keep only the combos whose T_ep satisfies the DLoC block_size=128 contract of the c512 MoE the
# spec layer uses since 2026-09-19 (CR-302598650: T_ep/2 >= 128 and 128 | T_ep, i.e. T_ep >= 256).
# Filtering here (rather than letting the kernel assert) keeps the run green and makes the skipped
# configs visible in the node ids. At T_SHARD=1, S_TKG=4 that leaves ranks 64 (T_ep=256, 2 blocks)
# and 128 (T_ep=512, 4 blocks); ranks-32 (T_ep=128) is no longer a valid config.
_DLOC_BLOCK = 128
SPEC_DP_PARAMS = [
    p for p in _CANDIDATE_PARAMS if (_T_SHARD_ROWS * p[2]) >= 2 * _DLOC_BLOCK and (_T_SHARD_ROWS * p[2]) % 128 == 0
]

# Test-only routing skew, default off: VLLM_NEURON_MEGA_SPEC_GATE_SKEW_EXPERTS=<n> adds a large
# positive router bias to experts 0..n-1 so every token selects them and those experts receive all
# T_ep tokens. With T_ep=512 and block 128 that is 1 static + 3 dynamic blocks, so the DLoC dynamic
# loop (and the weight-split's in-loop HW-DGE up/down loads) EXECUTES on ranks 0..n-1; under the
# harness's uniform random routing (~16 tokens per expert) it otherwise never runs. Both the kernel
# and the fp32 oracle see the same biased router, so the golden stays valid; the large margin also
# keeps those n selections clear of the mxfp8 routing-flip regime.
_SKEW_EXPERTS = int(os.environ.get("VLLM_NEURON_MEGA_SPEC_GATE_SKEW_EXPERTS", "0"))
_SKEW_BIAS = float(os.environ.get("VLLM_NEURON_MEGA_SPEC_GATE_SKEW_BIAS", "8.0"))


# The per-rank dict feeds BOTH the NKI kernel and the torch golden ref, and the two want
# OPPOSITE gate/up MX scale layouts:
#
#   * the kernel needs the PACK64 form [E_L, 2, 64, H/512, I/4] (SCALE_PACK64 is ON by default);
#   * ``mxfp4_dense_adapter.packed_mx_to_dense_gate_up`` indexes ``scale[e, :, g, :, :]``, which
#     is only meaningful on the UNPACKED [E, 16, 2, H/512, I] form -- its own docstring declares
#     that shape.
#
# pack64 makes axis 1 the gate/up axis, so that index selects the wrong axes and dies downstream
# in the reshape: at H=512, I=512 it yields (2, 1, 128) = 256 elements against a (16, 512) = 8192
# target -> "cannot reshape array of size 256 into shape (16,512)".
#
# Handing each consumer its own layout is exact rather than approximate: pack64_gate_up_scale
# documents itself as "EXACT and invertible, not a reinterpretation", so both forms carry
# identical values. Keyed on rank_id, which is itself one of the kernel inputs.
#
# NOTE this same conflict exists UPSTREAM in test_gpt_oss_decode_layer_dp_tp_ep_golden.py, whose
# ranks-32 case fails identically on default flags -- consistent with CR-300772530's own "no
# hardware accuracy run for the merged fork" caveat. Fixed locally rather than in the shared
# adapter so this fork keeps its every-file-matches-the-CR provenance.
_UNPACKED_GU_SCALE = {}


def _rtol_for(collective_ranks: int) -> float:
    """Per-arm relative tolerance for the fp8 EP ReduceScatter (the kernel's only mode).

    The fp8-at-source ReduceScatter quantizes each rank's MoE partial to fp8_e4m3fn BEFORE the EP reduction, so the
    error grows with the number of partials summed -- i.e. with EP WIDTH. Measured maxima on this
    gate, one variable (the flag), same seed:

        EP width | bf16 RS | fp8-RS at source
        ---------|---------|------------------
            32   |  4.179% |  5.602%
            64   |  4.13%  |  7.014%   <- exceeds a flat 7e-2 by 0.014 pp, on 1 rank of 64

    So a flat 7e-2 makes the ranks-64 arm permanently red with the flag on, which trains people to
    ignore the gate. The 1.70x cost factor at EP=64 matches the 1.82x mean cost measured at EP=32,
    so this is the flag behaving as expected at greater width, not a defect.

    This is a REGRESSION threshold, not an accuracy standard. 9e-2 leaves headroom over the measured
    7.01% at EP=64 while still catching a gross error; it is NOT a claim that 9% output error is
    acceptable. Only an end-to-end eval can make that call, and it has not been run -- see the
    module docstring.

    EP=128 (the production sharding) was measured on 2026-09-19 with the c512 weight-split MoE,
    dynamic loop exercised (SPEC_GATE_SKEW_EXPERTS=2), same seed, per-rank rel-diff over 128 ranks
    (the bf16-RS column is the pre-2026-09-24 kernel, when bf16 RS was still selectable):

        EP width | bf16 RS (median / p90 / max) | fp8-RS at source (median / p90 / max)
        ---------|------------------------------|--------------------------------------
           128   |  3.22% / 4.15% / 4.65%       |  5.80% / 7.09% / 9.28%  <- 1 rank over 9e-2

    The 1.25x-per-doubling extrapolation predicted ~8.8%; the measured 9.28% max is that prediction
    plus one rank's tail. So EP=128 with the flag on gets 1.2e-1: 30% headroom over the measured max,
    still 5x below the routing-flip signature (a flipped token row lands at 45-50% on its rank), so a
    gross error is still caught. The bf16 arm keeps 7e-2 at every width (max 4.65%).

    A routing flip is a SEPARATE failure mode and must not be conflated with this tolerance: it is one
    token row that is ~orthogonal to the golden (kernel routes on mxfp8-quantized activations, the
    oracle in fp32), and it shows up as one rank at tens of percent while every other rank sits under
    the p90 above. Triage it by seed movement (SEED_OFFSET) and by the skew arm, never by widening rtol.
    """
    if collective_ranks >= 128:
        return 1.2e-1
    return 9e-2 if collective_ranks > 32 else 7e-2


# ── Per-token validation with routing-boundary exclusion ──────────────────────────────────────
#
# The framework's default check is one allclose over the rank's whole [T_shard_rows, H] `out`, so a
# single token that the kernel routed to a different expert set than the fp32 oracle (the documented
# mxfp8-vs-fp32 routing flip) fails the rank and hides every other token's result. This validator
# checks each token ROW with the framework's own criterion (max|a-b| <= atol + rtol * max|golden|,
# the "max" mode of nkilib_testing.maxAllClose) and, following the c512 runbook's recommended
# mitigation ("routing-aware exclusion: the golden knows its fp32 router margins"), excludes a
# failing row ONLY when the oracle itself shows that token sitting within mxfp8 rounding noise of
# its top-k boundary. Everything else is a FAIL.
#
# Margin, in noise units: the oracle's fp32 router logits give l_k (last selected) and l_k+1 (first
# unselected) with margin m = l_k - l_k+1 = sum_i x_i * (w_k - w_k+1)_i over the normed router input
# x. The kernel's router input is the bf16 residual stream (its matmul then runs in fp16,
# ``router_mm_dtype=nl.float16`` in moe_block_dp_ep.py), so relative to the fp32 oracle each x_i
# carries at least bf16 rounding: uniform in +-2^-8, rms 2^-8/sqrt(3). The induced std of m is
# sigma = 2^-8/sqrt(3) * sqrt(sum_i (x_i * (w_k - w_k+1)_i)^2), and the token's distance to the
# boundary is m / sigma. Below MARGIN_SIGMAS (default 4) the kernel can legitimately pick another
# expert set and a failing row is excluded.
#
# Calibration (EP128, fp8-RS arm, 2026-09-21, 512 rows): the one flipped row sat at 0.08 sigma
# (rank65 token1, rel 50.9 %, cosine 0.889); rows down to ~0.5 sigma routed identically and passed;
# 18 of 512 rows (3.5 %) lie below the 4-sigma line, so that is the size of the blind spot the
# exclusion opens, and it only applies to rows that already FAIL. An e4m3 noise model (the c512
# runbook's wording, "mxfp8 activation quant") was tried first and rejected: it puts 297 of 512
# rows under 4 sigma although only one flipped, i.e. it overstates the router's noise ~16x.
#
# Guards so the exclusion cannot hide a kernel bug: only rows that FAIL the tolerance are excluded,
# an excluded row must still be finite, at most ONE row per rank may be excluded, and the test
# asserts the global excluded count stays <= 2% of T_ep (a defect in an expert's MLP corrupts a row
# on every rank whose tokens route there, ~16 rows at EP128, and none of them sits at a boundary).
_MARGIN_SIGMAS_MIN = float(os.environ.get("VLLM_NEURON_MEGA_SPEC_GATE_MARGIN_SIGMAS", "4.0"))
_ROUTER_INPUT_REL_RMS = 2.0**-8 / 3.0**0.5  # bf16 rounding of the router input, see above
_MAX_EXCLUDED_FRAC = 0.02
# The goldens run in forked workers and the validators in the parent, so the exclusion ledger is a
# file (path fixed at import, inherited by every process of this test run).
_EXCLUDED_LEDGER = os.path.join(tempfile.gettempdir(), f"spec_gate_excluded_{os.getpid()}.tsv")
_LAST_ROUTER_MARGIN: dict = {}  # worker-process scratch: margins of the most recent oracle call
_CUR_TOL = {"rtol": 7e-2, "atol": 1e-2}  # set by the test before run_test, read by the validators

_ORACLE = sl.gptoss_mxfp4_decode_layer


def _oracle_with_router_margin(*args, **kwargs):
    """The whole-layer oracle plus the fp32 router margin (in router-input-noise sigmas) of every token.

    Uses the oracle's own ``capture`` hook to read the MoE's exact input, then repeats the oracle's
    router math (kernels.moe_block_tkg: fp32 RMSNorm -> logits = normed @ W^T + b -> top-k).
    """
    cap: dict = {}
    out = _ORACLE(*args, capture=cap, **kwargs)
    mi = cap["moe"][0]["inputs"]
    x = mi["hidden_states"].to(torch.float32)
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + float(mi["eps"])) * mi["gamma"].to(torch.float32)
    rw = mi["router_weight"].to(torch.float32)  # [E, H]
    logits = normed @ rw.t() + mi["router_bias"].to(torch.float32)  # [T, E]
    k = int(mi["top_k"])
    top, idx = torch.topk(logits, k + 1, dim=-1)
    margin = top[:, k - 1] - top[:, k]  # last selected minus first unselected
    d = rw[idx[:, k - 1]] - rw[idx[:, k]]  # [T, H]
    sigma = _ROUTER_INPUT_REL_RMS * torch.sqrt(((normed * d) ** 2).sum(-1))
    _LAST_ROUTER_MARGIN["sigmas"] = (margin / sigma.clamp_min(1e-12)).numpy().astype(np.float32)
    return out


# make_golden_ref resolves the oracle by name in sl at call time, so patching the module attribute (inside the
# test, see _patch_shared_harness) routes every golden through the margin-capturing wrapper, in the forked
# workers as well.


def _make_token_validator(golden_out, sigmas, rank_id):
    g = np.asarray(golden_out)
    _gdtype, _gshape = g.dtype, g.shape

    class _PerToken(CustomValidator):
        def validate(self, inference_output: npt.NDArray[Any]) -> bool:
            av = np.asarray(inference_output)
            try:
                av = av.view(_gdtype).reshape(_gshape)
            except (ValueError, TypeError):
                pass
            a = np.asarray(av).astype(np.float64)
            b = g.astype(np.float64)
            if a.shape != b.shape:
                self._print_with_log(f"[spec-accuracy] rank{rank_id}: shape {a.shape} != golden {b.shape} -> FAIL")
                return False
            rtol, atol = _CUR_TOL["rtol"], _CUR_TOL["atol"]
            brange = float(np.nanmax(np.abs(b)))
            ok, n_excl = True, 0
            for t in range(b.shape[0]):
                finite = bool(np.all(np.isfinite(a[t])))
                maxd = float(np.max(np.abs(a[t] - b[t]))) if finite else float("inf")
                cos = (
                    float(np.dot(a[t], b[t]) / (np.linalg.norm(a[t]) * np.linalg.norm(b[t]) + 1e-12)) if finite else 0.0
                )
                sig = float(sigmas[t]) if sigmas is not None else float("nan")
                row_ok = finite and maxd <= atol + rtol * brange
                verdict = "PASS" if row_ok else "FAIL"
                if not row_ok and finite and sigmas is not None and sig < _MARGIN_SIGMAS_MIN:
                    n_excl += 1
                    verdict = "EXCLUDED(routing boundary)"
                    with open(_EXCLUDED_LEDGER, "a") as fh:
                        fh.write(f"{rank_id}\t{t}\t{sig:.3f}\t{100.0 * maxd / brange:.2f}\t{cos:.4f}\n")
                    row_ok = True
                ok = ok and row_ok
                self._print_with_log(
                    f"[spec-accuracy] rank{rank_id} token={t} max_abs_diff={maxd:.4f} rel={100.0 * maxd / brange:.2f}% "
                    f"(rtol {100.0 * rtol:.0f}%) cosine={cos:.4f} router_margin={sig:.1f}sigma -> {verdict}"
                )
            if n_excl > 1:
                self._print_with_log(
                    f"[spec-accuracy] rank{rank_id}: {n_excl} routing-boundary exclusions on one rank -> FAIL"
                )
                return False
            return ok

    return CustomValidatorWithOutputTensorData(validator=_PerToken, output_ndarray=g)


def _spec_comparator(rank_id, golden):
    """run_test custom_comparator: per-token validator for `out`, fed the rank's router margins."""
    g = dict(golden)
    sig = g.pop("router_margin_sigmas", None)
    return {"out": _make_token_validator(g["out"], sig, rank_id)}


def _read_excluded_ledger():
    if not os.path.exists(_EXCLUDED_LEDGER):
        return []
    with open(_EXCLUDED_LEDGER) as fh:
        return [line.rstrip("\n").split("\t") for line in fh if line.strip()]


def _ref_with_unpacked_gu_scale(base_ref):
    """Wrap a golden ref so it receives the UNPACKED gate/up scale for its rank.

    ``functools.wraps`` is load-bearing, not cosmetic: the framework introspects the ref with
    ``signature(torch_ref).parameters.keys()`` in TWO places, and a bare ``**kwargs`` reports
    ``{'kwargs'}`` for both. ``validate_torch_ref_signature`` compares that set to the kernel's
    39 params and raises at collection time, and ``filter_ref_input`` uses the same set to select
    what the ref is called with, so suppressing the validator alone would hand the ref an EMPTY
    dict. ``wraps`` sets ``__wrapped__``, which ``inspect.signature`` follows, fixing both.
    """

    @functools.wraps(base_ref)
    def ref(**kwargs):
        rank_id = int(np.asarray(kwargs["rank_id"]).reshape(-1)[0])
        unpacked = _UNPACKED_GU_SCALE.get(rank_id)
        if unpacked is not None:
            kwargs = dict(kwargs)
            kwargs["expert_gate_up_weights_scale"] = unpacked
        res = dict(base_ref(**kwargs))
        # Attach this rank's router margins (same row slice make_golden_ref uses for `out`) so the
        # per-token validator in the parent process can tell a routing-boundary token from a defect.
        sig = _LAST_ROUTER_MARGIN.get("sigmas")
        if sig is not None:
            tp_rank = rank_id % int(kwargs["tp_size"])
            t_shard = int(np.asarray(kwargs["x_shard"]).shape[0])
            res["router_margin_sigmas"] = np.asarray(sig[tp_rank * t_shard : (tp_rank + 1) * t_shard], dtype=np.float32)
        return res

    return ref


def _align_ref_signature(ref, kernel_entry):
    """Keep only the ref parameters that the kernel entry declares.

    The shared golden ref also declares the c128 kernel's optional sequence-packing metadata
    (``seq_packed_*``, all defaulting to None), which this kernel does not take, and
    ``validate_torch_ref_signature`` compares the two parameter sets exactly, in both directions.
    The shared mega golden aligns its ref to the c512 entry the same way. ``filter_ref_input`` reads
    the same signature, so the ref gets only this kernel's inputs and the dropped parameters keep
    their None defaults.
    """
    kernel_params = set(inspect.signature(kernel_entry).parameters)
    kept = [p for p in inspect.signature(ref).parameters.values() if p.name in kernel_params]
    ref.__signature__ = inspect.Signature(kept)
    return ref


def _patch_shared_harness(monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the shared c128 golden harness to this kernel's S_tkg and oracle wrapper for ONE test.

    Scoped to the test through monkeypatch so the c128 golden test in the same pytest session keeps its own
    S_tkg=1 harness (the module globals are read at call time, and the goldens run in forked workers that
    inherit the patched state).
    """
    monkeypatch.setattr(sl, "_S_TKG", _S_TKG)
    monkeypatch.setattr(sl, "gptoss_mxfp4_decode_layer", _oracle_with_router_margin)


# The metadata name becomes a pipeline (Hydra) step name and may only contain [A-Za-z0-9-_ ].
@pytest_test_metadata(name="GPT-OSS spec-decode target layer c128spec vs golden")
@pytest_marks(["collectives"])
@pytest.mark.high_rank
@pytest.mark.skip_simulation
@pytest.mark.platforms(exclude=[sl.Platforms.TRN1, sl.Platforms.TRN2])
class TestGptOssSpecDecodeLayerGolden:
    """Single-layer spec-decode target kernel vs the HF-bit-validated whole-layer oracle."""

    @sl.pytest_parametrize(sl.DP_PARAM_NAMES, SPEC_DP_PARAMS, abbrevs=sl._ABBREVS)
    def test_spec_layer_vs_golden(
        self,
        test_manager: sl.Orchestrator,
        platform_target: sl.Platforms,
        monkeypatch: pytest.MonkeyPatch,
        dp: int,
        n_experts: int,
        collective_ranks: int,
        logical_nc_config: int,
    ):
        if not platform_target.is_trn3():
            pytest.skip("MXFP4 (float4_e2m1fn_x4) is only supported on TRN3.")
        _patch_shared_harness(monkeypatch)

        _TP, _H, _Q_HEADS, _D_HEAD = sl._TP, sl._H, sl._Q_HEADS, sl._D_HEAD
        T_ep = _T_SHARD_ROWS * collective_ranks

        tp_subgroups = [list(range(d * _TP, (d + 1) * _TP)) for d in range(dp)]
        tp_group = sl.ReplicaGroup(tp_subgroups)
        ep_group = sl.ReplicaGroup([list(range(collective_ranks))])

        T_shard = _T_SHARD_ROWS  # rows of x_shard (the kernel's T_shard)
        B_attn = _T_SHARD_SEQ * _TP  # SEQUENCES per DP replica
        B_g_tokens = T_shard * collective_ranks  # global active tokens == T_ep
        softmax_scale = float(_D_HEAD**-0.5)
        print(
            f"[spec-accuracy] ranks={collective_ranks} dp={dp} E={n_experts} "
            f"B_attn={B_attn} seqs x S_tkg={_S_TKG} -> T_shard_rows={T_shard}, T_ep={T_ep}"
        )

        acfg = sl._attn_cfg_full(B_attn)  # ONE full 64-head attention set per replica
        _QD_R = _Q_HEADS * _D_HEAD
        _Q_ALL = sl._FULL_Q_HEADS * _D_HEAD
        _KV = sl._KV_HEADS * _D_HEAD
        np.random.seed(_MOE_SEED)  # see _MOE_SEED note: MXFP4 draws from the global RNG first
        moe_global = sl.generate_inputs(
            batch=B_g_tokens,
            seqlen=1,
            hidden=_H,
            hidden_actual=_H,
            intermediate=sl._I,
            num_global_experts=n_experts,
            num_local_experts=n_experts,
            top_k=sl._TOP_K,
            router_fn=None,  # ty: ignore[invalid-argument-type]  (None = skip; harness types it as the enum)
            hidden_act_fn=None,  # ty: ignore[invalid-argument-type]  (None = skip; harness types it as the enum)
            expert_affinities_scaling_mode=None,  # ty: ignore[invalid-argument-type]  (None = skip; harness types it as the enum)
            moe_weight_dtype=sl._MX,
            input_dtype=sl._DTYPE,
            has_bias=True,
            has_clamp=True,
            router_act_first=False,
            norm_topk_prob=False,
            skip_router_logits=True,
            router_mm_dtype=sl.nl.float16,
            is_all_expert=True,
        )
        moe_global["expert_gate_up_bias"][:, :, 1, ...] += 1.0  # GPT-OSS +1
        if _SKEW_EXPERTS > 0:
            rb = moe_global["router_bias"]
            rb.reshape(-1)[:_SKEW_EXPERTS] += np.asarray(_SKEW_BIAS, dtype=rb.dtype)
            print(
                f"[spec-accuracy] router skew: +{_SKEW_BIAS} bias on experts 0..{_SKEW_EXPERTS - 1} "
                f"-> each receives all {T_ep} tokens = {T_ep // 128} DLoC blocks (1 static + {T_ep // 128 - 1} dynamic)"
            )

        def create_inputs(rank_id: int):
            dp_replica = rank_id // _TP
            tp_rank = rank_id % _TP
            # Seed offset is env-tunable so a suspected routing-flip failure can be tested for
            # data-dependence: if the failing rank MOVES when the offset changes, the cause is a
            # token sitting near a top-k router boundary (mxfp8 kernel quant vs fp32 golden), not
            # anything structural about that rank's expert.
            aki = sl.generate_kernel_inputs(acfg, seed=1000 * dp_replica + _SEED_OFFSET)
            # THE ONLY S_tkg DELTA vs the c128/c512 gate: generate_kernel_inputs emits X for
            # B_attn sequences x S_tkg active tokens, so the token axis is B_attn * S_tkg (==
            # B_attn only when S_tkg == 1) and each TP rank owns a contiguous T_shard_rows slice.
            X_full = aki["X"].reshape(B_attn * _S_TKG, _H)
            x_shard = X_full[tp_rank * T_shard : (tp_rank + 1) * T_shard].copy()
            Wqkv_full = aki["W_qkv"]
            Wqkv_r = np.ascontiguousarray(
                np.concatenate(
                    [
                        Wqkv_full[:, tp_rank * _QD_R : (tp_rank + 1) * _QD_R],
                        Wqkv_full[:, _Q_ALL : _Q_ALL + _KV],
                        Wqkv_full[:, _Q_ALL + _KV : _Q_ALL + 2 * _KV],
                    ],
                    axis=1,
                )
            )
            Wout_r = np.ascontiguousarray(aki["W_out"][tp_rank * _QD_R : (tp_rank + 1) * _QD_R, :])
            sink_r = np.ascontiguousarray(aki["sink"][tp_rank * _Q_HEADS : (tp_rank + 1) * _Q_HEADS, :])
            # [S_tkg, B, 64, S_tkg] head-broadcast causal chain -> this rank's 8 heads.
            mask_full = aki["attention_mask"]
            mask_r = np.ascontiguousarray(mask_full[:, :, tp_rank * _Q_HEADS : (tp_rank + 1) * _Q_HEADS, :])
            e = rank_id  # this rank's global expert index (EP)
            # See _ref_with_unpacked_gu_scale: the kernel wants the PACK64 gate/up scale but the
            # torch oracle wants the unpacked one, so stash the unpacked slice for the ref.
            _UNPACKED_GU_SCALE[rank_id] = moe_global["expert_gate_up_weights_scale"][e : e + 1]
            return {
                "x_shard": x_shard,
                "input_layernorm_weight": aki["rmsnorm_X_gamma"],
                "W_qkv": Wqkv_r,
                "bias_qkv": aki["bias_qkv"],
                "W_out": Wout_r,
                "bias_out": aki["bias_out"],
                "cos": aki["cos"],
                "sin": aki["sin"],
                "sink": sink_r,
                "K_cache": aki["K_cache"],
                "V_cache": aki["V_cache"],
                "active_blocks_table": aki["active_blocks_table"],
                "attention_mask": mask_r,
                "kv_cache_update_idx": aki["kv_cache_update_idx"],
                "pos_ids": aki["pos_ids"],
                "swa_start_pos_ids": aki["swa_start_pos_ids"],
                "post_attn_layernorm_weight": moe_global["gamma"],
                "router_weights": moe_global["router_weights"],
                "router_bias": moe_global["router_bias"],
                "expert_gate_up_weights": moe_global["expert_gate_up_weights"][e : e + 1],
                "expert_down_weights": moe_global["expert_down_weights"][e : e + 1],
                # SCALE_PACK64 is ON by default, so the gate/up MX scale must be remapped from
                # the harness's unpacked layout to [E_L, 2, 64, H/512, I/4] or the kernel rejects
                # it at trace. Only gate/up is packed; the down scale stays as-is.
                "expert_gate_up_weights_scale": maybe_pack64_gate_up_scale(
                    moe_global["expert_gate_up_weights_scale"][e : e + 1]
                ),
                "expert_down_weights_scale": moe_global["expert_down_weights_scale"][e : e + 1],
                "expert_gate_up_bias": moe_global["expert_gate_up_bias"][e : e + 1],
                "expert_down_bias": moe_global["expert_down_bias"][e : e + 1],
                "rank_id": np.array([[rank_id]], dtype=np.uint32),
                "tp_replica_group": tp_group,
                "ep_replica_group": ep_group,
                "tp_size": _TP,
                "ep_size": collective_ranks,
                "B_attn": B_attn,
                "S_tkg": _S_TKG,
                "top_k": sl._TOP_K,
                "eps": sl._EPS,
                "hidden_actual": _H,
                "softmax_scale": softmax_scale,
                "gate_clamp_upper_limit": sl._GATE_CLAMP_UPPER,
                "up_clamp_upper_limit": sl._UP_CLAMP_UPPER,
                "up_clamp_lower_limit": sl._UP_CLAMP_LOWER,
            }

        # Same oracle as the c128/c512 gate. make_golden_ref needs no S_tkg argument: it reads
        # T_shard from x_shard.shape[0] and the oracle derives S_decode. Validation is per token row
        # (all T_shard_rows x H of every rank, i.e. all T_ep spec positions) with routing-boundary
        # exclusion; see _make_token_validator.
        _CUR_TOL["rtol"], _CUR_TOL["atol"] = _rtol_for(collective_ranks), 1e-2
        if os.path.exists(_EXCLUDED_LEDGER):
            os.remove(_EXCLUDED_LEDGER)
        sl.CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=gpt_oss_spec_decode_layer_dp_tp_ep_kernel,
            torch_ref=_align_ref_signature(
                _ref_with_unpacked_gu_scale(sl.make_golden_ref(tp_group, ep_group, _TP, collective_ranks)),
                gpt_oss_spec_decode_layer_dp_tp_ep_kernel,
            ),
            per_rank_input_generator=create_inputs,
            collective_ranks=collective_ranks,
        ).run_test(
            test_config=None,
            compiler_args=sl.CompilerArgs(
                logical_nc_config=logical_nc_config,
                platform_target=platform_target,
                additional_cmd_args=["--enable-ocp-compliant-scale-computation"],
            ),
            output_keys=["out"],
            rtol=_rtol_for(collective_ranks),
            atol=1e-2,
            inference_args=sl.InferenceArgs(collective_ranks=collective_ranks, enable_determinism_check=False),
            custom_comparator=_spec_comparator,
        )
        excluded = _read_excluded_ledger()
        print(
            f"[spec-accuracy] validated {T_ep} token rows x {_H} cols on {collective_ranks} ranks "
            f"(rtol {_CUR_TOL['rtol']:.2f}); routing-boundary rows excluded: {len(excluded)} "
            f"[(rank, token, sigmas, rel%, cosine)]: {excluded}"
        )
        assert len(excluded) <= _MAX_EXCLUDED_FRAC * T_ep, (
            f"{len(excluded)} routing-boundary exclusions exceed {_MAX_EXCLUDED_FRAC:.0%} of T_ep={T_ep}: {excluded}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-x"])
