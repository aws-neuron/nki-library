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

"""Driver for the GPT-OSS SPECULATIVE-DECODE TARGET giga kernel (c128 attn + c512 DLoC MoE).

Profiling driver for ``gpt_oss_spec_decode_mega_kernel``. It reuses every input generator,
helper and constant from :mod:`test_gpt_oss_decode_mega_golden` -- that module is imported,
never modified -- and differs from it in exactly three ways:

1. **S_tkg is env-tunable** (``VLLM_NEURON_MEGA_S_TKG``, default 4 = spec_len 3 + 1 bonus).
   ``T_SHARD`` keeps its meaning of *per-rank* count but is now counted in **sequences**,
   so a rank owns ``T_SHARD`` sequences x ``S_tkg`` active tokens = ``T_SHARD * S_tkg`` rows.
2. **The token axis carries S_tkg.** The golden harness does
   ``aki["X"].reshape(B_attn, H)`` and slices ``T_shard`` rows, both of which assume one
   active token per sequence. Here X is ``[B_attn * S_tkg, H]`` and each TP rank takes a
   contiguous ``T_shard * S_tkg`` slice. These are the ONLY two lines that had to change --
   everything else in the harness is already S_tkg-general.
3. **It drives the spec kernel** and is profile-only.

The default shape is the production spec-decode target config, chosen to match the c128
giga-kernel measurement so the two are comparable:

    VLLM_NEURON_MEGA_FULL_120B=1  ->  DP16 x TP8 x EP128 = 128 ranks, 36 layers
    T_SHARD=1 (sequences/rank)    ->  128 sequences globally  (= the c128 run's concurrency)
    S_TKG=4   (spec_len 3 + 1)    ->  512 active tokens globally
      => attention: B_attn = 8 sequences/replica, s_active = 4   (c128-shaped)
      => MoE:       T_ep = (1 * 4) * 128 = 512 tokens            (c512 DLoC-shaped)

**The active mask needs no special handling.** The harness's
``build_active_attention_mask`` already emits the causal chain overlay for any
``s_active``: ``tril(ones(s,s))`` permuted to ``[key, B, head, query]``, i.e. 1 iff
``key <= query``. That is the same tensor vllm-neuron's ``_build_default_active_mask``
builds as ``triu(ones(s,s))`` in that layout, so this driver exercises the production mask
bit-for-bit. vllm-neuron has no tree-draft path, so a chain is the only topology needed.

Golden is NOT available at S_tkg > 1 (the oracle reshapes the hidden state to ``(B_attn, H)``
and slices ``T_shard`` rows -- both S_tkg=1 assumptions), so this driver requires
``VLLM_NEURON_MEGA_SKIP_GOLDEN=1`` and makes no correctness claim.

Run (128-rank profile, on a trn3 pod with the pinned toolchain):

    export VLLM_NEURON_MEGA_FULL_120B=1 VLLM_NEURON_MEGA_SKIP_GOLDEN=1
    export VLLM_NEURON_MEGA_S_TKG=4 VLLM_NEURON_MEGA_T_SHARD=1
    export NEURON_RT_ENABLE_OCP=1 NEURON_RT_ENABLE_OCP_SATURATION=1   # the EP ReduceScatter is fp8
    pytest -c test/pytest.ini \\
      test/integration/nkilib/experimental/gpt_oss/test_gpt_oss_spec_decode_mega.py \\
      --platform-target trn3 --nki-compilation-mode tracer --enable-dge-notifs -v -s
"""

import functools
import logging
import os

import numpy as np
import pytest
from nkilib_src.nkilib.models.gpt_oss.c128spec_giga_kernel.experimental.gpt_oss.gpt_oss_spec_decode_layer_dp_tp_ep import (  # noqa: E501
    gpt_oss_spec_decode_mega_kernel,
)

import test.integration.nkilib.experimental.gpt_oss.test_gpt_oss_decode_mega_golden as mega
from test.utils.mx_utils import maybe_pack64_gate_up_scale
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata

# Active tokens per sequence per verify pass = spec_len + 1. Default 4 == spec_len 3.
_S_TKG = int(os.environ.get("VLLM_NEURON_MEGA_S_TKG", "4"))


def _patch_shared_harness(monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the shared mega harness to this kernel's S_tkg for ONE test.

    The harness computes its attention config from the MODULE-GLOBAL _S_TKG (mega._attn_cfg_full reads it at call
    time). Its aki memo is keyed on (seed, sliding, batch) -- NOT on S_tkg -- so the test gets a fresh memo as
    well. Both are scoped to the test through monkeypatch: pytest collects every module of this directory into
    one session, and an import-time patch would also change the c128 mega golden test that shares the module.
    """
    monkeypatch.setattr(mega, "_S_TKG", _S_TKG)
    monkeypatch.setattr(mega, "_ATTN_AKI_MEMO", {})


# The shared mega harness gained a `giga_variant` axis that selects which vendored fork its
# LAZY loader imports. This driver imports the spec kernel directly, so the axis selects
# nothing here -- relabel it `c128spec` over the harness's own layer configs so the test id
# names the kernel that actually ran (and does not duplicate when VLLM_NEURON_MEGA_GIGA lists
# several variants for the non-spec test).
_SPEC_LAYER_PARAMS = [("c128spec", nlyr, lnc) for (nlyr, lnc) in mega._LAYER_CFGS]


def _make_spec_stub_ref():
    """Zero-returning stand-in for the golden under SKIP_GOLDEN (the mega golden is S_tkg=1-only).

    Built on THIS kernel's parameter set: ``functools.wraps`` makes ``inspect.signature`` follow the kernel, which
    is what the framework's signature validator and input filter read. The shared harness's ``make_stub_ref``
    mirrors the c128 mega kernel instead, whose parameter set has since diverged (per-layer KV-cache buffers).
    Returns zeros in the kernel's output shapes/dtypes so the output descriptors are right; validation against
    it is meaningless and the comparator accepts everything.
    """

    @functools.wraps(gpt_oss_spec_decode_mega_kernel)
    def ref(**kw):
        x = np.asarray(kw["x_shard"])
        res = {"out": np.zeros(x.shape, dtype=x.dtype)}
        if mega._KV_UPDATE:
            res["K_cache"] = np.zeros_like(np.asarray(kw["K_cache"]))
            res["V_cache"] = np.zeros_like(np.asarray(kw["V_cache"]))
        return res

    return ref


# The metadata name becomes a pipeline (Hydra) step name and may only contain [A-Za-z0-9-_ ].
@pytest_test_metadata(name="GPT-OSS spec-decode target GIGA kernel c128spec profile driver")
@pytest_marks(["collectives"])
@pytest.mark.high_rank
@pytest.mark.skip_simulation
@pytest.mark.platforms(exclude=[mega.Platforms.TRN1, mega.Platforms.TRN2])
class TestGptOssSpecDecodeMega:
    """128-rank spec-decode target giga kernel: c128 attention + c512 DLoC MoE."""

    @mega.pytest_parametrize(mega.LAYER_PARAM_NAMES, _SPEC_LAYER_PARAMS, abbrevs=mega._ABBREVS)
    def test_spec_decode_mega(
        self,
        test_manager: mega.Orchestrator,
        platform_target: mega.Platforms,
        monkeypatch: pytest.MonkeyPatch,
        giga_variant: str,
        num_layers: int,
        logical_nc_config: int,
    ):
        if not platform_target.is_trn3():
            pytest.skip("MXFP4 (float4_e2m1fn_x4) is only supported on TRN3.")
        _patch_shared_harness(monkeypatch)
        # This is the profile driver: the golden reshapes the hidden state to (B_attn, H) and slices
        # T_shard rows, both assuming one active token per sequence, so it cannot validate S_tkg=4.
        # Without the SKIP_GOLDEN opt-in (the module docstring's recipe) skip with the reason, which
        # is what a pipeline run without the profile environment should do, rather than validate
        # against a wrong-shaped reference or fail.
        if _S_TKG != 1 and not mega._SKIP_GOLDEN:
            pytest.skip(
                f"profile driver: VLLM_NEURON_MEGA_S_TKG={_S_TKG} requires VLLM_NEURON_MEGA_SKIP_GOLDEN=1 "
                "(the mega golden is S_tkg=1 only)"
            )
        if mega._FULL_120B_PROFILE and not mega._SKIP_GOLDEN:
            pytest.skip("profile driver: VLLM_NEURON_MEGA_FULL_120B requires VLLM_NEURON_MEGA_SKIP_GOLDEN=1")

        _TP, _RANKS, _DP = mega._TP, mega._RANKS, mega._DP
        _H, _D_HEAD = mega._H, mega._D_HEAD
        _Q_HEADS, _KV_HEADS = mega._Q_HEADS, mega._KV_HEADS
        _FULL_KV_HEADS = mega._FULL_KV_HEADS

        tp_subgroups = [list(range(d * _TP, (d + 1) * _TP)) for d in range(_DP)]
        tp_group = mega.ReplicaGroup(tp_subgroups)
        ep_group = mega.ReplicaGroup([list(range(_RANKS))])

        # SEQUENCES per rank (not tokens). Each contributes S_tkg active tokens, so the
        # kernel sees T_shard_rows = T_shard_seq * S_tkg rows of x_shard.
        T_shard_seq = int(os.environ.get("VLLM_NEURON_MEGA_T_SHARD", "1"))
        T_shard_rows = T_shard_seq * _S_TKG
        B_attn = T_shard_seq * _TP  # sequences per DP replica
        B_g_seq = T_shard_seq * _RANKS  # global sequences (= concurrency)
        B_g_tokens = T_shard_rows * _RANKS  # global active tokens == T_ep
        softmax_scale = float(_D_HEAD**-0.5)
        layer_input_count = 1 if mega._SHARE_LAYER_INPUTS else num_layers
        logging.info(
            "spec target: %d sequences x S_tkg %d = %d tokens; B_attn=%d, T_ep=%d",
            B_g_seq,
            _S_TKG,
            B_g_tokens,
            B_attn,
            B_g_tokens,
        )

        input_cache = None
        if mega._INPUT_CACHE_DIR:
            source_files = [
                s
                for s in (
                    __file__,
                    mega.inspect.getsourcefile(mega.generate_inputs),
                    mega.inspect.getsourcefile(mega.generate_kernel_inputs),
                )
                if s is not None
            ]
            # s_tkg MUST be in the key: it changes every per-rank token-axis shape. Without
            # it an S_tkg=4 run silently loads the S_tkg=1 cache and dies at trace with a
            # reshape error (the same failure mode the golden harness documents for t_shard).
            cache_key = mega.build_input_cache_key(
                {
                    "tag": mega._INPUT_CACHE_TAG,
                    "variant": "spec_target",
                    "num_layers": num_layers,
                    "layer_input_count": layer_input_count,
                    "logical_nc_config": logical_nc_config,
                    "dp": _DP,
                    "tp": _TP,
                    "ranks": _RANKS,
                    "experts": mega._N_EXPERTS,
                    "hidden": _H,
                    "hidden_actual": mega._H_ACTUAL,
                    "intermediate": mega._I,
                    "head_dim": _D_HEAD,
                    "q_heads_per_rank": _Q_HEADS,
                    "kv_heads_per_rank": _KV_HEADS,
                    "context": mega._S_CTX,
                    "block_len": mega._BLOCK_LEN,
                    "sliding_window": mega._SLIDING_WINDOW,
                    "moe_seed": mega._MOE_SEED,
                    "t_shard": T_shard_seq,
                    "s_tkg": _S_TKG,
                    "fp8_kv": mega._FP8_KV,
                    "kv_quant_dtype": str(mega._KV_QUANT_DTYPE) if mega._FP8_KV else None,
                },
                source_files,
            )
            input_cache = mega.PersistentPerRankInputCache(mega._INPUT_CACHE_DIR, cache_key)
            logging.info("spec-target input cache namespace: %s", input_cache.directory)

        mg_shared = None

        def get_moe_inputs():
            nonlocal mg_shared
            if mg_shared is None:
                np.random.seed(mega._MOE_SEED)
                mg_shared = mega.generate_inputs(
                    batch=B_g_tokens,
                    seqlen=1,
                    hidden=_H,
                    hidden_actual=mega._H_ACTUAL,
                    intermediate=mega._I,
                    num_global_experts=mega._N_EXPERTS,
                    num_local_experts=mega._N_EXPERTS,
                    top_k=mega._TOP_K,
                    router_fn=None,  # ty: ignore[invalid-argument-type]  (None = skip; harness types it as the enum)
                    hidden_act_fn=None,  # ty: ignore[invalid-argument-type]  (None = skip; harness types it as the enum)
                    expert_affinities_scaling_mode=None,  # ty: ignore[invalid-argument-type]  (None = skip; harness types it as the enum)
                    moe_weight_dtype=mega._MX,
                    input_dtype=mega._DTYPE,
                    has_bias=True,
                    has_clamp=True,
                    router_act_first=False,
                    norm_topk_prob=False,
                    skip_router_logits=True,
                    router_mm_dtype=mega.nl.float16,
                    is_all_expert=True,
                )
                mg_shared["expert_gate_up_bias"][:, :, 1, ...] += 1.0
                mega._zero_moe_hidden_padding(mg_shared)
            return mg_shared

        def generate_array_inputs(rank_id: int):
            dp_replica = rank_id // _TP
            tp_rank = rank_id % _TP
            e = rank_id  # this rank's global expert index (EP)
            qd_r = _Q_HEADS * _D_HEAD
            q_all = mega._FULL_Q_HEADS * _D_HEAD
            kv1 = _D_HEAD
            moe_inputs = get_moe_inputs()

            (iln, wqkv, bqkv, wout, bout, sink, kcache, vcache, paln, rw, rb, egu, edn, egus, edns, egub, edb, swa) = (
                [] for _ in range(18)
            )
            x_shard = None
            cos0 = sin0 = abt0 = mask0 = upd0 = pos0 = None
            k_scale0 = v_scale0 = None
            for li in range(layer_input_count):
                sliding = mega._SLIDING_WINDOW if mega._is_sliding(li) else 0
                aki = mega._gen_attn_inputs(mega._attn_seed(dp_replica, li), sliding, B_attn)

                # THE S_tkg CHANGE. generate_kernel_inputs emits X for B_attn sequences x
                # S_tkg active tokens, so the token axis is B_attn * S_tkg (== B_attn only
                # when S_tkg == 1). Each TP rank owns a contiguous T_shard_rows slice; the
                # kernel takes its T_shard from x_shard.shape[0] and AllGather[TP]
                # reconstructs T_attn == B_attn * S_tkg rows before reshaping to
                # (B_attn, S_tkg, H).
                X_full = aki["X"].reshape(B_attn * _S_TKG, _H)
                if li == 0:
                    x_shard = X_full[tp_rank * T_shard_rows : (tp_rank + 1) * T_shard_rows].copy()

                Wqkv_full = aki["W_qkv"]
                kbase = q_all
                vbase = q_all + _FULL_KV_HEADS * _D_HEAD
                Wqkv_r = np.ascontiguousarray(
                    np.concatenate(
                        [
                            Wqkv_full[:, tp_rank * qd_r : (tp_rank + 1) * qd_r],
                            Wqkv_full[:, kbase + tp_rank * kv1 : kbase + (tp_rank + 1) * kv1],
                            Wqkv_full[:, vbase + tp_rank * kv1 : vbase + (tp_rank + 1) * kv1],
                        ],
                        axis=1,
                    )
                )
                Wout_r = np.ascontiguousarray(aki["W_out"][tp_rank * qd_r : (tp_rank + 1) * qd_r, :])
                sink_r = np.ascontiguousarray(aki["sink"][tp_rank * _Q_HEADS : (tp_rank + 1) * _Q_HEADS, :])
                Kc_full, Vc_full = aki["K_cache"], aki["V_cache"]
                Kc_r = np.ascontiguousarray(Kc_full[:, tp_rank : tp_rank + 1])
                Vc_r = np.ascontiguousarray(Vc_full[:, tp_rank : tp_rank + 1, :, :])
                abt_full = aki["active_blocks_table"]
                abt_head = abt_full[:, tp_rank, :]
                abt_r = np.ascontiguousarray(np.where(abt_head < 0, abt_head, abt_head // _FULL_KV_HEADS))
                # [S_tkg, B, 64, S_tkg] head-broadcast causal chain overlay -> this rank's heads.
                mask_full = aki["attention_mask"]
                mask_r = np.ascontiguousarray(mask_full[:, :, tp_rank * _Q_HEADS : (tp_rank + 1) * _Q_HEADS, :])
                upd_full = aki["kv_cache_update_idx"]
                if upd_full.ndim == 3:
                    if mega._KV_UPDATE:
                        # Localize PER OWNED HEAD, matching the non-spec driver. Unlike the read
                        # block table (head-independent `base`), the WRITE index carries the
                        # local-head factor, so each of this rank's _KV_HEADS heads must be
                        # localized with its own local_head rather than sharing head 0. At TP8
                        # (_KV_HEADS == 1) this reduces to base*block_len + offset; at TP<8 taking
                        # a single head would mislocalize the rest.
                        #
                        # This is a hang-not-crash site: a wrong slot addresses out of bounds, so
                        # every scatter is skipped/clamped, and under FP8_KV the gather-modify-
                        # scatter desyncs ranks into a collective hang instead of failing loudly.
                        upd_r = np.stack(
                            [
                                mega._localize_update_idx(
                                    upd_full[:, tp_rank * _KV_HEADS + h, :],
                                    mega._BLOCK_LEN,
                                    _FULL_KV_HEADS,
                                    _KV_HEADS,
                                    h,
                                )
                                for h in range(_KV_HEADS)
                            ],
                            axis=1,
                        )  # [B, _KV_HEADS, S_tkg]
                        # kv_heads == 1 wants the squeezed [B, S_tkg] form; >1 stays 3D and the
                        # kernel folds it to [B*_KV_HEADS, S_tkg].
                        if _KV_HEADS == 1:
                            upd_r = upd_r[:, 0, :]
                        upd_r = np.ascontiguousarray(upd_r)
                    else:
                        upd_r = np.ascontiguousarray(upd_full[:, tp_rank * _KV_HEADS : tp_rank * _KV_HEADS + 1, :])
                else:
                    upd_r = upd_full
                qkv_cols = _Q_HEADS * _D_HEAD + 2 * _KV_HEADS * _D_HEAD
                bqkv_r = np.zeros((1, qkv_cols), dtype=np.float32).astype(mega._np_dtype())
                bout_r = np.zeros((1, _H), dtype=np.float32).astype(mega._np_dtype())

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
                if li == 0:
                    cos0, sin0 = aki["cos"], aki["sin"]
                    abt0, mask0, upd0, pos0 = abt_r, mask_r, upd_r, aki["pos_ids"]
                    if mega._FP8_KV:
                        k_scale0 = np.ascontiguousarray(aki["k_scale"])
                        v_scale0 = np.ascontiguousarray(aki["v_scale"])

            abt0_swa, pos0_swa = mega._trim_block_table_and_positions(
                np.asarray(abt0),
                np.asarray(pos0).astype(np.int64),
                mega._SLIDING_WINDOW,
                mega._BLOCK_LEN,
            )
            _start_swa = np.clip(pos0_swa - mega._SLIDING_WINDOW + 1, 0, None).astype(np.float32)
            _zero_start = np.zeros((B_attn, _S_TKG), dtype=np.float32)
            swa = [
                np.ascontiguousarray(_start_swa if mega._is_sliding(li) else _zero_start) for li in range(num_layers)
            ]

            def _stack(lst):
                return np.ascontiguousarray(np.stack(lst, axis=0))

            arrays = {
                "x_shard": x_shard,
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
            if mega._FP8_KV:
                arrays["k_scale"] = k_scale0
                arrays["v_scale"] = v_scale0
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
                "top_k": mega._TOP_K,
                "eps": mega._EPS,
                "hidden_actual": mega._H_ACTUAL,
                "softmax_scale": softmax_scale,
                "gate_clamp_upper_limit": mega._GATE_CLAMP_UPPER,
                "up_clamp_upper_limit": mega._UP_CLAMP_UPPER,
                "up_clamp_lower_limit": mega._UP_CLAMP_LOWER,
            }
            if mega._FP8_KV:
                k_scale_scalar = float(np.asarray(arrays["k_scale"]).reshape(-1)[0])
                result["softmax_scale"] = softmax_scale / k_scale_scalar
            if mega._KV_UPDATE:
                result["K_cache.must_alias_input"] = result.pop("K_cache")
                result["V_cache.must_alias_input"] = result.pop("V_cache")
            return result

        # Profile-only: stub ref + accept-all comparator. The kernel still traces, compiles,
        # loads on all 128 ranks and is profiled; correctness is NOT checked (see docstring).
        torch_ref = _make_spec_stub_ref()
        comparator = mega._passthrough_comparator

        _extra_cc_args = ["--enable-ocp-compliant-scale-computation"]

        mega.CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=gpt_oss_spec_decode_mega_kernel,
            torch_ref=torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=_RANKS,
        ).run_test(
            test_config=None,
            compiler_args=mega.CompilerArgs(
                logical_nc_config=logical_nc_config,
                platform_target=platform_target,
                nki_compilation_mode=mega.NKICompilationMode.tracer,
                additional_cmd_args=_extra_cc_args,
            ),
            output_keys=(["out", "K_cache", "V_cache"] if mega._KV_UPDATE else ["out"]),
            rtol=7e-2,
            atol=1e-2,
            custom_comparator=comparator,
            inference_args=mega.InferenceArgs(
                collective_ranks=_RANKS,
                enable_determinism_check=False,
                profile_all_ranks=mega._env_enabled("VLLM_NEURON_MEGA_PROFILE_ALL_RANKS"),
                env_vars=mega._WORKER_ENV or None,
            ),
            profile_only=True,
            input_artifacts_directory=mega._REUSE_INPUT_ARTIFACTS or None,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-x"])
