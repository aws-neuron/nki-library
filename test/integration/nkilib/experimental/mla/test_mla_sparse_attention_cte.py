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

"""Integration tests for the standalone sparse MLA latent + RoPE attention kernel
(``mla_sparse_attention_cte_kernel``), kernel A of the split DeepSeek-V3.2 sparse-MLA forward.
"""

import ml_dtypes
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.experimental.mla.deepseek.mla_common_cte import MlaPrecision
from nkilib_src.nkilib.experimental.mla.deepseek.mla_sparse_attention_cte import (
    mla_sparse_attention_cte_kernel,
)
from nkilib_src.nkilib.experimental.mla.deepseek.mla_sparse_attention_cte_torch import (
    _owned_global_pos,
    mla_sparse_attention_cte_torch_ref,
)

from test.utils.common_dataclasses import CompilerArgs, ModelTestType, Platforms, prepare_model_parametrize
from test.utils.model_test_configs import no_model_configs
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

try:
    from test.integration.nkilib.experimental.mla.config.test_mla_sparse_attention_cte import (
        MODEL_PARAM_NAMES,
        mla_sparse_attention_cte_model_configs,
    )
except ImportError:
    MODEL_PARAM_NAMES = "batch, seqlen, S_kv, hidden, K, L, R, dense, precision"
    mla_sparse_attention_cte_model_configs = no_model_configs()


def _build_topk_causal_future(batch, seqlen, K, S_kv, prior_tokens=0, cp_rank=0, cp_world_size=1, interleave=1):
    """Topk exactly as the real DSA indexer emits it -- PHYSICAL cache slots, the sole
    sparse-flow topk builder.

    For a query at GLOBAL position ``q_global = owned(cp_rank, prior + s)`` the topk holds the
    causal physical slots (those whose GLOBAL position owned(slot) <= q_global), taking the K
    nearest, then padding to K with FUTURE slots (global > q_global) from the indexer's -inf tie
    region. The sparse kernel causal-re-masks those fillers so the query attends only its true
    causal prefix. ``cp_world_size == 1`` is the non-CP case: physical slot == global position, so
    this reduces to the original contiguous builder.

    numpy, not Python lists: keeps it O(seqlen * S_kv) so 128k+ caches build in seconds."""
    slots = np.arange(S_kv, dtype=np.int64)
    S_kv_local = S_kv // cp_world_size
    # GLOBAL position of every physical cache slot (identity when cp_world_size == 1).
    key_global = _owned_global_pos(slots // S_kv_local, slots % S_kv_local, interleave, cp_world_size)
    order = np.argsort(key_global, kind="stable")  # physical slots by ascending global position
    kg_sorted = key_global[order]
    topk = np.zeros((batch, seqlen, K), dtype=np.int32)
    for b in range(batch):
        for s in range(seqlen):
            q_global = int(_owned_global_pos(cp_rank, prior_tokens + s, interleave, cp_world_size))
            n_valid = int(np.searchsorted(kg_sorted, q_global, side="right"))  # causal slots
            if n_valid >= K:
                sel = order[n_valid - K : n_valid]  # enough valid keys: no fillers
            else:
                n_fut = K - n_valid
                fut = order[n_valid : n_valid + n_fut]
                if fut.size < n_fut:  # tiny S_kv: repeat the last future slot
                    fut = np.concatenate([fut, np.full(n_fut - fut.size, order[-1], dtype=order.dtype)])
                sel = np.concatenate([order[:n_valid], fut])
            topk[b, s, :] = sel.astype(np.int32)
    return topk


_PACK_CHUNK = 65536  # tokens per numpy packing chunk


def _generate_inputs(
    batch,
    seqlen,
    S_kv,
    hidden,
    K,
    L,
    R,
    dense=False,
    prior_tokens=0,
    cp_rank=0,
    cp_world_size=1,
    cp_kv_cache_interleave_size=1,
    precision=MlaPrecision.MX,
    packed=False,
    stash=None,
):
    np.random.seed(42)
    inputs = {
        "q_lift_hbm": np.random.randn(batch, seqlen, hidden, L).astype(ml_dtypes.bfloat16),
        "q_pe_hbm": np.random.randn(batch, seqlen, hidden, R).astype(ml_dtypes.bfloat16),
        "c_kv_hbm": np.random.randn(batch, S_kv, L).astype(ml_dtypes.bfloat16),
        "k_pe_hbm": np.random.randn(batch, S_kv, R).astype(ml_dtypes.bfloat16),
        # Arbitrary but shared with the ref, so validation is scale-invariant.
        "softmax_scale": float(1.0 / np.sqrt(L + R)),
        "precision": precision,
    }
    if packed:
        """
        Fuse the KV into ONE bf16 row per token, the layout the QKV kernel's paged store writes:
            [0 : L]           the L fp8 latent BYTES (2 per bf16 container)
            [L : L+2]         the per-token bf16 dequantization scale
            [L+2 : L+2+2R]    k_pe, genuine bf16
        Quantization mirrors _quantize_c_kv_per_token: per-token amax, clipped before the cast
        (the amax element lands on exactly fp8_amax, whose next e4m3 encoding is inf).
        """
        fp8_np, fp8_amax = ml_dtypes.float8_e4m3, 240.0
        row = -(-(L // 2 + 1 + R) // 8) * 8
        n_tok = batch * S_kv
        c_all = np.asarray(inputs["c_kv_hbm"]).reshape(n_tok, L)
        kp_all = np.asarray(inputs["k_pe_hbm"]).reshape(n_tok, R)
        packed_rows = np.zeros((n_tok, row), dtype=ml_dtypes.bfloat16)
        pb = packed_rows.view(np.uint8).reshape(n_tok, row * 2)
        # Chunked so peak host memory stays bounded at long S_kv.
        for beg in range(0, n_tok, _PACK_CHUNK):
            end = min(beg + _PACK_CHUNK, n_tok)
            c = c_all[beg:end].astype(np.float32)
            amax = np.maximum(np.abs(c).max(axis=1), 1e-30)
            scale = (amax / fp8_amax).astype(np.float32)
            q = np.clip(c / scale[:, None], -fp8_amax, fp8_amax).astype(fp8_np)
            pb[beg:end, 0:L] = q.view(np.uint8)
            pb[beg:end, L : L + 2] = scale.astype(ml_dtypes.bfloat16).view(np.uint8).reshape(end - beg, 2)
            pb[beg:end, L + 2 : L + 2 + R * 2] = kp_all[beg:end].astype(ml_dtypes.bfloat16).view(np.uint8)
        if stash is not None:
            # Keep the UNQUANTIZED tensors before packing overwrites c_kv_hbm.
            stash["c_kv_orig"] = inputs["c_kv_hbm"]
            stash["k_pe_orig"] = inputs["k_pe_hbm"]
        inputs["c_kv_hbm"] = packed_rows

    if dense:
        # Dense (no-indexer): attend all S_kv keys, causal-masked in-kernel. topk_indices is
        # not needed at all -- omit it entirely (the kernel arg is optional / defaults None).
        inputs["dense"] = True
    else:
        # Sparse: the causal re-mask is the default flow, so the topk always carries the
        # indexer's future fillers.
        inputs["topk_indices_hbm"] = _build_topk_causal_future(
            batch, seqlen, K, S_kv, prior_tokens or 0, cp_rank or 0, cp_world_size, cp_kv_cache_interleave_size
        )
    # prior_tokens / cp_rank are runtime [1, 1] TENSORS (not kernel constants) so one traced graph
    # serves every CP rank/step. prior_tokens None omits it (non-CP / first-chunk default 0).
    if prior_tokens is not None:
        inputs["prior_tokens_hbm"] = np.array([[prior_tokens]], dtype=np.int32)
    if cp_world_size > 1:
        # CP interleave: the cache is rank-major interleaved; pass rank + the static interleave
        # params so the kernel converts physical slots to global positions for the mask.
        inputs["cp_rank_hbm"] = np.array([[cp_rank]], dtype=np.int32)
        inputs["cp_world_size"] = cp_world_size
        inputs["cp_kv_cache_interleave_size"] = cp_kv_cache_interleave_size
    if stash is not None and "c_kv_orig" in stash:
        # A golden that never saw fp8: everything the kernel got, but with the original latents.
        stash["unquantized"] = {
            **inputs,
            "c_kv_hbm": stash["c_kv_orig"],
            "k_pe_hbm": stash["k_pe_orig"],
        }
    return inputs


def _output_tensors(kernel_input):
    batch, seqlen, hidden, L = kernel_input["q_lift_hbm"].shape
    return {"out_attn": np.zeros((batch, seqlen, hidden * L), dtype=ml_dtypes.bfloat16)}


PARAM_NAMES = "batch, seqlen, S_kv, hidden, K, L, R, prior_tokens, cp_rank, cp_world_size, cp_kv_cache_interleave_size"

SPARSE_PARAMS = [
    pytest.param(1, 64, 64 * 64, 128, 2048, 512, 64, 0, 0, 64, 2, id="ds_S64_r0_b2_sparse"),
    pytest.param(1, 128, 128 * 64, 128, 2048, 512, 64, 0, 8, 64, 4, id="ds_S128_r8_b4_sparse"),
    pytest.param(1, 256, 256 * 64, 128, 2048, 512, 64, 0, 16, 64, 8, id="ds_S256_r16_b8_sparse"),
    pytest.param(1, 512, 512 * 64, 128, 2048, 512, 64, 0, 24, 64, 16, id="ds_S512_r24_b16_sparse"),
    pytest.param(1, 1024, 1024 * 64, 128, 2048, 512, 64, 0, 32, 64, 32, id="ds_S1k_r32_b32_sparse"),
    pytest.param(1, 2048, 2048 * 64, 128, 2048, 512, 64, 0, 40, 64, 64, id="ds_S2k_r40_b64_sparse"),
]

SPARSE_PARAMS_BF16 = [
    pytest.param(1, 64, 64 * 64, 64, 2048, 512, 64, 0, 0, 64, 1, id="glm_S64_r0_b1_sparse"),
    # Isolates cp_rank != 0 with prior_tokens = 0 on the UNPACKED cache: the packed twin of this
    # geometry fails, and this says whether the cause is the packing or the CP rank handling.
    pytest.param(1, 128, 128 * 64, 64, 2048, 512, 64, 0, 8, 64, 2, id="glm_S128_r8_b2_sparse"),
    pytest.param(1, 16384, 16384 * 64, 64, 2048, 512, 64, 0, 63, 64, 128, id="glm_S16k_r63_b128_sparse"),
]

# PACKED fp8 KV cache (one fused row per token: fp8 latent + per-token scale + k_pe).
BF16_PACKED_PARAMS = [
    pytest.param(1, 128, 8192, 64, 2048, 512, 64, 0, 0, 1, 1, id="packed_glm_S128_Skv8192"),
    pytest.param(1, 256, 16384, 64, 2048, 512, 64, 0, 0, 1, 1, id="packed_glm_S256_Skv16k"),
    pytest.param(1, 1024, 131072, 64, 2048, 512, 64, 0, 0, 1, 1, id="packed_glm_S1024_Skv128k"),
    pytest.param(1, 2048, 131072, 64, 2048, 512, 64, 0, 0, 1, 1, id="packed_glm_S2048_Skv128k"),
    pytest.param(1, 2048, 131072, 64, 2048, 512, 64, 129024, 0, 1, 1, id="packed_glm_S2048_Skv128k_lastrank"),
    # APC, non-CP: a PARTIAL resident prefix, so the causal window ends mid-context and the
    # re-mask fires on a subset of keys (last-rank above masks nothing, prior=0 masks the most).
    pytest.param(1, 256, 16384, 64, 2048, 512, 64, 8192, 0, 1, 1, id="packed_glm_S256_Skv16k_apc_half"),
    pytest.param(1, 1024, 131072, 64, 2048, 512, 64, 65536, 0, 1, 1, id="packed_glm_S1024_Skv128k_apc_half"),
    pytest.param(1, 2048, 131072, 64, 2048, 512, 64, 2048, 0, 1, 1, id="packed_glm_S2048_Skv128k_apc_small"),
    pytest.param(1, 4096, 262144, 64, 2048, 512, 64, 0, 0, 1, 1, id="packed_glm_S4096_Skv256k"),
    pytest.param(1, 8192, 524288, 64, 2048, 512, 64, 0, 0, 1, 1, id="packed_glm_S8192_Skv512k"),
    pytest.param(1, 16384, 1048576, 64, 2048, 512, 64, 0, 0, 1, 1, id="packed_glm_S16384_Skv1M"),
    # The CP-interleaved sparse ladder, now on the packed cache: same geometries as the unpacked
    # rungs above, each also run with an APC prior (prior_tokens = S_kv - seqlen, the last rank's
    # offset) so a cached prefix is exercised at every shape.
    pytest.param(1, 128, 128 * 64, 64, 2048, 512, 64, 0, 8, 64, 2, id="packed_glm_S128_r8_b2_cp"),
    pytest.param(1, 128, 256 * 64, 64, 2048, 512, 64, 128, 8, 64, 2, id="packed_glm_S128_r8_b2_cp_apc"),
    pytest.param(1, 256, 256 * 64, 64, 2048, 512, 64, 0, 16, 64, 4, id="packed_glm_S256_r16_b4_cp"),
    pytest.param(1, 256, 512 * 64, 64, 2048, 512, 64, 256, 16, 64, 4, id="packed_glm_S256_r16_b4_cp_apc"),
    pytest.param(1, 512, 512 * 64, 64, 2048, 512, 64, 0, 24, 64, 8, id="packed_glm_S512_r24_b8_cp"),
    pytest.param(1, 512, 1024 * 64, 64, 2048, 512, 64, 512, 24, 64, 8, id="packed_glm_S512_r24_b8_cp_apc"),
    pytest.param(1, 1024, 1024 * 64, 64, 2048, 512, 64, 0, 32, 64, 16, id="packed_glm_S1k_r32_b16_cp"),
    pytest.param(1, 1024, 2048 * 64, 64, 2048, 512, 64, 1024, 32, 64, 16, id="packed_glm_S1k_r32_b16_cp_apc"),
    pytest.param(1, 2048, 2048 * 64, 64, 2048, 512, 64, 0, 40, 64, 32, id="packed_glm_S2k_r40_b32_cp"),
    pytest.param(1, 2048, 4096 * 64, 64, 2048, 512, 64, 2048, 40, 64, 32, id="packed_glm_S2k_r40_b32_cp_apc"),
    pytest.param(1, 4096, 4096 * 64, 64, 2048, 512, 64, 0, 48, 64, 64, id="packed_glm_S4k_r48_b64_cp"),
    pytest.param(1, 8192, 8192 * 64, 64, 2048, 512, 64, 0, 56, 64, 128, id="packed_glm_S8k_r56_b128_cp"),
]

DENSE_PARAMS = [
    pytest.param(1, 2, 128, 128, 2048, 512, 64, 0, 0, 64, 2, id="ds_S128_r0_b2_dense"),
    pytest.param(1, 8, 512, 128, 2048, 512, 64, 0, 16, 64, 8, id="ds_S512_r16_b8_dense"),
    pytest.param(1, 16, 1024, 128, 2048, 512, 64, 0, 32, 64, 16, id="ds_S1024_r32_b16_dense"),
    pytest.param(1, 32, 2048, 128, 2048, 512, 64, 0, 63, 64, 32, id="ds_S2048_r63_b32_dense"),
]

# DENSE (no-indexer) against the PACKED fp8 cache. Dense attends every slot, so the cache is
# query-independent and is dequantized once at load time rather than per query. Non-CP (W=1, I=1):
# the packed layout is what a paged deployment writes, and CP-interleaved dense rides the ladder
# below on the unpacked cache.
DENSE_PACKED_PARAMS_BF16 = [
    pytest.param(1, 2, 128, 64, 2048, 512, 64, 0, 0, 1, 1, id="packed_glm_S128_dense"),
    pytest.param(1, 8, 512, 64, 2048, 512, 64, 0, 0, 1, 1, id="packed_glm_S512_dense"),
    pytest.param(1, 16, 1024, 64, 2048, 512, 64, 0, 0, 1, 1, id="packed_glm_S1024_dense"),
    pytest.param(1, 32, 2048, 64, 2048, 512, 64, 0, 0, 1, 1, id="packed_glm_S2048_dense"),
    pytest.param(1, 32, 2048, 64, 2048, 512, 64, 1024, 0, 1, 1, id="packed_glm_S2048_prior_dense"),
    pytest.param(1, 16, 1024, 64, 2048, 512, 64, 512, 0, 1, 1, id="packed_glm_S1024_prior_dense"),
    # CP-interleaved dense on the packed cache, with and without an APC prior.
    pytest.param(1, 8, 512, 64, 2048, 512, 64, 0, 16, 64, 8, id="packed_glm_S512_r16_b8_dense_cp"),
    pytest.param(1, 8, 16 * 64, 64, 2048, 512, 64, 8, 16, 64, 8, id="packed_glm_S512_r16_b8_dense_cp_apc"),
    pytest.param(1, 16, 1024, 64, 2048, 512, 64, 0, 32, 64, 16, id="packed_glm_S1024_r32_b16_dense_cp"),
    pytest.param(1, 32, 2048, 64, 2048, 512, 64, 0, 63, 64, 32, id="packed_glm_S2048_r63_b32_dense_cp"),
    pytest.param(1, 16, 32 * 64, 64, 2048, 512, 64, 16, 63, 64, 32, id="packed_glm_S2048_r63_b32_dense_cp_apc"),
]

DENSE_PARAMS_BF16 = [
    # UNPACKED control, two rungs only (see SPARSE_PARAMS_BF16).
    pytest.param(1, 2, 128, 64, 2048, 512, 64, 0, 0, 64, 2, id="glm_S128_r0_b2_dense"),
    pytest.param(1, 32, 2048, 64, 2048, 512, 64, 0, 63, 64, 32, id="glm_S2048_r63_b32_dense"),
]


@pytest_test_metadata(name="SparseMlaLatentAttnCteStandalone", tags=["model"])
@pytest_marks(["attention", "mla_sparse_attention_cte_kernel"])
class TestSparseMlaLatentAttnCte:
    """Standalone sparse MLA latent + RoPE attention (kernel A).

    ``tags=["model"]`` is what enrols this file as its own step in the model-regression
    lane, which is the only lane that emits results under the model run type the
    kernel-to-model dashboard reads. It sits here rather than on the model-driven class
    below because one metadata annotation per file is enforced.
    """

    def _run(
        self,
        test_manager,
        platform_target,
        batch,
        seqlen,
        S_kv,
        hidden,
        K,
        L,
        R,
        dense=False,
        prior_tokens=0,
        cp_rank=0,
        cp_world_size=1,
        cp_kv_cache_interleave_size=1,
        precision=MlaPrecision.MX,
        packed=False,
        unquantized_ref=False,
        lnc=2,
    ):
        stash: dict = {}

        def input_generator(test_config):
            return _generate_inputs(
                batch,
                seqlen,
                S_kv,
                hidden,
                K,
                L,
                R,
                dense=dense,
                prior_tokens=prior_tokens,
                cp_rank=cp_rank,
                cp_world_size=cp_world_size,
                cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
                precision=precision,
                packed=packed,
                stash=stash if unquantized_ref else None,
            )

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=mla_sparse_attention_cte_kernel,
            torch_ref=torch_ref_wrapper(mla_sparse_attention_cte_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_output_tensors,
        )
        if unquantized_ref:

            def _unquantized_comparator(golden_dict, output_tensors):
                """Golden from the UNQUANTIZED c_kv / k_pe, so the reported diff is fp8's real cost.

                The framework's own golden comes from the ref fed the packed cache, which dequantizes
                the very bytes the kernel reads -- that proves the arithmetic but says nothing about
                accuracy. This recomputes the reference from the pre-packing tensors.
                """
                u = stash["unquantized"]

                def _t(x, dt=torch.float32):
                    # The stashed values may be numpy or already-converted torch tensors depending
                    # on where the harness intercepts the input dict.
                    if x is None:
                        return None
                    if isinstance(x, torch.Tensor):
                        return x.to(dt)
                    return torch.from_numpy(np.asarray(x).astype(np.float32)).to(dt)

                ref = mla_sparse_attention_cte_torch_ref(
                    q_lift_hbm=_t(u["q_lift_hbm"]),
                    q_pe_hbm=_t(u["q_pe_hbm"]),
                    c_kv_hbm=_t(u["c_kv_hbm"]),
                    k_pe_hbm=_t(u["k_pe_hbm"]),
                    softmax_scale=u["softmax_scale"],
                    # dense omits topk_indices entirely.
                    topk_indices_hbm=_t(u.get("topk_indices_hbm"), torch.int32),
                    dense=u.get("dense", False),
                    prior_tokens_hbm=_t(u.get("prior_tokens_hbm"), torch.int32),
                    precision=u["precision"],
                )
                # numpy golden: the harness's comparison path calls .astype on it.
                return {"out_attn": ref["out_attn"].to(torch.float32).numpy()}

            framework.run_test(
                test_config=None,
                compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
                atol=0.0,
                rtol=8e-2,
                custom_comparator=_unquantized_comparator,
            )
            return
        if packed:
            # atol 0 so the reported rel diff is a real |kernel - ref| / max|ref|: the harness
            # subtracts atol before dividing, so a non-zero atol reports 0 % for anything under it.
            # The ref decodes the same fp8 bytes the kernel reads, so only bf16 arithmetic error
            # is left.
            framework.run_test(
                test_config=None,
                compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
                atol=0.0,
                rtol=5e-2,
            )
            return
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
            atol=5e-2,
            rtol=5e-2,
        )

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(PARAM_NAMES, SPARSE_PARAMS_BF16)
    def test_sparse_mla_latent_attn_cte_cp_bf16(
        self,
        test_manager,
        platform_target,
        batch,
        seqlen,
        S_kv,
        hidden,
        K,
        L,
        R,
        prior_tokens,
        cp_rank,
        cp_world_size,
        cp_kv_cache_interleave_size,
    ):
        """CP interleave with the bf16 natural-order latent layout (GLM), incl. the 128k target."""
        self._run(
            test_manager,
            platform_target,
            batch,
            seqlen,
            S_kv,
            hidden,
            K,
            L,
            R,
            prior_tokens=prior_tokens,
            cp_rank=cp_rank,
            cp_world_size=cp_world_size,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            precision=MlaPrecision.BF16,
        )

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.fast
    @pytest.mark.parametrize(PARAM_NAMES, BF16_PACKED_PARAMS[:1])
    def test_sparse_mla_latent_attn_cte_bf16_packed_fp8_fast(
        self,
        test_manager,
        platform_target,
        batch,
        seqlen,
        S_kv,
        hidden,
        K,
        L,
        R,
        prior_tokens,
        cp_rank,
        cp_world_size,
        cp_kv_cache_interleave_size,
    ):
        """Fast lane: the 8k rung only. The long rungs trace a module far too large to compile
        inside the fast lane's per-test timeout; they run in the full suite instead."""
        self._run(
            test_manager,
            platform_target,
            batch,
            seqlen,
            S_kv,
            hidden,
            K,
            L,
            R,
            prior_tokens=prior_tokens,
            cp_rank=cp_rank,
            cp_world_size=cp_world_size,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            precision=MlaPrecision.BF16,
            packed=True,
        )

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(PARAM_NAMES, BF16_PACKED_PARAMS)
    def test_sparse_mla_latent_attn_cte_bf16_packed_fp8(
        self,
        test_manager,
        platform_target,
        batch,
        seqlen,
        S_kv,
        hidden,
        K,
        L,
        R,
        prior_tokens,
        cp_rank,
        cp_world_size,
        cp_kv_cache_interleave_size,
    ):
        """PACKED fp8 KV cache read path: one fused bf16 row per token holding the fp8 latent, its
        per-token scale and k_pe, so the topk gather is a single dma_copy per token. Exercises the
        mixed-dtype MM2 (fp8 moving operand against the bf16 p_t stationary) and both dequant
        folds -- the per-key scale into the pre-softmax scores, and into p after row_sum."""
        self._run(
            test_manager,
            platform_target,
            batch,
            seqlen,
            S_kv,
            hidden,
            K,
            L,
            R,
            prior_tokens=prior_tokens,
            cp_rank=cp_rank,
            cp_world_size=cp_world_size,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            precision=MlaPrecision.BF16,
            packed=True,
        )

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.fast
    @pytest.mark.parametrize(PARAM_NAMES, BF16_PACKED_PARAMS[:1])
    def test_sparse_mla_latent_attn_cte_bf16_packed_fp8_vs_unquantized(
        self,
        test_manager,
        platform_target,
        batch,
        seqlen,
        S_kv,
        hidden,
        K,
        L,
        R,
        prior_tokens,
        cp_rank,
        cp_world_size,
        cp_kv_cache_interleave_size,
    ):
        """ACCURACY of the packed fp8 cache: compares against a golden built from the UNQUANTIZED
        latents, so the reported diff is what quantization actually costs the attention output.
        The sibling packed test compares against a golden decoded from the same fp8 bytes, which
        proves the arithmetic but is blind to quantization error by construction."""
        self._run(
            test_manager,
            platform_target,
            batch,
            seqlen,
            S_kv,
            hidden,
            K,
            L,
            R,
            prior_tokens=prior_tokens,
            cp_rank=cp_rank,
            cp_world_size=cp_world_size,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            precision=MlaPrecision.BF16,
            packed=True,
            unquantized_ref=True,
        )

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(PARAM_NAMES, SPARSE_PARAMS)
    def test_sparse_mla_latent_attn_cte_cp_mx(
        self,
        test_manager,
        platform_target,
        batch,
        seqlen,
        S_kv,
        hidden,
        K,
        L,
        R,
        prior_tokens,
        cp_rank,
        cp_world_size,
        cp_kv_cache_interleave_size,
    ):
        """CP interleave with the bf16 natural-order latent layout (GLM), incl. the 128k target."""
        self._run(
            test_manager,
            platform_target,
            batch,
            seqlen,
            S_kv,
            hidden,
            K,
            L,
            R,
            prior_tokens=prior_tokens,
            cp_rank=cp_rank,
            cp_world_size=cp_world_size,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        )

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.fast
    @pytest.mark.parametrize(PARAM_NAMES, DENSE_PACKED_PARAMS_BF16[:1])
    def test_dense_mla_latent_attn_cte_packed_fp8_fast(
        self,
        test_manager,
        platform_target,
        batch,
        seqlen,
        S_kv,
        hidden,
        K,
        L,
        R,
        prior_tokens,
        cp_rank,
        cp_world_size,
        cp_kv_cache_interleave_size,
    ):
        self._run(
            test_manager,
            platform_target,
            batch,
            seqlen,
            S_kv,
            hidden,
            K,
            L,
            R,
            dense=True,
            prior_tokens=prior_tokens,
            cp_rank=cp_rank,
            cp_world_size=cp_world_size,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            precision=MlaPrecision.BF16,
            packed=True,
        )

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(PARAM_NAMES, DENSE_PACKED_PARAMS_BF16)
    def test_dense_mla_latent_attn_cte_packed_fp8(
        self,
        test_manager,
        platform_target,
        batch,
        seqlen,
        S_kv,
        hidden,
        K,
        L,
        R,
        prior_tokens,
        cp_rank,
        cp_world_size,
        cp_kv_cache_interleave_size,
    ):
        """DENSE path against the PACKED fp8 cache. Dense has no topk gather, so the resident cache
        is dequantized ONCE at load time and the per-query path is identical to the unpacked one --
        no per-key scale on the scores and none on the p_t eviction. MM2's keys-on-partition
        stationary comes straight off the packed row by DMA, needing no transposes at all."""
        self._run(
            test_manager,
            platform_target,
            batch,
            seqlen,
            S_kv,
            hidden,
            K,
            L,
            R,
            dense=True,
            prior_tokens=prior_tokens,
            cp_rank=cp_rank,
            cp_world_size=cp_world_size,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            precision=MlaPrecision.BF16,
            packed=True,
        )

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.fast
    @pytest.mark.parametrize(PARAM_NAMES, DENSE_PACKED_PARAMS_BF16[:1])
    def test_dense_mla_latent_attn_cte_packed_fp8_vs_unquantized(
        self,
        test_manager,
        platform_target,
        batch,
        seqlen,
        S_kv,
        hidden,
        K,
        L,
        R,
        prior_tokens,
        cp_rank,
        cp_world_size,
        cp_kv_cache_interleave_size,
    ):
        """ACCURACY of the packed fp8 cache on the dense path, against a golden built from the
        UNQUANTIZED latents."""
        self._run(
            test_manager,
            platform_target,
            batch,
            seqlen,
            S_kv,
            hidden,
            K,
            L,
            R,
            dense=True,
            prior_tokens=prior_tokens,
            cp_rank=cp_rank,
            cp_world_size=cp_world_size,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            precision=MlaPrecision.BF16,
            packed=True,
            unquantized_ref=True,
        )

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(PARAM_NAMES, DENSE_PARAMS_BF16)
    def test_dense_mla_latent_attn_cte_cp_bf16(
        self,
        test_manager,
        platform_target,
        batch,
        seqlen,
        S_kv,
        hidden,
        K,
        L,
        R,
        prior_tokens,
        cp_rank,
        cp_world_size,
        cp_kv_cache_interleave_size,
    ):
        self._run(
            test_manager,
            platform_target,
            batch,
            seqlen,
            S_kv,
            hidden,
            K,
            L,
            R,
            dense=True,
            prior_tokens=prior_tokens,
            cp_rank=cp_rank,
            cp_world_size=cp_world_size,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            precision=MlaPrecision.BF16,
        )

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(PARAM_NAMES, DENSE_PARAMS)
    def test_dense_mla_latent_attn_cte_cp_mx(
        self,
        test_manager,
        platform_target,
        batch,
        seqlen,
        S_kv,
        hidden,
        K,
        L,
        R,
        prior_tokens,
        cp_rank,
        cp_world_size,
        cp_kv_cache_interleave_size,
    ):
        self._run(
            test_manager,
            platform_target,
            batch,
            seqlen,
            S_kv,
            hidden,
            K,
            L,
            R,
            dense=True,
            prior_tokens=prior_tokens,
            cp_rank=cp_rank,
            cp_world_size=cp_world_size,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        )


@pytest_marks(["attention", "cte", "model", "mx", "mla_sparse_attention_cte_kernel"])
class TestSparseMlaLatentAttnCteModel:
    """Model-driven tests for the sparse MLA latent attention CTE kernel, organized by tier.

    Parametrized from configuration/test_vector_metadata/, which is also what feeds the
    kernel-to-model benchmark dashboard, so a config registered there is a config that runs.

    Carries no @pytest_test_metadata of its own: one annotation per file is enforced, and the
    kernel is already named by the class above.

    The ``mx`` mark does not describe this kernel's arithmetic: GLM-5.3 runs the bf16 MLA
    path. The trn3 model lane requires one of the MX marks to admit a test
    (``-m "model and (optimal or tier0) and (mx or mxfp4 or mxfp8)"``), and per the lane
    owner we carry ``mx`` for now rather than widening that group to accept ``bf16``. Do not
    read it as a precision declaration; the precision field in the config table is what
    identifies the path.
    """

    _OPTIMAL_PARAMS, _OPTIMAL_IDS = prepare_model_parametrize(
        {
            ModelTestType.OPTIMAL: mla_sparse_attention_cte_model_configs.get(ModelTestType.OPTIMAL, []),
        }
    )

    def _run_model_test(self, test_manager, platform_target, batch, seqlen, S_kv, hidden, K, L, R, dense, precision):
        # The config table carries the precision by NAME so it stays free of kernel imports;
        # resolve it here, where the kernel is already a dependency.
        TestSparseMlaLatentAttnCte()._run(
            test_manager,
            platform_target,
            batch,
            seqlen,
            S_kv,
            hidden,
            K,
            L,
            R,
            cp_world_size=64,
            dense=dense,
            precision=MlaPrecision[precision],
        )

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.optimal
    @pytest.mark.parametrize(MODEL_PARAM_NAMES, _OPTIMAL_PARAMS, ids=_OPTIMAL_IDS)
    def test_optimal(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        seqlen,
        S_kv,
        hidden,
        K,
        L,
        R,
        dense,
        precision,
    ):
        """OPTIMAL: the model configs the benchmark dashboard tracks, DeepSeek-V3.2 and GLM."""
        self._run_model_test(test_manager, platform_target, batch, seqlen, S_kv, hidden, K, L, R, dense, precision)

    # GLM-5.3 at a 16k GLOBAL context, the same silicon shared two ways. S_kv is the gathered
    # global cache either way, so only the query count and the shard count move: under LNC1 each
    # physical core is its own rank, so the context splits 128 ways instead of 64. TP stays 1, so
    # the head count per rank is 64 in both legs and the two rows are directly comparable.
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(
        "seqlen, S_kv, cp_world_size, lnc",
        [
            pytest.param(128, 16384, 128, 1, id="sparseattn_glm_S128_Sglob16k_cp128_lnc1"),
            pytest.param(256, 16384, 64, 2, id="sparseattn_glm_S256_Sglob16k_cp64_lnc2"),
        ],
    )
    def test_lnc_shard_sweep(self, test_manager: Orchestrator, platform_target, seqlen, S_kv, cp_world_size, lnc):
        TestSparseMlaLatentAttnCte()._run(
            test_manager,
            platform_target,
            1,
            seqlen,
            S_kv,
            64,
            2048,
            512,
            64,
            cp_world_size=cp_world_size,
            precision=MlaPrecision.BF16,
            lnc=lnc,
        )
