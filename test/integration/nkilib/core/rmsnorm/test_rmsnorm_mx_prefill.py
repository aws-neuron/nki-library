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

"""
Integration tests for the standalone RMSNorm[T,H] + swizzle-transpose + MX-quant prefill kernel.
"""

from typing import Any, final

import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.core.rmsnorm.rmsnorm_mx_prefill import rmsnorm_mx_prefill
from nkilib_src.nkilib.core.rmsnorm.rmsnorm_mx_prefill_torch import (
    decode_packed_output,
    reference_mx_dequant,
    rmsnorm_mx_prefill_torch_ref,
    swizzle_h_index,
)
from nkilib_src.nkilib.core.utils.common_types import RouterActFnType
from typing_extensions import override

from test.utils.common_dataclasses import (
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    Platforms,
)
from test.utils.comparators import maxAllClose
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, mark_uncacheable, torch_ref_wrapper

# fmt: off
# RMSNorm + MX-quant only (no router, no residual). (batch, seqlen, hidden, lnc); H multiple of 512.
_NO_ROUTER_PARAMS = [
    pytest.param(1, 128, 512, 1, marks=pytest.mark.fast),  # smallest: 1 token tile, 1 H512 block (fast)
    (1, 128, 3072, 1),    # running example H, 1 token tile
    (1, 256, 3072, 1),    # 2 token tiles
    (1, 384, 2048, 1),    # 3 token tiles, non-folding-aligned num_H512=4
    (1, 512, 3072, 2),    # LNC=2: 512 tokens sharded -> 256/core (2 tiles each)
    (1, 384, 2048, 2),    # LNC=2: 384 tokens -> 192/core (uneven last tile)
]
# fmt: on


def _gen_inputs(batch, seqlen, hidden, pack_scales):
    rng = np.random.default_rng(42)
    hid = (rng.standard_normal((batch, seqlen, hidden)) * 0.5).astype(dt.bfloat16)
    gamma = (rng.standard_normal((1, hidden)) * 0.2 + 1.0).astype(dt.bfloat16)
    return {"hidden_states": hid, "gamma": gamma, "eps": 1e-6, "pack_scales": pack_scales}


@pytest_test_metadata(name="RMSNorm MX Prefill", tags=["model"])
@pytest_marks(["rmsnorm", "quantization", "mx"])
@final
class TestRmsNormMxPrefill:
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize("batch, seqlen, hidden, lnc", _NO_ROUTER_PARAMS)
    def test_rmsnorm_mx_prefill(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        collector: IMetricsCollector,
        batch: int,
        seqlen: int,
        hidden: int,
        lnc: int,
    ):
        T = batch * seqlen
        pack_scales = True

        def input_generator(_):
            return _gen_inputs(batch, seqlen, hidden, pack_scales)

        @torch_ref_wrapper
        def _ref(  # noqa: ARG001 - router_*/dtype args unused on the no-router path; present to match kernel sig
            hidden_states,
            gamma,
            router_weights=None,
            router_bias=None,
            eps=1e-6,
            top_k=1,
            router_act_fn=None,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
            qmx_output_dtype=None,
            pack_scales=True,
            pack_affinities=False,
            unpadded_hidden_size=None,
            residual=None,
            emit_norm_bf16=False,
        ):
            # No router, no residual -> the unified ref returns just {"out"}, matching this descriptor.
            return rmsnorm_mx_prefill_torch_ref(hidden_states, gamma, eps=eps, pack_scales=pack_scales)

        # Folded scale layout: 4 H512 tiles packed per 128-wide block -> scale_region = n_packed*128.
        num_h512 = hidden // 512
        n_packed = (num_h512 + 3) // 4 if pack_scales else num_h512
        scale_region = n_packed * 128

        def _comparator(golden_dict, output_tensors):
            norm_golden = golden_dict["out"]  # fp32 [T, H]

            class _Validator(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    packed = np.frombuffer(inference_output.view(dtype=nl.bfloat16), dtype=dt.float8_e4m3fn).reshape(
                        T, hidden + scale_region
                    )
                    decoded = decode_packed_output(packed, T, hidden, pack_scales=pack_scales)
                    mx_golden = reference_mx_dequant(norm_golden)
                    return maxAllClose(decoded, mx_golden, rtol=5e-2, atol=1e-5, verbose=1, min_pass_rate=0.99)

            return {
                "out": CustomValidatorWithOutputTensorData(
                    validator=_Validator,
                    output_ndarray=output_tensors["out"],
                )
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=rmsnorm_mx_prefill,
            torch_ref=_ref,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=lambda _: {
                "out": np.ndarray(shape=[T, hidden + scale_region], dtype=dt.float8_e4m3fn)
            },
            collector=collector,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                logical_nc_config=lnc,
                platform_target=platform_target,
            ),
            custom_comparator=_comparator,
        )

    # fmt: off
    # (batch, seqlen, hidden, unpadded_hidden_size, E, top_k, act, lnc, use_residual, pack_affinities)
    # hidden is the padded (multiple-of-512) dim; unpadded_hidden_size is the unpadded size used for the
    # RMS mean denominator. The GPT-OSS prefill path pads 2880 -> 3072 and routes with fp16 weights.
    # use_residual adds a residual (hidden = hidden_states + residual) before the norm and returns it.
    # pack_affinities concatenates the dense [T,E] affinities into the packed row tail (downstream-gather
    # layout) instead of a standalone tensor; orthogonal to H/E/k, so it is enabled on a few representative
    # shapes only (fast, LNC=2, decode-shape, and an odd-E config that exercises the row 4-pad).
    _FUSED_PARAMS = [
        pytest.param(1, 128, 512, 512, 16, 2, RouterActFnType.SOFTMAX, 1, False, False, marks=pytest.mark.fast),  # fast
        pytest.param(1, 128, 512, 512, 16, 2, RouterActFnType.SOFTMAX, 1, False, True, marks=pytest.mark.fast),  # fast, packed affinities
        (1, 512, 2048, 2048, 16, 1, RouterActFnType.SOFTMAX, 2, False, False),  # LNC=2 fused
        (1, 512, 2048, 2048, 16, 1, RouterActFnType.SOFTMAX, 2, False, True),  # LNC=2 fused, packed affinities
        (1, 2048, 3072, 2880, 128, 4, RouterActFnType.SOFTMAX, 2, True, True),  # LNC=2 fused, padded 2880 -> 3072
        # Decode-shape A/B vs rmsnorm_router_topk_tkg: small T (decode regime), same H/E/k.
        (16, 1, 3072, 2880, 128, 4, RouterActFnType.SOFTMAX, 2, False, False),  # T=16 decode shape, LNC=2
        (16, 1, 3072, 2880, 128, 4, RouterActFnType.SOFTMAX, 2, False, True),  # T=16 decode shape, packed affinities
        (16, 1, 3072, 2880, 128, 4, RouterActFnType.SOFTMAX, 2, True, False),   # T=16 decode shape, residual add
        (1, 1, 3072, 2880, 128, 4, RouterActFnType.SOFTMAX, 2, False, False),   # T=1 decode shape, LNC=2
        (1, 2048, 7168, 7168, 256, 4, RouterActFnType.SOFTMAX, 2, True, False),    # T=1 prefill shape, residual add
        # Odd E: E*2 not a multiple of 4, so the packed row is padded up to the next multiple of 4 fp8 cols.
        (1, 256, 2048, 2048, 17, 2, RouterActFnType.SOFTMAX, 2, False, True),  # odd-E row-pad, packed affinities
    ]
    # fmt: on

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(
        "batch, seqlen, hidden, unpadded_hidden_size, expert, top_k, act, lnc, use_residual, pack_affinities",
        _FUSED_PARAMS,
    )
    def test_rmsnorm_mx_prefill_fused_router(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        collector: IMetricsCollector,
        batch: int,
        seqlen: int,
        hidden: int,
        unpadded_hidden_size: int,
        expert: int,
        top_k: int,
        act: RouterActFnType,
        lnc: int,
        use_residual: bool,
        pack_affinities: bool,
    ):
        E = expert  # local alias: E is the math symbol used throughout the body
        T = batch * seqlen
        num_h512 = hidden // 512
        n_packed = (num_h512 + 3) // 4
        scale_region = n_packed * 128
        perm = swizzle_h_index(hidden)  # ht-major slot -> original H index (pre-permute router weights)

        # Packed-affinity row geometry: affinities (E bf16 == E*2 fp8 cols) follow the scale region;
        # the total row is padded up to a multiple of 4 fp8 cols (the hidden fp32-reinterpret transpose
        # in the downstream consumer needs it). row_region == hidden + scale_region when not packed.
        _BF16_AS_FP8 = 2
        affin_off = hidden + scale_region
        row_region = ((affin_off + E * _BF16_AS_FP8 + 3) // 4) * 4 if pack_affinities else affin_off

        def input_generator(_):
            rng = np.random.default_rng(5)
            hid = (rng.standard_normal((batch, seqlen, hidden)) * 0.5).astype(dt.bfloat16)
            gamma = (rng.standard_normal((1, hidden)) * 0.2 + 1.0).astype(dt.bfloat16)
            # unpadded_hidden_size < hidden exercises the padded-mean path: both kernel and reference sum
            # squares over the full H and divide by unpadded_hidden_size, so they agree on any pad contents
            # (matches test_rmsnorm_mx_quantize_tkg.py, which also leaves the pad region random).
            # Router weights are fp16 -> kernel compute_dtype = router_weights.dtype = fp16 (GPT-OSS path).
            w = (rng.standard_normal((hidden, E)) * 0.1).astype(np.float16)
            wb = (rng.standard_normal((1, E)) * 0.1).astype(np.float16)
            inputs = {
                "hidden_states": hid,
                "gamma": gamma,
                "router_weights": w[perm],  # pre-permuted into swizzle H order
                "router_bias": wb,
                "eps": 1e-6,
                "top_k": top_k,
                "router_act_fn": act,
                "pack_scales": True,
                "pack_affinities": pack_affinities,
                "unpadded_hidden_size": unpadded_hidden_size,
            }
            if use_residual:
                inputs["residual"] = (rng.standard_normal((batch, seqlen, hidden)) * 0.5).astype(dt.bfloat16)
            return inputs

        # _ref runs before _comparator; stash the affinity golden here so the packed path (where
        # expert_affinities is not a standalone output) can still validate the row-tail affinities.
        _aff_golden_holder = {}

        # Uncacheable: the comparator reads expert_affinities from _aff_golden_holder, populated
        # only as a side effect of running this ref. A golden-cache HIT skips the ref, leaving the
        # holder empty -> comparator KeyError. mark_uncacheable forces a recompute so the stash fires.
        @mark_uncacheable
        @torch_ref_wrapper
        def _ref(
            hidden_states,
            gamma,
            router_weights=None,
            router_bias=None,
            eps=1e-6,
            top_k=1,
            router_act_fn=RouterActFnType.SIGMOID,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
            qmx_output_dtype=None,
            pack_scales=True,
            pack_affinities=False,
            unpadded_hidden_size=None,
            residual=None,
            emit_norm_bf16=False,
        ):  # noqa: ARG001 - qmx_output_dtype/pack_scales/n_group/topk_group/routed_scaling_factor unused on SIGMOID/SOFTMAX
            ref = rmsnorm_mx_prefill_torch_ref(
                hidden_states,
                gamma,
                eps=eps,
                residual=residual,
                router_weights=router_weights,
                router_bias=router_bias,
                top_k=top_k,
                router_act_fn=router_act_fn,
                # Absent means unpadded: the reference then divides by the full hidden size.
                **({} if unpadded_hidden_size is None else {"unpadded_hidden_size": unpadded_hidden_size}),
            )
            # Affinity golden is always needed by the comparator, but when pack_affinities it is NOT a
            # standalone output (it lives in the packed row), so it must not be a golden_dict key the
            # framework would match against the descriptor. Stash it in a closure holder instead.
            _aff_golden_holder["expert_affinities"] = ref["expert_affinities"]
            # Remap the shared ref's keys to this test's output names.
            golden = {
                "norm_quant_packed": ref["out"],  # used only via reference_mx_dequant in the comparator
                "expert_index": ref["expert_index"],
            }
            if not pack_affinities:
                golden["expert_affinities"] = ref["expert_affinities"]
            if use_residual:
                golden["output_residual"] = ref["out_residual"]
            return golden

        def _comparator(golden_dict, output_tensors):
            norm_golden = golden_dict["norm_quant_packed"]
            idx_golden = golden_dict["expert_index"]
            # Affinity golden comes from the holder (always set by _ref), since when packed it is not
            # a golden_dict key (no standalone expert_affinities output in that layout).
            aff_golden = _aff_golden_holder["expert_affinities"]

            class _QuantValidator(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    # The raw packed row is row_region fp8 cols wide (hidden+scale, plus the affinity
                    # tail when pack_affinities). decode_packed_output reads only [:, :hidden+scale_region].
                    packed = np.frombuffer(inference_output.view(dtype=nl.bfloat16), dtype=dt.float8_e4m3fn).reshape(
                        T, row_region
                    )
                    decoded = decode_packed_output(packed, T, hidden, pack_scales=True)
                    # Element-wise vs the reference MX quantizer; min_pass_rate tolerates the few
                    # hardware fp8-boundary flips (sim is byte-exact). See no-router test for rationale.
                    # Kernel quantizes from fp16 (compute_dtype) -> round the reference through fp16 too.
                    quant_ok = maxAllClose(
                        decoded,
                        reference_mx_dequant(norm_golden, round_dtype=np.float16),
                        rtol=5e-2,
                        atol=1e-5,
                        verbose=1,
                        min_pass_rate=0.99,
                    )
                    if not pack_affinities:
                        return quant_ok
                    # Packed: affinities live in the row tail (E bf16 at fp8 col affin_off). View the
                    # WHOLE contiguous row as bf16 (a strided column-slice .view() is not contiguous and
                    # would misread), then slice the affinity columns: fp8 col affin_off == bf16 col
                    # affin_off//_BF16_AS_FP8, width E. Validate against the golden inline (no standalone
                    # expert_affinities output exists in this layout).
                    row_bf16 = packed.view(dt.bfloat16).reshape(T, row_region // _BF16_AS_FP8)
                    aff_col = affin_off // _BF16_AS_FP8
                    aff_tail = row_bf16[:, aff_col : aff_col + E]
                    return quant_ok and _validate_affinities(aff_tail)

            # Top-K boundary ties: with large E and top_k>1 the kth vs (k+1)th logit gap is tiny for
            # a few percent of tokens. The kernel's fp16 matmul vs the fp32 reference disagree by ~1e-6,
            # flipping which expert lands at the boundary on those tokens -- a genuine MoE coin-flip, not
            # a kernel error. validate the top-K index by SET OVERLAP per token, and the affinities by COSINE similarity
            # (robust to those flips, since a flipped expert moves an affinity from one column to a neighbor while
            # the overall distribution stays directionally aligned) + an allclose pass-rate.
            _MIN_TOPK_SET_OVERLAP = 0.90
            _AFF_MIN_COS = 0.98  # MX precision budget

            def _validate_affinities(a_bf16: npt.NDArray[Any]) -> bool:
                # a_bf16: [T, E] bf16 affinities (fixed dtype, decoupled from router/compute precision).
                a = dt.static_cast(a_bf16, np.float32).reshape(T, E)
                flat_a, flat_g = a.flatten(), aff_golden.flatten()
                cos = float(np.dot(flat_a, flat_g) / (np.linalg.norm(flat_a) * np.linalg.norm(flat_g) + 1e-12))
                allclose_ok = maxAllClose(a, aff_golden, rtol=5e-2, atol=1e-3, verbose=1, min_pass_rate=0.95)
                print(f"INFO: expert_affinities cos={cos:.6f} (min {_AFF_MIN_COS}), allclose@0.95={allclose_ok}")
                return cos >= _AFF_MIN_COS and allclose_ok

            class _IdxValidator(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    a = inference_output.view(np.int32).reshape(T, top_k).astype(np.int64)
                    overlap = np.array(
                        [len(set(a[t].tolist()) & set(idx_golden[t].tolist())) / top_k for t in range(T)]
                    )
                    mean_overlap = float(overlap.mean())
                    print(
                        f"INFO: expert_index mean top-K set overlap = {mean_overlap:.4f} (min {_MIN_TOPK_SET_OVERLAP})"
                    )
                    return mean_overlap >= _MIN_TOPK_SET_OVERLAP

            class _AffValidator(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    # expert_affinities is always bf16 (fixed dtype, decoupled from router precision).
                    return _validate_affinities(inference_output.view(dt.bfloat16))

            validators = {
                "norm_quant_packed": CustomValidatorWithOutputTensorData(
                    validator=_QuantValidator, output_ndarray=output_tensors["norm_quant_packed"]
                ),
                "expert_index": CustomValidatorWithOutputTensorData(
                    validator=_IdxValidator, output_ndarray=output_tensors["expert_index"]
                ),
            }
            # Standalone affinity output only exists in the non-packed layout; when packed the
            # affinities are validated inside _QuantValidator from the packed row tail.
            if not pack_affinities:
                validators["expert_affinities"] = CustomValidatorWithOutputTensorData(
                    validator=_AffValidator, output_ndarray=output_tensors["expert_affinities"]
                )

            if use_residual:
                res_golden = golden_dict["output_residual"]  # fp32 [T, H] pre-norm sum

                class _ResidualValidator(CustomValidator):
                    @override
                    def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                        # output_residual dtype = residual.dtype = bf16; the sum is exact in bf16
                        # (input + residual both bf16), so it should match the rounded reference tightly.
                        a = dt.static_cast(inference_output.view(dt.bfloat16), np.float32).reshape(T, hidden)
                        return maxAllClose(a, res_golden, rtol=1e-2, atol=1e-5, verbose=1)

                validators["output_residual"] = CustomValidatorWithOutputTensorData(
                    validator=_ResidualValidator, output_ndarray=output_tensors["output_residual"]
                )

            return validators

        def _output_descriptor(_):
            # row_region == hidden + scale_region when not packed, or the affinity-padded width when packed.
            desc = {
                "norm_quant_packed": np.ndarray(shape=[T, row_region], dtype=dt.float8_e4m3fn),
                "expert_index": np.ndarray(shape=[T, top_k], dtype=np.int32),
            }
            # Standalone affinity output only in the non-packed layout (packed folds it into the row).
            # Always bf16 (fixed dtype, decoupled from router/compute precision).
            if not pack_affinities:
                desc["expert_affinities"] = np.ndarray(shape=[T, E], dtype=dt.bfloat16)
            if use_residual:
                desc["output_residual"] = np.ndarray(shape=[T, hidden], dtype=dt.bfloat16)
            return desc

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=rmsnorm_mx_prefill,
            torch_ref=_ref,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_output_descriptor,
            collector=collector,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                logical_nc_config=lnc,
                platform_target=platform_target,
            ),
            custom_comparator=_comparator,
        )

    # fmt: off
    # noaux_tc router fused into rmsnorm_mx_prefill. Emits dense fp32 affinities (no expert_index),
    # optionally packed into the row tail (as fp32). (batch, seqlen, hidden, E, n_group, topk_group,
    # top_k, routed_scaling_factor, lnc, pack_affinities)
    _NOAUX_PARAMS = [
        pytest.param(1, 128, 512, 32, 4, 2, 2, 2.5, 1, False, marks=pytest.mark.fast),
        pytest.param(1, 128, 512, 32, 4, 2, 2, 2.5, 1, True, marks=pytest.mark.fast),  # fast, packed fp32 affinities
        (1, 512, 2048, 256, 8, 4, 8, 2.5, 2, False),
        (1, 512, 2048, 256, 8, 4, 8, 2.5, 2, True),   # packed fp32 affinities
        (1, 128, 7168, 256, 8, 4, 8, 2.5, 2, False),
    ]
    # fmt: on

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(
        "batch, seqlen, hidden, expert, n_group, topk_group, top_k, routed_scaling_factor, lnc, pack_affinities",
        _NOAUX_PARAMS,
    )
    def test_rmsnorm_mx_prefill_noaux_tc_router(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        collector: IMetricsCollector,
        batch: int,
        seqlen: int,
        hidden: int,
        expert: int,
        n_group: int,
        topk_group: int,
        top_k: int,
        routed_scaling_factor: float,
        lnc: int,
        pack_affinities: bool,
    ):
        E = expert
        T = batch * seqlen
        num_h512 = hidden // 512
        n_packed = (num_h512 + 3) // 4
        scale_region = n_packed * 128
        # noaux packs fp32 affinities (E*4 fp8 cols) after the scale region; the total row is padded to a
        # multiple of 4 fp8 cols (already a multiple, since fp32 is 4 cols). Non-packed: row = hidden+scale.
        _FP32_AS_FP8 = 4
        affin_off = hidden + scale_region
        row_region = ((affin_off + E * _FP32_AS_FP8 + 3) // 4) * 4 if pack_affinities else affin_off
        num_h_tiles = hidden // 128
        perm = swizzle_h_index(hidden)

        def _to_hpart(w_swz):
            # Offline weight pre-arrange for the noaux CONTIGUOUS load: the kernel reshapes its [H, E]
            # input straight to [128, num_h_tiles, E] (no in-kernel permute), so the HBM bytes must be
            # in (partition p, tile t, expert e) order. w_swz is [H, E] with H = t*128 + p.
            # w_hpart[p,t,e] = w_swz[t*128+p, e], flattened back to [H, E].
            return w_swz.reshape(num_h_tiles, 128, E).transpose(1, 0, 2).reshape(hidden, E)

        def input_generator(_):
            rng = np.random.default_rng(7)
            hid = (rng.standard_normal((batch, seqlen, hidden)) * 0.5).astype(dt.bfloat16)
            gamma = (rng.standard_normal((1, hidden)) * 0.2 + 1.0).astype(dt.bfloat16)
            # The fused swizzle-transpose requires a 16-bit compute_dtype = router_weights.dtype, so the
            # Router weight is routed as bf16 here (checkpoint fp32 -> bf16). Bias stays fp32.
            w = (rng.standard_normal((hidden, E)) * 0.1).astype(dt.bfloat16)
            wb = (rng.standard_normal((1, E)) * 0.1).astype(np.float32)
            return {
                "hidden_states": hid,
                "gamma": gamma,
                # swizzle-permute (H reorder) THEN hpart (partition-transpose for the contiguous load).
                "router_weights": _to_hpart(w[perm]),
                "router_bias": wb,
                "eps": 1e-6,
                "top_k": top_k,
                "router_act_fn": RouterActFnType.NOAUX_TC,
                "n_group": n_group,
                "topk_group": topk_group,
                "routed_scaling_factor": routed_scaling_factor,
                "pack_scales": True,
                "pack_affinities": pack_affinities,
            }

        # _ref runs before _comparator; stash the affinity golden here so the packed path (where
        # expert_affinities is not a standalone output) can still validate the row-tail affinities.
        _aff_golden_holder = {}

        # Uncacheable: the comparator reads expert_affinities from _aff_golden_holder, populated
        # only as a side effect of running this ref. A golden-cache HIT skips the ref, leaving the
        # holder empty -> comparator KeyError. mark_uncacheable forces a recompute so the stash fires.
        @mark_uncacheable
        @torch_ref_wrapper
        def _ref(
            hidden_states,
            gamma,
            router_weights=None,
            router_bias=None,
            eps=1e-6,
            top_k=1,
            router_act_fn=RouterActFnType.SIGMOID,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
            qmx_output_dtype=None,
            pack_scales=True,
            pack_affinities=False,
            unpadded_hidden_size=None,
            residual=None,
            emit_norm_bf16=False,
        ):  # noqa: ARG001 - qmx_output_dtype/pack_scales dtype-only
            # Undo the hpart partition-transpose so the shared ref sees the swizzle-[H,E] it expects
            # (it un-permutes via swizzle_h_index). Inverse of _to_hpart: [128,nt,E]->transpose->[H,E].
            w_in = router_weights.numpy() if hasattr(router_weights, "numpy") else np.asarray(router_weights)
            w_swz = w_in.reshape(128, num_h_tiles, E).transpose(1, 0, 2).reshape(hidden, E)
            ref = rmsnorm_mx_prefill_torch_ref(
                hidden_states,
                gamma,
                eps=eps,
                router_weights=w_swz,
                router_bias=router_bias,
                top_k=top_k,
                router_act_fn=router_act_fn,
                n_group=n_group,
                topk_group=topk_group,
                routed_scaling_factor=routed_scaling_factor,
            )
            # Affinity golden is always needed by the comparator, but when pack_affinities it is NOT a
            # standalone output (it lives in the packed row tail), so it must not be a golden_dict key
            # the framework would match against the descriptor. Stash it in a closure holder instead.
            _aff_golden_holder["expert_affinities"] = ref["expert_affinities"]
            golden = {"norm_quant_packed": ref["out"]}
            if not pack_affinities:
                golden["expert_affinities"] = ref["expert_affinities"]
            return golden

        # noaux_tc affinities are fp32 dense [T, E]. The kernel routes from bf16 weights vs the fp32 torch
        # ref, so a few tie tokens flip group/expert selection -- and because each selected weight is
        # L1-normalized and x routed_scaling_factor, a flip moves a LARGE value to a different column,
        # denting cosine more than the smooth softmax path. Validate by cosine + an allclose pass-rate.
        _NOAUX_MIN_COS = 0.93

        def _validate_affinities(a_fp32: npt.NDArray[Any]) -> bool:
            a = a_fp32.reshape(T, E)
            aff_golden = _aff_golden_holder["expert_affinities"]
            flat_a, flat_g = a.flatten(), aff_golden.flatten()
            cos = float(np.dot(flat_a, flat_g) / (np.linalg.norm(flat_a) * np.linalg.norm(flat_g) + 1e-12))
            allclose_ok = maxAllClose(a, aff_golden, rtol=5e-2, atol=1e-3, verbose=1, min_pass_rate=0.95)
            print(f"INFO: noaux affinities cos={cos:.6f} (min {_NOAUX_MIN_COS}), allclose@0.95={allclose_ok}")
            return cos >= _NOAUX_MIN_COS and allclose_ok

        def _comparator(golden_dict, output_tensors):
            norm_golden = golden_dict["norm_quant_packed"]

            class _QuantValidator(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    packed = np.frombuffer(inference_output.view(dtype=nl.bfloat16), dtype=dt.float8_e4m3fn).reshape(
                        T, row_region
                    )
                    decoded = decode_packed_output(packed, T, hidden, pack_scales=True)
                    # Router routes from fp32 here -> the swizzle/quant compute_dtype is bf16 (no 16-bit
                    # router weight to force fp16), so round the reference quantizer through bf16.
                    quant_ok = maxAllClose(
                        decoded,
                        reference_mx_dequant(norm_golden, round_dtype=dt.bfloat16),
                        rtol=5e-2,
                        atol=1e-5,
                        verbose=1,
                        min_pass_rate=0.99,
                    )
                    if not pack_affinities:
                        return quant_ok
                    # Packed: fp32 affinities live in the row tail at fp8 col affin_off. View the WHOLE
                    # contiguous row as fp32 (fp8 col affin_off == fp32 col affin_off//4, width E), then
                    # validate inline (no standalone expert_affinities output exists in this layout).
                    row_fp32 = packed.view(np.float32).reshape(T, row_region // _FP32_AS_FP8)
                    aff_col = affin_off // _FP32_AS_FP8
                    aff_tail = row_fp32[:, aff_col : aff_col + E]
                    return quant_ok and _validate_affinities(aff_tail)

            class _AffValidator(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    # noaux_tc affinities are fp32 dense [T, E]. Delegate to the shared helper (which
                    # reads aff_golden from the holder) so this matches the packed-row validation path.
                    return _validate_affinities(inference_output.view(np.float32))

            validators = {
                "norm_quant_packed": CustomValidatorWithOutputTensorData(
                    validator=_QuantValidator, output_ndarray=output_tensors["norm_quant_packed"]
                ),
            }
            # Standalone affinity output only exists in the non-packed layout; when packed the
            # affinities are validated inside _QuantValidator from the packed row tail.
            if not pack_affinities:
                validators["expert_affinities"] = CustomValidatorWithOutputTensorData(
                    validator=_AffValidator, output_ndarray=output_tensors["expert_affinities"]
                )
            return validators

        def _output_descriptor(_):
            desc = {
                "norm_quant_packed": np.ndarray(shape=[T, row_region], dtype=dt.float8_e4m3fn),
            }
            if not pack_affinities:
                desc["expert_affinities"] = np.ndarray(shape=[T, E], dtype=np.float32)
            return desc

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=rmsnorm_mx_prefill,
            torch_ref=_ref,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_output_descriptor,
            collector=collector,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                logical_nc_config=lnc,
                platform_target=platform_target,
            ),
            custom_comparator=_comparator,
        )

    # fmt: off
    # emit_norm_bf16: also return the token-major bf16 RMSNorm output (natural H) for un-quantized
    # consumers (attention/indexer). (batch, seqlen, hidden, use_residual, lnc)
    _NORM_BF16_PARAMS = [
        pytest.param(1, 128, 512, False, 1, marks=pytest.mark.fast),  # smallest, 1 tile
        (1, 512, 2048, False, 2),    # LNC=2, multi-tile
        (1, 128, 7168, False, 2),    # no residual
        (1, 128, 7168, True, 2),     # residual add + LNC=2
    ]
    # fmt: on

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize("batch, seqlen, hidden, use_residual, lnc", _NORM_BF16_PARAMS)
    def test_rmsnorm_mx_prefill_emit_norm_bf16(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        collector: IMetricsCollector,
        batch: int,
        seqlen: int,
        hidden: int,
        use_residual: bool,
        lnc: int,
    ):
        T = batch * seqlen
        num_h512 = hidden // 512
        n_packed = (num_h512 + 3) // 4
        scale_region = n_packed * 128

        def input_generator(_):
            rng = np.random.default_rng(11)
            hid = (rng.standard_normal((batch, seqlen, hidden)) * 0.5).astype(dt.bfloat16)
            gamma = (rng.standard_normal((1, hidden)) * 0.2 + 1.0).astype(dt.bfloat16)
            inputs = {"hidden_states": hid, "gamma": gamma, "eps": 1e-6, "pack_scales": True, "emit_norm_bf16": True}
            if use_residual:
                inputs["residual"] = (rng.standard_normal((batch, seqlen, hidden)) * 0.5).astype(dt.bfloat16)
            return inputs

        @torch_ref_wrapper
        def _ref(  # noqa: ARG001 - router/dtype args unused here; present to match kernel sig
            hidden_states,
            gamma,
            router_weights=None,
            router_bias=None,
            eps=1e-6,
            top_k=1,
            router_act_fn=None,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
            qmx_output_dtype=None,
            pack_scales=True,
            pack_affinities=False,
            unpadded_hidden_size=None,
            residual=None,
            emit_norm_bf16=False,
        ):
            ref = rmsnorm_mx_prefill_torch_ref(
                hidden_states,
                gamma,
                eps=eps,
                pack_scales=pack_scales,
                residual=residual,
                emit_norm_bf16=True,
            )
            # Keys/order match the kernel return list: [norm_quant_packed, output_residual?, norm_bf16].
            # The quant validator reads the fp32 norm golden through reference_mx_dequant.
            golden = {"norm_quant_packed": ref["out"]}
            if residual is not None:
                golden["output_residual"] = ref["out_residual"]
            golden["norm_bf16"] = ref["norm_bf16"]
            return golden

        def _comparator(golden_dict, output_tensors):
            norm_golden = golden_dict["norm_quant_packed"]  # fp32 [T, H] norm, checked via MX dequant
            norm_bf16_golden = golden_dict["norm_bf16"]  # bf16 [T, H]

            class _QuantValidator(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    packed = np.frombuffer(inference_output.view(dtype=nl.bfloat16), dtype=dt.float8_e4m3fn).reshape(
                        T, hidden + scale_region
                    )
                    decoded = decode_packed_output(packed, T, hidden, pack_scales=True)
                    return maxAllClose(
                        decoded, reference_mx_dequant(norm_golden), rtol=5e-2, atol=1e-5, verbose=1, min_pass_rate=0.99
                    )

            class _NormBf16Validator(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    # bf16 norm computed fp32 then cast; should match the bf16-rounded reference tightly.
                    a = dt.static_cast(inference_output.view(dt.bfloat16), np.float32).reshape(T, hidden)
                    g = dt.static_cast(norm_bf16_golden, np.float32).reshape(T, hidden)
                    return maxAllClose(a, g, rtol=1e-2, atol=1e-5, verbose=1)

            validators = {
                "norm_quant_packed": CustomValidatorWithOutputTensorData(
                    validator=_QuantValidator, output_ndarray=output_tensors["norm_quant_packed"]
                ),
                "norm_bf16": CustomValidatorWithOutputTensorData(
                    validator=_NormBf16Validator, output_ndarray=output_tensors["norm_bf16"]
                ),
            }
            if use_residual:
                res_golden = golden_dict["output_residual"]

                class _ResidualValidator(CustomValidator):
                    @override
                    def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                        a = dt.static_cast(inference_output.view(dt.bfloat16), np.float32).reshape(T, hidden)
                        return maxAllClose(a, res_golden, rtol=1e-2, atol=1e-5, verbose=1)

                validators["output_residual"] = CustomValidatorWithOutputTensorData(
                    validator=_ResidualValidator, output_ndarray=output_tensors["output_residual"]
                )
            return validators

        def _output_descriptor(_):
            # Order MUST match the kernel's return list: [norm_quant_packed, output_residual?, norm_bf16].
            desc = {
                "norm_quant_packed": np.ndarray(shape=[T, hidden + scale_region], dtype=dt.float8_e4m3fn),
            }
            if use_residual:
                desc["output_residual"] = np.ndarray(shape=[T, hidden], dtype=dt.bfloat16)
            desc["norm_bf16"] = np.ndarray(shape=[T, hidden], dtype=dt.bfloat16)
            return desc

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=rmsnorm_mx_prefill,
            torch_ref=_ref,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_output_descriptor,
            collector=collector,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                logical_nc_config=lnc,
                platform_target=platform_target,
            ),
            custom_comparator=_comparator,
        )
