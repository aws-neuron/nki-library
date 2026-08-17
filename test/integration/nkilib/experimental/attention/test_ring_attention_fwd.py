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
"""Tests for ring attention forward kernel."""

import math
from typing import List, Optional

import ml_dtypes
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.attention.ring_attention_fwd import ring_attention_spmd_fwd
from nkilib_src.nkilib.experimental.attention.ring_attention_fwd_torch import ring_attention_spmd_fwd_torch_ref

from test.integration.nkilib.utils.sequence_packing_helpers import (
    cu_seqlens_to_striped_bounds,
)
from test.utils.common_dataclasses import (
    CompilerArgs,
    InferenceArgs,
    Platforms,
)
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_collective_framework import CollectiveUnitTestFramework

# ============================================================
# NumPy reference: attention forward
# ============================================================


class TestRingAttentionFwd:
    """Integration tests for ring attention forward kernel."""

    @pytest.mark.parametrize(
        "bs, nheads, nkv_heads, seqlen_per_rank, d, cp_degree, lnc, causal, striped, cu_seqlens_g",
        # fmt: off
        [
            # ──── Non-causal, MHA ────
            pytest.param(2, 2, 2, 4096, 128, 2, 1, False, False, None, id="nocausal_mha_seqsmall_cp2_lnc1"),
            pytest.param(2, 2, 2, 4096, 128, 2, 2, False, False, None, id="nocausal_mha_seqsmall_cp2_lnc2_even"),
            # ──── Causal contiguous, MHA ────
            pytest.param(2, 2, 2, 4096, 128, 2, 1, True, False, None, id="causal_contig_mha_seqsmall_cp2_lnc1"),
            pytest.param(2, 2, 2, 4096, 128, 2, 2, True, False, None, id="causal_contig_mha_seqsmall_cp2_lnc2_even"),
            # ──── Causal striped, MHA ────
            pytest.param(2, 2, 2, 4096, 128, 2, 1, True, True, None, id="causal_striped_mha_seqsmall_cp2_lnc1"),
            pytest.param(2, 2, 2, 4096, 128, 2, 2, True, True, None, id="causal_striped_mha_seqsmall_cp2_lnc2_even"),
            # ──── Tiny per-rank seqlen, cp=4 striped: probe whether lse drift requires large seqlen ────
            pytest.param(1, 2, 2, 512, 128, 4, 2, True, True, None, id="causal_striped_mha_seqtiny_cp4_lnc2"),
            # ──── LNC=2 odd cases (bs * nheads is odd) ────
            pytest.param(3, 1, 1, 4096, 128, 2, 2, False, False, None, id="nocausal_mha_seqsmall_cp2_lnc2_odd_bs3"),
            pytest.param(1, 3, 3, 4096, 128, 2, 2, True, False, None, id="causal_contig_mha_seqlarge_cp2_lnc2_odd"),
            # ──── Non-causal, MHA (large seqlen_per_rank, above 10k FA threshold) ────
            pytest.param(1, 1, 1, 1024 * 10, 128, 4, 1, False, False, None, id="nocausal_mha_seqlarge_cp4_lnc1"),
            pytest.param(1, 2, 2, 1024 * 10, 128, 4, 2, False, False, None, id="nocausal_mha_seqlarge_cp4_lnc2_even"),
            # ──── Causal contiguous, MHA (large seqlen_per_rank) ────
            pytest.param(1, 1, 1, 1024 * 10, 128, 4, 1, True, False, None, id="causal_contig_mha_seqlarge_cp4_lnc1"),
            pytest.param(
                1, 2, 2, 1024 * 10, 128, 4, 2, True, False, None, id="causal_contig_mha_seqlarge_cp4_lnc2_even"
            ),
            # ──── Causal striped, MHA (large seqlen_per_rank) ────
            pytest.param(1, 1, 1, 1024 * 10, 128, 4, 1, True, True, None, id="causal_striped_mha_seqlarge_cp4_lnc1"),
            pytest.param(
                1, 2, 2, 1024 * 10, 128, 4, 2, True, True, None, id="causal_striped_mha_seqlarge_cp4_lnc2_even"
            ),
            # ──── 16K profiling configs (dense + sequence packing), bs=1 nh=2 cp=4 lnc=2 ────
            pytest.param(1, 2, 2, 4096, 128, 4, 2, True, True, None, id="causal_striped_s16384_cp4_lnc2"),
            pytest.param(
                1,
                2,
                2,
                4096,
                128,
                4,
                2,
                True,
                True,
                [0, 1024, 4096, 6144, 10240, 12288, 15360, 16384],
                id="causal_striped_s16384_packed_cp4_lnc2",
            ),
            # ──── 32K profiling configs (dense + sequence packing), bs=1 nh=2 cp=4 lnc=2 ────
            pytest.param(1, 2, 2, 8192, 128, 4, 2, True, True, None, id="causal_striped_s32768_cp4_lnc2"),
            pytest.param(
                1,
                2,
                2,
                8192,
                128,
                4,
                2,
                True,
                True,
                [0, 2048, 8192, 12288, 20480, 24576, 30720, 32768],
                id="causal_striped_s32768_packed_cp4_lnc2",
            ),
            # ──── Ragged seqlen (not a multiple of 128) ────
            pytest.param(1, 1, 1, 4160, 128, 2, 1, False, False, None, id="nocausal_ragged_cp2_lnc1"),
            pytest.param(1, 2, 2, 4160, 128, 2, 2, False, False, None, id="nocausal_ragged_cp2_lnc2_even"),
            pytest.param(1, 1, 1, 4160, 128, 2, 2, False, False, None, id="nocausal_ragged_cp2_lnc2_odd_seqshard"),
            pytest.param(1, 1, 1, 576, 128, 2, 2, False, False, None, id="nocausal_ragged_cp2_lnc2_odd_singlecore"),
            pytest.param(1, 1, 1, 4160, 128, 2, 1, True, False, None, id="causal_contig_ragged_cp2_lnc1"),
            pytest.param(1, 2, 2, 4160, 128, 2, 2, True, False, None, id="causal_contig_ragged_cp2_lnc2_even"),
            pytest.param(1, 1, 1, 4160, 128, 2, 2, True, False, None, id="causal_contig_ragged_cp2_lnc2_odd"),
            pytest.param(1, 1, 1, 4160, 128, 2, 1, True, True, None, id="causal_striped_ragged_cp2_lnc1"),
            pytest.param(1, 2, 2, 4160, 128, 2, 2, True, True, None, id="causal_striped_ragged_cp2_lnc2_even"),
        ],
        # fmt: on
    )
    def test_ring_attention_spmd_fwd(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        bs: int,
        nheads: int,
        nkv_heads: int,
        seqlen_per_rank: int,
        d: int,
        cp_degree: int,
        lnc: int,
        causal: bool,
        striped: bool,
        cu_seqlens_g: Optional[List[int]],
    ):
        """Test ring attention forward pass against reference.

        When ``cu_seqlens_g`` is provided (list of global document boundaries,
        each a multiple of cp_degree), the kernel is invoked with sequence
        packing bounds and the golden applies an additional same-document mask.
        Packing requires causal=True and striped=True.
        """
        np.random.seed(42)
        scale = 1.0 / math.sqrt(d)
        is_packed = cu_seqlens_g is not None
        if is_packed:
            assert causal and striped, "sequence packing requires causal=True and striped=True"
            # Large striped+packed configs trip an internal compiler scheduler error
            # on trn2 (NCC_ISCH900). Validated on trn3_a0; skip on trn2.
            if platform_target == Platforms.TRN2 and seqlen_per_rank * cp_degree >= 16384:
                pytest.skip("large striped+packed CP configs hit NCC_ISCH900 on trn2; trn3_a0 validated")

        # The kernel expects q_h == k_h (pre-broadcasted for GQA).
        # We generate data at the KV-head granularity, then broadcast Q heads.
        q_h_per_kv_h = nheads // nkv_heads
        # bs_flat folds KV heads into batch: each (batch, kv_head) pair is independent
        bs_flat = bs * nkv_heads

        if striped:
            # Generate global data in natural position order
            seqlen = seqlen_per_rank * cp_degree
            q_global = np.random.randn(bs_flat, seqlen, d).astype(np.float32)
            k_global = np.random.randn(bs_flat, seqlen, d).astype(np.float32)
            v_global = np.random.randn(bs_flat, seqlen, d).astype(np.float32)

            # Stripe-slice: rank r gets positions [r, r+cp, r+2*cp, ...]
            q_per_rank = [q_global[:, r::cp_degree, :] for r in range(cp_degree)]
            k_per_rank = [k_global[:, r::cp_degree, :] for r in range(cp_degree)]
            v_per_rank = [v_global[:, r::cp_degree, :] for r in range(cp_degree)]
        else:
            # Contiguous: generate per-rank data independently
            q_per_rank = [np.random.randn(bs_flat, seqlen_per_rank, d).astype(np.float32) for _ in range(cp_degree)]
            k_per_rank = [np.random.randn(bs_flat, seqlen_per_rank, d).astype(np.float32) for _ in range(cp_degree)]
            v_per_rank = [np.random.randn(bs_flat, seqlen_per_rank, d).astype(np.float32) for _ in range(cp_degree)]

        replica_groups = (tuple(range(cp_degree)),)

        # Sequence packing bounds: shape (bs_flat, seqlen_per_rank, 1) fp32, identical across ranks.
        bmin_3d = None
        bmax_3d = None
        if is_packed:
            bound_min_local, bound_max_local = cu_seqlens_to_striped_bounds(
                np.asarray(cu_seqlens_g), seqlen_per_rank * cp_degree, cp_degree
            )
            bmin_3d = (
                np.broadcast_to(bound_min_local.reshape(1, seqlen_per_rank, 1), (bs_flat, seqlen_per_rank, 1))
                .astype(np.float32)
                .copy()
            )
            bmax_3d = (
                np.broadcast_to(bound_max_local.reshape(1, seqlen_per_rank, 1), (bs_flat, seqlen_per_rank, 1))
                .astype(np.float32)
                .copy()
            )

        def _to_kernel_layout_q(rank_id):
            """(bs_flat, seqlen_per_rank, d) -> (bs, nheads, d, seqlen_per_rank).

            Broadcast each KV head q_h_per_kv_h times to fill all Q heads.
            The kernel requires q_h == k_h, so after broadcast nheads == nheads.
            """
            arr = q_per_rank[rank_id]  # (bs_flat, spr, d)
            arr_4d = arr.reshape(bs, nkv_heads, seqlen_per_rank, d)  # (bs, nkv_heads, spr, d)
            arr_broadcast = np.repeat(arr_4d, q_h_per_kv_h, axis=1)  # (bs, nheads, spr, d)
            return arr_broadcast.transpose(0, 1, 3, 2)  # (bs, nheads, d, spr)

        def _to_kernel_layout_k(rank_id, data_per_rank):
            """(bs_flat, seqlen_per_rank, d) -> (bs, nheads, d, seqlen_per_rank).

            Broadcast KV heads to match Q heads (kernel requires q_h == k_h).
            K uses transposed layout (d, seqlen).
            """
            arr = data_per_rank[rank_id]  # (bs_flat, spr, d)
            arr_4d = arr.reshape(bs, nkv_heads, seqlen_per_rank, d)
            arr_broadcast = np.repeat(arr_4d, q_h_per_kv_h, axis=1)  # (bs, nheads, spr, d)
            return arr_broadcast.transpose(0, 1, 3, 2)  # (bs, nheads, d, spr)

        def _to_kernel_layout_v(rank_id, data_per_rank):
            """(bs_flat, seqlen_per_rank, d) -> (bs, nheads, seqlen_per_rank, d).

            Broadcast KV heads to match Q heads (kernel requires q_h == k_h).
            V uses non-transposed layout (seqlen, d).
            """
            arr = data_per_rank[rank_id]  # (bs_flat, spr, d)
            arr_4d = arr.reshape(bs, nkv_heads, seqlen_per_rank, d)
            arr_broadcast = np.repeat(arr_4d, q_h_per_kv_h, axis=1)  # (bs, nheads, spr, d)
            return arr_broadcast  # (bs, nheads, spr, d) — no transpose

        def create_inputs(rank_id: int):
            q_input = _to_kernel_layout_q(rank_id).astype(ml_dtypes.bfloat16)
            kernel_scale = scale

            # When testing pre-scaled Q path: multiply Q by scale on the host
            # and pass softmax_scale=1.0 so the kernel skips its own pre-scaling.
            if causal:
                q_input = (q_input.astype(np.float32) * scale).astype(ml_dtypes.bfloat16)
                kernel_scale = 1.0

            inputs = {
                "q": q_input,
                "k": _to_kernel_layout_k(rank_id, k_per_rank).astype(ml_dtypes.bfloat16),
                "v": _to_kernel_layout_v(rank_id, v_per_rank).astype(ml_dtypes.bfloat16),
                "replica_groups": replica_groups,
                "num_workers": cp_degree,
                "softmax_scale": kernel_scale,
                "use_causal_mask": causal,
                "striped_input": striped,
                "training": True,
            }
            if is_packed:
                # bmin_3d / bmax_3d already shaped (bs_flat, spr, 1) — kernel flattens
                # q, k, v to (bs_flat, ...) internally, so this shape matches.
                inputs["bound_min"] = bmin_3d
                inputs["bound_max"] = bmax_3d
            return inputs

        env_vars = {"NEURON_RT_ULTRASERVER_MODE": "4"} if (platform_target.is_trn3() and cp_degree > 1) else None
        framework = CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=ring_attention_spmd_fwd,
            torch_ref=ring_attention_spmd_fwd_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=cp_degree,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
            inference_args=InferenceArgs(collective_ranks=cp_degree, env_vars=env_vars),
            output_keys=["out_o", "out_lse"],
            atol=1e-3,
        )

    @pytest.mark.parametrize(
        "bs, nheads, seqlen_per_rank, d, cp_degree, lnc, causal, striped",
        # fmt: off
        [
            # ──── Causal contiguous with high cp_degree: ranks 0..cp-2 have fully-masked
            #      rows because their Q positions are before the K/V they receive from
            #      higher-numbered ranks. This is the config that triggered the nan-grad
            #      bug in torchtitan (small per-rank seqlen + high cp_degree + causal).
            pytest.param(1, 2, 512, 128, 4, 2, True, False, id="causal_contig_fullymask_cp4_lnc2"),
            pytest.param(1, 2, 256, 128, 4, 2, True, False, id="causal_contig_fullymask_cp4_spr256_lnc2"),
            # ──── Small head_dim (head_dim=16 case from the bug report) ────
            pytest.param(1, 2, 512, 16, 4, 2, True, False, id="causal_contig_fullymask_d16_cp4_lnc2"),
            # ──── Causal striped (also produces fully-masked rows for rank < cp_degree-1) ────
            pytest.param(1, 2, 512, 128, 4, 2, True, True, id="causal_striped_fullymask_cp4_lnc2"),
        ],
        # fmt: on
    )
    def test_ring_attention_fwd_fully_masked_rows(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        bs: int,
        nheads: int,
        seqlen_per_rank: int,
        d: int,
        cp_degree: int,
        lnc: int,
        causal: bool,
        striped: bool,
    ):
        """Test ring attention forward LSE is finite for configs with fully-masked rows.

        Regression test for the nan-grad bug: under causal ring attention, ranks whose
        Q positions are all before the incoming K/V block have fully-masked rows. Before
        the fix, these rows' LSE was the sentinel (-3.4e38), causing the backward to
        produce inf/nan gradients. After the fix, fully-masked rows get LSE=0.0.

        This test verifies:
        1. Output o is finite (no nan/inf).
        2. LSE is finite (no nan/inf) — the sentinel (-3.4e38) is neutralized.
        3. Output o and LSE match a reference that correctly zeros fully-masked rows.
        """
        np.random.seed(42)
        scale = 1.0 / math.sqrt(d)

        if striped:
            seqlen = seqlen_per_rank * cp_degree
            q_global = np.random.randn(bs * nheads, seqlen, d).astype(np.float32)
            k_global = np.random.randn(bs * nheads, seqlen, d).astype(np.float32)
            v_global = np.random.randn(bs * nheads, seqlen, d).astype(np.float32)
            q_per_rank = [q_global[:, r::cp_degree, :] for r in range(cp_degree)]
            k_per_rank = [k_global[:, r::cp_degree, :] for r in range(cp_degree)]
            v_per_rank = [v_global[:, r::cp_degree, :] for r in range(cp_degree)]
        else:
            q_per_rank = [np.random.randn(bs * nheads, seqlen_per_rank, d).astype(np.float32) for _ in range(cp_degree)]
            k_per_rank = [np.random.randn(bs * nheads, seqlen_per_rank, d).astype(np.float32) for _ in range(cp_degree)]
            v_per_rank = [np.random.randn(bs * nheads, seqlen_per_rank, d).astype(np.float32) for _ in range(cp_degree)]

        replica_groups = (tuple(range(cp_degree)),)

        def create_inputs(rank_id: int):
            q_arr = q_per_rank[rank_id]  # (bs*nheads, spr, d)
            q_4d = q_arr.reshape(bs, nheads, seqlen_per_rank, d).transpose(0, 1, 3, 2)  # (bs, nh, d, spr)
            k_4d = k_per_rank[rank_id].reshape(bs, nheads, seqlen_per_rank, d).transpose(0, 1, 3, 2)
            v_4d = v_per_rank[rank_id].reshape(bs, nheads, seqlen_per_rank, d)

            kernel_scale = scale
            if causal:
                q_4d = (q_4d.astype(np.float32) * scale).astype(np.float16)
                kernel_scale = 1.0
            else:
                q_4d = q_4d.astype(np.float16)

            return {
                "q": q_4d.astype(ml_dtypes.bfloat16),
                "k": k_4d.astype(ml_dtypes.bfloat16),
                "v": v_4d.astype(ml_dtypes.bfloat16),
                "replica_groups": replica_groups,
                "num_workers": cp_degree,
                "softmax_scale": kernel_scale,
                "use_causal_mask": causal,
                "striped_input": striped,
                "training": True,
            }

        env_vars = {"NEURON_RT_ULTRASERVER_MODE": "4"} if (platform_target.is_trn3() and cp_degree > 1) else None
        framework = CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=ring_attention_spmd_fwd,
            torch_ref=ring_attention_spmd_fwd_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=cp_degree,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
            inference_args=InferenceArgs(collective_ranks=cp_degree, env_vars=env_vars),
            output_keys=["out_o", "out_lse"],
            atol=1e-3,
        )

    @pytest.mark.parametrize(
        "bs, nheads, seqlen_per_rank, d, num_workers, total_ranks, lnc, additional_cmd_args",
        [
            # Flux: TP=4, CP=2, 8 total ranks, 4 replica groups of 2
            pytest.param(1, 6, 2304, 128, 2, 8, 1, [], id="flux_tp4_cp2_lnc1"),
            # Pipeline host is trn2.3xl, which has 8 cores with LNC1, but 4 cores if you're using LNC2.
            # TP4/CP2 doesn't work on pipeline with LNC2. Manual test only.
            # pytest.param(1, 6, 2304, 128, 2, 8, 2, [
            #     # "--internal-backend-options=--print-format=condensed",
            #     # "--internal-compiler-debug-mode=all",
            # ], id="flux_tp4_cp2_lnc2"),
        ],
    )
    def test_ring_attention_spmd_fwd_multi_group(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        bs: int,
        nheads: int,
        seqlen_per_rank: int,
        d: int,
        num_workers: int,
        total_ranks: int,
        lnc: int,
        additional_cmd_args: list,
    ):
        """Test ring attention with multiple replica groups (Flux CP layout).

        Flux uses TP=4, DP=2 with context parallelism over the DP dimension.
        This gives 8 total ranks with 4 independent replica groups:
        ((0,4), (1,5), (2,6), (3,7)).
        Each group runs ring attention with num_workers=2 independently.
        """
        np.random.seed(42)
        scale = 1.0 / math.sqrt(d)
        num_groups = total_ranks // num_workers

        # Build replica groups: Flux DP groups pattern
        # ranks [0..tp-1] are TP group 0, [tp..2*tp-1] are TP group 1
        # CP pairs: (i, i + num_groups) for i in range(num_groups)
        replica_groups = tuple(tuple(i + g * num_groups for g in range(num_workers)) for i in range(num_groups))
        # e.g. ((0,4), (1,5), (2,6), (3,7)) for num_workers=2, total_ranks=8

        # Generate independent data for each replica group
        # group_data[group_idx] = (q_per_worker, k_per_worker, v_per_worker)
        group_data = []
        for _ in range(num_groups):
            q_per_worker = [
                np.random.randn(bs * nheads, seqlen_per_rank, d).astype(np.float32) for _ in range(num_workers)
            ]
            k_per_worker = [
                np.random.randn(bs * nheads, seqlen_per_rank, d).astype(np.float32) for _ in range(num_workers)
            ]
            v_per_worker = [
                np.random.randn(bs * nheads, seqlen_per_rank, d).astype(np.float32) for _ in range(num_workers)
            ]
            group_data.append((q_per_worker, k_per_worker, v_per_worker))

        # Map global rank -> (group_idx, worker_idx_within_group)
        rank_to_group = {}
        for group_idx, group in enumerate(replica_groups):
            for worker_idx, rank in enumerate(group):
                rank_to_group[rank] = (group_idx, worker_idx)

        def create_inputs(rank_id: int):
            group_idx, worker_idx = rank_to_group[rank_id]
            q_w, k_w, v_w = group_data[group_idx]

            # (bs*nheads, spr, d) -> (bs, nheads, spr, d) for Q/K/V (non-transposed layout)
            # tp_q=True and tp_k=True let attention_cte handle transpose via dma_transpose
            q_arr = q_w[worker_idx].reshape(bs, nheads, seqlen_per_rank, d)
            k_arr = k_w[worker_idx].reshape(bs, nheads, seqlen_per_rank, d)
            v_arr = v_w[worker_idx].reshape(bs, nheads, seqlen_per_rank, d)

            return {
                "q": q_arr.astype(np.float16),
                "k": k_arr.astype(np.float16),
                "v": v_arr.astype(np.float16),
                "replica_groups": replica_groups,
                "num_workers": num_workers,
                "softmax_scale": scale,
                "use_causal_mask": False,
                "striped_input": False,
                "training": True,
                "tp_q": True,
                "tp_k": True,
            }

        framework = CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=ring_attention_spmd_fwd,
            torch_ref=ring_attention_spmd_fwd_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=total_ranks,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                logical_nc_config=lnc,
                platform_target=platform_target,
                additional_cmd_args=additional_cmd_args,
            ),
            output_keys=["out_o", "out_lse"],
        )
