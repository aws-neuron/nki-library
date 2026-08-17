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
"""Tests for ring attention backward kernel."""

import math
from typing import List, Optional

import numpy as np
import pytest
import torch
from nkilib_src.nkilib.experimental.attention.ring_attention_bwd import ring_attention_spmd_bwd
from nkilib_src.nkilib.experimental.attention.ring_attention_bwd_torch import (
    _ring_attention_spmd_bwd_full,
    compute_per_rank_o_lse,
    ring_attention_spmd_bwd_torch_ref,
)

from test.integration.nkilib.utils.sequence_packing_helpers import (
    cu_seqlens_to_striped_bounds,
    stripe_tensor,
)
from test.utils.common_dataclasses import (
    CompilerArgs,
    InferenceArgs,
    Platforms,
)
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_collective_framework import CollectiveUnitTestFramework

# ======================================================================
# Torch reference for sequence packing (causal + same-document).
# Shared by any packed test configs below.
# ======================================================================


def _full_attention_fwd_bwd_with_packing(q_full, k_full, v_full, dy_full, cu_seqlens, scale):
    """Full-sequence causal + same-document fwd+bwd via torch autograd.

    Args:
        q_full, k_full, v_full, dy_full: (bs_flat, total_seqlen, d) fp32 numpy.
        cu_seqlens: np.ndarray of document boundaries in [0, total_seqlen].
        scale: float softmax scale.

    Returns:
        o_full, lse_full, dq_full, dk_full, dv_full (all numpy).
    """
    q_t = torch.from_numpy(q_full.astype(np.float32)).requires_grad_(True)
    k_t = torch.from_numpy(k_full.astype(np.float32)).requires_grad_(True)
    v_t = torch.from_numpy(v_full.astype(np.float32)).requires_grad_(True)

    _, total_seqlen, _ = q_t.shape
    scores = torch.matmul(q_t, k_t.transpose(-1, -2)) * scale

    q_pos = torch.arange(total_seqlen).unsqueeze(-1)
    k_pos = torch.arange(total_seqlen).unsqueeze(0)
    causal = q_pos >= k_pos

    doc_id = torch.zeros(total_seqlen, dtype=torch.long)
    for doc_idx in range(len(cu_seqlens) - 1):
        doc_id[int(cu_seqlens[doc_idx]) : int(cu_seqlens[doc_idx + 1])] = doc_idx
    same_doc = doc_id.unsqueeze(-1) == doc_id.unsqueeze(0)
    mask = causal & same_doc

    scores_masked = torch.where(mask.unsqueeze(0), scores, torch.tensor(-float("inf")))
    probs = torch.softmax(scores_masked, dim=-1)
    probs = torch.where(torch.isnan(probs), torch.zeros_like(probs), probs)
    out_t = torch.matmul(probs, v_t)

    lse_full = torch.logsumexp(scores_masked, dim=-1).detach().numpy()

    dy_t = torch.from_numpy(dy_full.astype(np.float32))
    out_t.backward(dy_t)

    return (
        out_t.detach().numpy(),
        lse_full,
        q_t.grad.numpy(),
        k_t.grad.numpy(),
        v_t.grad.numpy(),
    )


@pytest_test_metadata(
    name="RingAttentionBwd",
    pytest_marks=["collectives", "ring_attention"],
)
class TestRingAttentionBwd:
    """Integration tests for ring attention backward kernel."""

    @pytest.mark.parametrize(
        "batch, nheads, seqlen, d, cp_degree, lnc, causal, striped, cu_seqlens_g",
        # fmt: off
        [
            # Non-causal configs
            pytest.param(1, 2, 8192, 128, 4, 2, False, False, None, id="bs1_nh2_s8192_d128_cp4_lnc2_nocausal"),
            pytest.param(2, 2, 8192, 128, 4, 2, False, False, None, id="bs2_nh2_s8192_d128_cp4_lnc2_nocausal"),
            pytest.param(3, 3, 4096, 128, 4, 2, False, False, None, id="bs3_nh3_s4096_d128_cp4_lnc2_nocausal"),
            # Causal configs
            pytest.param(1, 2, 8192, 128, 4, 2, True, False, None, id="bs1_nh2_s8192_d128_cp4_lnc2_causal"),
            pytest.param(2, 2, 8192, 128, 4, 2, True, False, None, id="bs2_nh2_s8192_d128_cp4_lnc2_causal"),
            pytest.param(3, 1, 8192, 128, 4, 2, True, False, None, id="bs3_nh1_s8192_d128_cp4_lnc2_causal"),
            pytest.param(3, 3, 4096, 128, 4, 2, True, False, None, id="bs3_nh3_s4096_d128_cp4_lnc2_causal"),
            # Striped causal configs
            pytest.param(1, 2, 8192, 128, 4, 2, True, True, None, id="bs1_nh2_s8192_d128_cp4_lnc2_striped"),
            # 32K seqlen configs (per-shard seqlen = 8192, equivalent to flash attn baseline)
            pytest.param(1, 2, 32768, 128, 4, 2, True, True, None, id="bs1_nh2_s32768_d128_cp4_lnc2_causal"),
            pytest.param(1, 2, 32768, 128, 4, 2, False, False, None, id="bs1_nh2_s32768_d128_cp4_lnc2_nocausal"),
            # 16K profiling configs (dense + sequence packing), bs=1 nh=1 cp=4 lnc=2
            # 16K profiling configs (dense + sequence packing), bs=1 nh=2 cp=4 lnc=2
            pytest.param(1, 2, 16384, 128, 4, 2, True, True, None, id="bs1_nh2_s16384_d128_cp4_lnc2_striped"),
            pytest.param(
                1,
                2,
                16384,
                128,
                4,
                2,
                True,
                True,
                [0, 1024, 4096, 6144, 10240, 12288, 15360, 16384],
                id="bs1_nh2_s16384_d128_cp4_lnc2_striped_packed",
            ),
            # 32K profiling configs (dense + sequence packing), bs=1 nh=2 cp=4 lnc=2
            pytest.param(1, 2, 32768, 128, 4, 2, True, True, None, id="bs1_nh2_s32768_d128_cp4_lnc2_striped"),
            pytest.param(
                1,
                2,
                32768,
                128,
                4,
                2,
                True,
                True,
                [0, 2048, 8192, 12288, 20480, 24576, 30720, 32768],
                id="bs1_nh2_s32768_d128_cp4_lnc2_striped_packed",
            ),
        ],
        # fmt: on
    )
    def test_ring_attention_spmd_bwd(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch: int,
        nheads: int,
        seqlen: int,
        d: int,
        cp_degree: int,
        lnc: int,
        causal: bool,
        striped: bool,
        cu_seqlens_g: Optional[List[int]],
    ):
        """Test ring attention backward pass against reference.

        When cu_seqlens_g is None -> dense path (uses _ring_attention_spmd_bwd_full).
        When cu_seqlens_g is provided -> sequence-packing path: full-sequence torch
        autograd reference, bounds derived from cu_seqlens and replicated per rank.
        Sequence packing requires causal=True and striped=True.
        """
        np.random.seed(42)
        scale = 1.0 / math.sqrt(d)
        seqlen_per_rank = seqlen // cp_degree
        bs_flat = batch * nheads
        is_packed = cu_seqlens_g is not None

        if is_packed:
            assert causal and striped, "sequence packing requires causal=True and striped=True"
            assert sum(cu_seqlens_g[i + 1] - cu_seqlens_g[i] for i in range(len(cu_seqlens_g) - 1)) == seqlen, (
                "cu_seqlens_g must span the global seqlen"
            )

        # ---- Generate Q/K/V/dY ----
        if striped:
            q_full = np.random.randn(bs_flat, seqlen, d).astype(np.float32)
            k_full = np.random.randn(bs_flat, seqlen, d).astype(np.float32)
            v_full = np.random.randn(bs_flat, seqlen, d).astype(np.float32)
            dy_full = np.random.randn(bs_flat, seqlen, d).astype(np.float32)
            q_all = [stripe_tensor(q_full, r, cp_degree, seq_axis=1) for r in range(cp_degree)]
            k_all = [stripe_tensor(k_full, r, cp_degree, seq_axis=1) for r in range(cp_degree)]
            v_all = [stripe_tensor(v_full, r, cp_degree, seq_axis=1) for r in range(cp_degree)]
            dy_all = [stripe_tensor(dy_full, r, cp_degree, seq_axis=1) for r in range(cp_degree)]
        else:
            q_all = [np.random.randn(bs_flat, seqlen_per_rank, d).astype(np.float32) for _ in range(cp_degree)]
            k_all = [np.random.randn(bs_flat, seqlen_per_rank, d).astype(np.float32) for _ in range(cp_degree)]
            v_all = [np.random.randn(bs_flat, seqlen_per_rank, d).astype(np.float32) for _ in range(cp_degree)]
            dy_all = [np.random.randn(bs_flat, seqlen_per_rank, d).astype(np.float32) for _ in range(cp_degree)]

        # ---- Compute goldens: dq/dk/dv and (for kernel inputs) o/lse ----
        if is_packed:
            cu_seqlens = np.asarray(cu_seqlens_g, dtype=np.int64)
            bound_min_local, bound_max_local = cu_seqlens_to_striped_bounds(cu_seqlens, seqlen, cp_degree)

            o_full, lse_full, dq_full, dk_full, dv_full = _full_attention_fwd_bwd_with_packing(
                q_full, k_full, v_full, dy_full, cu_seqlens, scale
            )
            o_per_rank = [stripe_tensor(o_full, r, cp_degree, seq_axis=1) for r in range(cp_degree)]
            lse_per_rank = [lse_full[:, r::cp_degree] for r in range(cp_degree)]

            # Bounds per rank: shape (bs_flat, seqlen_per_rank) fp32 — identical across ranks.
            bmin_2d = (
                np.broadcast_to(bound_min_local.reshape(1, seqlen_per_rank), (bs_flat, seqlen_per_rank))
                .astype(np.float32)
                .copy()
            )
            bmax_2d = (
                np.broadcast_to(bound_max_local.reshape(1, seqlen_per_rank), (bs_flat, seqlen_per_rank))
                .astype(np.float32)
                .copy()
            )
        else:
            q_torch = [torch.from_numpy(q) for q in q_all]
            k_torch = [torch.from_numpy(k) for k in k_all]
            v_torch = [torch.from_numpy(v) for v in v_all]
            dy_torch = [torch.from_numpy(dy) for dy in dy_all]
            _ring_attention_spmd_bwd_full(
                q_torch,
                k_torch,
                v_torch,
                dy_torch,
                scale,
                cp_degree,
                causal=causal,
                striped=striped,
            )
            o_per_rank, lse_per_rank = compute_per_rank_o_lse(
                q_torch,
                k_torch,
                v_torch,
                scale,
                cp_degree,
                causal=causal,
                striped=striped,
            )
            # compute_per_rank_o_lse returns numpy-like arrays; normalize to numpy for uniform handling below
            o_per_rank = [np.asarray(o) for o in o_per_rank]
            lse_per_rank = [np.asarray(lse) for lse in lse_per_rank]

        replica_groups = (tuple(range(cp_degree)),)

        def _to_kernel_layout(arr_per_rank, rank_id):
            """(bs_flat, seqlen_per_rank, d) -> (bs, nheads, d, seqlen_per_rank)."""
            return arr_per_rank[rank_id].transpose(0, 2, 1).reshape(batch, nheads, d, seqlen_per_rank)

        def _o_to_kernel_layout(rank_id):
            """O layout mapping, handles packed (bs_flat, spr, d) and dense (bs_flat, 1, d, spr)."""
            arr = o_per_rank[rank_id]
            if arr.ndim == 3:  # packed: (bs_flat, spr, d)
                return arr.transpose(0, 2, 1).reshape(batch, nheads, d, seqlen_per_rank)
            # dense compute_per_rank_o_lse: already (bs_flat, 1, d, spr)
            return arr.reshape(batch, nheads, d, seqlen_per_rank)

        def _lse_to_kernel_layout(rank_id):
            """LSE layout: packed (bs_flat, spr) vs dense (bs_flat, 1, 128, spr/128)."""
            arr = lse_per_rank[rank_id]
            if arr.ndim == 2:  # packed: (bs_flat, spr)
                return arr.reshape(batch, nheads, seqlen_per_rank // 128, 128).transpose(0, 1, 3, 2).astype(np.float32)
            return arr.reshape(batch, nheads, 128, seqlen_per_rank // 128).astype(np.float32)

        def create_inputs(rank_id: int):
            inputs = {
                "q_ref": _to_kernel_layout(q_all, rank_id).astype(np.float16),
                "k_ref": _to_kernel_layout(k_all, rank_id).astype(np.float16),
                "v_ref": _to_kernel_layout(v_all, rank_id).astype(np.float16),
                "o_ref": _o_to_kernel_layout(rank_id).astype(np.float16),
                "dy_ref": _to_kernel_layout(dy_all, rank_id).astype(np.float16),
                "lse_ref": _lse_to_kernel_layout(rank_id),
                "use_causal_mask": causal,
                "mixed_precision": True,
                "softmax_scale": scale,
                "num_workers": cp_degree,
                "lnc_size": lnc,
                "replica_groups": replica_groups,
            }
            if striped:
                inputs["striped_attention"] = True
            if is_packed:
                inputs["bound_min"] = bmin_2d
                inputs["bound_max"] = bmax_2d
            return inputs

        env_vars = {"NEURON_RT_ULTRASERVER_MODE": "4"} if (platform_target.is_trn3() and cp_degree > 1) else None
        framework = CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=ring_attention_spmd_bwd,
            torch_ref=ring_attention_spmd_bwd_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=cp_degree,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
            rtol=1e-2,
            atol=1e-2,
            inference_args=InferenceArgs(collective_ranks=cp_degree, env_vars=env_vars),
            output_keys=["out_dq_ref", "out_dk_ref", "out_dv_ref"],
        )
