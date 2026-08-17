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

# ty: ignore — vendored GPT-OSS test; heterogeneous config dict trips ty's type mapping
"""Standalone LNC=1 attention-block unit test for the GPT-OSS-120B decode shard.

This is the *attention half* of the GPT-OSS MXFP4 decode layer, exercised on its
own at the single-rank / single-NeuronCore (LNC=1) configuration we target for
trn3.3xl bring-up. It reuses the production ``attention_block_tkg`` kernel and its
established test harness (``_run_attention_block_test`` + ``AttnBlkTestConfig``);
the only thing that makes it "standalone" is the pinned GPT-OSS TP8/LNC=1 shape
ladder below.

Shard math (GPT-OSS-120B under tensor-parallel TP8):
  * 64 query heads total  -> 8 q heads  per rank  (``q_heads=8``)
  *  8 kv  heads total  -> 1 kv head   per rank  (``kv_heads=1``, GQA group 8)
  * head_dim = 64, hidden H = 2880 (padded to 3072 for the 128-tile), attention
    "local batch" B = 8, KV context S_ctx = 2048, single decode token S_tkg = 1.

Every config sets ``lnc=1`` — ``attention_block_tkg`` supports LNC in {1, 2} (its
KV-cache-update paths have explicit ``n_prgs == 1`` branches), but no prior test
pins LNC=1, so this file is the first LNC=1 coverage for the kernel. The three
configs form a progressive ladder so a failure localizes to the feature it adds:

  1. plain GQA decode (matches the existing gptoss_120b OPTIMAL model config,
     just at LNC=1);
  2. + attention sinks (``test_sink=True``) — GPT-OSS per-head softmax sink;
  3. + sliding-window attention (``use_pos_id=True`` + ``sliding_window``) — the
     even-layer SWA path, with sink kept on.

The shared runner builds all tensors (QKV/o_proj weights, paged K/V cache,
block_table, slot_mapping, cos/sin, mask, cache lengths, sink), runs the torch
reference (``AttentionBlockTkgTorchRef``) as the golden, and diffs via
``UnitTestFramework`` with cosine-similarity validation. With no Neuron device the
session resolves to compile/trace-only; on ``trn3pds-pdx10-1`` (TRN3) it executes
and numerically validates.
"""

import pytest

from test.integration.nkilib.experimental.transformer.test_attention_block_tkg import (
    _run_attention_block_test,
)
from test.integration.nkilib.experimental.transformer.test_attention_block_tkg_utils import (
    AttnBlkTestConfig,
)
from test.utils.common_dataclasses import Platforms
from test.utils.test_orchestrator import Orchestrator

# ── GPT-OSS-120B TP8 attention shard, LNC=1 ──────────────────────────────────
# Common shape shared by every config in the ladder. Kept as a dict so each
# config only spells out what it changes on top of the plain decode case.
#
# NOTE on layout: the GPT-OSS-120B model config runs this shape with
# ``transposed_in=True``/``transposed_out=True``, but that layout shards the
# hidden dim across programs and the kernel hard-requires ``LNC>=2`` for it
# (``attention_block_tkg`` asserts ``X.shape[1] >= 2`` under transposed_in). At
# LNC=1 there is only one program, so we use the plain (non-transposed) HBM
# layout — matching the existing non-transposed gptoss_120b model-config entry.
_GPTOSS_TP8_LNC1 = {
    "batch": 8,  # attention "local batch" B = 8
    "q_heads": 8,  # TP8 shard of 64 total q heads
    "kv_heads": 1,  # TP8 shard of 8 total kv heads (GQA group = 8)
    "d_head": 64,  # GPT-OSS head dim
    "H": 3072,  # padded hidden (multiple of 128)
    "H_actual": 2880,  # true GPT-OSS-120B hidden width
    "S_ctx": 2048,  # KV context length (~2k)
    "S_max_ctx": 2048,
    "S_tkg": 1,  # single decode token
    "block_len": 32,  # paged block KV
    "lnc": 1,  # <-- the whole point: single NeuronCore
}


# Progressive ladder — each step adds one GPT-OSS feature on top of the previous.
GPTOSS_TP8_LNC1_CONFIGS = [
    # 1. Plain GQA decode at LNC=1 (baseline: no sink, full attention).
    AttnBlkTestConfig(**_GPTOSS_TP8_LNC1),
    # 2. + attention sinks (per-head softmax sink column).
    AttnBlkTestConfig(**_GPTOSS_TP8_LNC1, test_sink=True),
    # 3. + sliding-window attention (even-layer SWA), sink kept on. SWA needs the
    #    in-kernel mask path (use_pos_id=True), which builds the mask directly from
    #    pos_ids/swa_start_pos_ids and bypasses the transposed_out/is_qk_swapped
    #    recomputation — so this rung drops the transposed layout and paged block
    #    KV to match the proven flat-KV SWA+sink precedent, changing only lnc->1.
    #    A short S_ctx keeps the window meaningful (window < context).
    AttnBlkTestConfig(
        batch=8,
        q_heads=8,
        kv_heads=1,
        d_head=64,
        H=3072,
        H_actual=2880,
        S_ctx=256,
        S_max_ctx=256,
        S_tkg=1,
        lnc=1,
        test_sink=True,
        use_pos_id=True,
        sliding_window=128,
    ),
]


# Not marked @pytest.mark.fast: the GPT-OSS-120B S_ctx=2048 config exceeds the
# release precommit's 240s compile CPU-timeout. Runs on shared fleet instead.
@pytest.mark.parametrize(
    "cfg",
    GPTOSS_TP8_LNC1_CONFIGS,
    ids=[cfg.test_id() for cfg in GPTOSS_TP8_LNC1_CONFIGS],
)
def test_attention_block_tkg_gptoss_lnc1(
    test_manager: Orchestrator,
    platform_target: Platforms,
    cfg: AttnBlkTestConfig,
):
    """Trace/compile (and, on TRN3 hardware, numerically validate) the GPT-OSS
    TP8 attention shard at LNC=1 against the torch reference."""
    _run_attention_block_test(
        test_manager=test_manager,
        platform_target=platform_target,
        cfg=cfg,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-x"])
