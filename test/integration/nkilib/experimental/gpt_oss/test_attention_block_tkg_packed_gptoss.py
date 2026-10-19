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

"""Sequence-packed attention on the GPT-OSS gigakernel's vendored dispatcher.

The gigakernel is a vendored snapshot, not a live import of ``core``, so its
dispatcher needs its own packed-path coverage: a green ``core`` test says nothing
about this tree.

Rather than rebuild the packed metadata here, these cases call the live runner
``_run_attn_blk_packed_integration`` with ``kernel_entry`` pointed at the vendored
dispatcher. The schedule construction, the torch reference and the tolerances are
therefore shared with the live test by construction, so the two cannot drift.

Both paths are asserted, against the same torch reference and the same tolerances,
so a divergence between them is a real difference in the dispatcher rather than in
the test. The unpacked arm is not redundant coverage of code this change leaves
alone. This change adds a ``not use_seq_packed_attention`` term to
``row_tile_q_for_attention``, so the unpacked arm is the only one that still reaches
the dispatcher's row-tiled-Q branch, and it is what goes red if a later change
alters that branch numerically. Deleting the new term sends the packed arm red on
trn3, on the index/data partition-match assert; the same four cases still compile
on trn3_a0, so that probe binds on one target and not the other.

The vendored tree re-roots ``common_types``, so its ``QuantizationType`` is a
different class than the live one the shared config carries and its membership
checks reject an equal-named member from there. Hence
``quantization_type_override``: the kernel-bound arguments must come from the
tree that validates them.

c128 only. The c512 variant differs from c128 in exactly one constant that packing
reads -- ``_FA_TILE_SIZE`` is 6K there against 8K here -- and the shared builders
hardcode 8K, so c512 packing arrives untestable through this harness. How much
else a c512 port would need is not measured here; the two dispatchers are not the
same length. ``test_packed_fa_tile_size_matches_builder_assumption`` below pins the
tile-size coupling so it cannot drift silently.
"""

from dataclasses import replace

import pytest
from nkilib_src.nkilib.models.gpt_oss.c128_giga_kernel.core.utils.common_types import (
    QuantizationType as QuantizationTypeC128,
)
from nkilib_src.nkilib.models.gpt_oss.c128_giga_kernel.experimental.transformer.attention_block_tkg import (
    attention_block_tkg as attention_block_tkg_c128,
)

from test.integration.nkilib.experimental.transformer.test_attention_block_tkg import (
    _ATTN_BLK_FP8_PACKED_AUTO_ALLOC_LNC1_CASES,
    _build_attn_blk_lnc1_schedule,
    _make_attn_blk_fp8_packed_auto_alloc_cfg,
    _run_attn_blk_packed_integration,
)
from test.utils.common_dataclasses import Platforms
from test.utils.test_orchestrator import Orchestrator


@pytest.fixture(autouse=True)
def _reset_vendored_trace_state():
    """Clear the vendored kernel's trace-scoped module state before every item here.

    These tests call ``attention_block_tkg`` as the OUTERMOST ``@nki.jit`` entry, and that is
    exactly the case the kernel's own contract does not cover: ``reset_trace_state``'s docstring
    says ``attention_block_tkg`` and ``moe_block_tkg`` deliberately do NOT reset, because the
    decode layer calls them as sub-kernels and clearing there would wipe the enclosing trace's
    prefetch FIFOs. So when they are entered directly, nobody resets.

    That matters because two of the caches are keyed on ``id()`` of trace-local tensors
    (``_ABT_FOLD_CACHE``, ``_ABT_CACHE``) and CPython recycles addresses, so a second trace in one
    process can hit a dead trace's entry. The failure is PACKED-ONLY and does not announce itself:
    earlier in this work it inflated packed cells by up to 6.76% while every dense cell stayed
    within 0.9%, and one cell was inflated 1.78% WHILE PASSING its accuracy check. A dense arm is
    not a canary and a green assert is not proof.

    This file parametrizes many items against that entry, so under ordinary pytest worker reuse
    they share the caches. Resetting here makes the isolation a property of the test rather than a
    requirement on whoever invokes it -- a future reader has no way to know to run one process per
    item.
    """
    from nkilib_src.nkilib.models.gpt_oss.c128_giga_kernel.experimental.trace_caches import (
        reset_trace_state,
    )

    reset_trace_state()
    yield
    # Reset on the way out too: an item that raises part-way through never cleans up after itself,
    # and the next item would inherit whatever it left behind.
    reset_trace_state()


@pytest.mark.fast
def test_packed_fa_tile_size_matches_builder_assumption():
    """Pin the FA tile size the shared packed builders assume against this tree's.

    ``_FA_TILE_SIZE`` is resolved per gigakernel variant -- c128 uses 8K, c512 uses
    6K -- while the builders in ``test_attention_tkg_seq_packed`` derive tile counts,
    mask widths and terminal offsets from a literal ``FA_TILE_SIZE``. On a mismatch
    the built mask width disagrees with the kernel's ``fa_n_sprior_tile`` and the
    kernel rejects it, so a future c512 port needs those builders parametrized rather
    than the constant edited. Editing it to 6K also fails silently at the
    ``block_len=64`` these cases use: the builders' own
    ``FA_TILE_SIZE // (P_MAX * block_len)`` fold count reaches 0, the mask-building
    loop never runs, and the mask stays all-zero. At their ``DEFAULT_BLOCK_LEN`` of 16
    the count is 3, so that mode is block-length specific. Assert the coupling here
    instead of meeting it as a trace-time error.
    """
    from nkilib_src.nkilib.models.gpt_oss.c128_giga_kernel.core.attention.attention_tkg_utils import (
        _FA_TILE_SIZE,
    )

    from test.integration.nkilib.core.attention.test_attention_tkg_seq_packed import FA_TILE_SIZE

    assert _FA_TILE_SIZE == FA_TILE_SIZE, (
        f"c128 gigakernel _FA_TILE_SIZE={_FA_TILE_SIZE} but the packed metadata "
        f"builders assume FA_TILE_SIZE={FA_TILE_SIZE}"
    )


@pytest.mark.fast
@pytest.mark.parametrize("use_seq_packed_attention", [True, False], ids=["packed", "unpacked"])
@pytest.mark.parametrize(
    "skip_output_projection,test_sink",
    [
        (False, True),
        (True, False),
    ],
    ids=[
        "output_projection_with_sink",
        "raw_attention_without_sink",
    ],
)
def test_attn_blk_packed_gptoss_integration(
    test_manager: Orchestrator,
    platform_target: Platforms,
    use_seq_packed_attention: bool,
    skip_output_projection: bool,
    test_sink: bool,
):
    """Attention on the vendored c128 dispatcher, with and without a sink.

    Mirrors the live parametrization so a divergence between the two files is a
    real difference in the dispatcher, not in the test.
    """
    from test.integration.nkilib.core.attention.test_attention_tkg_seq_packed import (
        GPT_B8_LENGTHS,
        GPT_B8_PADDED_SCHEDULE,
    )

    _run_attn_blk_packed_integration(
        test_manager=test_manager,
        platform_target=platform_target,
        use_seq_packed_attention=use_seq_packed_attention,
        skip_output_projection=skip_output_projection,
        test_sink=test_sink,
        batch=8,
        lengths=GPT_B8_LENGTHS,
        schedule=GPT_B8_PADDED_SCHEDULE,
        block_len=64,
        kernel_entry=attention_block_tkg_c128,
        quantization_type_override=QuantizationTypeC128.NONE,
    )


# These four compile a b16/c32k graph and measure 208-224 CPU-s locally, against the
# release gate's --cpu-timeout 240; the heaviest pre-existing attention fast test is 209.
# Budget them explicitly so host contention cannot inflate them past a shared default.
@pytest.mark.cpu_timeout(480)
@pytest.mark.fast
@pytest.mark.parametrize(
    "arm",
    ["packed_grouped", "packed_ordered", "unpacked"],
)
@pytest.mark.parametrize(
    "sweep_case_name",
    ["util50_bimodal_b16_c32k_s4", "util100_uniform_b16_c32k_s4"],
    ids=["util50", "util100"],
)
def test_attn_blk_packed_gptoss_sweep_integration(
    test_manager: Orchestrator,
    platform_target: Platforms,
    arm: str,
    sweep_case_name: str,
):
    """The 50%- and 100%-utilization sweep schedules, three accumulator arms.

    The two utilizations are the packed path's best and worst case: util50 has the
    length variance packing exists to exploit, while util100 is uniform full
    context whose packed rows are already full, so packing can only add routing
    overhead. Correctness must hold at both, since utilization changes how many
    slots carry real work and therefore which routing paths execute.

    ``packed_grouped`` and ``packed_ordered`` differ only by whether the two
    accumulator route tables are supplied, so the pair is the accumulator A/B on
    one binary and one schedule -- and it keeps a caller on the ordered merge,
    which the grouped default would otherwise leave uncovered on this tree.
    """
    from test.integration.nkilib.core.attention.test_attention_tkg_seq_packed import (
        GENERATED_PACKED_SWEEP_CASES,
    )

    case = next(c for c in GENERATED_PACKED_SWEEP_CASES if c["name"] == sweep_case_name)

    _run_attn_blk_packed_integration(
        test_manager=test_manager,
        platform_target=platform_target,
        use_seq_packed_attention=arm != "unpacked",
        skip_output_projection=True,
        test_sink=False,
        batch=len(case["lengths"]),
        lengths=case["lengths"],
        schedule=case["schedule"],
        # 64-token cache blocks rather than the sweep's default 16; the lengths
        # and the packing schedule are unchanged.
        block_len=64,
        kernel_entry=attention_block_tkg_c128,
        quantization_type_override=QuantizationTypeC128.NONE,
        use_grouped_accumulator=arm == "packed_grouped",
    )


_GPTOSS_LNC1_128K_RATIO_CASE_NAMES = (
    # Lower utilizations, where packing actually eliminates tile work: 0.50x, 0.62x and
    # 0.75x of dense's constant 256 tiles respectively. This is the region the ratio has
    # to cross 1.0 in, if it crosses at all.
    "util50_bimodal_b16_c128k_s4",
    "util62_bimodal_b16_c128k_s4",
    "util75_bimodal_b16_c128k_s4",
    "util87_bimodal_b16_c128k_s4",
    "util90_bimodal_b16_c128k_s4",
    "util100_uniform_b16_c128k_s4",
    # Slot width at the acceptance utilization. All of s2/s4/s8 do the SAME 232
    # slot-tiles here (num_slots * max(T_max, ceil(T_total/num_slots)) with T_total=232,
    # T_max=16), so they vary the merge workspace -- staged_records reserves
    # num_groups x num_sources cells with identity padding -- without varying tile work.
    "util90_bimodal_b16_c128k_s2",
    "util90_bimodal_b16_c128k_s8",
    "util90_bimodal_b16_c128k_s16",
)
_GPTOSS_LNC1_128K_RATIO_CASES = tuple(
    next(case for case in _ATTN_BLK_FP8_PACKED_AUTO_ALLOC_LNC1_CASES if case["name"] == name)
    for name in _GPTOSS_LNC1_128K_RATIO_CASE_NAMES
)


@pytest.mark.mx
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
@pytest.mark.parametrize("use_seq_packed_attention", [True, False], ids=["packed", "unpacked"])
@pytest.mark.parametrize(
    "case",
    _GPTOSS_LNC1_128K_RATIO_CASES,
    ids=[case["name"] for case in _GPTOSS_LNC1_128K_RATIO_CASES],
)
def test_attn_blk_packed_gptoss_lnc1_128k_ratio(
    test_manager: Orchestrator,
    platform_target: Platforms,
    case,
    use_seq_packed_attention: bool,
):
    """Packed against unpacked on the vendored dispatcher at 128K, one logical NC.

    Deliberately NOT marked fast. The sibling 32K cases above measure 208-224 CPU-s
    against the release gate's 240, and these are 128K, so a fast marking would blow
    that budget rather than merely stretch it.

    Both arms live in one test so the ratio comes from a matched pair: identical
    geometry, identical torch reference, differing only in the packed schedule. The
    utilization ladder brackets 90.6%, which is where the acceptance bar sits.
    """
    cfg = _make_attn_blk_fp8_packed_auto_alloc_cfg(case["lengths"], case.get("curr_sprior"), lnc=1)
    _run_attn_blk_packed_integration(
        test_manager=test_manager,
        platform_target=platform_target,
        use_seq_packed_attention=use_seq_packed_attention,
        skip_output_projection=False,
        test_sink=True,
        batch=cfg.batch,
        lengths=case["lengths"],
        schedule=_build_attn_blk_lnc1_schedule(case) if use_seq_packed_attention else None,
        block_len=cfg.block_len,
        cfg=cfg,
        kernel_entry=attention_block_tkg_c128,
        quantization_type_override=QuantizationTypeC128.NONE,
    )


_FA_TILING_ASSERT_TEXT = "Packed attention requires FA prior-sequence tiling."


@pytest.mark.fast
@pytest.mark.mx
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
def test_packed_without_fa_s_prior_tiling_is_rejected_at_trace_time(
    test_manager: Orchestrator,
    platform_target: Platforms,
):
    """Pin the blocker that made packing unreachable from the gigakernel decode entry.

    This is the POSITIVE control for that claim. The decode layer used to resolve
    ``enable_fa_s_prior_tiling`` to False for dense and packed alike, and the packed
    branch rejects that combination -- so packed metadata could not get through at any
    context length. Without this test the claim rests on reading the assert rather than
    on seeing it fire, and an assert nobody has triggered is exactly the kind of guard
    that turns out to be unreachable.

    Cheap despite the geometry: ``kernel_assert`` runs during tracing, so this fails
    long before anything is compiled.

    Deliberately asserts on the MESSAGE, not merely that something raised. Packed
    validation has a dozen preconditions and several would also raise here; matching the
    text is what makes this a test of the FA-tiling requirement rather than of packed
    validation in general.
    """
    from test.integration.nkilib.core.attention.test_attention_tkg_seq_packed import (
        GENERATED_PACKED_SWEEP_CASES,
    )

    case = next(c for c in GENERATED_PACKED_SWEEP_CASES if c["name"] == "util50_bimodal_b16_c32k_s4")
    cfg = replace(
        _make_attn_blk_fp8_packed_auto_alloc_cfg(case["lengths"]),
        enable_fa_s_prior_tiling=False,
    )

    with pytest.raises(Exception) as excinfo:  # noqa: PT011 -- the message is the assertion
        _run_attn_blk_packed_integration(
            test_manager=test_manager,
            platform_target=platform_target,
            use_seq_packed_attention=True,
            skip_output_projection=False,
            test_sink=True,
            batch=cfg.batch,
            lengths=case["lengths"],
            schedule=case["schedule"],
            block_len=cfg.block_len,
            cfg=cfg,
            kernel_entry=attention_block_tkg_c128,
            quantization_type_override=QuantizationTypeC128.NONE,
        )

    # Walk the cause chain: the framework may wrap the trace-time AssertionError.
    chain, err = [], excinfo.value
    while err is not None and len(chain) < 10:
        chain.append(str(err))
        err = err.__cause__ or err.__context__
    joined = "\n".join(chain)
    assert _FA_TILING_ASSERT_TEXT in joined, (
        f"expected the packed path to reject enable_fa_s_prior_tiling=False with "
        f"{_FA_TILING_ASSERT_TEXT!r}, got:\n{joined}"
    )


_GPTOSS_BF16_RATIO_CASE = "util90_bimodal_b16_c128k_s4"


@pytest.mark.mx
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
@pytest.mark.parametrize("use_seq_packed_attention", [True, False], ids=["packed", "unpacked"])
def test_attn_blk_packed_gptoss_lnc1_128k_bf16_ratio(
    test_manager: Orchestrator,
    platform_target: Platforms,
    use_seq_packed_attention: bool,
):
    """The same matched pair at BF16 KV, to separate the dtype from packing.

    The fp8-packed arms leave an ambiguity: FP8 halves the KV bytes, and the dense path
    at 128K is plausibly byte-bound while the packed path gathers through indirect DMA
    under a descriptor cap and so gains little from fewer bytes. If that is right, part
    of dense's advantage is a dtype effect rather than a packing cost, and the honest
    comparison shifts. One utilization answers it; the whole ladder is not needed.
    """
    case = next(c for c in _ATTN_BLK_FP8_PACKED_AUTO_ALLOC_LNC1_CASES if c["name"] == _GPTOSS_BF16_RATIO_CASE)
    cfg = replace(
        _make_attn_blk_fp8_packed_auto_alloc_cfg(case["lengths"], case.get("curr_sprior"), lnc=1),
        kv_quant=False,
        fp8_packed=False,
    )
    _run_attn_blk_packed_integration(
        test_manager=test_manager,
        platform_target=platform_target,
        use_seq_packed_attention=use_seq_packed_attention,
        skip_output_projection=False,
        test_sink=True,
        batch=cfg.batch,
        lengths=case["lengths"],
        schedule=_build_attn_blk_lnc1_schedule(case) if use_seq_packed_attention else None,
        block_len=cfg.block_len,
        cfg=cfg,
        kernel_entry=attention_block_tkg_c128,
        quantization_type_override=QuantizationTypeC128.NONE,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-x"])
