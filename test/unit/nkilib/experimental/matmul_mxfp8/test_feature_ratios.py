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

"""Ratio policies must preserve cache defaults and select actual eligible operations."""

import math
from types import SimpleNamespace
from typing import cast

import nki.language as nl
import pytest
from nkilib_src.nkilib.experimental.matmul_mxfp8 import matmul_mxfp8_blocks as blocks
from nkilib_src.nkilib.experimental.matmul_mxfp8 import matmul_mxfp8_generic_backward_kernel as backward
from nkilib_src.nkilib.experimental.matmul_mxfp8._matmul_mxfp8_generic_api import _make_feature_schedules
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_config import MatmulMxfp8KernelConfig
from nkilib_src.nkilib.experimental.moe_mxfp8 import _best_config_cache
from nkilib_src.nkilib.experimental.moe_mxfp8._config_codec import decode_config
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils import _load_apis, _load_block
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.common_dataclasses import (
    QuantScheme,
    TensorDescriptor,
    _RatioSchedule,
    _TileLocation,
)


@pytest.mark.parametrize("field", ["psum_copy_in_ratio", "strided_copy_swizzle_1x32_ratio"])
@pytest.mark.parametrize("value", [-0.01, 1.01, math.nan, math.inf, -math.inf])
def test_invalid_ratio(field, value):
    with pytest.raises(AssertionError, match=field):
        MatmulMxfp8KernelConfig(128, 512, 256, **{field: value})


def test_cache_records_use_explicit_copy_in_ratios():
    count = 0
    for index, (name, _) in enumerate(_best_config_cache._CONFIG_RECORDS):
        if name == "MatmulMxfp8KernelConfig":
            config = decode_config(index, _best_config_cache._CONFIG_SCHEMAS, _best_config_cache._CONFIG_RECORDS)
            assert config.psum_copy_in_ratio in (0.0, 1 / 3)
            assert not isinstance(config.psum_copy_in_ratio, bool)
            assert config.strided_copy_swizzle_1x32_ratio in (0.0, 1.0)
            count += 1
    assert count > 0


def _load_descriptor(k=512, quantized=False, swizzled=False, f_by_k=False, disable_dma_transpose=False):
    return SimpleNamespace(
        is_quantized=quantized,
        is_swizzled=swizzled or quantized,
        is_unswizzled_bf16=not (quantized or swizzled),
        is_f_by_k=f_by_k,
        quant_scheme=QuantScheme._1x32,
        data=_Tensor(shape=(k, 128), dtype=nl.bfloat16),
        logical_shape=(k, 128),
        sharded_logical_shape=(k, 128),
        effective_f_dim=None,
        disable_dma_transpose=disable_dma_transpose,
        fast_dma_transpose=False,
        uses_pe_swizzle=True,
        indirect_dma_vector_offset=None,
    )


def _load_one(td, schedule, k=0):
    _load_block._load_single_tensor(
        td,
        TensorDescriptor(data=cast(nl.NkiTensor, _Tensor(dtype=nl.bfloat16)), is_swizzled=True),
        k,
        0,
        0,
        0,
        512,
        128,
        128,
        strided_copy_swizzle_1x32_schedule=schedule,
    )


@pytest.mark.parametrize("ratio", [None, 0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("disable_dma_transpose", [False, True])
def test_loader_frequency_ignores_other_formats(monkeypatch, ratio, disable_dma_transpose):
    calls = []
    monkeypatch.setattr(_load_block, "get_active_sbm", lambda: None)
    monkeypatch.setattr(_load_apis, "load_tile_tensor_copy_1x32", lambda *a, **k: calls.append("tensor_copy"))
    monkeypatch.setattr(_load_apis, "load_tile_PE_Swizzle_1x32", lambda *a, **k: calls.append("pe"))
    monkeypatch.setattr(_load_apis, "_load_tile_dma_copy", lambda *a, **k: None)
    monkeypatch.setattr(_load_apis, "_load_tile_dense_scales", lambda *a, **k: None)
    schedule = _RatioSchedule(ratio) if ratio is not None else None
    for _ in range(8):
        for fmt in [{"quantized": True}, {"swizzled": True}, {"f_by_k": True}, {}]:
            _load_one(_load_descriptor(disable_dma_transpose=disable_dma_transpose, **fmt), schedule)
    assert calls.count("tensor_copy") == (int(8 * ratio) if ratio is not None else 0)
    assert calls.count("pe") == 16 - calls.count("tensor_copy")
    if schedule is not None:
        assert schedule.index == 8


def test_operand_schedules_are_independent(monkeypatch):
    choices = []
    monkeypatch.setattr(_load_block, "get_active_sbm", lambda: None)
    monkeypatch.setattr(
        _load_block,
        "load_tile",
        lambda load_loc, **kwargs: choices.append(kwargs["use_strided_copy_swizzle_1x32"]),
    )
    config = MatmulMxfp8KernelConfig(128, 512, 256, strided_copy_swizzle_1x32_ratio=0.5)
    _, lhs, rhs = _make_feature_schedules(config)
    for _ in range(4):
        _load_one(_load_descriptor(), lhs)
        _load_one(_load_descriptor(), rhs)
    assert lhs is not rhs
    assert choices[::2] == choices[1::2] == [False, True, False, True]
    assert lhs.index == rhs.index == 4


def test_split_k_tail_uses_one_schedule_selection(monkeypatch):
    choices = []
    monkeypatch.setattr(_load_block, "get_active_sbm", lambda: SimpleNamespace(alloc_stack=_Tensor))
    monkeypatch.setattr(_load_block.nisa, "tensor_copy", lambda **kwargs: None)
    monkeypatch.setattr(
        _load_block,
        "load_tile",
        lambda load_loc, **kwargs: choices.append((load_loc.tile_k, kwargs["use_strided_copy_swizzle_1x32"])),
    )
    schedule = _RatioSchedule(0.5)
    td = _load_descriptor(k=896)
    _load_one(td, schedule)
    _load_one(td, schedule, k=512)
    _load_one(td, schedule)
    assert choices == [(512, False), (256, True), (128, True), (512, False)]
    assert schedule.index == 3
    _load_one(td, schedule, k=896)
    assert schedule.index == 3


def test_non_2d_descriptor_without_sharded_shape(monkeypatch):
    calls = []
    monkeypatch.setattr(_load_block, "get_active_sbm", lambda: None)
    monkeypatch.setattr(_load_block, "load_tile", lambda *args, **kwargs: calls.append(args[0]))
    td = _load_descriptor(swizzled=True)
    td.data = _Tensor(shape=(128, 2, 128), dtype=nl.bfloat16)
    td.sharded_logical_shape = None
    _load_one(td, _RatioSchedule(1.0))
    assert len(calls) == 1


@pytest.mark.parametrize("override", [None, 0.0, 0.75])
def test_backward_forwards_phase_ratios(monkeypatch, override):
    calls = []
    monkeypatch.setattr(backward, "matmul_mxfp8", lambda **kwargs: calls.append(kwargs))
    dx = MatmulMxfp8KernelConfig(128, 512, 256, psum_copy_in_ratio=0.25, strided_copy_swizzle_1x32_ratio=0.5)
    dw = MatmulMxfp8KernelConfig(128, 512, 256, psum_copy_in_ratio=0.5, strided_copy_swizzle_1x32_ratio=1.0)
    backward.matmul_mxfp8_backward(
        None,
        None,
        None,
        input_grad_config=dx,
        weight_grad_config=dw,
        psum_copy_in_ratio=override,
        strided_copy_swizzle_1x32_ratio=override,
    )
    assert [call["psum_copy_in_ratio"] for call in calls] == ([0.25, 0.5] if override is None else [override] * 2)
    assert [call["strided_copy_swizzle_1x32_ratio"] for call in calls] == (
        [0.5, 1.0] if override is None else [override] * 2
    )


@pytest.mark.parametrize(
    "dtype,disable_dma_transpose,destination_shape,effective_f_dim,f_offset,expected",
    [
        (nl.bfloat16, False, (128, 1, 1, 512), None, 0, "pe"),
        (nl.float16, True, (128, 1, 1, 512), None, 0, "pe"),
        (nl.bfloat16, True, None, None, 0, "pe"),
        (nl.bfloat16, True, (128, 1, 1, 512), None, 0, "pe"),
        (nl.bfloat16, True, (128, 1, 512), None, 0, "pe"),
        (nl.bfloat16, True, (128, 512), None, 0, "pe"),
        (nl.bfloat16, True, (128, 1, 1, 512), 0, 0, "pe"),
        (nl.bfloat16, True, (128, 1, 1, 512), None, 128, "pe"),
        (nl.bfloat16, True, (128, 1, 1, 512), 72, 32, "pe"),
    ],
)
def test_default_loader_choice(
    monkeypatch, dtype, disable_dma_transpose, destination_shape, effective_f_dim, f_offset, expected
):
    calls = []
    monkeypatch.setattr(
        _load_apis,
        "load_tile_tensor_copy_1x32",
        lambda load_loc, data_store_loc=None, dma_engine=None: calls.append("tensor_copy"),
    )
    monkeypatch.setattr(
        _load_apis,
        "load_tile_PE_Swizzle_1x32",
        lambda load_loc, data_store_loc=None, dma_engine=None: calls.append("pe"),
    )
    load_loc = SimpleNamespace(
        tensor=SimpleNamespace(
            is_quantized=False,
            is_swizzled=False,
            is_f_by_k=False,
            quant_scheme=QuantScheme._1x32,
            data=SimpleNamespace(dtype=dtype),
            disable_dma_transpose=disable_dma_transpose,
            logical_shape=(512, 128),
            effective_f_dim=effective_f_dim,
        ),
        tile_k=512,
        tile_f=128,
        f_offset=f_offset,
    )
    store_loc = (
        SimpleNamespace(tensor=SimpleNamespace(data=SimpleNamespace(dtype=dtype, shape=destination_shape)))
        if destination_shape is not None
        else None
    )
    _load_apis.load_tile(cast(_TileLocation, load_loc), cast(_TileLocation | None, store_loc))
    assert calls == [expected]


class _Tensor:
    def __init__(self, shape=(128, 1, 128), dtype=nl.float32, buffer=nl.sbuf):
        self.shape, self.dtype, self.buffer = shape, dtype, buffer

    def ap(self, **kwargs):
        return self

    def __getitem__(self, key):
        return self


@pytest.mark.parametrize("ratio", [None, 0.0, 0.25, 0.5, 1.0])
def test_psum_frequency_follows_schedule(monkeypatch, ratio):
    copies, adds, accumulations = [], [], []
    monkeypatch.setattr(blocks.nl, "ndarray", _Tensor)
    monkeypatch.setattr(blocks.nisa, "tensor_copy", lambda dst, src: copies.append(dst.buffer))
    monkeypatch.setattr(blocks.nisa, "tensor_tensor", lambda *a, **k: adds.append(True))
    monkeypatch.setattr(blocks, "matmul_mx_x4_wrapper", lambda **k: accumulations.append(k["accumulate"]))
    td = cast(TensorDescriptor, SimpleNamespace(data=_Tensor(), scales=_Tensor(), scales_are_packed=False))
    schedule = _RatioSchedule(ratio) if ratio is not None else None
    for index in range(9):
        blocks._k_inner_loop_helper(
            cast(nl.NkiTensor, _Tensor()),
            td,
            td,
            0,
            0,
            128,
            128,
            1,
            index != 0,
            128,
            128,
            1,
            128,
            psum_copy_in_schedule=schedule,
        )
    expected = int(8 * ratio) if ratio is not None else 0
    assert copies.count(nl.psum) == expected
    assert sum(accumulations) == expected
    assert len(adds) == 8 - expected
    # A masked tile must not consume a scheduled operation.
    blocks._k_inner_loop_helper(
        cast(nl.NkiTensor, _Tensor()),
        td,
        td,
        1,
        0,
        128,
        128,
        1,
        True,
        128,
        128,
        1,
        128,
        psum_copy_in_schedule=schedule,
    )
    if schedule is not None:
        assert schedule.index == 8


@pytest.mark.parametrize("cache", [_best_config_cache._MOE_FWD_CONFIG_CACHE, _best_config_cache._MOE_BWD_CONFIG_CACHE])
def test_cached_loader_policy_preserves_measured_paths(cache):
    for key, (_, index) in cache.items():
        config = decode_config(index, _best_config_cache._CONFIG_SCHEMAS, _best_config_cache._CONFIG_RECORDS)
        names = (
            ("gate_up_config", "down_config")
            if hasattr(config, "gate_up_config")
            else ("phase1_config", "phase2_config", "phase3_config", "phase4_config")
        )
        expected = 1.0 if key.quant_scheme == QuantScheme._1x32 and not key.dma_transpose_enabled else 0.0
        for name in names:
            assert getattr(config, name).strided_copy_swizzle_1x32_ratio == expected
