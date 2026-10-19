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

"""Numerical coverage for general K-by-F loading and mixed matmul feature ratios."""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.matmul_mxfp8 import matmul_mxfp8_config
from nkilib_src.nkilib.experimental.matmul_mxfp8._matmul_mxfp8_constants import (
    PRECISION_BFLOAT16,
    PRECISION_FP32,
    PRECISION_MXFP8_X4,
)
from nkilib_src.nkilib.experimental.matmul_mxfp8._matmul_mxfp8_generic_api import generic_matmul_mxfp8_api
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_config import _validate_shapes, auto_generate_default
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_torch import matmul_mxfp8_torch_ref
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils import _load_apis
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils._common_utils import (
    create_and_set_active_sbm,
    get_active_sbm,
    with_active_sbm,
)
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils._load_apis import (
    load_tile_PE_Swizzle_1x32,
    load_tile_tensor_copy_1x32,
)
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.common_dataclasses import (
    LncShardingMode,
    QuantScheme,
    SwizzleMode,
    TensorDescriptor,
    TensorOrientation,
    _RatioSchedule,
    _TileLocation,
)

from test.integration.nkilib.experimental.matmul_mxfp8 import config_helper, constants
from test.integration.nkilib.experimental.matmul_mxfp8 import utils as matmul_utils
from test.integration.nkilib.experimental.matmul_mxfp8.test_matmul_mxfp8_generic_kernel import (
    _mxfp8_comparator,
    build_matmul_inputs,
)
from test.utils import common_dataclasses
from test.utils.unit_test_framework import UnitTestFramework

pytestmark = pytest.mark.fast


@nki.jit
@with_active_sbm
def _load_kernel(
    tensor_kf,
    tile_k,
    tile_f,
    k_offset,
    f_offset,
    store_rank,
    store_k_offset,
    store_f_offset,
    effective_f_dim,
    pe_loader=False,
):
    create_and_set_active_sbm()
    sbm = get_active_sbm()
    sbm.open_scope("GENERAL_1X32_LOAD")
    p = min(tile_k, 512) // 4
    slots = max(1, tile_k // 512)
    total_slots = slots + store_k_offset
    width = (tile_f + store_f_offset + 8) * 4
    shape = (
        (p, width) if store_rank == 2 else ((p, total_slots, width) if store_rank == 3 else (p, total_slots, 1, width))
    )
    data = sbm.alloc_stack(shape, dtype=tensor_kf.dtype, buffer=nl.sbuf)
    nisa.memset(dst=data, value=-7)
    td = TensorDescriptor(
        data=tensor_kf,
        orientation=TensorOrientation.K_BY_F,
        quant_scheme=QuantScheme._1x32,
        effective_f_dim=effective_f_dim,
        disable_dma_transpose=pe_loader,
    )
    store = _TileLocation(
        TensorDescriptor(data=data, is_swizzled=True),
        tile_k=p,
        tile_f=tile_f,
        k_offset=store_k_offset,
        f_offset=store_f_offset,
    )
    load_loc = _TileLocation(td, tile_k=tile_k, tile_f=tile_f, k_offset=k_offset, f_offset=f_offset)
    if pe_loader:
        load_tile_PE_Swizzle_1x32(load_loc, store)
    else:
        load_tile_tensor_copy_1x32(load_loc, store)
    out = nl.ndarray((p, total_slots, width), dtype=tensor_kf.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=data.reshape(out.shape))
    sbm.close_scope()
    return out


def _load_ref(
    tensor_kf,
    tile_k,
    tile_f,
    k_offset,
    f_offset,
    store_rank,
    store_k_offset,
    store_f_offset,
    effective_f_dim,
    pe_loader=False,
):
    p = min(tile_k, 512) // 4
    slots = max(1, tile_k // 512)
    out = np.full((p, slots + store_k_offset, (tile_f + store_f_offset + 8) * 4), -7, dtype=tensor_kf.dtype)
    for slot in range(slots):
        raw = np.zeros((p * 4, tile_f), dtype=tensor_kf.dtype)
        start = k_offset + slot * p * 4
        end = min(start + p * 4, tensor_kf.shape[0])
        f_end = min(f_offset + tile_f, effective_f_dim or tensor_kf.shape[1])
        if end > start and f_end > f_offset:
            raw[: end - start, : f_end - f_offset] = tensor_kf[start:end, f_offset:f_end]
        # The independent CPU golden defines consecutive-K quantization groups.
        swizzled = matmul_utils.swizzle_tensor_1x32(raw)
        valid_f = max(0, f_end - f_offset)
        out[:, store_k_offset + slot, store_f_offset * 4 : (store_f_offset + valid_f) * 4] = swizzled[:, : valid_f * 4]
    return {"out": out}


@pytest.mark.parametrize("dtype", [nl.bfloat16, nl.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize(
    "shape,tile,k_offset,f_offset,store_rank,store_k_offset,effective_f_dim",
    [
        ((512, 64), (512, 32), 0, 16, 2, 0, None),
        ((512, 64), (128, 32), 256, 16, 3, 1, None),
        ((512, 160), (384, 64), 128, 96, 4, 1, None),
        ((640, 72), (512, 96), 512, 32, 4, 1, None),
        ((1024, 48), (1024, 64), 0, 8, 3, 1, None),
        ((256, 64), (32, 8), 64, 16, 2, 0, None),
        ((512, 160), (512, 128), 0, 32, 4, 1, 72),
    ],
)
def test_general_loader(
    test_manager, platform_target, dtype, shape, tile, k_offset, f_offset, store_rank, store_k_offset, effective_f_dim
):
    if not platform_target.is_trn3():
        pytest.skip("MXFP8 requires TRN3.")
    kwargs = {
        'tensor_kf': np.random.RandomState(42).normal(size=shape).astype(dtype),
        'tile_k': tile[0],
        'tile_f': tile[1],
        'k_offset': k_offset,
        'f_offset': f_offset,
        'store_rank': store_rank,
        'store_k_offset': store_k_offset,
        'store_f_offset': 8,
        'effective_f_dim': effective_f_dim,
    }
    golden = _load_ref(**kwargs)
    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=_load_kernel,
        torch_ref=_load_ref,
        kernel_input_generator=lambda _: kwargs,
        output_tensor_descriptor=lambda _: golden,
    )
    framework.run_test(
        None,
        compiler_args=common_dataclasses.CompilerArgs(platform_target=platform_target, logical_nc_config=1),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("dtype", [nl.bfloat16, nl.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize(
    "tile_k,tile_f,store_rank,effective_f_dim",
    [(128, 32, 3, None), (512, 128, 4, 72)],
)
def test_pe_loader_independent(
    test_manager, platform_target, monkeypatch, dtype, tile_k, tile_f, store_rank, effective_f_dim
):
    if not platform_target.is_trn3():
        pytest.skip("MXFP8 requires TRN3.")

    def reject_tensor_copy(*args, **kwargs):
        raise AssertionError("The PE loader must not call the tensor-copy loader.")

    monkeypatch.setattr(_load_apis, "load_tile_tensor_copy_1x32", reject_tensor_copy)
    kwargs = {
        "tensor_kf": np.random.RandomState(42).normal(size=(512, 160)).astype(dtype),
        "tile_k": tile_k,
        "tile_f": tile_f,
        "k_offset": 0,
        "f_offset": 32,
        "store_rank": store_rank,
        "store_k_offset": 1,
        "store_f_offset": 8,
        "effective_f_dim": effective_f_dim,
        "pe_loader": True,
    }
    golden = _load_ref(**kwargs)
    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=_load_kernel,
        torch_ref=_load_ref,
        kernel_input_generator=lambda _: kwargs,
        output_tensor_descriptor=lambda _: golden,
    )
    framework.run_test(
        None,
        compiler_args=common_dataclasses.CompilerArgs(platform_target=platform_target, logical_nc_config=1),
        rtol=0,
        atol=0,
    )


@nki.jit
@with_active_sbm
def _ratio_matmul_kernel(
    lhs, rhs, lhs_scales, rhs_scales, config, lhs_orientation, rhs_orientation, disable_dma_transpose
):
    create_and_set_active_sbm()
    sbm = get_active_sbm()
    sbm.open_scope("FEATURE_RATIOS")
    lhs_td = TensorDescriptor(
        data=lhs,
        scales=lhs_scales,
        is_swizzled=lhs_scales is not None,
        orientation=lhs_orientation,
        quant_scheme=QuantScheme._1x32,
        swizzle_mode=SwizzleMode.PE,
        disable_dma_transpose=disable_dma_transpose,
    )
    rhs_td = TensorDescriptor(
        data=rhs,
        scales=rhs_scales,
        is_swizzled=rhs_scales is not None,
        orientation=rhs_orientation,
        quant_scheme=QuantScheme._1x32,
        swizzle_mode=SwizzleMode.PE,
        disable_dma_transpose=disable_dma_transpose,
    )
    auto_generate_default(
        config,
        PRECISION_MXFP8_X4 if lhs_td.is_quantized else PRECISION_BFLOAT16,
        PRECISION_MXFP8_X4 if rhs_td.is_quantized else PRECISION_BFLOAT16,
        PRECISION_FP32,
        lhs_is_swizzled=lhs_td.is_swizzled,
        rhs_is_swizzled=rhs_td.is_swizzled,
        lhs_load_with_PE_swizzle=lhs_td.uses_pe_swizzle,
        rhs_load_with_PE_swizzle=rhs_td.uses_pe_swizzle,
    )
    config.lnc_sharding.shard_operands(lhs_td, rhs_td)
    _validate_shapes(config, lhs_td, rhs_td)
    output = nl.ndarray((config.M, config.N), dtype=nl.float32, buffer=nl.shared_hbm)
    out_view = output
    rhs_offset = 0
    if config.lnc_sharding == LncShardingMode.SHARD_N:
        rhs_shape = rhs_td.sharded_logical_shape
        assert rhs_shape is not None
        n_per_core = rhs_shape[1]
        rhs_offset = nl.program_id(0) * n_per_core
        out_view = output[:, rhs_offset : rhs_offset + n_per_core]
    generic_matmul_mxfp8_api(
        lhs_td,
        rhs_td,
        config=config,
        output_td=TensorDescriptor(data=out_view),
        output_dtype=nl.float32,
        tile_loop_order=config.tile_loop_order.value,
        float8_dtype=config.float8_dtype,
        use_scale_packing=config.enable_scale_packing,
        rhs_n_offset=rhs_offset,
    )
    sbm.close_scope()
    return output


def _ratio_matmul_ref(
    lhs, rhs, lhs_scales, rhs_scales, config, lhs_orientation, rhs_orientation, disable_dma_transpose
):
    return matmul_mxfp8_torch_ref(
        lhs,
        rhs,
        lhs_scales=lhs_scales,
        rhs_scales=rhs_scales,
        lhs_is_swizzled=lhs_scales is not None,
        rhs_is_swizzled=rhs_scales is not None,
        lhs_orientation=lhs_orientation,
        rhs_orientation=rhs_orientation,
        float8_dtype=config.float8_dtype,
        use_scale_packing=config.enable_scale_packing,
        output_dtype=nl.float32,
        quant_scheme="1x32",
    )


@pytest.mark.parametrize("ratio", [0.0, 0.25, 0.5, 1.0], ids=["off", "quarter", "half", "all"])
@pytest.mark.parametrize(
    "case", ['small', 'partial', 'packed', 'prequantized', 'k_tail', 'k_tail_32', 'wide', 'lnc2', 'nmk']
)
def test_matmul_feature_ratios(test_manager, platform_target, ratio, case):
    if not platform_target.is_trn3():
        pytest.skip("MXFP8 requires TRN3.")
    M, K, N, tm, tn = (128, 1536, 256, 128, 128)
    lhs_dtype = rhs_dtype = constants.MatrixPrecision.BFLOAT16
    lhs_orientation = TensorOrientation.K_BY_F
    packing, lnc2 = False, False
    if case == "small":
        M, K, N, tm, tn = 64, 1024, 96, 32, 32
    elif case in ("partial", "packed"):
        M, K, N = 160, 1536, 288
        rhs_dtype = constants.MatrixPrecision.MXFP8_X4
        packing = case == "packed"
    elif case == "prequantized":
        lhs_dtype = rhs_dtype = constants.MatrixPrecision.MXFP8_X4
        packing = True
    elif case in ("k_tail", "k_tail_32"):
        K = 896 if case == "k_tail" else 672
        lhs_orientation = TensorOrientation.F_BY_K
    elif case == "wide":
        M, K, N, tm, tn = 128, 1024, 1280, 128, 1280
        rhs_dtype = constants.MatrixPrecision.MXFP8_X4
        packing = True
    elif case == "lnc2":
        N, lnc2 = 512, True
    elif case == "nmk":
        M, N = 288, 544
    conf = config_helper.TestConfig(
        M,
        K,
        N,
        tile_m=tm,
        tile_n=tn,
        tile_k=512,
        TILES_IN_BLOCK_M=2 if case == "nmk" else 1,
        TILES_IN_BLOCK_N=2 if case == "nmk" else 1,
        TILES_IN_BLOCK_K=1,
        TILES_IN_LOAD_M=1,
        TILES_IN_LOAD_N=1,
        run_with_lnc2=lnc2,
        lnc_2_shard_rhs=True,
        lhs_dtype=lhs_dtype,
        rhs_dtype=rhs_dtype,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        lhs_orientation=lhs_orientation,
        rhs_orientation=TensorOrientation.K_BY_F,
        quant_scheme="1x32",
        enable_scale_packing=packing,
        output_dtype=constants.MatrixPrecision.FP32,
        tile_loop_order="nmk" if case == "nmk" else "mnk",
    )
    config = conf.kernel_config
    config.psum_copy_in_ratio, config.strided_copy_swizzle_1x32_ratio = ratio, ratio
    inputs = build_matmul_inputs(conf)
    kwargs = {name: inputs[name] for name in ("lhs", "rhs", "lhs_scales", "rhs_scales")}
    for name in ("lhs_scales", "rhs_scales"):
        if kwargs[name] is not None and kwargs[name].dtype == np.uint8:
            kwargs[name] = kwargs[name].view(nl.float8_e8m0fnu)
    kwargs.update(
        config=config,
        lhs_orientation=lhs_orientation,
        rhs_orientation=TensorOrientation.K_BY_F,
        disable_dma_transpose=True,
    )
    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=_ratio_matmul_kernel,
        torch_ref=_ratio_matmul_ref,
        kernel_input_generator=lambda _: kwargs,
        output_tensor_descriptor=lambda _: {"out": np.zeros((M, N), dtype=np.float32)},
    )
    framework.run_test(
        None,
        compiler_args=common_dataclasses.CompilerArgs(
            platform_target=platform_target, logical_nc_config=2 if lnc2 else 1
        ),
        custom_comparator=_mxfp8_comparator(conf, nl.float32),
    )


def _fill_schedule(out, schedule, offset):
    for tile in range(4):
        nisa.memset(dst=out[:, offset + tile : offset + tile + 1], value=1 if schedule.select() else 0)


@nki.jit
def _schedule_kernel(ratio):
    schedule = _RatioSchedule(ratio)
    data = nl.ndarray((1, 12), dtype=nl.int32, buffer=nl.sbuf)
    for block in range(3):
        _fill_schedule(data, schedule, block * 4)
    out = nl.ndarray(data.shape, dtype=data.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=data)
    return out


def _schedule_ref(ratio):
    return {"out": np.diff(np.floor(np.arange(13) * ratio)).astype(np.int32).reshape(1, 12)}


@pytest.mark.parametrize("ratio", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_compiled_schedule(test_manager, platform_target, ratio):
    golden = _schedule_ref(ratio)
    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=_schedule_kernel,
        torch_ref=_schedule_ref,
        kernel_input_generator=lambda _: {"ratio": ratio},
        output_tensor_descriptor=lambda _: golden,
    )
    framework.run_test(
        None,
        compiler_args=common_dataclasses.CompilerArgs(platform_target=platform_target, logical_nc_config=1),
        rtol=0,
        atol=0,
    )


def test_cached_ratios_reach_auto_generated_configs(monkeypatch):
    """A K-by-F 1x32 cache hit carries both cached ratios; an explicit ratio wins."""
    cached = {
        "tile_m": 128,
        "tile_k": 512,
        "tile_n": 512,
        "TILES_IN_BLOCK_M": 10,
        "TILES_IN_BLOCK_N": 2,
        "TILES_IN_BLOCK_K": 4,
        "TILES_IN_LOAD_M": 10,
        "TILES_IN_LOAD_N": 2,
        "spill_reload": True,
        "psum_copy_in_ratio": 0.5,
        "strided_copy_swizzle_1x32_ratio": 1.0,
    }
    monkeypatch.setitem(matmul_mxfp8_config._AUTOTUNE_CACHE, "2560x8192x2560_pe_1x32_kbyf_pe_1x32_kbyf", cached)

    def weight_grad_conf():
        return config_helper.TestConfig(
            M=2560,
            K=8192,
            N=5120,
            lhs_dtype=constants.MatrixPrecision.BFLOAT16,
            rhs_dtype=constants.MatrixPrecision.BFLOAT16,
            lhs_is_swizzled=False,
            rhs_is_swizzled=False,
            lhs_orientation=TensorOrientation.K_BY_F,
            rhs_orientation=TensorOrientation.K_BY_F,
            quant_scheme="1x32",
        )

    (hit,) = weight_grad_conf().autoGenerateDefault()
    assert (hit.TILES_IN_BLOCK_M, hit.TILES_IN_LOAD_M) == (10, 10)
    assert (hit.psum_copy_in_ratio, hit.strided_copy_swizzle_1x32_ratio) == (0.5, 1.0)

    explicit = weight_grad_conf()
    explicit.strided_copy_swizzle_1x32_ratio = 0.0
    (pinned,) = explicit.autoGenerateDefault()
    assert (pinned.psum_copy_in_ratio, pinned.strided_copy_swizzle_1x32_ratio) == (0.5, 0.0)
