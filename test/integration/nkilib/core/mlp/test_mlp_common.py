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
import functools
import math
from collections import namedtuple
from typing import Callable

import neuron_dtypes as dt
import nki.isa as nisa
import nki.language as nl
import numpy as np
import torch
from nkilib_src.nkilib.core.mlp.mlp import mlp
from nkilib_src.nkilib.core.mlp.mlp_parameters import TKG_BS_SEQLEN_THRESHOLD
from nkilib_src.nkilib.core.mlp.mlp_tkg.projection_mx_constants import (
    _pmax,
    _psum_fmax,
    _q_width,
)
from nkilib_src.nkilib.core.mlp.mlp_torch import mlp_torch_ref
from nkilib_src.nkilib.core.utils.allocator import SbufManager
from nkilib_src.nkilib.core.utils.common_types import (
    ActFnType,
    ComputationMode,
    DtypeMode,
    MLPGateUpWeightLayout,
    NormType,
    QuantizationType,
)
from nkilib_src.nkilib.core.utils.kernel_assert import kernel_assert
from nkilib_src.nkilib.core.utils.kernel_helpers import (
    get_max_positive_value_for_dtype,
    get_verified_program_sharding_info,
)
from nkilib_src.nkilib.core.utils.logging import Logger

from test.integration.nkilib.utils.tensor_generators import (
    TensorTemplate,
    gaussian_tensor_generator,
    generate_stabilized_mx_data,
    update_func_str,
)
from test.integration.nkilib.utils.test_kernel_common import resolve_dtype_mode_for_torch_ref
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def dedup_test_vectors(vectors, ignore_indices):
    """Deduplicate test vectors, ignoring fields at the given indices.

    Certain fields (e.g., tpbSgCyclesSum, vnc_degree) are excluded from the
    dedup key. Keeps the first occurrence of each unique vector. Vectors may
    be raw lists/tuples or pytest.param() objects; pytest.param marks are
    preserved on the kept entry.

    Args:
        vectors: List of test parameter tuples/lists or pytest.param objects.
        ignore_indices: Set/tuple of field indices to exclude from the dedup key.
    """
    seen = set()
    result = []
    for v in vectors:
        values = v.values if hasattr(v, "values") and hasattr(v, "marks") else v
        key = tuple(val for i, val in enumerate(values) if i not in ignore_indices)
        if key not in seen:
            seen.add(key)
            result.append(v)
    return result


def mlp_output_tensor_descriptor(kernel_input):
    """Return output tensor shapes for the MLP kernel.

    Uses down_proj_weights_tensor to derive the true hidden dim, which
    correctly handles the ROW-quant H+4 padding case.
    """
    hidden_shape = kernel_input["hidden_tensor"].shape
    hidden = kernel_input["down_proj_weights_tensor"].shape[1]
    dtype = nl.bfloat16

    transposed_out = kernel_input.get("transposed_out", False)
    if transposed_out:
        # Transposed output: [H0, n_prgs, H1_shard, BxS]
        transposed_in = kernel_input.get("transposed_in", False)
        if transposed_in:
            H0 = hidden_shape[0]
            n_prgs = hidden_shape[1]
            H1_shard = hidden_shape[2]
            BxS = hidden_shape[3]
        else:
            batch, seqlen, _ = hidden_shape
            BxS = batch * seqlen
            H0 = 128
            n_prgs = 2  # LNC=2
            H1_shard = hidden // (H0 * n_prgs)
        output = {"out": np.zeros((H0, n_prgs, H1_shard, BxS), dtype=dtype)}
    else:
        transposed_in = kernel_input.get("transposed_in", False)
        if transposed_in:
            # Input is 4D [H0, n_prgs, H1_shard, BxS], output is [1, BxS, H]
            BxS = hidden_shape[3]
            output = {"out": np.zeros((1, BxS, hidden), dtype=dtype)}
        else:
            batch, seqlen, _ = hidden_shape
            output = {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}

    if kernel_input.get("fused_add_tensor") is not None and kernel_input.get("store_fused_add_result", False):
        batch_fa, seqlen_fa, _ = (
            hidden_shape if not kernel_input.get("transposed_in", False) else (1, hidden_shape[3], hidden)
        )
        output["add_out"] = np.zeros((batch_fa, seqlen_fa, hidden), dtype=dtype)
    return output


def _run_mlp_test(
    test_manager,
    kernel_input,
    compiler_args,
    output_tensor_descriptor,
    rtol=2e-2,
    atol=1e-5,
    is_negative_test=False,
    inference_args=None,
    kernel_entry=None,
):
    """Shared helper to run an MLP test (CTE or TKG) via UnitTestFramework.

    Args:
        test_manager: Orchestrator instance.
        kernel_input: Pre-built kernel input dict (numpy arrays).
        compiler_args: CompilerArgs for the test.
        output_tensor_descriptor: Callable that returns output tensor shapes.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.
        is_negative_test: Whether this is expected to fail.
        inference_args: Optional InferenceArgs (e.g. TKG_INFERENCE_ARGS for determinism checking).
        kernel_entry: Optional kernel callable. Defaults to mlp.
    """
    lnc = compiler_args.logical_nc_config
    kernel_fn = kernel_entry if kernel_entry is not None else mlp
    transposed_out = kernel_input.get("transposed_out", False)

    # Pre-resolve DtypeMode.AUTO for the torch ref using the platform target.
    # The kernel still receives the original dtype_mode and resolves at trace
    # time; the torch ref runs on CPU and can't query hardware directly.
    platform_target = compiler_args.platform_target
    kernel_dtype_mode = kernel_input.get("dtype_mode", DtypeMode.NON_OCP)
    torch_ref_dtype_mode = resolve_dtype_mode_for_torch_ref(kernel_dtype_mode, platform_target)

    @functools.wraps(mlp)
    def mlp_torch_wrapper(**kwargs):
        # Override with the pre-resolved mode so the ref clips to the same FP8
        # range the kernel allocates.
        kwargs["dtype_mode"] = torch_ref_dtype_mode
        result = mlp_torch_ref[lnc](**kwargs)
        if transposed_out:
            # Convert torch ref [B, S, H] output to transposed [H0, n_prgs*H1_shard*BxS]
            if isinstance(result, dict):
                out = result["out"]
            elif isinstance(result, torch.Tensor):
                out = result
            else:
                out = result
            if isinstance(out, torch.Tensor):
                out_np = out.float().numpy()
            else:
                out_np = out
            hidden = kernel_input["down_proj_weights_tensor"].shape[1]
            H0 = 128
            n_prgs = lnc if lnc > 1 else 2
            H1_shard = hidden // (H0 * n_prgs)
            BxS = out_np.reshape(-1, hidden).shape[0]
            # [BxS, H] -> [BxS, n_prgs, H0, H1_shard] -> [H0, n_prgs, H1_shard, BxS]
            transposed = out_np.reshape(BxS, n_prgs, H0, H1_shard).transpose(2, 1, 3, 0)
            if isinstance(result, dict):
                result["out"] = transposed
            else:
                result = transposed
        return result

    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=kernel_fn,
        torch_ref=torch_ref_wrapper(mlp_torch_wrapper),
        kernel_input_generator=lambda _: kernel_input,
        output_tensor_descriptor=output_tensor_descriptor,
        check_unused_params=True,
    )
    framework.run_test(
        test_config=None,
        compiler_args=compiler_args,
        rtol=rtol,
        atol=atol,
        is_negative_test=is_negative_test,
        inference_args=inference_args,
    )


def build_fused_norm_mlp(
    batch,
    seqlen,
    hidden,
    intermediate,
    dtype,
    quant_dtype=None,
    quantization_type=QuantizationType.NONE,
    is_input_quantized=False,
    eps=1e-6,
    fused_add=False,
    norm_type=NormType.RMS_NORM,
    store_add=False,
    lnc_degree=1,
    tiling_degree=None,
    skip_gate=False,
    act_fn_type=ActFnType.SiLU,
    gate_bias=False,
    up_bias=False,
    down_bias=False,
    norm_bias=False,
    use_tkg_gate_up_proj_column_tiling=True,
    use_tkg_down_proj_column_tiling=True,
    use_tkg_down_proj_optimized_layout=False,
    use_contiguous_x4_gate_up=False,
    gate_clamp_lower_limit=None,
    gate_clamp_upper_limit=None,
    up_clamp_lower_limit=None,
    up_clamp_upper_limit=None,
    transposed_in=False,
    transposed_out=False,
    gate_up_w_layout=MLPGateUpWeightLayout.CONTIGUOUS,
    tensor_generator: Callable = gaussian_tensor_generator(),
    mode: ComputationMode = ComputationMode.AUTO,
    use_mx_block_scale_input: bool = False,
):
    np.random.seed(42)
    rng = np.random.default_rng(42)

    # For MX flows, default gate_up_w_layout to H_X4_MIDDLE if not explicitly set
    if gate_up_w_layout == MLPGateUpWeightLayout.CONTIGUOUS and quantization_type in (
        QuantizationType.STATIC_MX,
        QuantizationType.ROW_MX,
    ):
        gate_up_w_layout = MLPGateUpWeightLayout.H_X4_MIDDLE

    is_tkg_mode = mode == ComputationMode.DECODE or (
        mode == ComputationMode.AUTO and (batch * seqlen) <= TKG_BS_SEQLEN_THRESHOLD
    )

    if isinstance(norm_type, bool):
        norm_type = NormType.RMS_NORM if norm_type else NormType.NO_NORM

    fused_add_tensor = (
        tensor_generator(shape=(batch, seqlen, hidden), dtype=dtype, name="fused_add_tensor") if fused_add else None
    )

    if quantization_type == QuantizationType.MX and is_tkg_mode:
        tokens = batch * seqlen
        n_H512_tile = hidden // _psum_fmax
        n_I512_tile = math.ceil(intermediate / (_pmax * _q_width))

        # Generate hidden_input as stabilized MX data
        hidden_states, _, _ = generate_stabilized_mx_data(
            mx_dtype=nl.float8_e4m3fn_x4, shape=(tokens * n_H512_tile * _pmax, _q_width), val_range=1.0
        )
        hidden_states = (
            hidden_states.reshape(tokens, n_H512_tile, _pmax, _q_width).transpose(0, 3, 1, 2).reshape(tokens, hidden)
        )
        hidden_input = dt.static_cast(hidden_states, dtype).reshape(batch, seqlen, hidden)
    elif is_input_quantized and use_mx_block_scale_input:
        from test.integration.nkilib.core.moe.moe_cte.test_utils import build_prequantized_hidden_concat

        tokens = batch * seqlen
        n_H512_tile = hidden // (_pmax * _q_width)
        hidden_concat, _mx_block_hidden_fp32 = build_prequantized_hidden_concat(tokens, n_H512_tile, hidden)
        hidden_input = dt.static_cast(hidden_concat.reshape(batch, seqlen, -1), quant_dtype)
    elif is_input_quantized:
        if quantization_type.is_logical_row() or quantization_type == QuantizationType.MX:
            hidden_input = tensor_generator(
                shape=(batch, seqlen, hidden + 4), dtype=quant_dtype, name='hidden'
            )  # +4 to allow space for an fp32 dequant value
        elif quantization_type.is_logical_static():
            hidden_input = tensor_generator(shape=(batch, seqlen, hidden), dtype=quant_dtype, name='hidden')
    else:
        hidden_input = tensor_generator(shape=(batch, seqlen, hidden), dtype=dtype, name="hidden")

    gamma = None
    norm_b = None
    if norm_type != NormType.NO_NORM and norm_type != NormType.RMS_NORM_SKIP_GAMMA:
        gamma = dt.static_cast(rng.uniform(low=-0.1, high=0.1, size=(1, hidden)), dtype)
        if quantization_type == QuantizationType.MX:
            gamma = (
                gamma.reshape(1, hidden // (_pmax * _q_width), _pmax, _q_width).transpose(0, 3, 1, 2).reshape(1, hidden)
            )
        if norm_bias:
            norm_b = tensor_generator(shape=(1, hidden), dtype=dtype, name="norm_b")

    # Pre-generate MX weights if using MX quantization
    if quantization_type == QuantizationType.MX:
        mx_quant_dtype = nl.float8_e4m3fn_x4 if quant_dtype in [nl.float8_e4m3, nl.float8_e4m3fn] else quant_dtype
        mx_weights = gen_mlp_mxfp_weights(hidden, intermediate, mx_quant_dtype)

    weight_dtype = quant_dtype if quant_dtype is not None else dtype

    # Use pre-generated MX weights or generate regular weights
    if quantization_type == QuantizationType.MX:
        if is_tkg_mode:
            gate_w = mx_weights.gate_w_qtz
            up_w = mx_weights.up_w_qtz
            down_w = mx_weights.down_w_qtz
        else:
            H0, H1 = mx_weights.gate_w_qtz.shape[0], mx_weights.gate_w_qtz.shape[1]
            I0, I1 = mx_weights.down_w_qtz.shape[0], mx_weights.down_w_qtz.shape[1]
            gate_w = mx_weights.gate_w_qtz.view(quant_dtype).reshape(H0, H1, I1, 4, I0, 4)
            up_w = mx_weights.up_w_qtz.view(quant_dtype).reshape(H0, H1, I1, 4, I0, 4)
            down_w = mx_weights.down_w_qtz.view(quant_dtype).reshape(I0, I1, hidden, 4)
    elif quantization_type == QuantizationType.STATIC_MX:
        # STATIC_MX weight format depends on kernel mode:
        # - TKG (BxS <= 96 or DECODE mode): x4-packed 3D weights [128, H//512, I] for nc_matmul_mx
        # - CTE (BxS > 96): 2D scalar fp8 weights [H, I] (CTE does its own internal packing)
        if is_tkg_mode:
            # TKG: generate scalar fp8 weights with per-tensor scale, then pack to x4 layout.
            gate_w_scalar, gate_w_scale_extracted = generate_and_quantize_to_fp8(
                shape=(hidden, intermediate),
                dtype=dtype,
                quant_dtype=quant_dtype,
                rng=rng,
                quantization_type=quantization_type,
                scale_shape=(_pmax, 1),
                quantize_dim=0,
            )
            up_w_scalar, up_w_scale_extracted = generate_and_quantize_to_fp8(
                shape=(hidden, intermediate),
                dtype=dtype,
                quant_dtype=quant_dtype,
                rng=rng,
                quantization_type=quantization_type,
                scale_shape=(_pmax, 1),
                quantize_dim=0,
            )
            down_w_scalar, down_w_scale_extracted = generate_and_quantize_to_fp8(
                shape=(intermediate, hidden),
                dtype=dtype,
                quant_dtype=quant_dtype,
                rng=rng,
                quantization_type=quantization_type,
                scale_shape=(_pmax, 1),
                quantize_dim=0,
            )
            gate_w = _fp8_to_gate_up_6d(gate_w_scalar, hidden, intermediate, gate_up_w_layout)
            up_w = _fp8_to_gate_up_6d(up_w_scalar, hidden, intermediate, gate_up_w_layout)
            down_w = _fp8_to_down_4d(down_w_scalar, intermediate, hidden)
        else:
            gate_w = tensor_generator(
                shape=(128, hidden // 512, intermediate // 512, 4, 128, 4), dtype=weight_dtype, name="gate_w"
            )
            up_w = tensor_generator(
                shape=(128, hidden // 512, intermediate // 512, 4, 128, 4), dtype=weight_dtype, name="up_w"
            )
            down_w = tensor_generator(shape=(128, intermediate // 512, hidden, 4), dtype=weight_dtype, name="down_w")
    elif quantization_type == QuantizationType.ROW_MX:
        # ROW_MX: weights use per-row scaling.
        # gate_up_weight [H, I] → scale [1, I] broadcast to [128, I]
        # down_weight [I, H] → scale [1, H] broadcast to [128, H]
        if is_tkg_mode:
            gate_w_scalar, gate_w_scale_extracted = generate_and_quantize_to_fp8(
                shape=(hidden, intermediate),
                dtype=dtype,
                quant_dtype=quant_dtype,
                rng=rng,
                quantization_type=QuantizationType.ROW,
                scale_shape=(_pmax, intermediate),
                quantize_dim=0,
            )
            up_w_scalar, up_w_scale_extracted = generate_and_quantize_to_fp8(
                shape=(hidden, intermediate),
                dtype=dtype,
                quant_dtype=quant_dtype,
                rng=rng,
                quantization_type=QuantizationType.ROW,
                scale_shape=(_pmax, intermediate),
                quantize_dim=0,
            )
            down_w_scalar, down_w_scale_extracted = generate_and_quantize_to_fp8(
                shape=(intermediate, hidden),
                dtype=dtype,
                quant_dtype=quant_dtype,
                rng=rng,
                quantization_type=QuantizationType.ROW,
                scale_shape=(_pmax, hidden),
                quantize_dim=0,
            )
            gate_w = _fp8_to_gate_up_6d(gate_w_scalar, hidden, intermediate, gate_up_w_layout)
            up_w = _fp8_to_gate_up_6d(up_w_scalar, hidden, intermediate, gate_up_w_layout)
            down_w = _fp8_to_down_4d(down_w_scalar, intermediate, hidden)
        else:
            gate_w = tensor_generator(
                shape=(128, hidden // 512, intermediate // 512, 4, 128, 4), dtype=weight_dtype, name="gate_w"
            )
            up_w = tensor_generator(
                shape=(128, hidden // 512, intermediate // 512, 4, 128, 4), dtype=weight_dtype, name="up_w"
            )
            down_w = tensor_generator(shape=(128, intermediate // 512, hidden, 4), dtype=weight_dtype, name="down_w")
    else:
        gate_w = tensor_generator(shape=(hidden, intermediate), dtype=weight_dtype, name="gate_w")
        up_w = tensor_generator(shape=(hidden, intermediate), dtype=weight_dtype, name="up_w")
        down_w = tensor_generator(shape=(intermediate, hidden), dtype=weight_dtype, name="down_w")

    # Generate Bias
    # Ensure n_I512_tile is defined for STATIC_MX TKG (already defined for MX above)
    if quantization_type == QuantizationType.STATIC_MX:
        n_I512_tile = math.ceil(intermediate / (_pmax * _q_width))
    if quantization_type == QuantizationType.ROW_MX:
        n_I512_tile = math.ceil(intermediate / (_pmax * _q_width))
    if (
        quantization_type == QuantizationType.MX
        or (quantization_type == QuantizationType.STATIC_MX and is_tkg_mode)
        or quantization_type == QuantizationType.ROW_MX
    ):
        i_p = intermediate // 4 if intermediate <= 512 else _pmax
        gate_b = tensor_generator(shape=(i_p, n_I512_tile, _q_width), dtype=dtype, name="gate_b") if gate_bias else None
        up_b = tensor_generator(shape=(i_p, n_I512_tile, _q_width), dtype=dtype, name="up_b") if up_bias else None
        down_b = tensor_generator(shape=(1, hidden), dtype=dtype, name="down_b") if down_bias else None
    else:
        gate_b = tensor_generator(shape=(1, intermediate), dtype=dtype, name="gate_b") if gate_bias else None
        up_b = tensor_generator(shape=(1, intermediate), dtype=dtype, name="up_b") if up_bias else None
        down_b = tensor_generator(shape=(1, hidden), dtype=dtype, name="down_b") if down_bias else None

    if quantization_type == QuantizationType.MX:
        # MX quantization uses uint8 scales
        if is_tkg_mode:
            gate_w_scale = mx_weights.gate_w_scale
            up_w_scale = mx_weights.up_w_scale
        else:
            # CTE expects gate/up scales pre-transposed to match the 6D weight layout's
            # physical I order: [16, H/512, I/512, 4, 128] (within each 512-tile, logical
            # order (128, 4) is transposed to physical order (4, 128)).
            n_I512_tile = math.ceil(intermediate / (_pmax * _q_width))
            gate_w_scale = mx_weights.gate_w_scale.reshape(16, hidden // 512, n_I512_tile, _pmax, _q_width).transpose(
                0, 1, 2, 4, 3
            )
            up_w_scale = mx_weights.up_w_scale.reshape(16, hidden // 512, n_I512_tile, _pmax, _q_width).transpose(
                0, 1, 2, 4, 3
            )
        down_w_scale = mx_weights.down_w_scale
        gate_up_in_scale = None
        down_in_scale = None
    elif quantization_type == QuantizationType.STATIC_MX:
        if is_tkg_mode:
            # TKG STATIC_MX: use extracted per-tensor fp32 dequant scales from weight quantization
            gate_w_scale = gate_w_scale_extracted
            up_w_scale = up_w_scale_extracted
            down_w_scale = down_w_scale_extracted
            # gate_up_in_scale must be a uniform per-tensor scalar broadcast to [_pmax, 1]
            gate_up_in_scale_scalar = np.float32(np.abs(rng.standard_normal()) * 10.0 + 1e-5)
            gate_up_in_scale = (
                np.broadcast_to(gate_up_in_scale_scalar.reshape(1, 1), (_pmax, 1)).copy().astype(np.float32)
            )
            # down_in_scale must be a uniform per-tensor scalar broadcast to [_pmax, 1]
            down_in_scale_scalar = np.float32(np.abs(rng.standard_normal()) * 10.0 + 1e-5)
            down_in_scale = np.broadcast_to(down_in_scale_scalar.reshape(1, 1), (_pmax, 1)).copy().astype(np.float32)
        else:
            # CTE STATIC_MX: same scale generation as mainline STATIC path
            gate_w_scale = tensor_generator(shape=(_pmax, 1), dtype=np.float32, name="gate_w_scale")
            up_w_scale = tensor_generator(shape=(_pmax, 1), dtype=np.float32, name="up_w_scale")
            down_w_scale = tensor_generator(shape=(_pmax, 1), dtype=np.float32, name="down_w_scale")
            gate_up_in_scale = tensor_generator(shape=(_pmax, 1), dtype=np.float32, name="gate_up_in_scale")
            down_in_scale = tensor_generator(shape=(_pmax, 1), dtype=np.float32, name="down_in_scale")
    elif quantization_type == QuantizationType.ROW_MX:
        if is_tkg_mode:
            # ROW_MX: per-row weight scales, no input scales (computed dynamically).
            # Pre-shuffle to match MX output layout for efficient per-partition dequant in kernel.
            gate_w_scale = _shuffle_gate_up_w_scale_for_row_mx_h_x4_middle(gate_w_scale_extracted, intermediate)
            up_w_scale = _shuffle_gate_up_w_scale_for_row_mx_h_x4_middle(up_w_scale_extracted, intermediate)
            down_w_scale = _shuffle_down_w_scale_for_row_mx(down_w_scale_extracted, hidden)
            gate_up_in_scale = None
            down_in_scale = None
        else:
            gate_w_scale = tensor_generator(
                shape=(_pmax, math.ceil(intermediate / 512), 4), dtype=np.float32, name="gate_w_scale"
            )
            up_w_scale = tensor_generator(
                shape=(_pmax, math.ceil(intermediate / 512), 4), dtype=np.float32, name="up_w_scale"
            )
            down_w_scale = np.broadcast_to(
                tensor_generator(shape=(1, hidden), dtype=np.float32, name="down_w_scale"), (_pmax, hidden)
            )
            gate_up_in_scale = None
            down_in_scale = None
    elif quantization_type in [QuantizationType.STATIC]:
        gate_w_scale = np.broadcast_to(
            tensor_generator(shape=(1, 1), dtype=np.float32, name="gate_w_scale"), (_pmax, 1)
        )
        up_w_scale = np.broadcast_to(tensor_generator(shape=(1, 1), dtype=np.float32, name="up_w_scale"), (_pmax, 1))
        down_w_scale = np.broadcast_to(
            tensor_generator(shape=(1, 1), dtype=np.float32, name="down_w_scale"), (_pmax, 1)
        )
        gate_up_in_scale = np.broadcast_to(
            tensor_generator(shape=(1, 1), dtype=np.float32, name="gate_up_in_scale"), (_pmax, 1)
        )
        down_in_scale = np.broadcast_to(
            tensor_generator(shape=(1, 1), dtype=np.float32, name="down_in_scale"), (_pmax, 1)
        )
    elif quantization_type == QuantizationType.ROW or quant_dtype is not None:
        gate_w_scale = np.broadcast_to(
            tensor_generator(shape=(intermediate,), dtype=np.float32, name="gate_w_scale"), (_pmax, intermediate)
        )
        up_w_scale = np.broadcast_to(
            tensor_generator(shape=(intermediate,), dtype=np.float32, name="up_w_scale"), (_pmax, intermediate)
        )
        down_w_scale = np.broadcast_to(
            tensor_generator(shape=(hidden,), dtype=np.float32, name="down_w_scale"), (_pmax, hidden)
        )
        gate_up_in_scale = None
        down_in_scale = None
    else:
        gate_w_scale = None
        up_w_scale = None
        down_w_scale = None
        gate_up_in_scale = None
        down_in_scale = None

    kernel_input = {
        "hidden_tensor": hidden_input,
        "gate_proj_weights_tensor": gate_w,
        "up_proj_weights_tensor": up_w,
        "down_proj_weights_tensor": down_w,
        "normalization_weights_tensor": gamma,
        "gate_proj_bias_tensor": gate_b,
        "up_proj_bias_tensor": up_b,
        "down_proj_bias_tensor": down_b,
        "normalization_bias_tensor": norm_b,
        "fused_add_tensor": fused_add_tensor,
        "store_fused_add_result": store_add,
        "activation_fn": act_fn_type,
        "normalization_type": norm_type,
        "quantization_type": quantization_type,
        "gate_w_scale": gate_w_scale,
        "up_w_scale": up_w_scale,
        "down_w_scale": down_w_scale,
        "gate_up_in_scale": gate_up_in_scale,
        "down_in_scale": down_in_scale,
        "output_dtype": nl.bfloat16,
        "store_output_in_sbuf": False,
        "eps": eps,
        "skip_gate_proj": skip_gate,
        "use_tkg_gate_up_proj_column_tiling": use_tkg_gate_up_proj_column_tiling,
        "use_tkg_down_proj_column_tiling": use_tkg_down_proj_column_tiling,
        "use_tkg_down_proj_optimized_layout": use_tkg_down_proj_optimized_layout,
        "use_contiguous_x4_gate_up": use_contiguous_x4_gate_up,
        "gate_clamp_upper_limit": gate_clamp_upper_limit,
        "gate_clamp_lower_limit": gate_clamp_lower_limit,
        "up_clamp_upper_limit": up_clamp_upper_limit,
        "up_clamp_lower_limit": up_clamp_lower_limit,
        "mode": mode,
        "gate_up_w_layout": gate_up_w_layout,
    }
    if transposed_in:
        kernel_input["transposed_in"] = transposed_in
    if transposed_out:
        kernel_input["transposed_out"] = transposed_out

    # Convert hidden_input to transposed layout [H0, n_prgs, H1_shard, BxS] when transposed_in=True
    if transposed_in:
        n_prgs = lnc_degree if lnc_degree and lnc_degree > 1 else 2
        H0 = 128
        H1_shard = hidden // (H0 * n_prgs)
        BxS = batch * seqlen
        # [B, S, H] -> [BxS, H] -> [BxS, n_prgs, H0, H1_shard] -> [H0, n_prgs, H1_shard, BxS]
        flat = dt.static_cast(hidden_input, dtype).reshape(BxS, hidden)
        transposed = flat.reshape(BxS, n_prgs, H0, H1_shard).transpose(2, 1, 3, 0)
        kernel_input["hidden_tensor"] = dt.static_cast(transposed, dtype)

    return kernel_input


float8_e5m2_x4 = nl.float8_e5m2_x4
float8_e4m3fn_x4 = nl.float8_e4m3fn_x4
float4_e2m1fn_x4 = nl.float4_e2m1fn_x4


# max normal
# float8_e5m2: S 11110 11 = ± 2^15 × 1.75 = ± 57,344
# float8_e4m3fn: S 1111 110 = ± 2^8 × 1.75 = ± 448
# float4_e2m1fn: S 11 1 = ± 2^2 × 1.5 = ± 6
def get_mx_fp_max(dst_dtype):
    max_values = {float8_e5m2_x4: 57344, float8_e4m3fn_x4: 448, float4_e2m1fn_x4: 6}
    assert dst_dtype in max_values, f"no max value provided for {dst_dtype}"
    return max_values.get(dst_dtype)


def get_mx_max_exp(dst_dtype):
    max_exp_values = {float8_e5m2_x4: 15, float8_e4m3fn_x4: 8, float4_e2m1fn_x4: 2}
    assert dst_dtype in max_exp_values, f"no max exp value provided for {dst_dtype}"
    return max_exp_values.get(dst_dtype)


# Get exponent for float32 in IEEE 754 standard
def get_float32_exp(float_data):
    man_nbits, exp_nbits = 23, 8
    return (float_data.astype(np.float32).view(np.uint32) >> man_nbits) & ((1 << exp_nbits) - 1)


MlpMxWeights = namedtuple(
    'MlpMxWeights',
    [
        'gate_w_qtz',
        'gate_w_scale',
        'up_w_qtz',
        'up_w_scale',
        'down_w_qtz',
        'down_w_scale',
    ],
)


def gen_mlp_mxfp_weights(hidden, intermediate, mx_dtype):
    """Generate MX quantized weights and scales for MLP gate, up, and down projections.

    Args:
        hidden: Hidden dimension size (must be divisible by 512)
        intermediate: Intermediate dimension size
        mx_dtype: MX quantization dtype (float4_e2m1fn_x4 or float8_e4m3fn_x2)

    Returns:
        MlpMxWeights: Named tuple containing quantized weights and scales
    """

    def split_last_dim(X, extra_last_dim):
        return X.reshape(*X.shape[:-1], -1, extra_last_dim)

    n_H512_tile = hidden // 512
    n_I512_tile = math.ceil(intermediate / 512)
    p_I = (intermediate // 4) if intermediate < 512 else 128  # do not pad I's pdim if I<512

    # Initialize tensors for gate projection
    gate_w_qtz = np.zeros((128, n_H512_tile, intermediate), dtype=mx_dtype)
    gate_w_scale = np.zeros((16, n_H512_tile, intermediate), dtype=np.uint8)

    # Initialize tensors for up projection
    up_w_qtz = np.zeros((128, n_H512_tile, intermediate), dtype=mx_dtype)
    up_w_scale = np.zeros((16, n_H512_tile, intermediate), dtype=np.uint8)

    # Initialize tensors for down projection
    down_w_qtz = np.zeros((p_I, n_I512_tile, hidden), dtype=mx_dtype)
    down_w_scale = np.zeros((p_I // 8, n_I512_tile, hidden), dtype=np.uint8)

    # Generate gate projection weights
    _, tmp_w_qtz, tmp_w_scale = generate_stabilized_mx_data(mx_dtype, (128, hidden // 128 * intermediate))
    gate_w_qtz[:, :, :] = split_last_dim(tmp_w_qtz, intermediate)  # [128, n_H512_tile, intermediate]
    gate_w_scale[:, :, :] = split_last_dim(tmp_w_scale, intermediate)  # [16, n_H512_tile, intermediate]

    # Generate up projection weights
    _, tmp_w_qtz, tmp_w_scale = generate_stabilized_mx_data(mx_dtype, (128, hidden // 128 * intermediate))
    up_w_qtz[:, :, :] = split_last_dim(tmp_w_qtz, intermediate)  # [128, n_H512_tile, intermediate]
    up_w_scale[:, :, :] = split_last_dim(tmp_w_scale, intermediate)  # [16, n_H512_tile, intermediate]

    # Generate down projection weights
    _, tmp_w_qtz, tmp_w_scale = generate_stabilized_mx_data(mx_dtype, (intermediate // 4, hidden * 4))
    for i_I512_tile in range(n_I512_tile):
        n_rows_qtz = min(128, intermediate // 4 - i_I512_tile * 128)  # every 512 tile has at most 512/4=128 x4 values
        n_rows_scale = min(16, intermediate // 32 - i_I512_tile * 16)  # every 512 tile has at most 512/32=16 scales
        down_w_qtz[:n_rows_qtz, i_I512_tile, :] = tmp_w_qtz[i_I512_tile * 128 : i_I512_tile * 128 + n_rows_qtz, :]
        down_w_scale[:n_rows_scale, i_I512_tile, :] = tmp_w_scale[i_I512_tile * 16 : i_I512_tile * 16 + n_rows_scale, :]

    return MlpMxWeights(gate_w_qtz, gate_w_scale, up_w_qtz, up_w_scale, down_w_qtz, down_w_scale)


MxAllTokensWeights = namedtuple(
    'MxAllTokensWeights',
    [
        'gate_up_w_qtz',
        'gate_up_w_scale',
        'down_w_qtz',
        'down_w_scale',
    ],
)


def gen_moe_mx_weights(hidden, intermediate, expert, mx_dtype=float4_e2m1fn_x4):
    """Generate MX weights and scales for all tokens gate/up (fused into one tensor) and down projection.

    Args:
        hidden: Hidden dimension size
        intermediate: Intermediate dimension size
        expert: Number of experts
        mx_dtype: MX quantization dtype (float4_e2m1fn_x4 or float8_e4m3fn_x4)
    """
    # x4 types pack 4 elements, so static_cast divides F-dim by 4 for both MXFP4 and MXFP8
    x4_pack_size = 4

    def split_last_dim(X, extra_last_dim):
        return X.reshape(*X.shape[:-1], -1, extra_last_dim)

    n_H512_tile = hidden // 512
    n_I512_tile = math.ceil(intermediate / 512)
    p_I = math.ceil(intermediate / x4_pack_size / 8) * 8 if intermediate < 512 else 128
    gate_up_w_qtz = np.zeros((expert, 128, 2, n_H512_tile, intermediate), dtype=mx_dtype)
    gate_up_w_scale = np.zeros((expert, 16, 2, n_H512_tile, intermediate), dtype=np.uint8)
    down_w_qtz = np.zeros((expert, p_I, n_I512_tile, hidden), dtype=mx_dtype)
    down_w_scale = np.zeros((expert, math.ceil(p_I / 8), n_I512_tile, hidden), dtype=np.uint8)

    for e in range(expert):
        # Gen weight for gate and up proj
        for i in range(2):
            _, tmp_w_qtz, tmp_w_scale = generate_stabilized_mx_data(mx_dtype, (128, hidden // 128 * intermediate))

            # Copy to full weights and scales
            gate_up_w_qtz[e, :, i, :, :] = split_last_dim(tmp_w_qtz, intermediate)  # [128, n_H512_tile, intermediate]
            gate_up_w_scale[e, :, i, :, :] = split_last_dim(
                tmp_w_scale, intermediate
            )  # [128, n_H512_tile, intermediate]

        # Gen weight for down proj - generate logical shape, static_cast handles x4 packing
        # Pad to multiple of 8 if needed
        I_p_actual = math.ceil(intermediate / x4_pack_size)
        I_p_padded = math.ceil(I_p_actual / 8) * 8
        _, tmp_w_qtz, tmp_w_scale = generate_stabilized_mx_data(mx_dtype, (I_p_padded, hidden * x4_pack_size))
        tmp_w_qtz = tmp_w_qtz[:I_p_actual, :]
        tmp_w_scale = tmp_w_scale[: math.ceil(I_p_actual / 8), :]
        for i_I512_tile in range(n_I512_tile):
            n_rows_qtz = min(128, intermediate // x4_pack_size - i_I512_tile * 128)
            n_rows_scale = min(16, math.ceil(intermediate / 32) - i_I512_tile * 16)
            down_w_qtz[e, :n_rows_qtz, i_I512_tile, :] = tmp_w_qtz[
                i_I512_tile * 128 : i_I512_tile * 128 + n_rows_qtz, :
            ]
            down_w_scale[e, :n_rows_scale, i_I512_tile, :] = tmp_w_scale[
                i_I512_tile * 16 : i_I512_tile * 16 + n_rows_scale, :
            ]

    return MxAllTokensWeights(gate_up_w_qtz, gate_up_w_scale, down_w_qtz, down_w_scale)


def modify_down_proj_lhs_rhs_swap_unit_stride_layout(tensor_template, tensor, lnc):
    if tensor_template.name == "down_w":
        I, H = tensor.shape
        kernel_assert(
            H // 128 >= lnc,
            f"Hidden dimension {H} must be at least {128 * lnc} to avoid zero dimension in reshape",
        )
        tensor = tensor.reshape((I, lnc, 128, H // 128 // lnc)).transpose((0, 1, 3, 2)).reshape((I, H))
    return tensor


def mlp_row_quant_tensor_generator(hidden_std: float = 1.0, weight_std: float = 1.0, scale_std: float = 0.001):
    """Create a tensor generator function that produces Gaussian-distributed tensors
    and allows for different distributions for different input tensor types.

    The hidden tensor will be generated with a packed fp32 scale. That scale will be
    generated with mean zero and scale_std.

    This factory function returns a tensor generator that creates tensors. The generator
    uses a configurable random seed for reproducibility.

    Args:
        hidden_std (float, optional): The standard deviation of the hidden tensor.
                                      Defaults to 1.0.
        weight_std (float, optional): The standard deviation of the weight tensors.
                                      Defaults to 1.0.
        scale_std (float, optional):  The standard deviation of the scale tensors.
                                      Defaults to 0.001.

    Returns:
        callable: A tensor generator function that accepts a tensor_template and
                  returns a NumPy array with the same shape and dtype as the template.
    """
    rng = np.random.default_rng(42)

    @update_func_str()
    def tensor_generator(shape, dtype, name):
        if name == "hidden":
            max_pos_val = get_max_positive_value_for_dtype(dtype)
            tensor = rng.normal(size=shape, scale=hidden_std).clip(-max_pos_val, max_pos_val).astype(dtype)
            B, S, H_PLUS_4 = shape
            scale_fp32 = rng.normal(size=(B, S, 1), scale=scale_std).astype(nl.float32)
            scale_fp8 = scale_fp32.view(dtype)
            tensor[:, :, -4:] = scale_fp8
        elif "_scale" in name:
            tensor = rng.normal(size=shape, scale=scale_std).astype(dtype)
        elif "_w" in name:
            max_pos_val = get_max_positive_value_for_dtype(dtype)
            tensor = rng.normal(size=shape, scale=weight_std).clip(-max_pos_val, max_pos_val).astype(dtype)
        else:
            tensor = rng.normal(size=shape)
        return tensor

    return tensor_generator


def modify_fp8_static_scale(tensor_template, tensor, lnc):
    rng = np.random.default_rng(0)
    if "scale" in tensor_template.name:
        scale = rng.normal() * 0.5
        return np.full(tensor_template.shape, scale, dtype=tensor_template.dtype)
    else:
        return tensor


def random_lhs_and_random_bound_weight_tensor_generator(weight_lower, weight_upper, modifier_fn=None, lnc=None):
    # make tests generate stable tensors
    rng = np.random.default_rng(42)

    @update_func_str()
    def tensor_generator(shape, dtype, name):
        """Generate tensor with specified shape, dtype, and name.

        Args:
            shape: Tuple specifying tensor dimensions
            dtype: Data type for the tensor
            name: Name for the tensor
        """
        if name in ("down_w", "gate_w", "up_w"):
            tensor = rng.uniform(weight_lower, weight_upper, shape).astype(dtype)
            if modifier_fn is not None:
                assert lnc is not None
                tensor = modifier_fn(
                    tensor_template=TensorTemplate(name=name, shape=shape, dtype=dtype),
                    tensor=tensor,
                    lnc=lnc,
                )
            return tensor
        elif name == "hidden":
            single_row = rng.uniform(weight_lower, weight_upper, size=(1, 1, shape[-1])).astype(dtype)
            full_tensor = single_row.repeat(shape[0], axis=0).repeat(shape[1], axis=1)
            return full_tensor
        elif name == "fused_add_tensor":
            return rng.uniform(weight_lower, weight_upper, shape).astype(dtype)
        elif name.endswith("in_scale") or name.endswith("w_scale"):
            return np.full(
                shape=shape,
                fill_value=rng.random() * 0.01,
                dtype=dtype,
            )
        else:
            return np.full(
                shape=shape,
                fill_value=rng.random(),
                dtype=dtype,
            )

    return tensor_generator


def generate_and_quantize_to_fp8(
    shape, dtype, quant_dtype, rng, quantization_type=QuantizationType.NONE, scale_shape=(), quantize_dim=0
):
    """Generate a random tensor, quantize to FP8, and return the quantized tensor with its scale.

    Args:
        shape: Tensor dimensions.
        dtype: Original dtype (e.g., nl.bfloat16).
        quant_dtype: Target FP8 dtype (nl.float8_e4m3 or nl.float8_e5m2).
        rng: NumPy random generator.
        quantization_type: Quantization type.
        scale_shape: Target shape for the returned scale tensor.
        quantize_dim: Axis along which to compute scale for ROW quantization.

    Returns:
        (fp8_tensor, fp32_scale): Quantized tensor and its scale with shape scale_shape.
    """
    tensor = (rng.standard_normal(shape) * 10.0).astype(dtype)

    fp8_max_map = {nl.float8_e4m3: 240.0, nl.float8_e5m2: 57344.0}
    max_val = fp8_max_map.get(quant_dtype)
    if max_val is None:
        raise ValueError(f"Unsupported quant_dtype: {quant_dtype}")

    if quantization_type == QuantizationType.ROW:
        original_shape = tensor.shape
        tensor_2d = tensor.reshape(-1, shape[-1])
        abs_max = np.max(np.absolute(tensor_2d), axis=quantize_dim, keepdims=True)
        scale = np.maximum(abs_max / max_val, 1e-05)
        fp8_tensor = (tensor_2d / scale).astype(quant_dtype).reshape(original_shape)
        scale_fp32 = np.broadcast_to(scale.astype(np.float32), scale_shape).copy()
    else:
        abs_max = np.max(np.absolute(tensor))
        scale = np.maximum(abs_max / max_val, 1e-05)
        fp8_tensor = (tensor / scale).astype(quant_dtype)

        if scale_shape == ():
            scale_fp32 = np.array(scale, dtype=np.float32)
        else:
            scale_fp32 = np.broadcast_to(np.array(scale, dtype=np.float32), scale_shape).copy()

    return fp8_tensor, scale_fp32


def _fp8_to_gate_up_x4(fp8_2d, H, I, contiguous_x4=False):
    """Convert scalar fp8 (H, I) to x4 packed (128, n_H512_tile, I) for gate/up projection.

    Supports two x4 packing conventions controlled by ``contiguous_x4``:

    **contiguous_x4=False (default):**
        Packs 4 consecutive H1 values per x4 element at each I column, matching the
        kernel's natural SBUF layout where H index = p + 128 * h1.

        In SBUF, activation at (p, h512, t) packs H indices:
            {p + 128*(4*h512+q) : q=0..3}   (stride-128)

        Weight at (p, h512, i) must pack the same H indices:
            fp8_2d[p + 128*(4*h512+0), i], fp8_2d[p + 128*(4*h512+1), i],
            fp8_2d[p + 128*(4*h512+2), i], fp8_2d[p + 128*(4*h512+3), i]

    **contiguous_x4=True (contiguous-4 H packing):**
        Packs 4 contiguous H values per x4 element:
            {4*p + q : q=0..3}   (stride-1)

        Weight at (p, h512, i) packs:
            fp8_2d[512*h512 + 4*p + 0, i], fp8_2d[512*h512 + 4*p + 1, i],
            fp8_2d[512*h512 + 4*p + 2, i], fp8_2d[512*h512 + 4*p + 3, i]
    """
    n_H512 = H // _psum_fmax
    if not contiguous_x4:
        # default: [H, I] → [n_H512, 4, 128, I] → transpose → [128, n_H512, I, 4] → pack x4
        # H index at (h512, q, p, i) = p + 128*(h512*4 + q)
        arr = fp8_2d.reshape(n_H512, _q_width, _pmax, I)  # [n_H512, 4, 128, I]
        arr = arr.transpose(2, 0, 3, 1)  # [128, n_H512, I, 4]
    else:
        # contiguous_x4: [H, I] → [n_H512, 128, 4, I] → transpose → [128, n_H512, I, 4] → pack x4
        # H index at (h512, p, q, i) = 512*h512 + 4*p + q
        arr = fp8_2d.reshape(n_H512, _pmax, _q_width, I)  # [n_H512, 128, 4, I]
        arr = arr.transpose(1, 0, 3, 2)  # [128, n_H512, I, 4]
    arr = arr.reshape(_pmax, n_H512, I * _q_width)  # [128, n_H512, I*4]
    packed = dt.static_cast(arr.astype(np.float32), nl.float8_e4m3fn_x4)  # [128, n_H512, I]
    return packed


def _fp8_to_down_x4(fp8_2d, I, H):
    """Convert scalar fp8 (I, H) to x4 packed (p_I, n_I512_tile, H) for down projection.

    Uses the I-contiguous x4 packing convention shared with the MLP CTE MX
    down projection: per I-512 tile, group 4 consecutive I rows per x4 element
    at the same H column::

        [I, H]
            → reshape [n_I512, 128_I, 4_I, H]
            → transpose [128_I, n_I512, H, 4_I]
            → x4 pack → [128_I, n_I512, H] (fp8_e4m3fn_x4)

    Element ``[p, tile, h]`` packs ``W[512*tile + 4p, h]``,
    ``W[512*tile + 4p+1, h]``, ``W[512*tile + 4p+2, h]``, ``W[512*tile + 4p+3, h]``
    — 4 consecutive I values at the same H column.

    This matches the contraction-dim layout consumed by ``nc_matmul_mx`` in
    ``down_projection_mx_tp_shard_H`` (partition dim × x4 = contraction = I)
    and aligns TKG with the CTE down-proj weight layout so both paths can
    share the same pre-packed weights.
    """
    n_I512 = math.ceil(I / _psum_fmax)
    p_I = I // _q_width if I < _psum_fmax else _pmax
    result = np.zeros((p_I, n_I512, H), dtype=nl.float8_e4m3fn_x4)
    for i_tile in range(n_I512):
        start = i_tile * _psum_fmax
        end = min(start + _psum_fmax, I)
        tile_rows = end - start
        tile = fp8_2d[start:end, :]  # [tile_rows, H]
        # Group 4 consecutive I rows (x4 lane) within the same H column.
        # [tile_rows, H] → [tile_rows//4, 4, H] → transpose → [tile_rows//4, H, 4]
        # → cast to x4 collapses the inner 4 into a single packed element.
        # static_cast preserves rank (trailing dim becomes size 1), so squeeze it.
        tile_i_contig = tile.reshape(tile_rows // _q_width, _q_width, H).transpose(0, 2, 1)
        tile_packed = dt.static_cast(np.ascontiguousarray(tile_i_contig).astype(np.float32), nl.float8_e4m3fn_x4)
        tile_packed = tile_packed.reshape(tile_rows // _q_width, H)
        result[: tile_rows // _q_width, i_tile, :] = tile_packed
    return result


def _fp8_to_gate_up_6d(fp8_2d, H, I, layout):
    """Convert scalar fp8 [H, I] to 6D [128, H/512, ceil(I/512), 4, 128, 4] for gate/up projection.

    Pads I to the nearest multiple of 512 with zeros if needed.

    Args:
        fp8_2d: scalar fp8 weight [H, I]
        H: hidden dimension
        I: intermediate dimension (may not be multiple of 512)
        layout: MLPGateUpWeightLayout.H_X4_MIDDLE or H_X4_INNERMOST

    Returns:
        [128, H/512, ceil(I/512), 4, 128, 4] in scalar fp8
    """
    n_H512 = H // _psum_fmax
    I_padded = math.ceil(I / _psum_fmax) * _psum_fmax
    n_I512 = I_padded // _psum_fmax

    # Pad I to multiple of 512
    if I < I_padded:
        padded = np.zeros((H, I_padded), dtype=fp8_2d.dtype)
        padded[:, :I] = fp8_2d
        fp8_2d = padded

    if layout == MLPGateUpWeightLayout.H_X4_MIDDLE:
        # [H, I] → [H/512, 4_H, 128_H, I/512, 128_I, 4_I] → transpose(2,0,3,5,4,1)
        # → [128_H, H/512, I/512, 4_I, 128_I, 4_H]
        return fp8_2d.reshape(n_H512, 4, _pmax, n_I512, _pmax, 4).transpose(2, 0, 3, 5, 4, 1).astype(nl.float8_e4m3fn)
    else:
        # H_X4_INNERMOST:
        # [H, I] → [H/512, 128_H, 4_H, I/512, 128_I, 4_I] → transpose(1,0,3,5,4,2)
        # → [128_H, H/512, I/512, 4_I, 128_I, 4_H]
        return fp8_2d.reshape(n_H512, _pmax, 4, n_I512, _pmax, 4).transpose(1, 0, 3, 5, 4, 2).astype(nl.float8_e4m3fn)


def _fp8_to_down_4d(fp8_2d, I, H):
    """Convert scalar fp8 [I, H] to 4D [128_I, I/512, H, 4_I] for down projection.

    Uses I-contiguous x4 packing: element [p, tile, h, q] = W[512*tile + 4*p + q, h].

    Args:
        fp8_2d: scalar fp8 weight [I, H]
        I: intermediate dimension
        H: hidden dimension

    Returns:
        [128_I, I/512, H, 4] in scalar fp8
    """
    n_I512 = math.ceil(I / _psum_fmax)
    p_I = I // _q_width if I < _psum_fmax else _pmax
    result = np.zeros((p_I, n_I512, H, _q_width), dtype=fp8_2d.dtype)
    for i_tile in range(n_I512):
        start = i_tile * _psum_fmax
        end = min(start + _psum_fmax, I)
        tile_rows = end - start
        tile = fp8_2d[start:end, :]  # [tile_rows, H]
        # Group 4 consecutive I rows: [tile_rows, H] → [tile_rows//4, 4, H] → [tile_rows//4, H, 4]
        tile_grouped = tile.reshape(tile_rows // _q_width, _q_width, H).transpose(0, 2, 1)
        result[: tile_rows // _q_width, i_tile, :, :] = tile_grouped
    return result.astype(nl.float8_e4m3fn)


def setup_sbuf_input(hidden_tensor):
    """Load HBM input into SBUF for mlp kernel. Returns (hidden_sb, sbm)."""
    B, S, H = hidden_tensor.shape
    T = B * S
    H0 = nl.tile_size.pmax
    H1 = H // H0

    _, num_shards, _ = get_verified_program_sharding_info()
    H1_shard = H1 // num_shards
    hidden_sb = nl.ndarray(shape=(H0, T, H1), dtype=hidden_tensor.dtype, buffer=nl.sbuf, name="hidden_sb")
    input_view = (
        hidden_tensor.flatten_dims(start_dim=0, end_dim=1)
        .reshape_dim(dim=1, shape=[num_shards, H0, H1_shard])
        .permute(dims=[2, 0, 1, 3])
    )
    dst_view = hidden_sb.reshape_dim(dim=2, shape=[num_shards, H1_shard])
    nisa.dma_copy(dst=dst_view, src=input_view)

    sbm = SbufManager(0, 200 * 1024, logger=Logger("mlp-tkg-sbuf-input"), use_auto_alloc=False)
    sbm.set_name_prefix("mlp_")
    return hidden_sb, sbm


def copy_sbuf_output_to_hbm(output_sb, hidden_tensor, down_proj_weights_tensor):
    """Copy SBUF output back to HBM. Returns [output_hbm] reshaped to (B, S, H_out)."""
    B, S, H = hidden_tensor.shape
    T = B * S
    _, num_shards, shard_id = get_verified_program_sharding_info()

    H_out = down_proj_weights_tensor.shape[-1]
    H_per_shard = H_out // num_shards
    output_hbm = nl.ndarray((T, H_out), dtype=hidden_tensor.dtype, buffer=nl.shared_hbm)
    output_pattern = [[H_out, T], [1, H_per_shard]]
    output_offset = shard_id * H_per_shard
    nisa.dma_copy(dst=output_hbm.ap(pattern=output_pattern, offset=output_offset), src=output_sb)
    return [output_hbm.reshape((B, S, H_out))]


def _shuffle_gate_up_w_scale_for_row_mx_h_x4_middle(original_scale, I):
    """Pre-shuffle gate/up weight scale for ROW_MX with I-dim in [n_I512, 4_I, 128_I] order.

    In the 6D layout, the I dimension within each 512-tile is ordered as (4, 128).
    The kernel iterates I as i_tile*512 + q*128 + p, so the element at kernel position
    (p, i_tile, q) corresponds to original I position i_tile*512 + p*4 + q.

    shuffled[p, i_tile*4 + q] = original[0, i_tile*512 + p*4 + q]
    """
    n_I512 = math.ceil(I / (_pmax * _q_width))
    shuffled = np.zeros((_pmax, n_I512 * _q_width), dtype=original_scale.dtype)
    for i_tile in range(n_I512):
        for q in range(_q_width):
            out_col = i_tile * _q_width + q
            for p in range(_pmax):
                i_pos = i_tile * _psum_fmax + p * _q_width + q
                if i_pos < I:
                    shuffled[p, out_col] = original_scale[0, i_pos]
    return shuffled


def _shuffle_down_w_scale_for_row_mx(original_scale, H):
    """Pre-shuffle down weight scale from [128, H] to [128, H//128] for ROW_MX.

    shuffled[p, h_col] = original[0, h_col * 128 + p]
    """
    H1 = H // _pmax
    # original[0, :] reshaped to [H1, 128] then transposed gives [128, H1]
    return original_scale[0, : H1 * _pmax].reshape(H1, _pmax).T.copy()
