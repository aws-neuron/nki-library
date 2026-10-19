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
This module implements a fused Adam/AdamW/AMSGrad optimizer kernel with SPMD tiling and Scalar Engine fusion.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil


@nki.jit
def adam_kernel(
    param_ptr: nl.NkiTensor,
    grad_ptr: nl.NkiTensor,
    exp_avg_ptr: nl.NkiTensor,
    exp_avg_sq_ptr: nl.NkiTensor,
    max_exp_avg_sq_ptr: nl.NkiTensor | None,
    step_size_ptr: nl.NkiTensor,
    inv_bc2_sqrt_ptr: nl.NkiTensor,
    wd_factor_ptr: nl.NkiTensor,
    numel: int,
    beta1: float,
    beta2: float,
    eps: float,
    amsgrad: bool = False,
):
    """
    Adam optimizer kernel with L2 regularization.

    This kernel implements the Adam optimizer with L2 weight regularization.
    For decoupled weight decay (AdamW), use adamw_kernel instead.

    TODO: Specify intended usage range (e.g., parameter count, batch size)

    Dimensions:
        N: Total number of elements in the parameter tensor (numel)
        ...: Any tensor shape containing N elements; all optimizer buffers share this shape
        P_MAX: Partition dimension size (128)

    Args:
        param_ptr (nl.NkiTensor): [...], Parameter tensor on HBM
        grad_ptr (nl.NkiTensor): [...], Gradient tensor on HBM
        exp_avg_ptr (nl.NkiTensor): [...], First moment estimate tensor on HBM
        exp_avg_sq_ptr (nl.NkiTensor): [...], Second moment estimate tensor on HBM
        max_exp_avg_sq_ptr (nl.NkiTensor): [...], Max second moment tensor on HBM (AMSGrad only)
        step_size_ptr (nl.NkiTensor): [P_MAX, 1], Bias-corrected step size on HBM
        inv_bc2_sqrt_ptr (nl.NkiTensor): [P_MAX, 1], Inverse bias correction sqrt on HBM
        wd_factor_ptr (nl.NkiTensor): [P_MAX, 1], Weight decay coefficient (lambda) on HBM
        numel (int): Number of elements in the parameter tensor
        beta1 (float): First moment decay factor (typically 0.9)
        beta2 (float): Second moment decay factor (typically 0.999)
        eps (float): Epsilon for numerical stability (typically 1e-8)
        amsgrad (bool): If True, use AMSGrad variant (default: False)

    Returns:
        param_out (nl.NkiTensor): [...], Updated parameter tensor on HBM
        exp_avg_out (nl.NkiTensor): [...], Updated first moment tensor on HBM
        exp_avg_sq_out (nl.NkiTensor): [...], Updated second moment tensor on HBM
        max_exp_avg_sq_out (nl.NkiTensor): [...], Updated max second moment (if amsgrad=True)

    Notes:
        - Uses L2 regularization: grad = grad + lambda * param
        - For decoupled weight decay (AdamW), use adamw_kernel instead
        - FP16 tensors compute in FP32 SBUF and store back in FP16.
          BF16 and FP32 arithmetic is unchanged.

    Pseudocode:
        # Adam with L2 regularization
        grad = grad + wd_factor * param
        exp_avg = beta1 * exp_avg + (1 - beta1) * grad
        exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
        denom = sqrt(exp_avg_sq) * inv_bc2_sqrt + eps
        update = exp_avg / denom * step_size
        param = param - update
    """
    _validate_adam_inputs(
        param_ptr,
        grad_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        max_exp_avg_sq_ptr,
        step_size_ptr,
        inv_bc2_sqrt_ptr,
        wd_factor_ptr,
        numel,
        beta1,
        beta2,
        eps,
        amsgrad,
    )
    return _adam_kernel_impl(
        param_ptr,
        grad_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        max_exp_avg_sq_ptr,
        step_size_ptr,
        inv_bc2_sqrt_ptr,
        wd_factor_ptr,
        numel,
        beta1,
        beta2,
        eps,
        decoupled_wd=False,
        amsgrad=amsgrad,
    )


@nki.jit
def adamw_kernel(
    param_ptr: nl.NkiTensor,
    grad_ptr: nl.NkiTensor,
    exp_avg_ptr: nl.NkiTensor,
    exp_avg_sq_ptr: nl.NkiTensor,
    max_exp_avg_sq_ptr: nl.NkiTensor | None,
    step_size_ptr: nl.NkiTensor,
    inv_bc2_sqrt_ptr: nl.NkiTensor,
    wd_factor_ptr: nl.NkiTensor,
    numel: int,
    beta1: float,
    beta2: float,
    eps: float,
    amsgrad: bool = False,
):
    """
    AdamW optimizer kernel with decoupled weight decay.

    This kernel implements the AdamW optimizer with decoupled weight decay.
    For L2 regularization (Adam), use adam_kernel instead.

    TODO: Specify intended usage range (e.g., parameter count, batch size)

    Dimensions:
        N: Total number of elements in the parameter tensor (numel)
        ...: Any tensor shape containing N elements; all optimizer buffers share this shape
        P_MAX: Partition dimension size (128)

    Args:
        param_ptr (nl.NkiTensor): [...], Parameter tensor on HBM
        grad_ptr (nl.NkiTensor): [...], Gradient tensor on HBM
        exp_avg_ptr (nl.NkiTensor): [...], First moment estimate tensor on HBM
        exp_avg_sq_ptr (nl.NkiTensor): [...], Second moment estimate tensor on HBM
        max_exp_avg_sq_ptr (nl.NkiTensor): [...], Max second moment tensor on HBM (AMSGrad only)
        step_size_ptr (nl.NkiTensor): [P_MAX, 1], Bias-corrected step size on HBM
        inv_bc2_sqrt_ptr (nl.NkiTensor): [P_MAX, 1], Inverse bias correction sqrt on HBM
        wd_factor_ptr (nl.NkiTensor): [P_MAX, 1], Weight decay factor (1 - lr*lambda) on HBM
        numel (int): Number of elements in the parameter tensor
        beta1 (float): First moment decay factor (typically 0.9)
        beta2 (float): Second moment decay factor (typically 0.999)
        eps (float): Epsilon for numerical stability (typically 1e-8)
        amsgrad (bool): If True, use AMSGrad variant (default: False)

    Returns:
        param_out (nl.NkiTensor): [...], Updated parameter tensor on HBM
        exp_avg_out (nl.NkiTensor): [...], Updated first moment tensor on HBM
        exp_avg_sq_out (nl.NkiTensor): [...], Updated second moment tensor on HBM
        max_exp_avg_sq_out (nl.NkiTensor): [...], Updated max second moment (if amsgrad=True)

    Notes:
        - Uses decoupled weight decay: param = param * wd_factor - update
        - For L2 regularization (Adam), use adam_kernel instead
        - FP16 tensors compute in FP32 SBUF and store back in FP16.
          BF16 and FP32 arithmetic is unchanged.

    Pseudocode:
        # AdamW with decoupled weight decay
        exp_avg = beta1 * exp_avg + (1 - beta1) * grad
        exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
        denom = sqrt(exp_avg_sq) * inv_bc2_sqrt + eps
        update = exp_avg / denom * step_size
        param = param * wd_factor - update
    """
    _validate_adam_inputs(
        param_ptr,
        grad_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        max_exp_avg_sq_ptr,
        step_size_ptr,
        inv_bc2_sqrt_ptr,
        wd_factor_ptr,
        numel,
        beta1,
        beta2,
        eps,
        amsgrad,
    )
    return _adam_kernel_impl(
        param_ptr,
        grad_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        max_exp_avg_sq_ptr,
        step_size_ptr,
        inv_bc2_sqrt_ptr,
        wd_factor_ptr,
        numel,
        beta1,
        beta2,
        eps,
        decoupled_wd=True,
        amsgrad=amsgrad,
    )


# Each step holds parameter, gradient, two moments, and a temporary SBUF tile;
# AMSGrad adds a max-moment tile. FP16 inputs compute in FP32 SBUF and use the
# FP32 tile width. BF16 keeps its existing SBUF dtype and wider tile.
_F_SIZE_BF16 = 4096
_F_SIZE_F32 = 2048


def _validate_adam_inputs(
    param_ptr: nl.NkiTensor,
    grad_ptr: nl.NkiTensor,
    exp_avg_ptr: nl.NkiTensor,
    exp_avg_sq_ptr: nl.NkiTensor,
    max_exp_avg_sq_ptr: nl.NkiTensor | None,
    step_size_ptr: nl.NkiTensor,
    inv_bc2_sqrt_ptr: nl.NkiTensor,
    wd_factor_ptr: nl.NkiTensor,
    numel: int,
    beta1: float,
    beta2: float,
    eps: float,
    amsgrad: bool,
):
    """Validate the public Adam and AdamW input contract before tiling."""
    kernel_assert(numel > 0, f"numel must be positive, got {numel}")
    kernel_assert(param_ptr.size == numel, f"param_ptr size {param_ptr.size} must equal numel {numel}")
    kernel_assert(
        param_ptr.dtype == nl.float16 or param_ptr.dtype == nl.bfloat16 or param_ptr.dtype == nl.float32,
        f"param_ptr dtype must be float16, bfloat16, or float32, got {param_ptr.dtype}",
    )
    kernel_assert(grad_ptr.shape == param_ptr.shape, f"grad_ptr shape {grad_ptr.shape} must equal {param_ptr.shape}")
    kernel_assert(
        exp_avg_ptr.shape == param_ptr.shape, f"exp_avg_ptr shape {exp_avg_ptr.shape} must equal {param_ptr.shape}"
    )
    kernel_assert(
        exp_avg_sq_ptr.shape == param_ptr.shape,
        f"exp_avg_sq_ptr shape {exp_avg_sq_ptr.shape} must equal {param_ptr.shape}",
    )
    kernel_assert(grad_ptr.dtype == param_ptr.dtype, f"grad_ptr dtype {grad_ptr.dtype} must equal {param_ptr.dtype}")
    kernel_assert(
        exp_avg_ptr.dtype == param_ptr.dtype, f"exp_avg_ptr dtype {exp_avg_ptr.dtype} must equal {param_ptr.dtype}"
    )
    kernel_assert(
        exp_avg_sq_ptr.dtype == param_ptr.dtype,
        f"exp_avg_sq_ptr dtype {exp_avg_sq_ptr.dtype} must equal {param_ptr.dtype}",
    )
    if amsgrad:
        kernel_assert(
            max_exp_avg_sq_ptr.shape == param_ptr.shape,
            f"max_exp_avg_sq_ptr shape {max_exp_avg_sq_ptr.shape} must equal {param_ptr.shape}",
        )
        kernel_assert(
            max_exp_avg_sq_ptr.dtype == param_ptr.dtype,
            f"max_exp_avg_sq_ptr dtype {max_exp_avg_sq_ptr.dtype} must equal {param_ptr.dtype}",
        )

    scalar_shape = (nl.tile_size.pmax, 1)
    kernel_assert(
        step_size_ptr.shape == scalar_shape, f"step_size_ptr shape {step_size_ptr.shape} must equal {scalar_shape}"
    )
    kernel_assert(
        inv_bc2_sqrt_ptr.shape == scalar_shape,
        f"inv_bc2_sqrt_ptr shape {inv_bc2_sqrt_ptr.shape} must equal {scalar_shape}",
    )
    kernel_assert(
        wd_factor_ptr.shape == scalar_shape, f"wd_factor_ptr shape {wd_factor_ptr.shape} must equal {scalar_shape}"
    )
    kernel_assert(
        step_size_ptr.dtype == inv_bc2_sqrt_ptr.dtype,
        f"step_size_ptr dtype {step_size_ptr.dtype} must equal {inv_bc2_sqrt_ptr.dtype}",
    )
    kernel_assert(
        step_size_ptr.dtype == wd_factor_ptr.dtype,
        f"step_size_ptr dtype {step_size_ptr.dtype} must equal {wd_factor_ptr.dtype}",
    )
    kernel_assert(0.0 <= beta1 < 1.0, f"beta1 must be in [0, 1), got {beta1}")
    kernel_assert(0.0 <= beta2 < 1.0, f"beta2 must be in [0, 1), got {beta2}")
    kernel_assert(eps >= 0.0, f"eps must be nonnegative, got {eps}")


def _adam_tile_compute(
    param: nl.NkiTensor,
    grad: nl.NkiTensor,
    exp_avg: nl.NkiTensor,
    exp_avg_sq: nl.NkiTensor,
    max_exp_avg_sq: nl.NkiTensor | None,
    step_size: nl.NkiTensor,
    inv_bc2_sqrt: nl.NkiTensor,
    weight_decay: nl.NkiTensor,
    beta1: float,
    beta2: float,
    eps: float,
    one_minus_beta1: float,
    one_minus_beta2: float,
    decoupled_wd: bool,
    amsgrad: bool,
):
    """
    Pure Adam compute for one tile.

    All inputs/outputs are SBUF tensors. Performs the core Adam/AdamW/AMSGrad
    update computation including moment updates and parameter updates.

    Args:
        param (nl.NkiTensor): Parameter tensor in SBUF
        grad (nl.NkiTensor): Gradient tensor in SBUF
        exp_avg (nl.NkiTensor): First moment estimate in SBUF
        exp_avg_sq (nl.NkiTensor): Second moment estimate in SBUF
        max_exp_avg_sq (nl.NkiTensor): Max second moment (AMSGrad only) in SBUF
        step_size (nl.NkiTensor): Bias-corrected learning rate in SBUF
        inv_bc2_sqrt (nl.NkiTensor): Inverse bias correction sqrt in SBUF
        weight_decay (nl.NkiTensor): Weight decay factor in SBUF
        beta1 (float): First moment decay factor
        beta2 (float): Second moment decay factor
        eps (float): Epsilon for numerical stability
        one_minus_beta1 (float): 1 - beta1
        one_minus_beta2 (float): 1 - beta2
        decoupled_wd (bool): If True, use AdamW decoupled weight decay
        amsgrad (bool): If True, use AMSGrad variant

    Returns:
        tuple: (param, exp_avg, exp_avg_sq, max_exp_avg_sq) updated tensors
    """
    temp0 = nl.ndarray(exp_avg.shape, dtype=exp_avg.dtype, buffer=nl.sbuf)
    if not decoupled_wd:
        nisa.scalar_tensor_tensor(
            dst=grad, data=param, op0=nl.multiply, operand0=weight_decay, op1=nl.add, operand1=grad
        )

    # First moment: exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    nisa.tensor_scalar(dst=temp0, data=exp_avg, op0=nl.multiply, operand0=beta1, engine=nisa.scalar_engine)
    nisa.scalar_tensor_tensor(
        dst=exp_avg, data=grad, op0=nl.multiply, operand0=one_minus_beta1, op1=nl.add, operand1=temp0
    )

    # Second moment: exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
    nisa.tensor_scalar(dst=temp0, data=exp_avg_sq, op0=nl.multiply, operand0=beta2, engine=nisa.scalar_engine)
    nisa.tensor_tensor(dst=grad, data1=grad, data2=grad, op=nl.multiply)  # grad = grad^2
    nisa.scalar_tensor_tensor(
        dst=exp_avg_sq, data=grad, op0=nl.multiply, operand0=one_minus_beta2, op1=nl.add, operand1=temp0
    )

    # AMSGrad: denom_src = max(max_exp_avg_sq, exp_avg_sq)
    if amsgrad:
        nisa.tensor_tensor(dst=max_exp_avg_sq, data1=max_exp_avg_sq, data2=exp_avg_sq, op=nl.maximum)
        denom_src = max_exp_avg_sq
    else:
        denom_src = exp_avg_sq

    # denom = sqrt(denom_src) * inv_bc2_sqrt + eps; update = exp_avg / denom * step_size
    nisa.activation(dst=temp0, op=nl.sqrt, data=denom_src)
    nisa.tensor_scalar(
        dst=temp0,
        data=temp0,
        op0=nl.multiply,
        operand0=inv_bc2_sqrt,
        op1=nl.add,
        operand1=eps,
        engine=nisa.scalar_engine,
    )
    nisa.activation(dst=temp0, data=temp0, op=nl.reciprocal)
    nisa.tensor_tensor(dst=grad, data1=exp_avg, data2=temp0, op=nl.multiply)
    nisa.tensor_scalar(dst=grad, data=grad, op0=nl.multiply, operand0=step_size, engine=nisa.scalar_engine)

    # param_new: AdamW decoupled decay param = param * weight_decay - update; Adam: param = param - update
    if decoupled_wd:
        nisa.scalar_tensor_tensor(
            dst=param, data=param, op0=nl.multiply, operand0=weight_decay, op1=nl.subtract, operand1=grad
        )
    else:
        nisa.tensor_tensor(dst=param, data1=param, data2=grad, op=nl.subtract)

    return param, exp_avg, exp_avg_sq, max_exp_avg_sq


def _adam_tile_step(
    param_ptr: nl.NkiTensor,
    grad_ptr: nl.NkiTensor,
    exp_avg_ptr: nl.NkiTensor,
    exp_avg_sq_ptr: nl.NkiTensor,
    max_exp_avg_sq_ptr: nl.NkiTensor | None,
    param_out: nl.NkiTensor,
    exp_avg_out: nl.NkiTensor,
    exp_avg_sq_out: nl.NkiTensor,
    max_exp_avg_sq_out: nl.NkiTensor,
    shape: tuple[int, int],
    access_pattern: list[list[int]],
    offset: int,
    step_size: nl.NkiTensor,
    inv_bc2_sqrt: nl.NkiTensor,
    weight_decay: nl.NkiTensor,
    beta1: float,
    beta2: float,
    eps: float,
    one_minus_beta1: float,
    one_minus_beta2: float,
    decoupled_wd: bool,
    amsgrad: bool,
):
    """
    Load inputs, call _adam_tile_compute, store outputs.

    Shape/access_pattern/offset differ between main and tail tiles.
    FP16 HBM tensors are loaded into FP32 SBUF tiles and converted back on
    store, keeping epsilon and the moment update representable during compute.
    BF16 and FP32 keep their existing SBUF arithmetic.

    Args:
        param_ptr (nl.NkiTensor): Parameter tensor pointer on HBM
        grad_ptr (nl.NkiTensor): Gradient tensor pointer on HBM
        exp_avg_ptr (nl.NkiTensor): First moment pointer on HBM
        exp_avg_sq_ptr (nl.NkiTensor): Second moment pointer on HBM
        max_exp_avg_sq_ptr (nl.NkiTensor): Max second moment pointer on HBM
        param_out (nl.NkiTensor): Output parameter tensor on HBM
        exp_avg_out (nl.NkiTensor): Output first moment on HBM
        exp_avg_sq_out (nl.NkiTensor): Output second moment on HBM
        max_exp_avg_sq_out (nl.NkiTensor): Output max second moment on HBM
        shape (tuple): Shape of the tile
        access_pattern (list): Access pattern for DMA
        offset (int): Offset for DMA
        step_size (nl.NkiTensor): Bias-corrected learning rate
        inv_bc2_sqrt (nl.NkiTensor): Inverse bias correction sqrt
        weight_decay (nl.NkiTensor): Weight decay factor
        beta1 (float): First moment decay factor
        beta2 (float): Second moment decay factor
        eps (float): Epsilon for numerical stability
        one_minus_beta1 (float): 1 - beta1
        one_minus_beta2 (float): 1 - beta2
        decoupled_wd (bool): If True, use AdamW decoupled weight decay
        amsgrad (bool): If True, use AMSGrad variant
    """
    # DMA converts FP16 HBM values into FP32 SBUF and rounds results back to
    # the caller's storage dtype on store.
    tile_dtype = nl.float32 if param_ptr.dtype == nl.float16 else param_ptr.dtype
    param = nl.ndarray(shape, dtype=tile_dtype, buffer=nl.sbuf)
    grad = nl.ndarray(shape, dtype=tile_dtype, buffer=nl.sbuf)
    exp_avg = nl.ndarray(shape, dtype=tile_dtype, buffer=nl.sbuf)
    exp_avg_sq = nl.ndarray(shape, dtype=tile_dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=param, src=param_ptr.ap(access_pattern, offset))
    nisa.dma_copy(dst=grad, src=grad_ptr.ap(access_pattern, offset))
    nisa.dma_copy(dst=exp_avg, src=exp_avg_ptr.ap(access_pattern, offset))
    nisa.dma_copy(dst=exp_avg_sq, src=exp_avg_sq_ptr.ap(access_pattern, offset))

    if amsgrad:
        max_exp_avg_sq = nl.ndarray(shape, dtype=tile_dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=max_exp_avg_sq, src=max_exp_avg_sq_ptr.ap(access_pattern, offset))
    else:
        max_exp_avg_sq = None

    param, exp_avg, exp_avg_sq, max_exp_avg_sq = _adam_tile_compute(
        param,
        grad,
        exp_avg,
        exp_avg_sq,
        max_exp_avg_sq,
        step_size,
        inv_bc2_sqrt,
        weight_decay,
        beta1,
        beta2,
        eps,
        one_minus_beta1,
        one_minus_beta2,
        decoupled_wd,
        amsgrad,
    )

    nisa.dma_copy(dst=param_out.ap(access_pattern, offset), src=param)
    nisa.dma_copy(dst=exp_avg_out.ap(access_pattern, offset), src=exp_avg)
    nisa.dma_copy(dst=exp_avg_sq_out.ap(access_pattern, offset), src=exp_avg_sq)
    if amsgrad:
        nisa.dma_copy(dst=max_exp_avg_sq_out.ap(access_pattern, offset), src=max_exp_avg_sq)


def _adam_kernel_impl(
    param_ptr: nl.NkiTensor,
    grad_ptr: nl.NkiTensor,
    exp_avg_ptr: nl.NkiTensor,
    exp_avg_sq_ptr: nl.NkiTensor,
    max_exp_avg_sq_ptr: nl.NkiTensor | None,
    step_size_ptr: nl.NkiTensor,
    inv_bc2_sqrt_ptr: nl.NkiTensor,
    wd_factor_ptr: nl.NkiTensor,
    numel: int,
    beta1: float,
    beta2: float,
    eps: float,
    decoupled_wd: bool,
    amsgrad: bool,
):
    """
    Internal implementation for Adam/AdamW/AMSGrad optimizer kernel with SPMD tiling and Scalar Engine fusion.

    This kernel implements the Adam optimizer family (Adam, AdamW, AMSGrad) with efficient
    SPMD parallelization across 2 cores and fused scalar operations. It uses flat memory
    access patterns via .ap() for optimal performance.

    TODO: Specify intended usage range (e.g., parameter count, batch size)

    Dimensions:
        N: Total number of elements in the parameter tensor (numel)
        ...: Any tensor shape containing N elements; all optimizer buffers share this shape
        P_MAX: Partition dimension size (128)

    Args:
        param_ptr (nl.NkiTensor): [...], Parameter tensor on HBM
        grad_ptr (nl.NkiTensor): [...], Gradient tensor on HBM
        exp_avg_ptr (nl.NkiTensor): [...], First moment estimate tensor on HBM
        exp_avg_sq_ptr (nl.NkiTensor): [...], Second moment estimate tensor on HBM
        max_exp_avg_sq_ptr (nl.NkiTensor): [...], Max second moment tensor on HBM (AMSGrad only)
        step_size_ptr (nl.NkiTensor): [P_MAX, 1], Bias-corrected step size on HBM
        inv_bc2_sqrt_ptr (nl.NkiTensor): [P_MAX, 1], Inverse bias correction sqrt on HBM
        wd_factor_ptr (nl.NkiTensor): [P_MAX, 1], Weight decay factor on HBM
        numel (int): Number of elements in the parameter tensor
        beta1 (float): First moment decay factor (typically 0.9)
        beta2 (float): Second moment decay factor (typically 0.999)
        eps (float): Epsilon for numerical stability (typically 1e-8)
        decoupled_wd (bool): If True, use AdamW decoupled weight decay; else Adam L2 reg
        amsgrad (bool): If True, use AMSGrad variant with max second moment

    Returns:
        param_out (nl.NkiTensor): [...], Updated parameter tensor on HBM
        exp_avg_out (nl.NkiTensor): [...], Updated first moment tensor on HBM
        exp_avg_sq_out (nl.NkiTensor): [...], Updated second moment tensor on HBM
        max_exp_avg_sq_out (nl.NkiTensor): [...], Updated max second moment tensor on HBM (if amsgrad=True)

    Notes:
        - SPMD parallelization splits work across num_programs cores
        - Scalar tensors (step_size, inv_bc2_sqrt, wd_factor) have shape [P_MAX, 1]
        - FP16 inputs use FP32 SBUF arithmetic; HBM parameters and optimizer
          state remain FP16. BF16 and FP32 retain their existing arithmetic.
        - For AdamW (decoupled_wd=True): param = param * wd_factor - update
        - For Adam (decoupled_wd=False): grad = grad + wd_factor * param, then param = param - update
        - AMSGrad uses max(max_exp_avg_sq, exp_avg_sq) as denominator

    Pseudocode:
        # Variable naming conventions:
        # param = parameter tensor
        # grad = gradient tensor
        # exp_avg = first moment estimate (m_t = E[grad])
        # exp_avg_sq = second moment estimate (v_t = E[grad^2])
        # max_exp_avg_sq = running max of v_t (AMSGrad only, v_hat_t)
        # step_size = lr / (1 - beta1^t), bias-corrected learning rate
        # inv_bc2_sqrt = 1 / sqrt(1 - beta2^t), bias correction for v_t
        # wd_factor = weight decay factor: (1 - lr*lambda) for AdamW, lambda for Adam

        one_minus_beta1 = 1.0 - beta1
        one_minus_beta2 = 1.0 - beta2

        # SPMD split work across cores
        per_core = div_ceil(numel, num_cores)
        core_base = core_id * per_core
        core_n = min(per_core, numel - core_base)

        # Process tiles
        for tile_idx in range(num_tiles):
            # Load tile from HBM to SBUF, promoting FP16 inputs to FP32
            param_tile = load(param_ptr[tile_offset:tile_offset+tile_size])
            grad_tile = load(grad_ptr[tile_offset:tile_offset+tile_size])
            exp_avg_tile = load(exp_avg_ptr[tile_offset:tile_offset+tile_size])
            exp_avg_sq_tile = load(exp_avg_sq_ptr[tile_offset:tile_offset+tile_size])

            # Adam L2 regularization (if not decoupled)
            if not decoupled_wd:
                grad_tile = grad_tile + wd_factor * param_tile

            # Update first moment
            exp_avg_tile = beta1 * exp_avg_tile + (1 - beta1) * grad_tile

            # Update second moment
            exp_avg_sq_tile = beta2 * exp_avg_sq_tile + (1 - beta2) * grad_tile^2

            # Compute denominator
            if amsgrad:
                max_exp_avg_sq_tile = max(max_exp_avg_sq_tile, exp_avg_sq_tile)
                denom = sqrt(max_exp_avg_sq_tile) * inv_bc2_sqrt + eps
            else:
                denom = sqrt(exp_avg_sq_tile) * inv_bc2_sqrt + eps

            # Compute update
            update = exp_avg_tile / denom * step_size

            # Update parameter
            if decoupled_wd:
                param_tile = param_tile * wd_factor - update
            else:
                param_tile = param_tile - update

            # Store tile back to HBM in the caller's dtype
            store(param_out[tile_offset:tile_offset+tile_size], param_tile)
            store(exp_avg_out[tile_offset:tile_offset+tile_size], exp_avg_tile)
            store(exp_avg_sq_out[tile_offset:tile_offset+tile_size], exp_avg_sq_tile)

        # Handle tail (remainder elements)
        if tail > 0:
            # Process tail with adjusted tile size
            ...
    """
    P_MAX = nl.tile_size.pmax
    one_minus_beta1 = 1.0 - beta1
    one_minus_beta2 = 1.0 - beta2

    param_out = nl.ndarray(param_ptr.shape, dtype=param_ptr.dtype, buffer=nl.shared_hbm)
    exp_avg_out = nl.ndarray(param_ptr.shape, dtype=param_ptr.dtype, buffer=nl.shared_hbm)
    exp_avg_sq_out = nl.ndarray(param_ptr.shape, dtype=param_ptr.dtype, buffer=nl.shared_hbm)
    max_exp_avg_sq_out = nl.ndarray(param_ptr.shape, dtype=param_ptr.dtype, buffer=nl.shared_hbm)

    # Load scalar tensors (shape [P_MAX, 1], same value across all partitions)
    step_size = nl.ndarray(step_size_ptr.shape, dtype=step_size_ptr.dtype, buffer=nl.sbuf)
    inv_bc2_sqrt = nl.ndarray(inv_bc2_sqrt_ptr.shape, dtype=inv_bc2_sqrt_ptr.dtype, buffer=nl.sbuf)
    weight_decay = nl.ndarray(wd_factor_ptr.shape, dtype=wd_factor_ptr.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=step_size, src=step_size_ptr)
    nisa.dma_copy(dst=inv_bc2_sqrt, src=inv_bc2_sqrt_ptr)
    nisa.dma_copy(dst=weight_decay, src=wd_factor_ptr)

    f_size = _F_SIZE_F32 if param_ptr.dtype == nl.float32 or param_ptr.dtype == nl.float16 else _F_SIZE_BF16
    tile_size = P_MAX * f_size

    # SPMD split
    num_progs = nl.num_programs(0)
    prog_id = nl.program_id(0)
    per_prog = div_ceil(numel, num_progs)
    core_base = prog_id * per_prog
    core_n = min(per_prog, numel - core_base)
    core_n_partitions, tail = divmod(core_n, P_MAX)
    core_n_aligned = core_n_partitions * P_MAX
    num_tiles = div_ceil(core_n_aligned, tile_size)

    for tile_idx in nl.affine_range(num_tiles):
        offset = core_base + tile_idx * tile_size
        masked_f = min(f_size, (core_n_aligned - tile_idx * tile_size) // P_MAX)
        _adam_tile_step(
            param_ptr,
            grad_ptr,
            exp_avg_ptr,
            exp_avg_sq_ptr,
            max_exp_avg_sq_ptr,
            param_out,
            exp_avg_out,
            exp_avg_sq_out,
            max_exp_avg_sq_out,
            (P_MAX, masked_f),
            [[masked_f, P_MAX], [1, masked_f]],
            offset,
            step_size,
            inv_bc2_sqrt,
            weight_decay,
            beta1,
            beta2,
            eps,
            one_minus_beta1,
            one_minus_beta2,
            decoupled_wd,
            amsgrad,
        )

    if tail > 0:
        tail_offset = core_base + core_n_aligned
        _adam_tile_step(
            param_ptr,
            grad_ptr,
            exp_avg_ptr,
            exp_avg_sq_ptr,
            max_exp_avg_sq_ptr,
            param_out,
            exp_avg_out,
            exp_avg_sq_out,
            max_exp_avg_sq_out,
            (tail, 1),
            [[1, tail], [1, 1]],
            tail_offset,
            step_size[0:tail, 0:1],
            inv_bc2_sqrt[0:tail, 0:1],
            weight_decay[0:tail, 0:1],
            beta1,
            beta2,
            eps,
            one_minus_beta1,
            one_minus_beta2,
            decoupled_wd,
            amsgrad,
        )

    if amsgrad:
        return param_out, exp_avg_out, exp_avg_sq_out, max_exp_avg_sq_out
    return param_out, exp_avg_out, exp_avg_sq_out
