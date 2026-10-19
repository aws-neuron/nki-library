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

"""PyTorch reference for the Adam NKI kernel."""

from ._adam_torch_common import _adam_ref, _recover_hyperparams


def adam_kernel_torch_ref(
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
    amsgrad=False,
):
    """Torch ref with signature matching adam_kernel."""
    lr, weight_decay, step = _recover_hyperparams(
        step_size_ptr,
        inv_bc2_sqrt_ptr,
        wd_factor_ptr,
        beta1,
        beta2,
        decoupled_wd=False,
    )
    _adam_ref(
        param_ptr,
        grad_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        max_exp_avg_sq_ptr,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        step,
        amsgrad,
        decoupled_wd=False,
    )

    result = {"param_out": param_ptr, "exp_avg_out": exp_avg_ptr, "exp_avg_sq_out": exp_avg_sq_ptr}
    if amsgrad:
        result["max_exp_avg_sq_out"] = max_exp_avg_sq_ptr
    return result
