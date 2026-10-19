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

"""Shared PyTorch Adam reference calculations."""

import math

import torch


def _recover_hyperparams(step_size_ptr, inv_bc2_sqrt_ptr, wd_factor_ptr, beta1, beta2, decoupled_wd):
    """Recover (lr, weight_decay, step) from pre-computed scalars."""
    ss = float(step_size_ptr.reshape(-1)[0])
    ibc = float(inv_bc2_sqrt_ptr.reshape(-1)[0])
    wd = float(wd_factor_ptr.reshape(-1)[0])

    step = round(math.log(1.0 - 1.0 / (ibc**2)) / math.log(beta2))
    lr = ss * (1.0 - beta1**step)
    weight_decay = (1.0 - wd) / lr if decoupled_wd else wd
    return lr, weight_decay, step


def _adam_ref(
    param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, lr, beta1, beta2, eps, weight_decay, step, amsgrad, decoupled_wd
):
    """Call torch.optim._functional.adam/adamw."""
    from torch.optim._functional import adam, adamw

    fn = adamw if decoupled_wd else adam
    fn(
        [param],
        [grad],
        [exp_avg],
        [exp_avg_sq],
        [max_exp_avg_sq] if amsgrad else [],
        # PyTorch increments the counter before applying this step's bias correction.
        [torch.tensor(float(step - 1))],
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=False,
    )
