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
Helper functions for output projection kernels.

"""

from ..utils.kernel_assert import kernel_assert


def calculate_head_packing(N, D, partition_size):
    """
    Optimize contraction dimension by folding N into D when D < partition_size.

    Args:
        N (int): Number of heads
        D (int): Head dimension size
        partition_size (int): Hardware constraint

    Returns:
        tuple: (new_N, new_D, group_size)
    """

    # Find largest divisor of N such that (group_size * D) <= partition_size
    group_size = N
    while (N % group_size) or (group_size * D) > partition_size:
        group_size -= 1
    kernel_assert(group_size > 0, f"group_size should be greater than or equal to 1, but got {group_size}.")

    new_N = N // group_size
    new_D = D * group_size
    return new_N, new_D, group_size
