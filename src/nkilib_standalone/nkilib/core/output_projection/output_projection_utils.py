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


def calculate_head_packing(N, D, partition_size, quant_config=None):
    """
    Optimize contraction dimension by folding N into D when D < partition_size.

    Args:
        N (int): Number of heads
        D (int): Head dimension size
        partition_size (int): Hardware constraint

    Returns:
        tuple: (new_N, new_D, group_size)
    """
    if D >= partition_size:
        N, D = calculate_double_row_head_packing(quant_config, N, D)
        return N, D, 1

    # Find largest divisor of N such that (group_size * D) <= partition_size
    group_size = N
    while (N % group_size) or (group_size * D) > partition_size:
        group_size -= 1
    assert group_size > 0, f"group_size should be greater than or equal to 1, but got {group_size}."

    new_N = N // group_size
    new_D = D * group_size
    new_N, new_D = calculate_double_row_head_packing(quant_config, new_N, new_D)

    return new_N, new_D, group_size


def calculate_double_row_head_packing(quant_config, N, D):
    """
    Manipulate num_heads dimension by folding D into N when using double row matmul and N is odd

    Args:
        quant_config: Quantization Config
        N (int): Number of heads
        D (int): Head dimension size

    Returns:
        tuple: (new_N, new_D)
    """
    if quant_config and quant_config.use_double_row:
        if N % 2 != 0:
            if D % 2 == 0:
                # Pack one more level: split D to increase N
                N = N * 2
                D = D // 2
            else:
                quant_config.use_double_row = False
    return N, D
