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

"""Torch reference for runtime element-offset indexing examples."""


def dynamic_elementwise_add_torch_ref(input_a, input_b, num_m_tiles):
    del num_m_tiles
    return input_a + input_b


def dynamic_elementwise_add_logical_tile_index_torch_ref(input_a, input_b, num_m_tiles):
    del num_m_tiles
    return input_a + input_b
