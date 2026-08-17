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

"""RNG kernels for getting/setting GPSIMD RNG state and generating random numbers."""

from .rng import generate_random, generate_random_fast, get_rng_state_gpsimd, set_rng_state_gpsimd

__all__ = ["get_rng_state_gpsimd", "set_rng_state_gpsimd", "generate_random", "generate_random_fast"]
