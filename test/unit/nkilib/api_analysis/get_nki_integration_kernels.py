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

"""Identify framework-ready kernels (proven HBM-safe)."""

from .get_nki_functions import KernelLocation
from .get_nki_kernels import get_nki_kernels_split
from .get_nki_wrapped_kernels import get_nki_wrapped_kernels


def get_nki_framework_kernels(proven_names: set[str]) -> list[KernelLocation]:
    """Return public kernels that are proven HBM-safe (framework-ready).

    Excludes wrapped kernels and subkernels, which are not directly
    invocable as top-level framework kernels.
    """
    public, _, _ = get_nki_kernels_split()
    wrapped_names: set[str] = {loc.name for loc in get_nki_wrapped_kernels().keys()}

    return [
        loc
        for loc in public
        if loc.name in proven_names and loc.name not in wrapped_names and '/subkernels/' not in loc.filepath
    ]
