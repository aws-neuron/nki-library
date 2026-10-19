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
"""Independent stable-sort reference for packed routing-vector construction."""

import torch


def build_packed_routing_torch_ref(ids, counts):
    if not torch.equal(torch.bincount(ids.long(), minlength=counts.numel()), counts.flatten().long()):
        raise ValueError("counts must equal the histogram of peer IDs")
    return {"source_indices": torch.argsort(ids, stable=True).to(torch.int32)}
