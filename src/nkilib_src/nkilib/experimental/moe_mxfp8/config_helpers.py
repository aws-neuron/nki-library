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

"""Public MXFP8 MoE/MLP execution-mode enum."""

from enum import Enum


class ExecutionMode(Enum):
    """Routing and affinity behavior for MXFP8 MoE and MLP."""

    MLP = "mlp"
    DENSE_MOE = "dense_moe"
    ROUTED_MOE = "routed_moe"

    @property
    def is_dense(self):
        return self != ExecutionMode.ROUTED_MOE
