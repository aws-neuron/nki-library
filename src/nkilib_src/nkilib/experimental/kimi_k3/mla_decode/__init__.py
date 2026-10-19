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

"""Kimi-K3 dense MLA decode kernel."""

from .mla_decode import kimi_k3_mla_decode
from .mla_decode_torch import kimi_k3_mla_decode_torch_ref

__all__ = ["kimi_k3_mla_decode", "kimi_k3_mla_decode_torch_ref"]
