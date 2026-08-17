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

"""GPT-OSS MXFP4 decode-layer kernel (experimental, under development).

This package hosts the single GPT-OSS decoder-block token-generation kernel:
RMSNorm -> GQA attention (per-head sinks + optional sliding window + YaRN RoPE,
paged KV cache) -> residual -> RMSNorm -> MXFP4 MoE (top-k softmax router +
SwiGLU experts) -> residual.

``gpt_oss_mxfp4_decode_layer`` is currently a **placeholder stub** that returns
a correct-shape output so the test harness, tensor contract, and reference
oracle can be developed ahead of the real kernel. The reference oracle is the
standalone pure-torch golden in ``private-vllm-neuron/gptoss_mxfp4_golden``.
"""

from .gpt_oss_mxfp4_decode_layer import gpt_oss_mxfp4_decode_layer

__all__ = ["gpt_oss_mxfp4_decode_layer"]
