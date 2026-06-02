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

import math

import torch
from torch.distributions import Exponential, LogNormal


class NKITestsPseudoRNG:
    K = 1024

    def __init__(self, seed=42):
        self._seed = seed
        self._blocks = {}
        self._counter = 0

    def _offset(self):
        self._counter += 1
        return (self._counter * 2654435761) % self.K

    def _block(self, key, gen_fn):
        if key not in self._blocks:
            torch.manual_seed(self._seed)
            self._blocks[key] = gen_fn()
        return self._blocks[key]

    def _sample(self, block, shape, dtype=None):
        numel = 1
        for s in shape:
            numel *= s
        offset = self._offset()
        repeats = (numel + offset + self.K - 1) // self.K + 1
        out = block.repeat(repeats)[offset:offset + numel].reshape(shape)
        if dtype is not None and out.dtype != dtype:
            out = out.to(dtype)
        return out

    def randn(self, *shape, dtype=None):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        block = self._block(("randn",), lambda: torch.randn(self.K))
        return self._sample(block, shape, dtype=dtype)

    def randint(self, low, high, size, dtype=None):
        block = self._block(("randint", low, high), lambda: torch.randint(low, high, (self.K,)))
        out = self._sample(block, size)
        if dtype is not None:
            out = out.to(dtype)
        return out

    def randperm(self, n):
        return torch.randperm(n)

    def kaiming_normal_(self, tensor, mode="fan_in", nonlinearity="leaky_relu"):
        fan = _calculate_fan(tensor, mode)
        gain = _calculate_gain(nonlinearity)
        std = gain / math.sqrt(fan)
        block = self._block(("randn",), lambda: torch.randn(self.K))
        numel = tensor.numel()
        offset = self._offset()
        repeats = (numel + offset + self.K - 1) // self.K + 1
        tensor.data.copy_((block.repeat(repeats)[offset:offset + numel] * std).reshape(tensor.shape))
        return tensor

    def kaiming_uniform_(self, tensor, mode="fan_in", nonlinearity="leaky_relu"):
        fan = _calculate_fan(tensor, mode)
        gain = _calculate_gain(nonlinearity)
        bound = math.sqrt(3.0) * gain / math.sqrt(fan)
        block = self._block(("uniform_std",), lambda: torch.empty(self.K).uniform_(-1, 1))
        numel = tensor.numel()
        offset = self._offset()
        repeats = (numel + offset + self.K - 1) // self.K + 1
        tensor.data.copy_((block.repeat(repeats)[offset:offset + numel] * bound).reshape(tensor.shape))
        return tensor

    def xavier_normal_(self, tensor, gain=1.0):
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        block = self._block(("randn",), lambda: torch.randn(self.K))
        numel = tensor.numel()
        offset = self._offset()
        repeats = (numel + offset + self.K - 1) // self.K + 1
        tensor.data.copy_((block.repeat(repeats)[offset:offset + numel] * std).reshape(tensor.shape))
        return tensor

    def xavier_uniform_(self, tensor, gain=1.0):
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
        block = self._block(("uniform_std",), lambda: torch.empty(self.K).uniform_(-1, 1))
        numel = tensor.numel()
        offset = self._offset()
        repeats = (numel + offset + self.K - 1) // self.K + 1
        tensor.data.copy_((block.repeat(repeats)[offset:offset + numel] * bound).reshape(tensor.shape))
        return tensor

    def normal_(self, tensor, mean=0.0, std=1.0):
        block = self._block(("randn",), lambda: torch.randn(self.K))
        numel = tensor.numel()
        offset = self._offset()
        repeats = (numel + offset + self.K - 1) // self.K + 1
        tensor.data.copy_((block.repeat(repeats)[offset:offset + numel] * std + mean).reshape(tensor.shape))
        return tensor

    def uniform_(self, tensor, a=0.0, b=1.0):
        block = self._block(("uniform_01",), lambda: torch.empty(self.K).uniform_(0, 1))
        numel = tensor.numel()
        offset = self._offset()
        repeats = (numel + offset + self.K - 1) // self.K + 1
        tensor.data.copy_((block.repeat(repeats)[offset:offset + numel] * (b - a) + a).reshape(tensor.shape))
        return tensor

    def tensor_uniform_(self, tensor, low=0.0, high=1.0):
        return self.uniform_(tensor, low, high)

    def exponential_sample(self, rate, shape):
        block = self._block(("exp", rate), lambda: Exponential(rate).sample((self.K,)))
        return self._sample(block, shape)

    def lognormal_sample(self, loc, scale, shape):
        block = self._block(("lognormal", loc, scale), lambda: LogNormal(loc, scale).sample((self.K,)))
        return self._sample(block, shape)


def _calculate_fan_in_and_fan_out(tensor):
    dims = tensor.dim()
    if dims < 2:
        raise ValueError("Fan requires at least 2D tensor")
    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    for s in tensor.shape[2:]:
        receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_fan(tensor, mode):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        return fan_in
    elif mode == "fan_out":
        return fan_out
    raise ValueError(f"Invalid mode: {mode}")


def _calculate_gain(nonlinearity):
    if nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        return math.sqrt(2.0 / (1 + 0.01**2))
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "sigmoid":
        return 1.0
    return 1.0


