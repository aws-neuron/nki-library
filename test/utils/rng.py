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


# Native torch RNG was consuming a significant amount of wall time in compile-only tests.
# NKITestsRNG repeats small true-randomness RNG blocks to produce large tensors of data,
# yielding major speedups (avg of 55.3% of wall time in measured tests).
class NKITestsRNG:
    # Prime block size: gcd(BLOCK_SIZE, N) == 1 for any layout width N < BLOCK_SIZE, so tiled
    # rows do not repeat and matmul/MLP/MoE golden inputs keep full rank (a 2^k size aliases to rank 1-2).
    # Rank saturates at BLOCK_SIZE; a dim that is an exact multiple of BLOCK_SIZE aliases back to rank 1.
    BLOCK_SIZE = 1031
    _KNUTH_MULTIPLIER = 2654435761

    def __init__(self, seed=42):
        self._seed = seed
        self.reset()

    def reset(self, seed=None):
        # Restore the instance to its initial seeded state. Call from a per-test
        # fixture so parallel/worksteal runs are deterministic regardless of how
        # many draws ran in earlier tests sharing this instance. Pass ``seed`` to
        # re-seed to a new value (mirrors the old torch.manual_seed(seed) reset).
        if seed is not None:
            self._seed = seed
        self._blocks = {}
        self._counter = 0
        # randperm cannot be tiled; a persistent generator (seeded once, then
        # advanced) yields distinct permutations across successive calls.
        self._perm_gen = torch.Generator()
        self._perm_gen.manual_seed(self._seed)

    def _offset(self):
        self._counter += 1
        return (self._counter * self._KNUTH_MULTIPLIER) % self.BLOCK_SIZE

    def _get_block(self, key, gen_fn):
        if key not in self._blocks:
            self._blocks[key] = gen_fn()
        return self._blocks[key]

    def _generator(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        return g

    def _tile(self, block, shape, dtype=None):
        numel = 1
        for s in shape:
            numel *= s
        offset = self._offset()
        block_len = block.numel()
        repeats = (numel + offset + block_len - 1) // block_len + 1
        out = block.repeat(repeats)[offset : offset + numel].reshape(shape)
        if dtype is not None and out.dtype != dtype:
            out = out.to(dtype)
        return out

    def _tile_into(self, tensor, block):
        numel = tensor.numel()
        offset = self._offset()
        block_len = block.numel()
        repeats = (numel + offset + block_len - 1) // block_len + 1
        tensor.data.copy_(block.repeat(repeats)[offset : offset + numel].reshape(tensor.shape))
        return tensor

    def randn(self, *shape, dtype=None):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        block = self._get_block(("randn",), lambda: torch.randn(self.BLOCK_SIZE, generator=self._generator()))
        return self._tile(block, shape, dtype=dtype)

    def randint(self, low, high, size, dtype=None):
        block = self._get_block(
            ("randint", low, high),
            lambda: torch.randint(low, high, (self.BLOCK_SIZE,), generator=self._generator()),
        )
        return self._tile(block, size, dtype=dtype)

    def randperm(self, n):
        # Tiling cannot produce valid permutations; use the persistent generator so
        # repeated calls advance RNG state and produce distinct permutations.
        return torch.randperm(n, generator=self._perm_gen)

    def kaiming_normal_(self, tensor, mode="fan_in", nonlinearity="leaky_relu"):
        # Compute std directly: std = gain / sqrt(fan)
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
        fan = fan_in if mode == "fan_in" else fan_out
        gain = torch.nn.init.calculate_gain(nonlinearity)
        std = gain / math.sqrt(fan)
        block = self._get_block(
            ("kaiming_normal", fan, nonlinearity),
            lambda: torch.empty(self.BLOCK_SIZE).normal_(0.0, std, generator=self._generator()),
        )
        return self._tile_into(tensor, block)

    def kaiming_uniform_(self, tensor, mode="fan_in", nonlinearity="leaky_relu"):
        # Compute bound directly: bound = gain * sqrt(3 / fan)
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
        fan = fan_in if mode == "fan_in" else fan_out
        gain = torch.nn.init.calculate_gain(nonlinearity)
        bound = gain * math.sqrt(3.0 / fan)
        block = self._get_block(
            ("kaiming_uniform", fan, nonlinearity),
            lambda: torch.empty(self.BLOCK_SIZE).uniform_(-bound, bound, generator=self._generator()),
        )
        return self._tile_into(tensor, block)

    def xavier_normal_(self, tensor, gain=1.0):
        # Compute std directly from fan values to avoid proxy-shape mismatch.
        # Xavier normal: std = gain * sqrt(2.0 / (fan_in + fan_out))
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        block = self._get_block(
            ("xavier_normal", fan_in, fan_out, gain),
            lambda: torch.empty(self.BLOCK_SIZE).normal_(0.0, std, generator=self._generator()),
        )
        return self._tile_into(tensor, block)

    def xavier_uniform_(self, tensor, gain=1.0):
        # Xavier uniform: bound = gain * sqrt(6.0 / (fan_in + fan_out))
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
        bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
        block = self._get_block(
            ("xavier_uniform", fan_in, fan_out, gain),
            lambda: torch.empty(self.BLOCK_SIZE).uniform_(-bound, bound, generator=self._generator()),
        )
        return self._tile_into(tensor, block)

    def normal_(self, tensor, mean=0.0, std=1.0):
        block = self._get_block(
            ("normal", mean, std),
            lambda: torch.empty(self.BLOCK_SIZE).normal_(mean, std, generator=self._generator()),
        )
        return self._tile_into(tensor, block)

    def uniform_(self, tensor, a=0.0, b=1.0):
        block = self._get_block(
            ("uniform", a, b),
            lambda: torch.empty(self.BLOCK_SIZE).uniform_(a, b, generator=self._generator()),
        )
        return self._tile_into(tensor, block)

    def tensor_uniform_(self, tensor, low=0.0, high=1.0):
        return self.uniform_(tensor, low, high)

    def exponential_sample(self, rate, shape):
        # Exp(rate) via the instance generator (distributions.sample() ignores it).
        block = self._get_block(
            ("exponential", rate),
            lambda: torch.empty(self.BLOCK_SIZE).exponential_(rate, generator=self._generator()),
        )
        return self._tile(block, shape)

    def lognormal_sample(self, loc, scale, shape):
        # LogNormal(loc, scale) == exp(Normal(loc, scale)), instance-seeded.
        block = self._get_block(
            ("lognormal", loc, scale),
            lambda: torch.empty(self.BLOCK_SIZE).normal_(loc, scale, generator=self._generator()).exp_(),
        )
        return self._tile(block, shape)
