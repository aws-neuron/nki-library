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
Dynamic expert selection on 3D tensors via scalar_offset indirect
indexing -- the MoE pattern. weights[E, P, F] indexed by a runtime
expert id produces a (P, F) view of the chosen expert.
"""

import nki
import nki.language as nl
import torch

from nkilib_src.nkilib.experimental import neurotile as nt

# ============================================================================
# Kernels
# ============================================================================


@nki.jit
def expert_select(weights, expert_id_tensor):
    """Select one expert dynamically. weights[E, P, F] -> output[P, F]."""
    # weights: [E, P, F],  expert_id_tensor: [1, 1]  ->  output: [P, F]
    E = weights.shape[0]
    P = weights.shape[1]
    F = weights.shape[2]
    out = nl.ndarray((P, F), dtype=weights.dtype, buffer=nl.shared_hbm)

    # 2D tile on 3D tensor: dim 0 (expert) is iterated/indexed.
    w_iter = nt.tiles(weights, tile_size=(P, F))
    eid_iter = nt.tiles(expert_id_tensor, tile_size=(1, 1))
    eid_tile = eid_iter[0, 0].load()  # tile: [1, 1]

    expert_data = w_iter[eid_tile].load()  # tile: [P, F]

    out_iter = nt.tiles(out, tile_size=(P, F))
    out_iter[0, 0].store(expert_data.data)
    return out


# ============================================================================
# Helpers
# ============================================================================


def to_device(t):
    import torch_xla.core.xla_model as xm

    return t.to(xm.xla_device())


def to_cpu(t):
    return t.cpu() if isinstance(t, torch.Tensor) else t


# ============================================================================
# Tests
# ============================================================================


def test_single_expert():
    torch.manual_seed(50)
    E, P, F = 8, 128, 64
    weights = torch.randn(E, P, F, dtype=torch.bfloat16)
    eid = torch.tensor([[5]], dtype=torch.int32)
    result = expert_select(to_device(weights), to_device(eid))
    expected = weights[5, :, :]
    torch.testing.assert_close(to_cpu(result).to(torch.float32), expected.to(torch.float32))
    print("expert_select: PASSED")


def main():
    test_single_expert()


if __name__ == "__main__":
    main()
