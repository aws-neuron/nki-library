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

"""CPU reference for the CSA tensor-parallel output all-reduce.

The reduction itself is an ordinary ``dist.all_reduce`` sum. What the kernel adds
around it -- staging the input into a NAMED ``shared_hbm`` buffer, launching on the
``[2]`` grid so the ``lnc=2`` lowering distributes one whole-tensor collective
across the rank's two logical cores, and copying the result back out -- is exactly
what the test is for, since each of those is a way to get a silently HALF-reduced
answer rather than an error.

Uses the same ``get_pg`` adapter as the other collective references, so it works
under both the simulated and the real distributed runner.
"""

import numpy as np
import torch
import torch.distributed as dist
from nki.collectives import ReplicaGroup

from ..collectives.distributed_adapter import get_pg


def nki_tp_all_reduce_torch_ref(input: np.ndarray, replica_group: ReplicaGroup) -> dict:
    """Sum ``input`` elementwise across the ranks of ``replica_group``.

    An all-reduce sum is elementwise and therefore layout-invariant, which is why
    the kernel is free to reshape a ``[B, S, dim]`` partial into a balanced ``[P, F]``
    tile before reducing -- the reshape cannot change the result, and it keeps both
    logical cores occupied where a single-partition input would leave one empty.
    """
    dtype = input.dtype
    try:
        tensor = torch.from_numpy(input.copy())
    except TypeError:
        tensor = torch.from_numpy(input.astype(np.float32))

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=get_pg(replica_group))

    out = tensor.numpy()
    if out.dtype != dtype:
        out = out.astype(dtype)
    return {"out": out}
