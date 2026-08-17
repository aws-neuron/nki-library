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

"""LNC sharding logic for the pad kernel.

Decides how to distribute work across LNC cores and splits the input,
output, and padding parameters accordingly.
"""

from typing import Optional, Tuple

import nki.language as nl

from ...core.utils.kernel_helpers import div_ceil
from ...core.utils.logging import get_logger
from .pad_params import PadParams
from .pad_tiling import compute_tiling_strategy

_logger = get_logger("pad_kernel")

ShardResult = Tuple[Optional[nl.NkiTensor], Optional[nl.NkiTensor], Optional[PadParams]]


def _shard_utilization(n_tiles: int, num_cores: int) -> float:
    """Fraction of work slots utilized when distributing tiles across cores."""
    if n_tiles == 0:
        return 0.0
    slots = div_ceil(n_tiles, num_cores) * num_cores
    return n_tiles / slots


def _choose_shard_dim(x_4d: nl.NkiTensor, params: PadParams, mode: str, dtype) -> int:
    """Choose which dimension to shard across LNC cores.

    Returns spatial dim index (0=D, 1=H, 2=W) or -1 for NC.
    """
    num_cores = nl.num_programs()
    NC = x_4d.shape[0]
    TILE_P = nl.tile_size.pmax

    nc_util = _shard_utilization(div_ceil(NC, TILE_P), num_cores)
    if nc_util >= 0.9:
        return -1

    # Replicate and constant modes have no cross-tile data dependencies for
    # padding (replicate uses the tile edge, constant uses a scalar), so we
    # can safely split a spatial dimension across cores. Reflect and circular
    # may need data from the other core's region, so they must shard on NC.
    if mode not in ("replicate", "constant"):
        return -1

    # Check if any spatial dim gives better utilization than NC
    D, H, W = x_4d.shape[1], x_4d.shape[2], x_4d.shape[3]
    tile_strategy = compute_tiling_strategy(D, H, W, params, mode, dtype)

    best_dim = -1
    best_util = nc_util
    for dim in (0, 1, 2):
        util = _shard_utilization(tile_strategy.n_tiles[dim], num_cores)
        if util > best_util:
            best_util = util
            best_dim = dim

    return best_dim


def _split_nc(
    x_4d: nl.NkiTensor,
    out_4d: nl.NkiTensor,
    params: PadParams,
    core_id: int,
    num_cores: int,
) -> ShardResult:
    """Split NC dimension across cores."""
    NC = x_4d.shape[0]
    if core_id >= NC:
        return None, None, None  # idle core; output is filled by other cores
    nc_per_core = div_ceil(NC, num_cores)
    nc_start = core_id * nc_per_core
    nc_count = min(nc_per_core, NC - nc_start)
    return (x_4d.slice(0, nc_start, nc_start + nc_count), out_4d.slice(0, nc_start, nc_start + nc_count), params)


def _split_spatial(
    x_4d: nl.NkiTensor,
    out_4d: nl.NkiTensor,
    params: PadParams,
    dim: int,
    core_id: int,
    num_cores: int,
) -> ShardResult:
    """Split a spatial dimension across cores, adjusting output and params."""
    axis = dim + 1
    src_size = x_4d.shape[axis]
    if core_id >= src_size:
        return None, None, None  # idle core; output is filled by other cores
    per_core = div_ceil(src_size, num_cores)
    start = core_id * per_core
    count = min(per_core, src_size - start)

    x_shard = x_4d.slice(axis, start, start + count)

    pad_before = params.before(dim) if core_id == 0 else 0
    pad_after = params.after(dim) if start + count == src_size else 0
    out_start = start + params.before(dim) - pad_before
    out_shard = out_4d.slice(axis, out_start, out_start + pad_before + count + pad_after)

    new_pad = list(params.pad)
    new_pad[dim] = (pad_before, pad_after)
    shard_params = PadParams(pad=tuple(new_pad), mode=params.mode)

    return x_shard, out_shard, shard_params


def shard_operation(
    x_4d: nl.NkiTensor,
    out_4d: nl.NkiTensor,
    params: PadParams,
    mode: str,
    dtype,
) -> ShardResult:
    """Shard the pad operation across LNC cores.

    Chooses the dimension with best utilization: NC by default, or a spatial
    dim for replicate/constant when NC is underutilized.

    Returns:
        ``(x_shard, out_shard, params_shard)`` or ``(None, None, None)``
        if this core has no work.
    """
    num_cores = nl.num_programs()
    core_id = nl.program_id(axis=0)

    shard_dim = _choose_shard_dim(x_4d, params, mode, dtype)

    _dim_names = ["D", "H", "W"]
    name = _dim_names[shard_dim] if shard_dim >= 0 else "NC"
    _logger.info(f"Sharding: dim={name}, num_cores={num_cores}")

    if shard_dim == -1:
        return _split_nc(x_4d, out_4d, params, core_id, num_cores)

    return _split_spatial(x_4d, out_4d, params, shard_dim, core_id, num_cores)
