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

"""Helper for LNC sendrecv that automatically selects gpsimd_dma when beneficial."""

import nki.isa as nisa

# Hardware limit: max free-dim elements per single gpsimd_dma sendrecv
_GPSIMD_HW_LIMIT = 256


def lnc_sendrecv(src, dst, send_to_rank: int, recv_from_rank: int, pipe_id: int = 0, allow_gpsimd: bool = True):
    """LNC sendrecv that automatically uses gpsimd_dma when beneficial.

    gpsimd_dma routes SBUF-to-SBUF directly without HBM, eliminating
    PSEUDO_CORE_BARRIER stalls. It has a hardware limit of 256 elements
    per partition in the free dimension. For transfers exceeding this
    limit, we fall back to regular DMA.

    Args:
        src: Source SBUF tensor to send.
        dst: Destination SBUF tensor to receive into.
        send_to_rank: Rank to send data to.
        recv_from_rank: Rank to receive data from.
        pipe_id: DMA pipe identifier. Defaults to 0.
        allow_gpsimd: Whether to use gpsimd_dma for small transfers. Defaults to True.
            Set to False when calling from contexts where gpsimd_dma is not supported.
    """
    free_elems = 1
    for dim_size in src.shape[1:]:
        free_elems *= dim_size

    if allow_gpsimd and free_elems <= _GPSIMD_HW_LIMIT:
        nisa.sendrecv(
            src=src,
            dst=dst,
            send_to_rank=send_to_rank,
            recv_from_rank=recv_from_rank,
            pipe_id=pipe_id,
            dma_engine=nisa.dma_engine.gpsimd_dma,
        )
    else:
        nisa.sendrecv(
            src=src,
            dst=dst,
            send_to_rank=send_to_rank,
            recv_from_rank=recv_from_rank,
            pipe_id=pipe_id,
        )
