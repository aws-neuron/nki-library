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

"""PyTorch reference for the chunked (wide-vocab) top-k kernel.

DELIBERATELY STRUCTURE-FREE. This does a single ``torch.topk`` over the WHOLE vocab and
knows nothing about chunks, snakes or the 65,535 ceiling. That is the point: the kernel's
entire claim is that splitting the vocab into P chunks and reducing is equivalent to one
global top-k, so the oracle must not reproduce the split. An oracle that chunked the same
way would agree with a wrong chunk offset.

``chunked_topk_ref_chunkwise`` below DOES reproduce the structure -- it exists only so the
test suite can prove its own gates fire when the chunk offset is injected wrong, and it is
never used as the golden.
"""

import torch


def chunked_topk_torch_ref(inp: torch.Tensor, config) -> dict[str, torch.Tensor]:
    """Global top-k over the last dimension. The golden for chunked_topk.

    Args:
        inp: ``[BxS, vocab]`` tensor (already in the kernel's dtype) -- or, when
            ``config.interleaved_peer_width`` is set, the all-to-all RECEIVE layout
            ``[BxS*(vocab/w), w]`` (row ``p*BxS + j`` holds token ``j``'s columns
            ``[p*w, (p+1)*w)``), which is de-interleaved here first: the config field
            DEFINES the input layout, so the reference honors it the same way the kernel does.
        config: ChunkedTopkConfig, read for ``.topk_config.k`` / ``.topk_config.sorted``.

    Returns:
        dict with ``topk_values [BxS, k]`` and ``topk_indices [BxS, k]`` (int32).
    """
    k = config.topk_config.k
    sorted_output = config.topk_config.sorted
    w = getattr(config, "interleaved_peer_width", 0)
    if w:
        bxs, vocab = config.BxS, config.vocab_size
        inp = inp.reshape(vocab // w, bxs, w).permute(1, 0, 2).reshape(bxs, vocab)
    values, indices = torch.topk(inp, k=k, dim=-1, largest=True, sorted=sorted_output)
    return {"topk_values": values, "topk_indices": indices.to(torch.int32)}


def chunked_topk_ref_chunkwise(inp, k: int, chunks: int, bug: str = "none"):
    """Numpy model of the kernel's OWN two-stage structure. Test-only.

    Mirrors chunked_topk: split the vocab into ``chunks`` contiguous pieces, take a top-k of
    each, map chunk-local indices to global ones, then reduce the ``chunks*k`` candidates.

    ``bug`` injects a specific index-mapping error so the suite can prove its gates are able
    to fail. A gate that cannot fail is not a gate, and the dangerous property of every bug
    below is that the returned VALUES stay correct -- only the indices lie:

        none         correct mapping
        no_offset    chunk-local indices returned as if global (offset dropped entirely)
        off_by_one   offset uses (c+1) instead of c
        reversed     offset uses (P-1-c) instead of c -- chunk order flipped
        scrambled    a wrong blocked-inverse INSIDE a chunk: (i % n_cols)*16 + i // n_cols
        wrong_row    row j's candidates taken from row j+1 (a token-major/chunk-major mixup)

    Returns ``(values [BxS, k], indices [BxS, k])`` as numpy arrays.
    """
    import numpy as np

    x = np.asarray(inp, dtype=np.float32)
    bxs, vocab = x.shape
    width = vocab // chunks
    n_cols = max(1, width // 16)
    cand_v = np.empty((bxs, chunks * k), dtype=np.float32)
    cand_i = np.empty((bxs, chunks * k), dtype=np.int64)
    for c in range(chunks):
        src_rows = np.roll(np.arange(bxs), -1) if bug == "wrong_row" else np.arange(bxs)
        block = x[:, c * width : (c + 1) * width]
        order = np.argsort(-block[src_rows], axis=-1, kind="stable")[:, :k]
        cand_v[:, c * k : (c + 1) * k] = np.take_along_axis(block[src_rows], order, axis=-1)
        local = order
        if bug == "scrambled":
            local = (order % n_cols) * 16 + order // n_cols
            local = np.clip(local, 0, width - 1)
        base = {
            "none": c,
            "scrambled": c,
            "wrong_row": c,
            "no_offset": 0,
            "off_by_one": c + 1,
            "reversed": chunks - 1 - c,
        }[bug]
        cand_i[:, c * k : (c + 1) * k] = local + base * width
    order2 = np.argsort(-cand_v, axis=-1, kind="stable")[:, :k]
    return (
        np.take_along_axis(cand_v, order2, axis=-1),
        np.take_along_axis(cand_i, order2, axis=-1),
    )


#: Every injectable mapping bug, for the suite's negative controls.
MAPPING_BUGS = ("no_offset", "off_by_one", "reversed", "scrambled", "wrong_row")
