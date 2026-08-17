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

"""Padding mode callables.

Each mode class implements three methods:

- ``fill(dst, interior, dim, pad_idx)``: fill a single padded slice in SBUF.
- ``fill_deferred(dst, src)``: fill a deferred padding slice via DMA.
- ``map_idx(dim, pad_idx)``: map a padding index to a source index.

All tensor arguments are NkiTensors.
"""

from typing import Union

import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert

# Type alias for any pad mode class
PadMode = Union["ConstantPad", "ReplicatePad", "ReflectPad", "CircularPad"]


class ConstantPad(nl.NKIObject):
    """Constant padding: fill with a fixed value (default 0)."""

    def __init__(self, hbm_src: nl.NkiTensor, value: float = 0):
        self._hbm = hbm_src
        self._value = value

    def map_idx(self, dim: int, pad_idx: int) -> int:
        return 0

    def fill(self, dst: nl.NkiTensor, interior: nl.NkiTensor, dim: int, pad_idx: int) -> None:
        nisa.memset(dst, self._value)

    def fill_deferred(self, dst: nl.NkiTensor, src: nl.NkiTensor) -> None:
        """For constant mode, fill via SBUF memset + DMA store."""
        tmp = nl.ndarray(dst.shape, dtype=dst.dtype, buffer=nl.sbuf)
        nisa.memset(tmp, self._value)
        nisa.dma_copy(dst=dst, src=tmp)


class ReplicatePad(nl.NKIObject):
    """Replicate padding: clamp to the nearest edge element."""

    def __init__(self, hbm_src: nl.NkiTensor):
        self._hbm = hbm_src

    def map_idx(self, dim: int, pad_idx: int) -> int:
        """Clamp to nearest edge."""
        return 0 if pad_idx < 0 else self._hbm.shape[dim + 1] - 1

    def fill(self, dst: nl.NkiTensor, interior: nl.NkiTensor, dim: int, pad_idx: int) -> None:
        axis = dim + 1
        src_idx = 0 if pad_idx < 0 else interior.shape[axis] - 1
        nisa.tensor_copy(dst=dst, src=interior.select(axis, src_idx))

    def fill_deferred(self, dst: nl.NkiTensor, src: nl.NkiTensor) -> None:
        nisa.dma_copy(dst=dst, src=src)


class ReflectPad(nl.NKIObject):
    """Reflect padding: bounce at boundaries (PyTorch ``reflect`` semantics)."""

    def __init__(self, hbm_src: nl.NkiTensor):
        self._hbm = hbm_src

    def map_idx(self, dim: int, pad_idx: int) -> int:
        """Before: -1→1, -2→2, ... After: 0→size-2, 1→size-3, ..."""
        src_size = self._hbm.shape[dim + 1]
        return -pad_idx if pad_idx < 0 else src_size - 2 - pad_idx

    def fill(self, dst: nl.NkiTensor, interior: nl.NkiTensor, dim: int, pad_idx: int) -> None:
        axis = dim + 1
        nisa.tensor_copy(dst=dst, src=interior.select(axis, self.map_idx(dim, pad_idx)))

    def fill_deferred(self, dst: nl.NkiTensor, src: nl.NkiTensor) -> None:
        nisa.dma_copy(dst=dst, src=src)


class CircularPad(nl.NKIObject):
    """Circular padding: wrap around (PyTorch ``circular`` semantics)."""

    def __init__(self, hbm_src: nl.NkiTensor):
        self._hbm = hbm_src

    def map_idx(self, dim: int, pad_idx: int) -> int:
        """Before: -1→size-1, -2→size-2, ... After: 0→0, 1→1, ..."""
        src_size = self._hbm.shape[dim + 1]
        return (src_size + pad_idx) % src_size if pad_idx < 0 else pad_idx % src_size

    def fill(self, dst: nl.NkiTensor, interior: nl.NkiTensor, dim: int, pad_idx: int) -> None:
        axis = dim + 1
        nisa.tensor_copy(dst=dst, src=interior.select(axis, self.map_idx(dim, pad_idx)))

    def fill_deferred(self, dst: nl.NkiTensor, src: nl.NkiTensor) -> None:
        nisa.dma_copy(dst=dst, src=src)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_pad_mode(mode_str: str, hbm_src: nl.NkiTensor, value: float = 0) -> PadMode:
    """Create a PadMode instance for the given mode string."""
    if mode_str == "constant":
        return ConstantPad(hbm_src, value=value)
    elif mode_str == "replicate":
        return ReplicatePad(hbm_src)
    elif mode_str == "reflect":
        return ReflectPad(hbm_src)
    elif mode_str == "circular":
        return CircularPad(hbm_src)
    else:
        kernel_assert(False, "Unknown pad mode: " + mode_str)
