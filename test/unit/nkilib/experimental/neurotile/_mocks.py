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

"""Shared tensor mocks for neurotile unit tests.

The real ``nl.ndarray`` is a single type that the layout / factory code
duck-types on across every memory space (HBM, SBUF, PSUM). The tests only
need two stand-ins that mirror that contract:

  - :class:`MockTensor` -- any tensor / ndarray source. Covers the identity
    HBM tensor (the common case), a sliced view (``pattern=`` + ``offset=``
    so ``get_pattern`` reports parent strides), an on-chip buffer
    (``buffer=nl.sbuf`` / ``nl.psum`` so ``tensor_view`` auto-detects the
    space), and a sub-width view of a wider allocation (``storage_shape=`` so
    ``_partition_row_stride`` reads the physical row width). It is
    subscriptable (``[:, a:b]`` narrows the free dim, preserving
    ``_storage_shape``) and reshapeable, which is what the SBUF / PSUM AP
    paths exercise.

  - :class:`MockApView` -- the value returned by ``MockTensor.ap(...)``.
    Exposes the logical AP ``.shape`` / ``.strides`` (per-level counts /
    strides) for on-chip AP assertions, and is also dict-indexable
    (``view["pattern"]``, ``"scalar_offset" in view``) for the HBM
    ``_apply_ap`` indirect-dispatch assertions.

One source class + one result class -- update them here when the layout
contract changes, instead of re-deriving an ndarray stand-in per test file.
"""


def _contiguous_pattern(shape):
    """Contiguous ``[[stride, count], ...]`` AP pattern for ``shape``."""
    pattern = []
    stride = 1
    for d in range(len(shape) - 1, -1, -1):
        pattern.append([stride, shape[d]])
        stride = stride * shape[d]
    pattern.reverse()
    return pattern


class MockApView:
    """Result of ``MockTensor.ap(pattern=..., offset=..., **indirect)``.

    Attribute access reports the logical AP: ``.shape`` is the per-level
    counts, ``.strides`` the per-level strides (so a test can assert the
    emitted partition stride / shape). Dict access exposes the raw call --
    ``view["pattern"]``, ``view["offset"]``, ``view["scalar_offset"]``,
    ``"vector_offset" in view`` -- for the HBM indirect-dispatch tests that
    assert on the keyword arguments rather than the strided view.
    """

    def __init__(self, pattern, offset=0, **indirect):
        self.pattern = pattern
        self.offset = offset
        self.shape = tuple(count for _stride, count in pattern)
        self.strides = tuple(stride for stride, _count in pattern)
        self._call = {"pattern": pattern, "offset": offset}
        self._call.update(indirect)

    def __getitem__(self, key):
        return self._call[key]

    def __contains__(self, key):
        return key in self._call


class MockTensor:
    """Stand-in for any ``nl.ndarray`` source the layout / factory code reads.

    Args:
        *shape: dims, as ``MockTensor(128, 512)`` or ``MockTensor((128, 512))``.
        dtype: element dtype string.
        buffer: an ``nl.MemoryRegion`` (``nl.sbuf`` / ``nl.psum``) to model an
            on-chip tensor whose space is auto-detected; omitted -> no
            ``.buffer``, which defaults to HBM.
        offset: storage element offset (a sliced source carries one).
        pattern: physical ``[[stride, count], ...]`` for a sliced view; ``None``
            (default) reports the contiguous pattern of ``shape``.
        storage_shape: physical buffer shape when this is a sub-width view of a
            wider allocation; defaults to ``shape``. ``_partition_row_stride``
            reads it, so a narrowed slice still reports the bank's true row width.
        indirect: model a handle that already carries a runtime (gather /
            dynamic-select) offset -- ``is_indirect()`` returns True, mirroring a
            real NkiTensor produced by ``select`` / ``vector_select`` / an
            indirect ``ap()``.
    """

    def __init__(
        self, *shape, dtype="float32", buffer=None, offset=0, pattern=None, storage_shape=None, indirect=False
    ):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        self.shape = tuple(shape)
        self.dtype = dtype
        self.offset = offset
        self._pattern = pattern
        self._storage_shape = tuple(storage_shape) if storage_shape is not None else self.shape
        self._indirect = indirect
        if buffer is not None:
            self.buffer = buffer

    def get_pattern(self):
        """Physical strides: explicit ``pattern`` for a slice, else contiguous."""
        if self._pattern is not None:
            return self._pattern
        return _contiguous_pattern(self.shape)

    def is_indirect(self):
        """True if this handle already carries a runtime (gather/select) offset."""
        return self._indirect

    def is_contiguous(self):
        """True if get_pattern() matches the contiguous pattern of ``shape``."""
        return self.get_pattern() == _contiguous_pattern(self.shape)

    def reshape(self, new_shape):
        # _as_2d may flatten an N-D buffer to 2D; the physical row width is
        # unchanged, so the reshaped view keeps _storage_shape.
        return MockTensor(tuple(new_shape), dtype=self.dtype, storage_shape=self._storage_shape)

    def __getitem__(self, key):
        # NKI ``[:, a:b]`` slice: narrow the free dim, accumulate the F-offset
        # into ``offset`` (a nonzero start advances the view's base, like the
        # real NkiTensor), and preserve the underlying _storage_shape (what
        # _partition_row_stride reads). The F key is a plain slice or an nl.ds
        # DynamicSlice (which carries .size).
        f_key = key[1] if isinstance(key, tuple) and len(key) > 1 else slice(None)
        if hasattr(f_key, "size"):  # nl.ds DynamicSlice
            new_f = f_key.size
            f_start = f_key.start
        elif isinstance(f_key, slice) and f_key.start is not None and f_key.stop is not None:
            f_start = f_key.start or 0
            new_f = f_key.stop - f_start
        else:
            new_f = self.shape[-1]
            f_start = 0
        return MockTensor(
            (self.shape[0], new_f),
            dtype=self.dtype,
            offset=self.offset + f_start,
            storage_shape=self._storage_shape,
        )

    def ap(self, pattern, offset=0, **indirect):
        return MockApView(pattern, offset, **indirect)
