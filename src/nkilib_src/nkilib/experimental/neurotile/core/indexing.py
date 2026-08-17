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

import nki.language as nl


def is_scalar_shape(shape):
    """Return True when every tensor dimension is a singleton."""
    if len(shape) < 1:
        return False
    for dim_extent in shape:
        if dim_extent != 1:
            return False
    return True


def is_sbuf_scalar_value(value):
    """Return True for scalar-shaped values that can feed SBUF scalar ops."""
    if not hasattr(value, "shape"):
        return False
    if not is_scalar_shape(tuple(value.shape)):
        return False
    if hasattr(value, "buffer"):
        return value.buffer == nl.sbuf
    return True


def assert_valid_element_offset_value(value):
    """Validate a normalized ElementOffset payload."""
    assert not isinstance(value, bool), (
        "nt.element_offset(value): bool offsets are not supported; use an int offset explicitly."
    )
    if isinstance(value, int):
        assert value >= 0, "nt.element_offset(value): static offsets must be non-negative; got " + str(value)
        return
    assert not isinstance(value, (list, tuple, dict, str, float)), (
        "nt.element_offset(value): unsupported offset type "
        + str(value)
        + ". Use a non-negative int, SBUF scalar tensor, SBUF scalar view, or NKI runtime scalar expression."
    )
    if hasattr(value, "shape"):
        assert is_sbuf_scalar_value(value), (
            "nt.element_offset(value): tensor value must be an SBUF scalar; got shape="
            + str(tuple(value.shape))
            + ", buffer="
            + str(value.buffer if hasattr(value, "buffer") else "<unknown>")
        )
        return
    assert hasattr(value, "dtype"), (
        "nt.element_offset(value): unsupported offset value "
        + str(value)
        + ". Use a non-negative int, SBUF scalar tensor, SBUF scalar view, or NKI runtime scalar expression with dtype metadata."
    )


class ElementOffset(nl.NKIObject):
    """Index key for a source-element offset relative to the current view."""

    def __init__(self, value):
        self.value = value


def element_offset(value):
    """element_offset(value) -> ElementOffset

    Mark ``value`` as a source-element offset for bracket indexing.

    Use this when a runtime counter is already maintained in element units
    relative to the current view, so NeuroTile should pass it through as an
    indirect scalar offset rather than treating it as a tile/block coordinate.

    .. warning::

       This API is experimental and may change in future releases.

    Args:
        value: Compile-time int, SBUF scalar, or scalar view containing a
            source-element offset relative to the current view dimension.

    Returns:
        ElementOffset: A marker object accepted by ``NDSlice.__getitem__``.
    """
    return ElementOffset(value)


__all__ = [
    "ElementOffset",
    "assert_valid_element_offset_value",
    "element_offset",
    "is_scalar_shape",
    "is_sbuf_scalar_value",
]
