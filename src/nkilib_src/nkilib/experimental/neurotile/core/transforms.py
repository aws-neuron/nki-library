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
# ============================================================================
# Transforms -- return (new_element_shape, new_strides).
#
# Every metadata transform (reshape / reshape_dim / permute / flatten_dims /
# squeeze_dim / broadcast / expand_dim) defers to the source NkiTensor's native
# view ops (see NDSlice._as_nki_view). compute_fold is the only one left here:
# it is a DMA recipe for merging non-adjacent / partition dims, which has no
# NkiTensor equivalent (it is not a single strided view).
# ============================================================================


def compute_fold(element_shape, strides, src_dim, into_dim, position="outer"):
    """Fold src_dim into into_dim (removes src_dim).

    If position="outer": into_dim size = src * into, stride = into's stride.
    If position="inner": into_dim size = into * src, stride = src's stride.

    The stride is kept from the into_dim (outer) or src_dim (inner) to
    preserve the HBM AP pattern for store operations. For load, the SBUF
    AP is built contiguously regardless.
    """
    assert src_dim != into_dim
    if position == "outer":
        new_size = element_shape[src_dim] * element_shape[into_dim]
        new_stride = strides[into_dim]
    else:
        new_size = element_shape[into_dim] * element_shape[src_dim]
        new_stride = strides[src_dim]

    new_shape = []
    new_strides = []
    for d in range(len(element_shape)):
        if d == src_dim:
            pass  # folded away
        elif d == into_dim:
            new_shape.append(new_size)
            new_strides.append(new_stride)
        else:
            new_shape.append(element_shape[d])
            new_strides.append(strides[d])
    return (tuple(new_shape), tuple(new_strides))
