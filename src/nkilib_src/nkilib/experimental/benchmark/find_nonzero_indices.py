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

"""This kernel finds nonzero indices in a 1D input tensor and returns them with a count."""

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert

SUPPORTED_INPUT_DTYPES = [nl.float32, nl.int32]


@nki.jit
def find_nonzero_indices_with_count(input_tensor):
    """
    Find nonzero indices in a 1D tensor and return them with a count.

    This kernel identifies all nonzero elements in the input tensor and returns
    their indices along with the total count. The output format is optimized for
    sparse tensor operations in MoE (Mixture of Experts) routing scenarios.

    Dimensions:
        T: Sequence length (number of elements in input)

    Args:
        input_tensor (nl.ndarray): [1, T], Input tensor on HBM. Supported dtypes: float32, int32.

    Returns:
        output (nl.ndarray): [1, T+1], Output tensor on HBM with dtype int32. Format:
            [idx1, idx2, ..., -1, -1, ..., count] where nonzero indices come first,
            followed by padding (-1), and count is the last element.

    Notes:
        - Input must be 2D with shape [1, T]
        - Supported input dtypes: float32, int32
        - Output is always int32
        - Padding value is -1 for unused index slots
        - Count is stored in the last position (index T)
        - TODO: Handle case where E>1 -> can we load multiple E at once to parallelize?
        - TODO: Handle SBUF output (return view of tensor that gets rid of the garbage data)
        - TODO: Specify intended usage range (e.g., sequence length, batch size)

    Pseudocode:
        TODO: Add pseudocode description
    """
    _validate_inputs(input_tensor)
    T = input_tensor.shape[-1]
    P_MAX = 128  # NOTE: nl.tile_size.pmax does not have correct value until trace time

    # Allocate buffers
    input_sb = nl.ndarray((P_MAX, T), dtype=input_tensor.dtype, buffer=nl.sbuf, name="input_sb")
    nonzero_indices_with_count_padded_sb = nl.ndarray(
        (P_MAX, T + 1), dtype=nl.int32, buffer=nl.sbuf, name="nonzero_with_count_sb"
    )
    nonzero_indices_with_count_hbm = nl.ndarray(
        (1, T + 1), dtype=nl.int32, buffer=nl.hbm, name="nonzero_with_count_hbm"
    )

    # Load input into partition 0 of input_sb
    nisa.dma_copy(
        src=input_tensor[...],
        dst=input_sb[0, :],
    )

    # Call ISA
    nisa.nonzero_with_count(
        src=input_sb,
        dst=nonzero_indices_with_count_padded_sb,
        padding_val=-1,
        index_offset=0,
    )

    # Store partition 0 of nonzero_indices_with_count_padded_sb
    nisa.dma_copy(
        src=nonzero_indices_with_count_padded_sb[0, :],
        dst=nonzero_indices_with_count_hbm[0, :],
    )

    return nonzero_indices_with_count_hbm


def _validate_inputs(input_tensor):
    """
    Validate input tensor shape and dtype.

    Args:
        input_tensor (nl.ndarray): Input tensor to validate
    """
    kernel_assert(
        len(input_tensor.shape) == 2,
        f"Expected 2D input, got {len(input_tensor.shape)}D input with shape {input_tensor.shape}",
    )
    kernel_assert(input_tensor.shape[0] == 1, f"Expected dim0 of input equal to 1, got {input_tensor.shape=}")
    kernel_assert(
        input_tensor.dtype in SUPPORTED_INPUT_DTYPES,
        f"Expected input.dtype in {SUPPORTED_INPUT_DTYPES}, got {input_tensor.dtype=}",
    )
