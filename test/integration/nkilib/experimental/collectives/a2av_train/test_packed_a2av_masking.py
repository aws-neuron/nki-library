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
"""CPU checks for packed A2AV routing and topology contracts."""

import numpy as np
import pytest
from nkilib_src.nkilib.experimental.collectives.a2av_train.a2av_train_utils import (
    _validate_trn3_a2av_group_size,
)
from nkilib_src.nkilib.experimental.collectives.a2av_train.permute_a2av_torch import (
    _validate_routed_row_permutation,
)


def test_packed_permutation_allows_empty_peer_segments():
    vector = np.array([4, 1, 3, 0, 2], dtype=np.int32)
    counts = np.array([[0, 2, 0, 3]], dtype=np.int32)
    _validate_routed_row_permutation(vector, counts)


@pytest.mark.parametrize("vector", [[0, 0, 2], [0, 1, 3], [-1, 1, 2]])
def test_packed_reference_rejects_duplicate_or_out_of_range_rows(vector):
    with pytest.raises(ValueError, match="permutation"):
        _validate_routed_row_permutation(np.array(vector, dtype=np.int32), np.array([[1, 2]], dtype=np.int32))


@pytest.mark.parametrize("counts", [[1, 1], [2, 2], [-1, 4]])
def test_packed_reference_rejects_counts_that_do_not_partition_rows(counts):
    with pytest.raises(ValueError, match="partition"):
        _validate_routed_row_permutation(np.arange(3, dtype=np.int32), np.array([counts], dtype=np.int32))


@pytest.mark.parametrize(
    "vector",
    [np.arange(3, dtype=np.int64), np.zeros((3, 2), dtype=np.int32)],
)
def test_packed_reference_rejects_wrong_index_shape_or_dtype(vector):
    with pytest.raises(ValueError, match="int32"):
        _validate_routed_row_permutation(vector, np.array([[1, 2]], dtype=np.int32))


@pytest.mark.parametrize("ep_size", [8, 12, 64, 128])
def test_trn3_group_size_accepts_complete_multi_device_groups(ep_size: int):
    _validate_trn3_a2av_group_size(ep_size)


@pytest.mark.parametrize("ep_size", [1, 4, 6, 10])
def test_trn3_group_size_rejects_incomplete_or_single_device_groups(ep_size: int):
    with pytest.raises(AssertionError, match="Trn3 LNC=2 A2AV requires"):
        _validate_trn3_a2av_group_size(ep_size)
