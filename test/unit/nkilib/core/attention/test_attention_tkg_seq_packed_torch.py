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

"""Unit tests for runtime packed-accumulator route construction."""

import pytest
import torch
from nkilib_src.nkilib.core.attention.attention_tkg_seq_packed_torch import (
    build_packed_accumulator_route_tables,
)


def test_build_packed_accumulator_route_tables_exact_values():
    """Check duplicate, padded, and NC-local sequence routing exactly."""
    # Row 0 groups slots 0 and 2 for sequence 0; row 1 belongs to NC1 and
    # converts absolute IDs [4, 6, 5, 6] to local IDs [0, 2, 1, 2].
    seq_id_table = torch.tensor(
        [
            [0, 1, 0, -1],
            [4, 6, 5, 6],
        ],
        dtype=torch.int32,
    )

    routes = build_packed_accumulator_route_tables(
        seq_id_table,
        d_head=16,
        s_active_qh=1,
        rows_per_nc=1,
        bs_per_nc=4,
    )

    # Here s_active_qh=1, so a partial offset is group * 5 + source for four
    # slots: source zero is prior state and slot i uses source i+1. Row-0 slot 2
    # routes to group 0/source 3, while row-1 slots 1 and 3 share group 1.
    expected_partial = torch.tensor(
        [
            [1, 7, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 7, 13, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.uint16,
    ).reshape(2, 16, 1)
    # Each used group names its NC-local sequence. Unused groups route to dummy
    # state 4, one past the four real sequences on that NC.
    expected_group_state = torch.tensor(
        [
            [0, 1, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.uint16,
    ).reshape(2, 16, 1)

    assert torch.equal(routes.partial_route_table, expected_partial)
    assert torch.equal(routes.group_state_route_table, expected_group_state)


def test_build_packed_accumulator_route_tables_repeats_snake_quadrants():
    """Check that logical offsets repeat across d_head's 16-lane quadrants."""
    routes = build_packed_accumulator_route_tables(
        torch.tensor([[0, 1, 0, -1]], dtype=torch.int32),
        d_head=64,
        s_active_qh=8,
        rows_per_nc=1,
        bs_per_nc=4,
    )

    assert routes.partial_route_table.shape == (1, 64, 2)
    assert routes.group_state_route_table.shape == (1, 64, 2)
    for table in routes:
        assert table.dtype == torch.uint16
        assert torch.equal(table[:, 0:16], table[:, 16:32])
        assert torch.equal(table[:, 0:16], table[:, 32:48])
        assert torch.equal(table[:, 0:16], table[:, 48:64])


def test_build_packed_accumulator_route_tables_rejects_odd_slot_count():
    """Reject metadata that the packed kernel cannot consume."""
    with pytest.raises(ValueError, match="num_slots .* must be even"):
        build_packed_accumulator_route_tables(
            torch.tensor([[0, 1, 2]], dtype=torch.int32),
            d_head=16,
            s_active_qh=1,
            rows_per_nc=1,
            bs_per_nc=4,
        )


@pytest.mark.parametrize(
    ("seq_id_table", "match"),
    [
        (torch.tensor([[0.0, 1.0]]), "integer dtype"),
        (torch.tensor([[0.0 + 0.0j, 1.0 + 0.0j]]), "integer dtype"),
        (torch.tensor([[0, -2]], dtype=torch.int32), "not padding or local"),
        (torch.tensor([[0, 4]], dtype=torch.int32), "not padding or local"),
    ],
)
def test_build_packed_accumulator_route_tables_rejects_invalid_ids(
    seq_id_table,
    match,
):
    """Reject non-integral IDs and IDs outside the row's NC-local batch."""
    error = TypeError if torch.is_floating_point(seq_id_table) or torch.is_complex(seq_id_table) else ValueError
    with pytest.raises(error, match=match):
        build_packed_accumulator_route_tables(
            seq_id_table,
            d_head=64,
            s_active_qh=8,
            rows_per_nc=1,
            bs_per_nc=4,
        )
