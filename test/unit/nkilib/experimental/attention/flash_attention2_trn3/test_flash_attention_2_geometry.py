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
"""Unit tests for ``flash_attention_2``'s walk geometry: ``FlashAttentionGeometry`` and ``StepInfo``.

These carry the whole of the kernel's unaligned-shape reasoning, and they are the only part of it that
can be checked without a device or the simulator. Worth testing hard, because the failure mode is
silent: a wrong ``is_last_tile`` means the reduce chain is never read out, so ``block_max`` keeps a
stale value from an earlier point and the softmax reference is quietly wrong -- plausible numbers, no
crash.

Every expectation below is computed from an independent reference (``_reference_steps``) that walks the
grid in plain Python, rather than restating the implementation's own arithmetic.
"""

import pytest
from nkilib_src.nkilib.experimental.attention.flash_attention2_trn3 import FlashAttentionConfig

_SQ_TILE = 128
_SV_TILE = 128


def _geometry(Sq, Sk, D=128):
    """The geometry owns the walk and the descriptor, so no tracing context is needed."""
    return FlashAttentionConfig().geometry(D=D, Sq=Sq, Sk=Sk)


def _reference_steps(Sq, Sk, sq_tiles_valid, sq_tiles_per_blk=4, sk_tiles_per_blk=4, sk_tile=512):
    """Independent model of the walk: one entry per step, in walk order.

    Built from the sequence lengths directly -- no ceil helpers, no reuse of the config -- so it fails
    if the implementation and this reference disagree about anything.
    """
    num_sk_tiles = len([t for t in range(10_000) if t * sk_tile < Sk])
    num_sk_blks = len([b for b in range(10_000) if b * sk_tiles_per_blk < num_sk_tiles])
    final_extent = Sk - (num_sk_tiles - 1) * sk_tile
    out = []
    for blk in range(num_sk_blks):
        for sq in range(sq_tiles_per_blk):
            for tile in range(sk_tiles_per_blk):
                global_tile = blk * sk_tiles_per_blk + tile
                tiles_here = min(sk_tiles_per_blk, num_sk_tiles - blk * sk_tiles_per_blk)
                is_final_tile = global_tile == num_sk_tiles - 1
                sk_valid = final_extent if is_final_tile else sk_tile
                out.append(
                    {
                        "index": (blk, sq, tile),
                        "exists": sq < sq_tiles_valid and tile < tiles_here,
                        "is_first_tile": tile == 0,
                        "is_last_tile": tile == tiles_here - 1,
                        "sk_valid": sk_valid,
                        "sv_tiles_valid": -(-sk_valid // _SV_TILE),
                        "sv_last_extent": sk_valid - (-(-sk_valid // _SV_TILE) - 1) * _SV_TILE,
                    }
                )
    return out


# Aligned, short on Sk tile count, partial final Sk tile, short Sq, and combinations.
_SHAPES = [
    (512, 2048),  # fully aligned: the regression baseline
    (512, 2560),  # Sk short on tile count (5 tiles), every tile whole
    (512, 2816),  # Sk's final tile partial (256 of 512)
    (512, 10240),  # the perf shape: 5 whole blocks
    (640, 2048),  # Sq short on tile count (5 tiles)
    (896, 2816),  # both axes ragged
    (128, 128),  # single tile on both axes
    (1024, 4096),
]


@pytest.mark.parametrize("Sq, Sk", _SHAPES)
class TestConfigCounts:
    def test_sk_blocks_cover_the_sequence_exactly(self, Sq, Sk):
        geometry = FlashAttentionConfig().geometry(D=128, Sq=Sq, Sk=Sk)
        assert len(geometry.sk_blocks) == geometry.num_sk_blks
        covered = sum((count - 1) * geometry.sk_tile + extent for count, extent in geometry.sk_blocks)
        assert covered == Sk, f"sk_blocks {geometry.sk_blocks} cover {covered}, not {Sk}"
        # Only the very last tile of the very last block may be short.
        for blk in range(len(geometry.sk_blocks)):
            count, extent = geometry.sk_blocks[blk]
            assert 0 < count <= geometry.sk_tiles_per_blk
            assert 0 < extent <= geometry.sk_tile
            if blk < len(geometry.sk_blocks) - 1:
                assert extent == geometry.sk_tile, f"block {blk} is short but is not the last"

    def test_sq_counts(self, Sq, Sk):
        geometry = FlashAttentionConfig().geometry(D=128, Sq=Sq, Sk=Sk)
        assert (geometry.num_sq_tiles - 1) * _SQ_TILE < Sq <= geometry.num_sq_tiles * _SQ_TILE
        assert (geometry.num_sq_blks - 1) * geometry.sq_tiles_per_blk < geometry.num_sq_tiles


@pytest.mark.parametrize("Sq, Sk", _SHAPES)
class TestStepInfo:
    def test_matches_the_reference_walk(self, Sq, Sk):
        geometry = _geometry(Sq, Sk)
        sq_tiles_valid = min(geometry.sq_tiles_per_blk, geometry.num_sq_tiles)
        expected = _reference_steps(Sq, Sk, sq_tiles_valid)
        assert len(expected) == geometry.num_steps
        for step in range(len(expected)):
            got = geometry.step_info(step, sq_tiles_valid)
            want = expected[step]
            for field in ("index", "exists", "is_first_tile", "is_last_tile"):
                assert getattr(got, field) == want[field], f"{field} at step {step}, {Sq=}, {Sk=}"
            # Extents only have to agree where the step is real.
            if want["exists"]:
                assert got.sk_valid == want["sk_valid"], f"sk_valid at step {step}"
                assert got.sv_tiles_valid == want["sv_tiles_valid"], f"sv_tiles_valid at step {step}"
                assert got.sv_last_extent == want["sv_last_extent"], f"sv_last_extent at step {step}"

    def test_valid_steps_cover_every_sk_element_once(self, Sq, Sk):
        """The union of valid Sk tiles is exactly the sequence: no gap, no double-count."""
        geometry = _geometry(Sq, Sk)
        covered = 0
        for step in range(geometry.num_steps):
            info = geometry.step_info(step, 1)  # one Sq tile, so each Sk tile is counted once
            if info.exists:
                covered = covered + info.sk_valid
        assert covered == Sk

    def test_every_point_closes_its_chain_exactly_once(self, Sq, Sk):
        """One is_first_tile and one is_last_tile per existing point.

        This is the property that a ragged trailing block breaks if ``is_last_tile`` is keyed off the
        tile index instead of validity: the chain would never be read out.
        """
        geometry = _geometry(Sq, Sk)
        sq_tiles_valid = min(geometry.sq_tiles_per_blk, geometry.num_sq_tiles)
        firsts = {}
        lasts = {}
        for step in range(geometry.num_steps):
            info = geometry.step_info(step, sq_tiles_valid)
            if not info.exists:
                continue
            point = (info.index[0], info.index[1])
            if info.is_first_tile:
                firsts[point] = firsts.get(point, 0) + 1
            if info.is_last_tile:
                lasts[point] = lasts.get(point, 0) + 1
        expected_points = geometry.num_sk_blks * sq_tiles_valid
        assert len(firsts) == expected_points
        assert len(lasts) == expected_points
        for point in firsts:
            assert firsts[point] == 1, f"point {point} resets its chain {firsts[point]} times"
            assert lasts[point] == 1, f"point {point} closes its chain {lasts[point]} times"

    def test_absent_steps_are_only_ever_in_the_trailing_blocks(self, Sq, Sk):
        geometry = _geometry(Sq, Sk)
        sq_tiles_valid = min(geometry.sq_tiles_per_blk, geometry.num_sq_tiles)
        for step in range(geometry.num_steps):
            info = geometry.step_info(step, sq_tiles_valid)
            if not info.exists:
                is_trailing_sk = info.index[0] == geometry.num_sk_blks - 1
                assert is_trailing_sk or info.index[1] >= sq_tiles_valid, (
                    f"step {step} {info.index} absent outside a trailing block"
                )


class TestRaggedBlockSpecifics:
    """Hand-checked cases, so a bug in the reference model cannot hide a bug in the kernel."""

    def test_sk_2816_last_block_holds_two_tiles(self):
        # 2816 = 5 x 512 + 256 -> 6 tiles -> blocks of 4 = [4, 2]; final tile is 256 wide.
        geometry = _geometry(512, 2816)
        assert geometry.sk_blocks == [(4, 512), (2, 256)]
        assert geometry.num_sk_blks == 2
        # Block 1 holds tiles 0 and 1 only; tile 1 is the last and is the short one.
        assert [geometry.step_info(4 * 4 + t, 1).exists for t in range(4)] == [True, True, False, False]
        assert geometry.step_info(4 * 4 + 1, 1).is_last_tile is True
        assert geometry.step_info(4 * 4 + 1, 1).sk_valid == 256
        assert geometry.step_info(4 * 4 + 1, 1).sv_tiles_valid == 2
        # ...and block 0's last tile is still index 3, with the full extent.
        assert geometry.step_info(3, 1).is_last_tile is True
        assert geometry.step_info(3, 1).sk_valid == 512

    def test_sk_2560_short_block_of_whole_tiles(self):
        # 2560 = 5 x 512 -> 5 tiles -> blocks [4, 1]; no partial tile anywhere.
        geometry = _geometry(512, 2560)
        assert geometry.sk_blocks == [(4, 512), (1, 512)]
        # Block 1 holds one tile, which is therefore both first and last.
        tail = geometry.step_info(4 * 4, 1)
        assert tail.exists and tail.is_first_tile and tail.is_last_tile
        assert tail.sk_valid == 512
        assert [geometry.step_info(4 * 4 + t, 1).exists for t in range(4)] == [True, False, False, False]

    def test_short_sq_block_marks_missing_sq_tiles_absent(self):
        geometry = _geometry(640, 2048)
        assert geometry.num_sq_tiles == 5
        assert geometry.num_sq_blks == 2
        # The trailing Sq block has one tile: sq index 0 exists, 1..3 do not.
        assert [geometry.step_info(sq * 4, 1).exists for sq in range(4)] == [True, False, False, False]

    def test_single_tile_shape(self):
        geometry = _geometry(128, 128)
        assert geometry.sk_blocks == [(1, 128)] and geometry.num_sk_blks == 1
        first = geometry.step_info(0, 1)
        assert first.exists and first.is_first_tile and first.is_last_tile
        assert first.sk_valid == 128 and first.sv_tiles_valid == 1


class TestArbitraryLengths:
    """Any Sq and Sk. A sub-V-tile tail is cut into a block of its own; see :func:`_split_sk`."""

    @pytest.mark.parametrize("Sq, Sk", [(64, 64), (1, 1), (100, 72), (128, 64), (64, 128)])
    def test_single_partial_tile(self, Sq, Sk):
        geometry = FlashAttentionConfig().geometry(D=128, Sq=Sq, Sk=Sk)
        assert geometry.sk_blocks == [(1, Sk)]
        info = geometry.step_info(0, 1)
        assert info.exists and info.is_first_tile and info.is_last_tile
        assert info.sk_valid == Sk
        assert info.sv_last_extent == Sk - (info.sv_tiles_valid - 1) * _SV_TILE

    @pytest.mark.parametrize(
        "Sk, expected",
        [
            (2880, [(4, 512), (2, 256), (1, 64)]),
            (200, [(1, 128), (1, 72)]),
            (1000, [(2, 384), (1, 104)]),
            (2048, [(4, 512)]),
            (2816, [(4, 512), (2, 256)]),
        ],
    )
    def test_sub_v_tile_tail_becomes_its_own_block(self, Sk, expected):
        assert FlashAttentionConfig().geometry(D=128, Sq=128, Sk=Sk).sk_blocks == expected

    @pytest.mark.parametrize("Sk", [64, 72, 200, 512, 1000, 2048, 2100, 2880, 3000, 10240])
    def test_no_block_mixes_whole_and_partial_v_tiles(self, Sk):
        """The invariant the cut exists for: V's partition axis needs every block to be a rectangle."""
        geometry = FlashAttentionConfig().geometry(D=128, Sq=128, Sk=Sk)
        for blk in range(len(geometry.sk_blocks)):
            count, extent = geometry.sk_blocks[blk]
            v_rows = (count - 1) * geometry.sk_tile + extent
            has_partial = v_rows % _SV_TILE != 0
            v_tiles = (v_rows + _SV_TILE - 1) // _SV_TILE
            assert not has_partial or v_tiles == 1, f"Sk={Sk} block {blk}: {v_tiles} V tiles with a partial last one"
