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

"""
Tests for HBMLayout: offset tracking, stride transforms, indirect state.

Pure Python -- no NKI, no tracer, no device required.
"""

import nki.language as nl
import pytest
from nki.language.tensor import NkiTensor  # ty: ignore[unresolved-import]
from nkilib_src.nkilib.experimental.neurotile.core._helpers import contiguous_ap_pattern as _contiguous_ap_pattern
from nkilib_src.nkilib.experimental.neurotile.core._helpers import contiguous_strides as _contiguous_strides
from nkilib_src.nkilib.experimental.neurotile.core._helpers import reachable_dim_extent
from nkilib_src.nkilib.experimental.neurotile.core.axis import IndirectKind, IndirectOffset
from nkilib_src.nkilib.experimental.neurotile.core.layout_hbm import HBMLayout
from nkilib_src.nkilib.experimental.neurotile.core.layout_sbuf import SBUFLayout
from nkilib_src.nkilib.experimental.neurotile.core.transforms import compute_fold

from test.unit.nkilib.experimental.neurotile._mocks import MockTensor
from test.utils.pytest_test_metadata import pytest_marks


def _scalar_indirect(value, dim):
    return IndirectOffset(kind=IndirectKind.SCALAR, value=value, dim=dim)


def _vector_indirect(value, dim):
    return IndirectOffset(kind=IndirectKind.VECTOR, value=value, dim=dim)


def _view(shape, strides, buffer=nl.shared_hbm):
    """A synthetic NkiTensor carrying explicit (shape, strides) -- the same
    logical view NDSlice._as_nki_view builds, for asserting transform stride
    math directly against the native view ops (now the implementation)."""
    t = NkiTensor(shape=tuple(shape), dtype="float32", storage=None, buffer=buffer)
    return t._copy(shape=tuple(shape), strides=tuple(strides))


# ============================================================================
# Helpers
# ============================================================================


def make_layout(shape, offset=0):
    """Create HBMLayout with contiguous strides for given shape."""
    strides = _contiguous_strides(shape)
    return HBMLayout(
        source=MockTensor(shape),  # source must implement .ap() for NDSlice.data
        offset=offset,
        strides=strides,
        dtype="float32",
        buffer_type=nl.shared_hbm,
    )


# ============================================================================
# Contiguous strides helper
# ============================================================================


@pytest_marks(["neurotile"])
class TestContiguousStrides:
    @pytest.mark.fast
    def test_2d(self):
        assert _contiguous_strides((128, 512)) == (512, 1)

    def test_3d(self):
        assert _contiguous_strides((8, 128, 64)) == (8192, 64, 1)

    def test_1d(self):
        assert _contiguous_strides((256,)) == (1,)

    def test_4d(self):
        s = _contiguous_strides((2, 4, 8, 16))
        assert s == (512, 128, 16, 1)


# ============================================================================
# Advance -- concrete (int k)
# ============================================================================


@pytest_marks(["neurotile"])
class TestAdvanceConcrete:
    @pytest.mark.fast
    def test_tile_level(self):
        """Advance by 1 tile along dim 0, step=128."""
        layout = make_layout((512, 2048))
        # strides = (2048, 1)
        result = layout.advance(dim=0, k=1, step=128)
        # offset += 1 * 128 * 2048 = 262144
        assert result.offset == 262144
        assert result.strides == layout.strides  # unchanged

    def test_block_level(self):
        """Advance by 1 block along dim 0, step=256 (bs=2, ts=128)."""
        layout = make_layout((512, 2048))
        result = layout.advance(dim=0, k=1, step=256)
        # offset += 1 * 256 * 2048 = 524288
        assert result.offset == 524288

    def test_element_level(self):
        """Advance by 42 elements along dim 0, step=1."""
        layout = make_layout((128, 64))
        result = layout.advance(dim=0, k=42, step=1)
        # offset += 42 * 1 * 64 = 2688
        assert result.offset == 2688

    def test_dim1_advance(self):
        """Advance along dim 1."""
        layout = make_layout((512, 2048))
        result = layout.advance(dim=1, k=2, step=512)
        # offset += 2 * 512 * 1 = 1024
        assert result.offset == 1024

    def test_cumulative_offset(self):
        """Multiple advances accumulate offset."""
        layout = make_layout((512, 2048))
        r1 = layout.advance(dim=0, k=1, step=128)  # +262144
        r2 = r1.advance(dim=1, k=2, step=512)  # +1024
        assert r2.offset == 262144 + 1024

    def test_preserves_indirect(self):
        """Concrete advance preserves an existing IndirectOffset tag."""
        layout = HBMLayout(
            source=None,
            offset=0,
            strides=(2048, 1),
            dtype="float32",
            buffer_type=nl.shared_hbm,
            indirect=_scalar_indirect("some_scalar", 0),
        )
        result = layout.advance(dim=1, k=1, step=512)
        assert result.indirect.value == "some_scalar"
        assert result.indirect.dim == 0


# ============================================================================
# Advance -- indirect (non-int k)
# ============================================================================


@pytest_marks(["neurotile"])
class TestAdvanceIndirect:
    @pytest.mark.fast
    def test_step_1_no_scaling(self):
        """Runtime k with step=1: stored as-is, no scaling."""
        layout = make_layout((8, 128, 64))
        runtime_k = 5.0  # non-int triggers indirect path
        result = layout.advance(dim=0, k=runtime_k, step=1)
        assert result.indirect.value == 5.0
        assert result.indirect.dim == 0
        assert result.offset == 0

    def test_step_1_sbuf_scalar_does_not_scale(self, monkeypatch):
        """Even tensor-like scalar indices pass through unchanged when step=1."""
        layout = make_layout((8, 128, 64))
        runtime_k = MockTensor((1, 1), dtype=nl.int32)

        def fail_if_called(value, scale):
            raise AssertionError("unexpected scaling")

        monkeypatch.setattr(HBMLayout, "_materialize_scaled_scalar_offset", staticmethod(fail_if_called))
        result = layout.advance(dim=0, k=runtime_k, step=1)
        assert result.indirect.value is runtime_k
        assert result.indirect.dim == 0

    def test_step_gt_1_sbuf_scalar_scaled(self, monkeypatch):
        """SBUF scalar logical index with step>1 is materialized once."""
        layout = make_layout((512, 2048))
        runtime_k = MockTensor((1, 1), dtype=nl.int32)
        monkeypatch.setattr(
            HBMLayout,
            "_materialize_scaled_scalar_offset",
            staticmethod(lambda value, scale: ("scaled", value, scale)),
        )
        result = layout.advance(dim=0, k=runtime_k, step=128)
        assert result.indirect.value == ("scaled", runtime_k, 128)
        assert result.indirect.dim == 0
        assert result.offset == 0

    def test_step_gt_1_buffered_sbuf_scalar_scaled(self, monkeypatch):
        """Scalar NKI tensors are accepted only when they are SBUF-backed."""
        layout = make_layout((512, 2048))
        runtime_k = MockTensor((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        monkeypatch.setattr(
            HBMLayout,
            "_materialize_scaled_scalar_offset",
            staticmethod(lambda value, scale: ("scaled", value, scale)),
        )
        result = layout.advance(dim=0, k=runtime_k, step=128)
        assert result.indirect.value == ("scaled", runtime_k, 128)

    def test_step_gt_1_loop_var_rejects(self):
        """Loop-variable-like runtime values cannot be scaled internally."""
        layout = make_layout((8, 128, 64))
        with pytest.raises(AssertionError, match="requires an SBUF scalar index"):
            layout.advance(dim=1, k=object(), step=128)

    def test_step_gt_1_vector_shaped_scalar_rejects(self):
        layout = make_layout((512, 2048))
        with pytest.raises(AssertionError, match="requires an SBUF scalar index"):
            layout.advance(dim=0, k=MockTensor((1, 2), dtype=nl.int32), step=128)

    def test_step_1_vector_shaped_scalar_rejects(self):
        layout = make_layout((8, 128, 64))
        with pytest.raises(AssertionError, match="requires an SBUF scalar index"):
            layout.advance(dim=0, k=MockTensor((1, 2), dtype=nl.int32), step=1)

    def test_runtime_logical_bool_index_rejects(self):
        layout = make_layout((512, 2048))
        with pytest.raises(AssertionError, match="does not accept bool"):
            layout.advance(dim=0, k=True, step=128)

    @pytest.mark.parametrize("step", [1, 128])
    def test_hbm_scalar_tensor_rejects(self, step):
        layout = make_layout((512, 2048))
        with pytest.raises(AssertionError, match="requires an SBUF scalar index"):
            layout.advance(dim=0, k=MockTensor((1, 1), dtype=nl.int32, buffer=nl.shared_hbm), step=step)

    @pytest.mark.parametrize("step", [0, -1, 1.5, True])
    def test_invalid_index_stride_rejects(self, step):
        layout = make_layout((512, 2048))
        with pytest.raises(AssertionError, match="index_stride_elements must be a positive int"):
            layout.advance(dim=0, k=0, step=step)

    @pytest.mark.parametrize("dim", [-1, 2, 1.0, True])
    def test_invalid_dim_rejects(self, dim):
        layout = make_layout((512, 2048))
        with pytest.raises(AssertionError, match="dim must be an int"):
            layout.advance(dim=dim, k=0, step=128)

    def test_indirect_dim_is_surviving_dim(self):
        """IndirectOffset dim is the dim parameter."""
        layout = make_layout((8, 128, 64))
        result = layout.advance(dim=1, k=2.0, step=1)
        assert result.indirect.dim == 1

    def test_indirect_replaces_vector(self):
        """Scalar advance replaces any pre-existing vector indirect."""
        layout = HBMLayout(
            source=None,
            offset=0,
            strides=(8192, 64, 1),
            dtype="float32",
            indirect=_vector_indirect("old_vector", 0),
        )
        result = layout.advance(dim=0, k=2.0, step=1)
        assert result.indirect.kind == IndirectKind.SCALAR
        assert result.indirect.value == 2.0

    def test_element_offset_static_folds_without_logical_stride(self):
        layout = make_layout((512, 2048))
        result = layout.advance_by_element_offset(dim=0, offset=128)
        assert result.offset == 128 * 2048
        assert result.indirect is None

    def test_element_offset_negative_static_rejects(self):
        layout = make_layout((512, 2048))
        with pytest.raises(AssertionError, match="must be non-negative"):
            layout.advance_by_element_offset(dim=0, offset=-1)

    @pytest.mark.parametrize("offset", [True, 1.0, None, [1], object()])
    def test_element_offset_invalid_payload_rejects(self, offset):
        layout = make_layout((512, 2048))
        with pytest.raises(AssertionError, match="nt.element_offset"):
            layout.advance_by_element_offset(dim=0, offset=offset)

    @pytest.mark.parametrize("buffer", [nl.shared_hbm, nl.psum])
    def test_element_offset_non_sbuf_tensor_rejects(self, buffer):
        layout = make_layout((512, 2048))
        with pytest.raises(AssertionError, match="SBUF scalar"):
            layout.advance_by_element_offset(dim=0, offset=MockTensor((1, 1), dtype=nl.int32, buffer=buffer))

    def test_element_offset_runtime_passes_through_without_scaling(self):
        layout = make_layout((512, 2048))
        runtime_offset = MockTensor((1, 1), dtype=nl.int32)
        result = layout.advance_by_element_offset(dim=0, offset=runtime_offset)
        assert result.indirect.kind == IndirectKind.SCALAR
        assert result.indirect.value is runtime_offset
        assert result.indirect.dim == 0


# ============================================================================
# Advance vector
# ============================================================================


@pytest_marks(["neurotile"])
class TestSetIndirectVector:
    @pytest.mark.fast
    def test_basic(self):
        layout = make_layout((512, 2048))
        vec = "mock_vector_tensor"
        result = layout.set_indirect(IndirectKind.VECTOR, vec, dim=0)
        assert result.indirect.kind == IndirectKind.VECTOR
        assert result.indirect.value == vec
        assert result.indirect.dim == 0


# ============================================================================
# Drop dim
# ============================================================================


@pytest_marks(["neurotile"])
class TestDropDim:
    @pytest.mark.fast
    def test_drop_dim0(self):
        """Drop dim 0: strides shrink, offset preserved."""
        layout = make_layout((8, 128, 64))
        # strides = (8192, 64, 1)
        result = layout.drop_dim(0)
        assert result.strides == (64, 1)
        assert result.offset == 0
        assert result.source is layout.source

    def test_drop_dim1(self):
        layout = make_layout((8, 128, 64))
        result = layout.drop_dim(1)
        assert result.strides == (8192, 1)

    def test_drop_preserves_indirect_dim(self):
        """IndirectOffset dim is source-relative -- drop_dim doesn't shift it."""
        layout = HBMLayout(
            source=None,
            offset=0,
            strides=(8192, 64, 1),
            dtype="float32",
            indirect=_scalar_indirect("eid", 0),
        )
        result = layout.drop_dim(0)
        assert result.indirect.dim == 0
        assert result.strides == (64, 1)

    def test_advance_then_drop(self):
        """Full iteration dim flow: advance(runtime) then drop."""
        layout = make_layout((8, 128, 64))
        advanced = layout.advance(dim=0, k=5.0, step=1)
        dropped = advanced.drop_dim(0)
        assert dropped.strides == (64, 1)
        assert dropped.indirect.value == 5.0
        assert dropped.indirect.dim == 0


# ============================================================================
# With strides (for transforms)
# ============================================================================


@pytest_marks(["neurotile"])
class TestApplyTransform:
    @pytest.mark.fast
    def test_basic(self):
        layout = make_layout((128, 512))
        new_layout = layout.apply_transform((1, 128))
        assert new_layout.strides == (1, 128)
        assert new_layout.offset == layout.offset
        assert new_layout.source is layout.source


# ============================================================================
# Stride transforms (standalone functions)
# ============================================================================


# Transform stride math (reshape_dim / reshape / permute / flatten_dims /
# squeeze_dim / broadcast) is now NkiTensor's native view ops -- the same ones
# NDSlice delegates to. These assert that math directly on a synthetic view.
@pytest_marks(["neurotile"])
class TestReshapeDim:
    @pytest.mark.fast
    def test_split_last_dim(self):
        """Split dim 1 of (128, 512) into (128, 8, 64)."""
        r = _view((128, 512), (512, 1)).reshape_dim(1, (8, 64))
        assert tuple(r.shape) == (128, 8, 64)
        assert tuple(r.strides) == (512, 64, 1)

    def test_split_first_dim(self):
        """Split dim 0 of (128, 512) into (4, 32, 512)."""
        r = _view((128, 512), (512, 1)).reshape_dim(0, (4, 32))
        assert tuple(r.shape) == (4, 32, 512)
        assert tuple(r.strides) == (16384, 512, 1)

    def test_split_middle_dim(self):
        """Split dim 1 of (4, 128, 64) into (4, 2, 64, 64)."""
        r = _view((4, 128, 64), (8192, 64, 1)).reshape_dim(1, (2, 64))
        assert tuple(r.shape) == (4, 2, 64, 64)
        assert tuple(r.strides) == (8192, 4096, 64, 1)

    def test_bad_product_rejected(self):
        """A sub-shape whose product != the dim size raises (strict validation
        inherited from NkiTensor; the old compute_reshape_dim mis-strided silently)."""
        with pytest.raises(AssertionError, match="[Ss]ize mismatch"):
            _view((128, 512), (512, 1)).reshape_dim(1, (8, 65))


@pytest_marks(["neurotile"])
class TestReshape:
    @pytest.mark.fast
    def test_basic(self):
        """Full reshape (128, 512) -> (64, 1024)."""
        r = _view((128, 512), (512, 1)).reshape((64, 1024))
        assert tuple(r.shape) == (64, 1024)
        assert tuple(r.strides) == (1024, 1)

    def test_noncontiguous_reshape_rejected(self):
        """Reshape of a permuted (non-contiguous) view raises -- it cannot be
        expressed as a view without copying. (The old compute_reshape silently
        mis-addressed this by scaling contiguous strides.)"""
        with pytest.raises(AssertionError, match="[Nn]on-contiguous"):
            _view((512, 128), (1, 512)).reshape((256, 256))


@pytest_marks(["neurotile"])
class TestPermute:
    @pytest.mark.fast
    def test_transpose_2d(self):
        r = _view((128, 512), (512, 1)).permute((1, 0))
        assert tuple(r.shape) == (512, 128)
        assert tuple(r.strides) == (1, 512)

    def test_permute_3d(self):
        r = _view((4, 128, 64), (8192, 64, 1)).permute((2, 0, 1))
        assert tuple(r.shape) == (64, 4, 128)
        assert tuple(r.strides) == (1, 8192, 64)


@pytest_marks(["neurotile"])
class TestFlattenDims:
    @pytest.mark.fast
    def test_flatten_last_two(self):
        """Flatten (4, 128, 64) dims [1,2] -> (4, 8192)."""
        r = _view((4, 128, 64), (8192, 64, 1)).flatten_dims(1, 2)
        assert tuple(r.shape) == (4, 8192)
        assert tuple(r.strides) == (8192, 1)  # innermost stride of merged range

    def test_flatten_first_two(self):
        """Flatten (4, 128, 64) dims [0,1] -> (512, 64)."""
        r = _view((4, 128, 64), (8192, 64, 1)).flatten_dims(0, 1)
        assert tuple(r.shape) == (512, 64)
        assert tuple(r.strides) == (64, 1)

    def test_noncontiguous_flatten_rejected(self):
        """Flattening dims that are not contiguous in memory raises (strict;
        the old compute_flatten_dims silently mis-strided)."""
        # (4,128,64) permuted so dims 1,2 are no longer contiguous.
        with pytest.raises(AssertionError, match="[Cc]ontiguous|contiguous"):
            _view((4, 64, 128), (8192, 1, 64)).flatten_dims(1, 2)


@pytest_marks(["neurotile"])
class TestSqueezeDim:
    @pytest.mark.fast
    def test_basic(self):
        r = _view((1, 128, 64), (8192, 64, 1)).squeeze_dim(0)
        assert tuple(r.shape) == (128, 64)
        assert tuple(r.strides) == (64, 1)


@pytest_marks(["neurotile"])
class TestExpandDim:
    @pytest.mark.fast
    def test_basic(self):
        # expand_dim defers to NkiTensor: inserts a size-1 dim. Its stride is
        # positional (a size-1 axis is never dereferenced; a following broadcast
        # overwrites it to 0). Assert the shape; stride[1] is don't-care.
        r = _view((128, 64), (64, 1)).expand_dim(1)
        assert tuple(r.shape) == (128, 1, 64)

    def test_pairs_with_broadcast(self):
        # The real usage: expand_dim then broadcast -> stride-0 broadcast axis.
        r = _view((128, 64), (64, 1)).expand_dim(1).broadcast(1, 8)
        assert tuple(r.shape) == (128, 8, 64)
        assert r.strides[1] == 0


@pytest_marks(["neurotile"])
class TestBroadcast:
    @pytest.mark.fast
    def test_basic(self):
        r = _view((128, 1, 64), (64, 0, 1)).broadcast(1, 16)
        assert tuple(r.shape) == (128, 16, 64)
        assert tuple(r.strides) == (64, 0, 1)


@pytest_marks(["neurotile"])
class TestComputeFold:
    @pytest.mark.fast
    def test_fold_outer(self):
        """Fold dim 2 into dim 0 (outer)."""
        shape = (4, 8, 16)
        strides = (128, 16, 1)
        new_shape, new_strides = compute_fold(shape, strides, src_dim=2, into_dim=0)
        assert new_shape == (64, 8)  # 4*16=64, dim 2 removed
        assert new_strides == (128, 16)  # into_dim gets its own stride

    def test_fold_inner(self):
        shape = (4, 8, 16)
        strides = (128, 16, 1)
        new_shape, new_strides = compute_fold(shape, strides, src_dim=2, into_dim=0, position="inner")
        assert new_shape == (64, 8)
        assert new_strides == (1, 16)  # src_dim's stride


# ============================================================================
# Chained transforms -- realistic pipeline
# ============================================================================


@pytest_marks(["neurotile"])
class TestTransformPipeline:
    @pytest.mark.fast
    def test_flatten_reshape_permute(self):
        """Realistic: (B, S, H) -> flatten(0,1) -> reshape_dim -> permute."""
        shape = (4, 128, 512)
        strides = _contiguous_strides(shape)
        assert strides == (65536, 512, 1)

        # flatten B,S -> reshape H into (H0,H1) -> permute to (H0, BS, H1).
        r = _view(shape, strides).flatten_dims(0, 1).reshape_dim(1, (8, 64)).permute((1, 0, 2))
        assert tuple(r.shape) == (8, 512, 64)
        assert tuple(r.strides) == (64, 512, 1)

    def test_layout_apply_transform_pipeline(self):
        """Apply a transform pipeline's resulting strides through HBMLayout.apply_transform."""
        layout = make_layout((4, 128, 512))

        r = _view((4, 128, 512), layout.strides).flatten_dims(0, 1).reshape_dim(1, (8, 64)).permute((1, 0, 2))

        result = layout.apply_transform(tuple(r.strides))
        assert result.strides == (64, 512, 1)
        assert result.offset == 0
        assert result.source is layout.source


# ============================================================================
# Repr
# ============================================================================


@pytest_marks(["neurotile"])
class TestRepr:
    @pytest.mark.fast
    def test_basic(self):
        layout = make_layout((128, 512))
        r = repr(layout)
        assert "HBMLayout(" in r
        assert "offset=0" in r

    def test_with_indirect(self):
        layout = HBMLayout(
            source=None,
            offset=100,
            strides=(64, 1),
            dtype="float32",
            indirect=_scalar_indirect("eid", 0),
        )
        r = repr(layout)
        assert "indirect=" in r


# ============================================================================
# compute_effective_tiles: clamping and tile count
# Regression: incorrect clamping caused wrong SBUF allocation shapes
# ============================================================================


@pytest_marks(["neurotile"])
class TestComputeEffectiveTiles:
    @pytest.mark.fast
    def test_exact_fit(self):
        eff, shape = HBMLayout.compute_effective_tiles((128, 512), (128, 512))
        assert eff == (128, 512)
        assert shape == (1, 1)

    def test_multi_tiles(self):
        eff, shape = HBMLayout.compute_effective_tiles((128, 512), (128, 128))
        assert eff == (128, 128)
        assert shape == (1, 4)

    def test_sub_tile_clamp(self):
        eff, shape = HBMLayout.compute_effective_tiles((64, 256), (128, 512))
        assert eff == (64, 256)
        assert shape == (1, 1)

    def test_remainder_ceiling(self):
        eff, shape = HBMLayout.compute_effective_tiles((128, 300), (128, 128))
        assert eff == (128, 128)
        assert shape == (1, 3)

    def test_3d(self):
        eff, shape = HBMLayout.compute_effective_tiles((8, 128, 256), (1, 128, 128))
        assert eff == (1, 128, 128)
        assert shape == (8, 1, 2)


# ============================================================================
# compute_sbuf_alloc: N-D vs flat 2D
# Regression: wrong allocation shape caused SBUF AP mismatch
# ============================================================================


@pytest_marks(["neurotile"])
class TestComputeSbufAlloc:
    @pytest.mark.fast
    def test_single_p_tile_keeps_nd(self):
        shape, p_tiles = HBMLayout.compute_sbuf_alloc((128, 512), (128, 512))
        assert p_tiles == 1
        assert shape == (128, 512)

    def test_multi_p_tile_flattens(self):
        shape, p_tiles = HBMLayout.compute_sbuf_alloc((256, 512), (128, 512))
        assert p_tiles == 2
        assert shape == (128, 1024)

    def test_3d_single_p_tile(self):
        shape, p_tiles = HBMLayout.compute_sbuf_alloc((128, 8, 64), (128, 8, 64))
        assert p_tiles == 1
        assert shape == (128, 8, 64)


# ============================================================================
# sbuf_f_extent and _contiguous_ap_pattern
# ============================================================================


@pytest_marks(["neurotile"])
class TestSbufFExtent:
    @pytest.mark.fast
    def test_single_p_tile(self):
        assert SBUFLayout.f_extent((128, 512), 128) == 512

    def test_multi_p_tile(self):
        assert SBUFLayout.f_extent((256, 512), 128) == 1024

    def test_3d(self):
        assert SBUFLayout.f_extent((128, 8, 64), 128) == 512


@pytest_marks(["neurotile"])
class TestContiguousApPattern:
    @pytest.mark.fast
    def test_2d(self):
        assert _contiguous_ap_pattern((128, 512)) == [[512, 128], [1, 512]]

    def test_3d(self):
        assert _contiguous_ap_pattern((128, 8, 64)) == [[512, 128], [64, 8], [1, 64]]


# ============================================================================
# indirect_dim preservation across multiple batch dim drops
# Regression: inline drop shifted dim indices but indirect_dim is source-relative
# ============================================================================


@pytest_marks(["neurotile"])
class TestIndirectDimBatchDrops:
    @pytest.mark.fast
    def test_drop_before_indirect(self):
        """Drop dim 0 with IndirectOffset.dim=1 keeps IndirectOffset.dim=1."""
        layout = HBMLayout(
            source=None,
            offset=0,
            strides=(512, 64, 1),
            dtype="float32",
            indirect=_scalar_indirect("pos", 1),
        )
        dropped = layout.drop_dim(0)
        assert dropped.indirect.dim == 1

    def test_drop_after_indirect(self):
        layout = HBMLayout(
            source=None,
            offset=0,
            strides=(512, 64, 1),
            dtype="float32",
            indirect=_scalar_indirect("pos", 1),
        )
        dropped = layout.drop_dim(2)
        assert dropped.indirect.dim == 1

    def test_two_drops_preserve_indirect(self):
        """IndirectOffset.dim is source-relative -- repeated drops don't shift it."""
        layout = HBMLayout(
            source=None,
            offset=0,
            strides=(4096, 512, 64, 1),
            dtype="float32",
            indirect=_scalar_indirect("pos", 2),
        )
        d1 = layout.drop_dim(0)
        assert d1.indirect.dim == 2
        d2 = d1.drop_dim(0)
        assert d2.indirect.dim == 2


# ============================================================================
# _apply_ap: pattern + offset, optional indirect dispatch
# ============================================================================


@pytest_marks(["neurotile"])
class TestBuildOverrideAp:
    @pytest.mark.fast
    def test_no_indirect(self):
        """No indirect: plain AP with pattern + offset."""
        result = HBMLayout._apply_ap(
            MockTensor(128, 512),
            offset=100,
            pattern=[[128, 4], [1, 64]],
            indirect=None,
        )
        assert result["pattern"] == [[128, 4], [1, 64]]
        assert result["offset"] == 100
        assert "scalar_offset" not in result
        assert "vector_offset" not in result

    def test_scalar_indirect(self):
        """Scalar IndirectOffset: passes scalar_offset + indirect_dim."""
        result = HBMLayout._apply_ap(
            MockTensor(128, 512),
            offset=0,
            pattern=[[128, 16], [2, 64]],
            indirect=_scalar_indirect("eid_scalar", 0),
        )
        assert result["scalar_offset"] == "eid_scalar"
        assert result["indirect_dim"] == 0
        assert "vector_offset" not in result

    def test_vector_indirect(self):
        """Vector IndirectOffset: passes vector_offset + indirect_dim."""
        result = HBMLayout._apply_ap(
            MockTensor(128, 512),
            offset=0,
            pattern=[[128, 16], [2, 64]],
            indirect=_vector_indirect("vec_tensor", 0),
        )
        assert result["vector_offset"] == "vec_tensor"
        assert result["indirect_dim"] == 0
        assert "scalar_offset" not in result


# ============================================================================
# DMA-transpose F-chunking helper
#
# `_f_chunks(f_dim)` is the pure tiling decision for the transpose free dim:
# it returns [(col_offset, width), ...] covering [0, f_dim) with full 128-wide
# chunks and a final partial chunk of width (f_dim % 128). It is a low-level
# helper and is tested here in isolation for all F. Note the load layer only
# EXERCISES the single-chunk (F <= 128) and all-full-chunk (F % 128 == 0)
# outputs: `.load(transpose=True)` rejects an F that is both > 128 and not a
# multiple of 128 to keep the single-DMA contract (see
# TestNeurotileTranspose.test_transpose_multi_dma_rejected). The partial-chunk
# rows below stay valid unit coverage of the helper itself.
# ============================================================================


@pytest_marks(["neurotile"])
class TestDmaTransposeFChunks:
    @pytest.mark.fast
    def test_small_single_chunk(self):
        """F <= 128: one chunk, full width."""
        assert HBMLayout._f_chunks(100) == [(0, 100)]

    def test_exact_128(self):
        """F == 128: one full chunk."""
        assert HBMLayout._f_chunks(128) == [(0, 128)]

    def test_exact_multiple(self):
        """F == k*128: k full chunks, no remainder (regression: today's only
        supported tiled case)."""
        assert HBMLayout._f_chunks(512) == [(0, 128), (128, 128), (256, 128), (384, 128)]

    def test_remainder_partial_last(self):
        """F = k*128 + r (r>0): k full chunks + one width-r partial chunk.
        This is the case today's assert rejects."""
        # 400 = 3*128 + 16
        assert HBMLayout._f_chunks(400) == [(0, 128), (128, 128), (256, 128), (384, 16)]

    def test_remainder_under_128(self):
        """F < 128 is itself the partial case -- single chunk of width F."""
        assert HBMLayout._f_chunks(16) == [(0, 16)]

    def test_just_over_128(self):
        """F = 129 -> one full chunk + a width-1 partial."""
        assert HBMLayout._f_chunks(129) == [(0, 128), (128, 1)]

    def test_chunks_cover_f_exactly(self):
        """Property: chunk widths sum to F and offsets are contiguous, for a
        spread of remainder shapes."""
        for f_dim in (1, 16, 128, 129, 256, 300, 400, 8192, 8200):
            chunks = HBMLayout._f_chunks(f_dim)
            # contiguous, non-overlapping, covering [0, f_dim)
            pos = 0
            for offset, width in chunks:
                assert offset == pos, f"f_dim={f_dim}: gap at {offset} != {pos}"
                assert 0 < width <= 128, f"f_dim={f_dim}: bad width {width}"
                pos += width
            assert pos == f_dim, f"f_dim={f_dim}: covered {pos} != {f_dim}"


# ============================================================================
# N-D gather-transpose: per-rank axes default + AP builder (Appendix D).
#
# `_default_transpose_axes(rank)` is nisa.dma_transpose's per-rank permutation;
# `_gather_transpose_aps(dims, axes)` builds the (hbm, sbuf, dst_shape) for a
# gather-transpose. Pure shape math -- no tracer/device.
# ============================================================================


@pytest_marks(["neurotile"])
class TestDefaultTransposeAxes:
    @pytest.mark.fast
    def test_per_rank(self):
        assert HBMLayout._default_transpose_axes(2) == (1, 0)
        assert HBMLayout._default_transpose_axes(3) == (2, 1, 0)
        assert HBMLayout._default_transpose_axes(4) == (3, 1, 2, 0)

    def test_unsupported_rank_returns_none(self):
        # Returns None (not raises) for an unsupported rank; the NDSlice.load
        # validation boundary turns that into a named error.
        assert HBMLayout._default_transpose_axes(5) is None
        assert HBMLayout._default_transpose_axes(1) is None


@pytest_marks(["neurotile"])
class TestGatherTransposeAps:
    """Per-rank AP shapes for gather-transpose -- dim 0 is the gathered-rows dim;
    dst is the transpose of `dims` under the permutation."""

    @pytest.mark.fast
    def test_2d(self):
        hbm, sbuf, shape = HBMLayout._gather_transpose_aps((16, 64), (1, 0))
        assert shape == (64, 16)
        assert hbm == [[64, 16], [1, 64]]
        assert sbuf == [[16, 64], [1, 16]]

    def test_3d(self):
        # src (rows=16, n_tiles=4, tile=64) -> dst (64, 4, 16)
        hbm, sbuf, shape = HBMLayout._gather_transpose_aps((16, 4, 64), (2, 1, 0))
        assert shape == (64, 4, 16)
        assert hbm == [[256, 16], [64, 4], [1, 64]]
        assert sbuf == [[64, 64], [16, 4], [1, 16]]

    def test_4d_has_dummy_dim(self):
        # src (rows=16, f_tiles=8, P=128) -> dst (128, 1, 8, 16); the [1,1] dim is
        # the hardware-required size-1 padding.
        hbm, sbuf, shape = HBMLayout._gather_transpose_aps((16, 8, 128), (3, 1, 2, 0))
        assert shape == (128, 1, 8, 16)
        assert hbm == [[1024, 16], [1, 1], [128, 8], [1, 128]]
        assert sbuf == [[128, 128], [1, 1], [16, 8], [1, 16]]

    def test_dst_shape_is_permutation_of_dims(self):
        # 4-D dst inserts a leading dummy; the non-dummy extents are the reverse-ish
        # transpose: P (last) -> partition, rows (first) -> free.
        _, _, shape = HBMLayout._gather_transpose_aps((32, 4, 64), (3, 1, 2, 0))
        assert shape == (64, 1, 4, 32)


# ============================================================================
# DMA-transpose dst partition stride: the emitted SBUF AP's level-0 stride must
# equal the dst buffer's PHYSICAL row width, not the logical p_dim*num_chunks.
# A `dst=` that is a sub-tile of a wider slot (e.g. one tile of an
# alloc_blocks rotating buffer) has a physical width > logical -- the partition
# stride must follow the physical width or the compiler rejects the AP. Captures
# the emitted sbuf_pattern by stubbing _issue_dma_transpose (no device).
# ============================================================================


@pytest_marks(["neurotile"])
class TestDmaTransposeDstPartitionStride:
    """level-0 of the emitted sbuf_pattern == physical row width of `dst`."""

    def _capture(self, monkeypatch):
        patterns = []

        def spy(source, offset, hbm_pattern, sbuf, sbuf_pattern, sbuf_offset, indirect, **kw):
            patterns.append(sbuf_pattern)

        monkeypatch.setattr(HBMLayout, "_issue_dma_transpose", staticmethod(spy))
        return patterns

    @pytest.mark.fast
    def test_subtile_dst_uses_physical_partition_stride(self, monkeypatch):
        # p_dim=128, f_dim=128 -> 1 chunk, logical transposed shape (128, 128).
        # dst is one tile of a (128, 512) slot -> physical row width 512.
        patterns = self._capture(monkeypatch)
        dst = MockTensor((128, 128), storage_shape=(128, 512))
        HBMLayout._dma_transpose(
            source="mock",
            offset=0,
            strides=(128, 1),
            remaining=(128, 128),
            dtype="bfloat16",
            dst=dst,
        )
        # level-0 partition stride must be the physical 512, not logical 128.
        assert patterns[0][0][0] == 512, f"got {patterns[0]}"

    @pytest.mark.fast
    def test_standalone_dst_partition_stride_equals_logical(self, monkeypatch):
        # Standalone dst (physical == logical): stride stays p_dim*num_chunks=128.
        patterns = self._capture(monkeypatch)
        dst = MockTensor((128, 128), storage_shape=(128, 128))
        HBMLayout._dma_transpose(
            source="mock",
            offset=0,
            strides=(128, 1),
            remaining=(128, 128),
            dtype="bfloat16",
            dst=dst,
        )
        assert patterns[0][0][0] == 128, f"got {patterns[0]}"

    @pytest.mark.fast
    def test_subtile_dst_tiled_multichunk_partition_stride(self, monkeypatch):
        # F=256 -> 2 chunks; logical transposed (128, 256). Sub-tile of a (128,512)
        # slot -> level-0 stride must be the physical 512 on the full-chunk batch.
        patterns = self._capture(monkeypatch)
        dst = MockTensor((128, 256), storage_shape=(128, 512))
        HBMLayout._dma_transpose(
            source="mock",
            offset=0,
            strides=(256, 1),
            remaining=(128, 256),
            dtype="bfloat16",
            dst=dst,
        )
        assert patterns[0][0][0] == 512, f"got {patterns[0]}"


# ============================================================================
# reachable_dim_extent: shared offset-clamp rule, mirrored across layouts
# ============================================================================


class _FakeGrid:
    """Minimal Grid stand-in carrying just the fields reachable_dim_extent reads."""

    def __init__(self, element_shape, remaining, remainder_dims=()):
        self.element_shape = tuple(element_shape)
        self.remaining = tuple(remaining)
        self.remainder_dims = tuple(remainder_dims)


class _FakeLayout:
    """Layout stand-in returning a fixed per-dim offset for dim_offset_elements."""

    def __init__(self, offsets):
        self._offsets = tuple(offsets)

    def dim_offset_elements(self, dim, element_shape):
        return self._offsets[dim]


@pytest_marks(["neurotile"])
class TestReachableDimExtent:
    """``reachable_dim_extent`` is the single offset-clamp rule shared by
    every layout's ``dim_addressable`` and by ``NDSlice.element_shape``.

    It subtracts the origin offset on a dim sitting on a partial trailing
    tile / block, and leaves a slice-narrowed dim untouched (where the source
    extent already equals the walk and the offset modulo is meaningless)."""

    @pytest.mark.fast
    def test_partial_block_subtracts_offset(self):
        # Trailing N-block: source extent 1792 still carried, walk 1024, origin
        # at 1024 -> only 1792 - 1024 = 768 reachable. element_shape > remaining
        # engages the clamp even though the dim is not in remainder_dims.
        g = _FakeGrid(element_shape=(256, 1792), remaining=(256, 1024))
        lay = _FakeLayout(offsets=(0, 1024))
        assert reachable_dim_extent(g, lay, 1) == 768

    @pytest.mark.fast
    def test_flagged_remainder_subtracts_offset(self):
        # remainder_dims pre-flagged (tile walk overshoot): subtract offset.
        g = _FakeGrid(element_shape=(128, 1792), remaining=(128, 256), remainder_dims=(1,))
        lay = _FakeLayout(offsets=(0, 1536))
        assert reachable_dim_extent(g, lay, 1) == 256

    @pytest.mark.fast
    def test_full_walk_not_clamped(self):
        # offset 0, element_shape > remaining -> subtract 0 (no-op): full block.
        g = _FakeGrid(element_shape=(256, 1792), remaining=(256, 1024))
        lay = _FakeLayout(offsets=(0, 0))
        assert reachable_dim_extent(g, lay, 1) == 1792

    @pytest.mark.fast
    def test_slice_narrowed_dim_not_clamped(self):
        # A slice narrows element_shape to match the walk (==), so the clamp
        # does NOT engage -- the offset modulo would be meaningless garbage.
        g = _FakeGrid(element_shape=(128, 4, 104), remaining=(128, 4, 104))
        lay = _FakeLayout(offsets=(0, 3, 0))  # 3 is the garbage modulo
        assert reachable_dim_extent(g, lay, 1) == 4


@pytest_marks(["neurotile"])
class TestDimAddressableMirroredAcrossLayouts:
    """Every layout exposes ``dim_addressable`` with identical semantics --
    a uniform interface so ``NDSlice`` clamps polymorphically without knowing
    the layout type. Each delegates to the shared ``reachable_dim_extent``."""

    @pytest.mark.fast
    def test_hbm_partial_block(self):
        from nkilib_src.nkilib.experimental.neurotile.core.factories import blocks

        trailing = blocks(MockTensor(512, 1792), tile_size=(128, 512), block_size=(2, 2))[0, 1]
        assert trailing._layout.dim_addressable(1, trailing._grid) == 768

    @pytest.mark.fast
    def test_all_layouts_expose_dim_addressable(self):
        # The method exists on all three layout classes (uniform interface).
        from nkilib_src.nkilib.experimental.neurotile.core.layout_psum import PSUMLayout
        from nkilib_src.nkilib.experimental.neurotile.core.layout_sbuf import SBUFLayout

        assert hasattr(HBMLayout, "dim_addressable")
        assert hasattr(SBUFLayout, "dim_addressable")
        assert hasattr(PSUMLayout, "dim_addressable")


# ============================================================================
# APEmitter: no neurotile-imposed AP-depth cap (compiler gates depth)
# ============================================================================


@pytest_marks(["neurotile"])
class TestAPEmitterNoLevelCap:
    """``APEmitter.emit`` no longer caps the access-pattern level count.

    Contiguous levels fold away in the merge pass (so high-rank *contiguous*
    views collapse to one level), and a genuinely deep *strided* pattern is
    validated by the compiler when the AP is consumed -- not gated by a
    neurotile assert. A >4-level strided pattern must therefore emit all its
    levels rather than raise. (Verified on trn2: a 5-level strided gather
    compiles and moves correct data; deeper patterns fail at compile time.)
    """

    @pytest.mark.fast
    def test_strided_5_level_pattern_emits_without_capping(self):
        from nkilib_src.nkilib.experimental.neurotile.core.ap_emitter import APEmitter
        from nkilib_src.nkilib.experimental.neurotile.core.axis import Axis, AxisLabel

        # 5 dims, each a non-mergeable strided gather (step > inner walk), so the
        # merge pass cannot fold them: source (4,)*5, pick indices {0,2} per dim.
        contig = [256, 64, 16, 4, 1]
        axes = tuple(
            Axis(count=2, step=2, dim=d, label=AxisLabel.ELEM if d > 0 else AxisLabel.PARTITION) for d in range(5)
        )
        levels = APEmitter.emit(axes, tuple(contig), (4, 4, 4, 4, 4))
        # All 5 strided levels survive -- no cap, no raise.
        assert len(levels) == 5, levels

    @pytest.mark.fast
    def test_max_ap_levels_constant_removed(self):
        # The neurotile-imposed cap is gone; the constant no longer exists.
        from nkilib_src.nkilib.experimental.neurotile.core import _helpers

        assert not hasattr(_helpers, "MAX_AP_LEVELS")
