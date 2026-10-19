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

"""Tests for the SBUF stack/heap BufferManager (allocator.py).

Two groups of tests:

  1. Host-side (pure-Python) tests that exercise the address-arithmetic and
     book-keeping logic which does not require a backend: dtype sizing, scope
     open/increment/close, stats tracking, and the defensive guards
     (OOM / no-open-scope / heap-underflow / auto-alloc-mode) which all raise
     before any ``nl.ndarray`` is constructed.

  2. Simulator tests that drive small ``@nki.jit`` kernels through
     ``UnitTestFramework`` so the normal allocation paths (which build
     ``nl.ndarray`` and therefore need a backend) actually execute: stack vs
     heap dispatch, HBM dispatch, explicit vs default alignment, manual vs
     auto-allocation addressing, and ``pop_heap`` of a live tensor.
"""

from typing import final

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.utils.allocator import (
    BufferManager,
    SbufManager,
    align_to,
    create_auto_alloc_manager,
    num_elts,
    sizeinbytes,
)
from nkilib_src.nkilib.core.utils.logging import get_logger

from test.utils.common_dataclasses import CompilerArgs
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

_SB = 192 * 1024  # per-partition SBUF budget used for manual-address managers


# =============================================================================
# Host-side (pure-Python) tests
# =============================================================================


@pytest.mark.fast
@pytest.mark.parametrize(
    "dtype,expected",
    [
        (nl.float32, 4),
        (nl.int32, 4),
        (nl.uint32, 4),
        (nl.bfloat16, 2),
        (nl.float16, 2),
        (nl.uint16, 2),
        (nl.int16, 2),
        (nl.float4_e2m1fn_x4, 2),
        (nl.int8, 1),
        (nl.uint8, 1),
        (nl.float8_e4m3fn_x4, 4),
        (nl.float8_e5m2_x4, 4),
    ],
)
def test_sizeinbytes_known_dtypes(dtype, expected):
    """Every recognised dtype returns its byte width (covers each elif arm)."""
    assert sizeinbytes(dtype) == expected


@pytest.mark.fast
def test_sizeinbytes_float8_string_variants():
    """The float8 string membership arm returns 1 byte."""
    assert sizeinbytes("float8_e4m3fn") == 1
    assert sizeinbytes("float8_e5m2") == 1


@pytest.mark.fast
def test_sizeinbytes_unknown_dtype_raises():
    """Unknown dtype hits the terminal kernel_assert guard."""
    with pytest.raises(AssertionError):
        sizeinbytes("not_a_real_dtype")


@pytest.mark.fast
def test_num_elts_multidim_and_empty():
    """num_elts multiplies free dims; empty shape returns the identity."""
    assert num_elts((2, 3, 4)) == 24
    assert num_elts(()) == 1


@pytest.mark.fast
def test_align_to_rounds_up():
    assert align_to(5, 4) == 8
    assert align_to(8, 4) == 8


@pytest.mark.fast
def test_init_default_logger_and_explicit_logger():
    """__init__ creates a default logger when none supplied, else keeps ours."""
    bm_default = BufferManager(0, _SB)
    assert bm_default.logger is not None

    my_logger = get_logger("custom")
    bm_explicit = BufferManager(0, _SB, logger=my_logger)
    assert bm_explicit.logger is my_logger


@pytest.mark.fast
def test_get_prefixed_name():
    """Prefix is applied to names but None passes through unchanged."""
    bm = BufferManager(0, _SB)
    bm.set_name_prefix("k_")
    assert bm.get_name_prefix() == "k_"
    assert bm._get_prefixed_name("t") == "k_t"
    assert bm._get_prefixed_name(None) is None


@pytest.mark.fast
def test_update_stats_tracks_new_maxima_only():
    """_update_stats raises maxima on growth and leaves them on shrink."""
    bm = BufferManager(0, _SB)
    # Grow stack and heap usage -> all three maxima update (true arcs).
    bm.stack_curr_addr = 100
    bm.heap_curr_addr = _SB - 200
    bm._update_stats()
    assert bm.max_stack_usage == 100
    assert bm.max_heap_usage == 200
    assert bm.max_combined_usage == 300
    # Shrink usage -> none update (false / fall-through arcs).
    bm.stack_curr_addr = 50
    bm.heap_curr_addr = _SB - 100
    bm._update_stats()
    assert bm.max_stack_usage == 100
    assert bm.max_heap_usage == 200
    assert bm.max_combined_usage == 300


@pytest.mark.fast
def test_increment_section_wrap_and_advance_single_scope():
    """increment_section advances within interleave, then wraps to the base."""
    bm = BufferManager(0, _SB)
    bm.open_scope(interleave_degree=2, name="s")
    bm.scopes[-1].min_independent_addr = 256
    bm.increment_section()  # advance: use min_independent_addr
    assert bm.scopes[-1].cur_section_id == 1
    assert bm.stack_curr_addr == 256
    bm.increment_section()  # wrap: reset to starting_addr
    assert bm.scopes[-1].cur_section_id == 0
    assert bm.stack_curr_addr == 0


@pytest.mark.fast
def test_increment_section_wrap_updates_parent_min_independent_addr():
    """Wrapping a nested scope propagates its min_independent_addr to parent."""
    bm = BufferManager(0, _SB)
    bm.open_scope(interleave_degree=1, name="outer")
    bm.open_scope(interleave_degree=1, name="inner")
    bm.scopes[-1].min_independent_addr = 500
    bm.increment_section()  # interleave 1 -> immediate wrap, parent gets 500
    assert bm.scopes[-2].min_independent_addr == 500


@pytest.mark.fast
def test_close_scope_parent_update_and_final_flush():
    """Closing a nested scope updates the parent; closing the last flushes."""
    bm = BufferManager(0, _SB)
    bm.open_scope(name="outer")
    bm.open_scope(name="inner")
    bm.scopes[-1].min_independent_addr = 700
    bm.close_scope()  # parent picks up child's min_independent_addr
    assert len(bm.scopes) == 1
    assert bm.scopes[-1].min_independent_addr == 700
    bm.close_scope()  # last scope closed -> auto-flush / stats path
    assert bm.scopes == []


@pytest.mark.fast
def test_space_accounting_helpers():
    bm = BufferManager(10, 10 + _SB)
    assert bm.get_total_space() == _SB
    assert bm.get_free_space() == _SB
    assert bm.get_used_space() == 0


@pytest.mark.fast
def test_alloc_stack_out_of_memory_guard():
    """alloc_stack aborts (before building a tensor) when it would overflow."""
    bm = BufferManager(0, 64)
    bm.open_scope()
    with pytest.raises(AssertionError):
        bm.alloc_stack((128, 1000), nl.float32)


@pytest.mark.fast
def test_alloc_stack_without_open_scope_guard():
    """alloc_stack aborts when there is no open scope."""
    bm = BufferManager(0, _SB)
    with pytest.raises(AssertionError):
        bm.alloc_stack((128, 4), nl.float32)


@pytest.mark.fast
def test_alloc_heap_out_of_memory_guard():
    """alloc_heap aborts (before building a tensor) when it would overflow."""
    bm = BufferManager(0, 64)
    with pytest.raises(AssertionError):
        bm.alloc_heap((128, 1000), nl.float32)


@pytest.mark.fast
def test_pop_heap_underflow_guard():
    """Popping an empty heap aborts."""
    bm = BufferManager(0, _SB)
    with pytest.raises(AssertionError):
        bm.pop_heap()


@pytest.mark.fast
def test_addr_accessors_manual_mode_and_auto_mode_guard():
    """Address accessors return in manual mode and abort in auto-alloc mode."""
    bm = BufferManager(0, _SB, use_auto_alloc=False)
    assert bm.get_stack_curr_addr() == 0
    assert bm.get_heap_curr_addr() == _SB
    bm.align_stack_curr_addr(32)  # no-op at 0, exercises the manual path
    assert bm.stack_curr_addr == 0

    auto = BufferManager(0, _SB, use_auto_alloc=True)
    assert auto.is_auto_alloc()
    with pytest.raises(AssertionError):
        auto.get_stack_curr_addr()
    with pytest.raises(AssertionError):
        auto.get_heap_curr_addr()
    with pytest.raises(AssertionError):
        auto.align_stack_curr_addr()


@pytest.mark.fast
def test_default_alloc_flags_and_alias():
    bm_stack = BufferManager(0, _SB, default_stack_alloc=True)
    assert bm_stack.is_default_stack_alloc() and not bm_stack.is_default_heap_alloc()
    bm_heap = BufferManager(0, _SB, default_stack_alloc=False)
    assert bm_heap.is_default_heap_alloc() and not bm_heap.is_default_stack_alloc()
    # Backward-compat alias points at the same class.
    assert SbufManager is BufferManager


# =============================================================================
# Simulator tests (normal allocation paths that build nl.ndarray)
# =============================================================================


@nki.jit
def kernel_manual_alloc(input_data, output):
    """Exercise manual-address stack + heap + HBM allocation and pop_heap.

    Data flows input -> stack(s) -> stack(s2) -> heap(h) -> heap(h2) -> output,
    so the result is an identity copy of the input.
    """
    sbm = BufferManager(0, _SB, use_auto_alloc=False, default_stack_alloc=True)
    sbm.set_name_prefix("k_")
    sbm.open_scope(name="outer")

    # alloc() dispatch -> stack (default_stack), default alignment, manual address.
    s = sbm.alloc(input_data.shape, input_data.dtype, buffer=nl.sbuf, name="s")
    nisa.dma_copy(dst=s, src=input_data)

    # Explicit alignment on the stack.
    s2 = sbm.alloc_stack(input_data.shape, input_data.dtype, name="s2", align=64)
    nisa.tensor_copy(dst=s2, src=s)

    # alloc() dispatch -> HBM branch.
    _tmp_hbm = sbm.alloc(input_data.shape, input_data.dtype, buffer=nl.hbm, name="tmp_hbm")

    # Heap allocation: default alignment then explicit alignment.
    h = sbm.alloc_heap(input_data.shape, input_data.dtype, name="h")
    nisa.tensor_copy(dst=h, src=s2)
    h2 = sbm.alloc_heap(input_data.shape, input_data.dtype, name="h2", align=64)
    nisa.tensor_copy(dst=h2, src=h)

    nisa.dma_copy(dst=output, src=h2)
    sbm.pop_heap()  # frees h2 (already stored) -> live-tensor pop path
    sbm.close_scope()
    return output


@nki.jit
def kernel_auto_alloc(input_data, output):
    """Exercise auto-allocation addressing and the default-heap dispatch."""
    sbm = create_auto_alloc_manager()
    sbm.set_auto_alloc(True)
    sbm.default_stack_alloc = False  # make alloc() dispatch to the heap
    sbm.open_scope(name="auto")

    # alloc() dispatch -> heap (default_stack False), auto address (no explicit addr).
    hh = sbm.alloc(input_data.shape, input_data.dtype, buffer=nl.sbuf, name="hh")
    nisa.dma_copy(dst=hh, src=input_data)

    # Stack allocation in auto mode.
    st = sbm.alloc_stack(input_data.shape, input_data.dtype, name="st")
    nisa.tensor_copy(dst=st, src=hh)

    nisa.dma_copy(dst=output, src=st)
    sbm.close_scope()
    return output


def _identity_ref(input_data, output):
    import torch

    return {"output": torch.clone(input_data)}


@final
@pytest_test_metadata(name="Allocator")
@pytest_marks(["allocator"])
class TestAllocatorSimulation:
    """Simulator tests that run the real allocation paths inside kernels."""

    def _run(self, kernel, test_manager, platform_target):
        def input_generator(test_config):
            np.random.seed(42)
            input_data = np.random.randn(128, 256).astype(np.float32)
            return {
                "input_data": input_data,
                "output.must_alias_input": np.zeros_like(input_data),
            }

        def output_tensors(kernel_input):
            return {"output": np.zeros((128, 256), dtype=np.float32)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref_wrapper(_identity_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
        )

    @pytest.mark.fast
    def test_manual_stack_heap_hbm_allocation(self, test_manager, platform_target):
        """Manual-address stack/heap/HBM allocation with pop_heap, identity copy."""
        self._run(kernel_manual_alloc, test_manager, platform_target)

    @pytest.mark.fast
    def test_auto_allocation_and_default_heap(self, test_manager, platform_target):
        """Auto-allocation addressing with default-heap dispatch, identity copy."""
        self._run(kernel_auto_alloc, test_manager, platform_target)
