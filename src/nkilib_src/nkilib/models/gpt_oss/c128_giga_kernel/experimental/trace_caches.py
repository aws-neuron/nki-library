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

"""One reset for all trace-scoped module state in this vendored kernel.

This kernel keeps three kinds of state in module globals that are valid only WITHIN a single
trace:

* **Prefetch hand-off FIFOs.** A producer enqueues an SBUF buffer and a later consumer in the
  same kernel pops it. Both consumers pop UNCONDITIONALLY whenever the FIFO is non-empty
  (``down_projection_mx.py`` ``_DOWN_PREFETCH_FIFO.pop(0)``, ``gate_up_projection_mx.py``
  ``_fifo.pop(0)``), so a leftover entry is silently consumed by the next trace.
* **Block-table memos**, keyed on ``id()`` of trace-local tensors -- values CPython recycles.
* **The DMA-name scope**, which qualifies every ``dma_name()`` and carries the
  duplicate-detection set.

Left populated, the next kernel traced in the same process misuses the previous trace's state.
Three failures were observed, all invisible in a single-trace profile run and all requiring two
kernels in one process:

* **Wrong weights.** A stale prefetch buffer was consumed as this layer's weight, regressing the
  golden gate to cosine 0.887 while profile runs -- which trace once per process -- looked clean.
* **Cross-trace collective operands.** ``_MOE_OUT_SBUF_CAPTURE.pop(0)`` returned a previous
  trace's buffer, so ``ncc.reduce_scatter`` got one operand from each trace and nki asserted
  ``All src & dst tensors must have the same buffer type``. Note that assert text ALSO belongs to
  the unrelated collective buffer-class rule (see ``_COLLECTIVE_HBM``), which is a standing trap
  when reading a failure log.
* **Duplicate DMA names.** A leaked scope made an unrelated kernel emit names and trip
  ``dma_name()``'s duplicate guard -- seen as ``duplicate DMA instruction name
  '_prequant_gamma_load'``. The leading underscore and missing layer index are the tell that the
  scope came from an empty layer tag, i.e. that it leaked rather than being set deliberately.

**Every OUTERMOST ``@nki.jit`` entry point in this package must call this first.** Resetting on
ENTRY rather than on exit is deliberate: a trace that raises part-way through never reaches its
own cleanup, and an entry point that never sets state can still inherit it.

Two functions deliberately do NOT reset: ``attention_block_tkg`` and ``moe_block_tkg`` are
``@nki.jit`` but are also called as sub-kernels by the decode layer, so clearing there would wipe
the enclosing kernel's prefetch FIFOs mid-trace.

The imports are function-local on purpose: this module is imported BY the entry points, and one
cache lives in ``transformer/attention_block_tkg.py``, which is itself an entry point. Deferring
the imports to call time breaks that cycle.
"""


def reset_trace_state() -> None:
    """Clear all trace-scoped module state. Call on entry to each outermost ``@nki.jit`` kernel."""
    from ..core.attention.attention_tkg import _ABT_CACHE, _ABT_V_REARRANGED_CACHE
    from ..core.moe.moe_tkg.down_projection_mx import (
        _DOWN_BIAS_PREFETCH_FIFO,
        _DOWN_PREFETCH_FIFO,
        _MOE_OUT_SBUF_CAPTURE,
    )
    from ..core.moe.moe_tkg.gate_up_projection_mx import _GATE_UP_PREFETCH_FIFO
    from ..core.utils.dma_names import set_dma_name_scope
    from .transformer.attention_block_tkg import _ABT_FOLD_CACHE, _CONST_IOTA_CACHE

    # Prefetch hand-off FIFOs (producer and consumer are both inside one trace).
    _GATE_UP_PREFETCH_FIFO[0].clear()
    _GATE_UP_PREFETCH_FIFO[1].clear()
    _DOWN_PREFETCH_FIFO.clear()
    _DOWN_BIAS_PREFETCH_FIFO.clear()
    from ..core.qkv import qkv_tkg as _qkv_mod
    from ..core.subkernels import rmsnorm_tkg as _rms_mod

    _qkv_mod._PIN_AFTER[0] = None
    _rms_mod._PIN_AFTER[0] = None
    _MOE_OUT_SBUF_CAPTURE.clear()
    # Memos keyed on id() of trace-local tensors.
    _ABT_CACHE.clear()
    _ABT_V_REARRANGED_CACHE.clear()
    _ABT_FOLD_CACHE.clear()
    # Cached iota tiles are SBUF tensors belonging to THIS trace.
    _CONST_IOTA_CACHE.clear()
    # Turn DMA naming off and drop the duplicate-detection set. A kernel that wants names re-scopes
    # immediately after this call; one that does not stays inert even though the call sites naming
    # uses live in shared sub-kernels.
    set_dma_name_scope(None)
