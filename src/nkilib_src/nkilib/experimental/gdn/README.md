# Gated DeltaNet (GDN) kernels

Gated DeltaNet linear attention for hybrid attention models: chunked **prefill**,
single-token **decode**, a fused decode **megakernel**, and the depthwise **conv1d** that
feeds them.

| Entry point | File | What it is |
|---|---|---|
| `gdn_cte` | `gdn_cte.py` | Prefill forward. Dispatches to one of two intra-chunk inverse algorithms; **the family's only prefill forward entry point**. |
| `gdn_cte_bwd` | `gdn_cte_bwd.py` | Prefill backward: reverse-mode VJP of the schur forward, returning dQ/dK/dV/dg_log/dbeta/dS0. Schur only — there is no neumann backward. |
| `gdn_tkg` | `gdn_tkg.py` | Single-token decode recurrence. |
| `gdn_block_tkg` | `gdn_block_tkg.py` | Fused decode megakernel (in_proj + conv1d + GQA + recurrence). |
| `gdn_conv1d_prefill` / `gdn_conv1d_decode` | `gdn_conv1d.py` | Short depthwise causal conv1d on the GDN input projection. |

The prefill FORWARD is ONE kernel with two candidate algorithms, selected by `algorithm=`:
`gdn_cte_schur.py` (default) and `gdn_cte_neumann.py`. Those two modules are *not* entry
points -- they hold plain traced functions that `gdn_cte` inlines, they are not
`@nki.jit`-decorated, and they are not exported from the package, so the family presents a
single interface with a single CPU reference,
`gdn_cte_torch.py::gdn_cte_torch_ref`. Both share `gdn_cte_utils.py` (constants,
host-built mask pack, tile helpers).

The prefill BACKWARD, `gdn_cte_bwd.py`, is a separate `@nki.jit` entry point covering the
schur forward only. Its reference is `gdn_cte_bwd_torch.py`, which holds both an autograd
oracle (the ground truth) and a hand-derived chunked form, cross-checked against each other.
Note its own limit on sequence length: the backward stashes each chunk's incoming recurrent
state in SBUF, which is linear in S, so `MAX_STATE_STASH_BYTES_PER_PARTITION` caps S at 49152
when `Dv = 128` (proportionally more at smaller `Dv`) — the forward has no such bound. The cap
sits just under the measured trn2 wall, where the stash alone fills SBUF at S = 53248; raising
it further means moving the stash to HBM.

## Prefill: two algorithms, one function

Both algorithms compute the same thing -- the chunked gated delta rule, same math as
FLA/CUDA `chunk_gated_delta_rule` -- from the same inputs, and are validated against the
same per-token fp32 reference. They differ only in how the intra-chunk triangular system
is solved:

| | `gdn_cte_schur` (default) | `gdn_cte_neumann` |
|---|---|---|
| inverse | recursive **Schur complement**: `inv([[P,0],[R,Q]]) = [[P^-1,0],[-Q^-1 R P^-1, Q^-1]]`, block size doubling 1 -> 128, i.e. 7 merge levels. Because every block of one level lands in a *disjoint* region of the same tile, one tile-wide chain resolves a whole level: **12 matmuls where the literal recursion is 127 block operations** | **Neumann series** `(I-A_ii)^-1 = sum_j A_ii^j` on 16x16 diagonal blocks by recursive doubling (exact -- such a block is nilpotent), then block forward substitution |
| forms powers of A? | never, so nothing depends on `‖A‖` | yes, safe only inside a 16x16 block (a row there has ≤15 nonzeros; at 128x128 the powers grow geometrically and silently destroy every digit before they overflow) |
| chunk length | 128 | 64 (forced by the substitution's `O(N^2)` coupling in the sub-block count) |
| S must be a multiple of | 128 | 64 |
| native argument layout | this dispatcher's, directly | its own flat `[B, S, D]` q/k/v + `[B, S]` gate/beta; the dispatcher adapts (see below) |
| GQA head sharing | yes, Hv may exceed Hk | no -- requires Hk == Hv |
| head dims | any `Dk, Dv <= 128`, `Dk != Dv` allowed | any `D <= 128`, but one `D` for q/k/v (`Dk == Dv`) |
| initial recurrent state | optional mutable in/out | none; always starts from zero |
| chunk grouping | `group_size` chunks (default 8) honored at every length -- full groups plus one residual group -- with the merge levels emitted **level-major** across the group | fixed 8 chunks per group, with a partial final group; sub-block work batched onto the 128-partition axis |
| bottleneck at S=12288 bf16 | **none singly**: the three engines that can reach PSUM all sit within 9% of each other at 71-77% busy, and the rest of the wall clock is cross-engine dependency stall | **scalar**, 5951 us of 9566 us wall, with 6.6x the DMA time |

Selecting one:

```python
from nkilib_src.nkilib.experimental.gdn import ALGORITHM_NEUMANN, build_mask_pack, gdn_cte

mask_pack = build_mask_pack()          # host-built; the schur algorithm cannot synthesize it
out, state = gdn_cte(query, key, value, g_log, beta, scale=head_dim**-0.5,
                     mask_pack=mask_pack)

# Continuing a sequence: pass the state and it is updated IN PLACE (aliased to output 1).
out, state = gdn_cte(query, key, value, g_log, beta, scale=head_dim**-0.5,
                     mask_pack=mask_pack, recurrent_state=state)

# The neumann algorithm: no GQA (repeat the key/query heads yourself if Hk < Hv), no
# initial state, and NO `mask_pack` -- it builds its constants on device, so passing one
# would add an unused 720 KB input to the NEFF.
out, state = gdn_cte(query, key, value, g_log, beta, scale=head_dim**-0.5,
                     algorithm=ALGORITHM_NEUMANN)
```

`algorithm` is a trace-time constant, so each choice compiles to its own NEFF. **Prefer
the default** unless the sequence length is a multiple of 64 but not 128, which only the
neumann algorithm accepts.

### Interface

The dispatcher's interface, which `gdn_cte_schur` takes directly and which the dispatcher
adapts onto `gdn_cte_neumann` (see above). `B` batch, `Hk` key/query heads, `Hv` value heads
(a multiple of `Hk`; value head `h` reads key head `h // (Hv//Hk)`), `Dk` the query/key head
dim and `Dv` the value head dim (both `<= 128`, the partition bound; `Dq == Dk` is required
since they contract, and `Dk == Dv` additionally on the neumann path). 128 is the efficient
value -- a `(Dk, CHUNK)` tile then fills the partition axis -- and is what the measurements
below use.
`S` sequence length.

| Tensor | Shape | Notes |
|---|---|---|
| `query` | `[B, Hk, S, Dk]` | **TOKEN-major** -- the natural projection output. L2-normalized, NOT pre-scaled. **Its dtype is the compute dtype** (matmul operands, q/k/v tiles, `out`). |
| `key` | `[B, Hk, S, Dk]` | Token-major, L2-normalized, same dtype as `query`. |
| `value` | `[B, Hv, S, Dv]` | Same dtype as `query`. |
| `g_log` | `[B, Hv, S]` | fp32 per-token log decay, negative. |
| `beta` | `[B, Hv, S]` | fp32 per-token update strength in (0, 1). |
| `mask_pack` | `[N_MASKS, 128, 128]` or omitted | fp32, from `build_mask_pack()`. 10 planes, ~650 KB. **Keyword, and required only by `"schur"`** -- omit it for `"neumann"`, which builds its constants on device, so an unused pack would otherwise sit in that NEFF. |
| `scale` | float | Query scaling (typically `1/sqrt(Dk)`), applied **on device** at no cost. |
| `recurrent_state` | `[B, Hv, Dk, Dv]` or omitted | fp32 **optional mutable in/out**: supplied, it is aliased to output 1 and updated in place, and its contents are the initial state; omitted, the kernel allocates it and starts from zero with no load emitted. `"neumann"` accepts ONLY the omitted form. |
| → `out` | `[B, Hv, S, Dv]` | Compute dtype, head-major, un-normalized (no fused epilogue). |
| → `recurrent_state` | `[B, Hv, Dk, Dv]` | Always fp32 -- it accumulates across the whole sequence, the one place a narrow mantissa would compound. |

## Device time

Every configuration below was scored against the per-token fp32 reference in `gdn_cte_torch.py`
BY THE RUN THAT TIMED IT, so these tables report time only. Worst case over the twelve bf16 rows,
both algorithms: `6.4e-03` on `out` and `5.4e-03` on the state; in fp32, `6.5e-06` and `2.8e-06`,
against the test suite's `2e-2` tolerance.

Nine of these shapes are also device cases in
`test/integration/nkilib/experimental/gdn/test_gdn_cte.py` -- both algorithms on `32,1,1,128`,
`1,2,4,768`, `1,1,1,8192`, `1,2,4,12288`, `32,1,1,2176`, `32,1,1,2304` and `1,1,1,8320`, schur
alone on `1,1,1,128` and `1,1,1,640`, and neumann alone on `1,1,1,64`. The `256`, GQA-`256`, `896`
and `32,1,1,64` rows and both fp32 rows are BENCHMARK-ONLY: the suite was trimmed to a minimal
branch-covering set, so it no longer exercises fp32 or a head dim other than 128 on device.

Device metrics come from a `neuron-explorer` trace of the third of three executions,
captured with the real inputs and with `recurrent_state` supplied (the aliased in/out
path). `speedup` is the neumann `total_time` over the schur `total_time`; MFU and MBU are
the profiler's estimated matmul-FLOP and memory-bandwidth utilizations.

Measured on trn2 (`trn2.48xlarge`, LNC=1, one NeuronCore), nki 0.5.0, neuronx-cc
2.26.6360.0, neuron-explorer 2.31.0, bf16 q/k/v with fp32 `g_log`/`beta`/state, zero
initial state, `group_size` at its default of 8.

The neumann rows also carry the layout adaptation described above: its key/query heads are
duplicated on the HOST before the call (so the GQA rows feed it Hv key heads), which is
outside everything measured here -- it happens before the NEFF is even compiled.

| B | Hk | Hv | S | schur | neumann | speedup | schur MFU | neumann MFU | schur MBU | neumann MBU |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 128 | 33.7 us | 61.3 us | **1.82x** | 3.64% | 0.55% | 3.94% | 1.72% |
| 32 | 1 | 1 | 128 | 524.8 us | 1401.3 us | **2.67x** | 7.48% | 0.77% | 2.69% | 2.41% |
| 1 | 1 | 1 | 256 | 35.4 us | 82.4 us | **2.33x** | 6.94% | 1.20% | 4.41% | 2.45% |
| 1 | 2 | 4 | 256 | 93.9 us | 270.8 us | **2.88x** | 10.45% | 1.46% | 3.43% | 2.98% |
| 1 | 1 | 1 | 640 | 57.0 us | 173.8 us | **3.05x** | 10.77% | 2.46% | 3.95% | 2.82% |
| 1 | 2 | 4 | 768 | 192.7 us | 688.0 us | **3.57x** | 15.28% | 2.87% | 3.30% | 3.41% |
| 1 | 2 | 4 | 896 | 221.9 us | 757.1 us | **3.41x** | 15.48% | 3.21% | 3.21% | 3.61% |
| 32 | 1 | 1 | 2176 | 3869.9 us | 14104.3 us | **3.64x** | 17.24% | 3.66% | 3.41% | 3.73% |
| 32 | 1 | 1 | 2304 | 3975.8 us | 14984.8 us | **3.77x** | 17.77% | 3.58% | 3.51% | 3.72% |
| 1 | 1 | 1 | 8192 | **441.9 us** | 1614.6 us | **3.65x** | 17.76% | 3.91% | 3.58% | 3.82% |
| 1 | 1 | 1 | 8320 | 462.1 us | 1649.7 us | **3.57x** | 17.25% | 3.85% | 3.48% | 3.80% |
| 1 | 2 | 4 | 12288 | **2555.8 us** | 9565.7 us | **3.74x** | 18.43% | 3.96% | 3.52% | 3.87% |

Sequence lengths only `gdn_cte_neumann` accepts (a multiple of 64, not of 128):

| B | Hk | Hv | S | neumann | neumann MFU | neumann MBU |
|---|---|---|---:|---:|---:|---:|
| 1 | 1 | 1 | 64 | 53.5 us | 0.26% | 1.07% |
| 32 | 1 | 1 | 64 | 1126.9 us | 0.40% | 1.63% |

Same kernels with an **fp32 compute dtype** (pass fp32 `query`/`key`/`value`; nothing else
changes):

| B | Hk | Hv | S | schur | neumann | speedup | schur MFU | neumann MFU | schur MBU | neumann MBU |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 8192 | 537.2 us | 1762.7 us | **3.28x** | 50.83% | 14.33% | 5.67% | 4.33% |
| 1 | 2 | 4 | 12288 | 3170.0 us | 10443.2 us † | **3.29x** | 51.69% | 14.52% | 5.61% | 4.38% |

† `gdn_cte_neumann` is NON-DETERMINISTIC on this configuration: two identical runs of the
same NEFF measured 6.5e-06 and 7.1e-05 error on `out` (the state is stable at 2.84e-06), and
`total_time` varies by ~1%. It reproduces only with more than one (batch x head) -- single-head
fp32 is bit-stable across runs, as is schur at the same 4-head shape.

### Reproducing

`test/integration/nkilib/experimental/gdn/profile_gdn_cte.py` produced all three tables. It
needs a Trainium host with a free NeuronCore and `neuron-explorer` on PATH:

```bash
# Everything, serially (compiles dominate; expect a long run at S=12288)
NEURON_RT_VISIBLE_CORES=<cores> python test/integration/nkilib/experimental/gdn/profile_gdn_cte.py

# One shape, one algorithm, one precision
NEURON_RT_VISIBLE_CORES=<cores> python .../profile_gdn_cte.py \
    --shapes 1,2,4,12288 --algorithms schur --dtypes bfloat16

# The neumann-only rows, and the fp32 rows
NEURON_RT_VISIBLE_CORES=<cores> python .../profile_gdn_cte.py \
    --shapes "1,1,1,64 32,1,1,64" --algorithms neumann
NEURON_RT_VISIBLE_CORES=<cores> python .../profile_gdn_cte.py \
    --shapes "1,1,1,8192 1,2,4,12288" --dtypes float32
```

The `--phase run` / `--phase profile` split exists so the compiles can be run concurrently
(they are CPU-bound) while profiling runs serially on an otherwise idle device --
concurrent profiling runs contend for HBM bandwidth and inflate every `total_time`. All
numbers above were taken that way, and `--phase table` re-emits the tables from the result
JSONs without touching the device.
