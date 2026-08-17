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

"""RNG kernels for GPSIMD engine state management and random number generation on Trainium2."""

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import assert_shape

NUM_LANES = 128
NUM_RNG_SEEDS = 6
DTYPE_SIZE_INT32 = 4


def _set_per_lane_rng_state(base_state):
    """Seed each of the 128 GPSIMD lanes with a distinct PRNG state (in-place helper).

    nisa.rand_set_state seeds each partition's PRNG from the corresponding partition of
    src_seeds, so unless the 128 lanes hold different seeds they emit identical streams
    (a 128-periodic sequence). Given a [NUM_LANES, NUM_RNG_SEEDS] uint32 ``base_state``
    tile already in SBUF, this adds a per-partition offset equal to the lane index and
    writes the result back to the GPSIMD engine, producing 128 independent streams.
    Lane 0 keeps the original base seed.

    This is a plain (non-@nki.jit) helper that emits instructions inline into the calling
    kernel; it is shared by set_rng_state_gpsimd and generate_random_fast so the per-lane
    seeding policy lives in one place.

    Args:
        base_state (nl.ndarray): [NUM_LANES, NUM_RNG_SEEDS] uint32 SBUF tile holding the
            base seed in each partition (typically a broadcast of one seed, or the value
            read back from nisa.rand_get_state).
    """
    # iota(dst, pattern, offset, channel_multiplier): pattern [[0, NUM_RNG_SEEDS]] gives a
    # zero free-step across the 6 seed words; channel_multiplier=1 makes the value equal
    # the partition (lane) index. Materialize the full [128, 6] tile so we can combine
    # with nisa.tensor_tensor, which preserves integer dtypes (nisa.tensor_scalar
    # arithmetic requires float32).
    lane_offset = nl.ndarray(shape=(NUM_LANES, NUM_RNG_SEEDS), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.iota(dst=lane_offset, pattern=[[0, NUM_RNG_SEEDS]], offset=0, channel_multiplier=1)

    per_lane_seeds = nl.ndarray(shape=(NUM_LANES, NUM_RNG_SEEDS), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=per_lane_seeds, data1=base_state, data2=lane_offset, op=nl.add)
    nisa.rand_set_state(src_seeds=per_lane_seeds, engine=nisa.gpsimd_engine)


@nki.jit
def get_rng_state_gpsimd(tensor_state: nl.ndarray):
    """
    Retrieve the current RNG state from the GPSIMD engine.

    Reads all 128 lanes of RNG state from the GPSIMD engine into SBUF,
    then copies only lane 0's seeds to a new output HBM tensor.

    Input shape range is constant [1, NUM_RNG_SEEDS]

    Dimensions:
        L: Number of GPSIMD lanes (128)
        S: Number of RNG seeds per lane (6)

    Args:
        tensor_state (nl.ndarray): [1, NUM_RNG_SEEDS], dtype uint32, HBM tensor
            used only for shape/dtype reference.

    Returns:
        output (nl.ndarray): [1, NUM_RNG_SEEDS], dtype uint32, HBM tensor
            containing the 6 RNG seeds from lane 0.

    Pseudocode:
        state = ndarray(shape=(128, 6), dtype=uint32)  # SBUF buffer
        state = gpsimd_engine.get_rng_state()           # Read all lanes
        output = state[0, 0:6]                          # Copy lane 0 to HBM
    """
    assert_shape(tensor_state, (1, NUM_RNG_SEEDS), "tensor_state")
    output = nl.ndarray(shape=(1, NUM_RNG_SEEDS), dtype=nl.uint32, buffer=nl.shared_hbm)
    state = nl.ndarray(shape=(NUM_LANES, NUM_RNG_SEEDS), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.rand_get_state(dst=state, engine=nisa.gpsimd_engine)
    nisa.dma_copy(dst=output, src=state[0:1, 0:NUM_RNG_SEEDS])
    return output


@nki.jit
def set_rng_state_gpsimd(tensor_state: nl.ndarray):
    """
    Set the RNG state for the GPSIMD engine with a distinct per-lane seed.

    Loads 6 seeds from HBM, broadcasts them across the 128 GPSIMD lanes, then perturbs
    each lane by its partition index (via _set_per_lane_rng_state) so the 128 lanes form
    128 independent PRNG streams. This makes generate_random_fast (which consumes all 128
    lanes) produce non-repeating output even when set_rng_state_gpsimd is the entry point;
    lane 0 keeps the original seed, so generate_random (lane-0 only) is unaffected.

    Input shape range is constant [1, NUM_RNG_SEEDS]

    Dimensions:
        L: Number of GPSIMD lanes (128)
        S: Number of RNG seeds per lane (6)

    Args:
        tensor_state (nl.ndarray): [1, NUM_RNG_SEEDS], dtype uint32, HBM tensor
            containing the 6 seeds to broadcast.

    Returns:
        output (nl.ndarray): [1, NUM_RNG_SEEDS], dtype uint32, HBM tensor
            echoing back the (lane-0) seeds that were set.

    Pseudocode:
        seed = dma_load(tensor_state)                    # Load seeds from HBM
        state = broadcast(seed, shape=(128, 6))          # Broadcast to all lanes
        state[p] += p                                    # Distinct per-lane seed
        gpsimd_engine.set_rng_state(state)               # Write state to engine
    """
    assert_shape(tensor_state, (1, NUM_RNG_SEEDS), "tensor_state")
    output = nl.ndarray(shape=(1, NUM_RNG_SEEDS), dtype=nl.uint32, buffer=nl.shared_hbm)
    seed = nl.ndarray(shape=(1, NUM_RNG_SEEDS), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(dst=seed, src=tensor_state)
    state = nl.broadcast_to(seed, (NUM_LANES, NUM_RNG_SEEDS))
    _set_per_lane_rng_state(state)
    nisa.dma_copy(dst=output, src=seed)
    return output.view(nl.int32)


@nki.jit
def generate_random(output: nl.ndarray, n_elements: int):
    """
    Generate random int32 values, tiling to fit SBUF.

    Generates n_elements random int32 values using the GPSIMD RNG engine
    and writes them to a new output HBM tensor. Uses sequential_range because
    rand carries implicit RNG state across iterations (loop-carried dependency).

    Output shape range should be [1, n_elements] where n_elements is unbounded

    Dimensions:
        N: Number of random elements to generate (n_elements)
        F: Tile size on free dimension, determined by available SBUF

    Args:
        output (nl.ndarray): [1, n_elements], dtype int32, HBM tensor
            to be filled with random values.
        n_elements (int): Number of random int32 values to generate.

    Returns:
        output (nl.ndarray): [1, n_elements], dtype int32, HBM tensor
            filled with random values.

    Notes:
        - Uses sequential_range (not affine_range) due to loop-carried RNG state dependency
        - Remainder tile is handled separately after full tiles

    Pseudocode:
        tile_free_size = total_sbuf_size // sizeof(int32)
        n_full_tiles = n_elements // tile_free_size
        remainder = n_elements % tile_free_size
        for tile_idx in range(n_full_tiles):
            random_buffer = rng(shape=(128, tile_free_size))
            output[0, tile_idx * tile_free_size : (tile_idx+1) * tile_free_size] = random_buffer[0]
        if remainder > 0:
            random_buffer = rng(shape=(128, remainder))
            output[0, n_full_tiles * tile_free_size : ...] = random_buffer[0]
    """

    # Compute tile size from SBUF capacity
    tile_free_size = nl.tile_size.total_available_sbuf_size // DTYPE_SIZE_INT32
    n_full_tiles, remainder = divmod(n_elements, tile_free_size)

    for tile_idx in nl.sequential_range(n_full_tiles):
        offset = tile_idx * tile_free_size
        random_buffer = nl.ndarray([NUM_LANES, tile_free_size], dtype=nl.int32, buffer=nl.sbuf)
        nisa.rng(dst=random_buffer, engine=nisa.engine.gpsimd)
        nisa.dma_copy(dst=output[0:1, offset : offset + tile_free_size], src=random_buffer[0:1, 0:tile_free_size])

    if remainder > 0:
        offset = n_full_tiles * tile_free_size
        random_buffer = nl.ndarray([NUM_LANES, remainder], dtype=nl.int32, buffer=nl.sbuf)
        nisa.rng(dst=random_buffer, engine=nisa.engine.gpsimd)
        nisa.dma_copy(dst=output[0:1, offset : offset + remainder], src=random_buffer[0:1, 0:remainder])
    return output


@nki.jit
def generate_random_fast(output: nl.ndarray, n_elements: int):
    """
    Generate random int32 values using ALL 128 GPSIMD lanes (fast, layout-dependent stream).

    nisa.rng fills all 128 GPSIMD lanes (partitions) in a single instruction. The
    original generate_random() keeps only lane 0 and discards the other 127 lanes, so
    it needs ~128x more serial sequential_range iterations than necessary. This kernel
    instead consumes all 128 partitions per nisa.rng call, yielding ~128x fewer
    iterations and a ~70x end-to-end speedup for large dropout shapes (measured on trn2:
    [1, 2048, 5120] RNG 73.1 ms -> 2.2 ms; full aten::native_dropout 72.1 ms -> 1.05 ms).

    Two pieces make this correct and fast:

      1. Per-lane seeding. nisa.rand_set_state seeds each partition's PRNG from the
         corresponding partition of src_seeds. We read the current state, add a
         per-partition offset (the lane index, via nisa.iota with channel_multiplier=1),
         and write it back so the 128 lanes produce 128 INDEPENDENT streams rather than
         128 identical copies. Without this, every lane would emit the same value.
      2. Full-partition consumption. The per-partition free dim is sized to
         (tile_free_size // NUM_LANES) so a single nisa.rng call produces
         NUM_LANES * lane_free usable elements, DMA-copied as a [128, lane_free] tile
         into a contiguous (partition-major) slice of the flat output.

    Trade-off: because values come from 128 interleaved streams, the value->element
    mapping depends on the output layout / tiling, so this kernel does NOT guarantee a
    layout-independent stream. Use generate_random() where that guarantee is required
    (e.g. tests asserting the same seed yields the same byte stream regardless of how
    the work is split across calls).

    Output shape range should be [1, n_elements] where n_elements is unbounded.

    Dimensions:
        N: Number of random elements to generate (n_elements)
        P: Number of GPSIMD lanes / partitions (NUM_LANES = 128)
        F: Per-partition tile size on free dimension (lane_free)

    Args:
        output (nl.ndarray): [1, n_elements], dtype int32, HBM tensor to be filled.
        n_elements (int): Number of random int32 values to generate.

    Returns:
        output (nl.ndarray): [1, n_elements], dtype int32, HBM tensor filled with random values.

    Notes:
        - Uses sequential_range due to the loop-carried RNG state dependency
        - Each full tile consumes all 128 partitions -> chunk = NUM_LANES * lane_free
        - Remainder (< one full chunk) is handled separately using lane 0 only

    Pseudocode:
        # Seed each lane distinctly so the 128 lanes are independent streams
        base_state = rand_get_state(shape=(128, NUM_RNG_SEEDS))
        lane_offset = iota(shape=(128, NUM_RNG_SEEDS), value=lane_index)  # per-partition
        rand_set_state(base_state + lane_offset)

        lane_free = (total_sbuf_size // sizeof(int32)) // 128
        chunk = 128 * lane_free
        n_full_tiles, remainder = divmod(n_elements, chunk)
        for tile_idx in range(n_full_tiles):
            random_buffer = rng(shape=(128, lane_free))  # all 128 lanes used
            output[0, tile_idx * chunk : (tile_idx + 1) * chunk] = random_buffer.reshape(chunk)
        if remainder > 0:
            random_buffer = rng(shape=(128, remainder))
            output[0, n_full_tiles * chunk : ...] = random_buffer[0]  # lane 0 only
    """

    # Per-lane seeding: re-derive 128 independent streams from the current engine state
    # on every call, so generate_random_fast is correct even when called without a
    # preceding set_rng_state_gpsimd (the common path). Shared with set_rng_state_gpsimd
    # via _set_per_lane_rng_state so the seeding policy lives in one place.
    base_state = nl.ndarray(shape=(NUM_LANES, NUM_RNG_SEEDS), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.rand_get_state(dst=base_state, engine=nisa.gpsimd_engine)
    _set_per_lane_rng_state(base_state)

    # Per-partition free-dim tile size: split the SBUF capacity across all 128 lanes
    # so a single nisa.rng call yields NUM_LANES * lane_free usable elements.
    lane_free = (nl.tile_size.total_available_sbuf_size // DTYPE_SIZE_INT32) // NUM_LANES
    chunk = NUM_LANES * lane_free
    n_full_tiles, remainder = divmod(n_elements, chunk)

    for tile_idx in nl.sequential_range(n_full_tiles):
        offset = tile_idx * chunk
        random_buffer = nl.ndarray([NUM_LANES, lane_free], dtype=nl.int32, buffer=nl.sbuf)
        nisa.rng(dst=random_buffer, engine=nisa.engine.gpsimd)
        # Consume all 128 partitions: DMA the full [128, lane_free] tile into a
        # contiguous (partition-major) view of the flat output.
        out_tile = output[0, offset : offset + chunk].reshape((NUM_LANES, lane_free))
        nisa.dma_copy(dst=out_tile, src=random_buffer)

    if remainder > 0:
        # Remainder is smaller than one full 128-lane chunk; fall back to lane 0.
        offset = n_full_tiles * chunk
        random_buffer = nl.ndarray([NUM_LANES, remainder], dtype=nl.int32, buffer=nl.sbuf)
        nisa.rng(dst=random_buffer, engine=nisa.engine.gpsimd)
        nisa.dma_copy(dst=output[0:1, offset : offset + remainder], src=random_buffer[0:1, 0:remainder])
    return output
