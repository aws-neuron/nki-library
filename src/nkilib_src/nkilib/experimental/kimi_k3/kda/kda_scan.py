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

"""Kimi Delta Attention scan using bounded NKI structured control flow.

For each head and timestep, with state ``S[K, V]``::

    S = S * exp(g_t)[:, None]
    delta = (v_t - k_t @ S) * beta_t
    S = S + outer(k_t, delta)
    out_t = q_t @ S

``q`` is pre-normalized and scaled, ``k`` is pre-normalized, ``g`` is the
final per-channel log decay, and ``beta`` is post-sigmoid. All inputs, outputs,
and recurrent state are fp32. The tiled variant emits only ``TILE_T`` steps in
the static body and uses ``nl.fori_loop`` for the sequence tiles, avoiding a
model graph whose body grows with every KDA layer and timestep.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert

P_MAX = 128


def _kda_step(S, g_col, beta, k_col, k_row, q_col, v_row, out_dst, dim):
    """Apply one gated delta-rule step to an SBUF-resident state."""
    nisa.tensor_scalar(dst=S, data=S, op0=nl.multiply, operand0=g_col)

    k_state_ps = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=k_state_ps, stationary=k_col, moving=S)
    k_state = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=k_state, src=k_state_ps)

    delta = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=delta, data1=v_row, data2=k_state, op=nl.subtract)
    nisa.tensor_scalar(dst=delta, data=delta, op0=nl.multiply, operand0=beta)

    outer_ps = nl.ndarray((dim, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=outer_ps, stationary=k_row, moving=delta)
    outer = nl.ndarray((dim, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=outer, src=outer_ps)
    nisa.tensor_tensor(dst=S, data1=S, data2=outer, op=nl.add)

    out_ps = nl.ndarray((dim, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=out_ps, stationary=S, moving=q_col)
    nisa.tensor_copy(dst=out_dst, src=out_ps)


def _seed_state(state, init_state_h, head, dim, has_init):
    """Load carry-in state or explicitly clear a recycled state slot."""
    if has_init:
        nisa.dma_copy(dst=state, src=init_state_h[head, 0:dim, 0:dim])
    else:
        nisa.memset(dst=state, value=0.0)


def kda_scan_fori_tiled(
    q_h,
    k_h,
    v_h,
    g_h,
    beta_h,
    H,
    T,
    D,
    TILE_T=128,
    init_state_h=None,
    HAS_INIT=False,
):
    """Run the KDA recurrence with a bounded static body.

    Args:
        q_h: ``[H, T, D]`` fp32 normalized and scaled queries.
        k_h: ``[H, T, D]`` fp32 normalized keys.
        v_h: ``[H, T, D]`` fp32 values.
        g_h: ``[H, T, D]`` fp32 final log decays.
        beta_h: ``[H, T]`` fp32 post-sigmoid write strengths.
        H: Compile-time head count.
        T: Compile-time sequence length.
        D: Compile-time head dimension.
        TILE_T: Static timesteps emitted inside each structured loop iteration.
        init_state_h: Optional ``[H, D, D]`` fp32 carry-in state.
        HAS_INIT: Compile-time carry-in-state selector.

    Returns:
        ``(out_hvt, state_h)`` with layouts ``[H, D, T]`` and ``[H, D, D]``.
    """
    kernel_assert(D <= P_MAX, f"D={D} must be <= {P_MAX}")
    kernel_assert(
        TILE_T <= P_MAX,
        f"TILE_T={TILE_T} must be <= {P_MAX} for nc_transpose",
    )
    kernel_assert(
        T % TILE_T == 0,
        f"T={T} must be a multiple of TILE_T={TILE_T}",
    )

    n_tiles = T // TILE_T
    out_hvt = nl.ndarray((H, D, T), dtype=nl.float32, buffer=nl.shared_hbm)
    state_h = nl.ndarray((H, D, D), dtype=nl.float32, buffer=nl.shared_hbm)

    for head in nl.affine_range(H):
        state = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.sbuf)
        _seed_state(state, init_state_h, head, D, HAS_INIT)

        t_cur = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=t_cur, value=0)

        def _tile_body(_tile):
            k_rows = nl.ndarray((TILE_T, D), dtype=nl.float32, buffer=nl.sbuf)
            q_rows = nl.ndarray((TILE_T, D), dtype=nl.float32, buffer=nl.sbuf)
            g_rows = nl.ndarray((TILE_T, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=k_rows,
                src=k_h.ap(
                    pattern=[[D, TILE_T], [1, D]],
                    offset=head * T * D,
                    scalar_offset=t_cur,
                    indirect_dim=1,
                ),
                dge_mode=nisa.dge_mode.hwdge,
                name=f"kda_load_k_h{head}",
            )
            nisa.dma_copy(
                dst=q_rows,
                src=q_h.ap(
                    pattern=[[D, TILE_T], [1, D]],
                    offset=head * T * D,
                    scalar_offset=t_cur,
                    indirect_dim=1,
                ),
                dge_mode=nisa.dge_mode.hwdge,
                name=f"kda_load_q_h{head}",
            )
            nisa.dma_copy(
                dst=g_rows,
                src=g_h.ap(
                    pattern=[[D, TILE_T], [1, D]],
                    offset=head * T * D,
                    scalar_offset=t_cur,
                    indirect_dim=1,
                ),
                dge_mode=nisa.dge_mode.hwdge,
                name=f"kda_load_g_h{head}",
            )

            beta_row = nl.ndarray((1, TILE_T), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=beta_row,
                src=beta_h.ap(
                    pattern=[[T, 1], [1, TILE_T]],
                    offset=head * T,
                    scalar_offset=t_cur,
                    indirect_dim=1,
                ),
                dge_mode=nisa.dge_mode.hwdge,
                name=f"kda_load_beta_h{head}",
            )

            k_transposed_ps = nl.ndarray((D, TILE_T), dtype=nl.float32, buffer=nl.psum)
            q_transposed_ps = nl.ndarray((D, TILE_T), dtype=nl.float32, buffer=nl.psum)
            g_transposed_ps = nl.ndarray((D, TILE_T), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=k_transposed_ps, data=k_rows)
            nisa.nc_transpose(dst=q_transposed_ps, data=q_rows)
            nisa.nc_transpose(dst=g_transposed_ps, data=g_rows)

            k_transposed = nl.ndarray((D, TILE_T), dtype=nl.float32, buffer=nl.sbuf)
            q_transposed = nl.ndarray((D, TILE_T), dtype=nl.float32, buffer=nl.sbuf)
            g_exp = nl.ndarray((D, TILE_T), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=k_transposed, src=k_transposed_ps)
            nisa.tensor_copy(dst=q_transposed, src=q_transposed_ps)
            nisa.activation(dst=g_exp, data=g_transposed_ps, op=nl.exp)

            out_tile = nl.ndarray((D, TILE_T), dtype=nl.float32, buffer=nl.sbuf)

            for step in range(TILE_T):
                beta = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=beta,
                    src=beta_row[0:1, nl.ds(step, 1)],
                )

                # One-partition rows must start at SBUF partition zero. Fetching
                # them directly from HBM avoids invalid partition-offset slices.
                k_row = nl.ndarray((1, D), dtype=nl.float32, buffer=nl.sbuf)
                v_row = nl.ndarray((1, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=k_row,
                    src=k_h.ap(
                        pattern=[[D, 1], [1, D]],
                        offset=head * T * D + step * D,
                        scalar_offset=t_cur,
                        indirect_dim=1,
                    ),
                    dge_mode=nisa.dge_mode.hwdge,
                    name=f"kda_krow_h{head}_t{step}",
                )
                nisa.dma_copy(
                    dst=v_row,
                    src=v_h.ap(
                        pattern=[[D, 1], [1, D]],
                        offset=head * T * D + step * D,
                        scalar_offset=t_cur,
                        indirect_dim=1,
                    ),
                    dge_mode=nisa.dge_mode.hwdge,
                    name=f"kda_vrow_h{head}_t{step}",
                )
                _kda_step(
                    state,
                    g_exp[0:D, nl.ds(step, 1)],
                    beta,
                    k_transposed[0:D, nl.ds(step, 1)],
                    k_row,
                    q_transposed[0:D, nl.ds(step, 1)],
                    v_row,
                    out_tile[0:D, nl.ds(step, 1)],
                    D,
                )

            nisa.dma_copy(
                dst=out_hvt.ap(
                    pattern=[[T, D], [1, TILE_T]],
                    offset=head * D * T,
                    scalar_offset=t_cur,
                    indirect_dim=2,
                ),
                src=out_tile,
                dge_mode=nisa.dge_mode.hwdge,
                name=f"kda_store_out_h{head}",
            )
            nisa.tensor_scalar(
                dst=t_cur,
                data=t_cur,
                op0=nl.add,
                operand0=TILE_T,
            )

        nl.fori_loop(0, n_tiles, _tile_body, step=1)
        nisa.dma_copy(dst=state_h[head, 0:D, 0:D], src=state)

    # Fence all shared-HBM writes before returning both outputs.
    nisa.dma_copy(dst=out_hvt, src=out_hvt)
    return out_hvt, state_h


def kda_scan_fori(
    q_h,
    k_h,
    v_h,
    g_h,
    beta_h,
    H,
    T,
    D,
    init_state_h=None,
    HAS_INIT=False,
):
    """Run one KDA timestep per ``nl.fori_loop`` iteration."""
    return kda_scan_fori_tiled(
        q_h,
        k_h,
        v_h,
        g_h,
        beta_h,
        H,
        T,
        D,
        TILE_T=1,
        init_state_h=init_state_h,
        HAS_INIT=HAS_INIT,
    )


kda_scan_fori_tiled_jit = nki.jit(kda_scan_fori_tiled)
