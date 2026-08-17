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

"""Test configuration helper for MXFP8 matmul tests."""

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from nkilib_src.nkilib.core.utils.kernel_helpers import div_ceil
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_config import (
    MatmulMxfp8KernelConfig,
    _generate_k_chains,
    _generate_m_chains,
    _generate_n_chains,
    auto_generate_default,
    calc_sbuf_free_dim_size,
    calculate_sbuf_usage,
    fits_in_sbuf,
    resolve_lnc2_sharding,
)

from test.integration.nkilib.experimental.matmul_mxfp8.constants import (
    DEFAULT_STRIDE,
    SBUF_LIMIT_BYTES,
    MatrixPrecision,
)
from test.integration.nkilib.experimental.matmul_mxfp8.random_input_generator import (
    get_random_distributions,
    set_seed,
)


def _load_bf16_baseline_cache() -> Dict[str, Any]:
    """Load BF16 baseline cache from bf16_baseline_cache.json if it exists."""
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bf16_baseline_cache.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as cache_file:
                return json.load(cache_file)
        except Exception:
            pass
    return {}


_BF16_BASELINE_CACHE = _load_bf16_baseline_cache()


class TestConfig:
    def __init__(
        self,
        M: int,
        K: int,
        N: int,
        TILES_IN_BLOCK_M: Optional[int] = None,
        TILES_IN_BLOCK_N: Optional[int] = None,
        TILES_IN_BLOCK_K: Optional[int] = None,
        TILES_IN_LOAD_M: Optional[int] = None,
        TILES_IN_LOAD_N: Optional[int] = None,
        tile_m: Optional[int] = None,
        tile_k: Optional[int] = None,
        tile_n: Optional[int] = None,
        tile_loop_order: Optional[str] = None,
        block_loop_order: Optional[str] = None,
        run_with_lnc2: Optional[bool] = None,
        float8_dtype: Optional[str] = None,
        lhs_dtype: Optional[str] = None,
        rhs_dtype: Optional[str] = None,
        output_dtype: Optional[str] = None,
        xfail: str = "pass",
        description: str = "",
        seed: int = 42,
        dists: Optional[List[str]] = None,
        params: Optional[List[Dict[str, Any]]] = None,
        stride: Optional[int] = None,
        spill_reload: Optional[bool] = None,
        lhs_is_swizzled: bool = True,
        rhs_is_swizzled: bool = True,
        enable_scale_packing: Optional[bool] = None,
        lnc_2_shard_rhs: Optional[bool] = None,
        load_with_PE_swizzle: bool = False,
        fast_subset: Optional[set] = None,
        # Both default to F-by-K ([F, K]) to match the kernel default and avoid the K-by-F
        # F%512 constraint on M/N for the common case. Set either to False per-operand to
        # exercise the K-by-F ([K, F]) layout.
        lhs_is_f_by_k: bool = True,
        rhs_is_f_by_k: bool = True,
        quant_scheme: str = "wrapX",
    ) -> None:
        # Create kernel config with kernel-level params
        self.kernel_config = MatmulMxfp8KernelConfig(
            M=M,
            K=K,
            N=N,
            tile_m=tile_m,
            tile_k=tile_k,
            tile_n=tile_n,
            TILES_IN_BLOCK_M=TILES_IN_BLOCK_M,
            TILES_IN_BLOCK_N=TILES_IN_BLOCK_N,
            TILES_IN_BLOCK_K=TILES_IN_BLOCK_K,
            TILES_IN_LOAD_M=TILES_IN_LOAD_M,
            TILES_IN_LOAD_N=TILES_IN_LOAD_N,
            block_loop_order=block_loop_order if block_loop_order is not None else 'mnk',
            tile_loop_order=tile_loop_order if tile_loop_order is not None else 'mnk',
            float8_dtype=float8_dtype if float8_dtype is not None else 'float8_e4m3fn',
            enable_scale_packing=enable_scale_packing if enable_scale_packing is not None else False,
            run_with_lnc2=run_with_lnc2 if run_with_lnc2 is not None else True,
            lnc_2_shard_rhs=lnc_2_shard_rhs,
            spill_reload=spill_reload if spill_reload is not None else False,
            lhs_is_swizzled=lhs_is_swizzled,
            rhs_is_swizzled=rhs_is_swizzled,
        )
        # Store original None-ness for optional params used in generation logic
        self._orig_run_with_lnc2 = run_with_lnc2
        self._orig_tile_loop_order = tile_loop_order
        self._orig_block_loop_order = block_loop_order
        self._orig_float8_dtype = float8_dtype
        self._orig_spill_reload = spill_reload
        self._orig_enable_scale_packing = enable_scale_packing
        self._orig_lnc_2_shard_rhs = lnc_2_shard_rhs

        # Original constructor params (None means "not specified by user")
        self.run_with_lnc2_param = run_with_lnc2
        self.tile_loop_order_param = tile_loop_order
        self.block_loop_order_param = block_loop_order
        self.float8_dtype_param = float8_dtype
        self.spill_reload_param = spill_reload
        self.enable_scale_packing_param = enable_scale_packing
        self.lnc_2_shard_rhs_param = lnc_2_shard_rhs

        # Test-only params
        self.xfail = xfail
        self.description = description
        self.seed = seed
        self.lhs_dtype = lhs_dtype
        self.rhs_dtype = rhs_dtype
        self.output_dtype = output_dtype
        self.dists = dists
        self.params = params
        self.stride = stride if stride is not None else DEFAULT_STRIDE
        self.load_with_PE_swizzle = load_with_PE_swizzle
        # Sub-indices (within this entry's autoGenerateRandomSubset output)
        # to mark pytest.mark.fast. Stable to grid reordering and additions;
        # sensitive to seed and TESTS_PER_SETUP changes. Consumed by populate_tests().
        self.fast_subset = frozenset(fast_subset) if fast_subset is not None else frozenset()
        self.lhs_is_f_by_k = lhs_is_f_by_k
        self.rhs_is_f_by_k = rhs_is_f_by_k
        self.quant_scheme = quant_scheme
        self.enable_psum_copy_in = None

    # ------------------------------------------------------------------
    # Property accessors delegating to kernel_config for backward compat
    # ------------------------------------------------------------------

    @property
    def M(self) -> int:
        return self.kernel_config.M

    @property
    def K(self) -> int:
        return self.kernel_config.K

    @property
    def N(self) -> int:
        return self.kernel_config.N

    @property
    def tile_m(self) -> Optional[int]:
        return self.kernel_config.tile_m

    @property
    def tile_k(self) -> Optional[int]:
        return self.kernel_config.tile_k

    @property
    def tile_n(self) -> Optional[int]:
        return self.kernel_config.tile_n

    @property
    def TILES_IN_BLOCK_M(self) -> Optional[int]:
        return self.kernel_config.TILES_IN_BLOCK_M

    @property
    def TILES_IN_BLOCK_N(self) -> Optional[int]:
        return self.kernel_config.TILES_IN_BLOCK_N

    @property
    def TILES_IN_BLOCK_K(self) -> Optional[int]:
        return self.kernel_config.TILES_IN_BLOCK_K

    @property
    def TILES_IN_LOAD_M(self) -> Optional[int]:
        return self.kernel_config.TILES_IN_LOAD_M

    @property
    def TILES_IN_LOAD_N(self) -> Optional[int]:
        return self.kernel_config.TILES_IN_LOAD_N

    @property
    def tile_loop_order(self) -> str:
        return self.kernel_config.tile_loop_order

    @property
    def block_loop_order(self) -> str:
        return self.kernel_config.block_loop_order

    @property
    def run_with_lnc2(self) -> bool:
        return self.kernel_config.run_with_lnc2

    @property
    def float8_dtype(self) -> str:
        return self.kernel_config.float8_dtype

    @property
    def spill_reload(self) -> bool:
        return self.kernel_config.spill_reload

    @property
    def lhs_is_swizzled(self) -> bool:
        return self.kernel_config.lhs_is_swizzled

    @property
    def rhs_is_swizzled(self) -> bool:
        return self.kernel_config.rhs_is_swizzled

    @property
    def enable_scale_packing(self) -> bool:
        return self.kernel_config.enable_scale_packing

    @property
    def lnc_2_shard_rhs(self) -> Optional[bool]:
        return self.kernel_config.lnc_2_shard_rhs

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def get_input_gen_config(self) -> Dict[str, Any]:
        """Get the input generation config dictionary for backward compatibility."""
        return {
            "shapes": (self.M, self.K, self.N),
            "dists": self.dists,
            "params": self.params,
            "stride": self.stride,
            "elem_dtype": self.float8_dtype,
            "seed": self.seed,
        }

    def to_metrics_dict(self) -> Dict[str, Any]:
        """Convert test config to dict for metrics emission."""
        metrics_dict = {
            "shape_name": self.description,
            "M": self.M,
            "K": self.K,
            "N": self.N,
            "TILES_IN_BLOCK_M": self.TILES_IN_BLOCK_M,
            "TILES_IN_BLOCK_N": self.TILES_IN_BLOCK_N,
            "TILES_IN_BLOCK_K": self.TILES_IN_BLOCK_K,
            "TILES_IN_LOAD_M": self.TILES_IN_LOAD_M,
            "TILES_IN_LOAD_N": self.TILES_IN_LOAD_N,
            "tile_m": self.tile_m,
            "tile_k": self.tile_k,
            "tile_n": self.tile_n,
            "tile_loop_order": self.tile_loop_order,
            "block_loop_order": self.block_loop_order,
            "run_with_lnc2": self.run_with_lnc2,
            "float8_dtype": self.float8_dtype,
            "lhs_dtype": self.lhs_dtype,
            "rhs_dtype": self.rhs_dtype,
            "output_dtype": self.output_dtype,
            "lhs_is_swizzled": self.lhs_is_swizzled,
            "rhs_is_swizzled": self.rhs_is_swizzled,
            "load_with_PE_swizzle": self.load_with_PE_swizzle,
            "quant_scheme": self.quant_scheme,
            "spill_reload": self.spill_reload,
            "enable_scale_packing": self.enable_scale_packing,
            "lnc_2_shard_rhs": self.lnc_2_shard_rhs,
            "seed": self.seed,
            "stride": self.stride,
            "conf": str(self),
        }

        baseline = self.get_bf16_baseline()
        if baseline:
            metrics_dict["BF16BaselineInferenceTime"] = baseline.get("inference_time", -1.0)
            metrics_dict["BF16BaselineActiveInferenceTime"] = baseline.get("active_inference_time", -1.0)
            metrics_dict["BF16BaselineMFU"] = baseline.get("mfu_percent", -1.0)
            metrics_dict["BF16BaselineMBU"] = baseline.get("mbu_percent", -1.0)

        return metrics_dict

    def get_bf16_baseline(self) -> Optional[Dict[str, Any]]:
        """Look up BF16 baseline for this shape from cache."""
        key = f"{self.M}x{self.K}x{self.N}"
        return _BF16_BASELINE_CACHE.get(key)

    def calculate_sbuf_usage(self) -> int:
        """Calculate the SBUF memory usage for this configuration in bytes."""
        return calculate_sbuf_usage(
            self.kernel_config,
            lhs_dtype=self.lhs_dtype or 'mxfp8_x4',
            rhs_dtype=self.rhs_dtype or 'mxfp8_x4',
            output_dtype_str=self.output_dtype or 'bfloat16',
        )

    def calc_sbuf_free_dim_size(self) -> int:
        """Calculate the maximum SBUF free dimension size for this configuration."""
        return calc_sbuf_free_dim_size(
            self.kernel_config,
            lhs_dtype=self.lhs_dtype or 'mxfp8_x4',
            rhs_dtype=self.rhs_dtype or 'mxfp8_x4',
            output_dtype_str=self.output_dtype or 'bfloat16',
        )

    def fits_in_sbuf(self) -> bool:
        """Check if this configuration fits within the SBUF memory limit."""
        return fits_in_sbuf(
            self.kernel_config,
            lhs_dtype=self.lhs_dtype or 'mxfp8_x4',
            rhs_dtype=self.rhs_dtype or 'mxfp8_x4',
            output_dtype_str=self.output_dtype or 'bfloat16',
        )

    @staticmethod
    def filter_configs_by_sbuf(configs: List["TestConfig"]) -> List["TestConfig"]:
        """Filter a list of test configurations to only include those that fit in SBUF."""
        return [config for config in configs if config.fits_in_sbuf()]

    # ------------------------------------------------------------------
    # Dimension chain generators (thin wrappers delegating to kernel_config)
    # ------------------------------------------------------------------

    def _generate_m_dimension_tile_configs(self) -> List[Tuple[int, int, int]]:
        """Generate M dimension chain: (tile_m, TILES_IN_BLOCK_M, TILES_IN_LOAD_M)."""
        return _generate_m_chains(self.kernel_config)

    def _generate_n_dimension_tile_configs(self) -> List[Tuple[int, int, int]]:
        """Generate N dimension chain: (tile_n, TILES_IN_BLOCK_N, TILES_IN_LOAD_N)."""
        return _generate_n_chains(self.kernel_config)

    def _generate_k_dimension_tile_configs(self) -> List[Tuple[int, int]]:
        """Generate K dimension chain: (tile_k, TILES_IN_BLOCK_K)."""
        return _generate_k_chains(self.kernel_config)

    # ------------------------------------------------------------------
    # Auto-generation
    # ------------------------------------------------------------------

    def autoGenerateDefault(self) -> List["TestConfig"]:
        """Generate default test configurations using the block_count_reducer strategy.

        Delegates kernel param generation to MatmulMxfp8KernelConfig.auto_generate_default(),
        then wraps the result with test-specific params.
        """
        assert self.M is not None and self.K is not None and self.N is not None

        # Resolve test-level defaults
        lhs_dtype = self.lhs_dtype if self.lhs_dtype is not None else MatrixPrecision.MXFP8_X4
        rhs_dtype = self.rhs_dtype if self.rhs_dtype is not None else MatrixPrecision.MXFP8_X4
        output_dtype = self.output_dtype if self.output_dtype is not None else MatrixPrecision.BFLOAT16

        dists = self._generate_random_dists(1)
        lhs_dist = random.choice(dists["lhs"])
        rhs_dist = random.choice(dists["rhs"])

        # Resolve the LNC2 flags on the full dims the way the kernel does, so the flags
        # recorded in metrics match the ones the kernel resolves and the sweep writer's
        # cache key matches the kernel's lookup. Pass prequant flags so the auto axis
        # choice matches the kernel for asymmetric (one-side-prequantized) shapes.
        _run_with_lnc2 = self.kernel_config.run_with_lnc2 if self._orig_run_with_lnc2 is not None else True
        _lhs_is_prequant = lhs_dtype in (MatrixPrecision.MXFP8, MatrixPrecision.MXFP8_X4)
        _rhs_is_prequant = rhs_dtype in (MatrixPrecision.MXFP8, MatrixPrecision.MXFP8_X4)
        _run_with_lnc2, _shard_rhs = resolve_lnc2_sharding(
            self.M,
            self.N,
            _run_with_lnc2,
            self.kernel_config.lnc_2_shard_rhs,
            lhs_is_prequant=_lhs_is_prequant,
            rhs_is_prequant=_rhs_is_prequant,
        )

        # Build a kernel config from current state, preserving already-set values.
        # auto_generate_default takes full M/N and applies the LNC2 shard itself.
        kc = MatmulMxfp8KernelConfig(
            M=self.M,
            K=self.K,
            N=self.N,
            tile_m=self.kernel_config.tile_m,
            tile_k=self.kernel_config.tile_k,
            tile_n=self.kernel_config.tile_n,
            TILES_IN_BLOCK_M=self.kernel_config.TILES_IN_BLOCK_M,
            TILES_IN_BLOCK_N=self.kernel_config.TILES_IN_BLOCK_N,
            TILES_IN_BLOCK_K=self.kernel_config.TILES_IN_BLOCK_K,
            TILES_IN_LOAD_M=self.kernel_config.TILES_IN_LOAD_M,
            TILES_IN_LOAD_N=self.kernel_config.TILES_IN_LOAD_N,
            block_loop_order=self.kernel_config.block_loop_order if self._orig_block_loop_order is not None else 'mnk',
            tile_loop_order=self.kernel_config.tile_loop_order if self._orig_tile_loop_order is not None else 'mnk',
            float8_dtype=self.kernel_config.float8_dtype if self._orig_float8_dtype is not None else 'float8_e4m3fn',
            enable_scale_packing=self.kernel_config.enable_scale_packing,
            run_with_lnc2=_run_with_lnc2,
            lnc_2_shard_rhs=_shard_rhs,
            spill_reload=self.kernel_config.spill_reload,
            lhs_is_swizzled=self.kernel_config.lhs_is_swizzled,
            rhs_is_swizzled=self.kernel_config.rhs_is_swizzled,
        )

        # Delegate kernel-level auto-generation
        auto_generate_default(
            kc,
            lhs_dtype=lhs_dtype,
            rhs_dtype=rhs_dtype,
            output_dtype_str=output_dtype,
        )

        # Resolve lnc_2_shard_rhs for the TestConfig wrapper
        if self._orig_lnc_2_shard_rhs is None:
            lnc_2_shard_rhs_val = kc.lnc_2_shard_rhs
        else:
            lnc_2_shard_rhs_val = self.kernel_config.lnc_2_shard_rhs

        desc = f"{self.description} [block_count_reducer]" if self.description else "block_count_reducer"
        config = TestConfig(
            M=self.M,
            K=self.K,
            N=self.N,
            tile_m=kc.tile_m,
            tile_k=kc.tile_k,
            tile_n=kc.tile_n,
            TILES_IN_BLOCK_M=kc.TILES_IN_BLOCK_M,
            TILES_IN_BLOCK_N=kc.TILES_IN_BLOCK_N,
            TILES_IN_BLOCK_K=kc.TILES_IN_BLOCK_K,
            TILES_IN_LOAD_M=kc.TILES_IN_LOAD_M,
            TILES_IN_LOAD_N=kc.TILES_IN_LOAD_N,
            tile_loop_order=kc.tile_loop_order,
            block_loop_order=kc.block_loop_order,
            run_with_lnc2=kc.run_with_lnc2,
            float8_dtype=kc.float8_dtype,
            lhs_dtype=lhs_dtype,
            rhs_dtype=rhs_dtype,
            output_dtype=output_dtype,
            xfail=self.xfail,
            description=desc,
            seed=self.seed,
            dists=[lhs_dist[0], rhs_dist[0]],
            params=[lhs_dist[1], rhs_dist[1]],
            stride=self.stride,
            spill_reload=kc.spill_reload if self._orig_spill_reload is not None else None,
            lhs_is_swizzled=kc.lhs_is_swizzled,
            rhs_is_swizzled=kc.rhs_is_swizzled,
            lhs_is_f_by_k=self.lhs_is_f_by_k,
            rhs_is_f_by_k=self.rhs_is_f_by_k,
            enable_scale_packing=kc.enable_scale_packing if self._orig_enable_scale_packing is not None else None,
            lnc_2_shard_rhs=lnc_2_shard_rhs_val
            if self._orig_lnc_2_shard_rhs is not None or not lnc_2_shard_rhs_val
            else None,
            load_with_PE_swizzle=self.load_with_PE_swizzle,
            quant_scheme=self.quant_scheme,
        )

        return [config]

    def autoGenerateRandomSubset(self, n_sample: int, distribution_population_size: int = 200) -> List["TestConfig"]:
        """Generate a random subset of test configurations by sampling each
        uninitialized field independently.

        Delegates kernel param generation to MatmulMxfp8KernelConfig.auto_generate_random(),
        then wraps results with test-specific params.
        """
        # make sure that the different (M,K,N) generate different distributions
        config_gen_seed = self.seed + 2 * self.M + 3 * self.K + 5 * self.N
        set_seed(config_gen_seed)

        # Pre-generate the option spaces
        m_chains = self._generate_m_dimension_tile_configs()
        n_chains = self._generate_n_dimension_tile_configs()
        k_chains = self._generate_k_dimension_tile_configs()
        dists = self._generate_random_dists(distribution_population_size)
        independent_params = self._generate_independent_params()

        independent_param_names = list(independent_params.keys())
        independent_param_values = [independent_params[name] for name in independent_param_names]

        configs: List[TestConfig] = []
        attempts = 0
        max_attempts = 1000

        while len(configs) < n_sample and attempts < max_attempts:
            set_seed(config_gen_seed + attempts)
            attempts += 1

            # Randomly sample from each chain/parameter space
            m_chain = random.choice(m_chains)
            n_chain = random.choice(n_chains)
            k_chain = random.choice(k_chains)
            independent_values = [random.choice(vals) for vals in independent_param_values]
            independent_dict = dict(zip(independent_param_names, independent_values, strict=True))
            lhs_dist = random.choice(dists["lhs"])
            rhs_dist = random.choice(dists["rhs"])

            # Randomize spill_reload if not explicitly set
            spill_reload = self.spill_reload if self._orig_spill_reload is not None else random.choice([True, False])
            """
            Randomize enable_scale_packing if not explicitly set.
            Pre-quantized MXFP8 with packed scales requires tile_k=512 because
            the packed scales format uses Q_TILE_K-sized tile indexing that doesn't
            align with smaller physical tile boundaries in the matmul instruction.
            """
            if self._orig_enable_scale_packing is not None:
                enable_scale_packing = self.enable_scale_packing
            else:
                enable_scale_packing = random.choice([True, False])
                if enable_scale_packing and k_chain[0] < 512:
                    lhs_dt = independent_dict.get('lhs_dtype', self.lhs_dtype)
                    rhs_dt = independent_dict.get('rhs_dtype', self.rhs_dtype)
                    has_prequantized = (lhs_dt in (MatrixPrecision.MXFP8, MatrixPrecision.MXFP8_X4)) or (
                        rhs_dt in (MatrixPrecision.MXFP8, MatrixPrecision.MXFP8_X4)
                    )
                    if has_prequantized:
                        enable_scale_packing = False

            # Generate run_with_lnc2 independently
            tile_m, tiles_in_block_m, _ = m_chain
            tile_n, tiles_in_block_n, _ = n_chain
            if self._orig_run_with_lnc2 is not None:
                run_with_lnc2 = self.run_with_lnc2
            else:
                num_blocks_in_m = div_ceil(self.M, tile_m * tiles_in_block_m) if (tile_m * tiles_in_block_m) > 0 else 0
                num_blocks_in_n = div_ceil(self.N, tile_n * tiles_in_block_n) if (tile_n * tiles_in_block_n) > 0 else 0
                if num_blocks_in_m >= 2 or num_blocks_in_n >= 2:
                    run_with_lnc2 = random.choice([True, False])
                else:
                    run_with_lnc2 = False

            # Randomize lnc_2_shard_rhs if not explicitly set
            if self._orig_lnc_2_shard_rhs is not None:
                lnc_2_shard_rhs = self.lnc_2_shard_rhs
            elif run_with_lnc2:
                num_blocks_in_m = div_ceil(self.M, tile_m * tiles_in_block_m) if (tile_m * tiles_in_block_m) > 0 else 0
                num_blocks_in_n = div_ceil(self.N, tile_n * tiles_in_block_n) if (tile_n * tiles_in_block_n) > 0 else 0
                can_shard_rhs = num_blocks_in_n >= 2
                can_shard_lhs = num_blocks_in_m >= 2
                if can_shard_rhs and can_shard_lhs:
                    lnc_2_shard_rhs = random.choice([True, False])
                elif can_shard_rhs:
                    lnc_2_shard_rhs = True
                elif can_shard_lhs:
                    lnc_2_shard_rhs = False
                else:
                    lnc_2_shard_rhs = True
            else:
                lnc_2_shard_rhs = True

            config = TestConfig(
                M=self.M,
                K=self.K,
                N=self.N,
                # M chain
                tile_m=m_chain[0],
                TILES_IN_BLOCK_M=m_chain[1],
                TILES_IN_LOAD_M=m_chain[2],
                # N chain
                tile_n=n_chain[0],
                TILES_IN_BLOCK_N=n_chain[1],
                TILES_IN_LOAD_N=n_chain[2],
                run_with_lnc2=run_with_lnc2,
                # K chain
                tile_k=k_chain[0],
                TILES_IN_BLOCK_K=k_chain[1],
                # Independent params
                tile_loop_order=independent_dict['tile_loop_order'],
                block_loop_order=independent_dict['block_loop_order'],
                float8_dtype=independent_dict['float8_dtype'],
                lhs_dtype=independent_dict['lhs_dtype'],
                rhs_dtype=independent_dict['rhs_dtype'],
                output_dtype=independent_dict['output_dtype'],
                spill_reload=spill_reload,
                enable_scale_packing=enable_scale_packing,
                # Fixed params
                xfail=self.xfail,
                description=self.description,
                seed=self.seed,
                # Input generation config
                dists=[lhs_dist[0], rhs_dist[0]],
                params=[lhs_dist[1], rhs_dist[1]],
                stride=self.stride,
                # Swizzled flags
                lhs_is_swizzled=self.lhs_is_swizzled,
                rhs_is_swizzled=self.rhs_is_swizzled,
                lhs_is_f_by_k=self.lhs_is_f_by_k,
                rhs_is_f_by_k=self.rhs_is_f_by_k,
                lnc_2_shard_rhs=lnc_2_shard_rhs,
            )

            # Only add if it fits in SBUF and isn't a duplicate
            if config.fits_in_sbuf() and not any(repr(config) == repr(existing) for existing in configs):
                configs.append(config)

        return configs

    # ------------------------------------------------------------------
    # Test-specific parameter generators
    # ------------------------------------------------------------------

    def _generate_independent_params(self) -> Dict[str, List[Any]]:
        """Generate independent parameters that don't depend on each other."""
        params: Dict[str, List[Any]] = {}

        if self._orig_tile_loop_order is not None:
            params['tile_loop_order'] = [self.tile_loop_order]
        else:
            params['tile_loop_order'] = ['mnk']

        if self._orig_block_loop_order is not None:
            params['block_loop_order'] = [self.block_loop_order]
        else:
            params['block_loop_order'] = ['mnk']

        if self._orig_float8_dtype is not None:
            params['float8_dtype'] = [self.float8_dtype]
        else:
            params['float8_dtype'] = ["float8_e4m3fn"]

        if self.lhs_dtype is not None:
            params['lhs_dtype'] = [self.lhs_dtype]
        else:
            params['lhs_dtype'] = [MatrixPrecision.BFLOAT16, MatrixPrecision.MXFP8, MatrixPrecision.MXFP8_X4]

        if self.rhs_dtype is not None:
            params['rhs_dtype'] = [self.rhs_dtype]
        else:
            params['rhs_dtype'] = [MatrixPrecision.BFLOAT16, MatrixPrecision.MXFP8, MatrixPrecision.MXFP8_X4]

        if self.output_dtype is not None:
            params['output_dtype'] = [self.output_dtype]
        else:
            params['output_dtype'] = [MatrixPrecision.BFLOAT16]

        return params

    def _generate_random_dists(self, distribution_population_size: int = 200) -> Dict[str, List]:
        res = {}
        if self.dists is not None and self.params is not None:
            assert len(self.dists) == 2, "dists needs to be of length 2"
            assert len(self.params) == 2, "params needs to be of length 2"
            res['lhs'] = [(self.dists[0], self.params[0])]
            res['rhs'] = [(self.dists[1], self.params[1])]
            return res
        elif self.dists is not None:
            assert len(self.dists) == 2, "dists needs to be of length 2"
            res['lhs'] = get_random_distributions(distribution_population_size, dist_names=[self.dists[0]])
            res['rhs'] = get_random_distributions(distribution_population_size, dist_names=[self.dists[1]])
        else:
            res['lhs'] = get_random_distributions(distribution_population_size)
            res['rhs'] = get_random_distributions(distribution_population_size)
        return res

    # ------------------------------------------------------------------
    # String representations
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        lines = []
        lines.append(f"TestConfig: {self.description if self.description else 'Matrix Multiplication Test'}")
        lines.append(f"  Matrix Dimensions: M={self.M}, K={self.K}, N={self.N}")

        if self.tile_m is not None or self.tile_k is not None or self.tile_n is not None:
            tile_info = []
            if self.tile_m is not None:
                tile_info.append(f"M={self.tile_m}")
            if self.tile_k is not None:
                tile_info.append(f"K={self.tile_k}")
            if self.tile_n is not None:
                tile_info.append(f"N={self.tile_n}")
            lines.append(f"  Tile Sizes: {', '.join(tile_info)}")

        if self.TILES_IN_BLOCK_M is not None or self.TILES_IN_BLOCK_K is not None or self.TILES_IN_BLOCK_N is not None:
            block_info = []
            if self.TILES_IN_BLOCK_M is not None:
                block_info.append(f"M={self.TILES_IN_BLOCK_M}")
            if self.TILES_IN_BLOCK_K is not None:
                block_info.append(f"K={self.TILES_IN_BLOCK_K}")
            if self.TILES_IN_BLOCK_N is not None:
                block_info.append(f"N={self.TILES_IN_BLOCK_N}")
            lines.append(f"  Tiles in Block: {', '.join(block_info)}")

        if self.TILES_IN_LOAD_M is not None or self.TILES_IN_LOAD_N is not None:
            load_info = []
            if self.TILES_IN_LOAD_M is not None:
                load_info.append(f"M={self.TILES_IN_LOAD_M}")
            if self.TILES_IN_LOAD_N is not None:
                load_info.append(f"N={self.TILES_IN_LOAD_N}")
            lines.append(f"  Tiles in Load: {', '.join(load_info)}")

        if self.lhs_dtype is not None or self.rhs_dtype is not None:
            dtype_info = []
            if self.lhs_dtype is not None:
                dtype_info.append(f"LHS={self.lhs_dtype}")
            if self.rhs_dtype is not None:
                dtype_info.append(f"RHS={self.rhs_dtype}")
            lines.append(f"  Data Types: {', '.join(dtype_info)}")

        if self.dists is not None and self.params is not None:
            dist_info = []
            for i, (dist, param) in enumerate(zip(self.dists, self.params, strict=True)):
                matrix_name = "LHS" if i == 0 else "RHS"
                if param:
                    dist_info.append(f"{matrix_name}={dist}({param})")
                else:
                    dist_info.append(f"{matrix_name}={dist}")
            lines.append(f"  Distributions: {', '.join(dist_info)}")

        if self._orig_tile_loop_order is not None:
            lines.append(f"  Tile Loop Order: {self.tile_loop_order}")
        if self._orig_block_loop_order is not None:
            lines.append(f"  Block Loop Order: {self.block_loop_order}")
        if self._orig_run_with_lnc2 is not None:
            lines.append(f"  Run with LNC2: {self.run_with_lnc2}")
        if self._orig_lnc_2_shard_rhs is not None:
            lines.append(f"  LNC2 Shard RHS: {self.lnc_2_shard_rhs}")
        if self._orig_float8_dtype is not None:
            lines.append(f"  Float8 DType: {self.float8_dtype}")
        if self.seed is not None:
            lines.append(f"  Seed: {self.seed}")

        try:
            sbuf_usage = self.calculate_sbuf_usage()
            sbuf_mb = sbuf_usage / (1024 * 1024)
            fits = "✓" if self.fits_in_sbuf() else "✗"
            lines.append(f"  SBUF Usage: {sbuf_mb:.2f} MB / {SBUF_LIMIT_BYTES / (1024 * 1024):.0f} MB {fits}")
        except (AssertionError, TypeError):
            pass

        baseline = self.get_bf16_baseline()
        if baseline:
            lines.append(
                f"  BF16 Baseline: InferenceTime={baseline.get('inference_time', -1.0)}, "
                f"ActiveInferenceTime={baseline.get('active_inference_time', -1.0)}, "
                f"MFU={baseline.get('mfu_percent', -1.0)}, "
                f"MBU={baseline.get('mbu_percent', -1.0)}"
            )

        if self.xfail != "pass":
            lines.append(f"  Expected Result: {self.xfail}")

        return '\n'.join(lines)

    def __repr__(self) -> str:
        """Return a detailed string representation that could recreate the object."""
        params = []
        params.append(f"M={self.M}")
        params.append(f"K={self.K}")
        params.append(f"N={self.N}")

        if self.TILES_IN_BLOCK_M is not None:
            params.append(f"TILES_IN_BLOCK_M={self.TILES_IN_BLOCK_M}")
        if self.TILES_IN_BLOCK_N is not None:
            params.append(f"TILES_IN_BLOCK_N={self.TILES_IN_BLOCK_N}")
        if self.TILES_IN_BLOCK_K is not None:
            params.append(f"TILES_IN_BLOCK_K={self.TILES_IN_BLOCK_K}")
        if self.TILES_IN_LOAD_M is not None:
            params.append(f"TILES_IN_LOAD_M={self.TILES_IN_LOAD_M}")
        if self.TILES_IN_LOAD_N is not None:
            params.append(f"TILES_IN_LOAD_N={self.TILES_IN_LOAD_N}")
        if self.tile_m is not None:
            params.append(f"tile_m={self.tile_m}")
        if self.tile_k is not None:
            params.append(f"tile_k={self.tile_k}")
        if self.tile_n is not None:
            params.append(f"tile_n={self.tile_n}")
        if self._orig_tile_loop_order is not None:
            params.append(f"tile_loop_order={self.tile_loop_order!r}")
        if self._orig_block_loop_order is not None:
            params.append(f"block_loop_order={self.block_loop_order!r}")
        if self._orig_run_with_lnc2 is not None:
            params.append(f"run_with_lnc2={self.run_with_lnc2}")
        if self._orig_float8_dtype is not None:
            params.append(f"float8_dtype={self.float8_dtype!r}")
        if self.lhs_dtype is not None:
            params.append(f"lhs_dtype={self.lhs_dtype!r}")
        if self.rhs_dtype is not None:
            params.append(f"rhs_dtype={self.rhs_dtype!r}")
        if self.xfail != "pass":
            params.append(f"xfail={self.xfail!r}")
        if self.description:
            params.append(f"description={self.description!r}")
        if self.seed is not None:
            params.append(f"seed={self.seed}")
        if self.output_dtype is not None:
            params.append(f"output_dtype={self.output_dtype!r}")
        if self.dists is not None:
            params.append(f"dists={self.dists!r}")
        if self.params is not None:
            params.append(f"params={self.params!r}")
        if self.stride != DEFAULT_STRIDE:
            params.append(f"stride={self.stride}")
        if self._orig_spill_reload is not None:
            params.append(f"spill_reload={self.spill_reload}")
        if self.enable_scale_packing:
            params.append(f"enable_scale_packing={self.enable_scale_packing}")
        if self._orig_lnc_2_shard_rhs is not None:
            params.append(f"lnc_2_shard_rhs={self.lnc_2_shard_rhs}")

        return f"TestConfig({', '.join(params)})"
