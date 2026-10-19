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

"""Tests for MXFP8 MoE backward pass configuration."""

from inspect import signature

import nki.language as nl
import pytest
from nki.dtype import float8_e4m3fn_x4  # ty: ignore[unresolved-import]
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_config import MatmulMxfp8KernelConfig
from nkilib_src.nkilib.experimental.moe.bwd.moe_bwd_parameters import (
    ActFnType,
    AffinityOption,
    ClampLimits,
    ShardOption,
    SkipMode,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.bwd.blockwise_mm_backward_mxfp8 import (
    _resolve_phase_configs,
    _validate_kernel_options,
    blockwise_mm_bwd_mxfp8,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.bwd.blockwise_mm_backward_mxfp8_torch import (
    blockwise_mm_bwd_mxfp8_torch_ref,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.bwd.config import (
    MXFP8MOEBwdConfig,
    SwizzleMode,
    TransposeMode,
)
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.common_dataclasses import QuantScheme, TensorDescriptor

from test.integration.nkilib.experimental.moe_mxfp8.tuned_blocking import (
    MAX_DGT_LOAD_WIDTH,
    SHAPE_TUNED_CONFIGS,
    derive_load_tiles,
    get_shape_tuned_config,
    tune_block_and_load_tiles,
)


class TestTensorDescriptorSwizzleMode:
    def test_uses_typed_tensor_local_mode(self):
        descriptor = TensorDescriptor(swizzle_mode=SwizzleMode.PE)

        assert descriptor.swizzle_mode == SwizzleMode.PE
        assert descriptor.uses_pe_swizzle

    def test_legacy_boolean_remains_supported(self):
        # load_with_PE_swizzle is now a read-only back-compat view of swizzle_mode.
        descriptor = TensorDescriptor(swizzle_mode=SwizzleMode.PE)

        assert descriptor.load_with_PE_swizzle
        assert descriptor.uses_pe_swizzle

    def test_quantized_input_is_always_swizzled(self):
        # Why the backward has no gate_up_weight_is_swizzled / down_weight_is_swizzled
        # knob: passing scales makes the tensor quantized, and __post_init__ then forces
        # is_swizzled=True, overriding whatever the caller asked for. Only the
        # non-None-ness of scales is read here, so a bare sentinel stands in for a tensor
        # (allocating one needs an active NKI trace context).
        descriptor = TensorDescriptor(scales=object(), is_swizzled=False)  # ty: ignore[invalid-argument-type]

        assert descriptor.is_quantized
        assert descriptor.is_swizzled


class TestMXFP8MOEBwdConfigDefaults:
    """Test default instantiation of MXFP8MOEBwdConfig."""

    def test_default_construction(self):
        config = MXFP8MOEBwdConfig()
        assert config.compute_dtype == nl.bfloat16
        assert config.fp8_x4_dtype == float8_e4m3fn_x4
        assert config.activation_type == ActFnType.SiLU
        assert config.shard_option == ShardOption.SHARD_ON_FREE
        # AFFINITY_ON_I is the default (the only mode the MXFP8 backward implements),
        # so a bare MXFP8MOEBwdConfig() passes the kernel's affinity validation.
        assert config.affinity_option == AffinityOption.AFFINITY_ON_I
        # Off by default: single_expert_dense (also default) requires it off.
        assert not config.accumulate_hidden_states_grad
        assert not config.skip_grad_initialization
        assert config.single_expert_dense
        assert config.num_static_block is None
        assert not config.fast_dma_transpose
        assert not config.bias

    def test_post_init_creates_phase_configs(self):
        config = MXFP8MOEBwdConfig()
        assert config.phase1_config is not None
        assert config.phase2_config is not None
        assert config.phase3_config is not None
        assert config.phase4_config is not None
        assert isinstance(config.phase1_config, MatmulMxfp8KernelConfig)
        assert isinstance(config.phase2_config, MatmulMxfp8KernelConfig)
        assert isinstance(config.phase3_config, MatmulMxfp8KernelConfig)
        assert isinstance(config.phase4_config, MatmulMxfp8KernelConfig)

    def test_post_init_phase_configs_have_default_blocking(self):
        config = MXFP8MOEBwdConfig()
        for phase_config in [
            config.phase1_config,
            config.phase2_config,
            config.phase3_config,
            config.phase4_config,
        ]:
            assert phase_config.M == 0
            assert phase_config.K == 0
            assert phase_config.N == 0
            assert phase_config.TILES_IN_BLOCK_M == 1
            assert phase_config.TILES_IN_BLOCK_N == 1
            assert phase_config.TILES_IN_BLOCK_K == 1

    def test_post_init_creates_clamp_limits(self):
        config = MXFP8MOEBwdConfig()
        assert config.clamp_limits is not None
        assert isinstance(config.clamp_limits, ClampLimits)
        assert config.clamp_limits.linear_clamp_upper_limit is None
        assert config.clamp_limits.linear_clamp_lower_limit is None
        assert config.clamp_limits.non_linear_clamp_upper_limit is None
        assert config.clamp_limits.non_linear_clamp_lower_limit is None

    def test_skip_dma_remains_none_after_post_init(self):
        config = MXFP8MOEBwdConfig()
        assert config.skip_dma is None

    def test_no_prequantized_weight_swizzle_flags(self):
        # gate_up_weight_is_swizzled / down_weight_is_swizzled were removed as dead:
        # pre-quantized weights are always swizzled, and TensorDescriptor.__post_init__
        # enforces that (is_quantized => is_swizzled), so the flag could never be read.
        # See TestTensorDescriptorSwizzleMode for the invariant itself.
        config = MXFP8MOEBwdConfig()
        assert not hasattr(config, "gate_up_weight_is_swizzled")
        assert not hasattr(config, "down_weight_is_swizzled")


class TestMXFP8MOEBwdConfigCustomValues:
    """Test custom instantiation of MXFP8MOEBwdConfig."""

    def test_custom_activation_type(self):
        config = MXFP8MOEBwdConfig(activation_type=ActFnType.GELU)
        assert config.activation_type == ActFnType.GELU

    def test_custom_shard_option(self):
        config = MXFP8MOEBwdConfig(shard_option=ShardOption.SHARD_ON_HIDDEN)
        assert config.shard_option == ShardOption.SHARD_ON_HIDDEN

    def test_custom_affinity_option(self):
        config = MXFP8MOEBwdConfig(affinity_option=AffinityOption.AFFINITY_ON_I)
        assert config.affinity_option == AffinityOption.AFFINITY_ON_I

    def test_custom_accumulation_flags(self):
        config = MXFP8MOEBwdConfig(
            accumulate_hidden_states_grad=False,
            skip_grad_initialization=True,
        )
        assert not config.accumulate_hidden_states_grad
        assert config.skip_grad_initialization

    def test_custom_num_static_block(self):
        config = MXFP8MOEBwdConfig(num_static_block=4)
        assert config.num_static_block == 4

    def test_custom_clamp_limits(self):
        limits = ClampLimits(
            linear_clamp_upper_limit=1.0,
            linear_clamp_lower_limit=-1.0,
            non_linear_clamp_upper_limit=2.0,
            non_linear_clamp_lower_limit=-2.0,
        )
        config = MXFP8MOEBwdConfig(clamp_limits=limits)
        assert config.clamp_limits.linear_clamp_upper_limit == 1.0
        assert config.clamp_limits.linear_clamp_lower_limit == -1.0
        assert config.clamp_limits.non_linear_clamp_upper_limit == 2.0
        assert config.clamp_limits.non_linear_clamp_lower_limit == -2.0

    def test_custom_skip_dma(self):
        skip = SkipMode(skip_token=True, skip_weight=False)
        config = MXFP8MOEBwdConfig(skip_dma=skip)
        assert config.skip_dma.skip_token
        assert not config.skip_dma.skip_weight

    def test_custom_bias(self):
        config = MXFP8MOEBwdConfig(bias=True)
        assert config.bias

    def test_custom_single_expert_dense(self):
        config = MXFP8MOEBwdConfig(single_expert_dense=True)
        assert config.single_expert_dense

    def test_custom_fast_dma_transpose(self):
        config = MXFP8MOEBwdConfig(fast_dma_transpose=True)
        assert config.fast_dma_transpose

    def test_custom_phase_config_not_overwritten(self):
        custom_phase1 = MatmulMxfp8KernelConfig(
            M=1024,
            K=512,
            N=2048,
            TILES_IN_BLOCK_M=4,
            TILES_IN_BLOCK_N=2,
            TILES_IN_BLOCK_K=1,
            quant_scheme=QuantScheme.WRAPX,
        )
        config = MXFP8MOEBwdConfig(phase1_config=custom_phase1)
        assert config.phase1_config.M == 1024
        assert config.phase1_config.K == 512
        assert config.phase1_config.N == 2048
        assert config.phase1_config.TILES_IN_BLOCK_M == 4
        assert config.phase1_config.TILES_IN_BLOCK_N == 2
        assert config.phase1_config.TILES_IN_BLOCK_K == 1
        # Other phases should still get defaults
        assert config.phase2_config.M == 0
        assert config.phase3_config.M == 0
        assert config.phase4_config.M == 0

    def test_phase_configs_own_spill_reload_and_scale_packing(self):
        # Spill/reload and scale packing are per-phase, so _resolve_phase_configs must
        # leave each phase's values alone while filling the shape-dependent fields.
        phase1 = MatmulMxfp8KernelConfig(
            M=0,
            K=0,
            N=0,
            quant_scheme=QuantScheme.WRAPX,
            spill_reload=False,
            enable_scale_packing=False,
        )
        phase3 = MatmulMxfp8KernelConfig(
            M=0,
            K=0,
            N=0,
            quant_scheme=QuantScheme.WRAPX,
            spill_reload=True,
            enable_scale_packing=True,
        )
        config = MXFP8MOEBwdConfig(phase1_config=phase1, phase3_config=phase3)
        phase_shapes = (
            (256, 512, 128),
            (256, 512, 256),
            (256, 256, 256),
            (256, 256, 256),
        )

        _resolve_phase_configs(
            config=config,
            phase_shapes=phase_shapes,
            hidden_size=512,
            run_with_lnc2=True,
        )

        assert not config.phase1_config.spill_reload
        assert not config.phase1_config.enable_scale_packing
        assert config.phase3_config.spill_reload
        assert config.phase3_config.enable_scale_packing
        # Omitted phases keep the _default_matmul_config values (both on).
        assert config.phase2_config.spill_reload
        assert config.phase2_config.enable_scale_packing

        for phase_config, phase_shape in zip(
            (
                config.phase1_config,
                config.phase2_config,
                config.phase3_config,
                config.phase4_config,
            ),
            phase_shapes,
            strict=True,
        ):
            assert (phase_config.M, phase_config.K, phase_config.N) == phase_shape
            assert phase_config.run_with_lnc2
            assert phase_config.tile_m is not None
            assert phase_config.tile_k is not None
            assert phase_config.tile_n is not None

    def test_load_m_defaults_to_one(self):
        phase3 = MatmulMxfp8KernelConfig(
            M=0,
            K=0,
            N=0,
            TILES_IN_BLOCK_M=12,
            TILES_IN_BLOCK_N=4,
            TILES_IN_BLOCK_K=2,
            quant_scheme=QuantScheme.WRAPX,
        )
        config = MXFP8MOEBwdConfig(phase3_config=phase3)

        _resolve_phase_configs(
            config=config,
            phase_shapes=(
                (4096, 4096, 768),
                (4096, 3072, 2048),
                (3072, 4096, 2048),
                (2048, 4096, 1536),
            ),
            hidden_size=4096,
            run_with_lnc2=True,
        )

        assert config.phase3_config.TILES_IN_LOAD_M == 1

    def test_resolve_accepts_1x32_quantization(self):
        phase_configs = tuple(MatmulMxfp8KernelConfig(M=0, K=0, N=0, quant_scheme=QuantScheme._1x32) for _ in range(4))
        config = MXFP8MOEBwdConfig(
            phase1_config=phase_configs[0],
            phase2_config=phase_configs[1],
            phase3_config=phase_configs[2],
            phase4_config=phase_configs[3],
        )

        _resolve_phase_configs(
            config=config,
            phase_shapes=((4096, 4096, 768), (4096, 3072, 2048), (3072, 4096, 2048), (2048, 4096, 1536)),
            hidden_size=4096,
            run_with_lnc2=True,
        )

        assert all(phase.quant_scheme == QuantScheme._1x32 for phase in phase_configs)


class TestMXFP8MOEBwdConfigValidation:
    def test_rejects_mixed_quantization_schemes(self):
        with pytest.raises(ValueError, match="same quantization scheme"):
            MXFP8MOEBwdConfig(
                phase1_config=MatmulMxfp8KernelConfig(M=0, K=0, N=0, quant_scheme=QuantScheme._1x32),
                phase2_config=MatmulMxfp8KernelConfig(M=0, K=0, N=0, quant_scheme=QuantScheme.WRAPX),
                phase3_config=MatmulMxfp8KernelConfig(M=0, K=0, N=0, quant_scheme=QuantScheme._1x32),
                phase4_config=MatmulMxfp8KernelConfig(M=0, K=0, N=0, quant_scheme=QuantScheme._1x32),
            )

    def test_preserves_nested_configs_and_modes(self):
        config = MXFP8MOEBwdConfig(
            phase1_config=MatmulMxfp8KernelConfig(
                M=0,
                K=0,
                N=0,
                tile_n=256,
                TILES_IN_BLOCK_N=3,
                quant_scheme=QuantScheme.WRAPX,
                spill_reload=True,
                enable_scale_packing=False,
            ),
            phase3_transpose_mode=TransposeMode.DMA,
        )

        assert config.phase1_config.spill_reload
        assert not config.phase1_config.enable_scale_packing
        assert config.phase1_config.tile_n == 256
        assert config.phase1_config.TILES_IN_BLOCK_N == 3
        assert config.phase1_config.quant_scheme == QuantScheme.WRAPX
        assert config.phase3_transpose_mode == TransposeMode.DMA
        assert config.phase4_transpose_mode == TransposeMode.NC


class TestMXFP8MOEBwdTunedBlocking:
    @staticmethod
    def _qwen3_tp1_1x32_config(pe_transpose_only=True):
        return get_shape_tuned_config(
            B=4096,
            H=4096,
            I_TP=1536,
            num_shards=2,
            shard_option=ShardOption.SHARD_ON_FREE,
            affinity_option=AffinityOption.AFFINITY_ON_I,
            compute_dtype=nl.bfloat16,
            single_expert_dense=True,
            use_scale_packing=True,
            quant_scheme=QuantScheme._1x32,
            pe_transpose_only=pe_transpose_only,
        )

    @staticmethod
    def _blocking(config):
        return [
            (
                phase.tile_n,
                phase.TILES_IN_BLOCK_M,
                phase.TILES_IN_BLOCK_N,
                phase.TILES_IN_BLOCK_K,
                phase.TILES_IN_LOAD_M,
                phase.TILES_IN_LOAD_N,
                phase.spill_reload,
                phase.enable_scale_packing,
            )
            for phase in (
                config.phase1_config,
                config.phase2_config,
                config.phase3_config,
                config.phase4_config,
            )
        ]

    def test_qwen3_tp1_1x32_pe_uses_tuned_blocking(self):
        config = self._qwen3_tp1_1x32_config()

        assert config is not None
        assert self._blocking(config) == [
            (256, 8, 3, 8, 8, 3, False, True),
            (512, 8, 4, 3, 4, 4, False, True),
            (512, 24, 2, 2, 24, 2, False, True),
            (512, 16, 3, 2, 16, 3, False, True),
        ]
        assert all(
            phase.quant_scheme == QuantScheme._1x32
            for phase in (
                config.phase1_config,
                config.phase2_config,
                config.phase3_config,
                config.phase4_config,
            )
        )

    def test_qwen3_tp1_1x32_pe_lookup_does_not_match_legacy_dense_tuning(self):
        one_by_32 = self._qwen3_tp1_1x32_config()
        legacy = get_shape_tuned_config(
            B=4096,
            H=4096,
            I_TP=1536,
            num_shards=2,
            shard_option=ShardOption.SHARD_ON_FREE,
            affinity_option=AffinityOption.AFFINITY_ON_I,
            compute_dtype=nl.bfloat16,
            single_expert_dense=True,
        )

        assert one_by_32 is not None
        assert one_by_32 is not legacy

    def test_qwen3_tp1_1x32_mixed_transpose_uses_tuned_blocking(self):
        mixed = self._qwen3_tp1_1x32_config(pe_transpose_only=False)
        pe_only = self._qwen3_tp1_1x32_config()

        assert mixed is not None
        assert self._blocking(mixed) == self._blocking(pe_only)
        assert mixed is not pe_only

    def test_load_tuning_respects_dgt_width(self):
        assert derive_load_tiles(4, 512) == 2
        assert derive_load_tiles(3, 256) == 3

    def test_odd_blocks_are_retuned_to_multi_tile_loads(self):
        assert tune_block_and_load_tiles(3, 512) == (2, 2)
        assert tune_block_and_load_tiles(5, 512) == (4, 2)
        assert tune_block_and_load_tiles(7, 512) == (6, 2)

    def test_all_tuned_phases_have_explicit_legal_blocking_and_loads(self):
        for tuning_key, config in SHAPE_TUNED_CONFIGS.items():
            for phase_config in (
                config.phase1_config,
                config.phase2_config,
                config.phase3_config,
                config.phase4_config,
            ):
                block_m = phase_config.TILES_IN_BLOCK_M
                block_n = phase_config.TILES_IN_BLOCK_N
                block_k = phase_config.TILES_IN_BLOCK_K
                load_m = phase_config.TILES_IN_LOAD_M
                load_n = phase_config.TILES_IN_LOAD_N
                tile_n = phase_config.tile_n
                assert block_m is not None
                assert block_n is not None
                assert block_k is not None
                assert load_m is not None
                assert load_n is not None
                assert tile_n is not None
                assert min(block_m, block_n, block_k, load_m, load_n) >= 1
                assert block_m % load_m == 0
                assert block_n % load_n == 0
                if block_m > 1:
                    assert load_m > 1
                if block_n > 1:
                    assert load_n > 1
                if tuning_key.quant_scheme == QuantScheme.WRAPX and not tuning_key.pe_transpose_only:
                    assert load_n * tile_n <= MAX_DGT_LOAD_WIDTH


class TestMXFP8MOEBwdKernelDefaults:
    """The backward kernel entry defaults block_size to 4096 and makes config optional.

    A framework can call ``blockwise_mm_bwd_mxfp8`` with only the required tensors:
    ``block_size`` falls back to 4096 and ``config=None`` resolves to a default
    ``MXFP8MOEBwdConfig()`` (one-tile-per-block phases) inside the kernel.
    """

    def test_block_size_defaults_to_4096(self):
        assert signature(blockwise_mm_bwd_mxfp8).parameters["block_size"].default == 4096

    def test_config_is_optional(self):
        assert signature(blockwise_mm_bwd_mxfp8).parameters["config"].default is None

    def test_config_none_resolves_to_default_config(self):
        # The kernel builds a default config when none is supplied, so a bare
        # MXFP8MOEBwdConfig() is a complete, valid configuration on its own.
        default = MXFP8MOEBwdConfig()
        assert default.shard_option == ShardOption.SHARD_ON_FREE
        assert default.affinity_option == AffinityOption.AFFINITY_ON_I
        for phase in (default.phase1_config, default.phase2_config, default.phase3_config, default.phase4_config):
            assert phase is not None
            assert phase.TILES_IN_BLOCK_M == 1
            assert phase.spill_reload
            assert phase.enable_scale_packing
        for mode in (
            default.output_grad_swizzle_mode,
            default.down_weight_swizzle_mode,
            default.d_gate_up_swizzle_mode,
            default.gate_up_weight_swizzle_mode,
            default.d_gate_up_t_swizzle_mode,
            default.hidden_states_t_swizzle_mode,
            default.output_grad_t_swizzle_mode,
            default.scaled_intermediate_t_swizzle_mode,
        ):
            assert mode == SwizzleMode.PE
        assert default.phase3_transpose_mode == TransposeMode.NC


class TestMXFP8MOEBwdWeightSwizzleValidation:
    """PE weight loads are rejected for E > 1 instead of silently reading expert 0.

    The routed path selects the per-expert weight slice through
    TensorDescriptor.scalar_offset, which load_tile_PE_swizzle_wrapX never applies. The
    weight swizzle modes default to PE, so multi-expert callers must be told to switch
    them back to DGT rather than getting wrong weight gradients.
    """

    def _validate(self, config, E):
        _validate_kernel_options(config=config, down_proj_act_checkpoint=None, E=E)

    def test_pe_weight_modes_rejected_for_multiple_experts(self):
        for field in ("gate_up_weight_swizzle_mode", "down_weight_swizzle_mode"):
            config = MXFP8MOEBwdConfig(
                gate_up_weight_swizzle_mode=SwizzleMode.DGT,
                down_weight_swizzle_mode=SwizzleMode.DGT,
            )
            setattr(config, field, SwizzleMode.PE)
            with pytest.raises(Exception, match="requires E == 1"):
                self._validate(config, E=2)

    def test_pe_weight_modes_allowed_for_single_expert(self):
        # E == 1 makes the ignored per-expert offset always 0, so the PE default stands.
        self._validate(MXFP8MOEBwdConfig(), E=1)

    def test_dgt_weight_modes_allowed_for_multiple_experts(self):
        config = MXFP8MOEBwdConfig(
            gate_up_weight_swizzle_mode=SwizzleMode.DGT,
            down_weight_swizzle_mode=SwizzleMode.DGT,
        )
        self._validate(config, E=8)

    def test_non_weight_pe_modes_unrestricted_for_multiple_experts(self):
        # Only the two weight operands carry a scalar_offset, so the activation and
        # gradient operands keep their PE default regardless of expert count.
        config = MXFP8MOEBwdConfig(
            gate_up_weight_swizzle_mode=SwizzleMode.DGT,
            down_weight_swizzle_mode=SwizzleMode.DGT,
        )
        assert config.output_grad_swizzle_mode == SwizzleMode.PE
        assert config.scaled_intermediate_t_swizzle_mode == SwizzleMode.PE
        self._validate(config, E=4)

    def test_torch_ref_mirrors_kernel_defaults(self):
        # The reference must resolve an omitted block_size / config identically to
        # the kernel, or a defaults-relying test would diverge from the golden.
        kernel = signature(blockwise_mm_bwd_mxfp8).parameters
        ref = signature(blockwise_mm_bwd_mxfp8_torch_ref).parameters
        assert ref["block_size"].default == kernel["block_size"].default == 4096
        assert ref["config"].default is None and kernel["config"].default is None
