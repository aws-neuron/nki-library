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
"""Does the load hoist speed up the prequantized MXFP8 input case?

Same shapes/tiling as test_pe_swizzle_hoist_proof.py, but both operands are
PREQUANTIZED MXFP8 (already quantized and swizzled in HBM). Like preswizzled BF16,
prequantized operands do NO on-chip transpose -- the load path is a DMA copy of the
already-laid-out data -- so the hoist can only remove a redundant DMA copy on the
invariant loop axis, not nc_transpose work.

Configs (K=4096, LNC1):
  ctrl_n1 : BLOCK_N=1 LOAD_N=1 -> tile_n loop x1 -> expected NO-OP (control)
  n3      : BLOCK_N=3 LOAD_N=1 -> tile_n loop x3
  n6      : BLOCK_N=6 LOAD_N=1 -> tile_n loop x6
  m4_rhs  : BLOCK_M=4 LOAD_M=1 -> tile_m loop x4 -> RHS
"""

import numpy as np
import pytest
from nkilib_src.nkilib.experimental.matmul_mxfp8 import matmul_mxfp8_generic_kernel
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_torch import matmul_mxfp8_torch_ref

from test.integration.nkilib.experimental.matmul_mxfp8 import config_helper, constants
from test.integration.nkilib.experimental.matmul_mxfp8.test_matmul_mxfp8_generic_kernel import (
    _mxfp8_comparator,
    build_matmul_inputs,
    get_output_dtype,
)
from test.utils import common_dataclasses
from test.utils.unit_test_framework import UnitTestFramework

_K = 4096

# (label, M, N, tile_n, BLOCK_M, BLOCK_N, BLOCK_K, LOAD_M, LOAD_N)
_SWEEP = [
    ("ctrl_n1", 512, 512, 512, 4, 1, 4, 4, 1),  # tile_n loop x1 -> no-op control
    ("n3", 512, 1536, 512, 4, 3, 4, 4, 1),  # tile_n loop x3
    ("n6", 512, 3072, 512, 4, 6, 4, 4, 1),  # tile_n loop x6
    ("m4_rhs", 512, 512, 512, 4, 1, 4, 1, 1),  # tile_m loop x4 -> RHS
]


def _make_configs():
    configs = []
    for label, M, N, tile_n, bm, bn, bk, lm, ln in _SWEEP:
        conf = config_helper.TestConfig(
            M=M,
            K=_K,
            N=N,
            tile_m=128,
            tile_k=512,
            tile_n=tile_n,
            TILES_IN_BLOCK_M=bm,
            TILES_IN_BLOCK_N=bn,
            TILES_IN_BLOCK_K=bk,
            TILES_IN_LOAD_M=lm,
            TILES_IN_LOAD_N=ln,
            description=f"prequant-hoist-{label}",
            run_with_lnc2=False,
            enable_scale_packing=False,
            output_dtype=constants.MatrixPrecision.BFLOAT16,
            lhs_dtype=constants.MatrixPrecision.MXFP8,
            rhs_dtype=constants.MatrixPrecision.MXFP8,
            lhs_is_swizzled=True,
            rhs_is_swizzled=True,
            spill_reload=True,
        )
        configs.append(conf)
    return configs


_CONFIGS = _make_configs()


class TestPrequantHoistProof:
    @pytest.mark.parametrize("conf", _CONFIGS, ids=[c.description for c in _CONFIGS])
    def test_prequant_hoist(self, test_manager, conf, platform_target):
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")

        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=2 if conf.run_with_lnc2 else 1,
            platform_target=platform_target,
        )

        test_manager.collector.set_kernel_params(conf.to_metrics_dict())
        output_dtype = get_output_dtype(conf)

        def input_generator(test_config):
            return build_matmul_inputs(conf)

        def output_tensors(kernel_input):
            return {"out": np.zeros((conf.M, conf.N), dtype=output_dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=matmul_mxfp8_generic_kernel.matmul_mxfp8,
            torch_ref=matmul_mxfp8_torch_ref,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            custom_comparator=_mxfp8_comparator(conf, output_dtype, False),
        )
