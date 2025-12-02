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


from typing import Optional, Tuple

import nki.language as nl

from .common_types import ActFnType, NormType

# TODO: Get this constant from the NKI API once it is available
NUM_HW_PSUM_BANKS = 8
PSUM_BANK_SIZE = 2048

#
# Local constants and data structures
_max_pos_range_map = {
    nl.float8_e4m3: 240.0,
    nl.float8_e5m2: 57344.0,
}

_act_fn_map = {
    ActFnType.SiLU: nl.silu,
    ActFnType.GELU: nl.gelu,
    ActFnType.GELU_Tanh_Approx: nl.gelu_apprx_tanh,
    # TODO: Need to add this to nl.lang
    # ActFnType.Swish: nl.gelu_apprx_sigmoid
}


# def is_sbuf_tensor(t):
#   """Checks whether the input nt.tensor or tensor view is in SBUF. """
#   return isinstance(t, NeuronSBTensor) or (hasattr(t, '_tensor') and isinstance(t._tensor, NeuronSBTensor))

# def is_psum_tensor(t):
#   """Checks whether the input nt.tensor or tensor view is in PSUM. """
#   return isinstance(t, NeuronPSUMTensor) or (hasattr(t, '_tensor') and isinstance(t._tensor, NeuronPSUMTensor))


def get_ceil_quotient(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def get_ceil_aligned_size(size: int, alignment_multiple: int) -> int:
    return get_ceil_quotient(size, alignment_multiple) * alignment_multiple


def get_floor_quotient(numerator: int, denominator: int) -> int:
    return numerator // denominator


def get_floor_aligned_size(size: int, alignment_multiple: int) -> int:
    return get_floor_quotient(size, alignment_multiple) * alignment_multiple


def get_nl_act_fn_from_type(act_fn: ActFnType):
    # TODO: NKIFE-375, currently new FE doesn't support enum as dict keys
    # return _act_fn_map[act_fn]
    return nl.silu


def get_nl_act_fn_from_type_hack(act_fn: ActFnType):
    if act_fn.value == ActFnType.SiLU.value:  # see NKIFE-347
        return nl.silu
    elif act_fn.value == ActFnType.GELU.value:
        return nl.gelu
    elif act_fn.value == ActFnType.GELU_Tanh_Approx.value:
        return nl.gelu_apprx_tanh
    elif act_fn.value == ActFnType.Swish.value:
        return nl.gelu_apprx_sigmoid
    assert False


def is_launched_as_spmd() -> bool:
    return nl.program_ndim() != 0 and nl.num_programs(axes=0) > 1


def is_rms_normalization(norm_type: NormType) -> bool:
    return norm_type == NormType.RMS_NORM or norm_type == NormType.RMS_NORM_SKIP_GAMMA


def normalization_uses_weights(norm_type: NormType) -> bool:
    return norm_type == NormType.RMS_NORM or norm_type == NormType.LAYER_NORM


def get_program_sharding_info() -> Tuple[int, int, int]:
    grid_ndim = nl.program_ndim()
    n_prgs, prg_id = (nl.num_programs(axes=0), nl.program_id(axis=0)) if grid_ndim != 0 else (1, 0)
    return grid_ndim, n_prgs, prg_id


def get_verified_program_sharding_info(
    kernel_name: str = "",
    allowed_ndims: Optional[Tuple[int, ...]] = None,
    max_sharding: Optional[int] = None,
) -> Tuple[int, int, int]:
    grid_ndim, n_prgs, prg_id = get_program_sharding_info()
    ndim_check = allowed_ndims is None or (grid_ndim == allowed_ndims[0] if len(allowed_ndims) == 1 else False)
    return grid_ndim, n_prgs, prg_id


def div_ceil(n, d):
    return (n + d - 1) // d


def get_max_positive_value_for_dtype(dtype) -> float:
    if str(dtype) == "float8e4":
        dtype = nl.float8_e4m3
    result = _max_pos_range_map.get(dtype, None)
    return result
