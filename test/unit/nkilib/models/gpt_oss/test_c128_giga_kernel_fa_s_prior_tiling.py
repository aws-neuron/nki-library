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

"""Pin the gigakernel's FA s_prior tiling resolution: packed forces it, dense unmoved.

Sequence packing was unreachable from the gigakernel decode path because both flag
sites resolved DISABLE_FA's default and ``attention_block_tkg`` asserts the tiling is
on whenever packed metadata is present. Packed metadata now forces it on.

The load-bearing claim is the NEGATIVE one -- that dense callers resolve exactly what
they resolved before -- and a test that only checked ``False`` for today's default
would pass just as happily if the packed disjunct had been written to fire on a dense
call. So the dense cases here are compared against ``not
mega_flag_default_on("VLLM_NEURON_MEGA_DISABLE_FA")`` recomputed independently, which
is the pre-change expression itself rather than a copy of its expected value: if that
helper's own truthiness definition changes, the dense assertion tracks it and only the
packed override is pinned.
"""

import pytest
from nkilib_src.nkilib.models.gpt_oss.c128_giga_kernel.core.utils.kernel_helpers import mega_flag_default_on
from nkilib_src.nkilib.models.gpt_oss.c128_giga_kernel.experimental.gpt_oss.gpt_oss_decode_layer_dp_tp_ep import (
    _resolve_fa_s_prior_tiling,
)

_FLAG = "VLLM_NEURON_MEGA_DISABLE_FA"

# The env values kernel_helpers treats as OFF, plus the on/unset ones. Explicit rather than
# imported so the matrix is reviewable -- but note it does NOT fail if _FLAG_OFF_VALUES
# widens: the expectation is recomputed from mega_flag_default_on, so both sides move
# together and a newly-added value is simply left uncovered here.
_ENV_VALUES = (None, "", "0", "false", "False", "1", "true", "anything")


@pytest.mark.parametrize("env_value", _ENV_VALUES)
def test_dense_resolution_is_the_pre_change_expression(monkeypatch, env_value):
    """With no packed mask the result IS ``not mega_flag_default_on(...)``, unchanged."""
    if env_value is None:
        monkeypatch.delenv(_FLAG, raising=False)
    else:
        monkeypatch.setenv(_FLAG, env_value)

    assert _resolve_fa_s_prior_tiling(None) == (not mega_flag_default_on(_FLAG))


@pytest.mark.parametrize("env_value", _ENV_VALUES)
def test_packed_metadata_forces_tiling_on_regardless_of_the_env(monkeypatch, env_value):
    """A packed mask forces the tiling on, including when DISABLE_FA is explicitly set."""
    if env_value is None:
        monkeypatch.delenv(_FLAG, raising=False)
    else:
        monkeypatch.setenv(_FLAG, env_value)

    # Any non-None sentinel: the resolution is an identity test, not a tensor read.
    assert _resolve_fa_s_prior_tiling(object()) is True


def test_unset_env_is_the_case_that_blocked_packed(monkeypatch):
    """The exact pre-change configuration: unset env, dense off, packed on."""
    monkeypatch.delenv(_FLAG, raising=False)

    assert mega_flag_default_on(_FLAG) is True, "unset must read as the flag being ON"
    assert _resolve_fa_s_prior_tiling(None) is False, "dense must still resolve tiling OFF"
    assert _resolve_fa_s_prior_tiling(object()) is True, "packed must now resolve tiling ON"


def test_both_flag_sites_call_the_shared_resolver():
    """Neither entry may re-derive the flag: a divergence is a wrong mask layout.

    The mega entry sizes the hoisted mask from its own resolution while the layer body
    passes its own to attention, so two copies that drift produce a mask whose layout
    does not match the shape attention expects.
    """
    import inspect

    from nkilib_src.nkilib.models.gpt_oss.c128_giga_kernel.experimental.gpt_oss import (
        gpt_oss_decode_layer_dp_tp_ep as module,
    )

    source = inspect.getsource(module)
    assignments = [line.strip() for line in source.splitlines() if "_enable_fa_s_prior_tiling =" in line.strip()]
    assert len(assignments) == 2, f"expected two flag sites, found {len(assignments)}: {assignments}"
    assert all(
        line == "_enable_fa_s_prior_tiling = _resolve_fa_s_prior_tiling(seq_packed_attention_mask)"
        for line in assignments
    ), assignments
