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

"""HBM safety proofs for framework-ready kernels."""

import importlib

import numpy as np
import pytest
from nki.language.buffers import is_hbm  # ty: ignore[unresolved-import]

from test.utils.simulation_setup import setup_simulation_mode

from .framework_ready_api_spec import HBM_SAFE_PROOF, HbmProof, ProofTensorSpec

# Cap BLAS threads in xdist workers; no import-order dependency
setup_simulation_mode()


def _assert_outputs_are_hbm(result: object, label: str):
    """Assert all tensor outputs are in HBM."""
    if result is None:
        return
    items = list(result) if isinstance(result, (list, tuple)) else [result]
    for i, item in enumerate(items):
        if hasattr(item, "buffer"):
            assert is_hbm(item.buffer), f"Output {i} of {label} is a tensor but not in HBM"


def _run_hbm_safe_test(proof: HbmProof):
    mod = importlib.import_module(proof.module_path)
    kernel_fn = getattr(mod, proof.name)

    np.random.seed(42)
    kwargs: dict[str, object] = {}
    for arg_name, spec in proof.args.items():
        if isinstance(spec, ProofTensorSpec):
            if np.issubdtype(spec.dtype, np.integer):
                kwargs[arg_name] = np.random.randint(0, max(spec.randint_max, 1), size=spec.shape).astype(spec.dtype)
            else:
                kwargs[arg_name] = np.random.randn(*spec.shape).astype(spec.dtype)
        else:
            assert not isinstance(spec, np.ndarray), f"Non-tensor arg '{arg_name}' must not be a numpy array"
            kwargs[arg_name] = spec

    lnc = kwargs.pop("_lnc", 1)
    label = f"{proof.module_path}.{proof.name}"
    original_fn = kernel_fn.__wrapped__ if hasattr(kernel_fn, "__wrapped__") else kernel_fn

    def checking_wrapper(*args, **kw):
        result = original_fn(*args, **kw)
        _assert_outputs_are_hbm(result, label)
        return result

    from nki.simulator import simulate_kernel  # ty: ignore[unresolved-import]

    simulate_kernel(checking_wrapper, args=[], kwargs=kwargs, lnc=lnc)


@pytest.mark.parametrize(
    "proof",
    HBM_SAFE_PROOF,
    ids=[f"{p.module_path}.{p.name}" for p in HBM_SAFE_PROOF],
)
def test_hbm_safe_kernel(proof: HbmProof):
    """Each framework-ready kernel must execute with all-HBM I/O in the simulator."""
    _run_hbm_safe_test(proof)
