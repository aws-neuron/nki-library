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

"""Trace-time scheduling hints (instruction handles) handed between subkernels."""

_LAST_PV_MATMUL: list = []


def record_pv_matmul(handle):
    """attention_tkg: remember the latest P@V matmul handle of this layer (the last one stays)."""
    if handle is not None:
        _LAST_PV_MATMUL[:] = [handle]


def take_pv_matmul():
    """output_projection_tkg: the layer's last P@V matmul handle (or None), cleared so a SWA layer cannot inherit a stale one."""
    if not _LAST_PV_MATMUL:
        return None
    h = _LAST_PV_MATMUL[0]
    _LAST_PV_MATMUL[:] = []
    return h


_QMX_STORE_COPY_HANDLES = []


def begin_qmx_store_copies():
    """Producer side, at the first copy of one rmsnorm_mx_quantize trace: drop stale handles no router consumed."""
    del _QMX_STORE_COPY_HANDLES[:]


def record_qmx_store_copy(handle):
    """Producer side (rmsnorm_mx_quantize_tkg): remember one PSUM->SBUF copy instruction handle."""
    if handle is not None:
        _QMX_STORE_COPY_HANDLES.append(handle)


def take_qmx_store_copies():
    """Consumer side (router_topk): the recorded handles, in trace order; the list is cleared."""
    deps = list(_QMX_STORE_COPY_HANDLES)
    del _QMX_STORE_COPY_HANDLES[:]
    return deps
