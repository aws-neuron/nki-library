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
"""
Centralized imports of compiler-internal utilities.

WARNING: TEMPORARY! DO NOT DEPEND ON THESE APIs
=========================================================
This module is a stop-gap fix until we officially provide the custom dtype support
functions in upcoming releases. These APIs are subject to change without
notice and should not be relied upon by external customers.

The functions re-exported here are internal implementation details and may be
changed or removed without notice.

If you are a customer reading this, please refrain from depending on these APIs.
"""

from neuronxcc.starfish.support import dtype as dt
from neuronxcc.starfish.support.dtype import is_float_type, static_cast

__all__ = ["dt", "is_float_type", "static_cast"]
