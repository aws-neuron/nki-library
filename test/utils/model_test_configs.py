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
"""Shared types for the per-kernel model test configuration tables."""

from .common_dataclasses import ModelTestType

# Model test configs grouped by tier: each entry is one pytest parametrize case.
ModelTestConfigs = dict[ModelTestType, list]


def no_model_configs() -> ModelTestConfigs:
    """Empty config table for kernels whose model config module is unavailable.

    Model config modules are optional: they are absent, or decline to import, in
    environments where model tests are disabled. Test files fall back to this so the
    model parametrization is empty instead of holding a value of an unrelated type.
    """
    return {}
