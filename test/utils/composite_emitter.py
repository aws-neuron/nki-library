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
"""Composite emitter that fans out to multiple emitters."""

import logging
from typing import final

from typing_extensions import override

from .metrics_collector import IMetricsCollector
from .metrics_emitter import IMetricsEmitter, OutputMode


@final
class CompositeEmitter(IMetricsEmitter):
    """Emitter that fans out to multiple child emitters."""

    def __init__(self, emitters: list[IMetricsEmitter]):
        """
        Initialize composite emitter.

        Args:
            emitters: List of emitters to fan out to
        """
        self._emitters = emitters
        self._logger = logging.getLogger(__name__)

    @override
    def get_metrics_enabled(self) -> bool:
        """Returns True if any child emitter has metrics enabled."""
        return any(emitter.get_metrics_enabled() for emitter in self._emitters)

    @override
    def get_output_mode(self) -> OutputMode | None:
        # Return first emitter's mode if any
        for emitter in self._emitters:
            mode = emitter.get_output_mode()
            if mode is not None:
                return mode
        return None

    @override
    def emit(self, collector: IMetricsCollector) -> None:
        """Call emit(collector) on all child emitters, log errors but don't fail."""
        for emitter in self._emitters:
            try:
                emitter.emit(collector)
            except Exception as e:
                self._logger.error(f"Emitter {type(emitter).__name__} failed: {e}")
