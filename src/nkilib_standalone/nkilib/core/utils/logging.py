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
Logger with support for different logging levels and hierarchical naming.

The class is implemented to run in a NKI environment.

Example usage:
  # Create top level logger
  logger = Logger("attn-mk", level=LogLevel.INFO)
  logger.info("validating input")   # Shows: [INFO] [attn-mk] validating input
  logger.debug("hidden size=3072")  # No output (below INFO level)

  # Create child logger
  qkv_logger = logger.create_child("qkv")
  qkv_logger.info("dispatching to QKV TKG")  # Shows: [INFO] [attn-mk.qkv] dispatching to QKV TKG

  # Child with different level
  qkv_logger = logger.create_child("qkv", level=LogLevel.DEBUG)
  qkv_logger.debug("hidden size=3072")  # Shows: [DEBUG] [attn-mk.qkv] hidden size=3072

"""

from enum import Enum
from typing import Optional

import nki.language as nl


class LogLevel(Enum):
    DEBUG = 0
    INFO = 1
    WARN = 2
    ERROR = 3
    OFF = 999


GLOBAL_LOG_LEVEL = LogLevel.INFO


class Logger(nl.NKIObject):
    def __init__(self, name: str, level: Optional[LogLevel] = None):
        self.name = name
        self.level = level if level != None else GLOBAL_LOG_LEVEL

    def create_child(self, name: str, level: Optional[LogLevel] = None):
        """Create a child logger with a hierarchical name and optionally adjust level"""
        child_level = level if level != None else self.level
        return Logger(self.name + "." + name, child_level)

    def _should_log(self, level: LogLevel) -> bool:
        return level.value >= self.level.value

    def is_enabled_for(self, level: LogLevel) -> bool:
        """Check if level is enabled before computing expensive message"""
        return self._should_log(level)

    def _print(self, msg: str, prefix: str = ""):
        name_prefix = f"[{self.name}] " if self.name else ""
        print(f"{prefix}{name_prefix}{msg}")

    def debug(self, msg: str):
        if self._should_log(LogLevel.DEBUG):
            self._print(msg, prefix="[DEBUG] ")

    def info(self, msg):
        if self._should_log(LogLevel.INFO):
            self._print(msg, prefix="[INFO] ")

    def warn(self, msg):
        if self._should_log(LogLevel.WARN):
            self._print(msg, prefix="[WARN] ")

    def error(self, msg):
        if self._should_log(LogLevel.ERROR):
            self._print(msg, prefix="[ERROR] ")


# global logger instance
logger = Logger("")
