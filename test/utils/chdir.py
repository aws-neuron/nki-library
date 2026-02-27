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
import os
from typing import Optional


class chdir:
    def __init__(self, path):
        self.path: str = path
        self.original: list[Optional[str]] = []

    def __enter__(self):
        # It is possible for getcwd() to fail if the current directory has been deleted
        try:
            self.original.append(os.getcwd())
        except FileNotFoundError as _:
            self.original.append(None)
            pass

        os.chdir(self.path)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        original_dir = self.original.pop()
        if original_dir:
            os.chdir(original_dir)
