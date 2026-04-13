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
"""Test that public symbols can be imported from nkilib.experimental.subkernels."""


def test_import_find_nonzero_indices_from_experimental():
    from nkilib.experimental.subkernels import find_nonzero_indices

    assert callable(find_nonzero_indices)


def test_import_indexed_flatten_from_experimental():
    from nkilib.experimental.subkernels import indexed_flatten

    assert callable(indexed_flatten)
