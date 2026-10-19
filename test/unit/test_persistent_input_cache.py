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

import numpy as np

from test.utils.persistent_input_cache import PersistentPerRankInputCache, build_input_cache_key, link_raw_memmap


def test_input_cache_key_tracks_payload_and_source(tmp_path):
    source = tmp_path / "generator.py"
    source.write_text("version = 1\n")

    key = build_input_cache_key({"layers": 36}, [source])
    assert key == build_input_cache_key({"layers": 36}, [source])
    assert key != build_input_cache_key({"layers": 8}, [source])

    source.write_text("version = 2\n")
    assert key != build_input_cache_key({"layers": 36}, [source])


def test_persistent_input_cache_generates_once_and_memory_maps_hits(tmp_path):
    calls = 0

    def generate():
        nonlocal calls
        calls += 1
        return {
            "x": np.arange(12, dtype=np.float16).reshape(3, 4),
            "rank": np.array([[7]], dtype=np.uint32),
        }

    cache = PersistentPerRankInputCache(tmp_path, "key")
    first = cache.get_or_create(7, generate)
    second = cache.get_or_create(7, generate)

    assert calls == 1
    assert isinstance(first["x"], np.memmap)
    assert isinstance(second["x"], np.memmap)
    np.testing.assert_array_equal(second["x"], np.arange(12, dtype=np.float16).reshape(3, 4))
    assert not second["x"].flags.writeable


def test_cached_memmap_is_hard_linked_when_raw_file_is_exact(tmp_path):
    source = tmp_path / "source.bin"
    np.arange(8, dtype=np.float32).tofile(source)
    value = np.memmap(source, dtype=np.float32, mode="r", shape=(8,))
    destination = tmp_path / "destination.bin"

    assert link_raw_memmap(value, destination)
    assert os.path.samefile(source, destination)
    np.testing.assert_array_equal(np.fromfile(destination, dtype=np.float32), value)


def test_cached_memmap_with_offset_is_not_linked(tmp_path):
    source = tmp_path / "source.bin"
    np.arange(8, dtype=np.float32).tofile(source)
    value = np.memmap(source, dtype=np.float32, mode="r", offset=4, shape=(7,))

    assert not link_raw_memmap(value, tmp_path / "destination.bin")
