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
User space stack allocator with support of multi-buffer.

The class is implemented to run in a NKI enviornment.

"""

from typing import Optional
from dataclasses import dataclass

import nki.language as nl

from .logging import Logger


def sizeinbytes(dtype):
    if str(dtype) == str(nl.float32):
        return 4
    elif str(dtype) == str(nl.bfloat16) or str(dtype) == str(nl.float16):
        return 2
    elif str(dtype) == str(nl.int8) or str(dtype) == str(nl.uint8):
        return 1
    elif str(dtype) == str(nl.int32) or str(dtype) == str(nl.uint32):
        return 4
    assert False, f"dtype size unknown! {dtype}"


def align_to(value, alignment):
    # This function is copied from the llvm::alignTo
    return ((value + alignment - 1) // alignment) * alignment


def _num_elts(shape):
    res = 1
    for i in range(len(shape)):
        res = res * shape[i]
    return res


@dataclass
class Scope(nl.NKIObject):
    starting_addr: int
    # number of independent sections in each stack frame,
    # used for multibuffer
    num_sections: int
    cur_section_id: int

    def __post_init__(self):
        self.cur_section_id = 0


class SbufManager(nl.NKIObject):
    def __init__(
        self,
        sb_lower_bound: int,
        sb_upper_bound: int,
        logger: Optional[Logger] = None,
        use_auto_alloc: bool = False,
        default_stack_alloc: bool = True,
    ):
        """
        Creates a SbufManager (referred to as SBM) instance
        with lower and upper bound, which jointly defines the contiguous region in sbuf
        that the SBUF manager may use.

        The Stack would grow upwards from the lower_bound, while the heap would grow downwards
        from the upper_bound.

        :param lower_bound: lower bound of the available sbuf memory region.
        :param upper_bound: upper bound of the available sbuf memory region.
        :param use_auto_alloc: Whether to use auto-allocation. Defaults to False.
        """
        self.lower_bound = sb_lower_bound
        self.upper_bound = sb_upper_bound
        self.stack_curr_addr = sb_lower_bound
        self.heap_curr_addr = sb_upper_bound
        self.use_auto_alloc = use_auto_alloc
        self.default_stack_alloc = default_stack_alloc
        self.logger = logger
        if self.logger == None:
            self.logger = Logger("SBM")
        self.scopes = []
        self.heap = []

    def is_auto_alloc(self):
        return self.use_auto_alloc

    def is_default_stack_alloc(self):
        return self.default_stack_alloc

    def is_default_heap_alloc(self):
        return not self.default_stack_alloc

    def open_scope(self, interleave_degree=1):
        """
        Add a new frame on the stack. SBUF addresses allocated on the stack will be
        automatically freed when its creation scope is closed.

        The optional argument `interleave_degree` helps manage multi-buffering in a loop.
        See the documentation of `increment_section` for more information.
        """
        self.scopes.append(Scope(self.stack_curr_addr, interleave_degree))

    def increment_section(self):
        """
        Increment the section count in the current scope. If the current section count reached the
        interleave_degree of the current scope, the address is reset to the beginning address of
        the current scope. Otherwise, continue allocate on the current address.

        Example:
        sbm = SbufManager(0, 128*1024, Logger())
        sbm.open_scope(interleave_degree=2)

        for i in range(4):
          sbm.alloc_stack((128, 128), dtype=nl.bfloat16)
          sbm.increment_section()
        sbm.close_scope()

        In the example above, the sbm would emit the following address for the 4 allocations,
                      address.      section_id after increment_section is called()
        Iteration 0:     0                     1
        Iteration 1:     256                   0
        Iteration 2:     0                     1
        Iteration 3:     256                   0

        This generally used to control multi-buffering in loops.
        """
        top_scope = self.scopes[-1]
        # assert top_scope.num_sections > 1, "The top scope only have one section, call close_scope instead."
        top_scope.cur_section_id = top_scope.cur_section_id + 1
        if top_scope.cur_section_id == top_scope.num_sections:
            self.logger.info("reset to section 0")
            top_scope.cur_section_id = 0
            self.stack_curr_addr = top_scope.starting_addr
        self.logger.info(f"Increment to section {top_scope.cur_section_id}, stack depth {self.scopes}")

    def close_scope(self):
        """
        Close the current stack scope. All tensors alloated within the scope will be freed
        """
        closing_scope = self.scopes[-1]

        self.stack_curr_addr = closing_scope.starting_addr
        self.scopes.pop()

    def alloc(self, shape, dtype, buffer=nl.sbuf, name=None, base_partition=0, align=1):
        """
        Allocate a tensor on the stack or the heap, depending on default allocation type.

        :param shape: shape of the tensor to be allocated
        :param dtype: dtype of the tensor to be allocated
        :param buffer: type of the buffer, currently only nl.sbuf is supported
        :param name: name of the tensor. Must be unique in the kernel
        :param base_partition: The base partition of the allocation, default to 0
        :param align: Alignment requirement of the address.
        :param is_stack: Whether to allocate on stack or heap.

        :return: a sbuf tensor described above.
        """
        if self.default_stack_alloc:
            return self.alloc_stack(
                shape,
                dtype,
                buffer=buffer,
                name=name,
                base_partition=base_partition,
                align=align,
            )
        else:
            return self.alloc_heap(shape, dtype, buffer=buffer, name=name, base_partition=base_partition)

    def alloc_stack(self, shape, dtype, buffer=nl.sbuf, name=None, base_partition=0, align=1):
        """
        Allocate a tensor on the stack which will be automatically freed when a scope closes.
        This method would raise an error if there are no open scope.

        :param shape: shape of the tensor to be allocated
        :param dtype: dtype of the tensor to be allocated
        :param buffer: type of the buffer, currently only nl.sbuf is supported
        :param name: name of the tensor. Must be unique in the kernel
        :param base_partition: The base partition of the allocation, default to 0
        :param align: Alignment requirement of the address.
        :return: a sbuf tensor described above.
        """
        assert buffer == nl.sbuf, "alloc_stack is only supported for SBUF tensors"
        N = _num_elts(shape[1:])
        bytes_per_partition = N * sizeinbytes(dtype)

        if self.stack_curr_addr + bytes_per_partition > self.heap_curr_addr:
            self.logger.error(
                f"Requested on stack: {bytes_per_partition}, " + f"free: {self.heap_curr_addr - self.stack_curr_addr}"
            )
            assert False

        if not self.scopes:
            self.logger.error("Cannot allocate in stack without an open scope")
            assert False

        self.stack_curr_addr = align_to(self.stack_curr_addr, align)
        if self.use_auto_alloc:
            mloc = nl.ndarray(shape=shape, dtype=dtype, buffer=buffer, name=name)
        else:
            mloc = nl.ndarray(
                shape=shape,
                dtype=dtype,
                buffer=buffer,
                name=name,
                address=(base_partition, self.stack_curr_addr),
            )
        self.stack_curr_addr = self.stack_curr_addr + bytes_per_partition
        self.stack_curr_addr = align_to(self.stack_curr_addr, align)
        self.logger.info(f"Allocating {bytes_per_partition}, current stack addr after {self.stack_curr_addr}")

        return mloc

    def alloc_heap(self, shape, dtype, buffer=nl.sbuf, name=None, base_partition=0):
        """
        Allocate a tensor on the heap.
        It will not be automatically freed when a scope closes,
        and must be released manually using pop_heap().

        :param shape: shape of the tensor to be allocated
        :param dtype: dtype of the tensor to be allocated
        :param buffer: type of the buffer, currently only nl.sbuf is supported
        :param name: name of the tensor. Must be unique in the kernel
        :param base_partition: The base partition of the allocation, default to 0
        :return: a sbuf tensor described above.
        """
        assert buffer == nl.sbuf, "alloc_heap is only supported for SBUF tensors"
        N = _num_elts(shape[1:])
        bytes_per_partition = N * sizeinbytes(dtype)

        if self.stack_curr_addr + bytes_per_partition > self.heap_curr_addr:
            self.logger.error(
                f"Requested on heap: {bytes_per_partition}, " f"free: {self.heap_curr_addr - self.stack_curr_addr}"
            )
            assert False

        base_addr = self.heap_curr_addr - bytes_per_partition
        self.heap_curr_addr -= bytes_per_partition
        self.heap_curr_addr = align_to(self.heap_curr_addr - 3, 4)  # heap grows down, so should the align

        if self.use_auto_alloc:
            mloc = nl.ndarray(shape=shape, dtype=dtype, buffer=buffer, name=name)
        else:
            mloc = nl.ndarray(
                shape=shape,
                dtype=dtype,
                buffer=buffer,
                name=name,
                address=(base_partition, base_addr),
            )
        self.logger.info(f"Allocating {bytes_per_partition}, current heap addr after {self.heap_curr_addr}")
        self.heap.append(mloc)

        return mloc

    def pop_heap(self):
        if not self.heap:
            self.logger.error("Invalid pop, heap is empty")
            assert False

        heap_top = self.heap[-1]
        N = _num_elts(heap_top.shape[1:])
        # Note to FE: the nl.ndarray or the sbuf.ptr should have a way of querying the shape
        bytes_per_partition = N * sizeinbytes(heap_top.dtype)
        self.heap_curr_addr = self.heap_curr_addr + bytes_per_partition
        self.heap_curr_addr = align_to(self.heap_curr_addr - 3, 4)
        self.logger.info(f"Releasing {bytes_per_partition}, current heap addr after {self.heap_curr_addr}")
        self.heap.pop()

    def get_free_space(self):
        if self.use_auto_alloc:
            self.logger.error("get_free_space() is not supported in auto-allocation mode.")
            assert False
        return self.heap_curr_addr - self.stack_curr_addr

    def get_stack_curr_addr(self):
        if self.use_auto_alloc:
            self.logger.error("get_stack_curr_addr() is not supported in auto-allocation mode.")
            assert False
        return self.stack_curr_addr

    def get_heap_curr_addr(self):
        if self.use_auto_alloc:
            self.logger.error("get_heap_curr_addr() is not supported in auto-allocation mode.")
            assert False
        return self.heap_curr_addr

    def align_stack_curr_addr(self, align=32):
        if self.use_auto_alloc:
            self.logger.error("align_stack_curr_addr() is not supported in auto-allocation mode.")
            assert False
        self.stack_curr_addr = align_to(self.stack_curr_addr, align)


def create_auto_alloc_manager(logger: Optional[Logger] = None):
    """create a default auto allocated SBM initialized with total SBUF space"""
    return SbufManager(0, nl.tile_size.total_available_sbuf_size, logger, use_auto_alloc=True)
