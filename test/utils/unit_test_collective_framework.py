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

"""Collective Unit Test Framework for multi-rank NKI kernel tests.

Extends UnitTestFramework validation to collective tests with parallel
torch_ref execution via multiprocessing, which can be switched to a real
backend(gloo/nccl) once it is available in future PyTorch dependency.
"""

import logging
import multiprocessing as mp
import os
import pickle
import shutil
import tempfile
import time
import tracemalloc
from typing import Callable, Optional

import torch
from nki.collectives import ReplicaGroup
from nkilib_src.nkilib.experimental.collectives.distributed_adapter import (
    SimDistAdapter,
    replica_group_key,
    set_adapter,
)
from torch.distributed import ProcessGroup, Work

from .common_dataclasses import (
    CompilerArgs,
    InferenceArgs,
    KernelArgs,
    NamedCallable,
    PerRankGenerator,
    PerRankLazyGoldenGenerator,
    PerRankLazyInputGenerator,
    ValidationArgs,
)
from .metadata_loader import load_model_configs
from .metrics_collector import IMetricsCollector, MetricName
from .test_orchestrator import Orchestrator
from .unit_test_framework import (
    check_unused_parameters,
    filter_kernel_input,
    filter_ref_input,
    validate_input_keys,
    validate_torch_ref_signature,
)

# ==================== SimDistRunner ====================


class _SimWork(Work):
    """No-op Work object for synchronous collective ops."""

    def is_completed(self):
        return True

    def wait(self, timeout=None):
        return True


class _StopAfterCollective(Exception):
    """Raised to abort torch ref after recording a collective's send data."""

    pass


class _MPSimProcessGroup(ProcessGroup):
    """ProcessGroup with multi-pass support for memory-efficient execution.

    Modes:
    - record (stop_after_call=K): run torch ref, record send for call K, raise _StopAfterCollective
    - replay (stop_after_call=None): all collective data on disk, read without waiting
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        coll_dir: str,
        group_id: str,
        global_counter: list,
        stop_after_call: int | None = None,
    ):
        # The base class declares only a longer form whose first parameter is a store. The
        # two-positional form used here is the one the runtime accepts for a Python subclass;
        # constructing it the declared way fails at runtime.
        super().__init__(rank, world_size)  # ty: ignore[missing-argument, invalid-argument-type]
        self._coll_dir = coll_dir
        self._group_id = group_id
        self._call_idx = 0
        self._stop_after_call = stop_after_call  # None = replay all (final pass)
        self._global_counter = global_counter  # shared [int] across all PGs for this rank

    def _file_path(self, call_id, r):
        return os.path.join(self._coll_dir, f"{self._group_id}_c{call_id}_r{r}.pkl")

    def _put_and_sync(self, data):
        self._call_idx += 1

        # Use global counter for file naming (consistent across all PGs)
        global_idx = self._global_counter[0]
        self._global_counter[0] += 1

        # Write this rank's send data (atomic) — skip if already exists from earlier pass
        path = self._file_path(global_idx, self.rank())
        if not os.path.exists(path):
            tmp_path = path + ".tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.rename(tmp_path, path)
        else:
            logging.debug(f"SimPG rank {self.rank()}: skipped redundant write for call {global_idx}")

        # If this is the target call for recording, stop execution
        if self._stop_after_call is not None and global_idx == self._stop_after_call:
            raise _StopAfterCollective(f"rank {self.rank()} recorded global call {global_idx}")

        return global_idx

    def _read_all(self, call_id):
        # Wait for all ranks' files (they should already exist from earlier passes)
        t0 = time.time()
        while True:
            missing = [r for r in range(self.size()) if not os.path.exists(self._file_path(call_id, r))]
            if not missing:
                break
            if time.time() - t0 > 600:
                raise RuntimeError(f"SimPG rank {self.rank()}: timeout waiting for {missing} at call {call_id}")
            time.sleep(0.01)

        results = {}
        for r in range(self.size()):
            with open(self._file_path(call_id, r), "rb") as f:
                results[r] = pickle.load(f)
        return results

    # The collective methods below override overload sets on the base class: each op declares a
    # list-of-tensors form and a single-tensor convenience form whose parameters differ in both
    # name and type. One implementation cannot be compatible with every overload, so the type
    # checker reports these overrides. They implement the list-based positional form, which is the
    # only form the collective wrappers and the torch reference kernels dispatch through -- for
    # example all_reduce calls allreduce([tensor], opts), and all_gather calls
    # allgather([tensor_list], [tensor], opts). No caller passes these arguments by keyword.
    def allreduce(self, tensors, opts=None):  # ty: ignore[invalid-method-override]
        call_id = self._put_and_sync(tensors[0].clone())
        data = self._read_all(call_id)
        tensors[0].copy_(sum(data[r] for r in range(self.size())))
        return _SimWork()

    def allgather(self, output_tensors_list, input_tensors, opts=None):  # ty: ignore[invalid-method-override]
        call_id = self._put_and_sync(input_tensors[0].clone())
        data = self._read_all(call_id)
        for r in range(self.size()):
            output_tensors_list[0][r].copy_(data[r])
        return _SimWork()

    def reduce_scatter(self, output_tensors, input_tensors_list, opts=None):  # ty: ignore[invalid-method-override]
        call_id = self._put_and_sync([t.clone() for t in input_tensors_list[0]])
        data = self._read_all(call_id)
        result = sum(data[src][self.rank()] for src in range(self.size()))
        output_tensors[0].copy_(result)
        return _SimWork()

    def alltoall(self, output_tensors, input_tensors, opts=None):  # ty: ignore[invalid-method-override]
        call_id = self._put_and_sync([t.clone() for t in input_tensors])
        data = self._read_all(call_id)
        for r in range(self.size()):
            output_tensors[r].copy_(data[r][self.rank()])
        return _SimWork()

    def alltoall_base(  # ty: ignore[invalid-method-override]
        self, output, input, output_split_sizes=None, input_split_sizes=None, opts=None
    ):
        """Variable-length all-to-all (backing dist.all_to_all_single)."""
        if not input_split_sizes:
            chunk_size = input.size(0) // self.size()
            input_split_sizes = [chunk_size] * self.size()
        # Send our input with split info
        call_id = self._put_and_sync((input.clone(), list(input_split_sizes)))
        data = self._read_all(call_id)
        # Each src sent us data[src].input[offset_for_us : offset_for_us + src_split_for_us]
        chunks = []
        for src in range(self.size()):
            src_input, src_splits = data[src]
            offset = sum(src_splits[: self.rank()])
            size = src_splits[self.rank()]
            chunks.append(src_input[offset : offset + size])
        output.copy_(torch.cat(chunks) if chunks else output.new_empty(0))
        return _SimWork()


def _log_vm_size(rank, label):
    """Log virtual memory size from /proc/self/status."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmSize:") or line.startswith("VmRSS:"):
                    logging.debug(f"SimDistRunner rank {rank} ({label}): {line.strip()}")
    except OSError:
        pass


class SimDistRunner:
    """Memory-efficient multi-pass collective torch ref runner.

    Splits execution into passes to bound peak memory to 1 rank at a time:
    - Pass 0 (parallel): all ranks record collective_0 sends, exit
    - Pass 1..N-1 (sequential): each rank replays earlier collectives, computes
      through next collective, records send, exits
    - Final pass (sequential): each rank replays all collectives, runs to completion

    Peak memory = max(1 rank's heaviest phase). No swap needed.
    """

    MAX_COLLECTIVE_PASSES = 10  # safety limit
    MAX_PARALLEL_RANKS = 8  # default; reduced dynamically if per-rank memory is high
    PARALLEL_RANK_MEMORY_THRESHOLD_MB = 500  # if per-rank peak > this, reduce to 4 parallel

    def __init__(self, num_ranks: int, replica_groups: list | None = None):
        self._num_ranks = num_ranks
        if replica_groups is None:
            replica_groups = [ReplicaGroup([list(range(num_ranks))])]
        self._replica_groups = replica_groups

    def _make_pg_map(self, rank, coll_dir, stop_after_call):
        """Create process group map for a rank with given stop_after_call."""
        pg_map = {}
        global_counter = [0]
        for rg in self._replica_groups:
            key = replica_group_key(rg)
            for group_ranks in rg._value:
                if rank in group_ranks:
                    group_rank = group_ranks.index(rank)
                    group_id = str(abs(hash((key, tuple(group_ranks)))) % 10**8)
                    pg_map[key] = _MPSimProcessGroup(
                        group_rank,
                        len(group_ranks),
                        coll_dir,
                        group_id,
                        stop_after_call=stop_after_call,
                        global_counter=global_counter,
                    )
        return pg_map

    def _run_rank(self, rank, torch_ref, per_rank_inputs, coll_dir, stop_after_call):
        """Run a single rank. Returns result dict or None if stopped early."""

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

        torch.set_num_threads(1)

        pg_map = self._make_pg_map(rank, coll_dir, stop_after_call)
        set_adapter(SimDistAdapter(rank, pg_map))
        try:
            result = torch_ref(**per_rank_inputs[rank])
            return result
        except _StopAfterCollective:
            return None  # expected — rank recorded its send and stopped
        finally:
            set_adapter(None)

    def run(self, torch_ref: Callable, per_rank_inputs: dict) -> dict:
        """Run torch_ref across all ranks with multi-pass memory optimization."""

        ctx = mp.get_context("fork")
        coll_dir = tempfile.mkdtemp(prefix="sim_coll_")
        result_dir = tempfile.mkdtemp(prefix="sim_result_")

        def _worker_record(rank, stop_after_call, error_dict):
            """Worker for recording pass — runs torch ref until stop_after_call."""
            try:
                _log_vm_size(rank, f"pass{stop_after_call} start")
                tracemalloc.start()
                self._run_rank(rank, torch_ref, per_rank_inputs, coll_dir, stop_after_call)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                _log_vm_size(rank, f"pass{stop_after_call} end")
                with open(os.path.join(result_dir, f"pass{stop_after_call}_rank{rank}_mem.txt"), "w") as f:
                    f.write(str(peak))
            except Exception:
                import traceback

                error_dict[rank] = traceback.format_exc()

        def _worker_final(rank, error_dict):
            """Worker for final pass — runs torch ref to completion."""
            try:
                tracemalloc.start()
                result = self._run_rank(rank, torch_ref, per_rank_inputs, coll_dir, stop_after_call=None)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                with open(os.path.join(result_dir, f"rank_{rank}.pkl"), "wb") as f:
                    pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(result_dir, f"rank_{rank}_mem.txt"), "w") as f:
                    f.write(str(peak))
            except Exception:
                import traceback

                error_dict[rank] = traceback.format_exc()

        # Discover number of collectives by doing a dry run of rank 0
        # Pass 0: run all ranks in parallel, stop after collective 0
        for pass_idx in range(self.MAX_COLLECTIVE_PASSES):
            error_dict = ctx.Manager().dict()

            # Probe with a single rank first to check if this collective exists
            probe_error = ctx.Manager().dict()
            probe_p = ctx.Process(target=_worker_record, args=(0, pass_idx, probe_error))
            probe_p.start()
            probe_p.join()

            if probe_p.exitcode and probe_p.exitcode < 0:
                import signal as _signal

                sig = -probe_p.exitcode
                sig_name = _signal.Signals(sig).name if sig in _signal.Signals._value2member_map_ else str(sig)
                probe_error[0] = f"probe rank 0 killed by signal {sig_name} ({sig})"

            if probe_error:
                shutil.rmtree(coll_dir, ignore_errors=True)
                shutil.rmtree(result_dir, ignore_errors=True)
                raise RuntimeError(f"SimDistRunner pass {pass_idx} probe failed:\n" + "\n".join(probe_error.values()))

            # Check if rank 0 hit this collective
            any_coll_written = any(
                os.path.exists(os.path.join(coll_dir, f))
                for f in os.listdir(coll_dir)
                if f.endswith(f"_c{pass_idx}_r0.pkl")
            )
            if not any_coll_written:
                # No collective at this index — all collectives discovered
                logging.debug(f"SimDistRunner: no collective {pass_idx} found, discovery complete")
                break

            # Collective exists — run remaining ranks in parallel to record their sends
            logging.debug(
                f"SimDistRunner: pass {pass_idx} — recording collective {pass_idx} (parallel, ranks 1..{self._num_ranks - 1})"
            )
            # Run ranks in batches to avoid OOM from too many forked processes
            remaining_ranks = list(range(1, self._num_ranks))
            for batch_start in range(0, len(remaining_ranks), self.MAX_PARALLEL_RANKS):
                batch = remaining_ranks[batch_start : batch_start + self.MAX_PARALLEL_RANKS]
                processes = []
                for rank in batch:
                    p = ctx.Process(target=_worker_record, args=(rank, pass_idx, error_dict))
                    processes.append((rank, p))
                    p.start()
                for _rank, p in processes:
                    p.join()
                if p.exitcode and p.exitcode < 0:
                    import signal

                    sig = -p.exitcode
                    sig_name = signal.Signals(sig).name if sig in signal.Signals._value2member_map_ else str(sig)
                    error_dict[rank] = f"rank {rank} killed by signal {sig_name} ({sig})"
                elif p.exitcode and p.exitcode > 0 and rank not in error_dict:
                    error_dict[rank] = f"rank {rank} exited with code {p.exitcode}"

            if error_dict:
                shutil.rmtree(coll_dir, ignore_errors=True)
                shutil.rmtree(result_dir, ignore_errors=True)
                raise RuntimeError(f"SimDistRunner pass {pass_idx} failed:\n" + "\n".join(error_dict.values()))

            # Summarize parallel pass peak memory
            pass_peaks = []
            for rank in range(self._num_ranks):
                mem_file = os.path.join(result_dir, f"pass{pass_idx}_rank{rank}_mem.txt")
                if os.path.exists(mem_file):
                    with open(mem_file) as f:
                        pass_peaks.append(int(f.read()) / 1e6)
            if pass_peaks:
                logging.debug(
                    f"SimDistRunner: parallel pass {pass_idx} summary — "
                    f"max={max(pass_peaks):.1f} MB, total={sum(pass_peaks):.1f} MB across {len(pass_peaks)} ranks"
                )
                # After pass 0, adapt parallel limit based on measured per-rank memory
                if pass_idx == 0 and max(pass_peaks) > self.PARALLEL_RANK_MEMORY_THRESHOLD_MB:
                    self.MAX_PARALLEL_RANKS = 4
                    logging.info(
                        f"SimDistRunner: per-rank peak {max(pass_peaks):.0f} MB > {self.PARALLEL_RANK_MEMORY_THRESHOLD_MB} MB threshold, "
                        f"reducing MAX_PARALLEL_RANKS to {self.MAX_PARALLEL_RANKS}"
                    )

        # Final pass: run each rank sequentially to completion (replay all collectives)
        error_dict = ctx.Manager().dict()
        for rank in range(self._num_ranks):
            p = ctx.Process(target=_worker_final, args=(rank, error_dict))
            p.start()
            p.join()  # sequential — wait for each rank to finish before starting next

            if error_dict:
                shutil.rmtree(coll_dir, ignore_errors=True)
                shutil.rmtree(result_dir, ignore_errors=True)
                raise RuntimeError(f"SimDistRunner final pass rank {rank} failed:\n" + "\n".join(error_dict.values()))

        # Read results
        results = {}
        for rank in range(self._num_ranks):
            with open(os.path.join(result_dir, f"rank_{rank}.pkl"), "rb") as f:
                results[rank] = pickle.load(f)
            mem_file = os.path.join(result_dir, f"rank_{rank}_mem.txt")
            if os.path.exists(mem_file):
                with open(mem_file) as f:
                    peak_bytes = int(f.read())
                logging.info(f"SimDistRunner rank {rank} peak memory (sequential final): {peak_bytes / 1e6:.1f} MB")

        shutil.rmtree(coll_dir, ignore_errors=True)
        shutil.rmtree(result_dir, ignore_errors=True)
        return results


# ==================== Validation Helpers ====================


def validate_cross_rank_consistency(
    per_rank_input_generator: PerRankGenerator, rank0_input: dict, num_ranks: int
) -> None:
    """Validate input tensor shapes/dtypes are consistent across all ranks."""
    import numpy as np

    for rank_id in range(1, num_ranks):
        rank_input = per_rank_input_generator(rank_id=rank_id)
        for key in rank0_input:
            v0, vr = rank0_input[key], rank_input.get(key)
            if isinstance(v0, np.ndarray) and isinstance(vr, np.ndarray):
                if v0.shape != vr.shape:
                    raise ValueError(
                        f"Input tensor '{key}' shape mismatch across ranks: rank 0 {v0.shape} vs rank {rank_id} {vr.shape}"
                    )
                if v0.dtype != vr.dtype:
                    raise ValueError(
                        f"Input tensor '{key}' dtype mismatch across ranks: rank 0 {v0.dtype} vs rank {rank_id} {vr.dtype}"
                    )


# ==================== Parallel Runner ====================


def _run_torch_refs_parallel(torch_ref, per_rank_inputs: dict, num_ranks: int) -> dict:
    """Run a collective torch_ref across all ranks.

    If torch.distributed is initialized (real hardware), runs torch_ref directly
    for this rank. Otherwise uses SimDistRunner to simulate multi-rank in threads.
    """
    import torch.distributed as dist

    if dist.is_initialized():
        # Real distributed mode — this process IS one rank, just run directly
        rank = dist.get_rank()
        return {rank: torch_ref(**per_rank_inputs[rank])}

    # Simulation mode — run all ranks in parallel threads
    from nki.collectives import ReplicaGroup

    replica_groups = []
    for v in per_rank_inputs[0].values():
        if isinstance(v, ReplicaGroup):
            replica_groups.append(v)

    sim = SimDistRunner(num_ranks=num_ranks, replica_groups=replica_groups or None)
    return sim.run(torch_ref, per_rank_inputs)


# ==================== CollectiveUnitTestFramework ====================


class CollectiveUnitTestFramework:
    """Framework for executing multi-rank collective NKI kernel tests.

    Extends the validation features of UnitTestFramework to collective (multi-rank) tests:
    - Signature validation (kernel_entry ↔ torch_ref)
    - .must_alias_input handling
    - Lazy per-rank golden generation (skipped in compile-only mode)
    - Cross-rank shape/dtype consistency check
    - Per-rank torch_ref input override (for KVDP-style tests)
    - Per-rank custom validation via callable custom_comparator
    """

    def __init__(
        self,
        test_manager: Orchestrator,
        kernel_entry: NamedCallable,
        torch_ref: Callable,
        per_rank_input_generator: PerRankGenerator,
        collective_ranks: int,
        check_unused_params: bool = True,
        collector: Optional[IMetricsCollector] = None,
    ):
        validate_torch_ref_signature(kernel_entry, torch_ref)
        if check_unused_params:
            check_unused_parameters(kernel_entry)

        self.test_manager = test_manager
        self.kernel_entry = kernel_entry
        self.torch_ref = torch_ref
        self.per_rank_input_generator = per_rank_input_generator
        self.collective_ranks = collective_ranks
        # Default to the orchestrator's collector, which is a NoopMetricsCollector
        # when metrics are disabled.
        self.collector = collector if collector is not None else test_manager.collector

    def run_test(
        self,
        test_config,
        compiler_args: CompilerArgs,
        output_keys: list,
        rtol: float = 1e-2,
        atol: float = 1e-2,
        inference_args: Optional[InferenceArgs] = None,
        custom_comparator: Optional[Callable] = None,
        metadata: Optional[dict] = None,
        golden_only: bool = False,
        profile_only: bool = False,
        input_artifacts_directory: Optional[str] = None,
    ):
        if metadata is not None:
            metadata_list = load_model_configs(metadata["config_name"])
            self.collector.match_and_add_metadata_dimensions(metadata["key"], metadata_list)

        # Validate rank-0 inputs
        rank0_input = self.per_rank_input_generator(rank_id=0)
        validate_input_keys(rank0_input, self.kernel_entry)

        # Cross-rank shape/dtype consistency check
        if self.collective_ranks >= 2 and not profile_only:
            validate_cross_rank_consistency(self.per_rank_input_generator, rank0_input, self.collective_ranks)

        # Build per-rank input generator with .must_alias_input filtering
        def _filtered_input_generator(rank_id: int) -> dict:
            raw = self.per_rank_input_generator(rank_id=rank_id)
            return filter_kernel_input(raw, self.kernel_entry)

        # Build per-rank golden via torch_ref.
        # Run all ranks in parallel processes so torch_refs
        # can communicate via torch.distributed API.
        torch_ref = self.torch_ref

        def _build_ref_input(rank_id: int) -> dict:
            raw_input = self.per_rank_input_generator(rank_id=rank_id)
            return filter_ref_input(raw_input, torch_ref)

        if not profile_only:
            per_rank_ref_inputs = {r: _build_ref_input(r) for r in range(self.collective_ranks)}

            def _generate_all_golden():
                """Lazy: only run multi-process golden gen when first rank is requested."""
                # Time just the reference compute as GoldenComputationTime (the collective
                # path has no golden cache, so this always runs). The comparator wrapping
                # below is validation, not golden compute, so it stays outside the timer.
                with self.collector.timer(MetricName.GOLDEN_COMPUTATION_TIME):
                    result = _run_torch_refs_parallel(torch_ref, per_rank_ref_inputs, self.collective_ranks)
                if custom_comparator is not None:
                    result = {r: custom_comparator(r, g) for r, g in result.items()}
                return result

        if golden_only:
            if profile_only:
                raise ValueError("golden_only and profile_only cannot be enabled together")
            return _generate_all_golden()

        # Resolve validation args.
        # In compile-only/trace-only mode, skip golden gen entirely.
        # Otherwise, generate golden EAGERLY (before execute) so the fork happens
        # before the compiler is loaded into memory (avoids OOM from large parent process).
        from .common_dataclasses import TraceMode

        is_compile_only = hasattr(self.test_manager, "trace_mode") and self.test_manager.trace_mode in (
            TraceMode.CompileOnly,
            TraceMode.TraceOnly,
        )
        if profile_only or (is_compile_only and output_keys):
            validation_args = None
        else:
            _cached_golden = {}

            def _golden_generator(rank_id: int) -> dict:
                if not _cached_golden:
                    _cached_golden.update(_generate_all_golden())
                return _cached_golden[rank_id]

            golden_output = PerRankLazyGoldenGenerator(_golden_generator, output_keys=output_keys or [])
            validation_args = ValidationArgs(
                golden_output=golden_output,
                relative_accuracy=rtol,
                absolute_accuracy=atol,
            )

        per_rank_input = PerRankLazyInputGenerator(
            _filtered_input_generator,
            input_artifacts_directory=input_artifacts_directory,
        )
        per_rank_input.base_input = rank0_input

        kernel_args = KernelArgs(
            kernel_func=self.kernel_entry,
            compiler_input=compiler_args,
            kernel_input=per_rank_input,
            inference_args=inference_args or InferenceArgs(collective_ranks=self.collective_ranks),
            validation_args=validation_args,
        )
        self.test_manager.execute(kernel_args)
