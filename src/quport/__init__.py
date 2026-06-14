# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""QuPort: Multi-QPU mapping and benchmarking toolkit."""

from quport.architecture import MultiQPUArchitecture
from quport.compiler import DistributedCompileResult, compile_distributed
from quport.config import LatencyModel, MultiQPUConfig
from quport.distributed import (
    DistributedProgram,
    RemoteOp,
    split_into_qpus,
    write_remote_ops_json,
)
from quport.pipeline import (
    benchmark_random_circuits,
    map_and_transpile,
    sweep_topologies,
)
from quport.schedule import (
    LayerScheduleTrace,
    RemoteRoundTrace,
    ScheduleSummary,
    TopologySchedulePlan,
    TopologyScheduleSummary,
    estimate_parallel_makespan,
    estimate_parallel_makespan_layered,
    estimate_parallel_makespan_topology,
    estimate_topology_schedule_plan,
)

__all__ = [
    "DistributedCompileResult",
    "DistributedProgram",
    "LatencyModel",
    "MultiQPUArchitecture",
    "MultiQPUConfig",
    "LayerScheduleTrace",
    "RemoteOp",
    "RemoteRoundTrace",
    "ScheduleSummary",
    "TopologySchedulePlan",
    "TopologyScheduleSummary",
    "benchmark_random_circuits",
    "compile_distributed",
    "estimate_parallel_makespan",
    "estimate_parallel_makespan_layered",
    "estimate_parallel_makespan_topology",
    "estimate_topology_schedule_plan",
    "map_and_transpile",
    "split_into_qpus",
    "write_remote_ops_json",
    "sweep_topologies",
]
