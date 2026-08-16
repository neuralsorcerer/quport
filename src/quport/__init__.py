# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""QuPort: Multi-QPU mapping and benchmarking toolkit."""

from quport.aggregation import (
    AggregationPlan,
    RemoteBlock,
    aggregate_remote_operations,
)
from quport.architecture import MultiQPUArchitecture
from quport.compiler import DistributedCompileResult, compile_distributed
from quport.config import LatencyModel, MultiQPUConfig
from quport.distributed import (
    DistributedProgram,
    RemoteOp,
    split_into_qpus,
    write_distributed_program,
    write_remote_ops_json,
)
from quport.entanglement import diagonal_positions
from quport.hypergraph import (
    DistributablePacket,
    EbitReport,
    PacketDecomposition,
    build_distributable_packets,
    ebit_cost,
    ebit_objective,
    ebit_report,
    ebit_traffic_matrix,
)
from quport.network import TopologyMetrics, topology_metrics
from quport.pipeline import (
    benchmark_random_circuits,
    map_and_transpile,
    sweep_topologies,
)
from quport.schedule import (
    EntanglementScheduleSummary,
    LayerScheduleTrace,
    RemoteRoundTrace,
    ScheduleSummary,
    TopologySchedulePlan,
    TopologyScheduleSummary,
    estimate_entanglement_schedule,
    estimate_parallel_makespan,
    estimate_parallel_makespan_layered,
    estimate_parallel_makespan_topology,
    estimate_topology_schedule_plan,
)

__all__ = [
    "AggregationPlan",
    "DistributablePacket",
    "DistributedCompileResult",
    "DistributedProgram",
    "EbitReport",
    "EntanglementScheduleSummary",
    "LatencyModel",
    "MultiQPUArchitecture",
    "MultiQPUConfig",
    "LayerScheduleTrace",
    "PacketDecomposition",
    "RemoteBlock",
    "RemoteOp",
    "RemoteRoundTrace",
    "ScheduleSummary",
    "TopologySchedulePlan",
    "TopologyMetrics",
    "TopologyScheduleSummary",
    "aggregate_remote_operations",
    "benchmark_random_circuits",
    "build_distributable_packets",
    "compile_distributed",
    "diagonal_positions",
    "ebit_cost",
    "ebit_objective",
    "ebit_report",
    "ebit_traffic_matrix",
    "estimate_entanglement_schedule",
    "estimate_parallel_makespan",
    "estimate_parallel_makespan_layered",
    "estimate_parallel_makespan_topology",
    "estimate_topology_schedule_plan",
    "map_and_transpile",
    "split_into_qpus",
    "write_distributed_program",
    "write_remote_ops_json",
    "sweep_topologies",
    "topology_metrics",
]
