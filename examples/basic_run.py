from quport import LatencyModel, MultiQPUConfig, map_and_transpile
from quport.pipeline import random_benchmark_circuit

cfg = MultiQPUConfig(
    n_qpus=10,
    compute_qubits_per_qpu=8,
    comm_qubits_per_qpu=1,
    intra_topology="clique",
    inter_topology="switch",
)

qc = random_benchmark_circuit(n_logical=80, depth=20, seed=7)
res = map_and_transpile(qc, cfg, latency=LatencyModel(), seed=7)

print("SWAPs:", res.metrics.swaps)
print("Remote2Q:", res.metrics.remote_2q)
print("Cost:", res.cost.total)
