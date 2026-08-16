# Entanglement model

Cut weight — the number of two-qubit gates whose operands land on different QPUs —
is the objective every classical partitioner minimizes, and it is not the quantity a
cat-entanglement machine pays. This page describes the model QuPort uses instead,
the three modules that implement it, and how they fit together.

| Module | Question it answers | When it runs |
|---|---|---|
| `quport.entanglement` | Which operand positions of an operation commute with `Z`? | Everywhere, as the shared correctness rule |
| `quport.hypergraph` | How many EPR pairs does this *partition* need? | Partition time, inside the search loop |
| `quport.aggregation` | Which EPR blocks does this *mapped circuit* need, under a real port budget? | Compile time, on physical qubits |
| `quport.schedule.estimate_entanglement_schedule` | How long does that plan take on this interconnect? | Schedule time |

## Why cat-entanglement changes the objective

A remote two-qubit gate is not executed by moving quantum state between QPUs. One
EPR pair is distributed, and QPU `B` builds a *cat copy* `b` of a root qubit `c`
living on QPU `A`:

1. Distribute an EPR pair, half on each QPU.
2. `A` applies `CX(c -> a)`, measures `a` in the `Z` basis, and sends the outcome;
   `B` applies a conditional `X`. The joint state is now
   `sum_z alpha_z |z>_c |z>_b`, so `b` carries `c`'s computational-basis label.
3. Every gate that uses `c` only through that label runs **locally on `B`** against
   `b` — one gate, ten gates, as many as the copy survives.
4. Cat-disentangler: measure `b` in the `X` basis, send the outcome back, apply a
   conditional `Z` to `c`.

Step 3 is the whole point, and it is correct exactly when every operation applied to
`c` while the copy is live commutes with `Z_c`. Such an operation maps `|z>_c |psi>`
to `|z>_c U_z|psi>`, so the label correspondence survives. An `X`, `H`, `SX`, or a
`CX` that uses `c` as its *target* destroys it and the copy must be released first.

Ten gates from one control into one QPU therefore cost ten units of cut weight and
**one** EPR pair.

## Diagonality analysis

`quport.entanglement.diagonal_positions(operation)` returns the operand positions on
which an operation commutes with `Z`, from three rules applied in order:

1. an explicit table for operations whose structure is not otherwise derivable
   (`rzz` on both operands, `rzx` on operand 0, `swap`/`ecr`/`rxx` on neither);
2. `ControlledGate` structure — every control operand is diagonal regardless of
   `ctrl_state`, and because `C(U) = P_0 (x) I + P_1 (x) U`, every operand that is
   diagonal for the base gate stays diagonal once controlled;
3. a table of diagonal single-qubit gates (`z`, `s`, `t`, `rz`, `p`, …).

Anything else is reported as non-diagonal on every operand. That direction of error
is the safe one: it can only over-count EPR pairs, never claim a copy that would not
survive. `tests/test_entanglement.py` checks every claim against the actual unitary
of every constructible gate in Qiskit's standard library.

## Distributable packets and the λ−1 metric

A **distributable packet** rooted at qubit `c` is a maximal run of gates over which
`c` stays diagonal. Diagonality is a property of the gate sequence alone, so packets
do not depend on placement: `build_distributable_packets` runs once per circuit, and
`ebit_cost` then evaluates any candidate partition in time linear in the number of
packet incidences. That is what makes the metric affordable inside annealing.

The EPR count for partition `pi` is the connectivity-minus-one (λ−1) metric of
hypergraph partitioning: one e-bit per packet per *distinct remote QPU* its partners
occupy.

Two kinds of gate cannot be served by one bipartite copy: two-qubit gates with no
diagonal operand, and operations on three or more qubits. A gate of either kind
spanning `k` QPUs is charged `2 * (k - 1)` e-bits — teleport each foreign operand to
one host and back.

Each gate is charged to exactly one root, so the count is exact for that assignment
and an upper bound over all assignments. Gates with two diagonal operands (`cz`,
`cp`, `crz`, `rzz`) admit a choice, controlled by `symmetric_root`:

| Policy | Behaviour |
|---|---|
| `"greedy"` (default) | reuse an operand that already roots an open packet; ties fall back to the lower qubit index |
| `"min_index"` | always the lower qubit index |
| `"first_operand"` | always operand 0, as written in the circuit |

## The `ebit` partitioning strategy

`ebit` runs the same TPCCAP-plus-annealing search as `tpccap_sa`, with the
weighted-cut-distance term replaced by hop-scaled e-bit demand: each e-bit is charged
the QPU-graph distance it has to cross, since entanglement swapping consumes a link
per hop. On an all-to-all fabric every distance is 1 and the term is exactly the λ−1
count. Port and congestion terms are unchanged.

`w_ebit` defaults to `0.0` on `tpccap_partition` and `tpccap_sa_partition`, so every
pre-existing objective is untouched. Passing `packets` with `w_ebit=0` fills in the
`ebits` and `weighted_ebit_distance` diagnostics without steering the search.

```{note}
Congestion is still estimated from gate-level traffic, which upper-bounds e-bit
traffic because aggregation only ever removes transactions. Use
`ebit_traffic_matrix` when the exact EPR demand matters.
```

## Communication aggregation

The λ−1 metric assumes comm ports are free. `aggregate_remote_operations` answers the
same question on a mapped circuit where they are not:

- a QPU with `P` ports hosts at most `P` cat copies at once;
- opening a block also needs a free port on the *root's* QPU, to run the entangler;
- when a port is needed and none is free, the least recently used copy is released,
  and a fresh EPR pair is spent if that root comes back. The plan reports those as
  `evictions`.

The two computations are independent implementations of the same quantity. With an
unbounded port budget `aggregate_remote_operations(...).epr_pairs` equals
`ebit_cost(...)` exactly, which
`tests/test_aggregation.py::test_unbounded_ports_match_hypergraph_ebits` pins down on
compiled random circuits.

`max_block_gates` caps how many gates one block may serve, which trades EPR savings
against how long a port stays pinned.

## Entanglement-aware scheduling

`estimate_entanglement_schedule` drops the DAG-layer abstraction. Layers impose a
global barrier between successive slices and charge one transaction per cross-QPU
gate; this estimator runs an as-soon-as-possible list schedule in program order over
explicit resources:

- one timeline per physical qubit, so QPUs sharing no qubits drift apart freely;
- a pool of `comm_qubits_per_qpu` ports per QPU, each held for a **whole block**;
- `link_capacity` channels on each link along the routed path;
- hop-scaled, probabilistic distribution: heralded entanglement needs `1 / p`
  attempts in expectation, so time scales as `hops * epr_gen / epr_success_prob`.

```{warning}
A plan and the schedule that consumes it must agree on the port budget. Passing a
plan built with a larger `ports_per_qpu` raises rather than silently reporting the
work as unschedulable.
```

## Reading the numbers

| Field | Where | Meaning |
|---|---|---|
| `ebits` | `EbitReport` | EPR pairs with unlimited ports — a lower bound on the plan |
| `epr_pairs` | `AggregationPlan` | EPR pairs under the real port budget |
| `baseline_epr_pairs` | `AggregationPlan` | what a per-gate telegate compiler would spend |
| `reduction` | both | fraction saved against that baseline |
| `evictions` | `AggregationPlan` | copies released early because a port was needed |
| `peak_cat_copies` | `AggregationPlan` | never exceeds the QPU's port budget, by construction |
| `entanglement_time` | `EntanglementScheduleSummary` | total link occupancy, summed over links |
| `unschedulable_gates` | `EntanglementScheduleSummary` | gates no port/link budget could serve |

Structured circuits gain most: a control that only picks up `Rz` rotations keeps its
packet open across a whole ladder. Random circuits gain less, because translating to
the default basis puts `sx` and `x` gates on most qubits and each one closes a
packet.

## Example

```python
from quport import MultiQPUConfig, compile_distributed
from quport.pipeline import random_benchmark_circuit

cfg = MultiQPUConfig(
    n_qpus=4,
    compute_qubits_per_qpu=4,
    comm_qubits_per_qpu=2,
    inter_topology="switch",
    optimization_level=0,
)

result = compile_distributed(
    random_benchmark_circuit(n_logical=16, depth=20, seed=0),
    cfg,
    seed=0,
    strategy="ebit",
)

print(result.ebits.ebits)                      # port-unconstrained lower bound
print(result.aggregation.epr_pairs)            # under the real port budget
print(result.aggregation.baseline_epr_pairs)   # without aggregation
print(result.entanglement_schedule.makespan)
```

The same report is available from the CLI:

```bash
quport ebits --n-logical 16 --depth 20 --seed 0 --out entanglement_plan.json
```
