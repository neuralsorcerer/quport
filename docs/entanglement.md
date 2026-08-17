# Entanglement model

Cut weight — the number of two-qubit gates whose operands land on different QPUs —
is the objective every classical partitioner minimizes, and it is not the quantity a
cat-entanglement machine pays. This page describes the model QuPort uses instead,
the modules that implement it, and how they fit together — from a partitioning
objective, through a port-constrained communication plan, down to the circuit that
performs it and a proof that it does.

| Module | Question it answers | When it runs |
|---|---|---|
| `quport.entanglement` | Which operand positions of an operation commute with `Z`? | Everywhere, as the shared correctness rule |
| `quport.hypergraph` | How many EPR pairs does this *partition* need? | Partition time, inside the search loop |
| `quport.aggregation` | Which EPR blocks does this *mapped circuit* need, under a real port budget? | Compile time, on physical qubits |
| `quport.schedule.estimate_entanglement_schedule` | How long does that plan take on this interconnect? | Schedule time |
| `quport.protocol` | What circuit actually performs the plan, and is it right? | Emission and verification |
| `quport.exact` | How close to optimal is the partition the heuristic found? | Offline, on small instances |

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
count.

`w_ebit` defaults to `0.0` on `tpccap_partition` and `tpccap_sa_partition`, so every
pre-existing objective is untouched. Passing `packets` with `w_ebit=0` fills in the
`ebits` and `weighted_ebit_distance` diagnostics without steering the search.

### The other terms have to be rescaled with it

Replacing the volume term is not a local change, because the remaining terms were
tuned against the term that was removed:

- **The port penalty is dropped** (`w_port=0.0`). `w_port` charges squared
  boundary-qubit overflow, which on realistic instances runs one to two orders of
  magnitude larger than an e-bit count — so with `w_dist=0` it is not a penalty on
  the objective, it *is* the objective. It also measures the wrong resource: what a
  cat-entanglement compiler needs a port for is a live cat copy, not every boundary
  qubit. And port pressure is already priced downstream, since
  `aggregate_remote_operations` converts a port shortage into evictions and fresh
  EPR pairs. Penalising it again double-counts a constraint in the wrong units.
- **Congestion is kept but re-sourced** (`congestion_source="ebits"`). Gate demand
  upper-bounds EPR demand — aggregation is exactly the business of removing
  transactions — so routing gate traffic reports congestion that never happens. The
  e-bit traffic matrix comes from the same sweep that computes the e-bit cost, so
  the congestion term and the volume term cannot describe different plans.
- **Both annealing stages use the same weight.** The default 4× congestion asymmetry
  between the TPCCAP seed and the annealer was tuned for the larger gate-traffic
  scale and costs e-bits at this one.

Measured over 36 configurations -- 9 to 20 logical qubits on 3 to 5 QPUs, across
`ring`, `switch` and `mesh` interconnects, six random circuits each -- rescaling
these terms moves the realised numbers as follows:

| | Before | After |
|---|---|---|
| EPR pairs actually spent | 28.3 | **21.4** |
| Port evictions | 2.72 | **2.06** |
| Entanglement-aware makespan | 5073 | **4237** |
| Peak link busy time | 2378 | 2406 |

Fewer EPR pairs *and* fewer evictions, for no material change in peak link load.
The evictions fall because minimising e-bits concentrates traffic into fewer,
longer-lived cat copies, which need fewer simultaneous ports than the many short
copies a boundary-minimising partition scatters around -- the e-bit objective was
already the better proxy for port pressure than the penalty meant to model it.

It also closes most of the distance to optimal. Over 24 instances -- 8 qubits on 2
QPUs, 9 on 3, and 12 on 3 and on 4, six random circuits each, all-to-all:

| Strategy | e-bit gap vs. proved optimum |
|---|---|
| `tpccap` | 55.8% |
| `cluster` | 46.5% |
| `tpccap_sa` | 44.3% |
| `balanced` | 36.6% |
| `ebit` | **7.7%** |

The row that motivated the change is `balanced`: before the rescaling, `ebit` sat at
43.5%, *behind plain balanced partitioning at the objective it is named for*. The
search was never the problem — given the e-bit objective alone, the annealer lands
within 0.2% of the proved optimum.

```{note}
`congestion_source` defaults to `"gates"` everywhere, so this affects the `ebit`
strategy only. Pass `congestion_source="ebits"` (with `packets`) to opt in from
your own call to `tpccap_partition` or `tpccap_sa_partition`.
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

## From plan to circuit

`build_telegate_circuit` expands a plan into the circuit it stands for. Each cat
block becomes, with root `c`, cat copy `b`, and EPR helper `a`:

```
entangler:     h(a); cx(a, b); cx(c, a); cx(a, b)
block gates:   every gate of the block, with c replaced by b
disentangler:  h(b); cz(b, c)
```

That is the deferred-measurement form: `measure a` plus `if m: x(b)` collapses to
`cx(a, b)`, and an X-basis measurement of `b` plus `if m: z(c)` collapses to
`h(b); cz(b, c)`. It is worth writing that way because a unitary circuit can be
checked with a state vector.

Both ancillas provably end in `|+>` regardless of the data, so an `h` returns
them to `|0>` and the next block reuses them. The emitted width therefore tracks
how many cat copies are live at once, not how many blocks there are.

`coherent=False` emits the executable form instead — real mid-circuit
measurement, `if` feedforward, and `reset` — which exports to OpenQASM 3.

```{note}
Teleport blocks show the state movement as a `swap` in and out of the host
ancilla rather than the Bell-measurement gadget. Expanding the return trip needs
a mid-circuit reset, which would make the program non-unitary and therefore
unverifiable by the route below. The two e-bits it costs are still accounted for.
```

## Verifying it

`verify_telegate_equivalence` runs the unitary form on a pseudo-random product
input, traces out the ancillas, and compares the reduced state of the data qubits
with the mapped circuit's. Unit fidelity certifies both halves at once: the data
are right, and the ancillas came back unentangled — leftover entanglement would
leave the reduced state mixed and the fidelity below one.

This is what makes the diagonality rule a tested property rather than a stated
assumption. A hand-built plan that keeps a cat copy live across an `X` on its
root — exactly what `aggregate_remote_operations` refuses to emit — drives the
fidelity to zero, not merely down.

```bash
quport ebits --n-logical 4 --depth 4 --config small.json --verify --emit-qasm telegate.qasm
```

Verification is a state-vector simulation and is refused above 24 qubits.

Because it compares state vectors it speaks about the state a circuit prepares.
Terminating measurements are dropped -- they read that state out without changing
it -- while a measurement or reset that later operations depend on changes what
the circuit computes, and is refused rather than quietly ignored.

## Calibrating against the optimum

A heuristic without a reference is a number without a scale: "the e-bit strategy
saved 30%" says nothing about how much was left behind. `quport.exact` solves the
same two problems exactly, by branch and bound, on instances small enough for that
to terminate:

```python
from quport.exact import optimal_partition, partition_gap

best = optimal_partition(9, 3, 3, objective="ebits", packets=packets)
gap = partition_gap(result.partition, 3, 3, objective="ebits", packets=packets)

print(best.objective, best.proved_optimal, best.nodes)
print(f"{gap.relative:.1%} above optimal")
```

Three things keep the tree small, and none of them is a heuristic shortcut:

- **Canonical form.** Both objectives are invariant under relabelling QPUs and the
  capacity is uniform, so only restricted-growth assignments are explored — a qubit
  joins a QPU already in use, or opens the lowest-numbered unused one. That collapses
  `n_qpus**n` candidates to set partitions of at most `n_qpus` blocks.
- **Monotone bounds.** Each incremental cost counts only what the new assignment
  *settles*, so the running total is an admissible lower bound and a node that
  already reaches the incumbent is cut.
- **A seeded incumbent**, so pruning bites from the first node.

`max_nodes` bounds the run. Exhausting it clears `proved_optimal`, and the returned
partition is then the best one found rather than the optimum — `PartitionGap` reports
that flag, and its `absolute` is a *lower* bound on the true gap in that case.

```{warning}
The tree is over set partitions, so this is for calibration on roughly a dozen
qubits, not for compiling. Use it to measure what a heuristic leaves behind, then
trust the heuristic at scale.
```

`partition_gap` raises rather than reporting a negative gap when a heuristic scores
*below* a proved optimum, because one of the two implementations would then be
wrong. That makes it a cross-check between two independent readings of both
objectives, which is why `tests/test_exact.py` runs it over every shipped strategy.
The branch and bound is itself checked against exhaustive enumeration over every
feasible assignment — 286 cut instances and 125 e-bit instances — since that is the
only real argument that the pruning and the canonical form do not silently lose an
optimum.

```bash
quport optimal --n-logical 9 --depth 10 --config small.json --strategy ebit
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
