<h1 align="center">
QuPort
</h1>

<div align="center">

QuPort is a research software framework developed in Python using the Qiskit toolkit for modeling, mapping, routing, splitting, scheduling, and benchmarking quantum circuits on modular multi-QPU machines. It treats the machine as a collection of QPUs with local compute qubits, communication-port qubits, an inter-QPU network, finite link capacity, finite port count, and a configurable latency model.

</div>

<div align="center">

[![Qiskit Ecosystem](https://qisk.it/e-390ee704)](https://qisk.it/e)
[![Current Release](https://img.shields.io/github/release/neuralsorcerer/quport.svg)](https://github.com/neuralsorcerer/quport/releases)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Test Linux](https://github.com/neuralsorcerer/quport/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/neuralsorcerer/quport/actions/workflows/ubuntu.yml?query=branch%3Amain)
[![Test Windows](https://github.com/neuralsorcerer/quport/actions/workflows/windows.yml/badge.svg)](https://github.com/neuralsorcerer/quport/actions/workflows/windows.yml?query=branch%3Amain)
[![Test MacOS](https://github.com/neuralsorcerer/quport/actions/workflows/macos.yml/badge.svg)](https://github.com/neuralsorcerer/quport/actions/workflows/macos.yml?query=branch%3Amain)
[![Lints](https://github.com/neuralsorcerer/quport/actions/workflows/lint.yml/badge.svg)](https://github.com/neuralsorcerer/quport/actions/workflows/lint.yml?query=branch%3Amain)
[![CodeQL](https://github.com/neuralsorcerer/quport/actions/workflows/codeql.yml/badge.svg)](https://github.com/neuralsorcerer/quport/actions/workflows/codeql.yml?query=branch%3Amain)
[![Documentation](https://github.com/neuralsorcerer/quport/actions/workflows/docs.yml/badge.svg)](https://github.com/neuralsorcerer/quport/actions/workflows/docs.yml?query=branch%3Amain)
[![License](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2605.12583-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2605.12583)
[![DOI:48550/arXiv.2605.12583](https://img.shields.io/badge/DOI-10.48550/arXiv.2605.12583-blue.svg)](https://doi.org/10.48550/arXiv.2605.12583)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/quport?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/quport)

</div>

---


The central problem solved by QuPort is:

Given a logical quantum circuit $C$ with $n$ logical qubits and two-qubit interactions $E$, choose

$$
\pi: \{0,\dots,n-1\}\rightarrow\{0,\dots,N-1\}
$$

that assigns every logical qubit to one of $N$ QPUs, then choose a physical layout

$$
\ell: \{0,\dots,n-1\}\rightarrow\{0,\dots,Q_{\mathrm{phys}}-1\}
$$

that places logical qubits on physical compute or communication qubits, and finally estimate or generate executable local programs plus remote-operation metadata while respecting capacity, topology, and routing constraints.

QuPort supports two complementary compilation modes:

1. **Global mapping and routing**: build one global directed Qiskit `CouplingMap` for all QPUs, provide a partition-aware initial layout, and let Qiskit/SABRE route the full circuit on the global graph.
2. **Distributed compilation**: partition the circuit, assign physical qubits, keep cross-QPU two-qubit operations as explicit remote events, split local operations into per-QPU circuits, and route only inside each QPU so remote execution is not hidden behind artificial cross-device SWAPs.

---

## What is implemented

QuPort implements an end-to-end stack for multi-QPU circuit experiments:

- Modular device construction with $N$ QPUs, $C$ compute qubits per QPU, and $P$ communication qubits per QPU.
- Local QPU topologies: `clique`, `line`, `ring`, and `grid2d`.
- Inter-QPU network topologies: `switch`, `mesh`, `ring`, `degree_d`, `clos`, and `fat_tree`.
- Directed Qiskit coupling maps where every undirected physical link is represented by two directed Qiskit edges.
- Logical interaction-graph extraction from arbitrary two-qubit circuit instructions.
- Optional temporal interaction weights that emphasize earlier two-qubit gates.
- Capacity-constrained partitioning baselines, topology-aware partitioning, and e-bit-aware partitioning.
- Computational-basis diagonality analysis that decides which gates one cat-entanglement can serve.
- Distributable-packet extraction and the hypergraph $\lambda-1$ e-bit metric.
- Communication aggregation of cross-QPU gates into cat-entanglement and teleport blocks under a comm-port budget.
- Communication-port placement hints for boundary-heavy and neighbor-diverse logical qubits.
- Global transpilation with configurable basis gates, layout method, routing method, optimization level, and seed.
- Distributed compilation into per-QPU OpenQASM 3 programs, remote-operation JSON, and schedule JSON.
- Schedule estimation under QPU-port, link-capacity, network-hop, switch-pair, and switch-reconfiguration constraints.
- Event-driven, resource-constrained entanglement scheduling with per-port hold times, per-link channels, hop-scaled EPR distribution, and a heralded-success retry model.
- Metrics for SWAP count, depth, circuit size, one-qubit gates, two-qubit gates, remote two-qubit operations, cut weight, congestion, remote rounds, peak link utilization, EPR pairs, and makespan.
- CLI commands for configuration generation, topology inspection, mapping, benchmarking, topology sweeps, schedule estimation, entanglement reporting, splitting, and distributed compilation.
- Programmatic APIs for custom pipelines and automated experiments.

---

## Architecture model

A QuPort device is configured with `MultiQPUConfig`.

Let:

- $N$ be `n_qpus`.
- $C$ be `compute_qubits_per_qpu`.
- $P$ be `comm_qubits_per_qpu`.
- $B=C+P$ be the physical block size of one QPU.
- $Q_{\mathrm{phys}}=N(C+P)$ be the total physical qubit count.

For QPU $q$, physical qubit indices are assigned contiguously:

$$
\mathrm{base}(q)=qB
$$

$$
\mathrm{compute}(q)=\{qB, qB+1, \dots, qB+C-1\}
$$

$$
\mathrm{comm}(q)=\{qB+C, qB+C+1, \dots, qB+C+P-1\}
$$

The physical-to-QPU map is:

```math
\mathrm{qpu\_of\_phys}(p)
=
\left\lfloor \frac{p}{B} \right\rfloor .
```

### Local QPU connectivity

For each QPU, QuPort builds local edges over `compute + comm` qubits:

| `intra_topology` | Meaning | Typical use |
|---|---|---|
| `clique` | Every local qubit connects to every other local qubit. | Idealized all-to-all QPU. |
| `line` | Local qubits form a path. | Strict nearest-neighbor baseline. |
| `ring` | Local qubits form a cycle when possible. | Slightly richer nearest-neighbor model. |
| `grid2d` | Local qubits are placed row-major on a 2D grid. | Planar/local-lattice style devices. |

For an undirected local edge $\{u,v\}$, QuPort inserts both directed Qiskit edges $(u,v)$ and $(v,u)$ because Qiskit coupling maps encode directed two-qubit operation support.

### Inter-QPU connectivity

Inter-QPU edges are created only between communication qubits.

| `inter_topology` | Meaning |
|---|---|
| `switch` | All QPU pairs can communicate through a switch-like all-to-all model. |
| `mesh` | All QPU pairs are adjacent in the QPU graph. |
| `ring` | QPU $q$ connects to $(q+1)\bmod N$. |
| `degree_d` | Each QPU connects to a bounded number of nearby QPUs controlled by `inter_degree`. |
| `clos` | Two-level approximation with pod-local and spine-style links when at least two ports exist. |
| `fat_tree` | Tree-like QPU graph; physical inter-QPU adjacency uses representative communication ports. |

The QPU graph is an undirected graph

$$
G_Q=(V_Q,E_Q),\qquad V_Q=\{0,\dots,N-1\}.
$$

For scheduling and congestion, shortest paths are computed on $G_Q$ with unweighted BFS distances:

$$
d(a,b)=\text{minimum number of QPU-network hops from }a\text{ to }b.
$$

If no path exists, QuPort treats the pair as unreachable and assigns a large unschedulable penalty in topology-aware estimators.

---

## Mathematical model

### Logical interaction graph

For a circuit $C$, QuPort scans all two-qubit instructions. If a two-qubit instruction acts on logical qubits $i$ and $j$, with $i\ne j$, it increments an undirected edge weight:

$$
w_{ij}\leftarrow w_{ij}+1,\qquad i<j.
$$

The weighted logical interaction graph is:

$$
G_L=(V_L,E_L,w),\qquad V_L=\{0,\dots,n-1\}.
$$

The weighted degree of logical qubit $i$ is:

$$
\deg(i)=\sum_{j:(i,j)\in E_L}w_{ij}.
$$

### Temporal interaction weighting

For strategies that use temporal weighting, QuPort orders two-qubit interactions by their two-qubit-operation index $t=0,1,2,\dots$ and applies exponential decay:

$$
w_t=\gamma^t,
$$

where `temporal_decay` is $\gamma\in(0,1]$.

For an edge $(i,j)$, the final temporal weight is:

$$
W_{ij}=\sum_{t\in T_{ij}}\gamma^t,
$$

where $T_{ij}$ is the set of times at which logical qubits $i$ and $j$ interact. If $\gamma=1$, temporal weights reduce to ordinary interaction counts.

Two consequences are worth keeping in mind when choosing $\gamma$:

- The total weight of a circuit converges to $\sum_{t\ge 0}\gamma^{t}=1/(1-\gamma)$ regardless of how long the circuit is. For the default $\gamma=0.98$ that is $50$, and $99\%$ of it falls in the first $\approx 228$ two-qubit operations. On a circuit with thousands of two-qubit gates the partitioner is therefore driven almost entirely by a prefix, which is the intended emphasis but is easy to overlook.
- Once $\gamma^{t}$ underflows to zero in double precision (at $t=36{,}883$ for $\gamma=0.98$), an edge that first appears after that point contributes exactly zero and is dropped from the interaction graph.

Which pipelines apply temporal weighting is not uniform, and it matters when comparing strategies:

| entry point | `tpccap` | `tpccap_sa` |
|---|---|---|
| `compile_distributed` | `temporal_decay`, default $\gamma=0.98$ | `temporal_decay`, default $\gamma=0.98$ |
| `map_and_transpile` | `temporal_decay`, default uniform counts | `temporal_decay`, default $\gamma=0.98$ |

Both entry points accept `temporal_decay`, and both ignore it for `balanced` and
`cluster`, which always partition on uniform interaction counts. What differs is
the default: `compile_distributed` applies $\gamma=0.98$ to either topology-aware
strategy, while `map_and_transpile` leaves `tpccap` on uniform counts and puts
`tpccap_sa` on $\gamma=0.98$.

That default asymmetry matters when reading benchmark output. `benchmark_random_circuits`
and `sweep_topologies` both call `map_and_transpile` without a decay, so their `method=2`
and `method=3` rows differ in the interaction weights as well as the search procedure and
are not an ablation of the annealing alone. On random circuits much of the gap between
those rows comes from the weighting rather than the annealing, and `tpccap_sa` can score
worse on uniform cut precisely because it is optimizing an early-weighted objective. To
measure the annealing itself, pass the same `temporal_decay` to both strategies, or use
`compile_distributed`, whose defaults already match.

### Partition capacity

Each QPU can host at most

$$
K=C+P
$$

logical qubits in the global mapping model. A partition $\pi$ is feasible if:

$$
\left|\{i:\pi(i)=q\}\right|\le K\qquad\forall q\in\{0,
\dots,N-1\}.
$$

### Cut weight

A two-qubit interaction is remote when its endpoints are assigned to different QPUs. The partition cut is:

$$
\mathrm{cut}(\pi)=\sum_{(i,j)\in E_L} w_{ij}\,\mathbf{1}[\pi(i)\ne\pi(j)].
$$

A lower cut usually means fewer remote two-qubit operations, although final routed metrics also depend on layout, topology, and Qiskit routing.

### Traffic matrix

For a partition $\pi$, QuPort computes a symmetric QPU-to-QPU traffic matrix $T$:

$$
T_{ab}=\sum_{(i,j)\in E_L} w_{ij}\,\mathbf{1}[\pi(i)=a,\pi(j)=b]
      +\sum_{(i,j)\in E_L} w_{ij}\,\mathbf{1}[\pi(i)=b,\pi(j)=a]
$$

for $a\ne b$, and

$$
T_{aa}=0.
$$

This matrix quantifies the amount of logical interaction weight that must cross between QPUs.

### Link-load routing

For each traffic pair $(a,b)$, QuPort can route $T_{ab}$ on QPU-network shortest paths.

In single-path mode, traffic follows one shortest path. If path edges are

$$
(a=v_{0},v_{1}),(v_{1},v_{2}),\dots,(v_{h-1},v_{h}=b),
$$

then each undirected link $\{v_{k},v_{k+1}\}$ receives load $T_{ab}$.

In ECMP mode, traffic is split evenly across all shortest paths. If there are $\sigma_{ab}$ shortest paths and a link $e$ appears in $\sigma_{ab}(e)$ of those paths, the load contribution is:

$$
L_e \mathrel{+}= T_{ab}\frac{\sigma_{ab}(e)}{\sigma_{ab}}.
$$

QuPort reports congestion metrics:

$$
L_{\max}=\max_{e\in E_Q}L_e
$$

and

$$
L_2=\sum_{e\in E_Q}L_e^2.
$$

---

## Partitioning strategies

QuPort supports five main partitioning strategies.

### `cluster`: heavy-edge clustering

This baseline uses a disjoint-set union structure.

1. Sort interaction edges by descending weight.
2. Merge clusters connected by heavy edges when the merged cluster size stays within capacity $K$.
3. Place clusters into QPUs with first-fit decreasing bin packing.
4. If a cluster cannot be placed whole, place its vertices individually.

The guiding idea is that large $w_{ij}$ means qubits $i$ and $j$ should preferably remain local, because cutting that edge contributes $w_{ij}$ to $\mathrm{cut}(\pi)$.

### `balanced`: balanced greedy partitioning

The balanced greedy strategy orders logical qubits by descending weighted degree. When placing a qubit $v$, it scores each non-full QPU $q$ as:

$$
\mathrm{score}(v,q)=
\sum_{u:\pi(u)=q}w_{uv}
-\alpha\frac{\mathrm{load}(q)}{K},
$$

where $\alpha$ is `alpha_balance` and $\mathrm{load}(q)$ is the number of already placed logical qubits on QPU $q$.

The first term rewards placing $v$ next to already assigned neighbors with high interaction weight. The second term discourages overfilling early QPUs and improves balance.

After greedy placement, QuPort runs local move refinement. Moving vertex $v$ from QPU $a$ to QPU $b$ changes the cut by comparing its external and internal incident weights. A move is accepted only when it decreases cut and respects capacity.

### `tpccap`: topology-, port-, and congestion-aware partitioning

`tpccap` extends cut minimization with architecture-aware terms. It considers:

- cut weight;
- QPU-network hop distance;
- communication-port pressure;
- routed link congestion;
- disconnected-pair penalties;
- load balance.

A simplified objective has the structure:

```math
J(\pi)
=
\lambda_{cut}\,cut(\pi)
+
\lambda_{hop}\sum_{a\lt b} T_{ab}\,d(a,b)
+
\lambda_{cong}\,L_2
+
\lambda_{port}\,\Phi_{port}
+
\lambda_{bal}\,\Phi_{bal}
+
\lambda_{disc}\,\Phi_{disc}.
```

The terms mean:

- $\mathrm{cut}(\pi)$ counts remote interaction weight.
- $\sum T_{ab}d(a,b)$ prefers remote traffic between nearby QPUs.
- $L_2$ penalizes concentrating routed traffic on the same network links.
- $\Phi_{\mathrm{port}}$ penalizes boundary pressure that exceeds available communication ports.
- $\Phi_{\mathrm{bal}}$ discourages imbalanced QPU loads.
- $\Phi_{\mathrm{disc}}$ penalizes traffic between disconnected QPU pairs.

The implementation validates all numeric controls and normalizes inputs before search so invalid capacities, probabilities, infinities, booleans, negative weights, malformed matrices, and disconnected routing cases fail deterministically or are penalized consistently.

### `tpccap_sa`: simulated-annealing refinement

`tpccap_sa` starts from the topology-aware partition and then performs simulated annealing moves. If a candidate move changes the objective by

$$
\Delta=J(\pi')-J(\pi),
$$

then QuPort accepts the move when $\Delta\le0$ and may accept it when $\Delta>0$ with probability

$$
P_{\mathrm{accept}}=\exp\left(-\frac{\Delta}{T}\right),
$$

where $T$ is a temperature that cools over iterations. This helps escape local minima created by greedy or local-search decisions.

The objective $J$ is not identical in the two stages. The seed is built with
`w_cong = 0.05`, matching `tpccap`, while the annealing optimizes
`anneal_w_cong = 0.2` — four times the congestion penalty. The annealing returns the
best state it saw under *its* objective, so it never loses ground there, but the
partition it hands back can score worse than its own seed when measured with the
seed's weighting; on random instances that happens for roughly $40\%$ of them. Both
are parameters of `tpccap_sa_partition`, and `anneal_w_cong=None` anneals on exactly
the objective the seed was built for. The defaults are the values every published
QuPort result was produced with.

Together with the interaction-weight difference described above, this is the second
reason a `tpccap` versus `tpccap_sa` comparison has to state its configuration: the
two strategies can otherwise be ranked on different scales.

### `ebit`: e-bit-aware partitioning

`ebit` runs the same search as `tpccap_sa` but measures communication in EPR pairs
rather than cut gates, because one cat-entanglement can serve many gates. It is
described in full under
[the entanglement model](#entanglement-model-packets-e-bits-and-communication-aggregation),
which first has to establish what a distributable packet is.

---

## Entanglement model: packets, e-bits, and communication aggregation

Every strategy above minimizes some form of *cut weight*: the number, or weighted
number, of two-qubit gates whose operands land on different QPUs. On a machine that
implements remote gates with cat-entanglement, that is the wrong quantity to
minimize, and this section describes the model QuPort uses instead.

### The cat-entanglement protocol and its correctness condition

A remote two-qubit gate is not executed by moving state between QPUs. The standard
construction distributes one EPR pair and builds a *cat copy* of a root qubit:

1. Distribute an EPR pair with half $a$ on QPU $A$ (which holds the root qubit $c$)
   and half $b$ on QPU $B$.
2. On $A$: apply $\mathrm{CX}(c\rightarrow a)$, measure $a$ in the $Z$ basis, send
   the outcome to $B$, which applies a conditional $X$. The joint state becomes

$$
\sum_z\alpha_z\lvert z\rangle_c\lvert z\rangle_b\otimes\lvert\psi_z\rangle,
$$

   so $b$ now carries the computational-basis label of $c$.
3. Run **every** gate that uses $c$ only through that label as a local gate on $B$,
   against $b$.
4. Cat-disentangler: measure $b$ in the $X$ basis, send the outcome back, and apply
   a conditional $Z$ to $c$.

Step 3 is correct exactly when every operation applied to $c$ while the copy is live
commutes with $Z_c$. Such an operation maps
$\lvert z\rangle_c\otimes\lvert\psi\rangle$ to
$\lvert z\rangle_c\otimes U_z\lvert\psi\rangle$, so the $c$/$b$ correspondence
survives and the disentangler restores $c$ exactly. An $X$, $H$, $\sqrt{X}$, or a
$\mathrm{CX}$ that uses $c$ as its *target* breaks it, and the copy must be released
first.

`quport.entanglement.diagonal_positions` answers, for one operation, which operand
positions commute with $Z$. It derives them from three rules, in order: an explicit
table for gates such as `rzz` and `rzx`; `ControlledGate` structure, where every
control operand is diagonal regardless of `ctrl_state` and every operand that is
diagonal for the base gate stays diagonal once controlled; and a table of diagonal
single-qubit gates. Anything else is reported as non-diagonal on every operand.
Under-reporting only costs extra EPR pairs, so the conservative default is the safe
one; `tests/test_entanglement.py` checks every claim against the actual unitary of
every constructible gate in Qiskit's standard library.

### Distributable packets and the $\lambda-1$ metric

A **distributable packet** rooted at qubit $c$ is a maximal run of gates over which
$c$ stays diagonal. Because diagonality is a property of the gate sequence alone,
packets are independent of where qubits are placed: they are built once per circuit
and re-evaluated for each candidate partition in time linear in the number of
packet incidences, which is what makes them cheap enough for the annealing loop.

Let $\mathcal{P}$ be the packets of a circuit, $\mathrm{root}(P)$ the root of packet
$P$, and $\mathrm{partners}(P)$ the other operand of each of its gates. The number of
EPR pairs a cat-entanglement compiler consumes under partition $\pi$ is the
connectivity-minus-one ($\lambda-1$) metric of hypergraph partitioning:

$$
E(\pi)=\sum_{P\in\mathcal{P}}
\Bigl\lvert\;\{\pi(t):t\in\mathrm{partners}(P)\}\setminus\{\pi(\mathrm{root}(P))\}\;\Bigr\rvert .
$$

One e-bit per packet per *distinct remote QPU*, not one per cut gate. Ten gates from
one control into one QPU cost ten units of cut weight and one e-bit.

Two kinds of gate cannot be served by a single cat copy: two-qubit gates with no
diagonal operand (`swap`, `iswap`, `ecr`, `rxx`), and operations on three or more
qubits, which a bipartite copy cannot bring together. A gate of either kind spanning
$k$ QPUs is charged $2(k-1)$ e-bits, the cost of teleporting every foreign operand to
one host and back, which is also the standard cost of an arbitrary non-local
two-qubit unitary.

Because each gate is charged to exactly one root, $E(\pi)$ is exact for the chosen
root assignment and an upper bound over all assignments. Gates whose *both* operands
are diagonal (`cz`, `cp`, `crz`, `rzz`) admit a choice; the default `"greedy"` policy
reuses an operand that already roots an open packet and falls back to the lower qubit
index.

### `ebit`: e-bit-aware partitioning

The `ebit` strategy runs the same TPCCAP plus simulated-annealing search as
`tpccap_sa`, but replaces the weighted-cut-distance term with hop-scaled e-bit
demand:

$$
J_{\mathrm{ebit}}(\pi)=
w_{\mathrm{ebit}}\sum_{P\in\mathcal{P}}\;\sum_{q\in R(P,\pi)}d(\pi(\mathrm{root}(P)),q)
+w_{\mathrm{port}}\sum_q\max(0,B_q-P)^2
+w_{\mathrm{cong}}L_2,
$$

where $R(P,\pi)$ is the set of distinct remote QPUs packet $P$ touches. On an
all-to-all fabric every distance is $1$ and the first term is exactly $E(\pi)$. The
port and congestion terms are unchanged from `tpccap`; congestion is still estimated
from gate-level traffic, which upper-bounds e-bit traffic because aggregation only
removes transactions. `quport.hypergraph.ebit_traffic_matrix` returns the exact EPR
demand for callers who want it.

`w_ebit` defaults to $0$ on `tpccap_partition` and `tpccap_sa_partition`, so every
pre-existing objective and every published number is unchanged. Passing `packets`
with `w_ebit=0` populates the `ebits` and `weighted_ebit_distance` diagnostics
without steering the search.

### Communication aggregation under a port budget

$E(\pi)$ assumes ports are free. `quport.aggregation.aggregate_remote_operations`
answers the same question on a real mapped circuit, where they are not: a QPU with
$P$ comm ports can host at most $P$ cat copies at once, and starting a new block also
needs a free port on the root's QPU to run the entangler. When a port is needed and
none is free, the least recently used copy is released, and a fresh EPR pair is spent
if that root is needed again. The plan records those `evictions`.

The two computations are independent implementations of the same quantity: with an
unbounded port budget, `aggregate_remote_operations(...).epr_pairs` equals
`ebit_cost(...)` exactly, which
`tests/test_aggregation.py::test_unbounded_ports_match_hypergraph_ebits` pins down
over compiled random circuits.

### Entanglement-aware scheduling

`estimate_entanglement_schedule` schedules the aggregated plan as a
resource-constrained system rather than a sequence of DAG layers. The layered and
topology-aware estimators charge each layer its slowest operation, which imposes a
global barrier between layers and charges one entanglement transaction per cross-QPU
gate. The entanglement-aware estimator instead runs an as-soon-as-possible list
schedule in program order against:

- one timeline per physical qubit, so QPUs that share no qubits drift apart freely;
- a pool of `comm_qubits_per_qpu` ports per QPU, each held for a whole block;
- `link_capacity` channels on every link along the routed path;
- hop-scaled, probabilistic distribution
  $\tau_{\mathrm{EPR}}(h)=h\cdot\tau_{\mathrm{EPR}}/p_{\mathrm{success}}$, since
  heralded entanglement needs $1/p$ attempts in expectation.

A block costs $\tau_{\mathrm{EPR}}(h)+\tau_{\mathrm{RTT}}^{\mathrm{eff}}+\tau_{\mathrm{remote\_gate}}$
to establish and $\tau_{\mathrm{RTT}}^{\mathrm{eff}}$ to disentangle; a teleport block
pays a second distribution for the return trip. Gates inside a block cost ordinary
local two-qubit time.

### Measured effect

Reproduce with `n_qpus=4`, `compute_qubits_per_qpu=4`, `comm_qubits_per_qpu=2`,
`inter_topology="switch"`, `optimization_level=0`, comparing
`aggregation.epr_pairs` against `aggregation.baseline_epr_pairs`:

| Circuit | Cross-QPU gates | EPR pairs, per gate | EPR pairs, aggregated | Saved |
|---|---|---|---|---|
| 16-qubit QFT, `strategy="ebit"`, `seed=0` | $190$ | $190$ | $88$ | $53.7\%$ |
| 16-qubit GHZ fan-out, `strategy="ebit"`, `seed=0` | $10$ | $10$ | $3$ | $70.0\%$ |
| 16-qubit random depth-20, `strategy="tpccap_sa"`, seeds $0..9$ | $990$ | $990$ | $639$ | $35.5\%$ |

Structured circuits benefit most, because a control that only ever picks up $R_z$
rotations keeps its packet open across the whole ladder. Random circuits benefit
less: translating to the default basis puts `sx` and `x` gates on most qubits, and
each of those closes a packet.

---

## Layout and communication-port placement

After partitioning, QuPort must map logical qubits onto physical qubits.

For each QPU $q$, there are two local physical pools:

- compute pool: ordinary local execution qubits;
- communication pool: qubits that can connect to other QPUs.

QuPort identifies boundary logical qubits:

$$
B_q=\{i:\pi(i)=q\text{ and }\exists j\text{ with }w_{ij}>0,\pi(j)\ne q\}.
$$

Boundary-heavy qubits are good candidates for communication ports because remote interactions require inter-QPU resources.

Two communication-selection modes are implemented:

- `topk`: choose the $P$ logical qubits in each QPU with the largest remote-boundary score;
- `diverse`: prefer qubits that interact with many distinct remote QPUs, which spreads port access across different network destinations.

A simple boundary score is:

$$
s_i=\sum_{j:\pi(j)\ne\pi(i)}w_{ij}.
$$

A diversity-aware score also considers

$$
d_i^{\mathrm{remote}}=\left|\{\pi(j):w_{ij}>0,\pi(j)\ne\pi(i)\}\right|.
$$

The diversity term is applied as a penalty against destinations already covered by
ports chosen earlier in the same QPU, so it has nothing to act on until the second
port. `tpccap` and `tpccap_sa` request `diverse` under both entry points, but at the
default `comm_qubits_per_qpu` of $1$ it selects exactly what `topk` would; give each
QPU at least two ports before attributing any result to port diversity.

The final layout maps selected boundary qubits to communication physical qubits first, then maps remaining qubits to compute qubits and any unused communication qubits.

---

## Global mapping pipeline

The `map_and_transpile` pipeline performs:

1. **Capacity check**: reject circuits where $n>Q_{\mathrm{phys}}$.
2. **Basis translation**: translate the circuit to configured basis gates, defaulting to `("rz", "sx", "x", "cx")`.
3. **Interaction extraction**: compute $w_{ij}$ or temporal weights $W_{ij}$.
4. **Partitioning**: apply `balanced`, `cluster`, `tpccap`, or `tpccap_sa`.
5. **Layout hinting**: choose communication-port logical qubits and create an initial Qiskit layout.
6. **Global coupling map construction**: create a directed coupling map for all local and inter-QPU physical links.
7. **Qiskit transpilation**: run Qiskit with the configured optimization, layout, and routing settings.
8. **Metric computation**: count SWAPs, depth, size, one-qubit gates, two-qubit gates, and remote two-qubit operations.
9. **Cost estimation**: evaluate the configured latency/cost model.

This mode is useful when you want one routed Qiskit circuit for the entire modular device graph.

---

## Distributed compilation pipeline

The `compile_distributed` pipeline is designed for explicit multi-QPU execution artifacts:

1. Translate the input circuit into the configured basis.
2. Extract logical interaction weights.
3. Partition logical qubits across QPUs.
4. Build a physical circuit with the partition-aware initial layout but without global inter-QPU routing.
5. Split the physical circuit into local per-QPU circuits plus remote operations.
6. Route each local circuit using that QPU's intra-QPU coupling map only.
7. Estimate topology-aware remote-operation scheduling.
8. Return all local circuits, remote-operation trace, metrics, and timing summaries.

A remote operation records:

- operation name;
- global instruction index;
- the two physical qubit indices it acts on;
- the QPU id owning each of those qubits;
- gate parameters;
- classical bit indices the operation reads or writes.

This split makes the boundary explicit: local gates remain in QPU-local programs, while cross-QPU two-qubit gates become remote events handled by orchestration, entanglement generation, teleportation-style protocols, or another execution backend.

---

## Scheduling and makespan estimation

QuPort includes progressively richer schedule estimators.

### Simple parallel estimator

The simple estimator treats QPUs as parallel local processors and adds synchronization costs at remote operations.

A local one-qubit operation costs `oneq`, a local two-qubit operation costs `twoq`, a SWAP costs `swap`, and a remote two-qubit operation costs:

$$
\tau_{\mathrm{remote}}=\tau_{\mathrm{EPR}}+\tau_{\mathrm{RTT}}+\tau_{\mathrm{remote\_gate}}.
$$

### Layered estimator

The layered estimator uses Qiskit DAG layers. Local operations in a layer can run in parallel across QPUs. The layer duration is approximately:

$$
\tau_{\mathrm{layer}}=
\max\left(\max_q \tau_{q,\mathrm{local}},\tau_{\mathrm{remote\_rounds}}\right).
$$

### Topology-aware estimator

The topology-aware estimator considers:

- available communication ports per QPU;
- per-link capacity `link_capacity`;
- QPU-network reachability;
- hop-dependent remote costs;
- switch pair limits through `switch_parallel_links`;
- switch reconfiguration delay through `switch_reconfig_delay`;
- optional classical-latency hiding through `async_classical` and `async_overlap`.

If classical latency hiding is enabled, the effective classical round-trip term is:

$$
\tau_{\mathrm{RTT,eff}}=(1-\rho)\tau_{\mathrm{RTT}},
$$

where $\rho=\mathtt{async\_overlap}$ clipped to $[0,1]$.

For QPU pair $(a,b)$ with shortest-path hop count $d(a,b)$, the remote cost is modeled as:

$$
\tau_{\mathrm{remote}}(a,b)=d(a,b)\tau_{\mathrm{EPR}}+\tau_{\mathrm{RTT,eff}}+\tau_{\mathrm{remote\_gate}}.
$$

Remote operations in the same DAG layer are greedily packed into rounds. A remote operation can be placed in a round only if:

$$
\mathrm{ports\_used}(a)<P,
$$

$$
\mathrm{ports\_used}(b)<P,
$$

and every link $e$ on the chosen QPU-network path has

$$
\mathrm{link\_used}(e) \lt \mathtt{link\_capacity}.
$$

The estimator returns:

- `makespan`;
- number of DAG `layers`;
- total `remote_ops`;
- `remote_rounds`;
- absolute per-layer `start_time` / `end_time` offsets;
- absolute per-round `start_time` / `end_time` offsets for timeline visualization and simulator ingestion;
- `peak_link_util`;
- `peak_qpu_ports_used`.

Use `schedule.to_dict()` or `schedule_plan.to_dict()` when exporting these values.
Those serializers normalize tuple-valued QPU pairs and link-utilization entries to
JSON-native arrays/objects and validate finite non-negative timings, non-negative
counts, and non-self QPU/link pairs before emitting a payload.

### Entanglement-aware estimator

`estimate_entanglement_schedule` drops the DAG-layer abstraction entirely. Layers
impose a global barrier between successive slices and charge one entanglement
transaction per cross-QPU gate; both are pessimistic. This estimator runs an
as-soon-as-possible list schedule in program order over the aggregated blocks of
[the entanglement model](#entanglement-model-packets-e-bits-and-communication-aggregation),
holding a comm port for a whole block and a link channel for each distribution
window. It returns:

- `makespan`;
- `blocks` and `epr_pairs`;
- `remote_gates` and `unschedulable_gates`;
- `entanglement_time`, the total link occupancy summed over links;
- `peak_ports_in_use`, `port_busy_time`, and `qpu_busy_time` per QPU;
- `link_busy_time` per inter-QPU link.

Because it neither serializes independent QPUs nor pays per gate, its makespan is
typically well below the topology-aware figure on the same circuit; the two answer
different questions and should not be mixed inside one comparison.

---

## Metrics and cost model

### Circuit metrics

For a transpiled or physical circuit, QuPort computes:

| Metric | Meaning |
|---|---|
| `swaps` | Number of `swap` instructions. |
| `depth` | Qiskit circuit depth. |
| `size` | Qiskit circuit size. |
| `n_1q` | Number of one-qubit instructions. |
| `n_2q` | Number of two-qubit instructions. |
| `remote_2q` | Number of two-qubit instructions whose physical endpoints belong to different QPUs. |

A two-qubit physical operation on physical qubits $p_{0},p_{1}$ is remote when:

`qpu_of_phys`$(p_{0}) \ne$ `qpu_of_phys`$(p_{1})$.

`swaps` counts instructions literally named `swap`, so it is basis-dependent. The
default `basis_gates` of `("rz", "sx", "x", "cx")` contains no `swap`, so Qiskit
rewrites every routing SWAP into CX gates and `swaps` reads $0$ for every run
while the routing overhead shows up inside `n_2q` instead. Add `"swap"` to
`basis_gates` to keep SWAP instructions intact and make the metric non-zero.
This applies wherever the metric surfaces, including the `SWAPs:` line printed
by `quport map`, the `swaps` column of the benchmark CSV, and `swaps_mean` in the
topology sweep.

A `swap` instruction is also a two-qubit instruction, so it is counted in both
`swaps` and `n_2q`.

### Cost model

The default `LatencyModel` contains:

| Field | Default | Meaning |
|---|---:|---|
| `oneq` | $1.0$ | Cost of one local one-qubit gate. |
| `twoq` | $10.0$ | Cost of one local two-qubit gate. |
| `swap` | $30.0$ | Cost of one SWAP. |
| `epr_gen` | $200.0$ | Entanglement-generation component of a remote operation. |
| `classical_rtt` | $20.0$ | Classical round-trip component. |
| `remote_gate_overhead` | $50.0$ | Additional remote-gate overhead. |
| `epr_success_prob` | $1.0$ | Heralded entanglement success probability per attempt, in $(0,1]$. |

`epr_success_prob` is read only by `estimate_entanglement_schedule` and
`LatencyModel.expected_epr_time`, which scale distribution time by the expected
attempt count $1/p$. The default of $1.0$ models a deterministic link, so
`estimate_cost` and every older estimator return exactly the values they always did.

The local component is:

$$
C_{\mathrm{local}}=c_{1q}n_{1q}+c_{2q}n_{2q}+c_{\mathrm{swap}}n_{\mathrm{swap}}.
$$

Because a `swap` instruction is counted in both $n_{2q}$ and $n_{\mathrm{swap}}$,
$c_{\mathrm{swap}}$ acts as an overhead charged on top of the ordinary two-qubit
cost: a surviving SWAP contributes $c_{2q}+c_{\mathrm{swap}}$, which is $40$ under
the defaults rather than $30$.

The remote component is:

$$
C_{\mathrm{remote}}=n_{\mathrm{remote}}
(c_{\mathrm{EPR}}+c_{\mathrm{RTT}}+c_{\mathrm{remote\_gate}}).
$$

The depth penalty is:

$$
C_{\mathrm{depth}}=0.1\,d_{\mathrm{circuit}}\,c_{2q}.
$$

The total reported cost is:

$$
C_{\mathrm{total}}=C_{\mathrm{local}}+C_{\mathrm{remote}}+C_{\mathrm{depth}}.
$$

---

## Installation

QuPort requires Python $\ge 3.10$.

### Runtime install

```bash
python -m pip install -e .
```

### Development and analysis install

```bash
python -m pip install -e ".[viz,yaml,graph]"
```

Optional extras:

| Extra | Installs | Why use it |
|---|---|---|
| `viz` | `pandas`, `matplotlib`, `tqdm` | CSV analysis, plotting, and progress helpers. |
| `yaml` | `PyYAML` | YAML config input/output. |
| `graph` | `networkx` | Graph-heavy downstream experiments. |

Check the CLI:

```bash
quport --help
```

or:

```bash
python -m quport --help
```

---

## Command-line usage

### Generate a config file

```bash
quport gen-config
```

This writes a default `MultiQPUConfig` to `quport_config.json`. Pass `--out` with a
`.yaml`/`.yml` suffix to emit YAML instead, which requires the `yaml` extra
(`pip install quport[yaml]`); the format follows the file extension.

### Map and globally transpile a random circuit

```bash
quport map --n-logical 80 --depth 20 --seed 7 --strategy tpccap_sa
```

Write the mapped circuit as OpenQASM 3:

```bash
quport map \
  --n-logical 80 \
  --depth 20 \
  --seed 7 \
  --strategy tpccap_sa \
  --out mapped.qasm
```

Use a custom config:

```bash
quport map \
  --n-logical 80 \
  --depth 20 \
  --seed 7 \
  --strategy tpccap_sa \
  --config quport_config.json
```

### Benchmark strategies

```bash
quport bench \
  --n-logical 80 \
  --depth 20 \
  --trials 20 \
  --seed 7 \
  --strategies baseline,balanced,tpccap \
  --out results.csv
```

### Sweep topologies and port counts

```bash
quport sweep \
  --n-logical 80 \
  --depth 20 \
  --trials 5 \
  --seed 7 \
  --out sweep.csv
```

Create a plot when `viz` dependencies are installed:

```bash
quport sweep \
  --n-logical 80 \
  --depth 20 \
  --trials 5 \
  --seed 7 \
  --out sweep.csv \
  --plot sweep.png
```

### Inspect the inter-QPU topology

```bash
quport topology-info --config quport_config.json
```

Prints structural metrics for the configured interconnect (degree, diameter,
average shortest path, connectivity) without compiling anything, which is a cheap
way to compare candidate topologies before a sweep.

### Estimate a schedule

```bash
quport schedule --n-logical 80 --depth 20 --seed 7 --strategy tpccap
```

### Report entanglement demand

```bash
quport ebits --n-logical 80 --depth 20 --seed 7 --out entanglement_plan.json
```

Prints cross-QPU gate count, EPR pairs with and without aggregation, the saving,
block count, port evictions, peak cat copies per QPU, the port-unconstrained
$\lambda-1$ e-bit count, and both makespan figures. `--out` additionally writes the
full plan, including every block's root, host QPU, protocol, and served gate
indices.

### Split a mapped global circuit into local circuits and remote operations

```bash
quport split \
  --n-logical 80 \
  --depth 20 \
  --seed 7 \
  --strategy tpccap \
  --out-dir distributed_out
```

### Distributed compile

```bash
quport compile-dist \
  --n-logical 80 \
  --depth 20 \
  --seed 7 \
  --strategy tpccap_sa \
  --temporal-decay 0.98 \
  --out-dir compile_out
```

This produces per-QPU routed programs, an ordered remote-operation trace, and a topology-aware schedule summary.

---

## Python API usage

### Basic global mapping

```python
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
result = map_and_transpile(qc, cfg, latency=LatencyModel(), seed=7, strategy="tpccap_sa")

print(result.metrics)
print(result.cost)
print(result.partition)
```

### Distributed compilation

```python
from quport.compiler import compile_distributed
from quport.config import LatencyModel, MultiQPUConfig
from quport.pipeline import random_benchmark_circuit

cfg = MultiQPUConfig(n_qpus=10, compute_qubits_per_qpu=8, comm_qubits_per_qpu=2)
qc = random_benchmark_circuit(n_logical=80, depth=20, seed=7)

result = compile_distributed(
    qc,
    cfg,
    latency=LatencyModel(),
    seed=7,
    strategy="tpccap_sa",
    temporal_decay=0.98,
)

print(result.schedule.makespan)
print(result.schedule.to_dict())
print(result.schedule_plan.to_dict()["summary"])
print(result.schedule_plan.layers[0].remote_rounds)
print(len(result.program.remote_ops))
print(result.local_metrics)
```

### Entanglement demand and aggregation

```python
from quport import (
    MultiQPUConfig,
    aggregate_remote_operations,
    build_distributable_packets,
    compile_distributed,
    ebit_cost,
    estimate_entanglement_schedule,
)
from quport.architecture import MultiQPUArchitecture
from quport.config import LatencyModel
from quport.pipeline import random_benchmark_circuit

cfg = MultiQPUConfig(
    n_qpus=4,
    compute_qubits_per_qpu=4,
    comm_qubits_per_qpu=2,
    inter_topology="switch",
    optimization_level=0,
)

qc = random_benchmark_circuit(n_logical=16, depth=20, seed=0)
result = compile_distributed(qc, cfg, seed=0, strategy="ebit")

print(result.ebits.ebits, "e-bits with unlimited ports")
print(result.aggregation.epr_pairs, "EPR pairs under the real port budget")
print(result.aggregation.baseline_epr_pairs, "EPR pairs without aggregation")
print(result.entanglement_schedule.makespan)

# The same analysis on any mapped circuit. A plan and the schedule that consumes
# it must agree on the port budget, so pass ports_per_qpu to both.
arch = MultiQPUArchitecture(cfg)
plan = aggregate_remote_operations(result.physical_circuit, arch, ports_per_qpu=8)
summary = estimate_entanglement_schedule(
    result.physical_circuit,
    arch,
    LatencyModel(epr_success_prob=0.5),
    plan=plan,
    ports_per_qpu=8,
)
print(summary.makespan, summary.peak_ports_in_use)

# Score a candidate partition without compiling anything. Reuse result.packets:
# they were built from the basis-translated circuit the partitioner actually saw.
print(ebit_cost(result.packets, result.partition, cfg.n_qpus))
print(ebit_cost(build_distributable_packets(qc), [0] * qc.num_qubits, cfg.n_qpus))
```

### Custom architecture inspection

```python
from quport.architecture import MultiQPUArchitecture
from quport.config import MultiQPUConfig

cfg = MultiQPUConfig(inter_topology="ring", intra_topology="grid2d", grid_rows=3)
arch = MultiQPUArchitecture(cfg)

print(arch.block_of_qpu(0))
print(arch.build_coupling_map())
print(arch.qpu_shortest_paths().dist)
```

---

## Configuration

`MultiQPUConfig` fields:

| Field | Default | Description |
|---|---:|---|
| `n_qpus` | `10` | Number of QPUs. |
| `compute_qubits_per_qpu` | `8` | Compute qubits in each QPU. |
| `comm_qubits_per_qpu` | `1` | Communication-port qubits in each QPU. |
| `intra_topology` | `clique` | Local QPU topology. |
| `inter_topology` | `switch` | Inter-QPU topology. |
| `inter_degree` | `2` | Degree control for `degree_d`. |
| `link_capacity` | `1` | Max simultaneous remote ops per inter-QPU link per round. |
| `switch_parallel_links` | `1000000` | Max distinct QPU pairs per round for switch-like models. |
| `switch_reconfig_delay` | `0.0` | Additional delay per switch communication round. |
| `async_classical` | `True` | Enable classical-latency overlap in topology-aware scheduling. |
| `async_overlap` | `0.5` | Fraction of `classical_rtt` hidden when async classical mode is enabled. |
| `grid_rows` | `None` | Optional row count for `grid2d`. |
| `grid_cols` | `None` | Optional column count for `grid2d`. |
| `basis_gates` | `("rz", "sx", "x", "cx")` | Basis gates for Qiskit translation/transpilation. |
| `optimization_level` | `3` | Qiskit optimization level. |
| `layout_method` | `sabre` | Qiskit layout method for global transpilation. |
| `routing_method` | `sabre` | Qiskit routing method. |

JSON example:

```json
{
  "n_qpus": 6,
  "compute_qubits_per_qpu": 8,
  "comm_qubits_per_qpu": 2,
  "intra_topology": "grid2d",
  "inter_topology": "ring",
  "link_capacity": 1,
  "async_classical": true,
  "async_overlap": 0.5
}
```

YAML example:

```yaml
n_qpus: 6
compute_qubits_per_qpu: 8
comm_qubits_per_qpu: 2
intra_topology: grid2d
inter_topology: ring
link_capacity: 1
async_classical: true
async_overlap: 0.5
```

Unknown config fields are rejected so typos do not silently alter experiments.

---

## Output artifacts

### `quport map --out mapped.qasm`

Writes a single OpenQASM 3 circuit after global mapping and routing.

### `quport split --out-dir distributed_out`

Produces:

| File | Description |
|---|---|
| `qpu_<id>.qasm` | Local OpenQASM 3 circuit for QPU `<id>`. |
| `remote_ops.json` | Ordered list of cross-QPU operations. |

### `quport compile-dist --out-dir compile_out`

Produces:

| File | Description |
|---|---|
| `qpu_<id>_routed.qasm` | Locally routed OpenQASM 3 circuit for QPU `<id>`. |
| `remote_ops.json` | Ordered remote-operation trace. |
| `schedule.json` | Strict JSON topology-aware schedule summary produced from `TopologyScheduleSummary.to_dict()`. |
| `schedule_trace.json` | Strict JSON per-layer/per-round communication plan produced from `TopologySchedulePlan.to_dict()`, with absolute timing, QPU-pair packing, port use, link utilization, and unschedulable penalty rounds. |
| `entanglement_plan.json` | Strict JSON bundle with the aggregated EPR blocks (`aggregation`), the $\lambda-1$ e-bit report for the chosen partition (`ebits`), and the entanglement-aware schedule summary (`schedule`). |

Remote operation entries have the shape:

```json
{
  "index": 12,
  "name": "cx",
  "q0_phys": 7,
  "q1_phys": 84,
  "qpu0": 0,
  "qpu1": 9,
  "params": [],
  "clbits": []
}
```

Schedule artifacts are written with `allow_nan=False`, so non-finite values are
rejected instead of being emitted as Python-specific `NaN`/`Infinity` tokens.

---

## CSV schemas

### Benchmark CSV

`quport bench` writes rows with:

| Column | Meaning |
|---|---|
| `trial` | Trial index. |
| `seed` | Random seed used for the trial. |
| `method` | Numeric method id: baseline `0`, balanced `1`, tpccap `2`, tpccap_sa `3`, cluster `4`, ebit `5`. |
| `strategy` | Strategy name. |
| `swaps` | SWAP count. |
| `remote_2q` | Remote two-qubit operation count. |
| `depth` | Circuit depth. |
| `size` | Circuit size. |
| `cost_total` | Total estimated cost. |
| `cost_local` | Local estimated cost. |
| `cost_remote` | Remote estimated cost. |
| `mapping_time_s` | Partition/layout time. |
| `transpile_time_s` | Qiskit transpilation time. |

### Sweep CSV

`quport sweep` writes summary rows with:

| Column | Meaning |
|---|---|
| `intra` | Local topology. |
| `inter` | Inter-QPU topology. |
| `ports` | Communication ports per QPU. |
| `method` | Numeric method id. |
| `swaps_mean` | Mean SWAP count. |
| `remote_2q_mean` | Mean remote two-qubit count. |
| `depth_mean` | Mean depth. |
| `cost_mean` | Mean total estimated cost. |
| `cost_median` | Median total estimated cost. |
| `transpile_time_mean` | Mean transpilation time. |

Cost is reported both ways because it is heavily skewed across random circuits.
Comparing two strategies instance by instance, the per-instance ratio spans roughly
$-50\%$ to $+200\%$, so a handful of hard instances can move the mean far enough to
reverse which strategy looks better while the median points the other way. Quote
whichever you prefer, but say which one, and do not read a small difference in
`cost_mean` as a ranking on its own.

---

## Testing

Install the project and run:

```bash
pytest
```

The repository sets `addopts = "-q"` under `[tool.pytest.ini_options]` in
`pyproject.toml`, so both invocations are quiet and pick up the same settings. The
module form additionally prepends the current directory to `sys.path`:

```bash
python -m pytest
```

Useful optional checks:

```bash
python -m compileall src tests examples
```

```bash
quport --help
```

---

## Design notes and limitations

- Qiskit `CouplingMap` edges are directed, so QuPort explicitly inserts both directions for physically symmetric links.
- Inter-QPU physical connectivity is modeled through communication qubits only.
- The default latency model is intentionally simple and configurable; values are comparative cost units unless you calibrate them to a hardware backend.
- Global mapping can insert cross-QPU routing operations because it exposes the whole modular graph to Qiskit. Use distributed compilation when you need remote operations to remain explicit.
- Topology-aware scheduling is a deterministic estimator, not a full hardware-control stack.
- Diagonality analysis is conservative: an operation QuPort cannot prove diagonal closes the packet, which over-counts EPR pairs rather than claiming a cat copy that would not survive.
- Each gate is charged to exactly one packet root, so the e-bit count is exact for that assignment and an upper bound over all assignments of symmetric gates.
- Teleport blocks are not merged: every non-diagonal cross-QPU gate pays its own round trip of two e-bits.
- Aggregation and entanglement scheduling describe a communication plan; QuPort does not emit the cat-entangler and cat-disentangler circuits themselves.
- Disconnected QPU pairs and zero-capacity communication resources are penalized rather than silently ignored.
- Random benchmark circuits are generated for repeatable experiments; application-specific circuits can be passed directly through the Python API.

---

## Citation

If you use quport in your work and wish to refer to it, please use the following BibTeX entry.

```bibtex
@misc{sarkar2026quporttopologyportcongestionaware,
      title={QuPort: Topology-, Port-, and Congestion-Aware Compilation for Modular Multi-QPU Quantum Systems},
      author={Soumyadip Sarkar and Subhasree Bhattacharjee},
      year={2026},
      eprint={2605.12583},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2605.12583},
}
```

## License

QuPort is licensed under the Apache License 2.0. See [`LICENSE`](LICENSE).
