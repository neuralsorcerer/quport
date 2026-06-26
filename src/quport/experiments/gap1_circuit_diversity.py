"""
gap1_circuit_diversity.py  (backward-compatible revision)
=========================================================
Gap 1 experiment for the QuPort revised manuscript.

Reviewer 1's request (R1Q1): sensitivity to circuit type.

Circuit families:
  qft           Quantum Fourier Transform (all-to-all, distance-decaying)
  efficient_su2 EfficientSU2 linear ansatz (chemistry proxy, nearest-neighbour)
  qaoa          QAOA ansatz on random Erdos-Renyi graph (sparse, irregular)
  random        Random two-qubit circuits (Section 9.1 baseline)

Compatible with Qiskit >= 0.45 (class-based API) AND Qiskit >= 2.1
(function-based API).  The import block below detects which API is present
and aliases everything to the same local names used throughout the script.

Outputs:
  gap1_raw.csv            per-trial raw rows
  gap1_summary.csv        mean +/- std by family/strategy
  gap1_obj_breakdown.png  J(pi) component bar chart
  gap1_proxy_vs_obj.png   proxy cost_total vs J(pi) scatter
  gap1_makespan.png       makespan by family and strategy
"""

import warnings
import statistics
import time
import csv

warnings.filterwarnings("ignore")

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

# ------------------------------------------------------------------
# Qiskit version-adaptive imports
# ------------------------------------------------------------------
import qiskit as _qiskit

_QISKIT_VERSION = tuple(
    int(x) for x in _qiskit.__version__.split(".")[:2]
    if x.isdigit()
)

if _QISKIT_VERSION >= (2, 1):
    # Modern function-based API (Qiskit >= 2.1)
    from qiskit.circuit.library import QFTGate, efficient_su2, qaoa_ansatz

    def _make_qft_circuit(n: int) -> QuantumCircuit:
        qc = QuantumCircuit(n)
        qc.append(QFTGate(n), range(n))
        return qc.decompose()

    def _make_su2_circuit(n: int, rng: np.random.Generator) -> QuantumCircuit:
        qc = efficient_su2(num_qubits=n, reps=3, entanglement="linear")
        params = {p: float(rng.uniform(0, 2 * np.pi)) for p in qc.parameters}
        return qc.assign_parameters(params)

    def _make_qaoa_circuit(
        n: int, rng: np.random.Generator, graph_seed: int
    ) -> QuantumCircuit:
        G = nx.erdos_renyi_graph(n, 0.3, seed=graph_seed)
        if G.number_of_edges() == 0:
            G.add_edge(0, 1)
        terms = [
            ("".join(reversed(["Z" if i in (u, v) else "I" for i in range(n)])), 0.5)
            for u, v in G.edges()
        ]
        H = SparsePauliOp.from_list(terms)
        qc = qaoa_ansatz(H, reps=2)
        params = {p: float(rng.uniform(0, 2 * np.pi)) for p in qc.parameters}
        return qc.assign_parameters(params)

else:
    # Legacy class-based API (Qiskit < 2.1)
    from qiskit.circuit.library import QFT, EfficientSU2

    def _make_qft_circuit(n: int) -> QuantumCircuit:
        return QFT(num_qubits=n, do_swaps=False).decompose()

    def _make_su2_circuit(n: int, rng: np.random.Generator) -> QuantumCircuit:
        qc = EfficientSU2(num_qubits=n, reps=3, entanglement="linear")
        params = {p: float(rng.uniform(0, 2 * np.pi)) for p in qc.parameters}
        return qc.assign_parameters(params)

    def _make_qaoa_circuit(
        n: int, rng: np.random.Generator, graph_seed: int
    ) -> QuantumCircuit:
        # QAOAAnsatz exists in 0.22 but fails with SparsePauliOp float
        # coefficients on that version. Build QAOA from primitive gates
        # (RZZ cost layer + RX mixer) instead — works on all Qiskit versions.
        G = nx.erdos_renyi_graph(n, 0.3, seed=graph_seed)
        if G.number_of_edges() == 0:
            G.add_edge(0, 1)
        qc = QuantumCircuit(n)
        qc.h(range(n))
        for _ in range(2):          # p = 2 layers, same as new-API version
            gamma = float(rng.uniform(0, 2 * np.pi))
            beta  = float(rng.uniform(0, 2 * np.pi))
            for u, v in G.edges():
                qc.rzz(gamma, u, v)
            for i in range(n):
                qc.rx(beta, i)
        return qc


from quport.config import MultiQPUConfig, LatencyModel
from quport.pipeline import (
    map_and_transpile,
    transpile_baseline,
    _translate_to_basis,
    random_benchmark_circuit,
)
from quport.compiler import compile_distributed
from quport.interaction import extract_twoq_weights
from quport.architecture import MultiQPUArchitecture
from quport.partition import _objective_tpccap

# ------------------------------------------------------------------
# Architecture  (identical to Section 9.1 for direct comparability)
# ------------------------------------------------------------------
CFG = MultiQPUConfig(
    n_qpus=4,
    compute_qubits_per_qpu=4,
    comm_qubits_per_qpu=1,
    intra_topology="line",
    inter_topology="ring",
    optimization_level=1,
)
LATENCY = LatencyModel()
N_LOGICAL = 15
DEPTH_RANDOM = 100
TRIALS = 20
STRATEGIES = ["baseline", "balanced", "tpccap", "tpccap_sa"]
TEMPORAL_DECAY = 1.0


# ------------------------------------------------------------------
# Circuit generators
# ------------------------------------------------------------------

def make_qft(n: int, seed: int) -> QuantumCircuit:
    """QFT — all-to-all, distance-decaying interaction weights."""
    return _make_qft_circuit(n)


def make_efficient_su2(n: int, seed: int) -> QuantumCircuit:
    """EfficientSU2 linear ansatz — chemistry proxy, nearest-neighbour."""
    rng = np.random.default_rng(seed)
    return _make_su2_circuit(n, rng)


def make_qaoa(n: int, seed: int) -> QuantumCircuit:
    """QAOA ansatz — sparse, irregular long-range interactions."""
    rng = np.random.default_rng(seed)
    graph_seed = int(rng.integers(0, 2 ** 31))
    return _make_qaoa_circuit(n, rng, graph_seed)


def make_random(n: int, seed: int) -> QuantumCircuit:
    """Random two-qubit circuit — matches Section 9.1 exactly."""
    return random_benchmark_circuit(n, DEPTH_RANDOM, seed)


CIRCUIT_FAMILIES = {
    "qft":           make_qft,
    "efficient_su2": make_efficient_su2,
    "qaoa":          make_qaoa,
    "random":        make_random,
}

FAMILY_LABELS = {
    "qft":           "QFT",
    "efficient_su2": "EfficientSU2\n(chemistry proxy)",
    "qaoa":          "QAOA",
    "random":        "Random",
}


# ------------------------------------------------------------------
# J(pi) helper
# ------------------------------------------------------------------

_ARCH = MultiQPUArchitecture(CFG)
_SP   = _ARCH.qpu_shortest_paths()


def evaluate_objective(weights, part):
    obj, diag = _objective_tpccap(
        weights=weights,
        part=part,
        n_qpus=CFG.n_qpus,
        comm_ports_per_qpu=CFG.comm_qubits_per_qpu,
        sp=_SP,
        w_dist=1.0,
        w_port=5.0,
        w_cong=0.05,
        congestion_routing="ecmp",
    )
    return obj, diag.weighted_cut_distance, diag.port_overflow_l2, diag.congestion_l2


# ------------------------------------------------------------------
# Single-trial runner
# ------------------------------------------------------------------

def run_trial(family_name: str, seed: int) -> list:
    gen = CIRCUIT_FAMILIES[family_name]
    qc  = gen(N_LOGICAL, seed)

    qc_basis = _translate_to_basis(qc, CFG.basis_gates, seed)
    weights  = extract_twoq_weights(qc_basis)

    rows = []
    for strategy in STRATEGIES:
        t0 = time.perf_counter()

        # global path
        if strategy == "baseline":
            gres = transpile_baseline(qc, CFG, latency=LATENCY, seed=seed)
        else:
            gres = map_and_transpile(
                qc, CFG, latency=LATENCY, seed=seed, strategy=strategy
            )

        J, cut_dist, port_ovf, cong = evaluate_objective(weights, gres.partition)

        # distributed path
        dist_strat = strategy if strategy != "baseline" else "balanced"
        dres = compile_distributed(
            qc, CFG, latency=LATENCY, seed=seed,
            strategy=dist_strat, temporal_decay=TEMPORAL_DECAY,
        )

        elapsed = time.perf_counter() - t0

        rows.append({
            "family":         family_name,
            "strategy":       strategy,
            "seed":           seed,
            "cost_total":     gres.cost.total,
            "remote_2q":      gres.metrics.remote_2q,
            "depth":          gres.metrics.depth,
            "swaps":          gres.metrics.swaps,
            "J":              J,
            "cut_distance":   cut_dist,
            "port_overflow":  port_ovf,
            "congestion":     cong,
            "makespan":       dres.schedule.makespan,
            "remote_rounds":  dres.schedule.remote_rounds,
            "dist_remote_2q": dres.global_metrics.remote_2q,
            "dist_cut":       dres.partition_cut,
            "wall_time_s":    elapsed,
        })
    return rows


# ------------------------------------------------------------------
# Full runner
# ------------------------------------------------------------------

def run_all() -> list:
    all_rows = []
    total = len(CIRCUIT_FAMILIES) * TRIALS
    done  = 0
    for family in CIRCUIT_FAMILIES:
        for seed in range(TRIALS):
            all_rows.extend(run_trial(family, seed))
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  {done}/{total} trials complete")
    return all_rows


# ------------------------------------------------------------------
# CSV helpers
# ------------------------------------------------------------------

def write_raw_csv(rows: list, path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def summarise(rows: list) -> list:
    numeric_keys = [
        "cost_total", "remote_2q", "depth", "swaps",
        "J", "cut_distance", "port_overflow", "congestion",
        "makespan", "remote_rounds", "dist_remote_2q", "dist_cut", "wall_time_s",
    ]
    groups = {}
    for row in rows:
        groups.setdefault((row["family"], row["strategy"]), []).append(row)

    summary = []
    for (family, strategy), grp in sorted(groups.items()):
        rec = {"family": family, "strategy": strategy, "n_trials": len(grp)}
        for k in numeric_keys:
            vals = [r[k] for r in grp]
            rec[f"{k}_mean"] = statistics.mean(vals)
            rec[f"{k}_std"]  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary.append(rec)
    return summary


def write_summary_csv(summary: list, path: str) -> None:
    if not summary:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"wrote {path} ({len(summary)} rows)")


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------

STRATEGY_COLORS = {
    "baseline":  "#4878CF",
    "balanced":  "#6ACC65",
    "tpccap":    "#D65F5F",
    "tpccap_sa": "#B47CC7",
}
STRATEGY_LABELS = {
    "baseline":  "Baseline",
    "balanced":  "Balanced greedy",
    "tpccap":    "TPCCAP",
    "tpccap_sa": "TPCCAP-SA",
}
FAMILIES = list(CIRCUIT_FAMILIES.keys())
N_FAM    = len(FAMILIES)


def _mean(summary, family, strategy, key):
    for row in summary:
        if row["family"] == family and row["strategy"] == strategy:
            return row[f"{key}_mean"]
    return float("nan")


def _std(summary, family, strategy, key):
    for row in summary:
        if row["family"] == family and row["strategy"] == strategy:
            return row[f"{key}_std"]
    return 0.0


def plot_objective_breakdown(summary, path):
    components  = ["cut_distance", "port_overflow", "congestion"]
    comp_labels = ["Weighted cut distance", "Port-overflow L2", "Congestion L2"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"J(\u03c0) component breakdown by circuit family and strategy\n"
        f"(n={N_LOGICAL}, P=1, 4 QPUs, {TRIALS} trials each)",
        fontsize=12,
    )
    x         = np.arange(N_FAM)
    bar_width = 0.18
    offsets   = np.linspace(
        -(len(STRATEGIES) - 1) / 2,
         (len(STRATEGIES) - 1) / 2,
        len(STRATEGIES),
    ) * bar_width

    for ax, comp, comp_label in zip(axes, components, comp_labels):
        for strat, offset in zip(STRATEGIES, offsets):
            means = [_mean(summary, fam, strat, comp) for fam in FAMILIES]
            stds  = [_std( summary, fam, strat, comp) for fam in FAMILIES]
            ax.bar(x + offset, means, bar_width, yerr=stds, capsize=3,
                   color=STRATEGY_COLORS[strat], label=STRATEGY_LABELS[strat],
                   alpha=0.88)
        ax.set_title(comp_label, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([FAMILY_LABELS[f] for f in FAMILIES], fontsize=8)
        ax.set_ylabel("Mean value")
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"wrote {path}")


def plot_proxy_vs_objective(summary, path):
    markers = {"qft": "o", "efficient_su2": "s", "qaoa": "^", "random": "D"}
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(7, 5))
    for strat in STRATEGIES:
        for fam in FAMILIES:
            ax.scatter(
                _mean(summary, fam, strat, "J"),
                _mean(summary, fam, strat, "cost_total"),
                color=STRATEGY_COLORS[strat],
                marker=markers[fam],
                s=90, zorder=3,
            )

    strat_handles = [
        Line2D([0], [0], marker="o", color=STRATEGY_COLORS[s], linestyle="none",
               markersize=8, label=STRATEGY_LABELS[s])
        for s in STRATEGIES
    ]
    fam_handles = [
        Line2D([0], [0], marker=markers[f], color="gray", linestyle="none",
               markersize=8, label=FAMILY_LABELS[f].replace("\n", " "))
        for f in FAMILIES
    ]
    leg1 = ax.legend(handles=strat_handles, title="Strategy",
                     loc="upper left", fontsize=8, title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=fam_handles, title="Circuit family",
              loc="lower right", fontsize=8, title_fontsize=8)

    ax.set_xlabel("J(\u03c0)  [TPCCAP objective \u2014 lower is better]", fontsize=10)
    ax.set_ylabel("cost_total  [proxy \u2014 lower is better]", fontsize=10)
    ax.set_title(
        "Proxy cost_total vs. TPCCAP objective across circuit families\n"
        f"(n={N_LOGICAL}, P=1, 4 QPUs, {TRIALS} trials each)",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"wrote {path}")


def plot_makespan(summary, path):
    strats_dist = ["balanced", "tpccap", "tpccap_sa"]
    labels_dist = ["Balanced greedy", "TPCCAP", "TPCCAP-SA"]
    colors_dist = [STRATEGY_COLORS[s] for s in strats_dist]

    x         = np.arange(N_FAM)
    bar_width = 0.22
    offsets   = np.linspace(
        -(len(strats_dist) - 1) / 2,
         (len(strats_dist) - 1) / 2,
        len(strats_dist),
    ) * bar_width

    fig, ax = plt.subplots(figsize=(10, 5))
    for strat, label, color, offset in zip(
        strats_dist, labels_dist, colors_dist, offsets
    ):
        means = [_mean(summary, fam, strat, "makespan") for fam in FAMILIES]
        stds  = [_std( summary, fam, strat, "makespan") for fam in FAMILIES]
        ax.bar(x + offset, means, bar_width, yerr=stds, capsize=3,
               color=color, label=label, alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABELS[f] for f in FAMILIES], fontsize=9)
    ax.set_ylabel("Makespan (abstract units)")
    ax.set_title(
        "Makespan by circuit family and strategy (distributed compilation path)\n"
        f"(n={N_LOGICAL}, P=1, 4 QPUs, {TRIALS} trials, temporal_decay=1.0)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"wrote {path}")


def print_headline_table(summary):
    print("\n" + "=" * 70)
    print(f"{'Family':<16}  {'J(pi) balanced':>14}  {'J(pi) tpccap':>12}  "
          f"{'Reduction':>10}  {'cost_total D':>13}")
    print("-" * 70)
    for fam in FAMILIES:
        J_bal = _mean(summary, fam, "balanced", "J")
        J_tp  = _mean(summary, fam, "tpccap",  "J")
        C_bal = _mean(summary, fam, "balanced", "cost_total")
        C_tp  = _mean(summary, fam, "tpccap",  "cost_total")
        J_red = (J_bal - J_tp) / J_bal * 100 if J_bal else float("nan")
        C_chg = (C_tp - C_bal) / C_bal * 100  if C_bal else float("nan")
        print(f"{fam:<16}  {J_bal:>14.1f}  {J_tp:>12.1f}  "
              f"{J_red:>+9.1f}%  {C_chg:>+12.1f}%")
    print("=" * 70)
    print("Negative J reduction = TPCCAP improves objective.")
    print("Positive cost_total D = TPCCAP proxy metric is worse.\n")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Gap 1 — circuit diversity experiment")
    print(f"  Qiskit version : {_qiskit.__version__}")
    print(f"  API mode       : {'new (>= 2.1)' if _QISKIT_VERSION >= (2,1) else 'legacy (< 2.1)'}")
    print(f"  families       : {list(CIRCUIT_FAMILIES)}")
    print(f"  strategies     : {STRATEGIES}")
    print(f"  n_logical      : {N_LOGICAL},  trials: {TRIALS}")
    print(f"  architecture   : {CFG.n_qpus} QPUs, "
          f"C={CFG.compute_qubits_per_qpu}, P={CFG.comm_qubits_per_qpu}, "
          f"{CFG.intra_topology}/{CFG.inter_topology}")
    print()

    t_start  = time.time()
    all_rows = run_all()
    print(f"\nAll trials done in {time.time() - t_start:.1f}s")

    write_raw_csv(all_rows,             "gap1_raw.csv")
    summary = summarise(all_rows)
    write_summary_csv(summary,          "gap1_summary.csv")
    print_headline_table(summary)

    print("Generating figures...")
    plot_objective_breakdown(summary,   "gap1_obj_breakdown.png")
    plot_proxy_vs_objective(summary,    "gap1_proxy_vs_obj.png")
    plot_makespan(summary,              "gap1_makespan.png")

    print("\nDone. Outputs:")
    for f in ["gap1_raw.csv", "gap1_summary.csv",
              "gap1_obj_breakdown.png", "gap1_proxy_vs_obj.png", "gap1_makespan.png"]:
        print(f"  {f}")
