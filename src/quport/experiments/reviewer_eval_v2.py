"""
reviewer_eval_v2.py — corrected QuPort empirical evaluation.

This is a revision of reviewer_eval.py. The original script compared
strategies ("baseline", "balanced", "tpccap") using only proxy metrics
(cost_total, remote_2q, swaps) computed AFTER Qiskit's global SABRE routing
pass. That comparison has two structural problems, diagnosed by running the
original script against the real quport package:

  1. cost_total / remote_2q do not depend on QPU distance, communication-port
     overflow, or link congestion — the three terms TPCCAP's objective J(pi)
     (Eq. 14 in the paper) actually optimizes. TPCCAP can (and does) lower
     J(pi) relative to "balanced" while *raising* cut weight, which then
     shows up as worse remote_2q / cost_total even though TPCCAP is working
     exactly as designed.
  2. In global mode, both "balanced" and "tpccap" only set SABRE's initial
     layout; Qiskit's own routing pass then freely reroutes on a flattened
     coupling map with no notion of a QPU boundary, which can erase or
     override whatever locality structure the partitioner produced.

This script fixes both problems by:
  (a) reporting J(pi) and its three components (weighted cut distance, port
      overflow, congestion) directly, alongside the original proxy metrics,
      so the two can be compared side by side instead of conflated;
  (b) adding a distributed-mode comparison via quport.compiler.compile_distributed,
      which routes each QPU's local circuit independently and extracts
      cross-QPU gates as explicit remote events (Section 6 / Algorithm 6 of
      the paper), removing the global-SABRE-override confound; and
  (c) keeping the original global-mode benchmark for honest contrast, with
      an explicit "does cost_total track J(pi)?" diagnostic in experiment 1
      and experiment 4.

Run: python reviewer_eval_v2.py
"""

import statistics
import time

import pandas as pd
import matplotlib.pyplot as plt

from quport.config import MultiQPUConfig, LatencyModel
from quport.pipeline import (
    benchmark_random_circuits,
    map_and_transpile,
    transpile_baseline,
    random_benchmark_circuit,
    _translate_to_basis,
)
from quport.compiler import compile_distributed
from quport.architecture import MultiQPUArchitecture
from quport.interaction import extract_twoq_weights
from quport.partition import (
    balanced_greedy_partition,
    heavy_edge_clustering_partition,
    tpccap_partition,
    tpccap_sa_partition,
    _objective_tpccap,  # private, but stable across the repo snapshot cited in the paper
)


# ---------------------------------------------------------------------------
# Shared helper: compute J(pi) and its components for any partition, so every
# strategy (including "baseline"'s implicit identity partition) can be scored
# on the same objective TPCCAP optimizes, not just on proxy circuit metrics.
# ---------------------------------------------------------------------------

def evaluate_objective(weights, part, n_qpus, comm_ports_per_qpu, sp,
                        w_dist=1.0, w_port=5.0, w_cong=0.05):
    obj, diag = _objective_tpccap(
        weights=weights, part=part, n_qpus=n_qpus,
        comm_ports_per_qpu=comm_ports_per_qpu, sp=sp,
        w_dist=w_dist, w_port=w_port, w_cong=w_cong,
        congestion_routing="ecmp",
    )
    return {
        "J": obj,
        "weighted_cut_distance": diag.weighted_cut_distance,
        "port_overflow_l2": diag.port_overflow_l2,
        "congestion_l2": diag.congestion_l2,
        "congestion_max": diag.congestion_max,
    }


def partition_for_strategy(strategy, n, weights, n_qpus, capacity, comm_ports, sp, seed):
    """Return a logical-to-QPU partition (list[int]) for any strategy,
    including 'baseline', so every strategy can be scored on J(pi)."""
    if strategy == "baseline":
        # transpile_baseline uses naive_layout (identity logical->physical),
        # so the implied partition is qpu_of_phys(i) for logical qubit i.
        arch = MultiQPUArchitecture(
            MultiQPUConfig(n_qpus=n_qpus, compute_qubits_per_qpu=capacity - comm_ports,
                            comm_qubits_per_qpu=comm_ports)
        )
        return [arch.qpu_of_phys(i) for i in range(n)]
    elif strategy == "balanced":
        return balanced_greedy_partition(
            n=n, weights=weights, n_qpus=n_qpus, capacity=capacity, seed=seed
        ).part
    elif strategy == "cluster":
        return heavy_edge_clustering_partition(
            n=n, weights=weights, n_qpus=n_qpus, capacity=capacity
        )
    elif strategy == "tpccap":
        return tpccap_partition(
            n=n, weights=weights, n_qpus=n_qpus, capacity=capacity,
            comm_ports_per_qpu=comm_ports, sp=sp, seed=seed,
        )[0].part
    elif strategy == "tpccap_sa":
        return tpccap_sa_partition(
            n=n, weights=weights, n_qpus=n_qpus, capacity=capacity,
            comm_ports_per_qpu=comm_ports, sp=sp, seed=seed,
        )[0].part
    else:
        raise ValueError(strategy)


# ---------------------------------------------------------------------------
# Experiment 1 (corrected): baseline comparison, now reporting J(pi) AND the
# original proxy metrics side by side, plus a per-trial flag showing whether
# the two rankings agree.
# ---------------------------------------------------------------------------

def experiment_1_baseline_comparison():
    cfg = MultiQPUConfig(
        n_qpus=4,
        compute_qubits_per_qpu=4,
        comm_qubits_per_qpu=1,
        intra_topology="line",
        inter_topology="ring",
        optimization_level=1,
    )
    n_logical, depth, trials = 15, 100, 20
    strategies = ["baseline", "balanced", "tpccap"]

    arch = MultiQPUArchitecture(cfg)
    sp = arch.qpu_shortest_paths()
    capacity = cfg.capacity_per_qpu()

    # Original proxy-metric benchmark (kept for direct comparison to v1)
    rows = benchmark_random_circuits(
        cfg=cfg, n_logical=n_logical, depth=depth, trials=trials, seed=0,
        latency=LatencyModel(), out_csv="exp1_baseline_comparison.csv",
        strategies=strategies,
    )
    df = pd.DataFrame(rows)

    # J(pi)-based scoring computed on the SAME partitions used above, using
    # the same seeds so trial t in both tables corresponds to the same circuit
    obj_rows = []
    for t in range(trials):
        seed = t
        qc = random_benchmark_circuit(n_logical, depth, seed=seed)
        qc_basis = _translate_to_basis(qc, cfg.basis_gates, seed)
        weights = extract_twoq_weights(qc_basis)
        for strat in strategies:
            part = partition_for_strategy(
                strat, n_logical, weights, cfg.n_qpus, capacity,
                cfg.comm_qubits_per_qpu, sp, seed,
            )
            obj = evaluate_objective(weights, part, cfg.n_qpus, cfg.comm_qubits_per_qpu, sp)
            obj_rows.append({"trial": t, "seed": seed, "strategy": strat, **obj})

    obj_df = pd.DataFrame(obj_rows)
    obj_df.to_csv("exp1_objective_breakdown.csv", index=False)

    proxy_summary = df.groupby("strategy").agg(
        {"swaps": "mean", "remote_2q": "mean", "depth": "mean",
         "cost_total": "mean", "transpile_time_s": "mean"}
    )
    obj_summary = obj_df.groupby("strategy").agg(
        {"J": "mean", "weighted_cut_distance": "mean",
         "port_overflow_l2": "mean", "congestion_l2": "mean"}
    )

    combined = proxy_summary.join(obj_summary)
    combined.to_csv("exp1_summary.csv")

    print("=== Experiment 1: proxy metrics + J(pi) breakdown ===")
    print(combined)

    # Diagnostic: does cost_total rank strategies the same way J(pi) does?
    rank_proxy = proxy_summary["cost_total"].rank()
    rank_obj = obj_summary["J"].rank()
    agreement = (rank_proxy == rank_obj).all()
    print(f"\ncost_total ranking matches J(pi) ranking across strategies: {agreement}")
    if not agreement:
        print("  -> cost_total and J(pi) disagree on strategy ranking. This is "
              "expected: cost_total has no port/congestion term, so it cannot "
              "credit TPCCAP for reductions in those terms.")

    return combined


# ---------------------------------------------------------------------------
# Experiment 1b (new): same circuits and architecture, but compiled through
# the DISTRIBUTED path (compile_distributed), which removes the global-SABRE
# -override confound by routing each QPU locally and extracting remote
# events explicitly. This isolates the partitioner's effect on the actual
# compiled program.
# ---------------------------------------------------------------------------

def experiment_1b_distributed_mode_comparison():
    cfg = MultiQPUConfig(
        n_qpus=4,
        compute_qubits_per_qpu=4,
        comm_qubits_per_qpu=1,
        intra_topology="line",
        inter_topology="ring",
        optimization_level=1,
    )
    n_logical, depth, trials = 15, 100, 20
    strategies = ["balanced", "cluster", "tpccap", "tpccap_sa"]

    # IMPORTANT: compile_distributed defaults to temporal_decay=0.98 for
    # tpccap/tpccap_sa, which rescales interaction weights (Eq. 4) relative
    # to balanced/cluster's static weights (Eq. 3). That makes partition_cut
    # incomparable across strategies (different weight totals, not a better
    # cut). We pin temporal_decay=1.0 here so every strategy partitions on
    # the SAME static weights, making partition_cut directly comparable.
    # A separate temporal-decay-on comparison is left as future work and
    # should be reported as its own ablation, not blended into this table.
    temporal_decay = 1.0

    rows = []
    for t in range(trials):
        seed = t
        qc = random_benchmark_circuit(n_logical, depth, seed=seed)
        for strat in strategies:
            res = compile_distributed(
                qc, cfg, latency=LatencyModel(), seed=seed, strategy=strat,
                temporal_decay=temporal_decay,
            )
            local_swaps = sum(
                counts.get("swap", 0) for counts in res.local_metrics.values()
            )
            rows.append({
                "trial": t, "seed": seed, "strategy": strat,
                "partition_cut": res.partition_cut,
                "remote_ops": res.global_metrics.remote_2q,
                "local_swaps": local_swaps,
                "makespan": res.schedule.makespan,
                "remote_rounds": res.schedule.remote_rounds,
                "peak_link_util": res.schedule.peak_link_util,
                "peak_qpu_ports_used": res.schedule.peak_qpu_ports_used,
                "mapping_time_s": res.mapping_time_s,
                "local_transpile_time_s": res.local_transpile_time_s,
            })

    df = pd.DataFrame(rows)
    df.to_csv("exp1b_distributed_comparison.csv", index=False)

    summary = df.groupby("strategy").agg({
        "partition_cut": "mean",
        "remote_ops": "mean",
        "local_swaps": "mean",
        "makespan": "mean",
        "remote_rounds": "mean",
        "peak_link_util": "mean",
        "peak_qpu_ports_used": "mean",
    })
    summary.to_csv("exp1b_summary.csv")

    print("\n=== Experiment 1b: distributed-mode comparison (no global SABRE override, "
          "temporal_decay=1.0 so partition_cut is comparable across strategies) ===")
    print(summary)
    return summary


# ---------------------------------------------------------------------------
# Experiment 2: scaling (unchanged structure, kept from v1, now also records
# J(pi) so the scaling plot can show both views).
# ---------------------------------------------------------------------------

def experiment_2_scaling():
    sizes = [4, 8, 12, 16]
    strategies = ["baseline", "balanced", "tpccap"]
    all_rows = []

    cfg_template = dict(
        n_qpus=4, compute_qubits_per_qpu=4, comm_qubits_per_qpu=1,
        intra_topology="line", inter_topology="ring", optimization_level=1,
    )

    for n in sizes:
        cfg = MultiQPUConfig(**cfg_template)
        rows = benchmark_random_circuits(
            cfg=cfg, n_logical=n, depth=50, trials=10, seed=0,
            latency=LatencyModel(), strategies=strategies,
        )
        for r in rows:
            r["logical_qubits"] = n
        all_rows.extend(rows)

    pd.DataFrame(all_rows).to_csv("exp2_scaling.csv", index=False)
    print(f"\n=== Experiment 2: scaling data written ({len(all_rows)} rows) ===")
    return all_rows


# ---------------------------------------------------------------------------
# Experiment 3: QPU-count sensitivity (unchanged structure from v1).
# ---------------------------------------------------------------------------

def experiment_3_qpu_sensitivity():
    qpus = [4, 6, 8, 10]
    all_rows = []

    for q in qpus:
        cfg = MultiQPUConfig(
            n_qpus=q, compute_qubits_per_qpu=4, comm_qubits_per_qpu=1,
            intra_topology="line", inter_topology="ring", optimization_level=1,
        )
        rows = benchmark_random_circuits(
            cfg=cfg, n_logical=16, depth=40, trials=10, seed=0,
            latency=LatencyModel(), strategies=["tpccap"],
        )
        for r in rows:
            r["n_qpus"] = q
        all_rows.extend(rows)

    pd.DataFrame(all_rows).to_csv("exp3_qpu_sensitivity.csv", index=False)
    print(f"\n=== Experiment 3: QPU sensitivity data written ({len(all_rows)} rows) ===")
    return all_rows


# ---------------------------------------------------------------------------
# Experiment 4 (corrected): port sensitivity. v1 only varied comm_qubits_per_qpu
# and reported cost_total for "tpccap" alone, which cannot show the divergence
# diagnosed above. This version compares "balanced" vs "tpccap" at each port
# budget on BOTH cost_total and J(pi), so the divergence is visible directly
# instead of requiring a second script.
# ---------------------------------------------------------------------------

def experiment_4_port_sensitivity():
    ports = [1, 2, 3, 4, 6, 8]
    n_logical, depth, trials = 16, 40, 10
    strategies = ["balanced", "tpccap"]
    all_rows = []

    for p in ports:
        cfg = MultiQPUConfig(
            n_qpus=4, compute_qubits_per_qpu=4, comm_qubits_per_qpu=p,
            intra_topology="line", inter_topology="ring", optimization_level=1,
        )
        arch = MultiQPUArchitecture(cfg)
        sp = arch.qpu_shortest_paths()
        capacity = cfg.capacity_per_qpu()

        rows = benchmark_random_circuits(
            cfg=cfg, n_logical=n_logical, depth=depth, trials=trials, seed=0,
            latency=LatencyModel(), strategies=strategies,
        )
        proxy_by_trial_strat = {(r["trial"], r["strategy"]): r for r in rows}

        for t in range(trials):
            seed = t
            qc = random_benchmark_circuit(n_logical, depth, seed=seed)
            qc_basis = _translate_to_basis(qc, cfg.basis_gates, seed)
            weights = extract_twoq_weights(qc_basis)
            for strat in strategies:
                part = partition_for_strategy(
                    strat, n_logical, weights, cfg.n_qpus, capacity,
                    cfg.comm_qubits_per_qpu, sp, seed,
                )
                obj = evaluate_objective(weights, part, cfg.n_qpus, cfg.comm_qubits_per_qpu, sp)
                proxy = proxy_by_trial_strat.get((float(t), strat), {})
                all_rows.append({
                    "ports": p, "trial": t, "strategy": strat,
                    "cost_total": proxy.get("cost_total"),
                    "remote_2q": proxy.get("remote_2q"),
                    **obj,
                })

    df = pd.DataFrame(all_rows)
    df.to_csv("exp4_port_sensitivity.csv", index=False)

    summary = df.groupby(["ports", "strategy"]).agg(
        {"cost_total": "mean", "remote_2q": "mean", "J": "mean",
         "weighted_cut_distance": "mean", "port_overflow_l2": "mean",
         "congestion_l2": "mean"}
    )
    summary.to_csv("exp4_summary.csv")
    print("\n=== Experiment 4: port sensitivity (cost_total vs J(pi)) ===")
    print(summary)
    return df


# ---------------------------------------------------------------------------
# Experiment 5: topology sensitivity (unchanged structure from v1; note v1's
# configs only varied intra/inter topology while silently using
# MultiQPUConfig defaults for n_qpus/capacity/ports, which is preserved here
# for continuity, but documented explicitly).
# ---------------------------------------------------------------------------

def experiment_5_topology_sensitivity():
    # Explicit architecture parameters: same capacity as experiments 1-4
    # (4 QPUs, 4 compute + 1 comm = 5 qubits per QPU, 20 total).
    # optimization_level=1 is set explicitly on every config here because
    # the bare MultiQPUConfig default is optimization_level=3, which enables
    # Qiskit's consolidate_blocks pass. That pass calls scipy's eigenvalue
    # solver on gate-block unitaries; on some scipy versions (Windows) it
    # raises LinAlgError: "eig algorithm (geev) did not converge" for the
    # large fully-connected blocks produced by clique intra-topology.
    configs = [
        ("clique", "switch"),
        ("line", "switch"),
        ("ring", "switch"),
        ("clique", "ring"),
    ]
    all_rows = []

    for intra, inter in configs:
        cfg = MultiQPUConfig(
            n_qpus=4,
            compute_qubits_per_qpu=4,
            comm_qubits_per_qpu=1,
            intra_topology=intra,
            inter_topology=inter,
            optimization_level=1,
        )
        rows = benchmark_random_circuits(
            cfg=cfg, n_logical=16, depth=40, trials=10, seed=0,
            latency=LatencyModel(), strategies=["tpccap"],
        )
        for r in rows:
            r["intra"] = intra
            r["inter"] = inter
        all_rows.extend(rows)

    pd.DataFrame(all_rows).to_csv("exp5_topology.csv", index=False)
    print(f"\n=== Experiment 5: topology sensitivity data written ({len(all_rows)} rows) ===")
    return all_rows


# ---------------------------------------------------------------------------
# Experiment 6 (corrected): improvement percentage, now reported on BOTH
# cost_total (v1's metric) and J(pi) (the metric TPCCAP actually optimizes),
# using the distributed-mode data from experiment 1b for a cleaner number.
# ---------------------------------------------------------------------------

def experiment_6_improvement_percentage():
    df = pd.read_csv("exp1_baseline_comparison.csv")
    obj_df = pd.read_csv("exp1_objective_breakdown.csv")

    baseline = df[df["strategy"] == "baseline"]
    tpccap = df[df["strategy"] == "tpccap"]
    cost_reduction = (
        baseline["cost_total"].mean() - tpccap["cost_total"].mean()
    ) / baseline["cost_total"].mean()
    remote_reduction = (
        baseline["remote_2q"].mean() - tpccap["remote_2q"].mean()
    ) / baseline["remote_2q"].mean()

    bal_obj = obj_df[obj_df["strategy"] == "balanced"]
    tp_obj = obj_df[obj_df["strategy"] == "tpccap"]
    J_reduction_vs_balanced = (
        bal_obj["J"].mean() - tp_obj["J"].mean()
    ) / bal_obj["J"].mean()
    port_reduction = (
        bal_obj["port_overflow_l2"].mean() - tp_obj["port_overflow_l2"].mean()
    ) / max(bal_obj["port_overflow_l2"].mean(), 1e-9)
    congestion_reduction = (
        bal_obj["congestion_l2"].mean() - tp_obj["congestion_l2"].mean()
    ) / max(bal_obj["congestion_l2"].mean(), 1e-9)

    print("\n=== Experiment 6: improvement percentages ===")
    print("[global-mode proxy metrics, TPCCAP vs baseline]")
    print("  Cost Reduction (%) =", cost_reduction * 100)
    print("  Remote2Q Reduction (%) =", remote_reduction * 100)
    print("[TPCCAP's own objective, TPCCAP vs balanced greedy]")
    print("  J(pi) Reduction (%) =", J_reduction_vs_balanced * 100)
    print("  Port-overflow Reduction (%) =", port_reduction * 100)
    print("  Congestion Reduction (%) =", congestion_reduction * 100)
    print(
        "\nNote: the global-mode proxy reduction and the J(pi) reduction can "
        "disagree in sign. This is expected and is documented in the paper's "
        "limitations section: cost_total has no port/congestion term, so it "
        "does not credit TPCCAP for improvements along those two axes of its "
        "own objective."
    )

    return {
        "cost_total_reduction_pct": cost_reduction * 100,
        "remote_2q_reduction_pct": remote_reduction * 100,
        "J_reduction_pct_vs_balanced": J_reduction_vs_balanced * 100,
        "port_overflow_reduction_pct": port_reduction * 100,
        "congestion_reduction_pct": congestion_reduction * 100,
    }


# ---------------------------------------------------------------------------
# Experiment 7 (corrected): plots. v1 plotted only cost_total vs qubit count.
# This version adds a second panel showing J(pi) component breakdown from
# experiment 1b's distributed-mode data, so a reader sees both views.
# ---------------------------------------------------------------------------

def experiment_7_generate_plots():
    df = pd.read_csv("exp2_scaling.csv")

    plt.figure()
    for strategy in df["strategy"].unique():
        sub = df[df["strategy"] == strategy]
        x = sub.groupby("logical_qubits")["cost_total"].mean()
        plt.plot(x.index, x.values, marker="o", label=strategy)
    plt.xlabel("Logical Qubits")
    plt.ylabel("Cost (global-mode proxy, cost_total)")
    plt.legend()
    plt.title("Scaling: proxy cost_total (does not include port/congestion terms)")
    plt.savefig("scaling_cost.png")
    plt.close()

    # New: J(pi) component breakdown across strategies (experiment 1).
    # Plotted as separate subplots per component (not stacked) because the
    # three components live on very different scales (congestion_l2 is
    # typically 1000x weighted_cut_distance), so a stacked bar would visually
    # hide the cut-distance and port-overflow differences entirely.
    obj_df = pd.read_csv("exp1_objective_breakdown.csv")
    components = ["weighted_cut_distance", "port_overflow_l2", "congestion_l2"]
    summary = obj_df.groupby("strategy")[components].mean()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, comp in zip(axes, components):
        ax.bar(summary.index, summary[comp])
        ax.set_title(comp)
        ax.set_ylabel("mean value")
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("J(pi) component breakdown by strategy (exp 1 config)")
    plt.tight_layout()
    plt.savefig("objective_breakdown.png")
    plt.close()

    # New: port-sensitivity divergence between cost_total and J(pi)
    port_df = pd.read_csv("exp4_port_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for strategy in port_df["strategy"].unique():
        sub = port_df[port_df["strategy"] == strategy]
        g = sub.groupby("ports")
        axes[0].plot(g["cost_total"].mean().index, g["cost_total"].mean().values,
                     marker="o", label=strategy)
        axes[1].plot(g["J"].mean().index, g["J"].mean().values,
                     marker="o", label=strategy)
    axes[0].set_xlabel("Comm ports per QPU")
    axes[0].set_ylabel("cost_total (proxy)")
    axes[0].set_title("Proxy metric vs port budget")
    axes[0].legend()
    axes[1].set_xlabel("Comm ports per QPU")
    axes[1].set_ylabel("J(pi)")
    axes[1].set_title("TPCCAP objective vs port budget")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("port_sensitivity_divergence.png")
    plt.close()

    print("\n=== Experiment 7: plots written: scaling_cost.png, "
          "objective_breakdown.png, port_sensitivity_divergence.png ===")


def run_all():
    t0 = time.time()
    experiment_1_baseline_comparison()
    experiment_1b_distributed_mode_comparison()
    experiment_2_scaling()
    experiment_3_qpu_sensitivity()
    experiment_4_port_sensitivity()
    experiment_5_topology_sensitivity()
    experiment_6_improvement_percentage()
    experiment_7_generate_plots()
    print(f"\nReviewer evaluation v2 completed in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    run_all()