from quport.pipeline import sweep_topologies

sweep_topologies(
    n_logical=24,
    depth=8,
    trials=1,
    seed=7,
    out_csv="sweep.csv",
    intra_topologies=("clique", "ring"),
    inter_topologies=("switch", "ring"),
    comm_ports=(1,),
    compute_per_qpu=4,
    n_qpus=6,
)
print("Wrote sweep.csv")
