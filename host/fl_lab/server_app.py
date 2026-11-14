from __future__ import annotations

import json

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from . import fl_model_ref as ref
from .fedlib import GlobalEvaluator, RunLogs, run_dir

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = dict(context.run_config)
    n = int(cfg["num-clients"])
    out = run_dir(cfg)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"{out} не порожній: оберіть інший results-dir")
    RunLogs(out)
    (out / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))

    initial = ref.init_weights(int(cfg["seed"]))          
    evaluator = GlobalEvaluator(cfg, initial)

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        rec = evaluator(server_round, arrays.to_numpy_ndarrays())
        return MetricRecord({"accuracy": rec["acc"], "loss": rec["loss"]})

    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=0.0,            
        min_train_nodes=n,
        min_available_nodes=n,
        weighted_by_key="num-examples",   
    )
    strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(initial),
        train_config=ConfigRecord({}),
        num_rounds=int(cfg["num-rounds"]),
        evaluate_fn=global_evaluate,
    )
