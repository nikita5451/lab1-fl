from __future__ import annotations

import json
import time

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common.logger import log
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from logging import INFO

from . import fl_model_ref as ref
from .serverlib import N_CLASSES, GlobalEvaluator, RunLogs, fedavg, run_dir

app = ServerApp()


class DeterministicFedAvg(FedAvg):
    def aggregate_train(self, server_round, replies):
        valid, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid:
            return None, None
        valid = sorted(valid, key=lambda m: int(m.content["metrics"]["partition-id"]))
        res = [(m.content["arrays"].to_numpy_ndarrays(),
                int(m.content["metrics"]["num-examples"])) for m in valid]
        arrays = ArrayRecord(fedavg(res))
        metrics = self.train_metrics_aggr_fn([m.content for m in valid],
                                             self.weighted_by_key)
        return arrays, metrics


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = dict(context.run_config)
    min_nodes = int(cfg["min-nodes"])
    rounds = int(cfg["num-rounds"])
    delay = float(cfg.get("round-delay", 0.0))
    out = run_dir(cfg)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"{out} не порожній: оберіть інший results-dir")
    RunLogs(out)
    (out / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))

    initial = ref.init_weights(int(cfg["seed"]))
    evaluator = GlobalEvaluator(cfg, initial)

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        rec = evaluator(server_round, arrays.to_numpy_ndarrays())
        per_class = " ".join(f"{c}:{a:.2f}" for c, a in enumerate(rec["class_acc"]))
        log(INFO, "раунд %d: точність %.4f, по класах %s", server_round, rec["acc"], per_class)
        if delay > 0 and 0 < server_round < rounds:
            log(INFO, "пауза %.0f с після раунду %d", delay, server_round)
            time.sleep(delay)
        metrics = {"accuracy": rec["acc"], "loss": rec["loss"]}
        metrics.update({f"acc-class-{c}": a for c, a in enumerate(rec["class_acc"])})
        return MetricRecord(metrics)

    strategy = DeterministicFedAvg(
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_train_nodes=min_nodes,
        min_available_nodes=min_nodes,
        weighted_by_key="num-examples",
    )
    strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(initial),
        train_config=ConfigRecord({}),
        num_rounds=rounds,
        evaluate_fn=global_evaluate,
    )
