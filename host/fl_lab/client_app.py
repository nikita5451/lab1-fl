from __future__ import annotations

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from .fedlib import client_round_node

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    run = dict(context.run_config)
    node = dict(context.node_config)
    arrays = msg.content["arrays"].to_numpy_ndarrays()
    server_round = int(msg.content["config"].get("server-round", 0))

    new, m = client_round_node(run, node, server_round, arrays)

    metrics = MetricRecord({
        "num-examples": int(m["n"]),
        "partition-id": int(node["partition-id"]),
        "train_acc": float(m["train_acc"]),
        "val_acc": float(m["val_acc"]),
        "dw": float(m["dw"]),
    })
    return Message(content=RecordDict({"arrays": ArrayRecord(new), "metrics": metrics}),
                   reply_to=msg)
