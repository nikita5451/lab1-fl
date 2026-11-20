from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import fl_model_ref as ref
from .protocol import BoardClient
from .serverlib import GlobalEvaluator, RunLogs, class_accuracy, fedavg, run_dir
from .transports import BoardUnavailable, ProcessTransport, SerialTransport

ROW_BYTES = ref.N_BYTES

def load_shard(root: str | Path, pid: int):
    blob = (Path(root) / "data" / f"data_{pid}.bin").read_bytes()
    return ref.parse_data_file(blob)


def load_path(path: str | Path):
    return ref.parse_data_file(Path(path).read_bytes())


def split_like_device(n: int, seed: int = 12345) -> tuple[np.ndarray, np.ndarray]:
    order = np.arange(n, dtype=np.int64)
    ref.Rng(seed).shuffle(order)
    n_val = n // 5
    return order[: n - n_val], order[n - n_val:]

class VirtualClient:
    backend = "csim"

    def __init__(self, root, pid: int | None = None, path: str | Path | None = None) -> None:
        x, y = load_path(path) if path is not None else load_shard(root, pid)
        tr, va = split_like_device(len(y))
        self.xt, self.yt = ref.normalize(x[tr]), y[tr].astype(np.int64)
        self.xv, self.yv = ref.normalize(x[va]), y[va].astype(np.int64)
        self.n_total = len(y)

    def train(self, arrays, epochs, lr, batch, seed, clip=0.0):
        m = ref.Model(arrays)
        t0 = time.perf_counter()
        r = m.fit(self.xt, self.yt, epochs, lr, batch, seed)
        us = (time.perf_counter() - t0) * 1e6
        new = m.get()
        scale = 1.0
        dw = ref.delta_norm(new, arrays)
        if clip > 0 and dw > clip:
            scale = clip / dw
            new = [(g + (a - g) * np.float32(scale)).astype(np.float32)
                   for a, g in zip(new, arrays)]
            dw = ref.delta_norm(new, arrays)
        vl, va = ref.Model(new).evaluate(self.xv, self.yv)
        return new, {
            "n": len(self.yt), "n_total": self.n_total,
            "train_loss": r["loss"], "train_acc": r["acc"],
            "val_loss": vl, "val_acc": va, "dw": dw, "dw_dev": dw,
            "ms": us / 1000.0, "us": us, "eval_ms": 0.0,
            "rx_bytes": ROW_BYTES, "tx_bytes": ROW_BYTES, "clip": scale,
        }


class BoardNode:

    def __init__(self, backend: str, target: str, transcript: Path) -> None:
        self.backend = {"serial": "serial", "proc": "csim-c"}[backend]
        self.backend_kind, self.target, self.transcript = backend, target, transcript

    def _open(self) -> BoardClient:
        if self.backend_kind == "serial":
            if not self.target:
                raise BoardUnavailable("не задано board-port: плата обов'язкова")
            tp = SerialTransport(self.target)
        else:
            tp = ProcessTransport(self.target.split("|"))
        return BoardClient(tp, transcript=self.transcript, label=self.backend)

    def train(self, arrays, epochs, lr, batch, seed, clip=0.0):
        c = self._open()
        try:
            new, m = c.train_round(arrays, epochs, lr, batch, seed, clip)
        finally:
            c.close()
        if abs(m["dw"] - m["dw_dev"]) > 1e-3 * max(1.0, m["dw"]):
            raise RuntimeError(f"|dw| хоста {m['dw']} != пристрою {m['dw_dev']}")
        return new, m


def make_client(cfg: dict, pid: int):
    bp = int(cfg.get("board-partition", -1))
    root = cfg["repo-root"]
    if pid == bp:
        backend = cfg.get("board-backend", "serial")
        target = cfg.get("board-port", "") if backend == "serial" else cfg["board-proc"]
        return BoardNode(backend, target, RunLogs(run_dir(cfg)).transcript)
    return VirtualClient(root, pid)


def client_round(cfg: dict, pid: int, server_round: int, arrays):
    """Один раунд клієнта + запис у журнал по клієнтах."""
    client = make_client(cfg, pid)
    seed = int(cfg["seed"]) + 1000 * server_round + 100 * pid
    new, m = client.train(arrays, int(cfg["local-epochs"]), float(cfg["lr"]),
                          int(cfg["batch"]), seed, float(cfg.get("clip-norm", 0.0)))
    RunLogs(run_dir(cfg)).client({"round": server_round, "client": pid,
                                  "backend": client.backend, **m})
    return new, m



def resolve_path(p: str, root: str | Path) -> Path:
    q = Path(p)
    return q if q.is_absolute() else Path(root) / q


def make_backend(node: dict, run: dict):
    kind = str(node["backend"])
    if kind == "serial":
        port = str(node.get("device-port", ""))
        if not port:
            raise BoardUnavailable("backend=serial потребує device-port")
        return BoardNode("serial", port, RunLogs(run_dir(run)).transcript)
    if kind == "sim":
        dp = node.get("data-path")
        if not dp:
            raise ValueError("backend=sim потребує data-path")
        return VirtualClient(run["repo-root"], path=resolve_path(str(dp), run["repo-root"]))
    raise ValueError(f"невідомий backend: {kind}")


def client_round_node(run: dict, node: dict, server_round: int, arrays):
    pid = int(node["partition-id"])
    backend = make_backend(node, run)
    seed = int(run["seed"]) + 1000 * server_round + 100 * pid
    new, m = backend.train(arrays, int(run["local-epochs"]), float(run["lr"]),
                           int(run["batch"]), seed, float(run.get("clip-norm", 0.0)))
    RunLogs(run_dir(run)).client({"round": server_round, "client": pid,
                                  "backend": backend.backend, **m})
    return new, m
