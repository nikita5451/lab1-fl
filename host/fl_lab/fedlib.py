from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import fl_model_ref as ref
from .protocol import BoardClient
from .transports import BoardUnavailable, ProcessTransport, SerialTransport

ROW_BYTES = ref.N_BYTES          

class RunLogs:
    def __init__(self, root: Path | str) -> None:
        self.dir = Path(root)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _append(self, name: str, rec: dict) -> None:
        with open(self.dir / name, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    def client(self, rec: dict) -> None:
        self._append("clients.jsonl", rec)

    def round(self, rec: dict) -> None:
        self._append("rounds.jsonl", rec)

    @property
    def transcript(self) -> Path:
        return self.dir / "transcript.log"


def run_dir(cfg: dict) -> Path:
    return Path(cfg["repo-root"]) / cfg["results-dir"]

def load_shard(root: str | Path, pid: int):
    blob = (Path(root) / "data" / f"data_{pid}.bin").read_bytes()
    return ref.parse_data_file(blob)


def split_like_device(n: int, seed: int = 12345) -> tuple[np.ndarray, np.ndarray]:
    order = np.arange(n, dtype=np.int64)
    ref.Rng(seed).shuffle(order)
    n_val = n // 5
    return order[: n - n_val], order[n - n_val:]

class VirtualClient:
    backend = "csim"

    def __init__(self, root, pid: int) -> None:
        x, y = load_shard(root, pid)
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

class GlobalEvaluator:
    def __init__(self, cfg: dict, initial: list[np.ndarray]) -> None:
        x, y = ref.parse_data_file((Path(cfg["repo-root"]) / "data" / "test.bin").read_bytes())
        self.x, self.y = ref.normalize(x), y.astype(np.int64)
        self.logs = RunLogs(run_dir(cfg))
        self.prev = [a.copy() for a in initial]
        self.t_prev = time.perf_counter()
        self.n_clients = int(cfg.get("num-clients", 0))

    def __call__(self, server_round: int, arrays: list[np.ndarray]) -> dict:
        now = time.perf_counter()
        loss, acc = ref.Model(arrays).evaluate(self.x, self.y)
        dw = ref.delta_norm(arrays, self.prev)
        rec = {"round": server_round, "acc": acc, "loss": loss, "dw_server": dw,
               "bytes_per_client": 0 if server_round == 0 else 2 * ROW_BYTES,
               "round_s": 0.0 if server_round == 0 else now - self.t_prev,
               "weights_crc": f"{ref.crc32(ref.pack(arrays)):08x}"}
        self.logs.round(rec)
        self.prev = [a.copy() for a in arrays]
        self.t_prev = time.perf_counter()
        return rec


def fedavg(results: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    total = sum(n for _, n in results)
    return [sum(np.float32(n / total) * arr[i] for arr, n in results).astype(np.float32)
            for i in range(len(results[0][0]))]
