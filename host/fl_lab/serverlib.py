from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import fl_model_ref as ref

ROW_BYTES = ref.N_BYTES
N_CLASSES = 10


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


def class_accuracy(arrays: list[np.ndarray], x: np.ndarray, y: np.ndarray) -> list[float]:
    z, _, _ = ref.Model(arrays)._logits(x)
    pred = z.argmax(1)
    out = []
    for c in range(N_CLASSES):
        mask = y == c
        out.append(float((pred[mask] == c).mean()) if mask.any() else 0.0)
    return out


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
               "class_acc": class_accuracy(arrays, self.x, self.y),
               "bytes_per_client": 0 if server_round == 0 else 2 * ROW_BYTES,
               "round_s": 0.0 if server_round == 0 else now - self.t_prev,
               "weights_crc": f"{ref.crc32(ref.pack(arrays)):08x}"}
        self.logs.round(rec)
        (self.logs.dir / "final_weights.bin").write_bytes(ref.pack(arrays))
        self.prev = [a.copy() for a in arrays]
        self.t_prev = time.perf_counter()
        return rec


def fedavg(results: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    total = sum(n for _, n in results)
    return [sum(np.float32(n / total) * arr[i] for arr, n in results).astype(np.float32)
            for i in range(len(results[0][0]))]
