from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from . import fl_model_ref as ref
from .fedlib import GlobalEvaluator, RunLogs, client_round, fedavg, run_dir


def load_cfg(path: str, overrides: list[str]) -> dict:
    cfg = tomllib.loads(Path(path).read_text())
    cfg.setdefault("repo-root", str(Path(__file__).resolve().parents[2]))
    for kv in overrides:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        cfg[k] = v
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], help="key=value")
    ap.add_argument("--port")
    a = ap.parse_args()
    cfg = load_cfg(a.config, a.set)
    if a.port:
        cfg["board-port"] = a.port
    out = run_dir(cfg)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} не порожній: оберіть інший results-dir")
    RunLogs(out)
    (out / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))

    arrays = ref.init_weights(int(cfg["seed"]))
    ev = GlobalEvaluator(cfg, arrays)
    r0 = ev(0, arrays)
    print(f"раунд 0: acc={r0['acc']:.4f} crc={r0['weights_crc']}")
    for rnd in range(1, int(cfg["num-rounds"]) + 1):
        res = []
        for pid in range(int(cfg["num-clients"])):
            new, m = client_round(cfg, pid, rnd, arrays)
            res.append((new, m["n"]))                     # вага = num-examples
        arrays = fedavg(res)
        r = ev(rnd, arrays)
        print(f"раунд {rnd}: acc={r['acc']:.4f} |dw|={r['dw_server']:.3f}")


if __name__ == "__main__":
    main()
