from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  


def main(dirs: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for d in dirs:
        rows = [json.loads(x) for x in (Path(d) / "rounds.jsonl").read_text().splitlines()]
        rows.sort(key=lambda r: r["round"])
        print(d)
        for r in rows:
            print(f"  раунд {r['round']:>2}  acc={r['acc']:.4f}  loss={r['loss']:.4f}")
        ax.plot([r["round"] for r in rows], [r["acc"] for r in rows], marker="o",
                label=Path(d).name)
    ax.set_xlabel("раунд")
    ax.set_ylabel("точність на test.bin (сервер)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    out = Path(dirs[0]) / "accuracy.png"
    fig.savefig(out, dpi=150)
    print("збережено", out)


if __name__ == "__main__":
    main(sys.argv[1:])
