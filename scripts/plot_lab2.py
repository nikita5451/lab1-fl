from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rows(d: str) -> list[dict]:
    r = [json.loads(x) for x in (Path(d) / "rounds.jsonl").read_text().splitlines() if x.strip()]
    return sorted(r, key=lambda x: x["round"])


def main(out_dir: str, a: str, b: str, g: str) -> None:
    out = Path(out_dir)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    styles = ((a, "А: плата сама", "o", "-", 6), (b, "Б: федерація", "o", "-", 8), (g, "Г: плату від'єднано", "s", "--", 5))
    for d, label, mk, ls, ms in styles:
        r = rows(d)
        ax.plot([x["round"] for x in r], [x["acc"] for x in r], marker=mk, linestyle=ls, markersize=ms, label=label)
    ax.axvline(3, color="gray", linestyle=":")
    ax.text(3.1, 0.12, "кабель вийнято\nпісля раунду 3", fontsize=8, color="gray")
    ax.set_xlabel("раунд")
    ax.set_ylabel("точність на test.bin (сервер)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "lab2-accuracy.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ra, rb = rows(a)[-1]["class_acc"], rows(b)[-1]["class_acc"]
    xs = range(10)
    ax.bar([i - 0.2 for i in xs], ra, width=0.4, label="А: плата сама")
    ax.bar([i + 0.2 for i in xs], rb, width=0.4, label="Б: федерація")
    ax.set_xticks(list(xs))
    ax.set_xlabel("клас (цифра)")
    ax.set_ylabel("точність на test.bin")
    ax.set_ylim(0, 1.2)
    ax.grid(alpha=.3, axis="y")
    ax.legend(loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(out / "lab2-classes.png", dpi=150)
    print("збережено", out / "lab2-accuracy.png", out / "lab2-classes.png")


if __name__ == "__main__":
    main(*sys.argv[1:5])
