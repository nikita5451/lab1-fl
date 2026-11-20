from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def main(dirs: list[str]) -> None:
    print("| Прогін | Вузлів (у раундах) | Загальна точність | Клас 2 | Клас 5 | Клас 8 | Байтів за прогін |")
    print("|-|-|-|-|-|-|-|")
    for d in dirs:
        rounds = sorted(jl(Path(d) / "rounds.jsonl"), key=lambda r: r["round"])
        clients = jl(Path(d) / "clients.jsonl")
        if not rounds:
            print(f"| {Path(d).name} | немає журналу | | | | | |")
            continue
        last = rounds[-1]
        per_round: dict[int, set[int]] = defaultdict(set)
        for c in clients:
            per_round[c["round"]].add(c["client"])
        counts = [len(per_round[r]) for r in sorted(per_round)]
        nodes = (f"{counts[0]}" if len(set(counts)) == 1 else f"{counts[0]} → {counts[-1]}") if counts else "0"
        sent = sum(c["rx_bytes"] + c["tx_bytes"] for c in clients)
        ca = last["class_acc"]
        print(f"| {Path(d).name} | {nodes} | {last['acc']:.4f} | {ca[2]:.2f} | {ca[5]:.2f} | {ca[8]:.2f} | {sent} |")
    print()
    for d in dirs:
        clients = jl(Path(d) / "clients.jsonl")
        per_round: dict[int, list[int]] = defaultdict(list)
        for c in clients:
            per_round[c["round"]].append(c["client"])
        print(d, {r: sorted(v) for r, v in sorted(per_round.items())})


if __name__ == "__main__":
    main(sys.argv[1:])
