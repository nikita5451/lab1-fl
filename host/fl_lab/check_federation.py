from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ALLOWED_CMDS = {"INFO", "CRC", "INIT", "SETW", "GETW", "FIT", "EVAL", "STATS",
                "RESETSTATS", "RESET", "W", "help", "info", "heap", "restart"}
ALLOWED_DEV_PREFIX = ("OK", "ERR", "EV", "W", "BOOT")
W_LINES_PER_XFER = 7
MODEL_BYTES = 4840


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def looks_like_sample(text: str) -> bool:
    nums = re.findall(r"(?<![\w.])\d{1,3}(?![\w.])", text)
    return len(nums) >= 64


def main(argv: list[str]) -> int:
    run = Path(argv[1] if len(argv) > 1 else ".")
    rounds, clients = jl(run / "rounds.jsonl"), jl(run / "clients.jsonl")
    if not rounds or not clients:
        print(f"порожні журнали у {run}")
        return 2
    glob = sorted((r for r in rounds), key=lambda r: r["round"])
    by_round: dict[int, list[dict]] = defaultdict(list)
    for c in clients:
        by_round[c["round"]].append(c)
    res: list[tuple[str, str, bool, str]] = []

    print("ГЛОБАЛЬНА МОДЕЛЬ (тест, якого не бачив жоден клієнт)")
    print(" раунд  точність   втрати  |dw| сервера  байтів/клієнт")
    for r in glob:
        print(f"{r['round']:6d} {r['acc']:9.4f} {r['loss']:8.4f} {r['dw_server']:12.4f} "
              f"{r['bytes_per_client']:14d}")
    print("\nКЛІЄНТИ (кожен навчався на своїх даних)")
    print(" клієнт  бекенд      n  train acc  val acc    |dw|   мс/раунд")
    per: dict[int, list[dict]] = defaultdict(list)
    for c in clients:
        per[c["client"]].append(c)
    for cid, rs in sorted(per.items()):
        m = lambda k: sum(x[k] for x in rs) / len(rs)  # noqa: E731
        print(f"{cid:7d} {rs[0]['backend']:>8} {rs[0]['n']:6d} {m('train_acc'):10.4f} "
              f"{m('val_acc'):8.4f} {m('dw'):8.4f} {m('ms'):10.1f}")

    # F1
    mins = min(len({c["client"] for c in cs}) for cs in by_round.values())
    res.append(("F1", "у кожному раунді >= 2 клієнтів", mins >= 2, f"мінімум {mins}"))
    # F2
    ns = sorted({rs[0]["n"] for rs in per.values()})
    accs = [sum(x["train_acc"] for x in rs) / len(rs) for rs in per.values()]
    ok2 = len(ns) >= 2 and (max(accs) - min(accs)) > 1e-3
    res.append(("F2", "дані клієнтів різні", ok2,
                f"n: {ns[0]}..{ns[-1]}, acc: {min(accs):.3f}..{max(accs):.3f}"))
    # F3
    ok3 = 0
    trained = [r for r in glob if r["round"] > 0]
    for r in trained:
        mx = max(c["dw"] for c in by_round[r["round"]])
        ok3 += r["dw_server"] < mx - 1e-9
    res.append(("F3", "сервер усереднив", ok3 == len(trained),
                f"у {ok3} з {len(trained)} раундів |dw|сервера < max|dw|клієнта"))
    # F4
    a0, aN = glob[0]["acc"], glob[-1]["acc"]
    res.append(("F4", "глобальна точність зросла", aN > a0, f"{a0:.4f} -> {aN:.4f}"))
    # F5
    board = [c for c in clients if c["backend"] == "serial"]
    if board:
        us = [c["us"] for c in board]
        res.append(("F5", "навчання виконувалось на платі", min(us) > 0,
                    f"{min(us)/1000:.1f}..{max(us)/1000:.1f} мс/раунд, {len(board)} раундів"))
    else:
        res.append(("F5", "навчання виконувалось на платі", False,
                    "у журналі немає клієнта з backend=serial (прогін без плати)"))
    # F6
    tr = run / "transcript.log"
    ok6, detail6 = True, "транскрипту немає (прогін без пристрою)"
    if tr.exists():
        n_getw = n_w = n_bad = 0
        bad_cmd: set[str] = set()
        for line in tr.read_text().splitlines():
            parts = line.split(None, 3)
            if len(parts) < 4:
                continue
            _, _, d, text = parts
            if d == ">":
                cmd = text.split()[0]
                if cmd.startswith("W["):      # шматок вагів хост -> пристрій
                    cmd = "W"
                n_getw += cmd == "GETW"
                if cmd not in ALLOWED_CMDS:
                    bad_cmd.add(cmd)
            elif d == "<":
                if text.startswith("W["):
                    n_w += 1
                elif not text.startswith(ALLOWED_DEV_PREFIX) and looks_like_sample(text):
                    n_bad += 1
        ok6 = not bad_cmd and n_bad == 0 and n_w == n_getw * W_LINES_PER_XFER
        detail6 = (f"рядків W від пристрою {n_w} = {n_getw} GETW x {W_LINES_PER_XFER}; "
                   f"невідомих команд {len(bad_cmd)}, рядків схожих на семпли {n_bad}")
    res.append(("F6", "семпли не йшли з пристрою на хост", ok6, detail6))

    print("\nПЕРЕВІРКИ ФЕДЕРАТИВНОСТІ")
    for fid, name, ok, detail in res:
        print(f"  [{'OK' if ok else 'FAIL'}] {fid} {name:<40} {detail}")
        
    bs = run / "board_stats.json"
    ok_stats = True
    if bs.exists():
        s = json.loads(bs.read_text())
        m = len(trained)
        exp = {"fits": m, "setw": m, "getw": m, "evals": m,
               "w_rx_bytes": m * MODEL_BYTES, "w_tx_bytes": m * MODEL_BYTES}
        print("\nЗВІРКА З ЛІЧИЛЬНИКАМИ ПЛАТИ")
        for k, v in exp.items():
            good = s.get(k) == v
            ok_stats &= good
            print(f"  {k:>11}: {s.get(k)!s:>6}  очікувано {v:<6} {'ok' if good else 'НЕ ЗБІГАЄТЬСЯ'}")
    return 0 if all(r[2] for r in res) and ok_stats else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
