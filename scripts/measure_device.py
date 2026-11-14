from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "host"))
from fl_lab.protocol import BoardClient, ProtocolError  
from fl_lab.transports import ProcessTransport, SerialTransport  

REPEATS = 5


def stat(xs):
    return f"{min(xs):.1f} / {st.mean(xs):.1f} / {max(xs):.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--proc", help="argv симулятора через | (лише для перевірки скрипта)")
    a = ap.parse_args()
    c = BoardClient(SerialTransport(a.port) if a.port else ProcessTransport(a.proc.split("|")))

    for diag in ("info", "heap"):
        try:
            c._send(diag)
            print(diag, "->", c._read_final(diag))
        except ProtocolError as e:
            print(diag, "-> недоступно:", e)

    i = c.info()
    n_tr = int(i["train"])
    print(f"\nпараметрів={i['params']} байтів={i['bytes']} семплів={i['n']} (train={i['train']}, val={i['val']})")
    print(f"розмір даних, байтів: {20 + int(i['n']) * 65}")

    c.init(1)
    ep_ms, ev_ms = [], []
    for _ in range(REPEATS):
        c.init(1)                                   
        r = c.fit(10, 0.3, 16, 1)                   
        ep_ms.append(r["us"] / 1000.0 / 10.0)
        ev_ms.append(c.eval()["us"] / 1000.0)
    print(f"\nчас ОДНІЄЇ епохи на {n_tr} семплах, мс (мін/сер/макс): {stat(ep_ms)}")
    print(f"час оцінки на валідації, мс (мін/сер/макс): {stat(ev_ms)}")
    print("\nПісля вимірювань ще раз виконайте `heap` у моніторі: запас стека має впасти до свого мінімуму.")
    c.close()


if __name__ == "__main__":
    main()
