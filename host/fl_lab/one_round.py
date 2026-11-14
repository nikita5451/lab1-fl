from __future__ import annotations

import argparse

from . import fl_model_ref as ref
from .protocol import BoardClient
from .transports import ProcessTransport, SerialTransport


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", choices=["serial", "proc"], default="serial")
    ap.add_argument("--port")
    ap.add_argument("--proc", help="argv симулятора через |")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    tp = SerialTransport(a.port) if a.transport == "serial" else ProcessTransport(a.proc.split("|"))
    c = BoardClient(tp)
    crc = c.init(1)
    print(f"init seed=1 crc={crc:08x} {'збігається з еталоном' if crc == 0xF8692CAC else 'НЕ ЗБІГАЄТЬСЯ (FL-03)!'}")
    w0 = ref.init_weights(1)
    new, m = c.train_round(w0, a.epochs, a.lr, a.batch, a.seed)
    print(f"передано ваг: {m['rx_bytes']} Б у пристрій, {m['tx_bytes']} Б назад")
    print(f"час навчання на пристрої: {m['ms']:.0f} мс ({a.epochs} епох, n={m['n']})")
    print(f"train acc={m['train_acc']:.3f}  val acc={m['val_acc']:.3f}")
    print(f"|dw| хост={m['dw']:.4f}  пристрій={m['dw_dev']:.4f}")
    print(f"лічильники пристрою: {c.stats()}")
    c.close()


if __name__ == "__main__":
    main()
