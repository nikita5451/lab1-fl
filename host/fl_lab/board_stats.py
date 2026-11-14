from __future__ import annotations

import argparse
import json

from .protocol import BoardClient
from .transports import ProcessTransport, SerialTransport


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["reset", "read"])
    ap.add_argument("--port")
    ap.add_argument("--proc", help="argv симулятора через |")
    ap.add_argument("--out")
    a = ap.parse_args()
    tp = SerialTransport(a.port) if a.port else ProcessTransport(a.proc.split("|"))
    c = BoardClient(tp)
    if a.action == "reset":
        crc = c.reset()          
        print(f"пристрій у стані холодного старту, crc ваг = {crc:08x}")
    else:
        s = c.stats()
        print(json.dumps(s, indent=2))
        if a.out:
            open(a.out, "w").write(json.dumps(s, indent=2))
    c.close()


if __name__ == "__main__":
    main()
