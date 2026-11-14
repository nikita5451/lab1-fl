from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from fl_lab import fl_model_ref as ref
from fl_lab.protocol import BoardClient, ProtocolError
from fl_lab.transports import ProcessTransport

ROOT = Path(__file__).resolve().parents[3]
SIM = ROOT / "firmware" / "host_test" / "proto_sim"


def main() -> int:
    c = BoardClient(ProcessTransport([str(SIM), str(ROOT / "data" / "data_0.bin")]),
                    transcript=ROOT / "results" / "sim_test_transcript.log")
    info = c.info()
    assert info["n"] == "152" and info["bytes"] == "4840", info
    
    for seed, want in ((1, 0xF8692CAC), (7, 0xC40A1C43)):
        got = c.init(seed)
        assert got == want, (seed, hex(got))
        assert ref.crc32(ref.pack(ref.init_weights(seed))) == want

    w0 = ref.init_weights(1)
    c.setw(ref.pack(w0))
    back = ref.unpack(c.getw())
    assert all(np.array_equal(a, b) for a, b in zip(w0, back))

    new, m = c.train_round(w0, epochs=3, lr=0.3, batch=16, seed=1)
    assert abs(m["dw"] - m["dw_dev"]) < 1e-3 * max(1.0, m["dw"]), m
    assert m["n"] == 122 and m["rx_bytes"] == m["tx_bytes"] == 4840

    assert m["n"] + 30 == m["n_total"]

    st = c.stats()
    assert st["fits"] == 1 and st["w_rx_bytes"] == 2 * 4840 and st["w_tx_bytes"] == 2 * 4840, st

    c._send("SETW 4840 deadbeef")
    c._read_final("ready")
    import base64
    zeros = bytes(4840)
    for off in range(0, 4840, 768):
        c._send("W " + base64.b64encode(zeros[off:off + 768]).decode())
    try:
        c._read_final("SETW")
        raise AssertionError("очікувалась помилка crc")
    except ProtocolError as e:
        assert "crc" in str(e)

    c._send("GETSAMPLE 0")
    try:
        c._read_final("GETSAMPLE")
        raise AssertionError("команда читання семплів існує!")
    except ProtocolError as e:
        assert "unknown_command" in str(e)
    c.close()
    print("test_protocol_sim: OK", {k: round(v, 4) if isinstance(v, float) else v
                                    for k, v in m.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
