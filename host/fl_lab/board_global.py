from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import fl_model_ref as ref
from .protocol import BoardClient
from .serverlib import class_accuracy
from .transports import ProcessTransport, SerialTransport


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--proc", help="argv симулятора через |")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--test", default="data/test.bin")
    ap.add_argument("--out")
    a = ap.parse_args()

    blob = Path(a.weights).read_bytes()
    tp = SerialTransport(a.port) if a.port else ProcessTransport(a.proc.split("|"))
    c = BoardClient(tp)
    info = c.info()
    c.setw(blob)
    ev = c.eval()
    back = c.getw()
    c.close()

    x, y = ref.parse_data_file(Path(a.test).read_bytes())
    arrays = ref.unpack(back)
    loss, acc = ref.Model(arrays).evaluate(ref.normalize(x), y.astype("int64"))
    res = {
        "weights": a.weights,
        "device_n": int(info["n"]),
        "roundtrip_identical": back == blob,
        "crc": f"{ref.crc32(back):08x}",
        "device_eval_local_val_acc": ev["acc"],
        "device_eval_local_val_loss": ev["loss"],
        "test_acc_of_device_weights": acc,
        "test_class_acc_of_device_weights": class_accuracy(arrays, ref.normalize(x), y.astype("int64")),
    }
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
