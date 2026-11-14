from __future__ import annotations

import base64
import time
import zlib
from pathlib import Path

from . import fl_model_ref as ref

W_LINE_BYTES = 768            # 1024 символи base64 = 768 байтів


class ProtocolError(RuntimeError):
    pass


def parse_fields(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in text.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
        else:
            out[tok] = "1"
    return out


class BoardClient:
    def __init__(self, transport, transcript: Path | str | None = None,
                 timeout: float = 15.0, label: str = "board") -> None:
        self.t = transport
        self.timeout = timeout
        self.label = label
        self.events: list[str] = []         # "EV ..." рядки пристрою
        self.skipped: list[str] = []        # сторонні рядки (логи прошивки)
        self.bytes_tx = 0                   # усі байти хост -> пристрій
        self.bytes_rx = 0                   # усі байти пристрій -> хост
        self.w_tx = 0                       # байти ВАГІВ хост -> пристрій
        self.w_rx = 0                       # байти ВАГІВ пристрій -> хост
        self._tr = open(transcript, "a", buffering=1) if transcript else None
        self._t0 = time.monotonic()

    def _log(self, direction: str, text: str) -> None:
        if self._tr:
            shown = text if not text.startswith("W ") else f"W[{len(text) - 2}] {text[2:18]}..."
            self._tr.write(f"{time.monotonic() - self._t0:9.3f} {self.label} {direction} {shown}\n")

    def _send(self, line: str) -> None:
        self.t.write_line(line)
        self.bytes_tx += len(line) + 1
        self._log(">", line)

    def _recv(self, timeout: float | None = None) -> str | None:
        line = self.t.read_line(self.timeout if timeout is None else timeout)
        if line is None:
            return None
        self.bytes_rx += len(line) + 1
        self._log("<", line)
        return line

    def _read_final(self, what: str) -> dict[str, str]:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            line = self._recv(0.5)
            if line is None:
                continue
            if line.startswith("OK "):
                body = line[3:]
                cmd, _, rest = body.partition(" ")
                f = parse_fields(rest)
                f["_cmd"] = cmd
                return f
            if line.startswith("ERR"):
                raise ProtocolError(f"{what}: {line}")
            if line.startswith("EV "):
                self.events.append(line)
            else:
                self.skipped.append(line)       # логи/шум: пропускаємо
        raise ProtocolError(f"{what}: таймаут {self.timeout}s")

    def _read_w_lines(self, n: int) -> bytes:
        chunks: list[str] = []
        deadline = time.monotonic() + self.timeout
        while len(chunks) < n and time.monotonic() < deadline:
            line = self._recv(0.5)
            if line is None:
                continue
            if line.startswith("W "):
                chunks.append(line[2:])
            elif line.startswith("EV "):
                self.events.append(line)
            else:
                self.skipped.append(line)
        if len(chunks) != n:
            raise ProtocolError(f"GETW: очікувано {n} рядків, прийшло {len(chunks)}")
        return b"".join(base64.b64decode(c) for c in chunks)

    def info(self) -> dict[str, str]:
        self._send("INFO")
        return self._read_final("INFO")

    def crc(self) -> int:
        self._send("CRC")
        return int(self._read_final("CRC")["crc"], 16)

    def init(self, seed: int) -> int:
        self._send(f"INIT {seed}")
        return int(self._read_final("INIT")["crc"], 16)

    def setw(self, blob: bytes) -> int:
        crc = zlib.crc32(blob) & 0xFFFFFFFF
        self._send(f"SETW {len(blob)} {crc:08x}")
        self._read_final("SETW ready")
        for off in range(0, len(blob), W_LINE_BYTES):
            self._send("W " + base64.b64encode(blob[off:off + W_LINE_BYTES]).decode())
        self.w_tx += len(blob)
        f = self._read_final("SETW")
        if int(f["crc"], 16) != crc:
            raise ProtocolError("SETW: crc на пристрої інший")
        return crc

    def getw(self) -> bytes:
        self._send("GETW")
        f = self._read_final("GETW")
        blob = self._read_w_lines(int(f["lines"]))
        self.w_rx += len(blob)
        want = int(f["crc"], 16)
        if (zlib.crc32(blob) & 0xFFFFFFFF) != want:
            raise ProtocolError("GETW: crc не збігається (FL-05)")
        return blob

    def fit(self, epochs: int, lr: float, batch: int = 16, seed: int = 1,
            clip: float = 0.0) -> dict[str, float]:
        self._send(f"FIT {epochs} {lr} {batch} {seed} {clip}")
        f = self._read_final("FIT")
        return {k: float(v) for k, v in f.items() if k != "_cmd"}

    def eval(self) -> dict[str, float]:
        self._send("EVAL")
        f = self._read_final("EVAL")
        return {k: float(v) for k, v in f.items() if k not in ("_cmd", "split")}

    def stats(self) -> dict[str, int]:
        self._send("STATS")
        f = self._read_final("STATS")
        return {k: int(v) for k, v in f.items() if k != "_cmd"}

    def reset(self) -> int:
        """Холодний старт пристрою (RESET): той самий стан перед кожним прогоном."""
        self._send("RESET")
        return int(self._read_final("RESET")["crc"], 16)

    def reset_stats(self) -> None:
        self._send("RESETSTATS")
        self._read_final("RESETSTATS")

    def close(self) -> None:
        if self._tr:
            self._tr.close()
        self.t.close()

    def train_round(self, arrays, epochs: int, lr: float, batch: int, seed: int,
                    clip: float = 0.0) -> tuple[list, dict]:
        """INFO -> SETW -> FIT -> GETW -> EVAL; повертає нові ваги й метрики."""
        info = self.info()
        blob_in = ref.pack(arrays)
        self.setw(blob_in)
        fit = self.fit(epochs, lr, batch, seed, clip)
        new = ref.unpack(self.getw())
        ev = self.eval()
        dw_host = ref.delta_norm(new, arrays)           # порахована на ХОСТІ
        metrics = {
            "n": int(fit["n"]), "n_total": int(info["n"]),
            "train_loss": fit["loss"], "train_acc": fit["acc"],
            "val_loss": ev["loss"], "val_acc": ev["acc"],
            "dw": dw_host, "dw_dev": fit["dw"],
            "ms": fit["ms"], "us": fit["us"],
            "eval_ms": ev["ms"],
            "rx_bytes": len(blob_in), "tx_bytes": len(ref.pack(new)),
        }
        return new, metrics
