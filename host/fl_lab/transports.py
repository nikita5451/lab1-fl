from __future__ import annotations

import queue
import subprocess
import threading


class BoardUnavailable(RuntimeError):
    pass

class SerialTransport:
    def __init__(self, port: str, baud: int = 115200, timeout: float = 0.5) -> None:
        import serial  

        try:
            ser = serial.Serial()
            ser.port = port
            ser.baudrate = baud
            ser.timeout = timeout
            ser.dtr = True
            ser.rts = True
            ser.open()
        except Exception as exc: 
            raise BoardUnavailable(f"не вдалося відкрити {port}: {exc}") from exc
        self._ser = ser

    def write_line(self, text: str) -> None:
        self._ser.write(text.encode("ascii") + b"\n")
        self._ser.flush()

    def read_line(self, timeout: float) -> str | None:
        self._ser.timeout = timeout
        raw = self._ser.readline()
        if not raw:
            return None
        return raw.decode("ascii", errors="replace").rstrip("\r\n")

    def close(self) -> None:
        self._ser.close()


class ProcessTransport:

    def __init__(self, argv: list[str]) -> None:
        self._p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   text=True, bufsize=1)
        self._q: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self._p.stdout is not None
        for line in self._p.stdout:
            self._q.put(line.rstrip("\n"))
        self._q.put(None)

    def write_line(self, text: str) -> None:
        assert self._p.stdin is not None
        self._p.stdin.write(text + "\n")
        self._p.stdin.flush()

    def read_line(self, timeout: float) -> str | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        if self._p.stdin:
            self._p.stdin.close()
        self._p.wait(timeout=5)
