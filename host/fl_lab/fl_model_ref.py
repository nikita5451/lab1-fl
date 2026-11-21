"""fl_model_ref — хостове дзеркало моделі з device/.  НАДАЄТЬСЯ.

Потрібне у двох місцях Flower-обгортки:

* серверна оцінка агрегованої моделі на глобальному тестовому наборі
  (вимога FL-06): сервер не має плати і рахує сам;
* віртуальний клієнт, щоб показати федерацію з N вузлів, маючи одну плату.

Дзеркало узгоджене з C-кодом побітово: той самий генератор випадкових
чисел, та сама розкладка параметрів, та сама математика.  Перевірка —
`python fl_model_ref.py`.

Зміна математики тут вимагає такої ж зміни в C (і навпаки): розбіжність
між двома реалізаціями не призводить до помилки, лише до нижчої точності
(FL-03).
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

N_IN, N_HIDDEN, N_OUT = 64, 16, 10

#: Форми параметрів у порядку розкладки: W1, b1, W2, b2.
SHAPES = [(N_HIDDEN, N_IN), (N_HIDDEN,), (N_OUT, N_HIDDEN), (N_OUT,)]
N_PARAMS = sum(int(np.prod(s)) for s in SHAPES)      # 1210
N_BYTES = N_PARAMS * 4                               # 4840

U32 = 0xFFFFFFFF


# --------------------------------------------------------------- RNG
# Дзеркало device/fl_rng.c

def rng_seed(seed: int) -> int:
    z = (int(seed) + 0x9E3779B9) & U32
    z ^= z >> 16
    z = (z * 0x21F0AAAD) & U32
    z ^= z >> 15
    z = (z * 0x735A2D97) & U32
    z ^= z >> 15
    return z if z else 0x6D2B79F5


class Rng:
    __slots__ = ("s",)

    def __init__(self, seed: int) -> None:
        self.s = rng_seed(seed)

    def next_u32(self) -> int:
        x = self.s
        x = (x ^ (x << 13)) & U32
        x ^= x >> 17
        x = (x ^ (x << 5)) & U32
        self.s = x
        return x

    def uniform(self) -> np.float32:
        return np.float32(self.next_u32() >> 8) * np.float32(1.0 / 16777216.0)

    def shuffle(self, a: np.ndarray, n: int | None = None) -> None:
        i = len(a) if n is None else n
        while i > 1:
            j = self.next_u32() % i
            a[i - 1], a[j] = a[j], a[i - 1]
            i -= 1

# ------------------------------------------------------- параметри

def init_weights(seed: int) -> list[np.ndarray]:
    """Дзеркало fl_model_init_weights(): Glorot uniform, зсуви в нуль."""
    rng = Rng(seed)
    two, one = np.float32(2.0), np.float32(1.0)

    def draw(n: int, lim: np.float32) -> np.ndarray:
        return np.array([(two * rng.uniform() - one) * lim for _ in range(n)],
                        dtype=np.float32)

    lim1 = np.float32(np.sqrt(np.float32(6.0) / np.float32(N_IN + N_HIDDEN)))
    w1 = draw(N_HIDDEN * N_IN, lim1).reshape(N_HIDDEN, N_IN)
    lim2 = np.float32(np.sqrt(np.float32(6.0) / np.float32(N_HIDDEN + N_OUT)))
    w2 = draw(N_OUT * N_HIDDEN, lim2).reshape(N_OUT, N_HIDDEN)
    return [w1, np.zeros(N_HIDDEN, np.float32), w2, np.zeros(N_OUT, np.float32)]


def pack(arrays: list[np.ndarray]) -> bytes:
    """Параметри -> плоский float32 little-endian, як на дроті."""
    out = bytearray()
    for a in arrays:
        out += np.ascontiguousarray(a, dtype="<f4").tobytes()
    return bytes(out)


def unpack(blob: bytes) -> list[np.ndarray]:
    arrays, off = [], 0
    for shape in SHAPES:
        n = int(np.prod(shape))
        arrays.append(np.frombuffer(blob, "<f4", n, off).reshape(shape).copy())
        off += n * 4
    if off != len(blob):
        raise ValueError(f"очікувано {off} B, отримано {len(blob)}")
    return arrays


def crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


# ---------------------------------------------------------- модель

def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z, dtype=np.float32)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


class Model:
    """Дзеркало fl_model.c + fl_train.c, але батчами (швидше на хості)."""

    def __init__(self, params: list[np.ndarray] | None = None, seed: int = 1):
        self.p = [a.astype(np.float32).copy()
                  for a in (params if params is not None else init_weights(seed))]

    def get(self) -> list[np.ndarray]:
        return [a.copy() for a in self.p]

    def set(self, params: list[np.ndarray]) -> None:
        for i, (dst, src) in enumerate(zip(self.p, params)):
            if dst.shape != src.shape:
                raise ValueError(f"параметр {i}: очікувано {dst.shape}, "
                                 f"отримано {src.shape}")
        self.p = [a.astype(np.float32).copy() for a in params]

    def _logits(self, x: np.ndarray):
        w1, b1, w2, b2 = self.p
        hpre = (x.astype(np.float32) @ w1.T + b1).astype(np.float32)
        h = np.maximum(hpre, np.float32(0.0))
        return (h @ w2.T + b2).astype(np.float32), h, hpre

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Середня перехресна ентропія і точність."""
        if len(x) == 0:
            return 0.0, 0.0
        z, _, _ = self._logits(x)
        p = _softmax(z)
        picked = np.clip(p[np.arange(len(y)), y], 1e-12, None)
        return float(-np.log(picked).mean()), float((z.argmax(1) == y).mean())

    def _sgd_step(self, x: np.ndarray, y: np.ndarray, lr: float) -> None:
        w1, _, w2, _ = self.p
        z, h, hpre = self._logits(x)
        dz = _softmax(z)
        dz[np.arange(len(y)), y] -= np.float32(1.0)

        gw2 = (dz.T @ h).astype(np.float32)
        gb2 = dz.sum(0, dtype=np.float32)
        dh = np.where(hpre > 0, (dz @ w2).astype(np.float32),
                      np.float32(0.0)).astype(np.float32)
        gw1 = (dh.T @ x.astype(np.float32)).astype(np.float32)
        gb1 = dh.sum(0, dtype=np.float32)

        scale = np.float32(lr / max(len(x), 1))
        for i, g in enumerate((gw1, gb1, gw2, gb2)):
            self.p[i] = (self.p[i] - scale * g).astype(np.float32)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int, lr: float,
            batch: int, seed: int = 1) -> dict[str, float]:
        """Дзеркало fl_train(): перемішує тим самим генератором, що C."""
        order = np.arange(len(y), dtype=np.int64)
        loss_sum = correct = seen = steps = 0
        for e in range(max(1, epochs)):
            Rng(seed + e).shuffle(order)
            loss_sum, correct, seen = 0.0, 0, 0
            for off in range(0, len(y), batch):
                idx = order[off:off + batch]
                bx, by = x[idx], y[idx]
                z, _, _ = self._logits(bx)
                p = _softmax(z)
                picked = np.clip(p[np.arange(len(by)), by], 1e-12, None)
                loss_sum += float(-np.log(picked).sum())
                correct += int((z.argmax(1) == by).sum())
                seen += len(by)
                self._sgd_step(bx, by, lr)
                steps += 1
        return {"loss": loss_sum / max(seen, 1), "acc": correct / max(seen, 1),
                "n": float(seen), "steps": float(steps)}


def delta_norm(new: list[np.ndarray], old: list[np.ndarray]) -> float:
    """||new - old||_2 по всьому плоскому масиву, як fl_model_delta_norm."""
    return float(np.sqrt(sum(float(((a - b).astype(np.float32) ** 2).sum())
                             for a, b in zip(new, old))))
                             
                        
# --------------------------------------------- формат файлу даних
# Той самий формат, що читає пристрій (п. 5.1 методички).

DS_MAGIC, DS_VERSION, DS_HDR, DS_FLAG_U8 = 0x53444C46, 1, 20, 0x0001


def build_data_file(x: np.ndarray, y: np.ndarray, n_classes: int = N_OUT) -> bytes:
    """uint8-ознаки (n, 64) + метки -> файл, який прошивається на пристрій."""
    x = np.ascontiguousarray(x, dtype=np.uint8)
    y = np.ascontiguousarray(y, dtype=np.uint8)
    n, n_feat = x.shape
    payload = np.concatenate([x, y[:, None]], axis=1).tobytes()
    head = struct.pack("<IHHHHII", DS_MAGIC, DS_VERSION, n_feat, n_classes,
                       DS_FLAG_U8, n, crc32(payload))
    return head + payload


def parse_data_file(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    magic, ver, n_feat, _, _, n, crc = struct.unpack("<IHHHHII", blob[:DS_HDR])
    if magic != DS_MAGIC or ver != DS_VERSION:
        raise ValueError("не той формат файлу даних")
    end = DS_HDR + n * (n_feat + 1)
    payload = blob[DS_HDR:end]
    if crc32(payload) != crc:
        raise ValueError("crc не збігається")
    raw = np.frombuffer(payload, np.uint8).reshape(n, n_feat + 1)
    return raw[:, :n_feat].copy(), raw[:, n_feat].copy()


def normalize(x: np.ndarray) -> np.ndarray:
    """Та сама нормалізація, що робить пристрій. Частина контракту даних."""
    return x.astype(np.float32) / np.float32(255.0)

# --------------------------------------------------------- самоперевірка

def _selftest() -> int:
    fails = 0
    print(f"params        : {N_PARAMS} (очікувано 1210)")
    print(f"байтів/напрям : {N_BYTES} (очікувано 4840)")
    if N_PARAMS != 1210:
        fails += 1

    for seed, want in ((1, 0xF8692CAC), (7, 0xC40A1C43)):
        got = crc32(pack(init_weights(seed)))
        ok = got == want
        print(f"init seed={seed}    : crc={got:08x} (очікувано {want:08x}) "
              f"{'ok' if ok else 'FAIL'}")
        fails += not ok
        
    # той самий синтетичний набір, що в device/selftest.c    
    xs, ys = [], []
    for i in range(256):
        cls = i % 10
        row = np.zeros(64, np.uint8)
        for k in range(6):
            row[(cls * 6 + k) % 64] = 200
        xs.append(row)
        ys.append(cls)
    x = normalize(np.array(xs))
    y = np.array(ys, dtype=np.int64)

    m = Model(init_weights(1))
    first = m.fit(x[:204], y[:204], 1, 0.3, 16, 1)
    later = m.fit(x[:204], y[:204], 40, 0.3, 16, 1)
    _, acc = m.evaluate(x[204:], y[204:])
    print(f"навчання      : 1 епоха acc={first['acc']:.3f} -> "
          f"41 епоха acc={later['acc']:.3f}, val acc={acc:.3f}")
    if not (later["acc"] > first["acc"] and later["acc"] > 0.9):
        print("FAIL: навчання не сходиться")
        fails += 1

    blob = build_data_file(np.array(xs), np.array(ys, np.uint8))
    bx, by = parse_data_file(blob)
    same = np.array_equal(bx, np.array(xs)) and np.array_equal(by, np.array(ys))
    print(f"файл даних    : {len(blob)} B, зворотний розбір "
          f"{'ok' if same else 'FAIL'}")
    fails += not same

    print("\n" + ("ЄСТЬ ПОМИЛКИ" if fails else "усі перевірки пройдено"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
