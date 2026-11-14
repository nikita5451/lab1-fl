#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fl_model_ref import build_data_file, parse_data_file  # noqa: E402


def split_non_iid(y: np.ndarray, n_clients: int, alpha: float, seed: int,
                  min_per_client: int = 40) -> list[np.ndarray]:

    rng = np.random.default_rng(seed)
    for _ in range(200):
        buckets: list[list[int]] = [[] for _ in range(n_clients)]
        for c in range(int(y.max()) + 1):
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            cuts = (np.cumsum(rng.dirichlet(np.repeat(alpha, n_clients)))
                    * len(idx)).astype(int)[:-1]
            for b, part in zip(buckets, np.split(idx, cuts)):
                b.extend(part.tolist())
        if min(len(b) for b in buckets) >= min_per_client:
            return [np.sort(np.asarray(b, dtype=int)) for b in buckets]
    raise RuntimeError(f"не вдалося дати кожному вузлу {min_per_client} семплів "
                       f"при alpha={alpha}; збільште alpha або зменште --clients")


def split_iid(y: np.ndarray, n_clients: int, seed: int) -> list[np.ndarray]:
    idx = np.arange(len(y))
    np.random.default_rng(seed).shuffle(idx)
    return [np.sort(p) for p in np.array_split(idx, n_clients)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--iid", action="store_true", help="рівномірне розбиття")
    ap.add_argument("--alpha", type=float, default=0.3,
                    help="концентрація Dirichlet (менше = сильніший перекіс)")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="типово: ../data")
    args = ap.parse_args()

    try:
        from sklearn.datasets import load_digits
    except ImportError:
        print("потрібен scikit-learn:  pip install scikit-learn", file=sys.stderr)
        return 1

    digits = load_digits()
    x = np.clip(np.rint(digits.images.reshape(-1, 64) / 16.0 * 255.0), 0, 255)
    x = x.astype(np.uint8)
    y = digits.target.astype(np.uint8)

    # Глобальний тест — стратифікований, і він не входить ні в один вузол.
    rng = np.random.default_rng(args.seed)
    test_mask = np.zeros(len(y), bool)
    for c in range(10):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        test_mask[idx[: int(round(len(idx) * args.test_frac))]] = True
    x_tr, y_tr = x[~test_mask], y[~test_mask]
    x_te, y_te = x[test_mask], y[test_mask]

    parts = (split_iid(y_tr, args.clients, args.seed) if args.iid
             else split_non_iid(y_tr, args.clients, args.alpha, args.seed))

    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "data"
    out.mkdir(parents=True, exist_ok=True)

    print(f"{'файл':<16} {'семплів':>8} {'байтів':>8}  класи 0..9")
    for i, p in enumerate(parts):
        blob = build_data_file(x_tr[p], y_tr[p])
        (out / f"data_{i}.bin").write_bytes(blob)
        hist = np.bincount(y_tr[p], minlength=10).tolist()
        print(f"{f'data_{i}.bin':<16} {len(p):>8} {len(blob):>8}  {hist}")

    blob = build_data_file(x_te, y_te)
    (out / "test.bin").write_bytes(blob)
    hist = np.bincount(y_te, minlength=10).tolist()
    print(f"{'test.bin':<16} {len(y_te):>8} {len(blob):>8}  {hist}   <- на плату НЕ йде")

    # Приклад для `make run` у device/ — просто перший вузол.
    (out / "data_example.bin").write_bytes((out / "data_0.bin").read_bytes())

    sizes = np.array([len(p) for p in parts], float)
    print(f"\nрозбиття: {'рівномірне' if args.iid else f'Dirichlet alpha={args.alpha}'}")
    print(f"розміри вузлів: {sizes.min():.0f}..{sizes.max():.0f} "
          f"(відношення {sizes.max() / sizes.min():.2f})")
    classes = [int((np.bincount(y_tr[p], minlength=10) > 0).sum()) for p in parts]
    print(f"класів у вузла: {min(classes)}..{max(classes)} з 10")

    # Перевірка: тестовий набір дійсно не перетинається з вузлами.
    node_rows = set()
    for i in range(len(parts)):
        bx, by = parse_data_file((out / f"data_{i}.bin").read_bytes())
        node_rows |= {bx[k].tobytes() + bytes([by[k]]) for k in range(len(by))}
    tx, ty = parse_data_file((out / "test.bin").read_bytes())
    overlap = sum(1 for k in range(len(ty))
                  if tx[k].tobytes() + bytes([ty[k]]) in node_rows)
    print(f"перетин тесту з вузлами: {overlap} семплів "
          f"({'ok' if overlap == 0 else 'ПОМИЛКА'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
