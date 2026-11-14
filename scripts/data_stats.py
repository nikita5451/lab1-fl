from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "host"))
from fl_lab import fl_model_ref as ref  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return ref.parse_data_file((ROOT / "data" / name).read_bytes())


def main() -> None:
    shards = [load(f"data_{i}.bin") for i in range(4)]
    xt, yt = load("test.bin")
    allx = np.concatenate([s[0] for s in shards])
    ally = np.concatenate([s[1] for s in shards])
    glob = np.bincount(ally, minlength=10) / len(ally)
    print(f"{'вузол':>6} {'n':>5} {'кл.':>4} {'TV до глоб.':>12}  гістограма класів 0..9")
    tvs = []
    for i, (x, y) in enumerate(shards):
        h = np.bincount(y, minlength=10)
        p = h / h.sum()
        tv = 0.5 * np.abs(p - glob).sum()
        tvs.append(tv)
        print(f"{i:>6} {len(y):>5} {int((h > 0).sum()):>4} {tv:>12.3f}  {h.tolist()}")
    sizes = [len(s[1]) for s in shards]
    print(f"\nрозмір: min={min(sizes)} max={max(sizes)} відношення={max(sizes)/min(sizes):.2f}")
    print(f"середній TV = {np.mean(tvs):.3f} (0 = рівномірно, більше = сильніший перекіс)")
    ht = np.bincount(yt, minlength=10)
    print(f"test.bin: n={len(yt)} гістограма {ht.tolist()}")
    keys = {bytes(r) for r in allx}
    inter = sum(bytes(r) in keys for r in xt)
    print(f"перетин test.bin з даними вузлів (за вмістом семпла): {inter}")


if __name__ == "__main__":
    main()
