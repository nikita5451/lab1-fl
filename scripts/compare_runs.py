import json
import sys
from pathlib import Path

KEYS = ("acc", "loss", "dw_server", "weights_crc")


def load(d):
    rows = [json.loads(x) for x in (Path(d) / "rounds.jsonl").read_text().splitlines()]
    return {r["round"]: r for r in rows}


a, b = load(sys.argv[1]), load(sys.argv[2])
ok = True
for rnd in sorted(a):
    same = all(a[rnd][k] == b.get(rnd, {}).get(k) for k in KEYS)
    ok &= same
    print(f"раунд {rnd}: crc {a[rnd]['weights_crc']} / {b.get(rnd, {}).get('weights_crc')}  "
          f"acc {a[rnd]['acc']:.6f} / {b.get(rnd, {}).get('acc', float('nan')):.6f}  "
          f"{'збігається' if same else 'РІЗНІ'}")
print("ВІДТВОРЮВАНО" if ok else "НЕ відтворюється: шукайте джерело недетермінізму")
sys.exit(0 if ok else 1)
