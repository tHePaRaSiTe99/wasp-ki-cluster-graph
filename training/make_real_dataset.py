"""
Baut aus deinen ECHTEN Seiten einen Trainings-/Testdatensatz (data/real/{train,val,test}.jsonl).

Eingabe:  data/raw/*.md   (Crawler)  und  data/gold/*.json  (von Hand korrigierte Lösungen, siehe prelabel_real.py)
Aufteilung nach Seiten (nicht nach Zeilen): keine Seite kommt in Train UND Test vor.
Die Test-Seiten sind deine eigentliche Bewertung für die Arbeit - nur dort darfst du nichts "schönen".
"""
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "data" / "real"

rows = []
for md in sorted((ROOT / "data" / "raw").glob("*.md")):
    js = ROOT / "data" / "gold" / f"{md.stem}.json"
    if js.exists():
        text = re.sub(r"<!--.*?-->", "", md.read_text(encoding="utf-8"), flags=re.S).strip() + "\n"
        rows.append({"source": md.name, "text": text, "gold": json.loads(js.read_text(encoding="utf-8"))})
if len(rows) < 6:
    raise SystemExit(f"Nur {len(rows)} korrigierte Seiten in data/gold - mindestens 6 nötig (besser 20+).")

random.Random(0).shuffle(rows)
n = len(rows)
n_test, n_val = max(2, n // 5), max(1, n // 10)
splits = {"test": rows[:n_test], "val": rows[n_test:n_test + n_val], "train": rows[n_test + n_val:]}
OUT.mkdir(parents=True, exist_ok=True)
for name, part in splits.items():
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(part)} Seiten")
