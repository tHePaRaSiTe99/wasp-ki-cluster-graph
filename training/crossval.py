"""
k-fache Kreuzvalidierung über ALLE echten Seiten (data/raw + data/gold).

Mit nur 5 festen Testseiten schwankt die Zahl stark (eine Seite wie Stadler zieht den Schnitt runter). Hier wird jede
Seite genau einmal getestet: Die Seiten werden in k Gruppen geteilt, je Durchgang ist eine Gruppe Test, der Rest
Training (+ 2 Validierungsseiten für "bestes Modell behalten"). Ergebnis: Mittelwert ± Standardabweichung über die
Durchgänge und ein Gesamtwert über alle Seiten.

  python crossval.py --folds 5 --epochs 6          # trainiert k Modelle (GPU: wenige Minuten)
  python crossval.py --folds 5 --skip-train        # nur neu auswerten

Die Aufteilung ist fest (gleicher Seed), damit Läufe vergleichbar sind. Seiten mit vielen Publikationen werden
reihum verteilt, damit jede Gruppe ähnlich viele Publikationen enthält.
"""
import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "data" / "real_cv"
SHOW = ["pubs", "pub_years", "pub_authors"]


def load_pages():
    rows = []
    for md in sorted((ROOT / "data" / "raw").glob("*.md")):
        js = ROOT / "data" / "gold" / f"{md.stem}.json"
        if js.exists():
            text = re.sub(r"<!--.*?-->", "", md.read_text(encoding="utf-8"), flags=re.S).strip() + "\n"
            rows.append({"source": md.name, "text": text, "gold": json.loads(js.read_text(encoding="utf-8"))})
    return rows


def split(rows, k):
    """Reihum nach Publikationszahl verteilen -> Gruppen mit ähnlich vielen Publikationen."""
    order = sorted(rows, key=lambda r: (-len(r["gold"]["publications"]), r["source"]))
    folds = [[] for _ in range(k)]
    for i, r in enumerate(order):
        folds[i % k if (i // k) % 2 == 0 else k - 1 - i % k].append(r)  # Schlangenlinie: 0..k-1, k-1..0, ...
    return folds


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--extra-train", default="data/fake")
    ap.add_argument("--skip-train", action="store_true", help="vorhandene Modelle checkpoints/cv/fold*.pt nur auswerten")
    args = ap.parse_args()

    sys.path.insert(0, str(HERE))
    from evaluate import backend_tagger, score

    folds = split(load_pages(), args.folds)
    fold_scores, all_pairs = [], []
    for i, test in enumerate(folds):
        rest = [r for j, f in enumerate(folds) if j != i for r in f]
        with_pubs = sorted((r for r in rest if r["gold"]["publications"]), key=lambda r: len(r["gold"]["publications"]))
        val = [with_pubs[len(with_pubs) // 2]] + [r for r in rest if not r["gold"]["publications"]][:1]
        train = [r for r in rest if r not in val]
        d = OUT / f"fold{i}"
        write(d / "train.jsonl", train); write(d / "val.jsonl", val); write(d / "test.jsonl", test)
        ck = f"checkpoints/cv/fold{i}.pt"
        print(f"\n=== Durchgang {i + 1}/{args.folds}: Test = {', '.join(r['source'][9:-3] for r in test)}", flush=True)
        if not args.skip_train:
            (HERE / ck).unlink(missing_ok=True)
            res = subprocess.run([sys.executable, "-u", str(HERE / "train_tagger.py"), "--data", str(d.relative_to(HERE)),
                                  "--extra-train", args.extra_train, "--chunk", "400", "--repeat", str(args.repeat),
                                  "--epochs", str(args.epochs), "--batch", "16", "--out", ck], cwd=HERE)
            if not (HERE / ck).exists():
                raise SystemExit(f"Training für Durchgang {i + 1} fehlgeschlagen (Code {res.returncode}), kein Modell geschrieben.")
        run = backend_tagger(ck)
        pairs = [(json.loads(run(r["text"])), r["gold"]) for r in test]
        all_pairs += pairs
        s = score(pairs)
        fold_scores.append(s)
        print("   " + " | ".join(f"{k} F1 {s[k][2]:.3f}" for k in SHOW), flush=True)

    total = score(all_pairs)
    print(f"\n{'Kategorie':12} {'Mittel F1':>9} {'± Std':>7} {'Gesamt P':>9} {'Gesamt R':>9} {'Gesamt F1':>9}")
    for k in SHOW:
        f1s = [s[k][2] for s in fold_scores]
        sd = statistics.stdev(f1s) if len(f1s) > 1 else 0.0
        p, r, f = total[k]
        print(f"{k:12} {statistics.mean(f1s):9.3f} {sd:7.3f} {p:9.3f} {r:9.3f} {f:9.3f}")
    print(f"\n{len(all_pairs)} Seiten, je einmal getestet. 'Gesamt' = über alle Seiten zusammen gezählt.")
