"""
Misst, wie gut ein Extraktions-Modell ist - gegen die bekannte richtige Lösung (gold).

Metriken (jeweils Precision / Recall / F1 über Mengen, Namen vereinheitlicht):
  people        Personennamen
  pubs          Publikationstitel
  pub_authors   Paare (Titel, Autor)        -> das sind die späteren Kanten im Graphen
  pub_years     Paare (Titel, Jahr)
  projects      Projektnamen
  proj_members  Paare (Projekt, Teilnehmer) -> ebenfalls Kanten
Zusätzlich: Anteil gültiger JSON-Ausgaben.

Backends:
  scratch   Mini-GPT als Textgenerator (checkpoints/tiny.pt)
  tagger    Wort-Tagger (checkpoints/tagger.pt)  <- der bessere Ansatz
  rules     regelbasierter Parser für das echte TH-Layout (real_rules.py) = Basislinie
  gold      Selbsttest: gibt die Lösung zurück, muss F1 = 1.0 ergeben

Beispiele:
  python evaluate.py --backend scratch --split test
  python evaluate.py --backend gold --split test      # Selbsttest
"""
import argparse
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s)).lower()
    return " ".join("".join(c for c in s if not unicodedata.combining(c)).split())


def facts(d: dict) -> dict[str, set]:
    f = defaultdict(set)
    for p in d.get("people", []) or []:
        n = norm(p.get("name", ""))
        f["people"].add(n)
        if p.get("role"): f["role"].add((n, norm(p["role"])))
        if p.get("faculty"): f["faculty"].add((n, norm(p["faculty"])))
        for e in p.get("expertise", []) or []:
            f["expertise"].add((n, norm(e)))
    for p in d.get("publications", []) or []:
        t = norm(p.get("title", ""))
        f["pubs"].add(t)
        if p.get("year") is not None:
            f["pub_years"].add((t, str(p["year"])))
        for a in p.get("authors", []) or []:
            f["pub_authors"].add((t, norm(a)))
    for p in d.get("projects", []) or []:
        n = norm(p.get("name", ""))
        f["projects"].add(n)
        for a in p.get("participants", []) or []:
            f["proj_members"].add((n, norm(a)))
    return f


KEYS = ["people", "role", "faculty", "expertise", "pubs", "pub_authors", "pub_years", "projects", "proj_members"]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score(pairs):
    """pairs = [(vorhergesagt, gold), ...]; Micro-Durchschnitt über alle Seiten."""
    c = {k: [0, 0, 0] for k in KEYS}
    for pred, gold in pairs:
        fp_, fg = facts(pred), facts(gold)
        for k in KEYS:
            a, b = fp_.get(k, set()), fg.get(k, set())
            c[k][0] += len(a & b); c[k][1] += len(a - b); c[k][2] += len(b - a)
    return {k: prf(*c[k]) for k in KEYS}


# ------------------------------- Backends ----------------------------------
def backend_scratch():
    import torch
    from tiny_gpt import TinyGPT, decode, encode_pair
    ck = torch.load(HERE / "checkpoints" / "tiny.pt", map_location="cpu")
    m = TinyGPT(d=ck["d"], layers=ck["layers"], ctx=ck["ctx"]); m.load_state_dict(ck["model"]); m.eval()

    def run(text):
        prompt, _ = encode_pair(text, None, m.ctx)
        return decode(m.generate(prompt, max_new=m.ctx - len(prompt)))
    return run


def backend_tagger():
    import torch
    from tagger import Tagger, predict, tags_to_record
    ck = torch.load(HERE / "checkpoints" / "tagger.pt", map_location="cpu")
    model = Tagger(len(ck["vocab"]) + 2); model.load_state_dict(ck["model"])

    def run(text):
        toks, tags = predict(model, ck["vocab"], text)
        return json.dumps(tags_to_record(text, toks, tags), ensure_ascii=False)
    return run


def backend_rules():
    from real_rules import parse_page
    return lambda text: json.dumps(parse_page(text), ensure_ascii=False)


def backend_gold():
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["scratch", "tagger", "rules", "gold"], default="scratch")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data", default="data/fake", help="Datenverzeichnis, z.B. data/real")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", type=int, default=2, help="so viele Fehlerbeispiele anzeigen")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(HERE / args.data / f"{args.split}.jsonl", encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    run = {"scratch": backend_scratch, "tagger": backend_tagger, "rules": backend_rules, "gold": backend_gold}[args.backend]()

    pairs, valid, bad = [], 0, []
    for i, r in enumerate(rows):
        out = json.dumps(r["gold"]) if run is None else run(r["text"])
        try:
            pred = json.loads(out); valid += 1
        except json.JSONDecodeError:
            pred = {}
        pairs.append((pred, r["gold"]))
        if pred != r["gold"] and len(bad) < args.show:
            bad.append((r["text"], out, r["gold"]))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)}", file=sys.stderr, flush=True)

    res = score(pairs)
    print(f"\nBackend: {args.backend} | Split: {args.split} | {len(rows)} Seiten | gültiges JSON: {valid / len(rows):.1%}")
    print(f"{'Kategorie':14} {'Precision':>9} {'Recall':>8} {'F1':>7}")
    for k in KEYS:
        p, r_, f = res[k]
        print(f"{k:14} {p:9.3f} {r_:8.3f} {f:7.3f}")
    print(f"{'Ø F1':14} {'':9} {'':8} {sum(v[2] for v in res.values()) / len(res):7.3f}")
    for text, out, gold in bad:
        print("\n--- Beispiel mit Abweichung ---\n" + text + "\nMODELL:  " + out + "\nGOLD:    " + json.dumps(gold, ensure_ascii=False, separators=(",", ":")))
