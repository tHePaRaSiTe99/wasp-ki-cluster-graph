"""
Extraktion mit dem selbst trainierten Wort-Tagger (training/tagger.py) - ohne Cloud-API, ohne Schlüssel.

Der Tagger markiert Wörter im Seitentext (Person, Titel, Jahr, Rolle ...). Er erzeugt keinen neuen Text, jede
Ausgabe ist also wörtlich ein Ausschnitt der Seite: Erfundene Namen sind ausgeschlossen.
Die Mitgliederliste (Crawler) und der Name in der Seitenüberschrift kommen ohne Modell direkt aus dem HTML.

Gibt es noch kein trainiertes Modell (training/checkpoints/tagger.pt), laufen nur die modellfreien Teile.
Wer später ein anderes Modell hat, tauscht nur `_run_model` aus.
"""
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
EXTRACTED_DIR = ROOT / "data" / "extracted"
CHECKPOINT = ROOT / "training" / "checkpoints" / "tagger.pt"

# Auf den TH-Köln-Personenseiten gibt es keinen Projekt-Abschnitt (nur Publikationen). Projekte aus dem Modell würden
# dort nur Fehlgriffe erzeugen; für andere Seiten auf True setzen.
INCLUDE_PROJECTS = False

_model = None
_model_failed = False


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return " ".join("".join(c for c in s if not unicodedata.combining(c)).casefold().split())


# ---------------------------------------------------------------------------
# Modell: Wort-Tagger (training/tagger.py)
# ---------------------------------------------------------------------------
def _load_model():
    global _model, _model_failed
    if _model is not None or _model_failed:
        return _model
    if not CHECKPOINT.exists():
        print(f"[extract] Kein Modell unter {CHECKPOINT} - nur modellfreie Extraktion (Namen, Mitglieder).")
        _model_failed = True
        return None
    try:
        import torch
        sys.path.insert(0, str(ROOT / "training"))
        from tagger import Tagger
        ck = torch.load(CHECKPOINT, map_location="cpu")
        m = Tagger(len(ck["vocab"]) + 2)
        m.load_state_dict(ck["model"])
        m.eval()
        _model = (m, ck["vocab"])
    except Exception as e:  # torch fehlt, Datei defekt, ...
        print(f"[extract] Modell konnte nicht geladen werden ({e}) - nur modellfreie Extraktion.")
        _model_failed = True
    return _model


def _run_model(text: str) -> dict:
    """Seitentext -> {"people":[...], "publications":[...], "projects":[...]} (leer, wenn kein Modell da ist).
    Der Tagger markiert Wörter im Text; jede Ausgabe ist deshalb wörtlich ein Ausschnitt der Seite."""
    loaded = _load_model()
    if loaded is None:
        return {}
    sys.path.insert(0, str(ROOT / "training"))
    from tagger import predict, tags_to_record
    model, vocab = loaded
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S).strip() + "\n"  # Quell-Kommentar des Crawlers ist kein Seiteninhalt
    toks, tags = predict(model, vocab, text)
    return tags_to_record(text, toks, tags) if toks else {}


def _h1(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return " ".join(line[2:].split())
    return ""


def extract_text(text: str) -> dict:
    """Seitentext -> {"people":[...], "projects":[...], "cooperations":[]} (Format wie von build_graph erwartet)."""
    rec = _run_model(text)
    owner_h1 = _h1(text)
    person = (rec.get("people") or [{}])[0]
    from build_graph import clean_name  # entfernt "Prof. Dr.-Ing. ..." (gleiche Regel wie beim Zusammenführen)
    owner = clean_name(owner_h1 or person.get("name", ""))  # Überschrift ist verlässlicher als die Modellvermutung
    people = []
    if owner:
        people.append({"name": owner, "role": person.get("role", ""), "faculty": person.get("faculty", ""),
                       "expertise": person.get("expertise", [])})

    projects: dict[str, dict] = {}
    items = [{"name": p["title"], "participants": p["authors"], "year": p.get("year"), "venue": p.get("venue", ""),
              "kind": "publication"} for p in rec.get("publications") or []]
    if INCLUDE_PROJECTS:
        items += [{"name": p["name"], "participants": p["participants"], "kind": "project"} for p in rec.get("projects") or []]
    for it in items:
        key = _fold(it["name"])
        if not key:
            continue
        parts = list(it["participants"])
        if owner and _fold(owner) not in {_fold(x) for x in parts}:
            parts.append(owner)  # steht auf der Seite der Person -> die Person ist beteiligt
        projects[key] = {**it, "description": "", "participants": parts}
    return {"people": people, "projects": list(projects.values()), "cooperations": []}


def extract_from_file(md_path: Path) -> dict:
    return extract_text(md_path.read_text(encoding="utf-8"))


def extract_files(md_paths: list[Path]) -> list[Path]:
    """Extrahiert die übergebenen Seiten und schreibt data/extracted/<name>.json (atomar)."""
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for i, md_path in enumerate(md_paths, 1):
        print(f"Extrahiere ({i}/{len(md_paths)}): {md_path.name}", flush=True)
        data = extract_from_file(md_path)
        target = EXTRACTED_DIR / f"{md_path.stem}.json"
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, target)
        out.append(target)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Extrahiert alle Seiten in data/raw neu (ohne Änderungserkennung).")
    ap.parse_args()
    extract_files(sorted(RAW_DIR.glob("*.md")))
