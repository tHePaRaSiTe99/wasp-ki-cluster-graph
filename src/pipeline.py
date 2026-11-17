"""
Ein Durchlauf: Crawlen -> (nur geänderte Seiten) extrahieren -> Graph bauen -> graph.json ersetzen.

  python src/pipeline.py            # normaler Lauf, nur Änderungen werden neu verarbeitet
  python src/pipeline.py --force    # alles neu laden und neu extrahieren

Sicherheitsnetz: graph.json wird erst ersetzt, wenn der neue Graph fertig und nicht leer ist.
Bei einem Fehler (Netz weg, Seite leer, ...) bleibt die bisherige, funktionierende Datei bestehen.
"""
import argparse
import os
import time

import state as state_store
from build_graph import OUTPUT_PATH, build_graph, export_for_cytoscape
from crawl import MEMBERS_URL, crawl
from extract import extract_files


def run(start_url: str = MEMBERS_URL, max_pages: int = 80, force: bool = False) -> str:
    st = state_store.load()
    st["last_attempt"] = time.time()
    state_store.save(st)

    result = crawl(start_url, max_pages=max_pages, force=force)
    extract_files(result.changed)

    changed = bool(result.changed or result.removed)
    if not changed and OUTPUT_PATH.exists() and not force:
        status = "Keine Änderungen - graph.json bleibt unverändert."
    else:
        graph = build_graph()
        if graph.number_of_nodes() == 0:
            raise RuntimeError("Der neue Graph ist leer - graph.json wird nicht überschrieben.")
        tmp = OUTPUT_PATH.with_suffix(".tmp")
        export_for_cytoscape(graph, tmp)
        os.replace(tmp, OUTPUT_PATH)  # atomar: der Browser sieht nie eine halbe Datei
        status = f"graph.json aktualisiert ({len(result.changed)} Seiten neu/geändert, {len(set(result.removed))} entfernt)."

    st = state_store.load()
    st["last_success"] = time.time()
    st["last_status"] = status
    state_store.save(st)
    print(status)
    return status


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=MEMBERS_URL)
    ap.add_argument("--max-pages", type=int, default=80)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run(a.url, a.max_pages, a.force)
