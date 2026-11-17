# THK-AI Kooperationsgraph (ohne Cloud-API)

Crawlt die Mitgliederseite des THK-AI Forschungsclusters, liest die Personenseiten, baut daraus einen Graphen
und zeigt ihn unter http://localhost:8000. Ein Hintergrunddienst hält die Daten täglich aktuell.

## Einrichtung (einmalig)
1. `python -m pip install -r requirements.txt`
2. Modell trainieren (liegt nicht im Projekt, ca. 15 Min.): in `training/`  `python fake_data.py --n-train 3000 --n-val 200 --n-test 200`
   und `python train_tagger.py --epochs 5 --batch 16` (ca. 15 Min.) → erzeugt `training/checkpoints/tagger.pt`. Ohne diese
   Datei läuft alles, aber nur mit den modellfreien Teilen (Mitglieder, Namen).
3. Einmal von Hand testen: `python src/pipeline.py`
4. Autostart einrichten (Windows): `python src/updater.py --install-autostart`
   (Entfernen: `--uninstall-autostart`). Log: `data/updater.log`

## Wer macht was
| Schritt | Quelle |
|---|---|
| Mitglieder, Namen | direkt aus dem HTML (Linktext / Überschrift), kein Modell |
| Rolle, Fakultät, Schwerpunkte, Publikationen, Projekte, Beteiligte | Tagger, **nur übernommen, wenn der Text auf der Seite wörtlich vorkommt** |
| Kanten zwischen Personen | aus gemeinsamen Projekten/Publikationen (`build_graph.py`) |

## Täglicher Ablauf (`src/updater.py`)
Beim Start und alle 30 Min. Prüfung: letzter Erfolg ≥ 24 h her → `pipeline.py`. Nur geänderte Seiten (Hash in
`data/state.json`) werden neu extrahiert; ausgetretene Personen werden entfernt; bei Fehlern bleibt die alte
`web/data/graph.json` bestehen. War der Rechner aus, wird beim nächsten Start nachgeholt.

## Wichtig: Qualität
Auf den zwei echten TH-Beispielseiten stimmt der Tagger mit dem Regel-Parser fast vollständig überein (Name, Fakultät, Rolle,
Publikationen, Jahre 1,00; Autoren 0,98). Das sind nur zwei Seiten: Vor einer Aussage in der Arbeit alle gecrawlten Seiten
mit `training/prelabel_real.py` etikettieren und den Tagger darauf testen (`training/README_TRAINING.md`).
Der Tagger ist auf das TH-Layout spezialisiert; ändert die TH ihr Seitenformat, muss neu trainiert werden.
Projekte werden bewusst nicht ausgegeben (`INCLUDE_PROJECTS` in `src/extract.py`): die Personenseiten haben keinen Projekt-Abschnitt.

## Online-Version (GitHub Pages)
`.github/workflows/pages.yml` veröffentlicht bei jedem Push auf `main`, der `web/` ändert, den Ordner `web/` als statische
Seite. Weil das Modell nicht im Repo liegt, wird `web/data/graph.json` lokal erzeugt und mitcommittet:
`python src/pipeline.py`, dann `git add web/data/graph.json && git commit -m "Graph aktualisiert" && git push`.
Einmalig nötig: unter *Settings → Pages* als Quelle „GitHub Actions“ wählen.

## Tests
`python -m pip install -r requirements-test.txt`, dann `python -m pytest`. Die Tests laufen ohne Netz und ohne Modell
(Webseiten und Tagger werden simuliert) und fassen `data/` nicht an. GitHub Actions führt sie bei jedem Push und PR aus
(`.github/workflows/tests.yml`).
