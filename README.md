# THK-AI Kooperationsgraph

**Wer forscht mit wem?** Ein interaktiver Graph der Zusammenarbeit im KI-Forschungscluster
[THK-AI](https://blogs.gm.fh-koeln.de/thk-ai/mitglieder/) der TH Köln. Er wird automatisch aus den öffentlichen
Profilseiten der Mitglieder erzeugt und täglich aktualisiert.

🔗 **Live:** https://theparasite99.github.io/wasp-ki-cluster-graph/

## Was der Graph zeigt
- **Personen:** Cluster-Mitglieder und externe Personen, die als Mitautorinnen und Mitautoren oder als
  Projektbeteiligte genannt werden. Angezeigt werden auch Rolle, Fakultät und Forschungsschwerpunkte, soweit sie auf
  den Seiten stehen.
- **Projekte** und **Publikationen** mit Jahr, Erscheinungsort und Link (DOI, wenn vorhanden). Für Publikationen ohne
  Link gibt es eine Suche bei Google Scholar.
- **Verbindungen:** Zwei Personen sind verbunden, wenn sie gemeinsam an einem Projekt oder einer Publikation
  beteiligt sind. Ein Klick auf eine Linie zeigt die gemeinsamen Arbeiten.

Die Oberfläche hat drei Ansichten (Personen-Netz, Projekte, Publikationen), eine Suche und zwei Filter: nur
Cluster-Mitglieder anzeigen und Personen ohne Verbindungen ein- oder ausblenden.

## So entsteht der Graph
```
Mitgliederseite ──► Personenseiten ──► Extraktion ──► Zusammenführen ──► graph.json ──► Webseite
   (crawl.py)         (crawl.py)       (extract.py)   (build_graph.py)                  (web/)
```

1. **Crawlen** (`src/crawl.py`): Der Crawler liest die Mitgliederliste und lädt alle verlinkten Personenseiten.
   `robots.txt` wird beachtet, zwischen zwei Seiten liegt eine Sekunde Pause. Er merkt sich für jede Seite einen Hash,
   damit später nur geänderte Seiten neu verarbeitet werden. Personen, die nicht mehr verlinkt sind, werden entfernt.
2. **Extrahieren** (`src/extract.py`):

   | Angabe | Verfahren |
   |---|---|
   | Mitglieder und Namen | direkt aus dem HTML (Linktext, Seitenüberschrift), ohne Modell |
   | Projekte | ohne Modell aus dem Abschnitt „Projekte / Kooperationen“ (Name, Beschreibung, Link) |
   | Rolle, Fakultät, Schwerpunkte, Publikationen (Titel, Autoren, Jahr, Ort) | selbst trainierter Wort-Tagger (`training/`) |
   | Links zu Publikationen | ohne Modell aus dem jeweiligen Listeneintrag, DOI bevorzugt |

   Der Tagger markiert nur Wörter, die auf der Seite stehen, und erzeugt keinen eigenen Text. Jede Angabe ist daher
   ein wörtlicher Ausschnitt der Quelle, erfundene Namen oder Titel kann es nicht geben. Der Tagger läuft lokal auf
   der CPU, eine Cloud-API wird nicht genutzt.
3. **Zusammenführen** (`src/build_graph.py`):
   - Akademische Titel werden entfernt, und Namensvarianten werden zu einer Person zusammengeführt (zum Beispiel
     „A. Muster“ und „Anna Muster“). Bei mehrdeutigen Initialen wird nicht geraten.
   - Dasselbe Projekt oder dieselbe Publikation auf mehreren Seiten wird auch bei Tippfehlern, anderer Schreibweise
     oder Abkürzung erkannt.
   - Personen, die in einer Projektbeschreibung mit vollem Namen genannt werden, gelten als beteiligt.
   - Arbeiten mit mehr als 15 Beteiligten erzeugen keine paarweisen Verbindungen.
4. **Veröffentlichen:** Das Ergebnis ist `web/data/graph.json`. Die Webseite (`web/`, [Cytoscape.js](https://js.cytoscape.org/))
   ist rein statisch und lädt nur diese Datei.

## Automatische Aktualisierung
Der Workflow `.github/workflows/pages.yml` führt die ganze Kette in GitHub Actions aus und veröffentlicht das Ergebnis
auf GitHub Pages:

- **Wann:** täglich um 04:17 UTC, bei jedem Push auf `main`, der `src/`, `web/` oder das Modell ändert, und von Hand
  unter *Actions → Crawl and deploy website → Run workflow*.
- **Zwischenspeicher:** Crawl-Daten und der zuletzt gebaute Graph liegen im Actions-Cache. Dadurch werden nur
  geänderte Seiten neu extrahiert, und eine Seite, die vorübergehend nicht lädt, behält ihre bisherigen Daten.
- **Bei Fehlern:** Schlägt das Crawlen fehl oder enthält der Graph keine Mitglieder bzw. Publikationen, wird nichts
  veröffentlicht. Die bisherige Version bleibt online.

`web/data/graph.json` im Repository ist nur ein Startwert und muss nicht von Hand gepflegt werden.

## Lokal ausführen
Voraussetzung ist Python 3.10 oder neuer.

```bash
python -m pip install -r requirements.txt
python src/pipeline.py            # crawlen, extrahieren, graph.json bauen (nur Änderungen)
python src/pipeline.py --force    # alle Seiten neu laden und extrahieren
python src/updater.py             # Graph unter http://localhost:8000 anzeigen und täglich aktualisieren
```

`src/updater.py` ist ein kleiner Hintergrunddienst. Er prüft alle 30 Minuten, ob der letzte erfolgreiche Lauf
24 Stunden zurückliegt, holt verpasste Läufe nach dem Hochfahren nach und schreibt ein Protokoll nach
`data/updater.log`. Unter Windows lässt er sich mit `--install-autostart` bei der Anmeldung starten
(`--uninstall-autostart` entfernt das wieder). `--once` führt nur einen einzigen Lauf aus.

Das trainierte Modell liegt unter `training/checkpoints/tagger.pt`. Fehlt es, läuft die Pipeline trotzdem, dann aber
nur mit den modellfreien Teilen: Mitglieder, Namen und Projekte, ohne Publikationen.

## Projektstruktur
```
src/          Pipeline: crawl.py, extract.py, build_graph.py, pipeline.py, updater.py, state.py
web/          statische Webseite (index.html, app.js, style.css) und data/graph.json
training/     Tagger-Modell: Trainingsdaten, Training, Auswertung (siehe training/README_TRAINING.md)
data/         lokale Arbeitsdaten (Rohseiten, Extraktionen, Zustand). Nicht im Repository.
.github/      GitHub-Actions-Workflows
```

## Qualität und Grenzen
- **Messwerte:** In einer 5-fachen Kreuzvalidierung über alle 28 Personenseiten erkennt der Tagger
  Publikationstitel mit einem F1-Wert von **0,95 ± 0,03**, Titel und Autoren zusammen mit 0,91. Einzelheiten und
  weitere Messungen stehen in [`training/README_TRAINING.md`](training/README_TRAINING.md).
- **Vergleichsdaten:** Die Vergleichsdaten stammen von einem regelbasierten Parser und sind nicht von Hand geprüft.
- **Layout-Abhängigkeit:** Der Tagger ist auf das Seitenlayout der TH Köln spezialisiert. In einem fremden Layout
  erkennt er keine Titel. Ändert die TH ihr Seitenformat, muss neu trainiert werden.
- **Abgeleitete Verbindungen:** Eine Verbindung bedeutet nur, dass zwei Personen gemeinsam bei einem Projekt oder
  einer Publikation genannt werden. Das ist eine automatisch abgeleitete Näherung, keine vollständige oder
  geprüfte Darstellung der Zusammenarbeit.
- **Datenquelle:** Alle Angaben stammen aus öffentlich zugänglichen Profilseiten. Was dort fehlt oder veraltet ist,
  fehlt auch im Graphen.
