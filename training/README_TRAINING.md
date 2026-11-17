# Extraktionsmodell: Training und Auswertung

Dieser Ordner enthält das Modell, mit dem die Pipeline Angaben aus den Personenseiten liest: Rolle, Fakultät,
Schwerpunkte und Publikationen mit Titel, Autoren, Jahr und Erscheinungsort. Alle Befehle werden in `training/`
ausgeführt. Daten liegen unter `training/data/`, Modelle unter `training/checkpoints/`.

Das Modell, das die Pipeline benutzt, ist `checkpoints/tagger.pt` und liegt im Repository. Weitere Checkpoints und
alle Trainingsdaten werden nicht versioniert.

## Ansätze
Alle Ansätze werden mit derselben Bewertung verglichen (`evaluate.py`: Precision, Recall und F1 je Kategorie).

| Ansatz | Dateien | Status |
|---|---|---|
| **Wort-Tagger** | `tagger.py`, `train_tagger.py` | wird von der Pipeline benutzt. Markiert Wörter im Text, erzeugt keinen eigenen Text, läuft auf der CPU. |
| Regel-Parser | `real_rules.py` | Basislinie ohne Training. Liest das feste Layout der TH-Seiten direkt und erzeugt die Vergleichsdaten. |
| Mini-GPT (Textgenerator) | `tiny_gpt.py`, `train_scratch.py` | Lernbeispiel. Ø-F1 ≈ 0,01, weil das Modell lernen muss, Text abzuschreiben. |
| LoRA-Feintuning | `finetune_lora.py` | nicht ausgewertet. Braucht eine GPU, gedacht als Ausweichlösung für deutlich andere Layouts. |

## Messwerte
**5-fache Kreuzvalidierung über alle 28 echten Personenseiten** (`crossval.py`): Jede Seite wird genau einmal
getestet. Trainiert wird mit den echten Seiten (zehnfach gewichtet) und künstlich erzeugten Seiten. Behalten wird
das Modell mit dem kleinsten Validierungsverlust.

| Trainingsdaten | Titel F1 (Mittel ± Std) | Titel + Jahr | Titel + Autor |
|---|---|---|---|
| Künstliche Seiten, nur einzeilige Literaturangaben | 0,902 ± 0,061 | 0,901 | 0,878 |
| **Künstliche Seiten mit zwei- und einzeiligem Format und Titeln mit Satzzeichen** | **0,950 ± 0,027** | 0,947 | 0,909 |
| Zum Vergleich: nur künstliche Seiten, fester Test mit 5 Seiten | 0,33 | 0,32 | 0,16 |

| Weitere Tests | Ergebnis |
|---|---|
| Künstliche Seiten im TH-Layout mit neuen Namen (`test_th`) | Ø-F1 1,00 |
| Künstliche Seiten in einem fremden Layout (`test_unseen`) | Ø-F1 0,52, Titel 0,00 |

So sind die Zahlen zu lesen:
- **Vergleichsdaten:** Sie stammen vom Regel-Parser (`prelabel_real.py`) und sind nicht von Hand geprüft.
- **Regel-Parser als Basislinie:** Auf den regelmäßig aufgebauten TH-Seiten erreicht der Regel-Parser selbst
  Ø-F1 1,00. Der Tagger zeigt, dass ein gelerntes Verfahren nahezu dasselbe erreicht. Bei Layout-Änderungen lässt er
  sich durch neues Training anpassen, statt Regeln umzuschreiben.
- **Layout-Abhängigkeit:** Das Modell ist auf das TH-Layout spezialisiert. In einem fremden Layout findet es keine
  Titel, ältere Modelle ebenfalls nicht.

Die echten Seiten nutzen zwei Formate für Literaturangaben: einzeilig `Autoren (Jahr): Titel. In: ...` und zweizeilig
`* Titel` / `  Autoren, Jahr, Hg.: ...`. Die Auswertung fasst daher einen Listeneintrag (Aufzählungspunkt mit
eingerückten Folgezeilen) zu einer Literaturangabe zusammen (`tagger.entry_ids`).

## Befehle
```bash
python fake_data.py --n-train 6000 --n-val 300 --n-test 300   # künstliche Seiten inkl. test_th und test_unseen
python train_tagger.py --epochs 5 --batch 16                   # mit GPU ca. 1 Min., auf der CPU ca. 30 Min.
python crossval.py --folds 5 --epochs 6                        # Kreuzvalidierung über alle echten Seiten
python evaluate.py --backend tagger --split test_th            # TH-Layout, neue Namen
python evaluate.py --backend tagger --split test_unseen        # fremdes Layout
python evaluate.py --backend rules  --split test_th            # Regel-Parser zum Vergleich
python evaluate.py --backend gold                              # Selbsttest der Bewertung (muss 1,0 ergeben)
```

Weitere nützliche Optionen:
- `train_tagger.py`:
  - `--data` und `--extra-train`: Datensätze mischen.
  - `--repeat`: echte Seiten höher gewichten.
  - `--chunk`: lange Seiten aufteilen.
  - `--device`, `--out`, `--seed`
- `evaluate.py`:
  - `--checkpoint`: Modelle vergleichen.
  - `--show N`: Fehlerbeispiele anzeigen.

## Mit echten Seiten trainieren
1. `python ../src/pipeline.py` crawlt die Seiten nach `data/raw/` im Projektordner.
2. `python prelabel_real.py` lässt den Regel-Parser alle gecrawlten Seiten etikettieren und schreibt die Ergebnisse
   nach `data/gold/` im Projektordner. Stichproben sollten von Hand geprüft und korrigiert werden, besonders Seiten mit ungewöhnlichen
   Literaturangaben.
3. `python make_real_dataset.py` teilt die Seiten in Trainings-, Validierungs- und Testdaten auf.
4. `python train_tagger.py --data data/real --extra-train data/fake` trainiert mit echten und künstlichen Seiten.
5. `python evaluate.py --backend tagger --data data/real --split test` wertet auf echten Testseiten aus.
6. Danach werden die Fehler angesehen (`--show 5`). Wo sich Muster zeigen, zum Beispiel ein neues Layout, werden
   weitere Seiten etikettiert, und der Ablauf wird wiederholt.

Ergebnisse auf künstlichen Seiten zeigen nur, dass das Verfahren grundsätzlich funktioniert. Aussagekräftig sind
Zahlen auf echten, von Hand geprüften Testseiten. Bei wenigen Testseiten schwanken sie stark, deshalb gibt es die
Kreuzvalidierung.

Ein neu trainiertes Modell kommt in Gebrauch, indem `checkpoints/tagger.pt` ersetzt und committet wird. Der Workflow
auf GitHub extrahiert danach alle Seiten neu und veröffentlicht den Graphen.

## Möglichkeiten zur Verbesserung
Nach erwarteter Wirkung sortiert:
1. Mehr echte, von Hand geprüfte Seiten.
2. Neue Layout-Varianten in `fake_data.py`, wenn Fehler zeigen, wie echte Seiten aussehen.
3. Mehr Epochen oder ein größeres Modell (`Tagger(hid=256)`). Das bringt meist weniger.
4. LoRA-Feintuning eines vortrainierten Modells (GPU).
