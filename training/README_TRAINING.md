# Modell trainieren, testen, verbessern

Es gibt drei Ansätze, alle mit derselben Bewertung (`evaluate.py`: Precision/Recall/F1 je Kategorie):

| Ansatz | Dateien | Ergebnis auf Fake-Test | Kommentar |
|---|---|---|---|
| **Wort-Tagger** (wird von der Pipeline benutzt) | `tagger.py`, `train_tagger.py` | siehe Messwerte unten | markiert Wörter im Text, erfindet nichts, CPU |
| Regel-Parser (Basislinie) | `real_rules.py` | 1,00 auf den echten Seiten | kein Training, liest das feste TH-Layout direkt |
| Mini-GPT (Textgenerator) | `tiny_gpt.py`, `train_scratch.py` | Ø-F1 ≈ 0,01 | muss Text "abschreiben" lernen, braucht weit mehr Rechenzeit; als Lernstück behalten |
| LoRA-Feintuning (vortrainiertes Modell) | `finetune_lora.py` | ungetestet | braucht GPU; Option, falls der Tagger an echten Seiten scheitert |

## Messwerte (Stand jetzt)
**5-fache Kreuzvalidierung über alle 28 echten Seiten** (`crossval.py`, jede Seite genau einmal getestet,
Training: echte Seiten ×10 + Fake-Daten, bestes Modell nach Validierungsverlust):

| Variante | Titel F1 (Mittel ± Std) | Titel+Jahr | Titel+Autor |
|---|---|---|---|
| Fake-Daten nur einzeilig (`data/fake_v1`) | 0,902 ± 0,061 | 0,901 | 0,878 |
| **Fake-Daten mit zweizeiligem Format + Satzzeichen-Titeln** (`data/fake`) | **0,950 ± 0,027** | 0,947 | 0,909 |
| Zum Vergleich: nur Fake-Daten, fester Test mit 5 Seiten | 0,33 | 0,32 | 0,16 |

| Weitere Tests (aktuelles Modell) | Ergebnis |
|---|---|
| Fake-Seiten im echten TH-Format (`test_th`), neue Namen | Ø-F1 1,00 |
| Fake-Seiten in einem fremden Layout (`test_unseen`) | Ø-F1 0,52, Titel 0,00 (auch ältere Modelle: Titel 0,00) - das Modell ist auf das TH-Layout spezialisiert |

Die echten Seiten haben zwei Literatur-Layouts: einzeilig `Autoren (Jahr): Titel. In: ...` und zweizeilig
`* Titel` / `  Autoren, Jahr, Hg.: ...`. Die Auswertung fasst deshalb einen Listeneintrag (Aufzählungspunkt + eingerückte
Folgezeilen) als eine Literaturangabe zusammen (`tagger.entry_ids`).

Wichtig zum Lesen: Die Vergleichslösung stammt vom Regel-Parser (`prelabel_real.py`), nicht von Hand geprüft.
Der Parser liest das feste TH-Layout direkt (`real_rules.py`, Ø-F1 1,00 auf den echten Seiten) und ist eine starke
Basislinie. Für die Arbeit ist das ein Ergebnis: Bei so regelmäßigen Seiten reichen Regeln; das Modell zeigt, dass ein
gelerntes Verfahren dasselbe erreicht und bei Layout-Änderungen flexibler sein kann.

## Tests, die du selbst ausführen kannst
```
python fake_data.py --n-train 6000 --n-val 300 --n-test 300   # Fake-Daten inkl. test_th und test_unseen
python train_tagger.py --epochs 5 --batch 16                   # GPU (falls vorhanden) ca. 1 Min., CPU ca. 30 Min.
python crossval.py --folds 5 --epochs 6                        # Kreuzvalidierung über alle echten Seiten (GPU ca. 10 Min.)
python evaluate.py --backend tagger --split test_th            # TH-Layout, neue Namen
python evaluate.py --backend tagger --split test_unseen        # fremdes Layout (strenger Test)
python evaluate.py --backend rules  --split test_th            # Regel-Parser zum Vergleich
python evaluate.py --backend gold                              # Selbsttest der Bewertung, muss 1,0 ergeben
```

## Auf echten Seiten verbessern (das zählt für die Arbeit)
1. `python src/pipeline.py` (Crawler füllt `data/raw`)
2. `python prelabel_real.py` – der Regel-Parser schreibt für ALLE gecrawlten Seiten Etiketten nach `data/gold/*.json`.
   Stichproben (5-10 Seiten) von Hand prüfen und korrigieren, vor allem Seiten mit ungewöhnlichen Literaturangaben.
3. `python make_real_dataset.py` – teilt nach Seiten in Train/Val/Test
4. `python train_tagger.py --data data/real --extra-train data/fake` – echte Seiten + Fake-Daten gemischt
5. `python evaluate.py --backend tagger --data data/real --split test` – Zahl für die Arbeit
6. Fehler ansehen (`--show 5`), Muster erkennen (z. B. neues Layout), Seiten dazulabeln, wiederholen.

Fake-Zahlen beweisen nur, dass die Technik funktioniert. Aussagekraft hat nur die Zahl auf echten, handkorrigierten Testseiten.
Mit nur wenigen Echtseiten ist die Zahl auf 2-3 Testseiten noch sehr wackelig.

## Weitere Hebel, nach Wirkung sortiert
1. Mehr echte, korrigierte Seiten (das wirkt am stärksten).
2. Neue Layout-Varianten in `fake_data.py` ergänzen, wenn Fehler zeigen, wie echte Seiten aussehen.
3. Mehr Epochen / größeres Modell (`Tagger(hid=256)`) - erst danach, bringt weniger.
4. Fallback LoRA mit GPU.
