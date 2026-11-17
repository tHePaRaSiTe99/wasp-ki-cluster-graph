# Modell trainieren, testen, verbessern

Es gibt drei Ansätze, alle mit derselben Bewertung (`evaluate.py`: Precision/Recall/F1 je Kategorie):

| Ansatz | Dateien | Ergebnis auf Fake-Test | Kommentar |
|---|---|---|---|
| **Wort-Tagger** (wird von der Pipeline benutzt) | `tagger.py`, `train_tagger.py` | siehe Messwerte unten | markiert Wörter im Text, erfindet nichts, CPU |
| Regel-Parser (Basislinie) | `real_rules.py` | 1,00 auf den echten Seiten | kein Training, liest das feste TH-Layout direkt |
| Mini-GPT (Textgenerator) | `tiny_gpt.py`, `train_scratch.py` | Ø-F1 ≈ 0,01 | muss Text "abschreiben" lernen, braucht weit mehr Rechenzeit; als Lernstück behalten |
| LoRA-Feintuning (vortrainiertes Modell) | `finetune_lora.py` | ungetestet | braucht GPU; Option, falls der Tagger an echten Seiten scheitert |

## Messwerte (Stand jetzt)
| Test | Ergebnis |
|---|---|
| Fake-Seiten im echten TH-Format (`test_th`), neue Namen | Ø-F1 0,999 |
| **2 echte TH-Seiten** (Gaida mit 10 Publikationen, Cerfontaine ohne) gegen den Regel-Parser | Name, Fakultät, Rolle, Publikationen, Jahre: 1,00; Autoren 0,98; Schwerpunkte 0,95 |
| Fake-Seiten in einem fremden Layout (`test_unseen`) | 0,64 (das Modell ist jetzt auf das TH-Layout spezialisiert) |

Wichtig zum Lesen: Die "echten Seiten" sind nur zwei, und die Vergleichslösung stammt vom Regel-Parser, nicht von Hand.
Der Parser liest das feste TH-Layout direkt (`real_rules.py`, Ø-F1 1,00 auf den echten Seiten) und ist eine starke
Basislinie. Für die Arbeit ist das ein Ergebnis: Bei so regelmäßigen Seiten reichen Regeln; das Modell zeigt, dass ein
gelerntes Verfahren dasselbe erreicht und bei Layout-Änderungen flexibler sein kann.

## Tests, die du selbst ausführen kannst
```
python fake_data.py --n-train 3000 --n-val 200 --n-test 200   # Fake-Daten inkl. test_th und test_unseen
python train_tagger.py --epochs 5 --batch 16                   # ca. 15 Min. auf CPU
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
