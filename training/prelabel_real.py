"""
Echte Seiten automatisch labeln, damit du sie nur noch stichprobenartig korrigieren musst.

  python prelabel_real.py
Liest data/raw/*.md (Crawler) und schreibt je Seite data/gold/<name>.json. Das Etikett kommt vom regelbasierten Parser
(real_rules.py), der das feste TH-Layout (Literaturangaben "Nachname, Vorname; ... (Jahr): Titel. ...") direkt liest.
>>> Öffne die JSON-Dateien, korrigiere/ergänze sie. Das Ergebnis ist die "richtige Lösung" für Training und Test. <<<
Format:  people[0] = Besitzer der Seite (name, role, faculty, expertise), publications (title, year, venue, authors),
         projects (name, participants). Texte müssen WÖRTLICH so auf der Seite stehen.
Bereits vorhandene Dateien werden nicht überschrieben (deine Korrekturen bleiben).
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "gold"
OUT.mkdir(parents=True, exist_ok=True)

from real_rules import parse_page

n_new = 0
for md in sorted((ROOT / "data" / "raw").glob("*.md")):
    target = OUT / f"{md.stem}.json"
    if target.exists():
        continue
    text = re.sub(r"<!--.*?-->", "", md.read_text(encoding="utf-8"), flags=re.S).strip() + "\n"
    target.write_text(json.dumps(parse_page(text), ensure_ascii=False, indent=2), encoding="utf-8")
    n_new += 1
print(f"{n_new} neue Etikett-Dateien in {OUT}. Stichproben prüfen/korrigieren, dann make_real_dataset.py.")
