"""
Regelbasierter Parser für die echten TH-Köln-Personenseiten (Markdown aus src/crawl.py).

Die Seiten folgen einem festen Muster (Überschrift, Fakultät, Institut, Kontakt, dann Abschnitte "Funktionen",
"Lehrgebiete", "Forschungsgebiete", "+ Publikationen" ...). Publikationen sind Literaturangaben:
    Nachname, Vorname; Nachname, Vorname (2016): Titel. In: Zeitschrift. Vol. 68, S. 1 - 9. <https://doi.org/...>
Dieser Parser liest genau dieses Muster. Er dient
  * als STARKE BASISLINIE für den Vergleich mit dem trainierten Modell (Thesis-Kapitel),
  * zum automatischen Vorlabeln echter Seiten (prelabel_real.py), die du danach nur noch korrigierst.
"""
import re

SECTIONS = {"funktionen": "roles", "lehrgebiete": "teach", "forschungsgebiete": "research", "publikationen": "pubs"}
ENTRY = re.compile(r"^(?P<authors>.+?)\s+\((?P<year>(?:19|20)\d\d)\)\s*:\s*(?P<rest>.+)$")
URL = re.compile(r"<(https?://[^>]+)>\s*$")


_TITLE = {"prof", "dr", "ing", "rer", "nat", "oec", "phil", "habil", "med", "pol", "mult", "dipl", "apl", "univ", "mba", "phd"}


def strip_titles(name: str) -> str:
    """'Prof. Dr.-Ing. habil. Jan Salmen' -> 'Jan Salmen'."""
    keep = [t for t in name.split() if not all(p in _TITLE for p in re.split(r"[.\-]+", t.lower()) if p)]
    return " ".join(keep)


def normalize_author(raw: str) -> str:
    """'Gaida, Daniel' -> 'Daniel Gaida'; 'Bernard, Olivier et al' -> 'Olivier Bernard'."""
    raw = re.sub(r"\s+et al\.?$", "", raw.strip())
    if "," in raw:
        last, first = (p.strip() for p in raw.split(",", 1))
        return f"{first} {last}".strip()
    return raw


def parse_entry(line: str) -> dict | None:
    line = line.strip()
    url = URL.search(line)
    link = url.group(1) if url else ""
    if url:
        line = line[: url.start()].strip()
    m = ENTRY.match(line)
    if not m:
        return None
    raw_authors = [a.strip() for a in m["authors"].split(";") if a.strip()]
    rest = m["rest"]
    title = re.split(r"\.\s", rest + " ", 1)[0].strip().rstrip(".")
    venue = ""
    vm = re.search(r"\bIn:\s*(.+?)(?:\.\s|\.$|$)", rest)
    if vm:
        venue = vm.group(1).strip()
    return {"title": title, "year": int(m["year"]), "venue": venue, "url": link,
            "authors": [normalize_author(a) for a in raw_authors], "authors_raw": raw_authors}


# Zweites Layout auf vielen Seiten (frei gepflegte Listen):
#     * Titel
#       Autor, Autor, 2016, Hg.: Verlag/Tagung
#       [Link](https://...)
_YEAR = re.compile(r"(?<![\w.-])((?:19|20)\d\d)(?![\w-])")
_INITIALS = re.compile(r"[A-ZÄÖÜ][\w'’\-]+(?:\s[a-z]{1,3})?,\s?[A-ZÄÖÜ]\.(?:\s?-?[A-ZÄÖÜ]\.)*")
_NOT_NAME = {"springer", "vieweg", "verlag", "ieee", "hg", "proceedings", "journal", "press", "gmbh", "university",
             "universität", "akademie", "kolloquium", "under", "review", "januar", "februar", "märz", "april", "mai", "juni",
             "juli", "august", "september", "oktober", "november", "dezember", "isbn", "vol", "band"}


def _looks_like_name(chunk: str) -> bool:
    words = chunk.split()
    if not 1 < len(words) <= 4 or any(c.isdigit() for c in chunk):
        return False
    if any(w.strip(".").lower() in _NOT_NAME for w in words):
        return False
    return all(w[:1].isupper() or w in {"van", "von", "de", "der", "den", "du", "la"} for w in words)


def _authors_two_line(line: str) -> list[str]:
    """Autorenteil einer Detailzeile (alles vor dem Jahr bzw. vor 'Hg.:'); wörtlich, damit er im Text markierbar bleibt."""
    cut = len(line)
    for m in (_YEAR.search(line), re.search(r",?\s*Hg\.:", line), re.search(r",?\s*In:", line)):
        if m:
            cut = min(cut, m.start())
    part = line[:cut].strip().rstrip(",").strip()
    part = re.sub(r"\s+(?:et al\.?|and others)$", "", part)
    if ";" in part:
        return [a for a in (re.sub(r"\s*\((?:Hg|Hrsg)\.\)$", "", a.strip()) for a in part.split(";"))
                if a and not any(c.isdigit() for c in a)]
    found = [] if re.match(r"[A-ZÄÖÜ]\.", part) else _INITIALS.findall(part)  # "T. Schneiders, ..." = Vorname zuerst
    if found:
        return found
    out = []
    for chunk in re.split(r",\s*|\s+and\s+|\s+und\s+", part):
        chunk = re.sub(r"\s+et al\.?$", "", chunk.strip())
        if not _looks_like_name(chunk):
            break  # Autoren stehen vorne; danach kommen Verlag/Tagung
        out.append(chunk)
    return out


def parse_two_line(first: str, details: list[str]) -> dict | None:
    title = first.strip().strip('"“”„«»').strip().rstrip(".").strip().strip('"“”„«»').lstrip("-–").strip()
    if re.match(r"(Alle|Automatisierte).*Publikation", title):  # Sammelverweis, keine Publikation
        return None
    details = [d for d in details if d and not d.startswith("[") and not re.match(r"https?://", d)]
    if not title or not details:
        return None
    joined = " ".join(details)
    year = _YEAR.search(joined)
    if not year:
        return None
    raw_authors = _authors_two_line(details[0])
    vm = re.search(r"Hg\.:\s*(.+)$", details[0])
    venue = vm.group(1).strip().rstrip(".").strip() if vm else ""
    return {"title": title, "year": int(year.group(1)), "venue": venue, "url": "",
            "authors": [normalize_author(a) for a in raw_authors], "authors_raw": raw_authors}


def pub_entries(block: list[str]) -> list[tuple[str, list[str]]]:
    """Aufzählungspunkt + eingerückte Folgezeilen (Leerzeilen dazwischen erlaubt) -> [(erste Zeile, [Folgezeilen])]."""
    out, open_ = [], False
    for line in block:
        if line.startswith("* "):
            out.append((line[2:], []))
            open_ = True
        elif open_ and line.startswith(" ") and line.strip():
            out[-1][1].append(line.strip())
        elif line.strip() and not line.startswith(" "):
            open_ = False  # z. B. Zwischenüberschrift "### Buch" beendet den Eintrag
    return out


def parse_pub(first: str, details: list[str]) -> dict | None:
    return parse_entry(first) or parse_two_line(first, details)


def bullets(block: list[str]) -> list[str]:
    out = []
    for line in block:
        if line.startswith("* "):
            text = line[2:].strip()
            text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # Markdown-Links -> Linktext
            out.append(text)
        elif out and line.startswith("  ") and line.strip():
            out[-1] += " " + line.strip()  # Fortsetzungszeile des Aufzählungspunkts
    return out


def parse_page(text: str) -> dict:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    lines = text.splitlines()
    name = strip_titles(next((l[2:].strip() for l in lines if l.startswith("# ")), ""))

    faculty = ""
    for l in lines[:25]:
        m = re.search(r"\[(Fakultät[^\]]+)\]", l)
        if m:
            faculty = m.group(1).strip()
            break

    blocks: dict[str, list[str]] = {}
    cur = None
    for l in lines:
        if l.startswith("## "):
            head = re.sub(r"^[+\s]+", "", l[3:]).strip().lower()
            cur = SECTIONS.get(head)
            if cur:
                blocks.setdefault(cur, [])  # manche Seiten haben den Abschnitt zweimal
        elif cur:
            blocks[cur].append(l)

    roles = bullets(blocks.get("roles", []))
    topics = [t for t in bullets(blocks.get("research", [])) + bullets(blocks.get("teach", []))
              if not t.startswith("Suchen Sie") and not t.startswith("Abschlussarbeiten")]
    topics = [re.split(r":\s", t, 1)[0].rstrip(":") for t in topics]  # "KI: Langtext" -> "KI"
    pubs = [e for e in (parse_pub(f, d) for f, d in pub_entries(blocks.get("pubs", []))) if e]
    return {
        "people": [{"name": name, "role": ", ".join(roles), "faculty": faculty, "expertise": list(dict.fromkeys(topics))}],
        "publications": pubs, "projects": [], "cooperations": [],
    }
