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
                blocks[cur] = []
        elif cur:
            blocks[cur].append(l)

    roles = bullets(blocks.get("roles", []))
    topics = [t for t in bullets(blocks.get("research", [])) + bullets(blocks.get("teach", []))
              if not t.startswith("Suchen Sie") and not t.startswith("Abschlussarbeiten")]
    topics = [re.split(r":\s", t, 1)[0].rstrip(":") for t in topics]  # "KI: Langtext" -> "KI"
    pubs = [e for e in (parse_entry(l[2:]) for l in blocks.get("pubs", []) if l.startswith("* ")) if e]
    return {
        "people": [{"name": name, "role": ", ".join(roles), "faculty": faculty, "expertise": list(dict.fromkeys(topics))}],
        "publications": pubs, "projects": [], "cooperations": [],
    }
