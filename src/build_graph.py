"""
Führt alle data/extracted/*.json zu einem Graphen zusammen und exportiert ihn
als web/data/graph.json im Format, das Cytoscape.js im Frontend erwartet.

Was hier zusätzlich zum reinen Zusammenfügen passiert:
  * Titel werden entfernt ("Prof. Dr.-Ing. habil. Jan Salmen" -> "Jan Salmen")
  * Namensvarianten werden zu einer Person zusammengeführt
    ("A. Stockem Novo" und "Alexander Stockem Novo")
  * Personen von der Mitgliederseite werden als Cluster-Mitglieder markiert
  * Personen, die gemeinsam an einem Projekt/einer Publikation beteiligt sind,
    bekommen eine Kooperationskante (type "collaboration")
"""
import json
import re
import unicodedata
from itertools import combinations
from pathlib import Path

import networkx as nx

from crawl import MEMBERS_URL, slugify

EXTRACTED_DIR = Path(__file__).resolve().parent.parent / "data" / "extracted"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "web" / "data" / "graph.json"

# Die Extraktion der Mitgliederseite ist die verlässliche Quelle für "wer gehört zum Cluster"
MEMBERS_STEM = slugify(MEMBERS_URL)

# Projekte/Publikationen mit mehr Beteiligten erzeugen keine Kooperationskanten
# zwischen allen Paaren (sonst entstehen riesige Knäuel durch Sammelpublikationen).
MAX_PARTICIPANTS_FOR_PAIRS = 15

# ---------------------------------------------------------------------------
# Namen bereinigen und vergleichen
# ---------------------------------------------------------------------------
_TITLE_PIECES = {
    "prof", "dr", "ing", "rer", "nat", "oec", "phil", "habil", "med", "pol",
    "mult", "dipl", "apl", "univ", "mba", "phd",
}
_TITLE_TOKENS = {"m.sc.", "b.sc.", "m.a.", "b.a.", "m.eng.", "b.eng.", "ph.d.", "h.c.", "m.sc", "b.sc"}
_INITIAL = re.compile(r"(\w\.)+|\w")  # "J.", "J.S.", "J" - aber nicht "Li"
# Wörter, an denen man erkennt, dass die LLM eine Organisation statt einer Person geliefert hat
_ORG_WORDS = {
    "cluster", "forschungscluster", "institut", "institute", "gmbh", "ag", "verein", "hochschule",
    "universitat", "university", "labor", "laboratory", "lab", "forum", "hub", "center", "centre",
    "zentrum", "consortium", "projekt", "project", "group", "gruppe", "team",
}


def clean_name(raw: str) -> str:
    """Entfernt akademische Titel: 'Prof. Dr.-Ing. habil. Jan Salmen' -> 'Jan Salmen'."""
    tokens = []
    for tok in raw.replace(" ", " ").split():
        tok = tok.strip(",;")
        if not tok:
            continue
        low = tok.lower()
        if low in _TITLE_TOKENS:
            continue
        pieces = [p for p in re.split(r"[.\-]+", low) if p]
        if pieces and all(p in _TITLE_PIECES for p in pieces):
            continue
        tokens.append(tok)
    return " ".join(tokens)


def fold(text: str) -> str:
    """Kleinschreibung ohne Akzente/Umlaute-Striche, damit 'Müller' == 'muller'."""
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).casefold()


def _is_initial(token: str) -> bool:
    return bool(_INITIAL.fullmatch(token))


def _compatible(a: dict, b: dict) -> bool:
    """Passen zwei Vornamen zusammen? Gleich, oder eine einzelne Initiale + passender Vorname.
    Mehrfach-Initialen wie 'J.S.' werden nur mit sich selbst zusammengeführt (zu unsicher)."""
    if a["first"] == b["first"]:
        return True
    if a["initial"] == b["initial"]:
        return False
    initial, full = (a, b) if a["initial"] else (b, a)
    return len(initial["first"]) == 1 and full["first"].startswith(initial["first"])


class PersonRegistry:
    """Ordnet jede Namensvariante einer stabilen Personen-ID zu."""

    def __init__(self) -> None:
        self.by_last: dict[str, list[dict]] = {}

    def resolve(self, raw_name: str) -> tuple[str, str] | None:
        """Gibt (person_id, anzeigename) zurück, oder None wenn es kein Personenname ist."""
        cleaned = clean_name(raw_name)
        tokens = cleaned.split()
        if len(tokens) < 2 or "(" in cleaned:  # Einzelwörter/Organisationen aussortieren
            return None
        if any(fold(t).strip(".,") in _ORG_WORDS for t in tokens):
            return None

        last = fold(tokens[-1])
        first_token = tokens[0]
        probe = {
            "first": fold(first_token).replace(".", ""),
            "initial": _is_initial(first_token),
        }
        candidates = self.by_last.setdefault(last, [])
        matches = [c for c in candidates if _compatible(c, probe)]

        entry = None
        if len(matches) == 1:
            entry = matches[0]
        elif len(matches) > 1:  # z.B. "M. Müller" passt auf mehrere -> nur exakte Treffer
            exact = [c for c in matches if c["first"] == probe["first"]]
            entry = exact[0] if exact else None

        if entry is None:
            slug = re.sub(r"\W+", "-", f"{last}-{probe['first']}").strip("-")
            entry = {"id": f"person:{slug}", **probe, "display": cleaned}
            candidates.append(entry)
        elif entry["initial"] and not probe["initial"]:
            # bisher nur Initiale bekannt, jetzt voller Vorname -> Anzeigename aufwerten
            entry.update(first=probe["first"], initial=False, display=cleaned)
        return entry["id"], entry["display"]


# ---------------------------------------------------------------------------
# Graph aufbauen
# ---------------------------------------------------------------------------
def _merge_list(existing: list, new: list) -> list:
    seen = {fold(x) for x in existing}
    return existing + [x for x in new if fold(x) not in seen and not seen.add(fold(x))]


def _add_person(g: nx.Graph, pid: str, display: str, member: bool = False, **attrs) -> None:
    if pid not in g:
        g.add_node(pid, type="person", label=display, member=False, faculty="", role="", expertise=[])
    node = g.nodes[pid]
    node["label"] = display
    node["member"] = node["member"] or member
    node["expertise"] = _merge_list(node["expertise"], attrs.get("expertise") or [])
    for key in ("faculty", "role"):
        value = (attrs.get(key) or "").strip()
        if value and (member or not node[key]):  # Angaben der Mitgliederseite haben Vorrang
            node[key] = value


def build_graph() -> nx.Graph:
    g = nx.Graph()
    registry = PersonRegistry()

    # Mitgliederseite zuerst: dort stehen die verlässlichen Vollnamen
    files = sorted(EXTRACTED_DIR.glob("*.json"), key=lambda p: (p.stem != MEMBERS_STEM, p.stem))

    for json_path in files:
        is_members_page = json_path.stem == MEMBERS_STEM
        data = json.loads(json_path.read_text(encoding="utf-8"))

        for person in data.get("people", []):
            resolved = registry.resolve(person.get("name", ""))
            if resolved:
                pid, display = resolved
                _add_person(g, pid, display, member=is_members_page, **person)

        for project in data.get("projects", []):
            name = " ".join((project.get("name") or "").split())
            if not name:
                continue
            proj_id = f"project:{fold(name)}"
            if proj_id not in g:
                g.add_node(proj_id, type="project", label=name, description="", kind=project.get("kind") or "project",
                           year=project.get("year"), venue=project.get("venue") or "", url="")
            node = g.nodes[proj_id]
            for key in ("year", "venue", "url"):  # dieselbe Publikation auf mehreren Seiten: fehlende Angaben ergänzen
                if project.get(key) and not node.get(key):
                    node[key] = project[key]
            if len(project.get("description") or "") > len(g.nodes[proj_id]["description"]):
                g.nodes[proj_id]["description"] = project["description"]
            for participant in project.get("participants", []):
                resolved = registry.resolve(participant)
                if resolved:
                    pid, display = resolved
                    _add_person(g, pid, display)
                    g.add_edge(pid, proj_id, type="member_of")

        for coop in data.get("cooperations", []):
            a, b = registry.resolve(coop.get("person_a", "")), registry.resolve(coop.get("person_b", ""))
            if not a or not b or a[0] == b[0]:
                continue
            _add_person(g, a[0], a[1])
            _add_person(g, b[0], b[1])
            contexts = g.edges[a[0], b[0]].get("contexts", []) if g.has_edge(a[0], b[0]) else []
            if coop.get("context"):
                contexts = _merge_list(contexts, [coop["context"]])
            g.add_edge(a[0], b[0], type="cooperation", contexts=contexts)

    return g


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def collaboration_edges(g: nx.Graph) -> dict[tuple[str, str], dict]:
    """Person-Person-Kanten: explizite Kooperationen + gemeinsame Projekte/Publikationen."""
    pairs: dict[tuple[str, str], dict] = {}

    def entry(a: str, b: str) -> dict:
        return pairs.setdefault(tuple(sorted((a, b))), {"projects": [], "items": [], "contexts": [], "explicit": False})

    for u, v, attrs in g.edges(data=True):
        if attrs.get("type") == "cooperation":
            e = entry(u, v)
            e["explicit"] = True
            e["contexts"] = _merge_list(e["contexts"], attrs.get("contexts", []))

    for node, attrs in g.nodes(data=True):
        if attrs["type"] != "project":
            continue
        people = [n for n in g.neighbors(node) if g.nodes[n]["type"] == "person"]
        if 2 <= len(people) <= MAX_PARTICIPANTS_FOR_PAIRS:
            for a, b in combinations(people, 2):
                e = entry(a, b)
                e["projects"].append(attrs["label"])
                e["items"].append({"title": attrs["label"], "url": attrs.get("url") or "", "year": attrs.get("year"),
                                   "venue": attrs.get("venue") or "", "kind": attrs.get("kind") or "project"})
    return pairs


def export_for_cytoscape(g: nx.Graph, out_path: Path) -> None:
    pairs = collaboration_edges(g)

    collab_degree: dict[str, int] = {}
    for a, b in pairs:
        collab_degree[a] = collab_degree.get(a, 0) + 1
        collab_degree[b] = collab_degree.get(b, 0) + 1

    nodes = []
    for n, attrs in g.nodes(data=True):
        data = {"id": n, **attrs}
        # Größe im Frontend: Personen nach Zahl der Kooperationspartner, Projekte nach Beteiligten
        data["degree"] = collab_degree.get(n, 0) if attrs["type"] == "person" else g.degree(n)
        nodes.append({"data": data})

    edges = [
        {"data": {"source": u, "target": v, "type": "member_of"}}
        for u, v, attrs in g.edges(data=True)
        if attrs.get("type") == "member_of"
    ]
    for (a, b), info in pairs.items():
        context = "; ".join(info["contexts"]) or (
            "Gemeinsam beteiligt an: " + "; ".join(info["projects"][:5]) if info["projects"] else ""
        )
        edges.append({"data": {
            "source": a, "target": b, "type": "collaboration",
            "explicit": info["explicit"],
            "weight": len(info["projects"]) + len(info["contexts"]),
            "projects": info["projects"], "items": info["items"], "context": context,
        }})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    members = sum(1 for n in nodes if n["data"].get("member"))
    print(f"Graph exportiert: {len(nodes)} Knoten ({members} Mitglieder), {len(edges)} Kanten -> {out_path}")


if __name__ == "__main__":
    graph = build_graph()
    export_for_cytoscape(graph, OUTPUT_PATH)
