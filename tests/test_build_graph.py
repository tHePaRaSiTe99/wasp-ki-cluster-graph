import json

import pytest

import build_graph as bg


# ---------------------------------------------------------------------------
# Namen
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("Prof. Dr.-Ing. habil. Jan Salmen", "Jan Salmen"),
    ("Dr. rer. nat. Anna Alpha", "Anna Alpha"),
    ("M.Sc. Bernd Beta", "Bernd Beta"),
    ("Prof. Dr. Dr. h.c. Carl Gamma, MBA", "Carl Gamma"),
    ("Jan Salmen", "Jan Salmen"),
])
def test_clean_name_removes_titles(raw, expected):
    assert bg.clean_name(raw) == expected


def test_fold_ignores_case_and_accents():
    assert bg.fold("Müller") == bg.fold("MULLER") == "muller"
    assert bg.fold("José") == "jose"


def test_registry_merges_initial_with_full_first_name():
    reg = bg.PersonRegistry()
    pid_short, _ = reg.resolve("A. Stockem Novo")
    pid_full, display = reg.resolve("Alexander Stockem Novo")
    assert pid_short == pid_full
    assert display == "Alexander Stockem Novo"
    # danach wird auch die Kurzform mit dem vollen Namen angezeigt
    assert reg.resolve("A. Stockem Novo") == (pid_full, "Alexander Stockem Novo")


def test_registry_merges_titles_and_umlauts():
    reg = bg.PersonRegistry()
    assert reg.resolve("Prof. Dr. Jörg Müller")[0] == reg.resolve("Jörg Müller")[0]


def test_registry_keeps_different_first_names_apart():
    reg = bg.PersonRegistry()
    assert reg.resolve("Max Müller")[0] != reg.resolve("Maria Müller")[0]


def test_registry_ambiguous_initial_gets_own_id():
    reg = bg.PersonRegistry()
    max_id = reg.resolve("Max Müller")[0]
    maria_id = reg.resolve("Maria Müller")[0]
    m_id = reg.resolve("M. Müller")[0]  # passt auf beide -> nicht raten
    assert m_id not in (max_id, maria_id)


@pytest.mark.parametrize("raw", ["Jan", "", "Prof. Dr.", "THK-AI Forschungscluster", "Institut für Informatik",
                                 "Projekt (ABC) Team"])
def test_registry_rejects_non_persons(raw):
    assert bg.PersonRegistry().resolve(raw) is None


# ---------------------------------------------------------------------------
# Projekte / Publikationen wiederfinden
# ---------------------------------------------------------------------------
def test_head_and_acronym():
    assert bg._head("KI-Assistenz – Passgenaue Hilfe") == "ki assistenz"
    assert bg._head("REDiEE: Ein Projekt") == "rediee"
    assert bg._acronym("Games Technology Network (GTN)") == "gtn"
    assert bg._acronym("Ohne Abkürzung") == ""


def test_work_index_matches_project_variants():
    idx = bg.WorkIndex()
    idx.add("project:a", "KI-Assistenz – Passgenaue Hilfe", "project")
    idx.add("project:b", "Games Technology Network (GTN)", "project")
    assert idx.find("KI-Assistenz - Passgenaue Hilfe", "project") == "project:a"
    assert idx.find("KI-Assistenz: etwas anderes", "project") == "project:a"  # gleicher Kurzname
    assert idx.find("GTN", "project") == "project:b"  # Abkürzung
    assert idx.find("Ganz anderes Vorhaben", "project") is None


def test_work_index_publications_need_same_year_and_near_identical_title():
    idx = bg.WorkIndex()
    title = "Deep learning for graph based collaboration analysis"
    idx.add("publication:a", title, "publication", 2020)
    assert idx.find(title, "publication", 2021) == "publication:a"  # exakt gleich: Jahr egal
    assert idx.find("Deep learnin for graph based collaboration analysis", "publication", 2020) == "publication:a"
    assert idx.find("Deep learnin for graph based collaboration analysis", "publication", 2021) is None
    assert idx.find(title, "project") is None  # andere Art


def test_work_index_short_publication_titles_are_not_fuzzy_matched():
    idx = bg.WorkIndex()
    idx.add("publication:a", "Short title A", "publication", 2020)
    assert idx.find("Short title B", "publication", 2020) is None


# ---------------------------------------------------------------------------
# Ganzer Graph aus data/extracted/*.json
# ---------------------------------------------------------------------------
ANNA, BERND, CARL, EVA = "person:alpha-anna", "person:beta-bernd", "person:gamma-carl", "person:extern-eva"
PAPER = "publication:shared paper on graphs and collaboration"
PROJECT = "project:projekt x"


@pytest.fixture
def extracted_dir(tmp_path, monkeypatch):
    d = tmp_path / "extracted"
    d.mkdir()
    pages = {
        bg.MEMBERS_STEM: {"people": [{"name": "Prof. Dr. Anna Alpha"}, {"name": "Bernd Beta"}],
                          "projects": [], "cooperations": []},
        "personen_anna.alpha": {
            "people": [{"name": "Anna Alpha", "role": "Professorin", "faculty": "F10", "expertise": ["KI", "Graphen"]}],
            "projects": [
                {"name": "Shared Paper on Graphs and Collaboration", "kind": "publication", "year": 2021,
                 "url": "https://doi.org/10.1/x", "participants": ["A. Alpha", "Bernd Beta", "Carl Gamma"]},
                {"name": "Projekt X", "kind": "project", "description": "Unter der Leitung von Bernd Beta.",
                 "participants": ["Anna Alpha"]},
            ],
            "cooperations": [],
        },
        "personen_eva.extern": {
            "people": [{"name": "Eva Extern"}],
            "projects": [
                {"name": "Projekt X", "kind": "project", "participants": ["Eva Extern"]},
                {"name": "Nur Extern Vorhaben", "kind": "project", "participants": ["Eva Extern"]},
            ],
            "cooperations": [],
        },
    }
    for stem, data in pages.items():
        (d / f"{stem}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(bg, "EXTRACTED_DIR", d)
    return d


def test_build_graph_people_and_membership(extracted_dir):
    g = bg.build_graph()
    assert g.nodes[ANNA]["member"] and g.nodes[BERND]["member"]
    assert not g.nodes[CARL]["member"] and not g.nodes[EVA]["member"]
    assert g.nodes[ANNA]["label"] == "Anna Alpha"
    assert g.nodes[ANNA]["role"] == "Professorin"
    assert g.nodes[ANNA]["faculty"] == "F10"
    assert g.nodes[ANNA]["expertise"] == ["KI", "Graphen"]


def test_build_graph_works_and_edges(extracted_dir):
    g = bg.build_graph()
    assert g.nodes[PAPER]["year"] == 2021
    assert g.nodes[PAPER]["url"] == "https://doi.org/10.1/x"
    assert {n for n in g.neighbors(PAPER)} == {ANNA, BERND, CARL}  # "A. Alpha" == Anna Alpha
    assert g.has_edge(BERND, PROJECT)  # in der Beschreibung namentlich genannt
    assert g.has_edge(EVA, PROJECT)  # Externe hängt sich an ein Mitgliederprojekt an
    assert "project:nur extern vorhaben" not in g  # eigene Projekte Externer werden nicht angelegt


def test_collaboration_edges(extracted_dir):
    pairs = bg.collaboration_edges(bg.build_graph())
    anna_bernd = pairs[tuple(sorted((ANNA, BERND)))]
    assert sorted(anna_bernd["projects"]) == ["Projekt X", "Shared Paper on Graphs and Collaboration"]
    assert not anna_bernd["explicit"]
    assert tuple(sorted((CARL, EVA))) not in pairs


def test_large_works_create_no_pair_edges(monkeypatch):
    import networkx as nx
    g = nx.Graph()
    g.add_node("project:big", type="project", label="Big", kind="project")
    for i in range(bg.MAX_PARTICIPANTS_FOR_PAIRS + 1):
        g.add_node(f"person:{i}", type="person")
        g.add_edge(f"person:{i}", "project:big", type="member_of")
    assert bg.collaboration_edges(g) == {}


def test_export_for_cytoscape(extracted_dir, tmp_path):
    out = tmp_path / "web" / "graph.json"
    bg.export_for_cytoscape(bg.build_graph(), out)
    data = json.loads(out.read_text(encoding="utf-8"))
    ids = {n["data"]["id"] for n in data["nodes"]}
    assert {ANNA, BERND, CARL, EVA, PAPER, PROJECT} <= ids
    for e in data["edges"]:
        assert e["data"]["source"] in ids and e["data"]["target"] in ids
    collab = [e["data"] for e in data["edges"] if e["data"]["type"] == "collaboration"]
    anna_bernd = next(e for e in collab if {e["source"], e["target"]} == {ANNA, BERND})
    assert anna_bernd["weight"] == 2
    assert anna_bernd["context"].startswith("Gemeinsam beteiligt an: ")
    degree = {n["data"]["id"]: n["data"]["degree"] for n in data["nodes"]}
    assert degree[ANNA] == 3  # Bernd, Carl, Eva
