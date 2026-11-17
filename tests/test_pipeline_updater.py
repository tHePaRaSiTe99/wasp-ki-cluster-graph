import json
import time
from pathlib import Path

import networkx as nx
import pytest

import pipeline
import state
import updater
from crawl import CrawlResult


def small_graph():
    g = nx.Graph()
    g.add_node("person:a", type="person", label="Anna Alpha", member=True)
    return g


@pytest.fixture
def out(tmp_path, monkeypatch):
    path = tmp_path / "web" / "data" / "graph.json"
    path.parent.mkdir(parents=True)
    monkeypatch.setattr(pipeline, "OUTPUT_PATH", path)
    monkeypatch.setattr(pipeline, "extract_files", lambda paths: [])
    monkeypatch.setattr(pipeline, "build_graph", small_graph)
    return path


def set_crawl(monkeypatch, changed=(), removed=()):
    monkeypatch.setattr(pipeline, "crawl", lambda *a, **kw: CrawlResult(changed=list(changed), removed=list(removed)))


def test_run_writes_graph_when_pages_changed(out, monkeypatch):
    set_crawl(monkeypatch, changed=[Path("personen_anna.alpha.md")])
    status = pipeline.run()
    assert "aktualisiert" in status
    assert [n["data"]["id"] for n in json.loads(out.read_text(encoding="utf-8"))["nodes"]] == ["person:a"]
    st = state.load()
    assert st["last_success"] >= st["last_attempt"]
    assert st["last_status"] == status


def test_run_without_changes_keeps_graph(out, monkeypatch):
    out.write_text("ALT", encoding="utf-8")
    set_crawl(monkeypatch)
    assert pipeline.run().startswith("Keine Änderungen")
    assert out.read_text(encoding="utf-8") == "ALT"


def test_run_rebuilds_on_removed_pages(out, monkeypatch):
    out.write_text("ALT", encoding="utf-8")
    set_crawl(monkeypatch, removed=["personen_bernd.beta"])
    pipeline.run()
    assert out.read_text(encoding="utf-8") != "ALT"


def test_empty_graph_keeps_old_file(out, monkeypatch):
    out.write_text("ALT", encoding="utf-8")
    set_crawl(monkeypatch, changed=[Path("x.md")])
    monkeypatch.setattr(pipeline, "build_graph", nx.Graph)
    with pytest.raises(RuntimeError):
        pipeline.run()
    assert out.read_text(encoding="utf-8") == "ALT"
    assert "last_success" not in state.load()
    assert "last_attempt" in state.load()


# ---------------------------------------------------------------------------
# updater._due: wann läuft die Pipeline?
# ---------------------------------------------------------------------------
H = 3600


@pytest.mark.parametrize("st, due", [
    ({}, True),                                                    # noch nie gelaufen
    ({"last_success": -2 * H, "last_attempt": -2 * H}, False),     # vor 2 h erfolgreich
    ({"last_success": -25 * H, "last_attempt": -25 * H}, True),    # Tag ist um
    ({"last_success": -48 * H, "last_attempt": -0.2 * H}, False),  # gerade fehlgeschlagen -> 1 h warten
    ({"last_success": -48 * H, "last_attempt": -2 * H}, True),     # Fehlschlag ist 2 h her
])
def test_due(st, due):
    now = time.time()
    state.save({k: now + v for k, v in st.items()})
    assert updater._due(interval_h=24, retry_h=1) is due
