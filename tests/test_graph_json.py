"""Prüft die committete web/data/graph.json, die GitHub Pages ausliefert."""
import json
from pathlib import Path

import pytest

GRAPH = Path(__file__).resolve().parent.parent / "web" / "data" / "graph.json"


@pytest.fixture(scope="module")
def graph():
    if not GRAPH.exists():
        pytest.skip("web/data/graph.json fehlt")
    return json.loads(GRAPH.read_text(encoding="utf-8"))


def test_node_ids_are_unique(graph):
    ids = [n["data"]["id"] for n in graph["nodes"]]
    assert len(ids) == len(set(ids))


def test_edges_point_to_existing_nodes(graph):
    types = {n["data"]["id"]: n["data"]["type"] for n in graph["nodes"]}
    for e in graph["edges"]:
        d = e["data"]
        assert d["source"] in types and d["target"] in types, d
        if d["type"] == "collaboration":
            assert types[d["source"]] == types[d["target"]] == "person", d


def test_has_cluster_members(graph):
    assert any(n["data"]["type"] == "person" and n["data"].get("member") for n in graph["nodes"])


def test_links_are_http(graph):
    for n in graph["nodes"]:
        url = n["data"].get("url") or ""
        assert url == "" or url.startswith(("http://", "https://")), n["data"]["id"]
