import json

import pytest

import extract

PAGE = """<!-- source: https://www.th-koeln.de/personen/anna.alpha/ -->

# Prof. Dr. Anna Alpha

Professorin an der Fakultät 10.

## Publikationen

* Ein Titel
  Anna Alpha, Bob Beispiel, 2020 <https://doi.org/10.1/abc>
* Zweiter Titel [Verlag](https://example.org/a) [Buch](https://doi.org/10.1/b) [Kapitel](https://doi.org/10.1/c)

## Projekte / Kooperationen

* **KI-Assistenz** – Hilfe für Lehrende
  Beschreibung Zeile zwei
  [Projektseite](https://example.org/ki)
* **Mitglied des Vereins XY**
* **Projekt \\_Beta**
  Text zum Projekt

## Kontakt

Raum 1
"""


@pytest.fixture
def no_model(monkeypatch):
    monkeypatch.setattr(extract, "_run_model", lambda text: {})


def test_pick_link_prefers_last_doi():
    assert extract._pick_link([]) == ""
    assert extract._pick_link(["https://a.org"]) == "https://a.org"
    assert extract._pick_link(["https://a.org", "http://dx.doi.org/1", "https://doi.org/2"]) == "https://doi.org/2"


def test_clean_text_and_h1():
    assert "source:" not in extract._clean_text(PAGE)
    assert extract._h1(PAGE) == "Prof. Dr. Anna Alpha"
    assert extract._h1("kein Titel") == ""


def test_entry_links():
    text = extract._clean_text(PAGE)
    urls = [url for _a, _b, url in extract._entry_links(text)]
    assert urls[:2] == ["https://doi.org/10.1/abc", "https://doi.org/10.1/c"]


def test_link_for_position():
    text = extract._clean_text(PAGE)
    entries = extract._entry_links(text)
    assert extract._link_for(entries, text.index("Zweiter Titel")) == "https://doi.org/10.1/c"
    assert extract._link_for(entries, None) == ""
    assert extract._link_for(entries, text.index("Professorin")) == ""


def test_project_sections():
    text = extract._clean_text(PAGE)
    (start, end), = extract._project_sections(text)
    assert text[start:].startswith("## Projekte")
    assert "KI-Assistenz" in text[start:end]
    assert "Raum 1" not in text[start:end]


def test_page_projects():
    projects = extract._page_projects(extract._clean_text(PAGE))
    assert [p["name"] for p in projects] == ["KI-Assistenz", "Projekt _Beta"]  # Mitgliedschaft fehlt
    ki = projects[0]
    assert ki["url"] == "https://example.org/ki"
    assert ki["description"] == "– Hilfe für Lehrende Beschreibung Zeile zwei"
    assert ki["kind"] == "project"


def test_page_projects_truncates_long_descriptions():
    text = "## Projekte\n\n* **Lang**\n  " + "x" * 700 + "\n"
    desc = extract._page_projects(text)[0]["description"]
    assert len(desc) == 601 and desc.endswith("…")


def test_extract_text_without_model(no_model):
    rec = extract.extract_text(PAGE)
    assert rec["people"] == [{"name": "Anna Alpha", "role": "", "faculty": "", "expertise": []}]
    assert rec["cooperations"] == []
    assert [p["name"] for p in rec["projects"]] == ["KI-Assistenz", "Projekt _Beta"]
    assert all(p["participants"] == ["Anna Alpha"] for p in rec["projects"])


def test_extract_text_with_model_output(monkeypatch):
    fake = {
        "people": [{"name": "Anna Alpha", "role": "Professorin", "faculty": "F10", "expertise": ["KI"]}],
        "publications": [
            {"title": "Ein Titel", "authors": ["Anna Alpha", "Bob Beispiel"], "year": 2020, "venue": "V",
             "url": "https://doi.org/10.1/abc"},
            {"title": "Ein  titel", "authors": ["Bob Beispiel"], "year": 2020},  # Dublette
        ],
    }
    monkeypatch.setattr(extract, "_run_model", lambda text: fake)
    rec = extract.extract_text(PAGE)
    assert rec["people"][0]["role"] == "Professorin"
    pubs = [p for p in rec["projects"] if p["kind"] == "publication"]
    assert len(pubs) == 1
    # die Seiteninhaberin ist immer beteiligt, aber nicht doppelt
    assert sorted(pubs[0]["participants"]) == ["Anna Alpha", "Bob Beispiel"]


def test_extract_files_writes_json(no_model, tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "EXTRACTED_DIR", tmp_path / "extracted")
    md = tmp_path / "personen_anna.alpha.md"
    md.write_text(PAGE, encoding="utf-8")
    (out,) = extract.extract_files([md])
    assert out == tmp_path / "extracted" / "personen_anna.alpha.json"
    assert json.loads(out.read_text(encoding="utf-8"))["people"][0]["name"] == "Anna Alpha"


def test_missing_checkpoint_disables_model(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "CHECKPOINT", tmp_path / "fehlt.pt")
    monkeypatch.setattr(extract, "_model", None)
    monkeypatch.setattr(extract, "_model_failed", False)
    assert extract._run_model(PAGE) == {}
