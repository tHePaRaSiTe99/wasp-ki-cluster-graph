import json

import pytest

import crawl
import state

BASE = "https://www.th-koeln.de/personen/"
ANNA_URL, BERND_URL = BASE + "anna.alpha/", BASE + "bernd.beta/"


def members_html(*people):
    links = "".join(f'<li><a href="{url}">{name}</a> <a href="{url}#kontakt">Mehr</a></li>' for url, name in people)
    return f'<html><body><a href="/impressum/">Impressum</a><ul>{links}</ul></body></html>'


def person_html(name, text="Lebenslauf"):
    return f"<html><body><nav>Menü</nav><main><h1>{name}</h1><p>{text}</p></main><script>alert(1)</script></body></html>"


@pytest.fixture
def site(tmp_path, monkeypatch):
    """Simuliertes Web: dict url -> html (None = Fehler). Leitet alle Ausgabeordner in tmp_path um."""
    pages = {
        crawl.MEMBERS_URL: members_html((ANNA_URL, "Prof. Dr. Anna Alpha"), (BERND_URL, "Bernd Beta")),
        ANNA_URL: person_html("Anna Alpha"),
        BERND_URL: person_html("Bernd Beta"),
    }
    monkeypatch.setattr(crawl, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(crawl, "EXTRACTED_DIR", tmp_path / "extracted")
    monkeypatch.setattr(crawl, "fetch", lambda url: pages.get(url))
    monkeypatch.setattr(crawl, "allowed_by_robots", lambda url, cache: True)
    return pages


def run(**kw):
    return crawl.crawl(delay_seconds=0, **kw)


def fake_extract(result):
    """Wie extract_files: zu jeder geänderten Seite gibt es danach eine JSON-Datei."""
    for md in result.changed:
        (crawl.EXTRACTED_DIR / f"{md.stem}.json").write_text("{}", encoding="utf-8")


@pytest.mark.parametrize("url, expected", [
    (ANNA_URL, "personen_anna.alpha"),
    (crawl.MEMBERS_URL, "thk-ai_mitglieder"),
    ("https://blogs.gm.fh-koeln.de/", "blogs_index"),
])
def test_slugify(url, expected):
    assert crawl.slugify(url) == expected


def test_find_person_links_dedupes_and_keeps_longest_text():
    html = members_html((ANNA_URL, "Prof. Dr. Anna Alpha")) + '<a href="/personen/bernd.beta/">Bernd Beta</a>'
    links = crawl.find_person_links(html, "https://www.th-koeln.de/mitglieder/")
    assert links == {ANNA_URL: "Prof. Dr. Anna Alpha", BERND_URL: "Bernd Beta"}


def test_html_to_markdown_drops_navigation_and_scripts():
    text = crawl.html_to_markdown(person_html("Anna Alpha"))
    assert "# Anna Alpha" in text and "Lebenslauf" in text
    assert "Menü" not in text and "alert" not in text


def test_first_crawl_stores_everything(site):
    result = run()
    assert sorted(p.name for p in result.changed) == ["personen_anna.alpha.md", "personen_bernd.beta.md"]
    md = (crawl.RAW_DIR / "personen_anna.alpha.md").read_text(encoding="utf-8")
    assert md.startswith(f"<!-- source: {ANNA_URL} -->")
    members = json.loads((crawl.EXTRACTED_DIR / f"{result.members_stem}.json").read_text(encoding="utf-8"))
    assert members["people"] == [{"name": "Prof. Dr. Anna Alpha"}, {"name": "Bernd Beta"}]
    assert set(state.load()["pages"]) == {ANNA_URL, BERND_URL}


def test_unchanged_pages_are_skipped(site):
    fake_extract(run())
    assert run().changed == []
    site[ANNA_URL] = person_html("Anna Alpha", "Neuer Lebenslauf")
    assert [p.name for p in run().changed] == ["personen_anna.alpha.md"]


def test_force_reprocesses_all(site):
    fake_extract(run())
    assert len(run(force=True).changed) == 2


def test_removed_member_is_deleted(site):
    fake_extract(run())
    site[crawl.MEMBERS_URL] = members_html((ANNA_URL, "Anna Alpha"))
    result = run()
    assert "personen_bernd.beta" in result.removed
    assert not (crawl.RAW_DIR / "personen_bernd.beta.md").exists()
    assert not (crawl.EXTRACTED_DIR / "personen_bernd.beta.json").exists()
    assert set(state.load()["pages"]) == {ANNA_URL}


def test_fetch_error_keeps_old_data(site):
    fake_extract(run())
    site[BERND_URL] = None
    result = run()
    assert result.removed == []
    assert (crawl.EXTRACTED_DIR / "personen_bernd.beta.json").exists()


def test_empty_members_page_aborts_without_deleting(site):
    fake_extract(run())
    site[crawl.MEMBERS_URL] = "<html><body>Wartungsarbeiten</body></html>"
    with pytest.raises(RuntimeError):
        run()
    assert (crawl.EXTRACTED_DIR / "personen_bernd.beta.json").exists()


def test_unreachable_members_page_raises(site):
    site[crawl.MEMBERS_URL] = None
    with pytest.raises(RuntimeError):
        run()
