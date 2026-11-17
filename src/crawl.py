"""
Crawler für den THK-AI Forschungscluster mit Änderungserkennung.

Ablauf:
  1. Mitgliederliste laden (MEMBERS_URL), alle Links auf Personenseiten sammeln
  2. Jede Personenseite laden, in Markdown umwandeln und speichern (data/raw/*.md)
  3. Pro Seite einen Hash merken (data/state.json): so weiß der Rest der Pipeline,
     WELCHE Seiten sich seit dem letzten Lauf geändert haben
  4. Die Mitgliederliste wird OHNE Sprachmodell ausgewertet (Linktext = Name) und direkt als
     data/extracted/<mitglieder>.json geschrieben - das ist deterministisch und fehlerfrei
  5. Seiten, die nicht mehr verlinkt sind (Person ausgetreten), werden samt Extraktion gelöscht
"""
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urldefrag, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

import state as state_store

# --- Konfiguration -----------------------------------------------------------
MEMBERS_URL = "https://blogs.gm.fh-koeln.de/thk-ai/mitglieder/"
PERSON_URL_PATTERN = "/personen/"  # nur Links mit diesem Teil im Pfad sind Personenseiten
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
EXTRACTED_DIR = ROOT / "data" / "extracted"
HEADERS = {"User-Agent": "THK-AI-Graph-Crawler/0.2 (Abschlussarbeit TH Köln)"}


@dataclass
class CrawlResult:
    members_stem: str = ""
    changed: list[Path] = field(default_factory=list)   # Markdown-Dateien, die neu oder geändert sind
    current_stems: set[str] = field(default_factory=set)
    removed: list[str] = field(default_factory=list)


def fetch(url: str) -> str | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        if "charset" not in resp.headers.get("content-type", "").lower():
            resp.encoding = "utf-8"  # requests rät sonst ISO-8859-1 und macht aus "ä" ein "Ã¤"
        return resp.text
    except requests.RequestException as e:
        print(f"  [!] Fehler bei {url}: {e}")
        return None


def allowed_by_robots(url: str, cache: dict) -> bool:
    """Prüft robots.txt der jeweiligen Domain. Ist sie nicht lesbar, wird erlaubt."""
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    if base not in cache:
        rp = RobotFileParser()
        rp.set_url(f"{base}/robots.txt")
        try:
            rp.read()
        except Exception:
            rp = None
        cache[base] = rp
    rp = cache[base]
    return True if rp is None else rp.can_fetch(HEADERS["User-Agent"], url)


def html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    return md(str(main), heading_style="ATX")


def find_person_links(html: str, base_url: str) -> dict[str, str]:
    """{url: Linktext} für alle Links auf Personenprofile, ohne Anker, ohne Duplikate.
    Der Linktext ist auf der Mitgliederseite der Name der Person."""
    soup = BeautifulSoup(html, "html.parser")
    found: dict[str, str] = {}
    for a in soup.find_all("a", href=True):
        url, _fragment = urldefrag(urljoin(base_url, a["href"]))
        if PERSON_URL_PATTERN not in urlparse(url).path:
            continue
        text = " ".join(a.get_text(" ", strip=True).split())
        if len(text) > len(found.get(url, "")):
            found[url] = text
    return found


def slugify(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_") or "index"
    return f"{parsed.netloc.split('.')[0]}_{path}" if path == "index" else path


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def crawl(start_url: str = MEMBERS_URL, max_pages: int = 80, delay_seconds: float = 1.0, force: bool = False) -> CrawlResult:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    result = CrawlResult(members_stem=slugify(start_url))
    robots_cache: dict = {}
    st = state_store.load()
    hashes: dict[str, str] = st.get("pages", {})

    members_html = fetch(start_url)
    if members_html is None:
        raise RuntimeError(f"Startseite {start_url} nicht erreichbar")
    people = find_person_links(members_html, start_url)
    if not people:
        # Sicherheitsnetz: Seite geändert/leer geliefert -> lieber abbrechen als alle Mitglieder zu löschen
        raise RuntimeError("Auf der Mitgliederseite wurden keine Personenlinks gefunden - Abbruch, nichts wird gelöscht.")
    print(f"{len(people)} Personenseiten gefunden (Muster: '{PERSON_URL_PATTERN}')")

    # Mitgliederliste ohne LLM: Linktext = Name
    members = [{"name": name} for name in people.values() if len(name.split()) >= 2]
    (EXTRACTED_DIR / f"{result.members_stem}.json").write_text(
        json.dumps({"people": members, "projects": [], "cooperations": []}, ensure_ascii=False, indent=2), encoding="utf-8")
    result.current_stems.add(result.members_stem)

    def store(url: str, html: str) -> None:
        text = f"<!-- source: {url} -->\n\n{html_to_markdown(html)}"
        stem = slugify(url)
        result.current_stems.add(stem)
        h = _hash(text)
        if force or hashes.get(url) != h or not (EXTRACTED_DIR / f"{stem}.json").exists():
            (RAW_DIR / f"{stem}.md").write_text(text, encoding="utf-8")
            result.changed.append(RAW_DIR / f"{stem}.md")
        hashes[url] = h

    for url in list(people)[:max_pages]:
        if not allowed_by_robots(url, robots_cache):
            print(f"Übersprungen (robots.txt): {url}")
            continue
        print(f"Lade: {url}")
        page_html = fetch(url)
        if page_html is None:
            result.current_stems.add(slugify(url))  # Fehler heute != Person weg: alte Daten behalten
            continue
        store(url, page_html)
        time.sleep(delay_seconds)  # höflich bleiben

    # Nicht mehr verlinkte Seiten samt Extraktion entfernen
    for folder, suffix in ((RAW_DIR, ".md"), (EXTRACTED_DIR, ".json")):
        for p in folder.glob(f"*{suffix}"):
            if p.stem not in result.current_stems:
                p.unlink()
                result.removed.append(p.stem)
    st["pages"] = {u: h for u, h in hashes.items() if u in people}
    state_store.save(st)
    print(f"Fertig. {len(result.changed)} neu/geändert, {len(set(result.removed))} entfernt.")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=MEMBERS_URL)
    parser.add_argument("--max-pages", type=int, default=80)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    crawl(args.url, max_pages=args.max_pages, force=args.force)
