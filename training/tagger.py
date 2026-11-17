"""
Wort-Tagger statt Textgenerator: Das Modell liest die Seite und markiert jedes Wort mit einem Etikett
(PER = Personenname, PUB = Publikationstitel, PROJ = Projektname, YEAR, VENUE, ROLE, FAC = Fakultät, EXP = Schwerpunkt).
Es muss nichts abschreiben, kann also keine Namen erfinden, und die Antwort steht immer wörtlich im Text.
Das ist die klassische Named-Entity-Recognition-Architektur und mit wenig Daten und CPU trainierbar.

Aufbau: Zeichen-GRU je Wort (erkennt auch unbekannte Namen) + Wort-Embedding + BiLSTM über die Seite -> Etikett je Wort.
Die Zuordnung "wer gehört zu welchem Titel" ergibt sich aus der Zeile: alle Personen in der Zeile einer Publikation
sind ihre Autoren.
"""
import re

import torch
import torch.nn as nn

TYPES = ["PER", "ROLE", "FAC", "EXP", "PUB", "PROJ", "YEAR", "VENUE"]
TAGS = ["O"] + [f"{p}-{t}" for t in TYPES for p in "BI"]
TAG_ID = {t: i for i, t in enumerate(TAGS)}
TOKEN_RE = re.compile(r"\n|https?://[^\s<>)\]]+|/[\w\-./%]+|[^\W_]+(?:[-.:][^\W_]+)*|[^\w\s]|_")  # URL/Pfad = ein Token; "_" ist Satzzeichen
MAXC = 20


def tokenize(text: str):
    """-> Liste (token, start, ende); Zeilenumbrüche sind eigene Token "\\n"."""
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


# ---------------------------------------------------------------------------
# Trainingsetiketten aus den Fake-Daten (gold -> Etikett je Wort)
# ---------------------------------------------------------------------------
def gold_to_tags(text: str, gold: dict):
    toks = tokenize(text)
    words = [t[0] for t in toks]
    tags = ["O"] * len(toks)

    def mark(strings, typ):
        for s in strings:
            seq = [t[0] for t in tokenize(s)]
            n = len(seq)
            if not n:
                continue
            for i in range(len(words) - n + 1):
                if words[i:i + n] == seq and all(tags[j] == "O" for j in range(i, i + n)):
                    tags[i] = f"B-{typ}"
                    for j in range(i + 1, i + n):
                        tags[j] = f"I-{typ}"

    pubs, projs = gold.get("publications", []), gold.get("projects", [])
    p0 = gold["people"][0]
    mark([p["title"] for p in pubs], "PUB")
    mark([p["name"] for p in projs], "PROJ")
    mark([p0.get("faculty", "")], "FAC")
    mark([p.get("venue", "") for p in pubs], "VENUE")
    mark([r for r in p0.get("role", "").split(", ") if r], "ROLE")
    mark(sorted({p0["name"], *[a for p in pubs for a in p.get("authors_raw") or p["authors"]],
                 *[a for p in projs for a in p["participants"]]}, key=len, reverse=True), "PER")
    # Jahr: nur in Einträgen mit einer Publikation, das erste Vorkommen des Jahrs dieser Publikation
    year_of = {" ".join(t[0] for t in tokenize(p["title"])): str(p["year"]) for p in pubs if p.get("year")}
    ent = entry_ids(text, toks)
    for a, b in spans(tags, "PUB"):
        y = year_of.get(" ".join(words[a:b]))
        same = [j for j in range(len(words)) if ent[j] == ent[a]]
        if y and not any(tags[j] == "B-YEAR" for j in same):
            j = next((j for j in same if words[j] == y and tags[j] == "O"), None)
            if j is not None:
                tags[j] = "B-YEAR"
    mark(p0.get("expertise", []), "EXP")
    return toks, tags


def entry_ids(text: str, toks) -> list[int]:
    """Nummer des Listeneintrags je Token. Ein Eintrag beginnt mit einer nicht eingerückten Zeile; eingerückte
    Folgezeilen (auch nach Leerzeilen) gehören dazu. So zählen zweizeilige Angaben ("* Titel" / "  Autoren, 2016")
    als EINE Literaturangabe, einzeilige bleiben wie bisher je Zeile getrennt."""
    out, cur = [], 0
    for i, (w, a, _) in enumerate(toks):
        if i == 0 or toks[i - 1][0] == "\n":
            ls = text.rfind("\n", 0, a) + 1
            if ls < len(text) and text[ls] not in " \t\n":
                cur += 1
        out.append(cur)
    return out


# ---------------------------------------------------------------------------
# Modell
# ---------------------------------------------------------------------------
def encode(words, vocab):
    wid = torch.tensor([vocab.get(w.lower(), 1) for w in words])
    ch = torch.zeros(len(words), MAXC, dtype=torch.long)
    for i, w in enumerate(words):
        b = w.encode("utf-8")[:MAXC]
        ch[i, : len(b)] = torch.tensor(list(b)) + 1
    feat = torch.tensor([[w[:1].isupper(), w.isdigit(), w == "\n", w.isupper() and len(w) > 1] for w in words], dtype=torch.float)
    return wid, ch, feat


class Tagger(nn.Module):
    def __init__(self, vocab_size, wdim=48, cdim=32, chid=48, hid=128):
        super().__init__()
        self.wemb = nn.Embedding(vocab_size, wdim, padding_idx=0)
        self.cemb = nn.Embedding(258, cdim, padding_idx=0)
        self.cgru = nn.GRU(cdim, chid, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(wdim + 2 * chid + 4, hid, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.drop = nn.Dropout(0.25)
        self.out = nn.Linear(2 * hid, len(TAGS))
        self.word_dropout = 0.3  # Wörter zufällig "vergessen": das Modell soll auf Wortform/Kontext achten, nicht auf Namenslisten

    def forward(self, wid, ch, feat):
        B, T, L = ch.shape
        if self.training:
            wid = torch.where((torch.rand_like(wid, dtype=torch.float) < self.word_dropout) & (wid > 0), torch.ones_like(wid), wid)
        _, h = self.cgru(self.cemb(ch.view(B * T, L)))
        c = torch.cat([h[0], h[1]], dim=-1).view(B, T, -1)
        x = self.drop(torch.cat([self.wemb(wid), c, feat], dim=-1))
        y, _ = self.lstm(x)
        return self.out(self.drop(y))


# ---------------------------------------------------------------------------
# Etiketten -> Struktur
# ---------------------------------------------------------------------------
def spans(tags, typ):
    """[(start, ende_exklusiv)] aller Spannen eines Typs (tolerant: I- ohne B- beginnt eine neue Spanne)."""
    out, cur = [], None
    for i, t in enumerate(tags + ["O"]):
        if t == f"B-{typ}" or (t == f"I-{typ}" and cur is None):
            if cur:
                out.append(tuple(cur))
            cur = [i, i + 1]
        elif t == f"I-{typ}" and cur:
            cur[1] = i + 1
        else:
            if cur:
                out.append(tuple(cur))
            cur = None
    return out


def normalize_name(raw: str) -> str:
    """'Gaida, Daniel' -> 'Daniel Gaida'; 'Bernard, Olivier et al' -> 'Olivier Bernard'."""
    raw = re.sub(r"\s+et al\.?$", "", raw.strip())
    if "," in raw:
        last, first = (p.strip() for p in raw.split(",", 1))
        return f"{first} {last}".strip()
    return raw


def tags_to_record(text, toks, tags):
    """Etikettierte Seite -> Dict im Format des Projekts (people / publications / projects)."""
    def txt(a, b):
        return text[toks[a][1]: toks[b - 1][2]]

    line_of, ln = [], 0
    for t in toks:
        line_of.append(ln)
        if t[0] == "\n":
            ln += 1
    line_start = {0: 0}
    for i, t in enumerate(toks):
        if t[0] == "\n":
            line_start[line_of[i] + 1] = t[2]
    per = [(a, b, line_of[a]) for a, b in spans(tags, "PER")]
    first_line = next((line_of[i] for i, t in enumerate(toks) if t[0] != "\n"), 0)
    owner = next((txt(a, b) for a, b, l in per if l == first_line), "")

    def first(typ):
        s = spans(tags, typ)
        return txt(*s[0]) if s else ""

    roles = list(dict.fromkeys(txt(a, b) for a, b in spans(tags, "ROLE")))

    person = {"name": owner, "role": ", ".join(roles), "faculty": first("FAC"),
              "expertise": list(dict.fromkeys(txt(a, b).rstrip(":").strip() for a, b in spans(tags, "EXP")
                                              if not text[toks[a][1] - 1: toks[a][1]].isspace() or text[line_start[line_of[a]]: line_start[line_of[a]] + 1] != " "))}
    pubs, projs = [], []
    ent = entry_ids(text, toks)
    per_ent = [(a, b, ent[a]) for a, b in spans(tags, "PER")]
    year_ents = {ent[i] for i, t in enumerate(tags) if t == "B-YEAR" and re.fullmatch(r"(19|20)\d\d", toks[i][0])}
    seen = set()
    for a, b in spans(tags, "PUB"):
        e = ent[a]
        # Plausibilität: Eine Literaturangabe = EIN Listeneintrag mit Jahr; pro Eintrag zählt nur der erste Titel
        if e not in year_ents or e in seen:
            continue
        seen.add(e)
        year = next((toks[i][0] for i, t in enumerate(tags) if t == "B-YEAR" and ent[i] == e and re.fullmatch(r"(19|20)\d\d", toks[i][0])), None)
        venue = next((txt(x, y) for x, y in spans(tags, "VENUE") if ent[x] == e), "")
        pubs.append({"title": txt(a, b), "pos": toks[a][1], "year": int(year) if year else None, "venue": venue,
                     "authors": list(dict.fromkeys(normalize_name(txt(x, y)) for x, y, ee in per_ent if ee == e))})
    for a, b in spans(tags, "PROJ"):
        l = line_of[a]
        projs.append({"name": txt(a, b), "participants": list(dict.fromkeys(normalize_name(txt(x, y)) for x, y, ll in per if ll == l))})
    return {"people": [person], "publications": pubs, "projects": projs, "cooperations": []}


@torch.no_grad()
def predict(model, vocab, text):
    model.eval()
    toks = tokenize(text)
    if not toks:
        return toks, []
    wid, ch, feat = encode([t[0] for t in toks], vocab)
    logits = model(wid[None], ch[None], feat[None])[0]
    return toks, [TAGS[i] for i in logits.argmax(-1).tolist()]
