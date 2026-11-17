"""
Erzeugt synthetische Trainings-/Testdaten: Personenseiten im Stil der THK-AI-Seite (Markdown)
zusammen mit der richtigen Lösung ("gold") im Format von GRAPH_SCHEMA aus src/extract.py.

Alles ist ausgedacht. Die Namen sind absichtlich in drei getrennte Töpfe aufgeteilt:
  train / val / test haben KEINE gemeinsamen Nachnamen. Ein Modell kann die Testnamen also
  nicht auswendig gelernt haben - es muss sie wirklich aus dem Text herauslesen.

Aufruf:  python fake_data.py            (erzeugt data/fake/{train,val,test}.jsonl)
         python fake_data.py --n-train 8000
"""
import argparse
import json
import random
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "data" / "fake"

FIRST = ["Anna", "Jonas", "Lea", "Felix", "Mara", "Tim", "Sophie", "Lukas", "Nina", "Paul", "Clara",
         "Jan", "Eva", "Max", "Laura", "Ben", "Julia", "Finn", "Marie", "Noah", "Katrin", "Stefan",
         "Petra", "Oliver", "Sabine", "Dirk", "Heike", "Tobias", "Nadine", "Markus"]
LAST = ["Aberle", "Brandt", "Cramer", "Dietz", "Eckert", "Fuchs", "Gerlach", "Hartung", "Imhof", "Jäger",
        "Kessler", "Lorenz", "Maurer", "Neumann", "Oswald", "Pfeiffer", "Quast", "Reuter", "Seidel",
        "Thiele", "Ullrich", "Vogel", "Winkler", "Ziegler", "Albrecht", "Becker", "Conrad", "Dreyer",
        "Ebert", "Fiedler", "Graf", "Hahn", "Jansen", "Keller", "Lange", "Meyer", "Noll", "Otto",
        "Probst", "Richter", "Schade", "Trapp", "Unger", "Voss", "Wagner", "Zimmer", "Arndt", "Busch",
        "Claus", "Dorn", "Engel", "Fink", "Götz", "Haas", "Ihle", "Jung", "Kraus", "Lenz", "Moll", "Nagel"]
TITLES = ["", "", "Prof. Dr. ", "Prof. Dr.-Ing. ", "Dr. ", "M.Sc. "]
ROLES = ["Professor:in", "Wissenschaftliche:r Mitarbeiter:in", "Doktorand:in", "Laboringenieur:in", "Postdoc"]
FACULTIES = ["Fakultät für Informatik und Ingenieurwissenschaften", "Fakultät für Angewandte Naturwissenschaften",
             "Fakultät für Anlagen, Energie- und Maschinensysteme", "Fakultät für Wirtschafts- und Rechtswissenschaften",
             "Fakultät für Informations-, Medien- und Elektrotechnik"]
EXPERTISE = ["Maschinelles Lernen", "Computer Vision", "Robotik", "Sprachmodelle", "Reinforcement Learning",
             "Datenanalyse", "Regelungstechnik", "Bioinformatik", "Optimierung", "Mensch-Maschine-Interaktion",
             "Zeitreihenanalyse", "Bildverarbeitung", "Embedded AI", "Erklärbare KI"]
T_A = ["Lernende", "Robuste", "Effiziente", "Erklärbare", "Skalierbare", "Adaptive", "Verteilte", "Probabilistische"]
T_B = ["Regelung", "Bilderkennung", "Prognose", "Optimierung", "Segmentierung", "Fehlererkennung", "Navigation",
       "Sprachverarbeitung", "Planung", "Klassifikation"]
T_C = ["in der Fertigung", "für Windkraftanlagen", "mit neuronalen Netzen", "bei geringer Datenlage",
       "im medizinischen Umfeld", "für mobile Roboter", "unter Echtzeitbedingungen", "mit Transformern"]
VENUES = ["IEEE Access", "ICML Workshop", "Automatisierungstechnik", "Sensors", "KI 2023", "ECML PKDD",
          "at - Automatisierungstechnik", "Journal of Machine Learning Applications", "NeurIPS Workshop", "ISPRS Annals"]
PROJ_A = ["KI", "Smart", "Open", "Edge", "Auto", "Robo", "Data"]
PROJ_B = ["Wind", "Care", "Fabrik", "Pilot", "Labor", "Mobil", "Vision", "Twin"]
NOISE = [
    "Die Fakultät für Informatik gehört zur Technischen Hochschule Köln.",
    "Sprechzeiten nach Vereinbarung.",
    "Kontakt: siehe Impressum. Barrierefreiheit | Datenschutz | Sitemap",
    "Mitglied im THK-AI Forschungscluster.",
    "Die TH Köln ist eine der größten Hochschulen für angewandte Wissenschaften in Deutschland.",
]


def split_last_names(seed: int):
    rng = random.Random(seed)
    names = LAST[:]
    rng.shuffle(names)
    n = len(names)
    return {"train": names[: int(n * 0.6)], "val": names[int(n * 0.6): int(n * 0.8)], "test": names[int(n * 0.8):]}


SYL = ["ba", "ber", "din", "gar", "hal", "ke", "lin", "mor", "nad", "ost", "pel", "rau", "sta", "tel", "ul",
       "ven", "wal", "zim", "bro", "fer", "gru", "hei", "kol", "lau", "mei", "nu", "sche", "tro", "wi", "dra"]
END = ["mann", "berg", "er", "ing", "stein", "hoff", "feld", "s", "dorf", "bach", "ke", "el"]


def made_up_name(rng):
    return "".join(rng.choice(SYL) for _ in range(rng.randint(1, 2))).capitalize() + rng.choice(END)


def person(rng, lasts):
    """Teils echte Namen (je Split getrennt), überwiegend frei erfundene: so kann das Modell Namen nicht
    auswendig lernen, sondern muss sie aus dem Text abschreiben."""
    first = rng.choice(FIRST) if rng.random() < 0.6 else made_up_name(rng)
    last = rng.choice(lasts) if rng.random() < 0.3 else made_up_name(rng)
    return f"{first} {last}"



# ---------------------------------------------------------------------------
# Layout 5: so sieht eine ECHTE TH-Köln-Personenseite aus (nach den Beispielseiten nachgebaut)
# ---------------------------------------------------------------------------
INSTITUTES = ["Institut für Informatik (INF)", "Institute of Computer and Communication Technology (ICCT)",
              "Institut für Medien- und Phototechnik (IMP)", "Institut für Data Science, Engineering, and Analytics (IDE+A)",
              "Institut für Produktentwicklung und Konstruktionstechnik (IPK)"]
CAMPUS = [("Gummersbach", "Steinmüllerallee 1", "51643"), ("Deutz", "Betzdorfer Straße 2", "50679"), ("Südstadt", "Claudiusstraße 1", "50678")]
FUNCTIONS = ["LaborleiterIn", "Prüfungsausschussvorsitzende/r", "Studiendekan/in", "Institutsleiter/in", "Sprecher/in", "Gleichstellungsbeauftragte/r"]
TEACH = ["Künstliche Intelligenz", "Algorithmik", "Machine Learning, Deep Learning und Künstliche Intelligenz", "Cyber-Physische Systeme",
         "Bildverarbeitung und Computer Vision", "MLOps", "Quantencomputing", "Mensch-zentrierte Künstliche Intelligenz",
         "Regelungstechnik", "Datenbanken", "Softwaretechnik", "Embedded Systems"]
RESEARCH = ["Verteilte Künstliche Intelligenz", "Physische Künstliche Intelligenz", "Cyber-Physische Systeme", "Erklärbare KI",
            "Optimierung", "Zeitreihenanalyse", "Robotik", "Bioinformatik", "Sprachmodelle"]
PUBLISHERS = ["Würzburg: Vogel Communications Group GmbH & Co. KG", "Köln: Technische Hochschule Köln", "Berlin: Springer Vieweg", "Wiesbaden: Springer Gabler"]
JOURNALS = ["IFAC-PapersOnLine", "Renewable and Sustainable Energy Reviews", "Automatisierungstechnik : AT", "Energy, Sustainability and Society",
            "Machine Learning for Cyber Physical Systems", "IEEE Access", "Sensors", "Reviews in Environmental Science and Bio-Technology"]


# Große, themenneutrale Wortlisten für abwechslungsreiche Titel (damit das Modell Titel an der STELLE erkennt,
# nicht an bekannten Wörtern). Wörter, die auf den echten Beispielseiten vorkommen, werden bewusst weggelassen.
EN_W = """adaptive analysis architecture assessment automated autonomous benchmark bounds calibration causal classification
clustering compression computation consensus control convergence data decentralised deep design detection diffusion
dimensionality distributed dynamic efficient embedding ensemble estimation evaluation exploration fault federated
filtering forecasting framework fusion generative graph heterogeneous hierarchical identification inference
integration interpretable kernel knowledge latent learning lightweight localisation logic mapping mechanisms
modelling monitoring multi-agent network neural noise nonlinear online optimal parallel perception planning
policy prediction privacy probabilistic quantum reasoning recognition reconstruction regression reinforcement
reliability representation robust sampling scalable scheduling search segmentation semantic sensor sequence
signal simulation sparse stability statistical stochastic structure supervised surrogate synthesis tracking
transfer transformer uncertainty unsupervised validation variational vision wireless workflow""".split()
DE_W = """Analyse Anwendung Auswertung Bewertung Datenqualität Entscheidung Erkennung Fertigung Gebäude Industrie
Instandhaltung Kommunikation Lernverfahren Messung Mobilität Modellierung Netzwerke Planung Prognose Prozess
Qualitätssicherung Regelung Sensorik Simulation Steuerung Strategie Systeme Überwachung Verfahren Vorhersage
Wartung Werkzeug Zuverlässigkeit""".split()
_LEAK = {"control", "data", "deep", "dynamic", "estimation", "learning", "online", "vision", "Systeme"}
EN_W = [w for w in EN_W if w not in _LEAK]
DE_W = [w for w in DE_W if w not in _LEAK]
EN_PAT = ["{a} {b} for {c} {d}", "A {a} Approach to {b} {c}", "{A} {b} in {c} {d}: A Review", "On the {b} of {a} {c}", "Towards {a} {b} using {c} {d}",
          "{A} {b}", "{A} and {b} {c} with {d} {e}", "{A}-based {b} for {c} Systems", "How {a} {b} Improves {c}", "{A} {b} {c} under {d} {e}"]


def rich_title(rng):
    if rng.random() < 0.3:
        w = rng.sample(DE_W, 4)
        t = rng.choice(["{0} und {1} {2}", "{0} von {1} mit {2} {3}", "{0} für {1} {2}", "{0} : {1} {2}"]).format(*w)
    else:
        w = rng.sample(EN_W, 5)
        t = rng.choice(EN_PAT).format(a=w[0], b=w[1], c=w[2], d=w[3], e=w[4], A=w[0].capitalize())
    if rng.random() < 0.4:  # Großschreibung wie in englischen Titeln
        t = " ".join(x if x in ("a", "of", "for", "to", "in", "and", "on", "with", "using", "under", "how") else x.capitalize() for x in t.split())
    if rng.random() < 0.2:
        t = t + " : " + " ".join(rng.sample(EN_W, rng.randint(2, 4))).capitalize()
    if rng.random() < 0.1:
        t = t.replace(" ", " 4.0 ", 1)
    return t[0].upper() + t[1:]


def punct_title(rng, t):
    """Titel mit den Satzzeichen echter Literaturlisten: Fragezeichen, Prozent, Klammer-Abkürzungen, Gedankenstrich-
    Untertitel, Anführungszeichen. Genau an solchen Stellen hat das Modell Titel früher zu kurz abgeschnitten."""
    r = rng.random()
    if r < 0.15:
        t = t + "?"
    elif r < 0.25:
        t = f"{t} ({''.join(w[0] for w in rng.sample(EN_W, rng.randint(2, 4))).upper()})"
    elif r < 0.35:
        t = f"{t} – {' '.join(rng.sample(EN_W + DE_W, rng.randint(2, 5))).capitalize()}"
    elif r < 0.42:
        t = t.replace(" ", f" {rng.randint(10, 100)} % ", 1)
    elif r < 0.50:
        t = f"„{rng.choice(DE_W)}“ – {t}"
    elif r < 0.56:
        t = f"{t}: {rng.choice(['Status quo', 'A Case Study', 'Teil 1', 'Lessons Learned', 'Herausforderungen und Potenziale'])}"
    elif r < 0.60:
        t = f"{t}, {rng.choice(DE_W)} & {rng.choice(DE_W)}"
    return t


MONTHS = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"]


def two_line_entry(rng, p, co):
    """Zweites echtes Layout (frei gepflegte Listen):
        * Titel
          Autoren, 2016, Hg.: Verlag
          [Link](https://...)
    Gibt die Zeilen zurück und setzt authors_raw/venue so, dass sie WÖRTLICH im Text stehen."""
    style = rng.randint(0, 4)
    if style == 0:    # Mai, Vanessa; Neef, Caterina; Richert, Anja, 2022, Hg.: ...
        raw = [f"{l}, {f}" for f, l in co]; a = "; ".join(raw)
    elif style == 1:  # Salmen, J. ; Suttorp, T. ; ...
        raw = [f"{l}, {f[0]}." for f, l in co]; a = " ; ".join(raw)
    elif style == 2:  # Kamau, E., Waschke, J.S., ...
        raw = [f"{l}, {f[0]}." for f, l in co]; a = ", ".join(raw)
    else:             # Andreas Schwenk, Bert Randerath / T. Schneiders, A. Löschel
        raw = [f"{f} {l}" if style == 3 else f"{f[0]}. {l}" for f, l in co]; a = ", ".join(raw)
    date = str(p["year"]) if rng.random() < 0.75 else rng.choice([f"{rng.choice(MONTHS)} {p['year']}", f"{rng.randint(1, 28):02d} {rng.choice(MONTHS)} {p['year']}"])
    venue = p["venue"] or rng.choice(PUBLISHERS + ["IEEE Xplore", "Springer", "CIRP", "Energieagentur NRW"])
    v = rng.random()
    t = p["title"]
    first = rng.choice([t, t, t, f'"{t}"', f"{t}."]) + "  "
    if v < 0.6:       # Jahr und Hg. in einer Zeile
        lines = [f"* {first}", f"  {a}, {date}, Hg.: {venue}"]
    elif v < 0.8:     # Jahr erst in der dritten Zeile
        lines = [f"* {first}", f"  {a}, Hg.: {venue}  ", f"  {rng.choice(JOURNALS)}, Volume {rng.randint(1, 80)}, {p['year']}, Pages {rng.randint(1, 300)}-{rng.randint(301, 600)}"]
    else:             # ohne Hg.: Zeitschrift und Jahr am Ende
        venue = ""
        lines = [f"* {first}", f"  {a}, {rng.choice(JOURNALS)} {rng.randint(1, 80)}, pp. {rng.randint(1, 30)}-{rng.randint(31, 60)}, {p['year']}"]
    if rng.random() < 0.4:
        url = p["_url"]
        lines += ["", f'  [{rng.choice([url, "Link", "Link Full Paper", "DOI"])}]({url} "Sie werden auf eine externe Webseite weitergeleitet.")']
    # Lösung: Autorennamen wie die Auswertung sie vergleicht ("Salmen, J." -> "J. Salmen")
    p["authors_raw"], p["venue"] = raw, venue
    p["authors"] = [f"{r.split(', ', 1)[1]} {r.split(', ', 1)[0]}" if ", " in r else r for r in raw]
    return lines


def th_name(rng, lasts):
    first = rng.choice(FIRST) if rng.random() < 0.6 else made_up_name(rng)
    if rng.random() < 0.15:
        first += rng.choice(["", " " + rng.choice("ABCDEFGHJKLMPRST") + "."])
    last = rng.choice(lasts) if rng.random() < 0.3 else made_up_name(rng)
    return first, last


def make_th_example(rng, lasts):
    mf, ml = th_name(rng, lasts)
    me = f"{mf} {ml}"
    title = rng.choice(["Prof. Dr. ", "Prof. Dr.-Ing. ", "Prof. Dr. rer. nat. ", "Prof. Dr.-Ing. habil. ", "Dr. "])
    fac = rng.choice(FACULTIES)
    inst = rng.choice(INSTITUTES)
    camp, street, plz = rng.choice(CAMPUS)
    roles = rng.sample(FUNCTIONS, rng.choice([0, 0, 1, 2]))
    teach = rng.sample(TEACH, rng.randint(1, 5))
    research = rng.sample(RESEARCH, rng.choice([0, 1, 2, 3]))
    pubs = []
    for _ in range(rng.choice([0, 0, 1, 2, 3, 5, 8])):
        co = [th_name(rng, lasts) for _ in range(rng.randint(0, 6))]
        co.insert(rng.randint(0, len(co)), (mf, ml))
        t = rich_title(rng) if rng.random() < 0.85 else f"{rng.choice(T_A)} {rng.choice(T_B)} {rng.choice(T_C)}"
        t = punct_title(rng, t)
        year = rng.randint(2008, 2025)
        kind = rng.random()
        venue = rng.choice(JOURNALS) if kind < 0.7 else ""
        raw = [f"{l}, {f}" for f, l in co]
        pubs.append({"title": t, "year": year, "venue": venue, "authors": [f"{f} {l}" for f, l in co], "authors_raw": raw,
                     "_tail": ((f"In: {venue}. Vol. {rng.randint(1, 80)}, S. {rng.randint(1, 400)} - {rng.randint(401, 500)}." if rng.random() < 0.6 else
                                f"In: {venue}. {rng.randint(2, 30)}. International Conference on {rng.choice(EN_W).capitalize()}; {rng.choice(['Berlin', 'Lisbon', 'Taipei', 'Wien'])}; "
                                f"{rng.randint(1, 28):02d}.0{rng.randint(1, 9)}.-{rng.randint(1, 28):02d}.0{rng.randint(1, 9)}.{year}., S. {rng.randint(1, 90)} - {rng.randint(91, 200)}.") if venue
                               else f"{rng.choice(PUBLISHERS)}."),
                     "_flag": rng.choice(["", "", " (peer-reviewed)", " (Open Access)", " (peer-reviewed/Open Access)"]),
                     "_url": rng.choice(["https://doi.org/10.%d/%s" % (rng.randint(1000, 1999), made_up_name(rng).lower()),
                                         "https://bibliografie.th-koeln.de/servlets/DozBibEntryServlet?mode=show&id=bibthk_mods_%08d" % rng.randint(1, 99999999)])})
    L = [f"# {title}{me}", "", rng.choice(["Dr.", "Dr.-Ing.", "Dr. rer. nat.", ""]), "", f"[{fac}](/{made_up_name(rng).lower()}/) ", ""]
    if rng.random() < 0.8:
        L += [inst, ""]
    L += [f"![{title}{me}](/pseimg/{rng.getrandbits(80):020x}.png)", "", f"Campus {camp}  ", street + "  ", f"{plz} Köln  ", "[Postanschrift](/hochschule/standorte_3914.php)", "", "  ", "",
          f"* ![Telefon](/img/icons/contact-telefon.svg)+49 221-8275-{rng.randint(1000, 9999)}",
          f"* ![Email](/img/icons/contact-email.svg){mf.split()[0].lower()}.{ml.lower()}@th-koeln.de", ""]
    if rng.random() < 0.4:
        L += [f"* [Webseite](https://www.{made_up_name(rng).lower()}.de)", ""]
    if rng.random() < 0.5:
        L += ["## Sprechstunden", "", "**Sprechstunde nach Vereinbarung**  ", f"Campus {camp}, Raum {rng.randint(1, 4)}.{rng.randint(100, 399)}", ""]
    sections = []
    if roles:
        sections.append(["## Funktionen", ""] + [f"* {r}" for r in roles] + [""])
    sections.append(["## Lehrgebiete", ""] + [f"* {t}" for t in teach] + (["* Suchen Sie ein Studierendenprojekt oder eine Abschlussarbeit?",
                     "  [PROX (Project Exchange TH Köln)](https://prox.innovation-hub.de/projects?filter=x)"] if rng.random() < 0.3 else []) + [""])
    if research:
        sections.append(["## Forschungsgebiete", ""] + [f"* {t}" for t in research] + [""])
    if pubs and rng.random() < 0.5:  # zweites echtes Layout: "* Titel" / "  Autoren, Jahr, Hg.: ..."
        rows = []
        for i, p in enumerate(pubs):
            if i and rng.random() < 0.1:
                rows += [rng.choice(["### Buch", "### Buchkapitel", "### Fachartikel", "### Konferenzbeiträge"])]
            co = [(a.rsplit(" ", 1)[0], a.rsplit(" ", 1)[1]) for a in p["authors"]]
            rows += two_line_entry(rng, p, co)
        sections.append([rng.choice(["## +Publikationen", "## + Publikationen"]), ""] + rows + [""])
    elif pubs:
        sections.append(["## + Publikationen", ""] + [f"* {'; '.join(p['authors_raw'])} ({p['year']}): {p['title']}. {p['_tail']}{p['_flag']}<{p['_url']}>" for p in pubs] + [""])
    if rng.random() < 0.4:
        sections.append(["## +Lebenslauf", "", "|  |  |", "| --- | --- |", f"| {rng.randint(1995, 2015)} bis {rng.randint(2016, 2022)} | Hahn-Schickard-Gesellschaft e.V. Aufbau der Forschungsgruppe |", ""])
    keep = sections[:2] if False else sections
    for sec in keep:
        L += sec
    text = "\n".join(L).strip() + "\n"
    for p in pubs:
        for k in ("_tail", "_flag", "_url"):
            p.pop(k)
    gold = {"people": [{"name": me, "role": ", ".join(roles), "faculty": fac, "expertise": list(dict.fromkeys(research + teach))}],
            "publications": pubs, "projects": [], "cooperations": []}
    return {"text": text, "gold": gold}


def make_example(rng, lasts, layout=None):
    if layout is None:
        layout = rng.choice([0, 1, 2, 4, 4, 5, 5, 5, 5, 5])
    if layout == 5:
        return make_th_example(rng, lasts)
    me = person(rng, lasts)
    title = rng.choice(TITLES)
    role = rng.choice(ROLES)
    fac = rng.choice(FACULTIES)
    exp = rng.sample(EXPERTISE, rng.randint(1, 3))

    pubs = []
    for _ in range(rng.randint(0, 2)):
        coauth = [person(rng, lasts) for _ in range(rng.randint(0, 2))]
        authors = coauth[:]
        authors.insert(rng.randint(0, len(authors)), me)
        pubs.append({
            "title": f"{rng.choice(T_A)} {rng.choice(T_B)} {rng.choice(T_C)}",
            "year": rng.randint(2018, 2026),
            "venue": rng.choice(VENUES),
            "authors": authors,
        })
    projs = []
    for _ in range(rng.randint(0, 1)):
        part = [me] + [person(rng, lasts) for _ in range(rng.randint(1, 2))]
        rng.shuffle(part)
        projs.append({"name": f"{rng.choice(PROJ_A)}{rng.choice(PROJ_B)}", "participants": part})

    if layout == 3:  # NIE im Training gesehenes Layout: Test, ob das Modell wirklich verallgemeinert
        lines = [f"**{title}{me}** – {role}, {fac}", "", "Arbeitsgebiete: " + "; ".join(exp), ""]
        blocks = []
        if pubs:
            blocks.append(["Publikationen", ""] + [f"{i}. {' & '.join(p['authors'])}: „{p['title']}“, {p['venue']} {p['year']}"
                                                   for i, p in enumerate(pubs, 1)] + [""])
        if projs:
            blocks.append(["Projekte", ""] + [f"* {pr['name']} (mit {' und '.join(pr['participants'])})" for pr in projs] + [""])
        rng.shuffle(blocks)
        for b in blocks:
            lines += b
        gold_person = {"name": me, "role": role, "faculty": fac, "expertise": exp}
        return {"text": "\n".join(lines).strip() + "\n",
                "gold": {"people": [gold_person], "publications": pubs, "projects": projs, "cooperations": []}}

    if layout is None:
        layout = rng.choice([0, 1, 2, 4, 4, 5, 5, 5, 5, 5])
    if layout == 5:
        return make_th_example(rng, lasts)
    if layout == 4:  # zufälliger "Stil": wechselnde Listenzeichen, Trennzeichen, Reihenfolgen (Layout 3 bleibt Testlayout!)
        mark = rng.choice(["-", "*", "•", "1."])
        sep = rng.choice([", ", " und ", " / ", "; "])
        q = rng.choice(["", '"', "'", "«"])
        qc = {"«": "»"}.get(q, q)
        order = rng.randint(0, 2)
        head = rng.choice([f"# {title}{me}", f"## {title}{me}", f"{title}{me}", f"**{title}{me}**"])
        info = rng.choice([f"{role}, {fac}", f"{fac} | {role}", f"Rolle: {role}\nFakultät: {fac}"])
        lines = [head, "", info, "", rng.choice(["Schwerpunkte", "Forschung", "Themen", "Expertise"]) + ": " + rng.choice([", ", " · ", " / "]).join(exp), ""]
        blocks = []
        if pubs:
            rows = []
            for p in pubs:
                a, t = sep.join(p["authors"]), f"{q}{p['title']}{qc}"
                rows.append(f"{mark} " + [f"{a}: {t}, {p['venue']} ({p['year']})", f"{t} — {a} ({p['year']}, {p['venue']})",
                                          f"{a} ({p['year']}) {t}. {p['venue']}."][order])
            blocks.append([rng.choice(["Publikationen", "Papers", "Veröffentlichungen", "Aktuelle Arbeiten"]), ""] + rows + [""])
        if projs:
            pj = rng.choice([0, 1])
            blocks.append([rng.choice(["Projekte", "Projektarbeit", "Förderprojekte"]), ""] +
                          [f"{mark} {pr['name']}" + (f": {sep.join(pr['participants'])}" if pj else f" (Team: {sep.join(pr['participants'])})") for pr in projs] + [""])
        rng.shuffle(blocks)
        for b in blocks:
            lines += b
        if rng.random() < 0.7:
            lines.append(rng.choice(NOISE))
        gold_person = {"name": me, "role": role, "faculty": fac, "expertise": exp}
        return {"text": "\n".join(lines).strip() + "\n",
                "gold": {"people": [gold_person], "publications": pubs, "projects": projs, "cooperations": []}}

    # --- Seite als Markdown rendern (drei Layouts, wechselnde Überschriften) ---
    layout = rng.randint(0, 2)
    h_pub = rng.choice(["Publikationen", "Veröffentlichungen", "Ausgewählte Publikationen"])
    h_proj = rng.choice(["Projekte", "Forschungsprojekte", "Laufende Projekte"])
    lines = [f"# {title}{me}", ""]
    if layout == 0:
        lines += [f"{role} · {fac}", ""]
    elif layout == 1:
        lines += [f"**Funktion:** {role}", f"**Fakultät:** {fac}", ""]
    else:
        lines += [f"{fac}", f"_{role}_", ""]
    lines += [("Schwerpunkte: " if layout != 1 else "Forschungsschwerpunkte: ") + ", ".join(exp), ""]
    blocks = []
    if pubs:
        b = [f"## {h_pub}", ""]
        for p in pubs:
            a = ", ".join(p["authors"])
            b.append(f"- {a} ({p['year']}): {p['title']}. {p['venue']}." if layout != 2
                     else f"- {p['title']} ({p['year']}), {p['venue']}. Autoren: {a}")
        blocks.append(b + [""])
    if projs:
        b = [f"## {h_proj}", ""]
        for pr in projs:
            b.append(f"- **{pr['name']}** – Team: {', '.join(pr['participants'])}"
                     if layout == 1 else f"- {pr['name']}: Beteiligte {', '.join(pr['participants'])}")
        blocks.append(b + [""])
    rng.shuffle(blocks)
    for b in blocks:
        lines += b
    if rng.random() < 0.7:
        lines.append(rng.choice(NOISE))
    text = "\n".join(lines).strip() + "\n"

    gold_person = {"name": me, "role": role, "faculty": fac, "expertise": exp}
    gold = {"people": [gold_person], "publications": pubs, "projects": projs, "cooperations": []}
    return {"text": text, "gold": gold}


def compact(d: dict) -> str:
    """Kanonische, kurze JSON-Schreibweise - das Modell lernt genau diese Ausgabe."""
    return json.dumps(d, ensure_ascii=False, separators=(",", ":"))


def build(n: int, lasts, seed: int, max_len: int, layout=None):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        ex = make_example(rng, lasts, layout)
        if len((ex["text"] + compact(ex["gold"])).encode("utf-8")) <= max_len:
            out.append(ex)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--max-len", type=int, default=100000, help="Bytes Text + Lösung (Kontextfenster des Mini-Modells)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lasts = split_last_names(seed=7)
    for name, n, seed in (("train", args.n_train, 1), ("val", args.n_val, 2), ("test", args.n_test, 3)):
        rows = build(n, lasts[name], seed, args.max_len)
        with open(OUT_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} Beispiele -> {OUT_DIR / (name + '.jsonl')}")
    rows = build(args.n_test, lasts["test"], 5, args.max_len, layout=5)
    with open(OUT_DIR / "test_th.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"test_th: {len(rows)} Beispiele (TH-Layout, neue Namen)")
    rows = build(args.n_test, lasts["test"], 4, args.max_len, layout=3)
    with open(OUT_DIR / "test_unseen.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"test_unseen: {len(rows)} Beispiele (neues Layout, nie im Training)")
    ex = build(1, lasts["train"], 99, args.max_len)[0]
    print("\n--- Beispielseite ---\n" + ex["text"] + "\n--- Lösung ---\n" + compact(ex["gold"]))
