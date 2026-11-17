// Farben und Schrift kommen aus den Design-Tokens in style.css (Abschnitt :root) -
// ein Designwechsel passiert also nur dort.
const cssVars = getComputedStyle(document.documentElement);
const token = (name, fallback) => cssVars.getPropertyValue(name).trim() || fallback;

const COLORS = {
  brand: token("--brand", "#d1001f"),
  member: token("--member", "#d1001f"),
  external: token("--external", "#9aa6b2"),
  project: token("--project", "#343a42"),
  publication: token("--publication", "#b59cff"),
  edge: token("--edge", "#aab3bd"),
  edgeStrong: token("--edge-strong", "#59636e"),
  paper: token("--bg", "#f2f3f5"),  // gleich dem Hintergrund, damit Label-Hintergründe nicht auffallen
  ink: token("--ink", "#1b1f24"),
};

const state = { view: "people", membersOnly: false, showIsolated: false };
let cy = null;
let focused = null;
let isolatedHidden = 0;

// ---------------------------------------------------------------------------
// Hilfsfunktionen
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const fold = (s) => String(s || "").normalize("NFD").replace(/\p{M}/gu, "").toLowerCase();
const byLabel = (a, b) => a.data("label").localeCompare(b.data("label"), "de");

// Baut DOM-Elemente ohne innerHTML - die Texte stammen aus gecrawlten Seiten/LLM-Ausgaben
// und dürfen deshalb nie als HTML interpretiert werden.
function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid !== null && kid !== undefined && kid !== false) el.append(kid);
  }
  return el;
}

function setHint(text) {
  $("hint").textContent = text || "";
}

// ---------------------------------------------------------------------------
// Daten laden
// ---------------------------------------------------------------------------
async function loadGraph() {
  const res = await fetch("data/graph.json", { cache: "no-store" });
  if (!res.ok) throw new Error(`graph.json nicht ladbar (HTTP ${res.status})`);
  return normalise(await res.json());
}

// Macht auch ältere graph.json-Dateien (ohne degree/member/items) darstellbar.
function normalise(graph) {
  graph.nodes.forEach(({ data: d }) => {
    d.degree = Number(d.degree) || 0;
    d.member = !!d.member;
    d.label = d.label || d.id;
    d.url = safeUrl(d.url);
    d.year = d.year || null;
    d.venue = d.venue || "";
    d.kind = d.kind || "project";
  });
  graph.edges.forEach(({ data: d }) => {
    if (d.type === "cooperation") {
      d.type = "collaboration";
      d.explicit = true;
    }
    d.weight = Number(d.weight) || 1;
    // ältere Dateien hatten nur eine Liste von Titeln ("projects")
    d.items = Array.isArray(d.items) ? d.items : (d.projects || []).map((title) => ({ title }));
  });
  return graph;
}

// Nur http(s)-Links werden verlinkt - nie javascript:, data: o. Ä., egal was in graph.json steht.
function safeUrl(value) {
  if (typeof value !== "string") return null;
  try {
    const u = new URL(value.trim());
    return u.protocol === "https:" || u.protocol === "http:" ? u.href : null;
  } catch (err) {
    return null;
  }
}

const scholarUrl = (title) => "https://scholar.google.com/scholar?q=" + encodeURIComponent(title);

function externalLink(href, text, cls) {
  return h("a", { class: cls, href, target: "_blank", rel: "noopener noreferrer" }, text);
}

// Eine Veröffentlichung/ein Projekt als Block: Titel (mit Link, falls vorhanden), Jahr · Zeitschrift.
function workRow(item) {
  const href = safeUrl(item.url);
  const meta = [item.year, item.venue].filter(Boolean).join(" · ");
  return h(
    "div",
    { class: "work" },
    href ? externalLink(href, `${item.title} ↗`, "work-title") : h("span", { class: "work-title" }, item.title),
    meta ? h("div", { class: "work-meta" }, meta) : null,
    // Kein DOI gefunden: bei Publikationen wenigstens eine Suche anbieten (klar als Suche beschriftet)
    !href && item.kind === "publication" ? externalLink(scholarUrl(item.title), "Auf Google Scholar suchen ↗", "work-search") : null
  );
}

const isWork = (n) => n.data("type") === "project";  // Projekt- und Publikationsknoten haben type "project"
const kindLabel = (kind) => (kind === "publication" ? "Publikation" : "Projekt");

const workOfNode = (n) => {
  const d = n.data();
  return { title: d.label, url: d.url, year: d.year, venue: d.venue, kind: d.kind };
};

// ---------------------------------------------------------------------------
// Darstellung
// ---------------------------------------------------------------------------
function buildStyle() {
  return [
    {
      selector: "node",
      style: {
        // Ob ein Name angezeigt wird, entscheidet updateLabels() (Klasse "lbl"),
        // damit sich Namen nie überdecken.
        label: "",
        "font-family": LABEL_FONT,
        "font-size": 11,
        color: COLORS.ink,
        "text-valign": "bottom",
        "text-margin-y": 5,
        "text-wrap": "ellipsis",
        "text-max-width": "120px",
        "text-background-color": COLORS.paper,
        "text-background-opacity": 0.85,
        "text-background-padding": 2,
        "border-width": 1.5,
        "border-color": COLORS.paper,
      },
    },
    {
      selector: 'node[type="person"]',
      style: {
        "background-color": COLORS.external,
        width: "mapData(degree, 0, 12, 8, 20)",
        height: "mapData(degree, 0, 12, 8, 20)",
      },
    },
    {
      selector: 'node[type="person"][?member]',
      style: {
        "background-color": COLORS.member,
        width: "mapData(degree, 0, 12, 16, 36)",
        height: "mapData(degree, 0, 12, 16, 36)",
        "font-size": 13,
        "font-weight": 600,
        "z-index": 10,  // Mitglieder (samt Namen) liegen über den übrigen Knoten
      },
    },
    {
      selector: 'node[type="project"]',
      style: {
        "background-color": COLORS.project,
        shape: "round-rectangle",
        width: "mapData(degree, 0, 12, 12, 24)",
        height: "mapData(degree, 0, 12, 12, 24)",
      },
    },
    {
      selector: 'node[type="project"][kind="publication"]',
      style: {
        "background-color": COLORS.publication,
        shape: "diamond",
        width: "mapData(degree, 0, 12, 10, 20)",
        height: "mapData(degree, 0, 12, 10, 20)",
      },
    },
    {
      selector: "edge",
      style: {
        width: "mapData(weight, 1, 6, 1.2, 5)",
        "line-color": COLORS.edge,
        opacity: 0.55,
        "curve-style": "bezier",
      },
    },
    { selector: 'edge[type="member_of"]', style: { width: 1, opacity: 0.35 } },
    { selector: "edge[?explicit]", style: { "line-color": COLORS.edgeStrong, opacity: 0.8 } },
    { selector: "node.lbl", style: { label: "data(label)" } },
    { selector: ".dim", style: { opacity: 0.08 } },
    {
      selector: "node.hl",
      style: { "text-wrap": "wrap", "text-max-width": "180px", "z-index": 20 },
    },
    { selector: "edge.hl", style: { opacity: 0.95, "line-color": COLORS.brand, width: 3.5, "z-index": 10 } },
    { selector: "node:selected", style: { "border-color": COLORS.ink, "border-width": 3 } },
    { selector: ".hidden", style: { display: "none" } },
  ];
}

function hasFcose() {
  try {
    return !!cytoscape("layout", "fcose");
  } catch (err) {
    return false;
  }
}

// Je mehr Knoten sichtbar sind, desto mehr Platz bekommt jeder einzelne.
function spreadFactor(nodeCount) {
  return Math.min(2.5, Math.max(1, Math.sqrt(nodeCount / 50)));
}

function fcoseOptions(nodeCount) {
  const k = spreadFactor(nodeCount);
  return {
    name: "fcose",
    quality: "proof",
    randomize: true,
    animate: false,
    fit: false,
    nodeDimensionsIncludeLabels: false,
    nodeRepulsion: () => 14000 * k,
    idealEdgeLength: (edge) => (edge.data("type") === "member_of" ? 60 : 100) * k,
    edgeElasticity: () => 0.25,
    gravity: 0.12,
    gravityRange: 3.8,
    numIter: 3000,
    nodeSeparation: 100 * k,
    packComponents: true,
    tile: true,
  };
}

function coseOptions(nodeCount) {
  const k = spreadFactor(nodeCount);
  return {
    name: "cose",
    animate: false,
    fit: false,
    nodeRepulsion: () => 20000 * k,
    idealEdgeLength: () => 100 * k,
    componentSpacing: 120 * k,
    nodeOverlap: 30,
    gravity: 0.3,
    numIter: 1500,
  };
}

function fitView(eles) {
  cy.fit(eles, 40);
  if (cy.zoom() > 1.1) {  // kleine Graphen nicht riesig aufzoomen
    cy.zoom(1.1);
    cy.center(eles);
  }
  scheduleLabels();
}

function runLayout() {
  const eles = cy.elements().not(".hidden");
  if (eles.empty()) return;
  const n = eles.nodes().length;
  const run = (opts) => {
    const layout = eles.layout(opts);
    layout.one("layoutstop", () => fitView(eles));
    layout.run();
  };
  try {
    run(hasFcose() ? fcoseOptions(n) : coseOptions(n));
  } catch (err) {
    console.warn("fcose-Layout fehlgeschlagen, nutze cose:", err);
    run(coseOptions(n));
  }
}

// ---------------------------------------------------------------------------
// Filter / Ansichten
// ---------------------------------------------------------------------------
function computeVisibility() {
  const peopleView = state.view === "people";
  const shownKind = state.view === "publications" ? "publication" : "project";
  cy.batch(() => {
    cy.nodes().forEach((n) => {
      const isProject = n.data("type") === "project";  // Werk: Projekt ODER Publikation (siehe kind)
      let hide = isProject && (peopleView || n.data("kind") !== shownKind);
      if (state.membersOnly && !isProject && !n.data("member")) hide = true;
      n.toggleClass("hidden", hide);
    });
    cy.edges().forEach((e) => {
      const type = e.data("type");
      let hide = e.source().hasClass("hidden") || e.target().hasClass("hidden");
      if (peopleView && type === "member_of") hide = true;
      if (!peopleView && type === "collaboration" && !e.data("explicit")) hide = true;
      e.toggleClass("hidden", hide);
    });
    isolatedHidden = 0;
    if (!state.showIsolated) {
      cy.nodes().not(".hidden").forEach((n) => {
        if (n.connectedEdges().not(".hidden").empty()) {
          n.addClass("hidden");
          isolatedHidden += 1;
        }
      });
    }
  });
}

function updateStats() {
  const nodes = cy.nodes().not(".hidden").length;
  const edges = cy.edges().not(".hidden").length;
  const vis = cy.nodes().not(".hidden");
  const persons = vis.filter((n) => n.data("type") === "person");
  $("stat-members").textContent = persons.filter((n) => n.data("member")).length;
  $("stat-externals").textContent = persons.filter((n) => !n.data("member")).length;
  $("stat-projects").textContent = vis.filter((n) => isWork(n) && n.data("kind") === "project").length;
  $("stat-publications").textContent = vis.filter((n) => isWork(n) && n.data("kind") === "publication").length;
  $("stat-edges").textContent = edges;
  $("stats").textContent = isolatedHidden ? `${isolatedHidden} Einzelne ausgeblendet` : "";
  setHint(nodes === 0 ? "Keine Verbindungen in dieser Ansicht." : "");
  scheduleLabels();
}

function applyFilters() {
  clearFocus();
  computeVisibility();
  runLayout();
  updateStats();
}

function syncControls() {
  document.querySelectorAll("#view-seg [data-view]").forEach((b) => {
    b.classList.toggle("active", b.dataset.view === state.view);
  });
  $("opt-members").checked = state.membersOnly;
  $("opt-isolated").checked = state.showIsolated;
}

// ---------------------------------------------------------------------------
// Beschriftungen ohne Überlappung
// ---------------------------------------------------------------------------
// Cytoscape würde jedes Label immer zeichnen, auch wenn sich Namen überdecken. Deshalb wird bei
// jeder Änderung von Zoom, Position oder Filter neu entschieden, welche Namen Platz haben.
// Nach Priorität (überfahrener/angeklickter Knoten, seine Nachbarn, Mitglieder, dann stark
// vernetzte Personen) bekommt ein Name sein Label nur, wenn er keinen schon platzierten berührt.
// Wer fehlt, erscheint beim Hineinzoomen, beim Darüberfahren oder über die Suche.
const LABEL_FONT = token("--font", 'system-ui, "Segoe UI", Arial, sans-serif');
const LABEL_PX = { member: 13, other: 11 };  // Schriftgröße auf dem Bildschirm
const LABEL_GAP_X = 6;  // Mindestabstand zwischen zwei Namen in Pixeln
const LABEL_GAP_Y = 2;
let activeNode = null;  // Knoten, der gerade überfahren oder angeklickt ist
let labelFrame = 0;
let measureCtx = null;
const widthCache = new Map();

function textWidth(text, fontPx, bold) {
  const key = `${bold ? 1 : 0}|${fontPx}|${text}`;
  let w = widthCache.get(key);
  if (w === undefined) {
    measureCtx = measureCtx || document.createElement("canvas").getContext("2d");
    measureCtx.font = `${bold ? 600 : 400} ${fontPx}px ${LABEL_FONT}`;
    w = measureCtx.measureText(text).width;
    widthCache.set(key, w);
  }
  return w;
}

function scheduleLabels() {
  if (labelFrame) return;  // höchstens einmal pro Frame neu berechnen
  labelFrame = requestAnimationFrame(() => {
    labelFrame = 0;
    updateLabels();
  });
}

function updateLabels() {
  if (!cy) return;
  const zoom = cy.zoom();
  const viewW = cy.width();
  const viewH = cy.height();
  const rank = (n) => (n === activeNode ? 0 : n.hasClass("hl") ? 1 : n.data("member") ? 2 : 3);

  const candidates = cy
    .nodes()
    .filter((n) => !n.hasClass("hidden") && !n.hasClass("dim"))
    .toArray()
    .sort((a, b) => rank(a) - rank(b) || b.data("degree") - a.data("degree") || byLabel(a, b));

  // Kreise aller sichtbaren Knoten: Ein Name soll auch nicht von einem fremden Knoten verdeckt werden.
  // Mitglieder werden über den z-index über anderen Knoten gezeichnet; ihr Name muss daher nur
  // anderen Mitgliedern ausweichen, alle übrigen Namen allen Knoten.
  const circles = candidates.map((n) => {
    const p = n.renderedPosition();
    return { id: n.id(), x: p.x, y: p.y, r: n.renderedWidth() / 2 + 1.5, member: n.data("member") };
  });

  // Alle Maße in Bildschirm-Pixeln: Die Schrift hat immer dieselbe Größe auf dem Bildschirm
  // (siehe fitLabelToZoom). Wer hineinzoomt, gewinnt dadurch Platz und sieht mehr Namen.
  const placed = [];
  const wanted = new Set();
  for (const n of candidates) {
    const d = n.data();
    const isActive = n === activeNode;
    const fontPx = d.member ? LABEL_PX.member : LABEL_PX.other;

    const pos = n.renderedPosition();
    if (pos.x < -40 || pos.x > viewW + 40 || pos.y < -40 || pos.y > viewH + 40) continue;  // außerhalb

    const hl = n.hasClass("hl");
    const maxW = hl ? 180 : 120;  // wie "text-max-width" in fitLabelToZoom
    const natural = textWidth(d.label, fontPx, d.member);
    const lines = hl ? Math.min(3, Math.ceil(natural / maxW)) : 1;
    const w = Math.min(natural, maxW) + 9;  // + Hintergrund-Polster, das Cytoscape um den Text zeichnet
    const top = pos.y + n.renderedHeight() / 2;  // gemessen: Cytoscapes Label-Rechteck beginnt ca. 1 px unter dem Knoten
    const box = { x0: pos.x - w / 2, x1: pos.x + w / 2, y0: top, y1: top + lines * fontPx * 1.25 + 6.5 };

    const clash = placed.some(
      (b) =>
        box.x0 < b.x1 + LABEL_GAP_X && box.x1 > b.x0 - LABEL_GAP_X &&
        box.y0 < b.y1 + LABEL_GAP_Y && box.y1 > b.y0 - LABEL_GAP_Y
    );
    if (clash && !isActive) continue;

    const covered = circles.some((c) => {
      if (c.id === n.id() || (d.member && !c.member)) return false;
      const dx = c.x - Math.max(box.x0, Math.min(c.x, box.x1));  // Abstand Kreismitte -> Rechteck
      const dy = c.y - Math.max(box.y0, Math.min(c.y, box.y1));
      return dx * dx + dy * dy < c.r * c.r;
    });
    if (covered && !isActive) continue;
    placed.push(box);
    wanted.add(n.id());
  }

  cy.batch(() => {
    cy.nodes().forEach((n) => {
      const want = wanted.has(n.id());
      if (n.hasClass("lbl") !== want) n.toggleClass("lbl", want);
      if (want) fitLabelToZoom(n, zoom);
    });
  });
}

// Cytoscape skaliert Schrift mit dem Zoom. Damit Namen auf dem Bildschirm immer gleich groß bleiben,
// wird die Schriftgröße (in Modell-Einheiten) hier durch den Zoom geteilt.
function fitLabelToZoom(n, zoom) {
  const hl = n.hasClass("hl");
  const key = `${Math.round(Math.log(zoom) * 200)}|${hl ? 1 : 0}`;  // ~0,5 % Zoom-Schritte
  if (n.scratch("labelKey") === key) return;
  n.scratch("labelKey", key);
  n.style({
    "font-size": (n.data("member") ? LABEL_PX.member : LABEL_PX.other) / zoom,
    "text-max-width": (hl ? 180 : 120) / zoom,
    "text-margin-y": 5 / zoom,
    "text-background-padding": 2 / zoom,
  });
}

// ---------------------------------------------------------------------------
// Hervorheben & Fokus
// ---------------------------------------------------------------------------
function highlight(node) {
  activeNode = node;
  cy.batch(() => {
    cy.elements().removeClass("dim hl");
    const hood = node.closedNeighborhood().not(".hidden");
    cy.elements().not(".hidden").not(hood).addClass("dim");
    hood.addClass("hl");
  });
  scheduleLabels();
}

function clearHighlight() {
  activeNode = null;
  cy.elements().removeClass("dim hl");
  scheduleLabels();
}

function focus(node) {
  focused = node;
  highlight(node);
  cy.elements().unselect();
  node.select();
  showNodePanel(node);
}

function clearFocus() {
  focused = null;
  if (cy) {
    clearHighlight();
    cy.elements().unselect();
  }
  $("panel").classList.add("hidden");
}

function centerOn(node) {
  cy.animate({ center: { eles: node }, zoom: Math.max(cy.zoom(), 1.3) }, { duration: 300 });
}

// Macht einen Knoten sichtbar (falls durch Filter ausgeblendet) und springt dorthin.
function reveal(node) {
  let changed = false;
  const adjust = [
    () => {
      if (state.membersOnly && node.data("type") === "person" && !node.data("member")) {
        state.membersOnly = false;
        return true;
      }
    },
    () => {
      // Werk: in die passende Ansicht (Projekte bzw. Publikationen); Person ohne Kooperationskante: Projektansicht
      const wanted = !isWork(node) ? "projects" : node.data("kind") === "publication" ? "publications" : "projects";
      if (isWork(node) ? state.view !== wanted : state.view === "people") {
        state.view = wanted;
        return true;
      }
    },
    () => {
      if (!state.showIsolated) {
        state.showIsolated = true;
        return true;
      }
    },
  ];
  for (const step of adjust) {
    if (!node.hasClass("hidden")) break;
    if (step()) {
      changed = true;
      computeVisibility();
    }
  }
  if (changed) {
    syncControls();
    runLayout();
    updateStats();
  }
  if (node.hasClass("hidden")) {
    setHint("Dieser Knoten ist in der aktuellen Ansicht ausgeblendet.");
    return;
  }
  focus(node);
  centerOn(node);
}

// ---------------------------------------------------------------------------
// Infopanel
// ---------------------------------------------------------------------------
function linkList(title, nodes) {
  if (nodes.length === 0) return null;
  const LIMIT = 40;
  const shown = nodes.slice(0, LIMIT);
  return h(
    "div",
    {},
    h("h4", {}, `${title} (${nodes.length})`),
    h(
      "div",
      { class: "links" },
      shown.map((n) => h("button", { class: "link", onclick: () => reveal(n) }, n.data("label"))),
      nodes.length > LIMIT ? h("span", { class: "more" }, `… und ${nodes.length - LIMIT} weitere`) : null
    )
  );
}

function showNodePanel(n) {
  const d = n.data();
  const hood = n.neighborhood("node").toArray().sort(byLabel);
  const people = hood.filter((x) => x.data("type") === "person");
  const projects = hood.filter((x) => isWork(x) && x.data("kind") === "project");
  const publications = hood.filter((x) => isWork(x) && x.data("kind") === "publication");
  const isPerson = d.type === "person";
  const meta = (isPerson
    ? [d.role, d.faculty]
    : [kindLabel(d.kind), d.year, d.venue]
  ).filter(Boolean).join(" · ");
  const href = safeUrl(d.url);

  const content = h(
    "div",
    {},
    d.member ? h("span", { class: "badge" }, "Cluster-Mitglied") : null,
    h("h3", {}, d.label),
    h("div", { class: "meta" }, meta),
    (d.expertise || []).map((e) => h("span", { class: "tag" }, e)),
    d.description ? h("p", {}, d.description) : null,
    !isPerson && href
      ? h("p", {}, externalLink(href, d.kind === "publication" ? "Zur Veröffentlichung ↗" : "Zur Projektseite ↗", "work-title"))
      : null,
    !isPerson && !href && d.kind === "publication"
      ? h("p", {}, externalLink(scholarUrl(d.label), "Auf Google Scholar suchen ↗", "work-search"))
      : null,
    isPerson ? linkList("Arbeitet zusammen mit", people) : linkList("Beteiligte", people),
    isPerson ? linkList("Projekte", projects) : null,
    isPerson ? linkList("Publikationen", publications) : null
  );
  $("panel-content").replaceChildren(content);
  $("panel").classList.remove("hidden");
}

// Panel für eine angeklickte Verbindungslinie: wer ↔ wer, und die zugehörigen Veröffentlichungen mit Link.
function showEdgePanel(e) {
  const d = e.data();
  const LIMIT = 50;
  let items = d.items;
  let heading = "Gemeinsame Veröffentlichungen & Projekte";
  if (d.type === "member_of") {  // Person ↔ Projekt: das Projekt selbst ist die "Veröffentlichung"
    const work = e.source().data("type") === "project" ? e.source() : e.target();
    items = [workOfNode(work)];
    heading = "Beteiligt an";
  }
  const shown = items.slice(0, LIMIT);

  const content = h(
    "div",
    {},
    h("h3", {}, `${e.source().data("label")} ↔ ${e.target().data("label")}`),
    h("div", { class: "meta" }, d.type === "member_of" ? "Beteiligung" : "Zusammenarbeit"),
    d.context ? h("p", {}, d.context) : null,
    items.length
      ? h(
          "div",
          {},
          h("h4", {}, `${heading} (${items.length})`),
          shown.map(workRow),
          items.length > LIMIT ? h("span", { class: "more" }, `… und ${items.length - LIMIT} weitere`) : null
        )
      : d.context ? null : h("p", { class: "more" }, "Keine Details erfasst.")
  );
  clearFocus();
  $("panel-content").replaceChildren(content);
  $("panel").classList.remove("hidden");
}

// ---------------------------------------------------------------------------
// Suche
// ---------------------------------------------------------------------------
function findMatches(query) {
  const q = fold(query).trim();
  if (!q) return [];
  return cy
    .nodes()
    .filter((n) => fold(n.data("label")).includes(q))
    .toArray()
    .sort((a, b) => {
      const vis = Number(a.hasClass("hidden")) - Number(b.hasClass("hidden"));
      const mem = Number(b.data("member")) - Number(a.data("member"));
      return vis || mem || byLabel(a, b);
    });
}

function setupSearch() {
  const input = $("search");
  const list = $("results");

  const renderResults = () => {
    const q = input.value.trim();
    const matches = findMatches(q);
    list.replaceChildren();
    list.hidden = !q || matches.length === 0;
    setHint(q && !matches.length ? "Kein Treffer" : "");
    matches.slice(0, 8).forEach((n) => {
      const kind = isWork(n) ? kindLabel(n.data("kind")) : n.data("member") ? "Mitglied" : "Extern";
      const btn = h("button", { type: "button" }, [
        h("span", { class: n.hasClass("hidden") ? "r-dim" : "" }, n.data("label")),
        h("span", { class: n.data("kind") === "publication" && isWork(n) ? "r-kind pub" : "r-kind" }, kind),
      ]);
      btn.addEventListener("click", () => {
        reveal(n);
        list.hidden = true;
      });
      list.append(h("li", {}, [btn]));
    });
  };

  input.addEventListener("input", renderResults);
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") { list.hidden = true; return; }
    if (ev.key !== "Enter") return;
    const matches = findMatches(input.value);
    if (matches.length) {
      reveal(matches[0]);
      list.hidden = true;
    }
  });
}

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
(async function init() {
  let graph;
  try {
    graph = await loadGraph();
  } catch (err) {
    setHint(String(err.message || err));
    return;
  }

  cy = cytoscape({
    container: $("cy"),
    elements: graph,
    style: buildStyle(),
    wheelSensitivity: 0.25,
    minZoom: 0.05,
    maxZoom: 4,
    boxSelectionEnabled: false,
  });

  cy.on("viewport resize", scheduleLabels);  // Zoom, Verschieben, Fenstergröße
  cy.on("drag", "node", scheduleLabels);     // Knoten per Maus verschieben
  cy.on("mouseover", "node", (ev) => { if (!focused) highlight(ev.target); });
  cy.on("mouseout", "node", () => { if (!focused) clearHighlight(); });
  cy.on("tap", "node", (ev) => focus(ev.target));
  cy.on("tap", "edge", (ev) => showEdgePanel(ev.target));
  cy.on("mouseover", "edge", (ev) => {
    $("cy").style.cursor = "pointer";
    if (!focused) ev.target.addClass("hl");
  });
  cy.on("mouseout", "edge", (ev) => {
    $("cy").style.cursor = "";
    if (!focused) ev.target.removeClass("hl");
  });
  cy.on("mouseover", "node", () => { $("cy").style.cursor = "pointer"; });
  cy.on("mouseout", "node", () => { $("cy").style.cursor = ""; });
  cy.on("tap", (ev) => { if (ev.target === cy) clearFocus(); });

  $("panel-close").addEventListener("click", clearFocus);

  // Zoom-Knöpfe
  const zoomBy = (f) => {
    const z = Math.min(cy.maxZoom(), Math.max(cy.minZoom(), cy.zoom() * f));
    cy.animate({ zoom: { level: z, renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } } }, { duration: 160 });
  };
  $("zoom-in").addEventListener("click", () => zoomBy(1.4));
  $("zoom-out").addEventListener("click", () => zoomBy(1 / 1.4));
  $("zoom-fit").addEventListener("click", () => fitView(cy.elements().not(".hidden")));

  // Die Graph-Fläche ändert ihre Größe, wenn das Detailfenster auf- oder zugeht
  if ("ResizeObserver" in window) {
    new ResizeObserver(() => { cy.resize(); scheduleLabels(); }).observe($("cy"));
  }
  document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") clearFocus(); });

  document.querySelectorAll("#view-seg [data-view]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.view = btn.dataset.view;
      syncControls();
      applyFilters();
    });
  });
  $("opt-members").addEventListener("change", (ev) => {
    state.membersOnly = ev.target.checked;
    applyFilters();
  });
  $("opt-isolated").addEventListener("change", (ev) => {
    state.showIsolated = ev.target.checked;
    applyFilters();
  });

  setupSearch();
  syncControls();
  applyFilters();
})();
