/*
 * Node runner for the front page's interference widget (apps/web/interfere.js).
 *
 * Two things need checking here, and neither can be seen from a driven browser
 * (that tab is always hidden, so it never lays anything out):
 *
 *   1. The canvas sizes itself to the pane it is actually laid out in. A canvas
 *      whose bitmap aspect does not match the aspect it is displayed at is
 *      silently stretched in one axis, which is the bug errors.js shipped in its
 *      two plot widgets: flattened on a desktop, stretched tall on a phone, with
 *      the text distorted to match.
 *   2. The picture is the identity it claims to illustrate. The widget exists to
 *      show cos(kx) + cos(kx - d) = 2cos(d/2)cos(kx - d/2); if the drawing and
 *      the closed form ever part company, the front page is illustrating
 *      something that is not true.
 *
 * So the widget is mounted against a hand-built stand-in for the parts of the
 * DOM it touches (there is no jsdom here), at several widths and pixel ratios.
 * The pane cap is read out of the widget's own stylesheet rather than restated,
 * so the layout below is driven by the CSS that ships.
 *
 * Prints one JSON object. Driven by tests/test_interference_widget.py.
 */
const fs = require("fs");
const path = require("path");

const SRC = path.join(__dirname, "..", "apps", "web", "interfere.js");
const SOURCE = fs.readFileSync(SRC, "utf8");

/** The plot pane's max-width, read off the stylesheet that ships. */
function readPlotCap(src) {
  const css = /const CSS = `([\s\S]*?)`;/.exec(src)[1];
  const rule = /\.if-plot\{([^}]*)\}/.exec(css);
  const m = rule && /max-width:\s*(\d+(?:\.\d+)?)px/.exec(rule[1]);
  return m ? parseFloat(m[1]) : Infinity;
}

const PLOT_CAP = readPlotCap(SOURCE);

function makeEnv(containerWidth, deviceRatio) {
  const styles = {};

  function layoutWidth(node) {
    // Only the plot pane constrains anything: one wide canvas, centred, capped.
    const pane = node.tagName === "CANVAS" ? node.parentNode : node;
    if (pane && pane.className === "if-plot") {
      return Math.min(containerWidth, PLOT_CAP);
    }
    return containerWidth;
  }

  function ctxStub() {
    return {
      fillStyle: "", strokeStyle: "", font: "", textAlign: "", lineWidth: 1,
      fillRect() {}, fillText() {}, setTransform() {}, beginPath() {},
      moveTo() {}, lineTo() {}, stroke() {}, fill() {}, setLineDash() {},
      measureText: (t) => ({ width: String(t).length * 6 }),
    };
  }

  function makeEl(tag) {
    const node = {
      tagName: String(tag).toUpperCase(),
      className: "",
      id: "",
      innerHTML: "",
      textContent: "",
      children: [],
      parentNode: null,
      style: {},
      _listeners: {},
      appendChild(c) { c.parentNode = node; node.children.push(c); return c; },
      setAttribute() {},
      addEventListener(t, fn) { (node._listeners[t] = node._listeners[t] || []).push(fn); },
      getBoundingClientRect() { return { width: layoutWidth(node), height: 0 }; },
    };
    Object.defineProperty(node, "parentElement", { get: () => node.parentNode });
    if (node.tagName === "CANVAS") {
      node.width = 300; node.height = 150;
      node.getContext = () => ctxStub();
    }
    return node;
  }

  const doc = {
    getElementById: (id) => styles[id] || null,
    createElement: (tag) => makeEl(tag),
    head: { appendChild(s) { if (s.id) styles[s.id] = s; } },
  };
  const win = {
    document: doc,
    devicePixelRatio: deviceRatio,
    addEventListener() {},
    // No ResizeObserver on purpose: the fallback path must work too.
  };
  return { win, doc, makeEl };
}

function load(env) {
  const mod = { exports: {} };
  const fn = new Function("window", "document", "module", SOURCE);
  fn(env.win, env.doc, mod);
  return env.win.PhotonnInterfere;
}

function find(root, cls) {
  let hit = null;
  const walk = (n) => {
    if (!hit && n.className === cls) hit = n;
    n.children.forEach(walk);
  };
  walk(root);
  return hit;
}

function canvases(root) {
  const out = [];
  const walk = (n) => {
    if (n.tagName === "CANVAS") {
      out.push({
        bitmapW: n.width,
        bitmapH: n.height,
        shownW: n.getBoundingClientRect().width,
        styleH: n.style.height ? parseFloat(n.style.height) : null,
        pane: n.parentNode ? n.parentNode.className : null,
      });
    }
    n.children.forEach(walk);
  };
  walk(root);
  return out;
}

const out = { plotCap: PLOT_CAP, layout: {}, physics: {}, readout: {} };

// A phone, a small tablet, and the front page's prose column on a desktop.
const WIDTHS = [300, 480, 1042];
for (const ratio of [1, 2]) {
  for (const width of WIDTHS) {
    const env = makeEnv(width, ratio);
    const api = load(env);
    const host = env.makeEl("div");
    api.mount(host);
    out.layout[width + "x" + ratio] = {
      width: width,
      dpr: ratio,
      canvases: canvases(host.children[0]),
    };
  }
}

// The identity, at the phases the widget's own copy talks about.
const env = makeEnv(640, 1);
const api = load(env);
const PHASES = [0, Math.PI / 4, Math.PI / 2, Math.PI, (3 * Math.PI) / 2, 2 * Math.PI];
for (const d of PHASES) {
  const n = 257;
  const s = api.samples(d, n);
  const env2 = api.envelope(d);
  let maxErr = 0;
  for (let i = 0; i < n; i++) {
    const kx = s.t[i] * api.CYCLES * 2 * Math.PI;
    const closed = env2 * Math.cos(kx - d / 2);
    maxErr = Math.max(maxErr, Math.abs(s.sum[i] - closed));
    // The two waves themselves, while we are here: equal amplitude, one delayed.
    maxErr = Math.max(maxErr, Math.abs(s.a[i] - Math.cos(kx)));
    maxErr = Math.max(maxErr, Math.abs(s.b[i] - Math.cos(kx - d)));
  }
  out.physics[d.toFixed(6)] = {
    dphi: d,
    envelope: env2,
    peak: Math.max.apply(null, Array.from(s.sum)),
    brightness: api.brightness(d),
    maxErr: maxErr,
  };
}

// What the reader is actually told, at the three phases the note branches on.
for (const d of [0, Math.PI / 2, Math.PI]) {
  const e = makeEnv(640, 1);
  const a = load(e);
  const host = e.makeEl("div");
  a.mount(host, { dphi: d });
  const root = host.children[0];
  out.readout[d.toFixed(6)] = {
    value: find(root, "if-val").textContent,
    swatch: find(root, "if-swatch").style.background,
    read: find(root, "if-read").innerHTML,
    note: find(root, "if-note").innerHTML,
  };
}

process.stdout.write(JSON.stringify(out));
