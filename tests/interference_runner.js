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
 * So the widget is mounted against the stand-in in tests/dom_stub.js (there is
 * no jsdom here), at several widths and pixel ratios. What this file supplies is
 * the part that is actually about this widget: its layout model and the canvas
 * methods it calls. The pane cap is read out of the widget's own stylesheet
 * rather than restated, so the layout below is driven by the CSS that ships.
 *
 * Prints one JSON object. Driven by tests/test_interference_widget.py.
 */
const fs = require("fs");
const path = require("path");
const { makeEnv: makeStubEnv, loadWidget } = require("./dom_stub.js");

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
  return makeStubEnv({
    dpr: deviceRatio,
    // Only the plot pane constrains anything: one wide canvas, centred, capped.
    layoutWidth(node) {
      const pane = node.tagName === "CANVAS" ? node.parentNode : node;
      if (pane && pane.className === "if-plot") {
        return Math.min(containerWidth, PLOT_CAP);
      }
      return containerWidth;
    },
    ctxStub() {
      return {
        fillStyle: "", strokeStyle: "", font: "", textAlign: "", lineWidth: 1,
        fillRect() {}, fillText() {}, setTransform() {}, beginPath() {},
        moveTo() {}, lineTo() {}, stroke() {}, setLineDash() {},
        measureText: (t) => ({ width: String(t).length * 6 }),
      };
    },
  });
}

function load(env) {
  loadWidget(SOURCE, env);
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
  const amp = api.envelope(d);
  let maxErr = 0;
  for (let i = 0; i < n; i++) {
    const kx = s.t[i] * api.CYCLES * 2 * Math.PI;
    const closed = amp * Math.cos(kx - d / 2);
    maxErr = Math.max(maxErr, Math.abs(s.sum[i] - closed));
    // The two waves themselves, while we are here: equal amplitude, one delayed.
    maxErr = Math.max(maxErr, Math.abs(s.a[i] - Math.cos(kx)));
    maxErr = Math.max(maxErr, Math.abs(s.b[i] - Math.cos(kx - d)));
  }
  out.physics[d.toFixed(6)] = {
    envelope: amp,
    peak: Math.max.apply(null, Array.from(s.sum)),
    brightness: api.brightness(d),
    maxErr: maxErr,
  };
}

// What the reader is actually told, at the three phases the note branches on.
for (const d of [0, Math.PI / 2, Math.PI]) {
  const host = env.makeEl("div");
  api.mount(host, { dphi: d });
  const root = host.children[0];
  out.readout[d.toFixed(6)] = {
    value: find(root, "if-val").textContent,
    swatch: find(root, "if-swatch").style.background,
    read: find(root, "if-read").innerHTML,
    note: find(root, "if-note").innerHTML,
  };
}

process.stdout.write(JSON.stringify(out));
