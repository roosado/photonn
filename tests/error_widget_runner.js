/*
 * Node runner for the error-mechanism widgets (apps/web/errors.js).
 *
 * These widgets have two layout properties that nothing else in the suite can
 * see, and that the driven browser cannot check either -- that tab is always
 * hidden, so it never lays anything out:
 *
 *   1. The three-panel widgets are a before/after/difference triptych. They only
 *      say anything if all three panels are visible while the slider moves, so
 *      the row must never wrap, at any width.
 *   2. The two plot widgets size their own canvas. A canvas whose bitmap aspect
 *      does not match the aspect it is displayed at is silently stretched in one
 *      axis -- which is exactly what a fixed 420-unit drawing space did inside a
 *      column of some other width: flattened on a desktop, stretched tall on a
 *      phone, with the text distorted to match.
 *
 * So errors.js is mounted here against a hand-built stand-in for the parts of
 * the DOM it touches, at several viewport widths. The stub is a small flexbox
 * for this one case, and it reads its rules out of the widget's own stylesheet
 * rather than restating them -- otherwise "three panels stay on one row" would
 * be a check on this file instead of on the CSS that ships.
 *
 * Prints one JSON object. Driven by tests/test_error_widgets.py.
 */
const fs = require("fs");
const path = require("path");

const SRC = path.join(__dirname, "..", "apps", "web", "errors.js");

const NARROW_AT = 560;   // the widget stylesheet's one breakpoint

/**
 * Read the pane layout rules out of the widget's own stylesheet.
 *
 * Parsed rather than restated so the layout below is driven by the CSS that
 * actually ships. Restating it would make the "three panels stay on one row"
 * check tautological -- it would be testing this file.
 */
function readPaneCss(src) {
  const css = /const CSS = `([\s\S]*?)`;/.exec(src)[1];
  // `sel` arrives already escaped for RegExp; `\{` pins the end of the selector
  // so that `\.ex-pane` does not also match `.ex-panes{`.
  const rule = (sel) => {
    const m = new RegExp(sel + "\\{([^}]*)\\}").exec(css);
    return m ? m[1] : "";
  };
  const px = (body, prop, dflt) => {
    const m = new RegExp(prop + ":\\s*(\\d+(?:\\.\\d+)?)px").exec(body);
    return m ? parseFloat(m[1]) : dflt;
  };
  const panes = rule("\\.ex-panes");
  const pane = rule("\\.ex-pane");
  const wide = rule("\\.ex-pane\\.ex-wide");
  // The one media query, for the narrow gap.
  const narrow = /@media \(max-width:560px\)\{([\s\S]*?)\n\}/.exec(css);
  const narrowPanes = narrow ? (/\.ex-panes\{([^}]*)\}/.exec(narrow[1]) || [, ""])[1] : "";
  const flex = /flex:\s*\d+\s+\d+\s+(\d+(?:\.\d+)?)(px)?/.exec(pane);
  return {
    wrap: /flex-wrap:\s*wrap/.test(panes),
    gap: px(panes, "gap", 0),
    narrowGap: px(narrowPanes, "gap", px(panes, "gap", 0)),
    basis: flex ? parseFloat(flex[1]) : 0,
    minWidth: px(pane, "min-width", /min-width:\s*0/.test(pane) ? 0 : 0),
    wideMax: px(wide, "max-width", Infinity),
  };
}

const PANE_CSS = readPaneCss(fs.readFileSync(SRC, "utf8"));

/** How many panes share one row, given the parsed rules. */
function perRow(containerWidth, count, css) {
  if (!css.wrap) return count;
  const gap = containerWidth <= NARROW_AT ? css.narrowGap : css.gap;
  // A wrapping row fits as many items at their hypothetical main size as it can.
  const hypothetical = Math.max(css.basis, css.minWidth) || 1;
  const fit = Math.floor((containerWidth + gap) / (hypothetical + gap));
  return Math.max(1, Math.min(count, fit));
}

function makeEnv(containerWidth, deviceRatio) {
  const styles = {};
  const css = PANE_CSS;

  function layoutWidth(node) {
    // A pane's share of its row, by the rules read off the stylesheet above.
    const pane = node.tagName === "CANVAS" ? node.parentNode : node;
    if (!pane || !pane.parentNode) return containerWidth;
    const row = pane.parentNode;
    const panes = row.children.filter((c) => c.className.indexOf("ex-pane") === 0);
    if (!panes.length) return containerWidth;
    if (pane.className.indexOf("ex-wide") >= 0) {
      return Math.min(containerWidth, css.wideMax);
    }
    const gap = containerWidth <= NARROW_AT ? css.narrowGap : css.gap;
    const n = perRow(containerWidth, panes.length, css);
    // Flex items grow to fill whatever row they land on.
    return Math.max(css.minWidth, (containerWidth - gap * (n - 1)) / n);
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

  function ctxStub() {
    return {
      fillStyle: "", font: "", textAlign: "",
      fillRect() {}, fillText() {}, setTransform() {}, beginPath() {}, arc() {},
      fill() {}, stroke() {}, moveTo() {}, lineTo() {}, setLineDash() {},
      createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
      putImageData() {},
    };
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

// A real-shaped mask: 128x128 of 8-bit phase codes, contents irrelevant here.
const N = 128;
const codes = Buffer.alloc(N * N);
for (let i = 0; i < codes.length; i++) codes[i] = (i * 37 + (i >> 7) * 11) & 255;
const MASK = { n: N, mask_b64: codes.toString("base64") };

function load(env) {
  const src = fs.readFileSync(SRC, "utf8");
  env.win.PHOTONN_ERR_MASK = MASK;
  const mod = { exports: {} };
  const fn = new Function("window", "document", "module", "atob", src);
  fn(env.win, env.doc, mod, (b64) => Buffer.from(b64, "base64").toString("binary"));
  return env.win.PhotonnErrors;
}

/** Every canvas in a mounted widget, with what it would actually display as. */
function survey(root, env) {
  const out = [];
  const walk = (node) => {
    if (node.tagName === "CANVAS") {
      const shown = node.getBoundingClientRect().width;
      const styleH = node.style.height ? parseFloat(node.style.height) : null;
      out.push({
        bitmapW: node.width,
        bitmapH: node.height,
        shownW: Math.round(shown * 1000) / 1000,
        styleH: styleH,
        wide: node.parentNode.className.indexOf("ex-wide") >= 0,
      });
    }
    node.children.forEach(walk);
  };
  walk(root);
  return out;
}

const out = {};
// `mesh` is mounted here with no window.PHOTONN_MESH on purpose: that exercises
// the stand-in path, and layout -- which is all this file tests -- does not depend
// on which mesh the operator came from.
const KINDS = ["crosstalk", "phase", "detector", "loss", "wavelength", "quant", "mesh"];
// A phone, a small tablet, and a desktop column on the tolerance page.
const WIDTHS = [300, 480, 1042];

for (const dprValue of [1, 2]) {
  for (const width of WIDTHS) {
    for (const kind of KINDS) {
      const env = makeEnv(width, dprValue);
      const api = load(env);
      const host = env.makeEl("div");
      api.mount(host, { kind: kind });
      const root = host.children[0];
      const panes = root.children.filter((c) => c.className === "ex-panes")[0];
      const count = panes ? panes.children.length : 0;
      out[`${kind}@${width}x${dprValue}`] = {
        kind: kind,
        width: width,
        dpr: dprValue,
        panes: count,
        perRow: count ? perRow(width, count, PANE_CSS) : 0,
        canvases: survey(root, env),
      };
    }
  }
}

out.kinds = KINDS;
process.stdout.write(JSON.stringify(out));
