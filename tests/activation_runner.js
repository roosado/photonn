/*
 * Node runner for the activation widget (apps/web/activation.js).
 *
 * Mounts both modes against the DOM stand-in in tests/dom_stub.js, at three
 * widths and two pixel ratios, and dumps what tests/test_activation_widget.py
 * needs to hold the widget to the physics it draws:
 *
 *   - the curve the device mode samples, for several (alpha, g, phi_b), so the
 *     test can compare it with the closed form pointwise;
 *   - field() on 2 000 seeded complex amplitudes, so the test can rebuild each
 *     one from photonn.mzi's beamsplitter and phase_shifter;
 *   - the landmarks and the readout text, so the printed numbers can be checked
 *     against the numbers they claim to be;
 *   - every canvas's bitmap and display size, for the aspect check.
 *
 * The plot cap is read off the widget's own stylesheet rather than restated.
 * Prints one JSON object.
 */
const fs = require("fs");
const path = require("path");
const { makeEnv: makeStubEnv, loadWidget } = require("./dom_stub.js");

const WEB = path.join(__dirname, "..", "apps", "web");
const SOURCE = fs.readFileSync(path.join(WEB, "activation.js"), "utf8");
const PLOT = fs.readFileSync(path.join(WEB, "plot.js"), "utf8");
const MESH = fs.readFileSync(path.join(WEB, "mesh.js"), "utf8");
const BUNDLE = require(path.join(WEB, "deep_mesh_weights.js"));

function readPlotCap(src) {
  const css = /const CSS = `([\s\S]*?)`;/.exec(src)[1];
  const rule = /\.ea-plot\{([^}]*)\}/.exec(css);
  const m = rule && /max-width:\s*(\d+(?:\.\d+)?)px/.exec(rule[1]);
  return m ? parseFloat(m[1]) : Infinity;
}
const PLOT_CAP = readPlotCap(SOURCE);
const atob = (b64) => Buffer.from(b64, "base64").toString("binary");

function makeEnv(containerWidth, deviceRatio) {
  return makeStubEnv({
    dpr: deviceRatio,
    layoutWidth(node) {
      const pane = node.tagName === "CANVAS" ? node.parentNode : node;
      if (pane && pane.className === "ea-plot") return Math.min(containerWidth, PLOT_CAP);
      return containerWidth;
    },
    ctxStub() {
      return {
        fillStyle: "", strokeStyle: "", font: "", textAlign: "", lineWidth: 1,
        fillRect() {}, fillText() {}, setTransform() {}, beginPath() {},
        moveTo() {}, lineTo() {}, stroke() {}, setLineDash() {}, arc() {}, fill() {},
        measureText: (t) => ({ width: String(t).length * 6 }),
        createImageData: (w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
        putImageData() {},
      };
    },
  });
}

function load(env, withBundle) {
  if (withBundle) env.win.PHOTONN_DEEP_MESH = BUNDLE;
  loadWidget(PLOT, env, {});
  loadWidget(MESH, env, { atob });
  loadWidget(SOURCE, env, { atob });
  return env.win.PhotonnActivation;
}

function canvases(root) {
  const out = [];
  const walk = (n) => {
    if (n.tagName === "CANVAS" && n.parentNode && n.parentNode.className === "ea-plot") {
      out.push({ bitmapW: n.width, bitmapH: n.height,
                 shownW: n.getBoundingClientRect().width,
                 styleH: n.style.height ? parseFloat(n.style.height) : null });
    }
    n.children.forEach(walk);
  };
  walk(root);
  return out;
}

const out = { plotCap: PLOT_CAP, layout: {}, curves: [], field: {}, landmarks: {},
              readout: {}, errors: {} };

const DEVICE = { alpha: 0.1, g_phi: 157.07963267948966, phi_b: Math.PI };

for (const ratio of [1, 2]) {
  for (const width of [300, 480, 1042]) {
    for (const mode of ["device", "network"]) {
      const env = makeEnv(width, ratio);
      const api = load(env, true);
      const host = env.makeEl("div");
      api.mount(host, mode === "device" ? Object.assign({ mode: mode }, DEVICE) : { mode: mode });
      out.layout[mode + ":" + width + "x" + ratio] = canvases(host.children[0]);
    }
  }
}

const env = makeEnv(640, 1);
const api = load(env, true);

// The curve, as the device mode samples it, for a spread of devices.
for (const [a, g, b] of [[0.1, 157.08, Math.PI], [0.1, 157.08, 0], [0.2, 3.0, 0.85 * Math.PI],
                         [0.05, 0.5, 0.5 * Math.PI]]) {
  const [lo, hi] = api.axisRange(g, b);
  const s = api.samples(a, g, b, lo, hi, 400);
  out.curves.push({ alpha: a, g: g, phiB: b, lo: lo, hi: hi,
                    p: Array.from(s.p), t: Array.from(s.t) });
}

// field() on seeded draws (a small LCG, so the draw is reproducible anywhere).
let state = 20261005;
const rnd = () => ((state = (1103515245 * state + 12345) % 2147483648) / 2147483648);
const draws = [];
for (let i = 0; i < 2000; i++) {
  const re = (rnd() - 0.5) * 6, im = (rnd() - 0.5) * 6;
  const a = rnd() * 0.5, g = rnd() * 2, b = rnd() * 2 * Math.PI;
  draws.push([re, im, a, g, b, api.field(re, im, a, g, b)]);
}
out.field.draws = draws;

// Low-power slope at phi_b = pi, over [1e-3, 1e-2] * pi/g, and the endpoints by name.
const g = 0.3, a = 0.1;
out.landmarks.pi = api.landmarks(a, g, Math.PI);
out.landmarks.zero = api.landmarks(a, g, 0);
out.landmarks.tOpenPi = api.transfer(Math.PI / g, a, g, Math.PI);
out.landmarks.tWeakZero = api.transfer(1e-9, a, g, 0);
const p1 = 1e-3 * Math.PI / g, p2 = 1e-2 * Math.PI / g;
out.landmarks.slopePi = Math.log10(api.transfer(p2, a, g, Math.PI) / api.transfer(p1, a, g, Math.PI))
  / Math.log10(p2 / p1);

// What the device mode tells the reader, at the trained device and with weak light passing.
// "nearDark" is pi rounded to a 0.01 step -- what a browser makes of pi on a 0.01
// slider. This stub does not snap values, so the runner reports the bias slider's
// step too, and the test holds it to "any".
for (const [name, opts] of [["trained", DEVICE], ["passing", { alpha: 0.1, g_phi: 157.08, phi_b: 0 }],
                            ["nearDark", { alpha: 0.1, g_phi: 157.08, phi_b: 3.14 }]]) {
  const host = env.makeEl("div");
  const m = api.mount(host, Object.assign({ mode: "device" }, opts));
  const inputs = [];
  const walk = (n) => { if (n.tagName === "INPUT") inputs.push(n); n.children.forEach(walk); };
  walk(host);
  out.readout[name] = { note: m.note.innerHTML, landmarks: m.landmarks(),
                        steps: inputs.map((i) => i.step) };
}
// The network mode's note at the design point, gallery digit 0.
{
  const host = env.makeEl("div");
  const m = api.mount(host, { mode: "network" });
  out.readout.network = { note: m.note.innerHTML };
}

// No constants, no bundle: it must refuse rather than invent a device.
{
  const bare = makeEnv(640, 1);
  const apiBare = load(bare, false);
  try {
    apiBare.mount(bare.makeEl("div"), { mode: "device" });
    out.errors.device = null;
  } catch (e) {
    out.errors.device = String(e.message);
  }
}

process.stdout.write(JSON.stringify(out));
