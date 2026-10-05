/*
 * Mount every widget in apps/web once, and report what happened.
 *
 * The widgets with a Node runner (errors.js, interfere.js) are tested properly.
 * The rest were tested by grepping their own source, because "mount() needs a
 * canvas and a 2D context, and there is no jsdom here" -- which was true when
 * those tests were written and stopped being true once tests/dom_stub.js existed.
 *
 * This is the cheap end of fixing that: not a behavioural test, but proof that a
 * widget still *loads and mounts* -- that its module scope evaluates, its CSS
 * injects, its DOM assembles, and its first paint runs without throwing. That is
 * the class of breakage a refactor of the shared canvas code would cause, and
 * nothing could see it before.
 *
 * The context stub is a recorder rather than a set of no-ops, so a widget that
 * reaches for a method nobody anticipated shows up as an unknown call instead of
 * a silent success. Anything genuinely absent throws, which is the dom_stub
 * convention and the right one.
 */
const fs = require("fs");
const path = require("path");
const { makeEnv, loadWidget } = require("./dom_stub.js");

const WEB = path.join(__dirname, "..", "apps", "web");

/* Widgets that mount into a container. The data bundles (d2nn_weights.js and
 * friends) and the pure-computation modules (asm.js, d2nn.js, analogy_geom.js)
 * publish no mount() and are exercised elsewhere. */
const WIDGETS = [
  { file: "errors.js", opts: { kind: "phase" }, globals: ["PhotonnErrors"] },
  { file: "interfere.js", opts: {}, globals: ["PhotonnInterfere"] },
  { file: "explorer.js", opts: {}, globals: ["PhotonnExplorer"] },
  { file: "digit_source.js", opts: {}, globals: ["PhotonnDigitSource"] },
  { file: "d2nn_stage.js", opts: {}, globals: ["PhotonnD2NNStage"] },
  { file: "scaling.js", opts: {}, globals: ["PhotonnScaling"] },
  { file: "optics.js", opts: {}, globals: ["PhotonnOptics"] },
  { file: "analogy.js", opts: {}, globals: ["PhotonnAnalogy"] },
  { file: "d2nn_demo.js", opts: {}, globals: ["PhotonnD2NN"] },
  { file: "d2nn_compare.js", opts: {}, globals: ["PhotonnD2NNCompare"] },
  // Network mode: it decodes the trained bundle and runs the chip, so it reaches
  // further than device mode, which test_activation_widget.py mounts on its own.
  { file: "activation.js", opts: { mode: "network" }, globals: ["PhotonnActivation"] },
];

/* Modules a widget reads off `window` at module scope. Loaded first, into the
 * same window, exactly as a built page emits them. */
/* Order matters and is the page's own: d2nn.js reads window.D2NN_WEIGHTS at
 * module scope and builds the default network there, so the weights bundle has
 * to be evaluated first. Listing it after produced a network of {} and a
 * "cannot read n_layers" three frames later -- the same module-scope ordering
 * hazard build_site.py guards for mesh_weights.js and errors.js. */
const DEPENDENCIES = {
  "errors.js": ["mesh_weights.js", "mesh.js", "error_mask.js"],
  "explorer.js": ["asm.js"],          // reads window.ASM inside compute()
  "digit_source.js": ["asm.js", "d2nn_weights.js", "d2nn.js"],
  "d2nn_stage.js": ["asm.js", "d2nn_weights.js", "d2nn.js"],
  "d2nn_demo.js": ["asm.js", "d2nn_weights.js", "d2nn.js"],
  "d2nn_compare.js": ["asm.js", "d2nn_weights.js", "d2nn_deep_weights.js",
                      "d2nn.js", "digit_source.js"],
  "scaling.js": ["optics_sweep.js"],
  "optics.js": ["optics_sweep.js"],
  "analogy.js": ["analogy_geom.js"],
  // Read at mount rather than module scope, but emitted first on the page anyway.
  "activation.js": ["mesh.js", "deep_mesh_weights.js"],
};

function recordingCtx(calls) {
  const noop = () => {};
  const ctx = {
    canvas: { width: 0, height: 0 },
    save: noop, restore: noop, beginPath: noop, closePath: noop,
    moveTo: noop, lineTo: noop, arc: noop, rect: noop, ellipse: noop,
    fill: noop, stroke: noop, fillRect: noop, strokeRect: noop, clearRect: noop,
    fillText: noop, strokeText: noop, translate: noop, scale: noop, rotate: noop,
    transform: noop, setTransform: noop, setLineDash: noop, clip: noop,
    drawImage: noop, putImageData: noop, quadraticCurveTo: noop, bezierCurveTo: noop,
    createLinearGradient: () => ({ addColorStop: noop }),
    createRadialGradient: () => ({ addColorStop: noop }),
    createImageData: (w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
    getImageData: (x, y, w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
    measureText: () => ({ width: 10 }),
  };
  return new Proxy(ctx, {
    get(target, prop) {
      if (prop in target) return target[prop];
      if (typeof prop === "string" && !prop.startsWith("_")) calls.push(prop);
      return undefined;
    },
    set(target, prop, value) { target[prop] = value; return true; },
  });
}

const results = [];

for (const w of WIDGETS) {
  const unknown = [];
  const env = makeEnv({
    dpr: 2,
    layoutWidth: () => 720,
    ctxStub: () => recordingCtx(unknown),
  });
  // The page script and mount queue are not present; widgets that look for them
  // must cope, which is itself part of the contract.
  env.win.matchMedia = () => ({ matches: false, addEventListener() {}, addListener() {} });
  env.win.requestAnimationFrame = () => 0;
  env.win.cancelAnimationFrame = () => {};
  env.win.setTimeout = (fn) => { return 0; };
  env.win.clearTimeout = () => {};
  env.win.getComputedStyle = () => ({ getPropertyValue: () => "" });
  env.doc.documentElement = { getAttribute: () => null, style: {} };
  env.doc.addEventListener = () => {};
  env.win.MutationObserver = function () { return { observe() {}, disconnect() {} }; };
  env.win.ResizeObserver = undefined;

  const atob = (b64) => Buffer.from(b64, "base64").toString("binary");
  // Widgets call these bare, not as window.*, and `new Function` gives them no
  // global scope to find them in -- so they are bound as parameters.
  const extras = {
    atob,
    getComputedStyle: env.win.getComputedStyle,
    requestAnimationFrame: env.win.requestAnimationFrame,
    cancelAnimationFrame: env.win.cancelAnimationFrame,
    setTimeout: env.win.setTimeout,
    clearTimeout: env.win.clearTimeout,
    MutationObserver: env.win.MutationObserver,
    ResizeObserver: function () { return { observe() {}, disconnect() {} }; },
    performance: { now: () => 0 },
  };

  let status = "ok", detail = "";
  try {
    // plot.js before everything: every drawing widget reads window.PhotonnPlot at
    // module scope, which is the load order a built page emits.
    for (const dep of ["plot.js", ...(DEPENDENCIES[w.file] || [])]) {
      const p = path.join(WEB, dep);
      if (!fs.existsSync(p)) continue;          // error_mask.js is built inline
      loadWidget(fs.readFileSync(p, "utf8"), env, extras);
    }
    const api = loadWidget(fs.readFileSync(path.join(WEB, w.file), "utf8"), env, extras);
    const mount = (api && api.mount) || (w.globals.map((g) => env.win[g]).find(Boolean) || {}).mount;
    if (typeof mount !== "function") {
      status = "no-mount";
    } else {
      // In the document, as a real host div is: explorer.js looks its own
      // controls back up with document.getElementById, which only works if the
      // container is actually attached.
      const host = env.makeEl("div");
      env.body.appendChild(host);
      mount(host, w.opts);
      if (!host.children.length) { status = "empty"; detail = "mounted but built no DOM"; }
    }
  } catch (e) {
    status = "threw";
    detail = String(e && e.message).split("\n")[0];
  }
  results.push({ widget: w.file, status, detail, unknownCtxCalls: [...new Set(unknown)] });
}

process.stdout.write(JSON.stringify({ results }, null, 1));
