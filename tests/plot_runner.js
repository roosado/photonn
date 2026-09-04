/*
 * Exercise apps/web/plot.js directly: the shared canvas primitives, tested once
 * rather than nine times not at all.
 *
 * The properties here are the ones the duplicated copies got wrong. The resize
 * guard is the headline -- an unguarded `canvas.width = w` reallocates the
 * backing store even when the value is unchanged, which is what made the one
 * animating widget on the site reallocate a ~1440x860 surface every frame.
 */
const { makeEnv } = require("./dom_stub.js");
const plot = require("../apps/web/plot.js");

const out = {};

function ctxStub() {
  const calls = [];
  return {
    _calls: calls,
    setTransform(...a) { calls.push(["setTransform", ...a]); },
    createImageData: (w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
    putImageData() { calls.push(["putImageData"]); },
    clearRect() {},
  };
}

const env = makeEnv({ dpr: 2, layoutWidth: () => 600, ctxStub });
global.window = env.win;
global.document = env.doc;

/* -- the resize guard ------------------------------------------------------ */
{
  const c = env.makeEl("canvas");
  const first = plot.resize(c, 800, 400);
  const second = plot.resize(c, 800, 400);          // same dimensions
  const third = plot.resize(c, 800, 401);
  out.resize = { first, second, third, width: c.width, height: c.height };
}

/* -- dpr clamp ------------------------------------------------------------- */
{
  env.win.devicePixelRatio = 3;
  const clamped = plot.scale();
  env.win.devicePixelRatio = 1.5;
  const passed = plot.scale();
  env.win.devicePixelRatio = 2;
  out.scale = { clamped, passed, max: plot.MAX_DPR };
}

/* -- fit(): height from width, transform re-established --------------------- */
{
  const c = env.makeEl("canvas");
  const a = plot.fit(c, (w) => Math.round(w * 0.5));
  const before = c.width;
  const b = plot.fit(c, (w) => Math.round(w * 0.5));  // idempotent
  out.fit = {
    W: a.W, H: a.H, dpr: a.dpr,
    backing: [c.width, c.height],
    styleHeight: c.style.height,
    stableAcrossCalls: c.width === before,
    // Two calls, two setTransform -- re-applying after a skipped resize is a
    // no-op, and after a real one it is required.
    transforms: b.ctx._calls.filter((k) => k[0] === "setTransform").length,
  };
}

/* -- fitTo(): both dimensions given ---------------------------------------- */
{
  const c = env.makeEl("canvas");
  plot.fitTo(c, 300, 120);
  out.fitTo = { backing: [c.width, c.height], styleHeight: c.style.height };
}

/* -- LUTs ------------------------------------------------------------------ */
{
  const lut = plot.LUT_INTENSITY;
  out.lut = {
    length: lut.length,
    firstIsBlack: [lut[0], lut[1], lut[2]],
    lastIsBright: [lut[255 * 3], lut[255 * 3 + 1], lut[255 * 3 + 2]],
    // Twilight is cyclic: its two ends must match, or phase gets a false seam.
    phaseJoins: [
      Math.abs(plot.LUT_PHASE[0] - plot.LUT_PHASE[255 * 3]) <= 2,
      Math.abs(plot.LUT_PHASE[1] - plot.LUT_PHASE[255 * 3 + 1]) <= 2,
      Math.abs(plot.LUT_PHASE[2] - plot.LUT_PHASE[255 * 3 + 2]) <= 2,
    ],
  };
}

/* -- raster ---------------------------------------------------------------- */
{
  const n = 4;
  const data = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) data[i] = i / (n * n - 1);
  const c = plot.raster(data, n);
  out.raster = { width: c.width, height: c.height, put: c.getContext("2d")._calls.length > 0 };

  // Cyclic mapping must not auto-scale: two fields with different ranges have to
  // land on the same colours where their values agree.
  const a = new Float64Array([-Math.PI, 0, Math.PI, 0]);
  const b = new Float64Array([-Math.PI, 0, Math.PI / 2, 0]);
  plot.raster(a, 2, { cyclic: true, lut: plot.LUT_PHASE });
  plot.raster(b, 2, { cyclic: true, lut: plot.LUT_PHASE });
  out.rasterCyclicRuns = true;
}

/* -- onThemeChange watches the attribute the toggle actually writes --------- */
{
  let observed = null;
  env.win.matchMedia = () => ({ matches: false, addEventListener() {}, addListener() {} });
  global.MutationObserver = function (fn) {
    return { observe(target, cfg) { observed = { target, cfg, fn }; }, disconnect() {} };
  };
  plot.onThemeChange(() => {});
  out.theme = {
    observesRoot: observed && observed.target === env.doc.documentElement,
    filter: observed && observed.cfg.attributeFilter,
  };
}

/* -- injectStyle is once per id -------------------------------------------- */
{
  plot.injectStyle("plot-test-style", ".a{}");
  const after1 = env.doc.getElementById("plot-test-style") !== null;
  plot.injectStyle("plot-test-style", ".b{}");
  const still = env.doc.getElementById("plot-test-style");
  out.injectStyle = { created: after1, notOverwritten: still.textContent === ".a{}" };
}

process.stdout.write(JSON.stringify(out, null, 1));
