/*
 * Node runner for the chip widget's operator build (apps/web/errors.js, KINDS.mesh).
 *
 * The /chip widget draws |U . diag(sigma) . V| for the trained mesh, before and
 * after a coupler-imbalance draw. That picture is only worth anything if the
 * matrix behind it is the one the trained chip actually realises, and the ways to
 * get it silently wrong are all cheap: a transposed index, the closed-form 2x2
 * block instead of the factored one, a Clements schedule that starts on the wrong
 * parity, an output phase screen applied on the wrong side.
 *
 * So this dumps the operator the shipped code builds, and tests/test_mesh_web.py
 * checks it against the same thing built in NumPy from photonn.mzi.
 *
 * Prints one JSON object.
 */
const fs = require("fs");
const path = require("path");

const WEB = path.join(__dirname, "..", "apps", "web");
const weights = require(path.join(WEB, "mesh_weights.js"));
const src = fs.readFileSync(path.join(WEB, "errors.js"), "utf8");

// errors.js only touches the DOM when a widget mounts; the operator build does
// not, so the stubs here need to be no more than enough to let the file load.
const doc = {
  getElementById: () => null,
  createElement: () => ({ appendChild() {}, style: {}, setAttribute() {} }),
  head: { appendChild() {} },
};
const win = {
  document: doc, devicePixelRatio: 1, addEventListener() {},
  PHOTONN_MESH: weights,
};
new Function("window", "document", "module", "atob", src)(
  win, doc, { exports: {} }, (b64) => Buffer.from(b64, "base64").toString("binary"));

const probe = win.PhotonnErrors._mesh;
const M = probe.load();
const n = M.n;

/** The widget's own imbalance draw, reproduced exactly. */
function splitsFor(eps) {
  const s = new Float64Array(4 * M.nMzi);
  for (let i = 0; i < s.length; i++) s[i] = 0.5 + eps * M.splitNoise[i];
  return s;
}

const out = {
  n: n,
  nMzi: M.nMzi,
  sigma: Array.from(M.sigma),
  theta: Array.from(M.theta),
  phi: Array.from(M.phi),
  outPhase: Array.from(M.outV).concat(Array.from(M.outU)),
  ideal: Array.from(probe.operatorMag(M, n, M.sigma, null)),
  // An imbalance of exactly zero is still a *different code path* -- the factored
  // block with both splits at 0.5 rather than no splits at all -- so it has to
  // land on the same matrix, or the ideal panel is not the ideal chip.
  zeroEps: Array.from(probe.operatorMag(M, n, M.sigma, splitsFor(0))),
  stressed: Array.from(probe.operatorMag(M, n, M.sigma, splitsFor(0.01))),
};
process.stdout.write(JSON.stringify(out));
