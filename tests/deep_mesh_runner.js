/*
 * Node runner for the activation page's live chip (activation.js, network mode).
 *
 * The widget runs the trained two-layer chip out of deep_mesh_weights.js through
 * mesh.js. That is only worth showing if it is the chip PyTorch trained and MATLAB
 * reproduced, so this dumps what the shipped code computes -- the decoded
 * parameters, and every gallery digit's logits, prediction and per-mode power into
 * the activation at the design power and at ten times it -- for
 * tests/test_deep_mesh_web.py to compare with photonn.mzi.deep_mesh_forward and
 * with tests/fixtures/deep_mesh_reference.json.
 *
 * Prints one JSON object.
 */
const fs = require("fs");
const path = require("path");
const { makeEnv, loadWidget } = require("./dom_stub.js");

const WEB = path.join(__dirname, "..", "apps", "web");
const bundle = require(path.join(WEB, "deep_mesh_weights.js"));
const atob = (b64) => Buffer.from(b64, "base64").toString("binary");

// The forward pass touches no DOM; the layout and context stubs are never reached.
const env = makeEnv({ dpr: 1, layoutWidth: () => 640, ctxStub: () => ({}) });
env.win.PHOTONN_DEEP_MESH = bundle;
loadWidget(fs.readFileSync(path.join(WEB, "plot.js"), "utf8"), env, {});
loadWidget(fs.readFileSync(path.join(WEB, "mesh.js"), "utf8"), env, { atob });
loadWidget(fs.readFileSync(path.join(WEB, "activation.js"), "utf8"), env, { atob });

const A = env.win.PhotonnActivation;
const net = A._net.load(bundle);
const P0 = bundle.operating_point.input_power_w;

const out = {
  decoded: {
    theta: Array.from(net.decoded.theta), phi: Array.from(net.decoded.phi),
    sigma: Array.from(net.decoded.sigma), outPhase: Array.from(net.decoded.outPhase),
  },
  inputs: Array.from(net.inputs),
  digits: [],
};
for (let i = 0; i < net.labels.length; i++) {
  const r = A._net.forward(net, i, P0);
  const r10 = A._net.forward(net, i, 10 * P0);
  out.digits.push({
    logits: Array.from(r.logits), pred: r.pred,
    taps: Array.from(r.taps[0]), taps10: Array.from(r10.taps[0]),
  });
}
process.stdout.write(JSON.stringify(out));
