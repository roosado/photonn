/*
 * Node runner for the JS<->torch D2NN cross-check.
 *
 * Loads the browser network (apps/web/d2nn.js, which pulls in asm.js and the
 * generated d2nn_weights.js) and the torch-generated fixture
 * (tests/fixtures/d2nn_reference.json), classifies each frozen digit, and prints
 * one JSON summary per case. Driven by tests/test_d2nn_crosscheck.py.
 */
const path = require("path");
const fs = require("fs");

const NET = require(path.join(__dirname, "..", "apps", "web", "d2nn.js"));
const fixture = JSON.parse(
  fs.readFileSync(path.join(__dirname, "fixtures", "d2nn_reference.json"), "utf8")
);

function toFloat(bytes) {
  const out = new Float64Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) out[i] = bytes[i] / 255;
  return out;
}

// 1. full forward pass: predictions and logits against torch
const cases = [];
for (const c of fixture.cases) {
  const res = NET.classify(toFloat(c.image28));
  let maxLogitErr = 0;
  for (let i = 0; i < 10; i++) {
    const e = Math.abs(res.logits[i] - c.logits[i]);
    if (e > maxLogitErr) maxLogitErr = e;
  }
  cases.push({
    index: c.index,
    label: c.label,
    predJs: res.pred,
    predRef: c.pred,
    maxLogitErr,
  });
}

// 2. the bilinear resize against torch's align_corners=false canvases
const resize = [];
for (const c of fixture.resize_cases) {
  const win = NET.resize(toFloat(c.image28), NET.DIGIT, fixture.window.size);
  let maxErr = 0;
  for (let i = 0; i < win.length; i++) {
    const e = Math.abs(win[i] - c.window[i]);
    if (e > maxErr) maxErr = e;
  }
  resize.push({ index: c.index, maxErr });
}

// 3. the geometry the JS actually runs, so the test can catch a stale bundle
const geometry = {
  n: NET.N,
  dx: NET.weights.dx,
  wavelength: NET.weights.wavelength,
  separation: NET.weights.separation,
  n_layers: NET.weights.n_layers,
  readout_gain: NET.weights.readout_gain,
  phase_scale: NET.weights.phase_scale,
  input_frac: NET.weights.input_frac,
};

// 4. accuracy over the digits shipped in the browser gallery
let galleryCorrect = 0;
for (let k = 0; k < NET.nGallery; k++) {
  if (NET.classify(NET.galleryDigit(k)).pred === NET.galleryLabel(k)) galleryCorrect++;
}

process.stdout.write(JSON.stringify({
  cases,
  resize,
  geometry,
  gallery: { correct: galleryCorrect, total: NET.nGallery },
}));
