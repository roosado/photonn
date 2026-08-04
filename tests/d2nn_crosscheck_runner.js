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

// 5. the 3D stage's intermediate slices are display-only and must not be able to
//    move a class score: classify() before and after must be bit-identical, and
//    the slices landing on the mask planes must reproduce the canonical hops.
const slices = [];
for (let k = 0; k < Math.min(4, NET.nGallery); k++) {
  const digit = NET.galleryDigit(k);
  const before = NET.classify(digit);
  const field = NET.encodeInput(before.canvas);
  const S = 4;
  const walk = NET.sliceForward(field[0], field[1], S);
  const after = NET.classify(digit);

  let logitDelta = 0;
  for (let i = 0; i < 10; i++) {
    const d = Math.abs(before.logits[i] - after.logits[i]);
    if (d > logitDelta) logitDelta = d;
  }

  // Relative agreement between each sub-stepped mask plane and forward()'s own.
  let planeRel = 0;
  for (let Lz = 0; Lz < NET.weights.n_layers; Lz++) {
    const got = walk[(Lz + 1) * S].I;
    const ref = before.planes[Lz];
    let err = 0, peak = 0;
    for (let i = 0; i < ref.length; i++) {
      const e = Math.abs(got[i] - ref[i]);
      if (e > err) err = e;
      if (ref[i] > peak) peak = ref[i];
    }
    const rel = peak > 0 ? err / peak : 0;
    if (rel > planeRel) planeRel = rel;
  }
  let detErr = 0, detPeak = 0;
  const last = walk[walk.length - 1].I;
  for (let i = 0; i < last.length; i++) {
    const e = Math.abs(last[i] - before.intensity[i]);
    if (e > detErr) detErr = e;
    if (before.intensity[i] > detPeak) detPeak = before.intensity[i];
  }

  slices.push({
    index: k,
    predBefore: before.pred,
    predAfter: after.pred,
    logitDelta,
    planeRel,
    detectorRel: detPeak > 0 ? detErr / detPeak : 0,
    count: walk.length,
    expectedCount: (NET.weights.n_layers + 1) * S + 1,
    firstZ: walk[0].z,
    lastZ: walk[walk.length - 1].z,
    totalZ: (NET.weights.n_layers + 1) * NET.weights.separation,
  });
}

process.stdout.write(JSON.stringify({
  cases,
  resize,
  geometry,
  gallery: { correct: galleryCorrect, total: NET.nGallery },
  slices,
}));
