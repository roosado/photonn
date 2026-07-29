/*
 * Node runner for the JS<->Python angular-spectrum cross-check.
 *
 * Loads the browser physics module (apps/web/asm.js) and the NumPy-generated
 * fixture (tests/fixtures/asm_reference.json), recomputes each case, and prints
 * one JSON summary per case (max abs error on peak-normalized intensity, plus the
 * JS sampling-distance values). Driven by tests/test_asm_crosscheck.py.
 */
const path = require("path");
const fs = require("fs");

const ASM = require(path.join(__dirname, "..", "apps", "web", "asm.js"));
const fixture = JSON.parse(
  fs.readFileSync(path.join(__dirname, "fixtures", "asm_reference.json"), "utf8")
);

const results = [];
for (const c of fixture.cases) {
  const I = ASM.propagateIntensity(c.n, c.dx, c.wavelength, c.z, c.shape, c.size);
  let peak = 0;
  for (let i = 0; i < I.length; i++) if (I[i] > peak) peak = I[i];

  const ref = c.intensity_norm;
  let maxErr = 0;
  for (let i = 0; i < I.length; i++) {
    const v = peak > 0 ? I[i] / peak : I[i];
    const e = Math.abs(v - ref[i]);
    if (e > maxErr) maxErr = e;
  }

  const zCritJs = ASM.zCrit(c.n, c.dx, c.wavelength);
  results.push({
    name: c.name,
    n: c.n,
    maxErr,
    zCritJs,
    zCritRef: c.z_crit,
    samplingOkRef: c.sampling_ok,
    samplingOkJs: Math.abs(c.z) <= zCritJs,
  });
}
process.stdout.write(JSON.stringify(results));
