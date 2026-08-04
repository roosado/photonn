/*
 * Node runner for the 3D stage's projection basis.
 *
 * Loads apps/web/d2nn_stage.js (its `basis` is pure arithmetic -- no DOM) and
 * reports the invariants tests/test_stage_projection.py asserts. Kept as a runner
 * rather than a port so the *shipped* code is what gets checked.
 */
const path = require("path");
const STAGE = require(path.join(__dirname, "..", "apps", "web", "d2nn_stage.js"));

const DEG = Math.PI / 180;
const ANGLES = [[0, 0], [34, 19], [10, -25], [80, 55], [45, 0], [0, 40]];

const rows = ANGLES.map(([tDeg, pDeg]) => {
  const b = STAGE.basis(tDeg * DEG, pDeg * DEG);
  const n2 = (v) => v[0] * v[0] + v[1] * v[1];
  return {
    theta: tDeg,
    phi: pDeg,
    eX: Array.from(b.eX),
    eY: Array.from(b.eY),
    eZ: Array.from(b.eZ),
    // Projecting the three columns of a rotation matrix onto a plane leaves the
    // sum of their squared norms equal to 2 -- the one identity that survives the
    // dropped dimension, and the cheapest proof the basis is a real projection.
    sumSq: n2(b.eX) + n2(b.eY) + n2(b.eZ),
  };
});

process.stdout.write(JSON.stringify({ rows }));
