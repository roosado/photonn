/*
 * Node runner for the JS<->Python optics-sweep geometry cross-check.
 *
 * The widget (apps/web/optics.js) recomputes the diffractive reach live from the
 * closed form rather than reading it from the data bundle, so it can move
 * continuously with its slider. That makes it a second implementation of
 * photonn.propagate.diffraction_reach_px, and the two must not drift.
 *
 * Prints one JSON object: the JS reach at each requested (z, n, dx, lambda), plus
 * the bundle's own stored values so the test can check the shipped data agrees
 * with the shipped code. Driven by tests/test_optics_sweep.py.
 */
const path = require("path");

const WEB = path.join(__dirname, "..", "apps", "web");
const OPTICS = require(path.join(WEB, "optics.js"));
const DATA = require(path.join(WEB, "optics_sweep.js"));

const cases = JSON.parse(process.argv[2] || "[]");
const reach = cases.map((c) =>
  OPTICS.reachPerHop(c.z_mm, { dx: c.dx, wavelength: c.wavelength })
);

process.stdout.write(JSON.stringify({
  reach: reach,
  bundle: {
    grid: DATA.grid,
    dx: DATA.dx,
    wavelength: DATA.wavelength,
    required_px: DATA.required_px,
    z_crit_mm: DATA.z_crit_mm,
    points: (DATA.points || []).map((p) => ({
      z_mm: p.z_mm,
      layers: p.layers,
      reach_hop: p.reach_hop,
      reach_total: p.reach_total,
      acc: p.acc,
    })),
  },
}));
