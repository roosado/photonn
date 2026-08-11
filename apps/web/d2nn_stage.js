/*
 * d2nn_stage.js -- the diffractive stack in three dimensions.
 *
 * The filmstrip beside this widget is accurate but flat: a row of squares asking
 * you to imagine they are the same light further along. This draws the machine
 * instead -- the entrance plane, the phase masks and the detector plane as
 * parallel panels receding along the optical axis, each carrying the light
 * actually computed on it, so the geometry reads at a glance.
 *
 * Three facts make it cheap and honest:
 *
 *  - An orthographic projection of a flat plane is *affine*, so one
 *    ctx.transform + drawImage renders a 128x128 plane as a correct
 *    parallelogram. No WebGL, no library.
 *  - The panels are parallel and never intersect, so back-to-front painting is
 *    exact occlusion -- no depth buffer.
 *  - The light *between* masks is real, not decorative: sub-stepping a hop is
 *    exact while z < z_crit (see d2nn.js:sliceForward).
 *
 * What is *not* drawn: rays. Scalar diffraction is not ray optics, and straight
 * lines from digit to detector would misrepresent the physics this project is
 * about. Travel is conveyed by real slices and by the sweep, not by fiction.
 *
 * The stack is far longer than it is wide -- 18 mm across a 1.02 mm aperture for
 * the shipped design, 30 mm for a 56-mask one -- so drawn to scale it is an
 * unreadable needle. The depth axis is compressed, and the figure says so on its
 * face.
 *
 * Three things scale with depth, and all three are derived from the bundle rather
 * than assumed, because a 56-mask network is not a 5-mask one drawn eleven times:
 *
 *  - **How many masks are drawn.** Fifty-six panels at 0.53 mm spacing are
 *    fifty-six near-identical pictures. Above MAX_MASK_PANELS the stack is
 *    sampled, and every label then names its *true* index ("mask 23 of 56") so
 *    the sampling can never be mistaken for the whole stack.
 *  - **How finely a hop is sub-stepped.** Sub-stepping shows travel only if the
 *    light actually spreads within a hop; at 2.19 px per hop it does not, so
 *    subSteps falls to 1. See defaultSubSteps.
 *  - **Whether it follows the pen.** A deep stack re-renders on a button press
 *    instead of on every stroke, so frame rate stops mattering. See MODE.
 *
 * Usage:  var stage = window.PhotonnD2NNStage.mount(container, opts);
 *         stage.setResult(res, {trueLabel: 7});   // res from a d2nn.js network
 *         stage.setDigit(img28, meta);            // or hand it the digit instead,
 *                                                 // which lets a deep stack defer
 *                                                 // the forward pass as well
 * opts (all optional):
 *   net       -- the network to draw (from d2nn.js:buildNet). Defaults to the
 *                shipped one, so one page can hold a stage per model.
 *   mode      -- "live" | "manual". Default: derived from depth.
 *   theta, phi, beam, subSteps -- degrees, degrees, bool, int (all overrides).
 *
 * Depends on window.PhotonnD2NN_Net (d2nn.js). Pure rendering: no physics of its
 * own beyond calling sliceForward, no network, no libraries.
 */
(function () {
  "use strict";

  // Backing-store scale, capped at 2x.
  //
  // A dpr-3 phone would otherwise get 2.25x the pixels of a dpr-2 one for a
  // difference nobody can see at arm's length, and the cost is quadratic in the
  // canvas area -- the 3D stage re-rasterises ~64 drawImage calls at
  // imageSmoothingQuality "high" on every orbit frame, so this is the difference
  // between a smooth orbit and a slideshow on exactly the devices least able to
  // afford it.
  const MAX_DPR = 2;
  function canvasScale() { return Math.min(window.devicePixelRatio || 1, MAX_DPR); }

  const NET = (typeof window !== "undefined" && window.PhotonnD2NN_Net)
    ? window.PhotonnD2NN_Net
    : (typeof require !== "undefined" ? require("./d2nn.js") : null);

  // Same inferno LUT as the filmstrip and the diffraction explorer, so every
  // optical-intensity image on the site speaks one visual language.
  const INFERNO = [
    [0, 0, 4], [22, 11, 57], [66, 10, 104], [106, 23, 110], [147, 38, 103],
    [188, 55, 84], [221, 81, 58], [243, 120, 25], [252, 255, 164],
  ];
  // Cyclic map for phase: it must join end-to-end, because -pi and +pi are the
  // same setting of the same mask. A sequential map would draw a false seam.
  const TWILIGHT = [
    [226, 217, 226], [151, 180, 212], [76, 123, 189], [48, 63, 125], [24, 24, 45],
    [56, 32, 58], [120, 52, 84], [186, 88, 89], [222, 148, 116], [226, 217, 226],
  ];

  function makeLUT(anchors) {
    const lut = new Uint8ClampedArray(256 * 3);
    const seg = anchors.length - 1;
    for (let i = 0; i < 256; i++) {
      const t = i / 255 * seg;
      const k = Math.min(seg - 1, Math.floor(t));
      const f = t - k;
      const a = anchors[k], b = anchors[k + 1];
      lut[i * 3] = a[0] + (b[0] - a[0]) * f;
      lut[i * 3 + 1] = a[1] + (b[1] - a[1]) * f;
      lut[i * 3 + 2] = a[2] + (b[2] - a[2]) * f;
    }
    return lut;
  }
  const LUT_I = makeLUT(INFERNO);
  const LUT_P = makeLUT(TWILIGHT);

  const GAMMA = 0.5;          // sqrt stretch; detector intensity spans decades
  const DEPTH_SPAN = 6.0;     // drawn stack length, in half-aperture units
  const SWEEP_MS = 3200;
  // Optical intensity is read against black in this project and everywhere else,
  // and inferno needs a dark ground to hold contrast -- so the plates stay dark in
  // both page themes, exactly like the filmstrip's canvases below. Only the chrome
  // (labels, outlines, captions) follows the theme.
  const PLATE_INK = "#0b1018";
  const PLATE_ALPHA = 0.78;

  //: Most mask panels drawn. Beyond this the stack is sampled -- six planes
  //: spread through a deep stack tell the same story as fifty-six, and are the
  //: only version that is legible. The shipped 5-mask model is under the limit,
  //: so it is drawn whole exactly as before.
  const MAX_MASK_PANELS = 6;

  //: Sub-hop slices are only worth computing if the light visibly spreads across
  //: one. This is a *legibility* threshold, not a physical bound: sliceForward is
  //: exact at any subSteps (z stays well under z_crit), but below a few pixels of
  //: spreading consecutive slices are the same picture, so the extra propagations
  //: buy nothing. At the shipped 12.47 px per hop this yields the historical 4.
  const MIN_SLICE_REACH_PX = 3;
  const MAX_SUB_STEPS = 4;

  //: Depth at which the stage stops following the pen. Below it a re-render is
  //: comfortably inside a frame budget; above it, chasing every pointer move
  //: makes the whole page feel broken for a picture nobody reads mid-stroke.
  const LIVE_MAX_LAYERS = 8;

  /**
   * Diffractive reach of one hop, in pixels: z*lambda / (2*dx^2).
   *
   * The same closed form as photonn.propagate.diffraction_reach_px, which is what
   * the optics page plots and what the Phase-3 correspondence figure is built on.
   */
  function reachPerHopPx(W) {
    return W.separation * W.wavelength / (2 * W.dx * W.dx);
  }

  /** Sub-hops per hop for this geometry: 4 for the shipped stack, 1 for a deep one. */
  function defaultSubSteps(W) {
    const s = Math.floor(reachPerHopPx(W) / MIN_SLICE_REACH_PX);
    return Math.max(1, Math.min(MAX_SUB_STEPS, s));
  }

  /**
   * Which mask indices to draw, always including the first and the last.
   *
   * Returns every index when the stack is short enough, which is what keeps the
   * shipped model's figure byte-for-byte what it was. Evenly spread otherwise,
   * de-duplicated, so `n` is an upper bound rather than a promise.
   */
  function sampleMaskIndices(nLayers, maxPanels) {
    const m = Math.max(1, maxPanels == null ? MAX_MASK_PANELS : maxPanels);
    if (nLayers <= m) {
      const all = [];
      for (let i = 0; i < nLayers; i++) all.push(i);
      return all;
    }
    const out = [];
    for (let i = 0; i < m; i++) {
      const k = Math.round(i * (nLayers - 1) / (m - 1));
      if (out[out.length - 1] !== k) out.push(k);
    }
    return out;
  }

  // Unique across apps/web/*.js. It was "ds-style", which digit_source.js also
  // claims -- injectStyle() bails when the id is already there, so whichever
  // widget mounted second silently ran with no CSS of its own. That cost nothing
  // visible until a page carried both: on the optics page the stage lost, and its
  // toolbar unwrapped and its canvas collapsed to the 300px default. Guarded by
  // tests/test_web_style_ids.py.
  const STYLE_ID = "d2nn-stage-style";
  const CSS = `
.ds-root{--pe-fg:#1b1f24;--pe-muted:#5a6472;--pe-panel:#f4f6f9;--pe-border:#d7dde5;
  --pe-accent:#3b6ea5;--pe-ok:#3f8f4e;--pe-warn:#c14a3d;
  color:var(--pe-fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  display:flex;flex-direction:column;gap:12px;}
@media (prefers-color-scheme:dark){.ds-root{--pe-fg:#e6eaf0;--pe-muted:#9aa6b5;
  --pe-panel:#1c2128;--pe-border:#30363d;--pe-accent:#6ea8e0;--pe-ok:#5cc06e;--pe-warn:#e0705f;}}
.ds-box{background:var(--pe-panel);border:1px solid var(--pe-border);border-radius:10px;
  padding:12px 14px;}
.ds-bar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:10px;}
.ds-seg{display:flex;border:1px solid var(--pe-border);border-radius:7px;overflow:hidden;}
.ds-seg button{border:0;background:transparent;color:var(--pe-muted);padding:6px 12px;
  font:inherit;font-size:12px;font-weight:600;cursor:pointer;}
.ds-seg button[aria-pressed=true]{background:var(--pe-accent);color:#fff;}
.ds-btn{border:1px solid var(--pe-border);background:transparent;color:var(--pe-fg);
  border-radius:7px;padding:6px 12px;font:inherit;font-size:12px;cursor:pointer;}
.ds-btn:hover{border-color:var(--pe-accent);color:var(--pe-accent);}
.ds-btn[aria-pressed=true]{border-color:var(--pe-accent);color:var(--pe-accent);
  background:color-mix(in srgb,var(--pe-accent) 12%,transparent);}
.ds-btn:disabled{opacity:.5;cursor:default;}
.ds-spacer{flex:1 1 auto;}
.ds-canvas{display:block;width:100%;touch-action:none;cursor:grab;border-radius:8px;}
.ds-canvas.drag{cursor:grabbing;}
.ds-foot{display:flex;gap:14px;flex-wrap:wrap;justify-content:space-between;
  font-size:11.5px;color:var(--pe-muted);margin-top:8px;}
.ds-foot b{color:var(--pe-fg);font-variant-numeric:tabular-nums;font-weight:600;}
.ds-note{font-size:12px;color:var(--pe-muted);margin:0;}
`;

  function injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = CSS;
    document.head.appendChild(s);
  }

  function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }

  /**
   * Orthographic basis for azimuth `theta` and elevation `phi` (radians).
   *
   * Screen y grows downward, which is also how image rows are stored, so a plane
   * bitmap drawn on (eX, eY) comes out the right way up. eZ points right and down
   * as z increases, i.e. deeper planes are drawn nearer the viewer -- so painting
   * in ascending z is the exact painter's order.
   *
   * At theta = phi = 0 the depth axis collapses to zero and every panel lands on
   * top of the others; that degeneracy is the cheapest check the basis is right
   * (tests/test_stage_projection.py).
   */
  function basis(theta, phi) {
    const ct = Math.cos(theta), st = Math.sin(theta);
    const cp = Math.cos(phi), sp = Math.sin(phi);
    return {
      eX: [ct, -st * sp],
      eY: [0, cp],
      eZ: [st, ct * sp],
    };
  }

  /** Peak-normalised colour render of an n x n map into an offscreen canvas.
   *
   * Opaque, deliberately: the light planes are composited with "lighter", and
   * additive blending already makes darkness contribute nothing. Alpha-keying on
   * top of that would attenuate every mid-tone twice and dim the whole beam.
   */
  function renderBitmap(data, n, lut, gamma, cyclic) {
    const off = document.createElement("canvas");
    off.width = n; off.height = n;
    const ctx = off.getContext("2d");
    const img = ctx.createImageData(n, n);
    const d = img.data;

    let lo = 0, hi = 1;
    if (cyclic) { lo = -Math.PI; hi = Math.PI; }
    else {
      hi = 0;
      for (let i = 0; i < data.length; i++) if (data[i] > hi) hi = data[i];
      if (hi <= 0) hi = 1;
    }
    const span = hi - lo;

    for (let i = 0; i < n * n; i++) {
      let v = (data[i] - lo) / span;
      if (v < 0) v = 0; else if (v > 1) v = 1;
      if (gamma !== 1) v = Math.pow(v, gamma);
      const li = (v * 255) | 0;
      d[i * 4] = lut[li * 3];
      d[i * 4 + 1] = lut[li * 3 + 1];
      d[i * 4 + 2] = lut[li * 3 + 2];
      d[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return off;
  }

  function palette(root) {
    const cs = getComputedStyle(root);
    const get = (k, f) => (cs.getPropertyValue(k).trim() || f);
    return {
      fg: get("--pe-fg", "#1b1f24"),
      muted: get("--pe-muted", "#5a6472"),
      panel: get("--pe-panel", "#f4f6f9"),
      border: get("--pe-border", "#d7dde5"),
      accent: get("--pe-accent", "#3b6ea5"),
      warn: get("--pe-warn", "#c14a3d"),
    };
  }

  function mount(container, opts) {
    opts = opts || {};
    injectStyle();

    // The network being drawn. Defaulting to the module's shipped one keeps every
    // existing caller working; passing opts.net is what lets one page carry a
    // stage for the shipped model and another for a candidate.
    const MODEL = opts.net || NET;
    const W = MODEL.weights;
    const N = MODEL.N;
    const HOPS = W.n_layers + 1;
    const TOTAL_Z = HOPS * W.separation;              // 18 mm shipped, 30 mm at 56 masks
    const APERTURE = N * W.dx;                        // 1.024 mm
    // The drawn stack is DEPTH_SPAN half-apertures long; anything else would be
    // a needle. Report the factor rather than let the reader assume scale.
    const COMPRESSION = (TOTAL_Z / APERTURE) / (DEPTH_SPAN / 2);

    //: True mask indices drawn, and whether that is all of them. Everything the
    //: figure says about "which mask is this" is derived from these two.
    const MASK_IDX = sampleMaskIndices(W.n_layers, opts.maxMaskPanels);
    const SAMPLED = MASK_IDX.length < W.n_layers;
    const MODE = opts.mode || (W.n_layers <= LIVE_MAX_LAYERS ? "live" : "manual");

    const state = {
      theta: (opts.theta == null ? 34 : opts.theta) * Math.PI / 180,
      phi: (opts.phi == null ? 19 : opts.phi) * Math.PI / 180,
      view: "light",                 // "light" | "phase"
      beam: opts.beam !== false,
      subSteps: opts.subSteps || defaultSubSteps(W),
      sweep: null,                   // null, or {start} while playing
      front: 1,                      // wavefront position, 0..1 of the axis
      res: null,                     // what is currently drawn
      meta: null,
      pending: null,                 // in manual mode: arrived but not yet drawn
      bitmaps: null,                 // {panels, phases, slices}
      sliceTimer: 0,
      drag: null,
    };

    const root = el("div", "pe-root ds-root");
    container.innerHTML = "";
    container.appendChild(root);

    const box = el("div", "ds-box");
    const bar = el("div", "ds-bar");

    const seg = el("div", "ds-seg");
    const btnLight = el("button", null, "Light arriving");
    const btnPhase = el("button", null, "Mask phase");
    seg.appendChild(btnLight); seg.appendChild(btnPhase);
    bar.appendChild(seg);

    const btnBeam = el("button", "ds-btn", "Beam between masks");
    const btnSweep = el("button", "ds-btn", "▶ Sweep");
    bar.appendChild(btnBeam); bar.appendChild(btnSweep);

    // Manual mode only: the stage stops chasing the pen and re-renders here.
    const btnRefresh = el("button", "ds-btn ds-refresh", "⟳ Refresh");
    if (MODE === "manual") bar.appendChild(btnRefresh);

    bar.appendChild(el("div", "ds-spacer"));
    const btnReset = el("button", "ds-btn", "Reset view");
    bar.appendChild(btnReset);
    box.appendChild(bar);

    const canvas = el("canvas", "ds-canvas");
    canvas.setAttribute("role", "img");
    box.appendChild(canvas);

    const foot = el("div", "ds-foot");
    const footL = el("div", null, "");
    const footR = el("div", null, "");
    foot.appendChild(footL); foot.appendChild(footR);
    box.appendChild(foot);
    root.appendChild(box);

    root.appendChild(el("p", "ds-note",
      "Drag to orbit. Every panel carries the light actually computed on it, and the haze "
      + "between them is the light at those in-between depths, computed the same way "
      + "rather than shaded in. No rays are drawn on purpose: light here behaves as a "
      + "wave, and straight arrows would misrepresent it."));

    // ------------------------------------------------------------- bitmap cache
    // Compute and render are separate: orbiting and sweeping only re-run the
    // affine draws. Bitmaps are rebuilt only when the digit or the view changes.
    function buildPanelBitmaps(res) {
      const panels = [];
      panels.push({
        z: 0, kind: "input", label: "input",
        bmp: renderBitmap(res.canvas, N, LUT_I, 0.7, false),
      });
      for (const L of MASK_IDX) {
        // The label carries the *true* index, and says what it is out of as soon
        // as any mask has been left out -- a visitor must never read six panels
        // as six masks, or panel 3 as the third plate in the machine.
        panels.push({
          z: (L + 1) * W.separation, kind: "mask", index: L,
          label: SAMPLED ? `mask ${L + 1} of ${W.n_layers}` : "mask " + (L + 1),
          bmp: renderBitmap(res.planes[L], N, LUT_I, GAMMA, false),
        });
      }
      panels.push({
        z: TOTAL_Z, kind: "detector", label: "detectors",
        bmp: renderBitmap(res.intensity, N, LUT_I, GAMMA, false),
      });
      return panels;
    }

    /** Phase plates for the drawn masks only, keyed by true mask index. */
    function buildPhaseBitmaps() {
      const out = new Map();
      for (const L of MASK_IDX) out.set(L, renderBitmap(MODEL.maskPhase(L), N, LUT_P, 1, true));
      return out;
    }

    /**
     * The depth-ordered intensity stack to draw, as [{z, I}] from the entrance
     * plane down.
     *
     * At subSteps === 1 the slices *are* the canonical planes: sliceForward would
     * walk exactly the hops classify() already walked, and
     * tests/test_d2nn_crosscheck.py pins those two to better than 1e-11 of peak.
     * So the beam is read straight out of the result rather than propagating the
     * whole stack a second time. On the 56-mask stage -- which is precisely the
     * one that lands on subSteps = 1, because its hops are 2.19 px of reach --
     * that is 57 propagations per digit that no longer happen.
     */
    function sliceStack(res, S) {
      const f = MODEL.encodeInput(res.canvas);
      if (S !== 1) return MODEL.sliceForward(f[0], f[1], S);

      const entrance = new Float64Array(N * N);
      for (let i = 0; i < entrance.length; i++) {
        entrance[i] = f[0][i] * f[0][i] + f[1][i] * f[1][i];
      }
      const out = [{ z: 0, I: entrance }];
      for (let L = 0; L < res.planes.length; L++) {
        out.push({ z: (L + 1) * W.separation, I: res.planes[L] });
      }
      out.push({ z: (res.planes.length + 1) * W.separation, I: res.intensity });
      return out;
    }

    function buildSliceBitmaps(res, panels) {
      const S = Math.max(1, state.subSteps);
      const dz = W.separation / S;
      const slices = sliceStack(res, S);

      // Skip only the depths a *drawn panel* already occupies -- those are drawn
      // opaque. The test used to be "does a mask sit here", which was the same
      // thing while every mask got a panel. Once the stack is sampled they part
      // company, and at subSteps=1 they part company completely: every slice
      // lands on some mask, so that test discarded the entire beam and left the
      // fifty undrawn plates as empty space. The light between the six panels is
      // the most honest thing in the figure -- it is where the other fifty are.
      const drawn = new Set(panels.map((p) => Math.round(p.z / dz)));
      const out = [];
      for (const s of slices) {
        if (drawn.has(Math.round(s.z / dz))) continue;
        out.push({ z: s.z, bmp: renderBitmap(s.I, N, LUT_I, GAMMA, false) });
      }
      return out;
    }

    /** Recompute slices off the critical path, then repaint.
     *
     * Slices cost (n_layers+1)*subSteps propagations -- ~130 ms -- so they are
     * debounced: while someone is drawing, setResult fires on every pointer move
     * and only the cheap panel bitmaps are rebuilt. The beam catches up once the
     * pen stops.
     */
    function requestSlices() {
      if (state.sliceTimer) clearTimeout(state.sliceTimer);
      if (!state.beam || !state.res || (state.bitmaps && state.bitmaps.slices)) return;
      state.sliceTimer = setTimeout(() => {
        state.sliceTimer = 0;
        if (!state.res || !state.beam) return;
        state.bitmaps.slices = buildSliceBitmaps(state.res, state.bitmaps.panels);
        draw();
      }, 180);
    }

    /** Render whatever is currently held, rebuilding the bitmaps for it. */
    function render(res, meta) {
      state.res = res;
      state.meta = meta || null;
      state.pending = null;
      state.bitmaps = {
        panels: buildPanelBitmaps(res),
        phases: state.bitmaps ? state.bitmaps.phases : buildPhaseBitmaps(),
        slices: null,                        // invalidated: new digit, new beam
      };
      requestSlices();
      draw();
      updateRefresh();
    }

    /**
     * A new result arrived from the classifier.
     *
     * In live mode this draws it, as it always has. In manual mode it is *held*:
     * a deep stack costs too much to rebuild on every pointer move, and a picture
     * nobody can read mid-stroke is not worth a stuttering page. The first result
     * still draws immediately -- there is nothing to be stale against yet, and an
     * empty stage would just look broken.
     */
    function setResult(res, meta) {
      if (MODE === "live" || !state.res) {
        render(res, meta);
        return;
      }
      state.pending = { res: res, meta: meta || null };
      updateRefresh();
      draw();
    }

    /**
     * Take a raw digit instead of a finished result, and classify it here.
     *
     * This is the form a deep stage wants. `setResult` obliges the caller to have
     * already run the forward pass, which for a 56-mask network is ~130 ms thrown
     * away on every pointer move that manual mode then declines to draw. Holding
     * the *digit* defers the physics as well as the drawing, so drawing beside a
     * deep stage costs nothing until Refresh is pressed.
     */
    function setDigit(img28, meta) {
      if (MODE === "live" || !state.res) {
        render(MODEL.classify(img28), meta);
        return;
      }
      state.pending = { digit: img28, meta: meta || null };
      updateRefresh();
      draw();
    }

    /** Draw whatever is held, running the forward pass now if it was deferred. */
    function commitPending() {
      const p = state.pending;
      if (!p) return;
      render(p.res || MODEL.classify(p.digit), p.meta);
    }

    /** Reflect whether what is drawn is still the current digit. */
    function updateRefresh() {
      if (MODE !== "manual") return;
      const stale = !!state.pending;
      btnRefresh.setAttribute("aria-pressed", String(stale));
      btnRefresh.disabled = !stale;
      btnRefresh.textContent = stale ? "⟳ Refresh" : "⟳ Up to date";
    }

    // ----------------------------------------------------------------- geometry
    /** Fit the whole stack, at the current orbit, into a w x h canvas. */
    function layout(w, h) {
      const b = basis(state.theta, state.phi);
      const pts = [];
      for (const zz of [0, DEPTH_SPAN]) {
        for (const ax of [-1, 1]) {
          for (const ay of [-1, 1]) {
            pts.push([ax * b.eX[0] + ay * b.eY[0] + zz * b.eZ[0],
                      ax * b.eX[1] + ay * b.eY[1] + zz * b.eZ[1]]);
          }
        }
      }
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const p of pts) {
        if (p[0] < x0) x0 = p[0]; if (p[0] > x1) x1 = p[0];
        if (p[1] < y0) y0 = p[1]; if (p[1] > y1) y1 = p[1];
      }
      const m = 30;                                       // room for plane labels
      const s = Math.min((w - 2 * m) / Math.max(x1 - x0, 1e-6),
                         (h - 2 * m) / Math.max(y1 - y0, 1e-6));
      return {
        b, s,
        ox: w / 2 - s * (x0 + x1) / 2,
        oy: h / 2 - s * (y0 + y1) / 2,
      };
    }

    /** Plane coords (u, v in [0,1]) at depth z (metres) -> screen [x, y]. */
    function project(L, u, v, z) {
      const a = (u - 0.5) * 2, c = (v - 0.5) * 2;         // to [-1, 1]
      const d = (z / TOTAL_Z) * DEPTH_SPAN;
      return [
        L.ox + L.s * (a * L.b.eX[0] + c * L.b.eY[0] + d * L.b.eZ[0]),
        L.oy + L.s * (a * L.b.eX[1] + c * L.b.eY[1] + d * L.b.eZ[1]),
      ];
    }

    /** Draw an N x N bitmap as the plane at depth z, via one affine transform.
     *
     * Uses ctx.transform (which *multiplies* onto the current matrix), never
     * ctx.setTransform (which replaces it). draw() puts a devicePixelRatio scale
     * on the context, and every other mark here -- plates, outlines, detector
     * boxes, labels -- is a plain path drawn under it. Replacing the matrix would
     * silently drop that scale for the light only, shrinking it by 1/dpr toward
     * the canvas origin while the rig stayed put: invisible at dpr 1, obvious on
     * any phone or scaled desktop.
     */
    function drawPlane(ctx, L, bmp, z, alpha) {
      const o = project(L, 0, 0, z);
      const px = project(L, 1, 0, z);
      const py = project(L, 0, 1, z);
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.transform((px[0] - o[0]) / N, (px[1] - o[1]) / N,
                    (py[0] - o[0]) / N, (py[1] - o[1]) / N, o[0], o[1]);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(bmp, 0, 0);
      ctx.restore();
    }

    /** Fill the plane's quad with a flat colour -- the "glass" a plate is made of.
     *
     * Optical intensity is mostly darkness, so an opaque intensity image renders as
     * a black card and the stack becomes seven black cards. Drawing a faint plate
     * and compositing alpha-keyed light onto it keeps the geometry visible *and*
     * lets you see through to the planes behind.
     */
    function fillPlane(ctx, L, z, color, alpha) {
      const c = [project(L, 0, 0, z), project(L, 1, 0, z),
                 project(L, 1, 1, z), project(L, 0, 1, z)];
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(c[0][0], c[0][1]);
      for (let i = 1; i < 4; i++) ctx.lineTo(c[i][0], c[i][1]);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    function outline(ctx, L, z, color, width, alpha) {
      const c = [project(L, 0, 0, z), project(L, 1, 0, z),
                 project(L, 1, 1, z), project(L, 0, 1, z)];
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(c[0][0], c[0][1]);
      for (let i = 1; i < 4; i++) ctx.lineTo(c[i][0], c[i][1]);
      ctx.closePath();
      ctx.stroke();
      ctx.restore();
    }

    // ------------------------------------------------------------------ drawing
    function draw() {
      const w = canvas.clientWidth || 720;
      const h = Math.max(300, Math.min(430, Math.round(w * 0.52)));
      const dpr = canvasScale();
      canvas.style.height = h + "px";
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const p = palette(root);
      if (!state.res) {
        ctx.fillStyle = p.muted;
        ctx.font = "13px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Pick or draw a digit below.", w / 2, h / 2);
        return;
      }

      const L = layout(w, h);
      const bm = state.bitmaps;
      const frontZ = state.front * TOTAL_Z;
      const reached = (z) => z <= frontZ + 1e-12;

      // Faint edges joining consecutive panels: the stack's solid geometry.
      ctx.save();
      ctx.strokeStyle = p.border;
      ctx.globalAlpha = 0.85;
      ctx.lineWidth = 1;
      const corners = [[0, 0], [1, 0], [1, 1], [0, 1]];
      for (let k = 0; k + 1 < bm.panels.length; k++) {
        const za = bm.panels[k].z, zb = bm.panels[k + 1].z;
        for (const [u, v] of corners) {
          const a = project(L, u, v, za), b2 = project(L, u, v, zb);
          ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b2[0], b2[1]); ctx.stroke();
        }
      }
      ctx.restore();

      // Ascending z is the exact painter's order: the panels are parallel and
      // never intersect. Substrate first, then light, because the masks are
      // *transmissive* -- light from plane k really does carry on through plate
      // k+1, so a plate must not darken the field behind it.
      const phaseOf = (pan) => (state.view === "phase" && pan.kind === "mask")
        ? bm.phases.get(pan.index) : null;

      for (const pan of bm.panels) {
        const alpha = reached(pan.z) ? 1 : 0.2;
        const ph = phaseOf(pan);
        // Phase is a property of the fabricated surface, so it *is* the plate.
        if (ph) drawPlane(ctx, L, ph, pan.z, alpha * 0.85);
        else fillPlane(ctx, L, pan.z, PLATE_INK, alpha * PLATE_ALPHA);
      }

      // Light, additively: intensities add, so overlapping planes glow instead of
      // occluding and the dark parts of a field contribute nothing -- which is
      // what makes a stack of mostly-black planes readable at all.
      const lights = [];
      for (const pan of bm.panels) {
        if (!phaseOf(pan)) lights.push({ z: pan.z, bmp: pan.bmp, a: 1 });
      }
      if (state.beam && bm.slices) {
        for (const s of bm.slices) lights.push({ z: s.z, bmp: s.bmp, a: 0.5 });
      }
      lights.sort((a, b) => a.z - b.z);
      ctx.save();
      ctx.globalCompositeOperation = "lighter";
      for (const li of lights) {
        drawPlane(ctx, L, li.bmp, li.z, reached(li.z) ? li.a : li.a * 0.12);
      }
      ctx.restore();

      for (const pan of bm.panels) {
        const lit = reached(pan.z);
        const edge = pan.kind === "detector" ? p.accent : p.border;
        outline(ctx, L, pan.z, edge, pan.kind === "detector" ? 1.6 : 1, lit ? 0.9 : 0.3);
        if (pan.kind === "detector") drawRegions(ctx, L, pan.z, lit ? 1 : 0.3, p);
      }

      // Labels last: a nearer plate would otherwise paint over the label of the
      // plate behind it, and a legend you cannot read is worse than a wrong depth.
      const placed = [];
      for (const pan of bm.panels) drawLabel(ctx, L, pan, reached(pan.z) ? 1 : 0.4, p, placed);

      // The wavefront itself, while sweeping.
      if (state.front < 1) outline(ctx, L, frontZ, p.warn, 2, 0.95);

      // Held-back digit: say so on the picture, not only on the button. Someone
      // who has drawn something and sees an unchanged stack needs to know it is
      // waiting rather than broken.
      if (state.pending) {
        const msg = "Showing the previous digit. Press Refresh to update.";
        ctx.save();
        ctx.font = "600 11.5px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.textAlign = "right";
        ctx.textBaseline = "top";
        ctx.strokeStyle = p.panel;
        ctx.lineWidth = 3;
        ctx.lineJoin = "round";
        ctx.strokeText(msg, w - 10, 8);
        ctx.fillStyle = p.warn;
        ctx.fillText(msg, w - 10, 8);
        ctx.restore();
      }

      updateFoot();
    }

    function drawRegions(ctx, L, z, alpha, p) {
      const regions = W.regions;
      ctx.save();
      ctx.globalAlpha = alpha;
      for (let c = 0; c < regions.length; c++) {
        const r = regions[c];                             // [y0, y1, x0, x1]
        const win = c === state.res.pred;
        const pts = [
          project(L, r[2] / N, r[0] / N, z), project(L, r[3] / N, r[0] / N, z),
          project(L, r[3] / N, r[1] / N, z), project(L, r[2] / N, r[1] / N, z),
        ];
        ctx.strokeStyle = win ? "#ffffff" : "rgba(255,255,255,0.45)";
        ctx.lineWidth = win ? 2 : 1;
        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < 4; i++) ctx.lineTo(pts[i][0], pts[i][1]);
        ctx.closePath();
        ctx.stroke();
        if (win) {
          ctx.fillStyle = "#ffffff";
          ctx.font = "600 13px ui-monospace, SFMono-Regular, Menlo, monospace";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(String(c), (pts[0][0] + pts[2][0]) / 2, (pts[0][1] + pts[2][1]) / 2);
        }
      }
      ctx.restore();
    }

    function drawLabel(ctx, L, pan, alpha, p, placed) {
      // Centred just above each plane's own top edge, so a label is unambiguously
      // attached to its plate at any orbit angle. The planes' cascade staggers the
      // labels for free -- while the planes are far apart.
      //
      // On a deep stack they are not. At 0.53 mm gaps the first mask sits 1.7% of
      // the way along the axis, so "mask 1 of 56" lands on top of "input", and the
      // last one lands on "detectors". So each label is lifted until it clears the
      // ones already drawn: the cascade direction is preserved, nothing is
      // dropped, and no label is ever moved off its own plate.
      const top = project(L, 0.5, 0, pan.z);
      const bottom = project(L, 0.5, 1, pan.z);
      const uy = [bottom[0] - top[0], bottom[1] - top[1]];
      let x = top[0] - uy[0] * 0.035, y = top[1] - uy[1] * 0.035;

      ctx.save();
      ctx.font = (pan.kind === "detector" ? "600 " : "")
        + "11px ui-monospace, SFMono-Regular, Menlo, monospace";
      const halfW = ctx.measureText(pan.label).width / 2 + 3;
      const H = 13;
      for (let guard = 0; guard < 40; guard++) {
        const hit = placed.find((q) => Math.abs(q.x - x) < q.halfW + halfW
                                    && Math.abs(q.y - y) < H);
        if (!hit) break;
        y = hit.y - H;
      }
      placed.push({ x, y, halfW });

      ctx.globalAlpha = alpha;
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      // A label can land on a dark plate or on the light card behind it depending
      // on the orbit, so it carries a halo in the panel colour and reads on both.
      ctx.strokeStyle = p.panel;
      ctx.lineWidth = 3;
      ctx.lineJoin = "round";
      ctx.strokeText(pan.label, x, y);
      ctx.fillStyle = pan.kind === "detector" ? p.accent : p.fg;
      ctx.fillText(pan.label, x, y);
      ctx.restore();
    }

    function updateFoot() {
      const showing = state.view === "phase"
        ? "masks show their trained phase (cyclic colour); the other planes show light"
        : "each plane shows the brightness arriving on it, on a square-root scale";
      // Sampling is stated in the running text as well as on every label: the
      // labels are only readable at some orbit angles, and this claim has to
      // survive the figure being screenshotted.
      footL.innerHTML = SAMPLED
        ? `${showing}, <b>${MASK_IDX.length} of ${W.n_layers} masks drawn</b>, `
          + "spread through the stack"
        : showing;
      const pred = state.res ? state.res.pred : null;
      footR.innerHTML =
        `<b>${(TOTAL_Z * 1e3).toFixed(0)} mm</b> of optics across a `
        + `<b>${(APERTURE * 1e3).toFixed(2)} mm</b> across, so the depth is squashed `
        + `<b>&times;${COMPRESSION.toFixed(1)}</b>, not to scale`
        + (pred == null ? "" : ` &middot; reads <b>${pred}</b>`);
    }

    // ------------------------------------------------------------------- events
    let raf = false;
    function requestDraw() {
      if (raf) return;
      raf = true;
      requestAnimationFrame(() => { raf = false; draw(); });
    }

    function setView(v) {
      state.view = v;
      btnLight.setAttribute("aria-pressed", String(v === "light"));
      btnPhase.setAttribute("aria-pressed", String(v === "phase"));
      requestDraw();
    }
    btnLight.addEventListener("click", () => setView("light"));
    btnPhase.addEventListener("click", () => setView("phase"));

    btnBeam.addEventListener("click", () => {
      state.beam = !state.beam;
      btnBeam.setAttribute("aria-pressed", String(state.beam));
      if (state.beam) requestSlices();
      requestDraw();
    });

    let sweepRaf = 0;
    function stopSweep() {
      if (sweepRaf) cancelAnimationFrame(sweepRaf);
      sweepRaf = 0;
      state.sweep = null;
      state.front = 1;
      btnSweep.textContent = "▶ Sweep";
      btnSweep.setAttribute("aria-pressed", "false");
      requestDraw();
    }
    function stepSweep(now) {
      if (!state.sweep) return;
      const t = (now - state.sweep.start) / SWEEP_MS;
      if (t >= 1) { stopSweep(); return; }
      state.front = t;
      draw();
      sweepRaf = requestAnimationFrame(stepSweep);
    }
    btnSweep.addEventListener("click", () => {
      if (state.sweep) { stopSweep(); return; }
      state.sweep = { start: performance.now() };
      state.front = 0;
      btnSweep.textContent = "■ Stop";
      btnSweep.setAttribute("aria-pressed", "true");
      sweepRaf = requestAnimationFrame(stepSweep);
    });

    btnRefresh.addEventListener("click", commitPending);

    btnReset.addEventListener("click", () => {
      state.theta = 34 * Math.PI / 180;
      state.phi = 19 * Math.PI / 180;
      requestDraw();
    });

    // Orbit. Recomputes nothing -- only the affine draws re-run.
    const LIM = { thetaMin: 6 * Math.PI / 180, thetaMax: 80 * Math.PI / 180,
                  phiMin: -28 * Math.PI / 180, phiMax: 58 * Math.PI / 180 };
    canvas.addEventListener("pointerdown", (ev) => {
      state.drag = { x: ev.clientX, y: ev.clientY, theta: state.theta, phi: state.phi };
      canvas.setPointerCapture(ev.pointerId);
      canvas.classList.add("drag");
      ev.preventDefault();
    });
    canvas.addEventListener("pointermove", (ev) => {
      if (!state.drag) return;
      const dx = ev.clientX - state.drag.x, dy = ev.clientY - state.drag.y;
      state.theta = Math.min(LIM.thetaMax, Math.max(LIM.thetaMin,
        state.drag.theta + dx * 0.0034));
      state.phi = Math.min(LIM.phiMax, Math.max(LIM.phiMin,
        state.drag.phi + dy * 0.0030));
      requestDraw();
      ev.preventDefault();
    });
    const endDrag = () => { state.drag = null; canvas.classList.remove("drag"); };
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);

    if (typeof MutationObserver !== "undefined") {
      new MutationObserver(requestDraw).observe(document.documentElement,
        { attributes: true, attributeFilter: ["data-theme"] });
    }
    if (typeof ResizeObserver !== "undefined") new ResizeObserver(requestDraw).observe(root);
    else window.addEventListener("resize", requestDraw);

    setView("light");
    btnBeam.setAttribute("aria-pressed", String(state.beam));
    updateRefresh();
    draw();

    const api = { setResult, setDigit, redraw: requestDraw, state,
                  // Geometry decisions, exposed so tests can assert them without a
                  // DOM and a page can print "6 of 56" without recomputing it.
                  model: MODEL, mode: MODE, maskIndices: MASK_IDX, sampled: SAMPLED,
                  refresh: commitPending,
                  setFront: (t) => { state.front = t; draw(); } };
    container.__photonnStage = api;   // handle for page scripts and browser tests
    return api;
  }

  // sampleMaskIndices/defaultSubSteps/reachPerHopPx are exported so the rules that
  // make a deep stack legible can be tested without a DOM -- there is no jsdom
  // here, and "the labels name the real mask" is the claim that matters most.
  const API = { mount, basis, sampleMaskIndices, defaultSubSteps, reachPerHopPx,
                MAX_MASK_PANELS, LIVE_MAX_LAYERS };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  if (typeof window !== "undefined") window.PhotonnD2NNStage = API;
})();
