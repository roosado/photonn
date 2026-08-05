/*
 * d2nn_stage.js -- the diffractive stack in three dimensions.
 *
 * The filmstrip beside this widget is accurate but flat: five squares in a row,
 * asking you to imagine they are the same light further along. This draws the
 * machine instead -- the entrance plane, the five phase masks and the detector
 * plane as parallel panels receding along the optical axis, each carrying the
 * light actually computed on it, so the geometry reads at a glance.
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
 * The stack is 18 mm long across a 1.02 mm aperture -- about 18:1 -- so drawn to
 * scale it is an unreadable needle. The depth axis is compressed, and the figure
 * says so on its face.
 *
 * Usage:  var stage = window.PhotonnD2NNStage.mount(container, opts);
 *         stage.setResult(res, {trueLabel: 7});   // res from PhotonnD2NN_Net
 * opts (all optional): { theta, phi, beam, subSteps }  -- degrees, degrees, bool, int
 *
 * Depends on window.PhotonnD2NN_Net (d2nn.js). Pure rendering: no physics of its
 * own beyond calling sliceForward, no network, no libraries.
 */
(function () {
  "use strict";

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

  const STYLE_ID = "ds-style";
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

    const W = NET.weights;
    const N = NET.N;
    const HOPS = W.n_layers + 1;
    const TOTAL_Z = HOPS * W.separation;              // 18 mm
    const APERTURE = N * W.dx;                        // 1.024 mm
    // The drawn stack is DEPTH_SPAN half-apertures long; anything else would be
    // an 18:1 needle. Report the factor rather than let the reader assume scale.
    const COMPRESSION = (TOTAL_Z / APERTURE) / (DEPTH_SPAN / 2);

    const state = {
      theta: (opts.theta == null ? 34 : opts.theta) * Math.PI / 180,
      phi: (opts.phi == null ? 19 : opts.phi) * Math.PI / 180,
      view: "light",                 // "light" | "phase"
      beam: opts.beam !== false,
      subSteps: opts.subSteps || 4,
      sweep: null,                   // null, or {start} while playing
      front: 1,                      // wavefront position, 0..1 of the axis
      res: null,
      meta: null,
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
      "Drag to orbit. Every panel carries the field computed on it; the haze between "
      + "them is the field at intermediate depths, which is exact physics here, not a "
      + "gradient. No rays are drawn &mdash; scalar diffraction is not ray optics."));

    // ------------------------------------------------------------- bitmap cache
    // Compute and render are separate: orbiting and sweeping only re-run the
    // affine draws. Bitmaps are rebuilt only when the digit or the view changes.
    function buildPanelBitmaps(res) {
      const panels = [];
      panels.push({
        z: 0, kind: "input", label: "input",
        bmp: renderBitmap(res.canvas, N, LUT_I, 0.7, false),
      });
      for (let L = 0; L < W.n_layers; L++) {
        panels.push({
          z: (L + 1) * W.separation, kind: "mask", index: L, label: "mask " + (L + 1),
          bmp: renderBitmap(res.planes[L], N, LUT_I, GAMMA, false),
        });
      }
      panels.push({
        z: TOTAL_Z, kind: "detector", label: "detectors",
        bmp: renderBitmap(res.intensity, N, LUT_I, GAMMA, false),
      });
      return panels;
    }

    function buildPhaseBitmaps() {
      const out = [];
      for (let L = 0; L < W.n_layers; L++) {
        out.push(renderBitmap(NET.maskPhase(L), N, LUT_P, 1, true));
      }
      return out;
    }

    function buildSliceBitmaps(res) {
      const f = NET.encodeInput(res.canvas);
      const slices = NET.sliceForward(f[0], f[1], state.subSteps);
      const out = [];
      for (const s of slices) {
        // Skip the depths a panel already occupies -- those are drawn opaque.
        const onPanel = Math.abs(s.z / W.separation - Math.round(s.z / W.separation)) < 1e-9;
        if (onPanel) continue;
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
        state.bitmaps.slices = buildSliceBitmaps(state.res);
        draw();
      }, 180);
    }

    function setResult(res, meta) {
      state.res = res;
      state.meta = meta || null;
      state.bitmaps = {
        panels: buildPanelBitmaps(res),
        phases: state.bitmaps ? state.bitmaps.phases : buildPhaseBitmaps(),
        slices: null,                        // invalidated: new digit, new beam
      };
      requestSlices();
      draw();
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
      const dpr = window.devicePixelRatio || 1;
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
        ? bm.phases[pan.index] : null;

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
      for (const pan of bm.panels) drawLabel(ctx, L, pan, reached(pan.z) ? 1 : 0.4, p);

      // The wavefront itself, while sweeping.
      if (state.front < 1) outline(ctx, L, frontZ, p.warn, 2, 0.95);

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

    function drawLabel(ctx, L, pan, alpha, p) {
      // Centred just above each plane's own top edge, so a label is unambiguously
      // attached to its plate at any orbit angle. The planes' cascade staggers the
      // labels for free.
      const mid = project(L, 0.5, 0, pan.z);
      const top = project(L, 0.5, 0, pan.z);
      const bottom = project(L, 0.5, 1, pan.z);
      const uy = [bottom[0] - top[0], bottom[1] - top[1]];
      const x = mid[0] - uy[0] * 0.035, y = mid[1] - uy[1] * 0.035;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.font = (pan.kind === "detector" ? "600 " : "")
        + "11px ui-monospace, SFMono-Regular, Menlo, monospace";
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
        : "each plane shows the intensity arriving on it, square-root stretched";
      footL.innerHTML = showing;
      const pred = state.res ? state.res.pred : null;
      footR.innerHTML =
        `<b>${(TOTAL_Z * 1e3).toFixed(0)} mm</b> of optics across a `
        + `<b>${(APERTURE * 1e3).toFixed(2)} mm</b> aperture &mdash; depth compressed `
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
    draw();

    const api = { setResult, redraw: requestDraw, state,
                  setFront: (t) => { state.front = t; draw(); } };
    container.__photonnStage = api;   // handle for page scripts and browser tests
    return api;
  }

  const API = { mount, basis };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  if (typeof window !== "undefined") window.PhotonnD2NNStage = API;
})();
