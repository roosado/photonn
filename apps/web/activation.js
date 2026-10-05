/*
 * activation.js -- the electro-optic activation, and the chip built around it.
 *
 * The activation page argues one device: a tap sends a tenth of a mode's light to a
 * photodiode, and the amplified photocurrent sets the phase inside an
 * interferometer that the rest of the *same* light crosses (Williamson et al.,
 * IEEE JSTQE 26(1):7700412, 2020). Two modes of one widget, because both draw the
 * same curve and two files would mean two copies of it:
 *
 *   mode "device"  -- the transfer function T(P) = (1-a) cos^2((g P + phi_b)/2)
 *                     on a log power axis, with sliders for the bias phi_b and the
 *                     gain g. The tap fraction a is shown, not slid: it sets a
 *                     ceiling, and in the real device g is proportional to a, so an
 *                     independent a slider would teach a coupling the hardware
 *                     does not have.
 *   mode "network" -- the trained two-layer chip (deep_mesh_weights.js), one test
 *                     digit at a time: where its sixteen modes land on the curve,
 *                     what the ten detectors read, and how much light gets there.
 *                     No sliders on a, g or phi_b -- perturbing a trained device and
 *                     showing the answer change is the as-built model's job, in
 *                     MATLAB. The input-power slider is allowed because the write-up
 *                     publishes the noiseless power sweep it walks along.
 *
 * MECHANISM ONLY, as errors.js: no accuracy is computed here, one digit at a time.
 * NO requestAnimationFrame: the driven browser tab never fires one, so nothing
 * animated could be verified. Constants are never literals in this file: they
 * come from the mount options or the bundle, and mounting with neither throws.
 *
 * tests/test_activation_widget.py holds the drawn curve to the closed form, the
 * complex field() to an MZI built from photonn.mzi, and both canvases to the
 * aspect they are shown at. tests/test_deep_mesh_web.py holds the network mode to
 * the float64 reference (photonn.mzi.deep_mesh_forward) on every gallery digit.
 *
 * Usage:  window.PhotonnActivation.mount(el, {mode: "device"})
 *         window.PhotonnActivation.mount(el, {mode: "network"})
 */
(function () {
  "use strict";

  const P = (typeof window !== "undefined" && window.PhotonnPlot)
    ? window.PhotonnPlot
    : (typeof require !== "undefined" ? require("./plot.js") : null);

  const STYLE_ID = "ea-style";

  const CSS = `
.ea-root{--ea-fg:#1b1f24;--ea-muted:#5a6472;--ea-panel:#f4f6f9;--ea-border:#d7dde5;
  --ea-accent:#3b6ea5;--ea-ok:#3f8f4e;--ea-warn:#c14a3d;
  color:var(--ea-fg);font:13px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  background:var(--ea-panel);border:1px solid var(--ea-border);border-radius:10px;padding:13px 15px;}
@media (prefers-color-scheme:dark){.ea-root{--ea-fg:#e6eaf0;--ea-muted:#9aa6b5;
  --ea-panel:#1c2128;--ea-border:#30363d;--ea-accent:#6ea8e0;--ea-ok:#5cc06e;--ea-warn:#e0705f;}}
:root[data-theme="dark"] .ea-root{--ea-fg:#e6eaf0;--ea-muted:#9aa6b5;--ea-panel:#1c2128;
  --ea-border:#30363d;--ea-accent:#6ea8e0;--ea-ok:#5cc06e;--ea-warn:#e0705f;}
:root[data-theme="light"] .ea-root{--ea-fg:#1b1f24;--ea-muted:#5a6472;--ea-panel:#f4f6f9;
  --ea-border:#d7dde5;--ea-accent:#3b6ea5;--ea-ok:#3f8f4e;--ea-warn:#c14a3d;}
.ea-row{display:flex;gap:16px;align-items:flex-end;flex-wrap:wrap;margin-bottom:11px;}
.ea-row label{font-size:11px;font-weight:600;color:var(--ea-muted);text-transform:uppercase;
  letter-spacing:.04em;display:block;margin-bottom:3px;}
.ea-row input[type=range]{width:220px;max-width:100%;accent-color:var(--ea-accent);display:block;}
.ea-val{font-variant-numeric:tabular-nums;font-weight:600;font-size:13px;white-space:nowrap;}
.ea-fixed{font-size:12px;color:var(--ea-muted);}
.ea-fixed b{color:var(--ea-fg);font-variant-numeric:tabular-nums;}
/* One wide plot per canvas, capped so it does not letterbox on a desktop. The
   canvases measure these panes before drawing (P.fit), one unit per CSS pixel. */
.ea-plot{max-width:640px;margin:0 auto;}
.ea-plot canvas{width:100%;height:auto;display:block;border-radius:6px;background:#0b0d10;}
.ea-cap{font-size:11px;color:var(--ea-muted);margin:5px 0 0;text-align:center;line-height:1.35;}
.ea-note{font-size:12px;color:var(--ea-muted);margin:11px 0 0;line-height:1.5;}
.ea-note b{color:var(--ea-fg);}
.ea-gallery{display:flex;flex-wrap:wrap;gap:4px;margin:0 0 11px;}
.ea-gallery button{padding:0;border:2px solid transparent;border-radius:4px;background:#000;
  cursor:pointer;line-height:0;}
.ea-gallery button[aria-pressed="true"]{border-color:var(--ea-accent);}
.ea-gallery button:focus-visible{outline:2px solid var(--ea-accent);outline-offset:1px;}
.ea-gallery canvas{width:28px;height:28px;display:block;image-rendering:pixelated;}
.ea-flag{font-weight:600;}
.ea-flag.ok{color:var(--ea-ok);}
.ea-flag.warn{color:var(--ea-warn);}
.ea-plot + .ea-plot{margin-top:10px;}
@media (max-width:560px){
  .ea-row{gap:10px;margin-bottom:9px;}
  .ea-row input[type=range]{width:min(220px,58vw);}
  .ea-cap{font-size:10px;line-height:1.3;}
}
`;

  const el = P.el;

  // ---------------------------------------------------------------- physics
  /** Transmission |f|^2 / |z|^2 at power p: Williamson et al. Eq. (6), squared. */
  function transfer(p, alpha, g, phiB) {
    const c = Math.cos(0.5 * (g * p + phiB));
    return (1 - alpha) * c * c;
  }

  /**
   * Eq. (6) on one complex amplitude: f = j sqrt(1-a) e^{-jh} cos(h) z,
   * h = (g|z|^2 + phi_b)/2. Since j e^{-jh} = sin h + j cos h, the multiplier is
   * sqrt(1-a) cos h (sin h + j cos h). Returns [re, im].
   */
  function field(re, im, alpha, g, phiB) {
    const h = 0.5 * (g * (re * re + im * im) + phiB);
    const a = Math.sqrt(1 - alpha) * Math.cos(h);
    const kr = a * Math.sin(h), ki = a * Math.cos(h);
    return [kr * re - ki * im, kr * im + ki * re];
  }

  /**
   * Where the curve does what: the first power at which it opens fully, the first
   * at which it reaches half its ceiling, and how it treats weak light.
   *
   * Closed forms, not a numerical search, so the readout and the drawing cannot
   * disagree about the curve they both describe.
   */
  function landmarks(alpha, g, phiB) {
    const TWO_PI = 2 * Math.PI;
    const b = ((phiB % TWO_PI) + TWO_PI) % TWO_PI;
    let open = (TWO_PI - b) % TWO_PI;                 // phase still to write
    if (open < 1e-12) open = TWO_PI;                  // open at P = 0: the next one
    const t0 = Math.cos(b / 2) * Math.cos(b / 2);     // weak-light transmission / ceiling
    // half-ceiling: h = pi/4 + k pi/2, first one past h0 = b/2 (crossing, not touching)
    const h0 = b / 2;
    let k = Math.ceil((h0 - Math.PI / 4) / (Math.PI / 2) + 1e-12);
    if (k < 0) k = 0;
    const hHalf = Math.PI / 4 + k * Math.PI / 2;
    return {
      openP: open / g,
      halfP: (2 * hHalf - b) / g,
      weakT: (1 - alpha) * t0,
      passesWeak: t0 >= 0.5,
      // Weak light passes as P^0 (a linear element) unless the bias sits on the dark
      // fringe, where the leading term is the cubic and transmission goes as P^2.
      slope: t0 > 1e-9 ? 0 : 2,
      // Off the dark fringe by a little: a fixed sliver gets through the faintest
      // light, and the cube takes over once the light writes more phase than the
      // offset. The case a dragged slider almost always lands in.
      nearDark: t0 > 1e-9 && t0 < 1e-3,
    };
  }

  /** The curve the device mode draws, sampled at log-spaced powers. */
  function samples(alpha, g, phiB, pLo, pHi, n) {
    const p = new Float64Array(n), t = new Float64Array(n);
    const a = Math.log10(pLo), b = Math.log10(pHi);
    for (let i = 0; i < n; i++) {
      p[i] = Math.pow(10, a + (b - a) * (n > 1 ? i / (n - 1) : 0));
      t[i] = transfer(p[i], alpha, g, phiB);
    }
    return { p: p, t: t };
  }

  /** The device mode's axis: three decades below the first full opening, half above. */
  function axisRange(g, phiB) {
    const L = landmarks(0, g, phiB);
    return [L.openP / 1000, L.openP * Math.sqrt(10)];
  }

  // -------------------------------------------------------------- formatting
  const PREFIX = [[1, "W"], [1e-3, "mW"], [1e-6, "µW"], [1e-9, "nW"], [1e-12, "pW"],
                  [1e-15, "fW"]];

  /** A power with an SI prefix and two significant figures: 0.0201 -> "20 mW". */
  function fmtPower(w, unit) {
    if (unit !== "W") return (+w.toPrecision(2)).toString() + " " + unit;
    for (let i = 0; i < PREFIX.length; i++) {
      if (w >= PREFIX[i][0] * 0.9995 || i === PREFIX.length - 1) {
        return (+(w / PREFIX[i][0]).toPrecision(2)).toString() + " " + PREFIX[i][1];
      }
    }
    return w + " W";
  }

  function fmtSci(v) {
    if (v === 0) return "0";
    const e = Math.floor(Math.log10(Math.abs(v)));
    if (e >= -2 && e <= 3) return (+v.toPrecision(2)).toString();
    const m = v / Math.pow(10, e);
    return (+m.toPrecision(2)).toString() + "×10<sup>" + e + "</sup>";
  }

  // ----------------------------------------------------------------- drawing
  // The plots stay dark in both themes, as the tolerance page's do; only the card
  // follows the theme.
  const PLOT_BG = "#0b0d10", AXIS = "#39414f", GRID = "#1d232c", MUTED = "#9aa6b5";
  const CURVE = "#f2f5f9", CEIL = "#6ea8e0", DOT = "#e0a25f", BAR = "#6ea8e0",
        BAR_HI = "#e0a25f";

  function frame(ctx, W, H, pad) {
    ctx.fillStyle = PLOT_BG;
    ctx.fillRect(0, 0, W, H);
    return { x0: pad.l, x1: W - pad.r, y0: pad.t, y1: H - pad.b };
  }

  /** Log-x axis with a tick per decade, labelled with SI prefixes. */
  function logAxis(ctx, box, lo, hi, unit, title) {
    const a = Math.log10(lo), b = Math.log10(hi);
    const X = (p) => box.x0 + (Math.log10(p) - a) / (b - a) * (box.x1 - box.x0);
    ctx.strokeStyle = GRID;
    ctx.lineWidth = 1;
    ctx.font = "10px ui-monospace, monospace";
    ctx.textAlign = "center";
    const first = Math.ceil(a), last = Math.floor(b);
    const every = (last - first) > 8 ? 3 : 1;
    for (let e = first; e <= last; e++) {
      const x = X(Math.pow(10, e));
      ctx.beginPath(); ctx.moveTo(x, box.y0); ctx.lineTo(x, box.y1); ctx.stroke();
      if ((e - first) % every === 0) {
        ctx.fillStyle = MUTED;
        ctx.fillText(fmtPower(Math.pow(10, e), unit), x, box.y1 + 12);
      }
    }
    ctx.strokeStyle = AXIS;
    ctx.beginPath(); ctx.moveTo(box.x0, box.y1); ctx.lineTo(box.x1, box.y1); ctx.stroke();
    ctx.fillStyle = MUTED;
    ctx.fillText(title, (box.x0 + box.x1) / 2, box.y1 + 25);
    return X;
  }

  /** The transfer curve, its ceiling, and (network mode) the modes on it. */
  function drawCurve(c, dev, lo, hi, dots, unit, title) {
    const { ctx, W, H } = P.fit(c, (w) => Math.max(170, Math.min(250, w * 0.42)));
    const box = frame(ctx, W, H, { l: 30, r: 12, t: 26, b: 34 });
    const X = logAxis(ctx, box, lo, hi, unit, title);
    const Y = (t) => box.y1 - t * (box.y1 - box.y0);

    ctx.fillStyle = MUTED;
    ctx.font = "10px ui-monospace, monospace";
    ctx.textAlign = "right";
    for (const t of [0, 0.5, 1]) ctx.fillText(t.toFixed(1), box.x0 - 4, Y(t) + 3);

    // The ceiling 1 - alpha: what the tap costs even when the device is fully open.
    ctx.strokeStyle = CEIL;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(box.x0, Y(1 - dev.alpha)); ctx.lineTo(box.x1, Y(1 - dev.alpha));
    ctx.stroke();
    ctx.setLineDash([]);

    const n = Math.max(160, Math.round(box.x1 - box.x0));
    const s = samples(dev.alpha, dev.g, dev.phiB, lo, hi, n);
    ctx.strokeStyle = CURVE;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = X(s.p[i]), y = Y(s.t[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    if (dots) {
      ctx.fillStyle = DOT;
      for (let k = 0; k < dots.length; k++) {
        const p = Math.min(hi, Math.max(lo, dots[k]));
        ctx.beginPath();
        ctx.arc(X(p), Y(transfer(dots[k], dev.alpha, dev.g, dev.phiB)), 3.2, 0, 2 * Math.PI);
        ctx.fill();
      }
    }

    // Legend in the canvas: these inks are chosen against the dark plot.
    ctx.font = "11px ui-monospace, monospace";
    ctx.textAlign = "left";
    const keys = [["transmission", CURVE], ["ceiling 1 − α", CEIL]];
    if (dots) keys.push(["this digit's 16 modes", DOT]);
    let kx = box.x0;
    for (const [label, ink] of keys) {
      ctx.fillStyle = ink;
      ctx.fillRect(kx, 9, 9, 3);
      ctx.fillText(label, kx + 13, 15);
      kx += 13 + ctx.measureText(label).width + 14;
    }
    return s;
  }

  /** Ten detector bars: each class's share of the light, as the readout computes it. */
  function drawBars(c, share, pred, label) {
    const { ctx, W, H } = P.fit(c, (w) => Math.max(110, Math.min(150, w * 0.24)));
    const box = frame(ctx, W, H, { l: 12, r: 12, t: 22, b: 18 });
    const n = share.length;
    const slot = (box.x1 - box.x0) / n;
    let top = 0;
    for (let i = 0; i < n; i++) top = Math.max(top, share[i]);
    ctx.font = "10px ui-monospace, monospace";
    ctx.textAlign = "center";
    for (let i = 0; i < n; i++) {
      const h = top > 0 ? share[i] / top * (box.y1 - box.y0) : 0;
      ctx.fillStyle = i === pred ? BAR_HI : BAR;
      ctx.fillRect(box.x0 + i * slot + slot * 0.18, box.y1 - h, slot * 0.64, h);
      ctx.fillStyle = i === label ? CURVE : MUTED;
      ctx.fillText(String(i), box.x0 + (i + 0.5) * slot, box.y1 + 12);
    }
    ctx.textAlign = "left";
    ctx.fillStyle = MUTED;
    ctx.fillText("share of the light reaching each class's detector", box.x0, 13);
  }

  /**
   * A labelled range input. `step` may be "any": a range input snaps its value to
   * the step grid, so a 0.01-step bias slider turns pi into 3.14 -- 1.6 mrad off the
   * dark setting, which is the whole difference this widget is about.
   */
  function slider(row, label, min, max, step, value, aria) {
    const box = el("div");
    box.appendChild(el("label", null, label));
    const input = document.createElement("input");
    input.type = "range";
    input.min = String(min); input.max = String(max); input.step = String(step);
    input.value = String(value);
    input.setAttribute("aria-label", aria);
    box.appendChild(input);
    const val = el("span", "ea-val");
    box.appendChild(val);
    row.appendChild(box);
    return { input: input, val: val };
  }

  // --------------------------------------------------------- the device mode
  function deviceConstants(opts) {
    if (opts.alpha != null && opts.g_phi != null && opts.phi_b != null) {
      return { alpha: +opts.alpha, g: +opts.g_phi, phiB: +opts.phi_b,
               unit: opts.unit || "W" };
    }
    const B = opts.bundle || (typeof window !== "undefined" && window.PHOTONN_DEEP_MESH);
    if (B && B.operating_point) {
      const op = B.operating_point;
      return { alpha: op.eo_alpha, g: op.eo_g_phi, phiB: op.eo_phi_b, unit: "W" };
    }
    throw new Error("activation.js: device mode needs alpha, g_phi and phi_b, from the "
      + "mount options or the deep_mesh_weights.js bundle; it carries no defaults.");
  }

  function mountDevice(root, opts) {
    const dev0 = deviceConstants(opts);
    const dev = { alpha: dev0.alpha, g: dev0.g, phiB: dev0.phiB };
    const row = el("div", "ea-row");
    root.appendChild(row);
    const sb = slider(row, "Bias phase φ_b", 0, 2 * Math.PI, "any", dev.phiB,
                      "Bias phase in radians");
    // Gain on a log slider, two decades either side of the trained device.
    const sg = slider(row, "Phase gain g_φ", -2, 2, 0.01, 0, "Phase gain, log10 multiple of design");
    const fixed = el("div", "ea-fixed");
    fixed.innerHTML = "tap α <b>" + dev.alpha.toFixed(2) + "</b> (fixed)";
    row.appendChild(fixed);

    const plot = el("div", "ea-plot");
    const canvas = el("canvas");
    plot.appendChild(canvas);
    plot.appendChild(el("p", "ea-cap",
      "How much of the light the device passes, against the power entering it. The dashed "
      + "line is what the tap costs even fully open."));
    root.appendChild(plot);
    const note = el("p", "ea-note");
    root.appendChild(note);

    const unitTitle = dev0.unit === "W" ? "power entering the activation"
                                        : "power entering, in units of " + dev0.unit;
    function update() {
      dev.phiB = +sb.input.value;
      dev.g = dev0.g * Math.pow(10, +sg.input.value);
      const L = landmarks(dev.alpha, dev.g, dev.phiB);
      const [lo, hi] = axisRange(dev.g, dev.phiB);
      drawCurve(canvas, dev, lo, hi, null, dev0.unit, unitTitle);
      sb.val.textContent = (dev.phiB / Math.PI).toFixed(2) + "π rad";
      sg.val.textContent = "×" + (+Math.pow(10, +sg.input.value).toPrecision(2)) + " · "
        + (+dev.g.toPrecision(3)) + " rad/" + dev0.unit;
      note.innerHTML = "Opens fully at <b>" + fmtPower(L.openP, dev0.unit) + "</b>, "
        + "half-open at <b>" + fmtPower(L.halfP, dev0.unit) + "</b>. It <b>"
        + (L.passesWeak ? "passes weak light" : "blocks weak light") + "</b>: "
        + (L.slope === 2
          ? "weak light gets through as the square of its own power (×P²), which makes "
            + "the output the cube of the input field."
          : L.nearDark
            ? "the faintest light gets through only as a fixed sliver, " + fmtSci(L.weakT)
              + " (×P⁰), and once it writes more phase than the bias is off its dark "
              + "setting, as the square of its power (×P²)."
            : "weak light gets through in a fixed fraction, " + L.weakT.toFixed(3)
              + " (×P⁰), so faint light sees a plain attenuator.")
        + " The bias decides <em>whether</em> weak light passes; the gain decides "
        + "<em>where</em> the device switches, and both are set by electronics.";
    }
    sb.input.addEventListener("input", update);
    sg.input.addEventListener("input", update);
    P.onWidthChange(canvas, update);
    update();
    return { landmarks: () => landmarks(dev.alpha, dev.g, dev.phiB), note: note };
  }

  // -------------------------------------------------------- the network mode
  let NET = null;

  /** Decode the bundle and build the two layers' matrices, once. */
  function loadNet(B) {
    if (NET && NET.src === B) return NET;
    const M = (typeof window !== "undefined" && window.PhotonnMesh)
      ? window.PhotonnMesh
      : (typeof require !== "undefined" ? require("./mesh.js") : null);
    const n = B.n, L = B.n_layers, nMzi = B.n_mzi, TWO_PI = 2 * Math.PI;
    const theta = M.decode16(B.theta_b64, TWO_PI);
    const phi = M.decode16(B.phi_b64, TWO_PI);
    const sigma = M.decode16(B.sigma_b64, 1);
    const outPhase = M.decode16(B.out_phase_b64, TWO_PI);
    const sch = M.schedule(n);
    const layers = [];
    for (let l = 0; l < L; l++) {
      const off = l * 2 * nMzi;
      layers.push(M.svdMatrix(n, sch, theta.subarray(off), phi.subarray(off),
                              outPhase.subarray((2 * l) * n, (2 * l + 1) * n),
                              outPhase.subarray((2 * l + 1) * n, (2 * l + 2) * n),
                              sigma.subarray(l * n, (l + 1) * n), 0));
    }
    const bytes = M.b64ToBytes(B.gallery_inputs_b64);
    const inputs = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 2);
    const gallery = M.b64ToBytes(B.gallery_b64);
    NET = { src: B, M: M, n: n, layers: layers, inputs: inputs, gallery: gallery,
            labels: B.gallery_labels, op: B.operating_point,
            decoded: { theta: theta, phi: phi, sigma: sigma, outPhase: outPhase } };
    return NET;
  }

  /**
   * One digit through the chip at input power pIn, in watts, noiseless.
   *
   * Exactly mzi.deep_mesh_forward: the field enters as sqrt(pIn) times the
   * unit-norm input, each layer is U.diag(sigma).V, and the activation acts on the
   * field in sqrt(W) between them. Returns the powers entering the activation, the
   * output intensities, and the readout's per-class share.
   */
  function forward(net, idx, pIn) {
    const n = net.n, op = net.op;
    let xr = new Float64Array(n), xi = new Float64Array(n);
    const s = Math.sqrt(pIn);
    for (let k = 0; k < n; k++) {
      xr[k] = s * net.inputs[2 * (idx * n + k)];
      xi[k] = s * net.inputs[2 * (idx * n + k) + 1];
    }
    const taps = [];
    for (let l = 0; l < net.layers.length; l++) {
      const y = net.M.apply(net.layers[l], n, xr, xi);
      xr = y.re; xi = y.im;
      if (l < net.layers.length - 1) {
        const pw = new Float64Array(n);
        for (let k = 0; k < n; k++) {
          pw[k] = xr[k] * xr[k] + xi[k] * xi[k];
          const f = field(xr[k], xi[k], op.eo_alpha, op.eo_g_phi, op.eo_phi_b);
          xr[k] = f[0]; xi[k] = f[1];
        }
        taps.push(pw);
      }
    }
    const intensity = new Float64Array(n);
    let total = 0;
    for (let k = 0; k < n; k++) {
      intensity[k] = xr[k] * xr[k] + xi[k] * xi[k];
      total += intensity[k];
    }
    const share = new Float64Array(10), logits = new Float64Array(10);
    let pred = 0;
    for (let c = 0; c < 10; c++) {
      share[c] = total > 0 ? intensity[c] / total : 0;
      logits[c] = share[c] * op.readout_gain;
      if (intensity[c] > intensity[pred]) pred = c;
    }
    return { taps: taps, intensity: intensity, total: total, share: share,
             logits: logits, pred: pred };
  }

  function digitCanvas(net, idx) {
    const c = el("canvas");
    c.width = 28; c.height = 28;
    const ctx = c.getContext("2d");
    const img = ctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = net.gallery[idx * 784 + i];
      img.data[4 * i] = v; img.data[4 * i + 1] = v; img.data[4 * i + 2] = v;
      img.data[4 * i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return c;
  }

  function mountNetwork(root, opts) {
    const B = opts.bundle || (typeof window !== "undefined" && window.PHOTONN_DEEP_MESH);
    if (!B) throw new Error("activation.js: network mode needs deep_mesh_weights.js.");
    const net = loadNet(B);
    const op = net.op;
    const dev = { alpha: op.eo_alpha, g: op.eo_g_phi, phiB: op.eo_phi_b };
    // Photon energy at the design wavelength, exact SI constants.
    const ePhoton = 6.62607015e-34 * 299792458 / op.wavelength_m;

    const gal = el("div", "ea-gallery");
    root.appendChild(gal);
    const buttons = [];
    for (let i = 0; i < net.labels.length; i++) {
      const b = el("button");
      b.type = "button";
      b.setAttribute("aria-label", "Test digit labelled " + net.labels[i]);
      b.setAttribute("aria-pressed", "false");
      b.appendChild(digitCanvas(net, i));
      gal.appendChild(b);
      buttons.push(b);
    }

    const row = el("div", "ea-row");
    root.appendChild(row);
    // log10 of input power, 1 nW to 1 W; the design point is the handoff's.
    const sp = slider(row, "Light into the chip", -9, 0, 0.05, Math.log10(op.input_power_w),
                      "Input power, log10 watts");

    const curvePane = el("div", "ea-plot");
    const curve = el("canvas");
    curvePane.appendChild(curve);
    root.appendChild(curvePane);
    const barPane = el("div", "ea-plot");
    const bars = el("canvas");
    barPane.appendChild(bars);
    root.appendChild(barPane);
    const note = el("p", "ea-note");
    root.appendChild(note);

    let pick = opts.gallery != null ? opts.gallery : 0;
    let last = null;
    function update() {
      const pIn = Math.pow(10, +sp.input.value);
      const r = forward(net, pick, pIn);
      last = r;
      buttons.forEach((b, i) => b.setAttribute("aria-pressed", i === pick ? "true" : "false"));
      drawCurve(curve, dev, 1e-12, 1, r.taps[0], "W", "power entering each activation");
      drawBars(bars, r.share, r.pred, net.labels[pick]);
      sp.val.textContent = fmtPower(pIn, "W");
      let past = 0, maxPhase = 0, bankIn = 0, bankOut = 0;
      const tap = r.taps[0];
      for (let k = 0; k < tap.length; k++) {
        const t = transfer(tap[k], dev.alpha, dev.g, dev.phiB);
        if (t >= 0.5 * (1 - dev.alpha)) past += 1;
        maxPhase = Math.max(maxPhase, dev.g * tap[k]);
        bankIn += tap[k];
        bankOut += t * tap[k];
      }
      const photons = r.total * op.integration_time_s / ePhoton;
      const right = r.pred === net.labels[pick];
      note.innerHTML = "This digit is a <b>" + net.labels[pick] + "</b>; the chip says <b>"
        + r.pred + "</b> <span class='ea-flag " + (right ? "ok" : "warn") + "'>"
        + (right ? "(right)" : "(wrong)") + "</span>. The brightest mode writes <b>"
        + fmtSci(maxPhase) + " rad</b> on its modulator, <b>" + past + "</b> of 16 are past "
        + "half-open, and the activations let through <b>" + fmtSci(bankIn > 0 ? bankOut / bankIn : 0)
        + "</b> of the light reaching them. At one input per "
        + Math.round(op.integration_time_s * 1e12) + " ps that leaves <b>" + fmtSci(photons)
        + " photons</b> at the detectors.";
    }
    buttons.forEach((b, i) => b.addEventListener("click", () => { pick = i; update(); }));
    sp.input.addEventListener("input", update);
    P.onWidthChange(curve, update);
    update();
    return { forward: (idx, pIn) => forward(net, idx, pIn), last: () => last, note: note };
  }

  // ------------------------------------------------------------------ mount
  function mount(container, opts) {
    opts = opts || {};
    const mode = opts.mode || "device";
    if (mode !== "device" && mode !== "network") {
      throw new Error("activation.js: unknown mode " + mode);
    }
    P.injectStyle(STYLE_ID, CSS);
    const root = el("div", "ea-root");
    container.innerHTML = "";
    container.appendChild(root);
    return mode === "device" ? mountDevice(root, opts) : mountNetwork(root, opts);
  }

  // transfer/field/landmarks/samples are exported because the drawing is only
  // worth trusting if something checks it: tests/test_activation_widget.py holds
  // them to the closed form and to an MZI built from photonn.mzi.
  const api = { mount, transfer, field, landmarks, samples, axisRange, fmtPower,
                _net: { load: loadNet, forward: forward } };
  if (typeof window !== "undefined") window.PhotonnActivation = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})();
