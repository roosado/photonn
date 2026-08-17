/*
 * errors.js -- what each fabrication error physically does to the machine.
 *
 * One widget per error source, all seven on the tolerance page: six against the
 * diffractive stack, and a seventh against the MZI mesh in the section that
 * compares the two. Each is purpose-built for its own mechanism rather than one
 * generic
 * slider: phase error scatters every pixel independently, quantisation collapses
 * a continuum onto steps, crosstalk smears neighbours together, loss drains a
 * photon budget, wavelength drift acts through two channels at once, detector
 * noise corrupts ten sums, and an imbalanced coupler sends the wrong fraction of
 * the light the wrong way, 72 columns in a row. Drawn as one file because they
 * share a palette, a stylesheet and a set of drawing helpers; seven files would
 * mean seven STYLE_IDs to keep distinct, and this repo has already shipped that
 * bug once.
 *
 * MECHANISM ONLY. Nothing here runs a classifier, and no accuracy number is
 * computed in any widget: the measured tolerance curve sits beside it, and is
 * the only thing that quotes accuracy. That keeps these honest -- they
 * illustrate what the error is, they do not make new claims about what it costs.
 *
 * Both models are the real thing. The mask is mask 0 of the trained D2NN,
 * carried by the bundle in window.PHOTONN_ERR_MASK, and its codes decode exactly
 * as d2nn.js decodes them: phi = code * 2*pi/256 - pi. The mesh is the trained
 * Phase-3 chip, carried by mesh_weights.js (window.PHOTONN_MESH) at 16 bits.
 * Either may be absent -- the widgets fall back to a deterministic stand-in so
 * the file still mounts under Node, where layout, not physics, is under test.
 *
 * Usage:  window.PhotonnErrors.mount(el, {kind: "crosstalk"})
 *   kind: phase | quant | crosstalk | loss | wavelength | detector | mesh
 */
(function () {
  "use strict";

  const STYLE_ID = "ex-style";
  const MAX_DPR = 2;
  function dpr() { return Math.min(window.devicePixelRatio || 1, MAX_DPR); }

  const SRC = (typeof window !== "undefined" && window.PHOTONN_ERR_MASK)
    ? window.PHOTONN_ERR_MASK
    : (typeof require !== "undefined" ? require("./error_mask.js") : null);

  const MESH_SRC = (typeof window !== "undefined" && window.PHOTONN_MESH)
    ? window.PHOTONN_MESH
    : null;

  const CSS = `
.ex-root{--ex-fg:#1b1f24;--ex-muted:#5a6472;--ex-panel:#f4f6f9;--ex-border:#d7dde5;
  --ex-accent:#3b6ea5;--ex-ok:#3f8f4e;--ex-warn:#c14a3d;
  color:var(--ex-fg);font:13px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  background:var(--ex-panel);border:1px solid var(--ex-border);border-radius:10px;padding:13px 15px;}
@media (prefers-color-scheme:dark){.ex-root{--ex-fg:#e6eaf0;--ex-muted:#9aa6b5;
  --ex-panel:#1c2128;--ex-border:#30363d;--ex-accent:#6ea8e0;--ex-ok:#5cc06e;--ex-warn:#e0705f;}}
:root[data-theme="dark"] .ex-root{--ex-fg:#e6eaf0;--ex-muted:#9aa6b5;--ex-panel:#1c2128;
  --ex-border:#30363d;--ex-accent:#6ea8e0;--ex-ok:#5cc06e;--ex-warn:#e0705f;}
:root[data-theme="light"] .ex-root{--ex-fg:#1b1f24;--ex-muted:#5a6472;--ex-panel:#f4f6f9;
  --ex-border:#d7dde5;--ex-accent:#3b6ea5;--ex-ok:#3f8f4e;--ex-warn:#c14a3d;}
.ex-row{display:flex;gap:14px;align-items:flex-end;flex-wrap:wrap;margin-bottom:11px;}
.ex-row label{font-size:11px;font-weight:600;color:var(--ex-muted);text-transform:uppercase;
  letter-spacing:.04em;display:block;margin-bottom:3px;}
.ex-row input[type=range]{width:220px;max-width:100%;accent-color:var(--ex-accent);display:block;}
.ex-val{font-variant-numeric:tabular-nums;font-weight:600;font-size:14px;white-space:nowrap;}
/* The three panels are a before/after/difference triptych: the widget only says
   anything if all three are on screen while the slider moves. So they never
   wrap -- a zero flex-basis plus min-width:0 lets them share whatever width
   there is, down to a phone, instead of overflowing onto a second row and
   leaving the reader scrolling between panels meant to be compared at a glance. */
.ex-panes{display:flex;gap:12px;align-items:flex-start;}
.ex-pane{flex:1 1 0;min-width:0;}
/* A single plot spanning the row, capped so it does not sprawl into a 1000 px
   letterbox on a desktop. */
.ex-pane.ex-wide{flex:1 1 100%;max-width:640px;margin:0 auto;}
.ex-pane canvas{width:100%;height:auto;display:block;border-radius:6px;background:#000;}
.ex-cap{font-size:11px;color:var(--ex-muted);margin:5px 0 0;text-align:center;line-height:1.35;}
@media (max-width:560px){
  .ex-panes{gap:7px;}
  .ex-cap{font-size:10px;line-height:1.3;}
  .ex-row{gap:10px;margin-bottom:9px;}
  .ex-row input[type=range]{width:min(220px,60vw);}
}
.ex-note{font-size:12px;color:var(--ex-muted);margin:11px 0 0;line-height:1.5;}
.ex-note b{color:var(--ex-fg);}
.ex-flag{font-weight:600;}
.ex-flag.ok{color:var(--ex-ok);} .ex-flag.warn{color:var(--ex-warn);}
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

  function b64ToBytes(b64) {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  // ---------------------------------------------------------------- the mask
  const N = SRC ? SRC.n : 128;
  const CODES = SRC ? b64ToBytes(SRC.mask_b64) : new Uint8Array(N * N);
  const PHASE = new Float32Array(N * N);
  for (let i = 0; i < PHASE.length; i++) PHASE[i] = CODES[i] * (2 * Math.PI / 256) - Math.PI;

  /** Cyclic colour wheel: phase is an angle, so its palette has to wrap. */
  function phaseRGB(phi) {
    const t = (phi + Math.PI) / (2 * Math.PI);          // 0..1
    const a = t * 2 * Math.PI;
    return [
      Math.round(128 + 118 * Math.cos(a)),
      Math.round(128 + 118 * Math.cos(a - 2.094)),
      Math.round(128 + 118 * Math.cos(a + 2.094)),
    ];
  }

  function drawPhase(canvas, field, size) {
    const n = size || N;
    canvas.width = n; canvas.height = n;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(n, n);
    for (let i = 0; i < n * n; i++) {
      const [r, g, b] = phaseRGB(field[i]);
      img.data[4 * i] = r; img.data[4 * i + 1] = g;
      img.data[4 * i + 2] = b; img.data[4 * i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  /** Grey magnitude map, auto-scaled, for difference panels. */
  function drawMag(canvas, field, size, cap) {
    const n = size || N;
    canvas.width = n; canvas.height = n;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(n, n);
    let hi = cap || 0;
    if (!cap) for (let i = 0; i < n * n; i++) hi = Math.max(hi, Math.abs(field[i]));
    hi = hi || 1;
    for (let i = 0; i < n * n; i++) {
      const v = Math.min(1, Math.abs(field[i]) / hi);
      const c = Math.round(255 * v);
      img.data[4 * i] = c; img.data[4 * i + 1] = Math.round(c * 0.55);
      img.data[4 * i + 2] = Math.round(c * 0.30); img.data[4 * i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  function pane(parent, caption, wide) {
    const p = el("div", "ex-pane" + (wide ? " ex-wide" : ""));
    const c = el("canvas");
    p.appendChild(c);
    p.appendChild(el("p", "ex-cap", caption));
    parent.appendChild(p);
    return c;
  }

  /**
   * Size a plot canvas to the width it is *actually* laid out at, so that one
   * drawing unit is one CSS pixel in both directions.
   *
   * The panel canvases carry square bitmaps and can be left to `width:100%`,
   * but a plot cannot. These used to draw into a fixed 420-unit space and then
   * let `width:100%` stretch the bitmap to whatever the column happened to be,
   * which scales x without scaling y: the chart came out flattened on a wide
   * screen and stretched tall on a narrow one, and the text with it. Measuring
   * first is the whole fix. `heightFor` maps the measured width to a height, so
   * the shape stays sane across the range rather than being pinned to one.
   *
   * `setTransform` is correct here, unlike in d2nn_stage.js: resizing a canvas
   * resets its context, so this call is establishing the dpr scale, not
   * replacing one that is already there.
   */
  function fitCanvas(c, heightFor) {
    const r = dpr();
    const W = Math.max(240, Math.round(c.getBoundingClientRect().width || 420));
    const H = Math.round(heightFor(W));
    c.width = Math.round(W * r);
    c.height = Math.round(H * r);
    c.style.height = H + "px";
    const ctx = c.getContext("2d");
    ctx.setTransform(r, 0, 0, r, 0, 0);
    return { ctx, W, H };
  }

  /**
   * Re-run `fn` when `target` changes width, and only then.
   *
   * Guarded on the measured width rather than firing on every observation
   * because `fn` sets the canvas height, which is itself a resize: an unguarded
   * observer would answer its own callback forever.
   */
  function onWidthChange(target, fn) {
    let last = -1;
    const check = function () {
      const w = Math.round(target.getBoundingClientRect().width);
      if (w === last) return;
      last = w;
      fn();
    };
    if (typeof window.ResizeObserver === "function") {
      new window.ResizeObserver(check).observe(target);
    } else {
      window.addEventListener("resize", check);
    }
  }

  function slider(parent, label, min, max, step, value) {
    const wrap = el("div");
    const lab = el("label", null, label);
    const input = document.createElement("input");
    input.type = "range";
    input.min = String(min); input.max = String(max);
    input.step = String(step); input.value = String(value);
    input.setAttribute("aria-label", label);
    const val = el("span", "ex-val", "");
    const valWrap = el("div");
    valWrap.appendChild(val);
    wrap.appendChild(lab); wrap.appendChild(input);
    parent.appendChild(wrap);
    parent.appendChild(valWrap);
    return { input, val };
  }

  /** Deterministic normal draws, so the picture is stable across redraws. */
  function gaussians(count, seed) {
    let s = seed >>> 0;
    const out = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      s = (s * 1664525 + 1013904223) >>> 0;
      const u1 = ((s >>> 8) / 16777216) || 1e-9;
      s = (s * 1664525 + 1013904223) >>> 0;
      const u2 = (s >>> 8) / 16777216;
      out[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    return out;
  }
  const NOISE = gaussians(N * N, 20260814);

  // ------------------------------------------------------------------ kinds
  const KINDS = {};

  // 1. Phase-setting error: every pixel scattered independently.
  KINDS.phase = function (root) {
    const ctl = el("div", "ex-row");
    root.appendChild(ctl);
    const s = slider(ctl, "Phase error σ", 0, 0.6, 0.01, 0.15);
    const panes = el("div", "ex-panes");
    root.appendChild(panes);
    const a = pane(panes, "as designed");
    const b = pane(panes, "as built");
    const d = pane(panes, "the difference");
    const note = el("p", "ex-note");
    root.appendChild(note);
    drawPhase(a, PHASE);
    const built = new Float32Array(N * N);
    const diff = new Float32Array(N * N);
    function update() {
      const sig = +s.input.value;
      for (let i = 0; i < N * N; i++) {
        built[i] = PHASE[i] + sig * NOISE[i];
        diff[i] = built[i] - PHASE[i];
      }
      drawPhase(b, built);
      drawMag(d, diff, N, 0.6);
      s.val.textContent = sig.toFixed(2) + " rad";
      const ok = sig <= 0.3;
      note.innerHTML = "Each pixel is set slightly wrong, independently of its neighbours, so "
        + "the error is <b>salt-and-pepper rather than structured</b>. The middle panel still "
        + "looks like the design because a small random offset does not move the large-scale "
        + "pattern. <span class='ex-flag " + (ok ? "ok" : "warn") + "'>"
        + (ok ? "Inside" : "Past") + " the measured limit of 0.3 rad.</span> A well-calibrated "
        + "device sits near 0.05 rad, an order of magnitude inside it.";
    }
    s.input.addEventListener("input", update);
    update();
  };

  // 2. Quantisation: a continuum collapsed onto 2^bits levels.
  KINDS.quant = function (root) {
    const ctl = el("div", "ex-row");
    root.appendChild(ctl);
    const s = slider(ctl, "Control resolution", 1, 8, 1, 3);
    const panes = el("div", "ex-panes");
    root.appendChild(panes);
    const a = pane(panes, "as designed");
    const b = pane(panes, "as built");
    const h = pane(panes, "phases actually used");
    const note = el("p", "ex-note");
    root.appendChild(note);
    drawPhase(a, PHASE);
    const built = new Float32Array(N * N);
    function update() {
      const bits = +s.input.value;
      const levels = 1 << bits;
      const step = 2 * Math.PI / levels;
      const hist = new Float64Array(levels);
      for (let i = 0; i < N * N; i++) {
        const k = Math.min(levels - 1, Math.floor((PHASE[i] + Math.PI) / step));
        built[i] = k * step - Math.PI + step / 2;
        hist[k]++;
      }
      drawPhase(b, built);
      // Histogram: a continuum of phases collapsing onto a comb of spikes.
      const n = 128;
      h.width = n; h.height = n;
      const ctx = h.getContext("2d");
      ctx.fillStyle = "#0b0d10"; ctx.fillRect(0, 0, n, n);
      const hi = Math.max.apply(null, Array.from(hist)) || 1;
      const w = Math.max(1, Math.floor(n / levels) - 1);
      for (let k = 0; k < levels; k++) {
        const x = Math.round((k + 0.5) / levels * n);
        const barH = Math.round(hist[k] / hi * (n - 12));
        const [r, g, bl] = phaseRGB(k * step - Math.PI + step / 2);
        ctx.fillStyle = `rgb(${r},${g},${bl})`;
        ctx.fillRect(x - (w >> 1), n - barH, Math.max(1, w), barH);
      }
      s.val.textContent = bits + (bits === 1 ? " bit" : " bits") + " · " + levels
        + (levels === 2 ? " setting" : " settings");
      const ok = bits >= 3;
      note.innerHTML = "A real driver cannot hold any phase you like, only a fixed number of "
        + "steps. The right-hand panel is every phase on the mask, counted: the smooth spread "
        + "the design wants <b>collapses onto a comb of " + levels + "</b>. "
        + "<span class='ex-flag " + (ok ? "ok" : "warn") + "'>"
        + (ok ? "Accuracy still holds here." : "Below the measured 3-bit limit.") + "</span> "
        + "Real hardware offers 8 bits, so this one is comfortable.";
    }
    s.input.addEventListener("input", update);
    update();
  };

  // 3. Crosstalk: neighbours bleed into each other. The binding constraint.
  KINDS.crosstalk = function (root) {
    const ctl = el("div", "ex-row");
    root.appendChild(ctl);
    const s = slider(ctl, "Bleed between pixels", 0, 1.5, 0.05, 0.25);
    const panes = el("div", "ex-panes");
    root.appendChild(panes);
    const a = pane(panes, "as designed");
    const b = pane(panes, "as built");
    const z = pane(panes, "close up, 24 px across");
    const note = el("p", "ex-note");
    root.appendChild(note);
    drawPhase(a, PHASE);
    const built = new Float32Array(N * N);
    const tmp = new Float32Array(N * N);
    // Blur the mask as a complex phasor, not as an angle: averaging raw angles
    // across the +pi/-pi wrap would invent phases that are in neither pixel.
    const re = new Float32Array(N * N), im = new Float32Array(N * N);
    const re2 = new Float32Array(N * N), im2 = new Float32Array(N * N);
    function update() {
      const sig = +s.input.value;
      if (sig < 0.02) {
        built.set(PHASE);
      } else {
        const rad = Math.max(1, Math.ceil(3 * sig));
        const k = [];
        let sum = 0;
        for (let t = -rad; t <= rad; t++) {
          const v = Math.exp(-(t * t) / (2 * sig * sig));
          k.push(v); sum += v;
        }
        for (let i = 0; i < k.length; i++) k[i] /= sum;
        for (let i = 0; i < N * N; i++) { re[i] = Math.cos(PHASE[i]); im[i] = Math.sin(PHASE[i]); }
        for (let y = 0; y < N; y++) {                       // horizontal pass
          for (let x = 0; x < N; x++) {
            let ar = 0, ai = 0;
            for (let t = -rad; t <= rad; t++) {
              const xx = Math.min(N - 1, Math.max(0, x + t)), w = k[t + rad];
              ar += w * re[y * N + xx]; ai += w * im[y * N + xx];
            }
            re2[y * N + x] = ar; im2[y * N + x] = ai;
          }
        }
        for (let x = 0; x < N; x++) {                       // vertical pass
          for (let y = 0; y < N; y++) {
            let ar = 0, ai = 0;
            for (let t = -rad; t <= rad; t++) {
              const yy = Math.min(N - 1, Math.max(0, y + t)), w = k[t + rad];
              ar += w * re2[yy * N + x]; ai += w * im2[yy * N + x];
            }
            built[y * N + x] = Math.atan2(ai, ar);
          }
        }
      }
      drawPhase(b, built);
      // Close-up: the same 24x24 corner, nearest-neighbour so pixels stay pixels.
      const M = 24, SCALE = 6;
      z.width = M * SCALE; z.height = M * SCALE;
      const ctx = z.getContext("2d");
      for (let y = 0; y < M; y++) {
        for (let x = 0; x < M; x++) {
          const [r, g, bl] = phaseRGB(built[(y + 52) * N + (x + 52)]);
          ctx.fillStyle = `rgb(${r},${g},${bl})`;
          ctx.fillRect(x * SCALE, y * SCALE, SCALE, SCALE);
        }
      }
      s.val.textContent = sig.toFixed(2) + " px";
      const ok = sig <= 0.25;
      note.innerHTML = "Setting one pixel disturbs the pixels around it, because heat spreads "
        + "and electric fields spill sideways. Watch the close-up: the sharp pixel-to-pixel "
        + "steps the design depends on <b>wash into each other</b>. "
        + "<span class='ex-flag " + (ok ? "ok" : "warn") + "'>"
        + (ok ? "At or inside the 0.25 px limit." : "Past the 0.25 px limit that still holds.")
        + "</span> A real LCoS device bleeds about <b>1 px</b>, four times too much, which is "
        + "why this is the source that decides the whole question.";
    }
    s.input.addEventListener("input", update);
    update();
  };

  // 4. Loss: no effect at all until the photon budget runs out.
  KINDS.loss = function (root) {
    const ctl = el("div", "ex-row");
    root.appendChild(ctl);
    const s = slider(ctl, "Lost per surface", 0, 3, 0.1, 1);
    const panes = el("div", "ex-panes");
    root.appendChild(panes);
    const c = pane(panes, "photons surviving, surface by surface", true);
    const note = el("p", "ex-note");
    root.appendChild(note);
    function update() {
      const db = +s.input.value;
      const { ctx, W, H } = fitCanvas(c, (w) => Math.max(150, Math.min(200, w * 0.34)));
      ctx.fillStyle = "#0b0d10"; ctx.fillRect(0, 0, W, H);
      const stages = 6;
      let frac = 1;
      const bw = W / (stages + 1);
      for (let i = 0; i <= stages; i++) {
        const h = Math.max(1, frac * (H - 34));
        ctx.fillStyle = i === stages ? "#6ea8e0" : "#3f8f4e";
        ctx.fillRect(i * bw + 6, H - 18 - h, bw - 12, h);
        ctx.fillStyle = "#9aa6b5";
        ctx.font = "10px ui-monospace, monospace";
        ctx.textAlign = "center";
        ctx.fillText(i === 0 ? "in" : ("m" + i), i * bw + bw / 2, H - 5);
        ctx.fillText((frac * 100).toFixed(frac < 0.1 ? 1 : 0) + "%", i * bw + bw / 2, H - 24 - h);
        frac *= Math.pow(10, -db / 10);
      }
      const surviving = Math.pow(10, -db * stages / 10);
      s.val.textContent = db.toFixed(1) + " dB/surface";
      note.innerHTML = "After five masks and the entrance, <b>" + (surviving * 100).toFixed(2)
        + "%</b> of the light is left. Here is the counter-intuitive part: on its own this "
        + "costs <b>no accuracy at all</b>. The readout compares the ten detector boxes "
        + "against each other, so dimming every one of them by the same factor changes "
        + "nothing about which is largest. Loss only bites once so few photons arrive that "
        + "counting them becomes unreliable, which is why it is swept together with the "
        + "light budget rather than alone.";
    }
    s.input.addEventListener("input", update);
    onWidthChange(c, update);
    update();
  };

  // 5. Wavelength drift: two mechanisms at once.
  KINDS.wavelength = function (root) {
    const ctl = el("div", "ex-row");
    root.appendChild(ctl);
    const s = slider(ctl, "Wavelength drift", 0, 40, 1, 10);
    const panes = el("div", "ex-panes");
    root.appendChild(panes);
    const a = pane(panes, "designed for 532 nm");
    const b = pane(panes, "at the drifted colour");
    const d = pane(panes, "the difference");
    const note = el("p", "ex-note");
    root.appendChild(note);
    drawPhase(a, PHASE);
    const built = new Float32Array(N * N);
    const diff = new Float32Array(N * N);
    function wrap(p) { return Math.atan2(Math.sin(p), Math.cos(p)); }
    function update() {
      const dl = +s.input.value;
      const lam0 = 532, lam = lam0 + dl;
      const scale = lam0 / lam;                 // a fixed etch depth is a phase that scales
      for (let i = 0; i < N * N; i++) {
        built[i] = wrap(PHASE[i] * scale);
        diff[i] = wrap(built[i] - PHASE[i]);
      }
      drawPhase(b, built);
      drawMag(d, diff, N, 0.6);
      const reach = 12.47 * (lam / lam0);
      s.val.textContent = "+" + dl + " nm · " + lam + " nm";
      const ok = dl <= 10;
      note.innerHTML = "Drift acts through <b>two channels at once</b>. The plates are a fixed "
        + "depth of glass, so the delay they impose scales as λ₀/λ and every "
        + "phase on the mask shifts (left and middle). And diffraction itself is "
        + "colour-dependent, so the light spreads further per gap: <b>"
        + reach.toFixed(2) + " px</b> against 12.47 at the design colour. "
        + "<span class='ex-flag " + (ok ? "ok" : "warn") + "'>"
        + (ok ? "Inside the measured 10 nm limit." : "Past the 10 nm limit.") + "</span> "
        + "Any temperature-controlled laser holds far tighter than this.";
    }
    s.input.addEventListener("input", update);
    update();
  };

  // 6. Detector noise: ten sums, corrupted three ways.
  KINDS.detector = function (root) {
    const ctl = el("div", "ex-row");
    root.appendChild(ctl);
    const s = slider(ctl, "Input power", 0, 6, 1, 4);
    const POWERS = [1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-9, 1e-3];
    const LABELS = ["1 fW", "10 fW", "100 fW", "1 pW", "10 pW", "1 nW", "1 mW"];
    const panes = el("div", "ex-panes");
    root.appendChild(panes);
    const c = pane(panes, "light collected by each of the ten class boxes", true);
    const note = el("p", "ex-note");
    root.appendChild(note);
    // A plausible clean readout: one clear winner, a runner-up, and eight others.
    const CLEAN = [0.055, 0.041, 0.062, 0.049, 0.038, 0.271, 0.052, 0.044, 0.180, 0.058];
    const NZ = gaussians(64, 991);
    function update() {
      const idx = +s.input.value;
      const power = POWERS[idx];
      // 532 nm photon = 3.73e-19 J (h*c/lambda); 1 ms exposure.
      const photons = power * 1e-3 / 3.73e-19;
      const { ctx, W, H } = fitCanvas(c, (w) => Math.max(158, Math.min(210, w * 0.36)));
      ctx.fillStyle = "#0b0d10"; ctx.fillRect(0, 0, W, H);
      let best = -1, bestV = -1;
      const vals = [];
      for (let k = 0; k < 10; k++) {
        const mean = CLEAN[k] * photons;
        // shot noise (Gaussian approximation) + 2 electrons of read noise
        const noisy = mean + Math.sqrt(Math.max(mean, 0)) * NZ[k] + 2 * NZ[k + 10];
        const v = Math.max(0, noisy) / Math.max(photons, 1e-9);
        vals.push(v);
        if (v > bestV) { bestV = v; best = k; }
      }
      const hi = Math.max.apply(null, vals) || 1;
      const bw = W / 10;
      for (let k = 0; k < 10; k++) {
        const h = Math.max(1, vals[k] / hi * (H - 40));
        ctx.fillStyle = k === best ? (best === 5 ? "#5cc06e" : "#e0705f") : "#42506a";
        ctx.fillRect(k * bw + 7, H - 20 - h, bw - 14, h);
        ctx.fillStyle = "#9aa6b5";
        ctx.font = "11px ui-monospace, monospace";
        ctx.textAlign = "center";
        ctx.fillText(String(k), k * bw + bw / 2, H - 6);
      }
      s.val.textContent = LABELS[idx];
      const right = best === 5;
      note.innerHTML = "Light arrives as a countable number of photons, so a dim readout is "
        + "<b>grainy</b>: roughly " + (photons < 1e4 ? photons.toFixed(0)
          : photons.toExponential(1)) + " photons over a 1 ms exposure, plus read noise from "
        + "the sensor. Class <b>5</b> is the true answer here. "
        + "<span class='ex-flag " + (right ? "ok" : "warn") + "'>"
        + (right ? "Still read correctly." : "The noise has flipped the answer to "
          + best + ".") + "</span> Accuracy holds down to 1 pW; the nominal operating point "
        + "is a million times brighter than that.";
    }
    s.input.addEventListener("input", update);
    onWidthChange(c, update);
    update();
  };

  // ------------------------------------------------------- the mesh, on demand
  // Everything below is used by one widget on one page. Decoding 5,300 codes at
  // load time would charge the tolerance page for a chip it never draws, so the
  // mesh is built the first time the mesh widget mounts and cached after.
  let MESH = null;

  /** Decode 16-bit little-endian codes back to the span they were quantised over. */
  function decode16(b64, span) {
    const bytes = b64ToBytes(b64);
    const out = new Float64Array(bytes.length >> 1);
    for (let i = 0; i < out.length; i++) {
      const code = bytes[2 * i] | (bytes[2 * i + 1] << 8);
      out[i] = (code + 0.5) / 65536 * span;
    }
    return out;
  }

  /**
   * The Clements schedule: column c couples mode pairs starting at c % 2.
   *
   * Derived rather than shipped. It is three lines of arithmetic and it is the
   * same rule MZIMeshLayer._schedule applies, so re-deriving it costs nothing and
   * removes one more array that could go stale against the model.
   */
  function schedule(n) {
    const cols = [];
    let idx = 0;
    for (let c = 0; c < n; c++) {
      const pairs = [];
      for (let m = c % 2; m < n - 1; m += 2) pairs.push([m, idx++]);
      cols.push(pairs);
    }
    return { columns: cols, nMzi: idx };
  }

  /**
   * The 2x2 MZI block, written as the two couplers and two phase shifters it is
   * physically made of: B(s2) . P(theta) . B(s1) . P(phi).
   *
   * With both splits at 0.5 this collapses to i.e^{i.theta/2}.[[e^{i.phi}s, c],
   * [e^{i.phi}c, -s]], which is the closed form photonn.mzi.mzi_matrix and the
   * torch layer both use. It is written factored anyway, because the factored
   * form is the only one an imbalanced coupler can enter -- exactly the reason
   * photonn-hw/+meshmodel/mzi_matrix.m is written this way too.
   */
  function mziBlock(theta, phi, s1, s2, out) {
    const k1 = Math.asin(Math.sqrt(Math.min(1, Math.max(0, s1))));
    const k2 = Math.asin(Math.sqrt(Math.min(1, Math.max(0, s2))));
    const c1 = Math.cos(k1), n1 = Math.sin(k1);
    const c2 = Math.cos(k2), n2 = Math.sin(k2);
    // A = B(s1) . P(phi): P(phi) = diag(e^{i.phi}, 1), so it scales column 0.
    const cp = Math.cos(phi), sp = Math.sin(phi);
    const a = [c1 * cp, c1 * sp, 0, n1,      // [0][0], [0][1] = i.n1
               -n1 * sp, n1 * cp, c1, 0];    // [1][0] = i.n1.e^{i.phi}, [1][1]
    // P(theta) scales row 0 of A.
    const ct = Math.cos(theta), st = Math.sin(theta);
    const b = [a[0] * ct - a[1] * st, a[0] * st + a[1] * ct,
               a[2] * ct - a[3] * st, a[2] * st + a[3] * ct,
               a[4], a[5], a[6], a[7]];
    // B(s2) . that.  B = [[c2, i.n2], [i.n2, c2]].
    out[0] = c2 * b[0] - n2 * b[5];  out[1] = c2 * b[1] + n2 * b[4];
    out[2] = c2 * b[2] - n2 * b[7];  out[3] = c2 * b[3] + n2 * b[6];
    out[4] = c2 * b[4] - n2 * b[1];  out[5] = c2 * b[5] + n2 * b[0];
    out[6] = c2 * b[6] - n2 * b[3];  out[7] = c2 * b[7] + n2 * b[2];
  }

  /**
   * One mesh, built column by column into an n-by-n complex matrix.
   *
   * Column by column rather than as one product because that is where a per-MZI
   * error has to be able to enter; `splits` is 2 per MZI, in schedule order.
   */
  function meshMatrix(n, sch, theta, phi, outPhase, off, splits) {
    const re = new Float64Array(n * n), im = new Float64Array(n * n);
    for (let i = 0; i < n; i++) re[i * n + i] = 1;
    const blk = new Float64Array(8);
    const rowA = new Float64Array(2 * n), rowB = new Float64Array(2 * n);
    for (let c = 0; c < sch.columns.length; c++) {
      const pairs = sch.columns[c];
      for (let p = 0; p < pairs.length; p++) {
        const m = pairs[p][0], j = pairs[p][1];
        mziBlock(theta[off + j], phi[off + j],
                 splits ? splits[2 * j] : 0.5, splits ? splits[2 * j + 1] : 0.5, blk);
        const r0 = m * n, r1 = (m + 1) * n;
        for (let k = 0; k < n; k++) {
          const ar = re[r0 + k], ai = im[r0 + k], br = re[r1 + k], bi = im[r1 + k];
          rowA[2 * k] = blk[0] * ar - blk[1] * ai + blk[2] * br - blk[3] * bi;
          rowA[2 * k + 1] = blk[0] * ai + blk[1] * ar + blk[2] * bi + blk[3] * br;
          rowB[2 * k] = blk[4] * ar - blk[5] * ai + blk[6] * br - blk[7] * bi;
          rowB[2 * k + 1] = blk[4] * ai + blk[5] * ar + blk[6] * bi + blk[7] * br;
        }
        for (let k = 0; k < n; k++) {
          re[r0 + k] = rowA[2 * k]; im[r0 + k] = rowA[2 * k + 1];
          re[r1 + k] = rowB[2 * k]; im[r1 + k] = rowB[2 * k + 1];
        }
      }
    }
    // The output phase screen: a diagonal e^{i.psi} on the left, so it scales rows.
    for (let i = 0; i < n; i++) {
      const cp = Math.cos(outPhase[i]), sp = Math.sin(outPhase[i]);
      for (let k = 0; k < n; k++) {
        const r = re[i * n + k], q = im[i * n + k];
        re[i * n + k] = r * cp - q * sp;
        im[i * n + k] = r * sp + q * cp;
      }
    }
    return { re: re, im: im };
  }

  /** |U . diag(sigma) . V| -- the magnitude of the map the chip computes. */
  function operatorMag(M, n, sigma, splits) {
    const v = meshMatrix(n, M.sch, M.theta, M.phi, M.outV, 0, splits);
    const u = meshMatrix(n, M.sch, M.theta, M.phi, M.outU, M.nMzi,
                         splits ? splits.subarray(2 * M.nMzi) : null);
    const out = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < n; k++) {
        let ar = 0, ai = 0;
        for (let j = 0; j < n; j++) {
          const s = sigma[j];
          const ur = u.re[i * n + j] * s, ui = u.im[i * n + j] * s;
          const vr = v.re[j * n + k], vi = v.im[j * n + k];
          ar += ur * vr - ui * vi;
          ai += ur * vi + ui * vr;
        }
        out[i * n + k] = Math.sqrt(ar * ar + ai * ai);
      }
    }
    return out;
  }

  function loadMesh() {
    if (MESH) return MESH;
    const n = MESH_SRC ? MESH_SRC.n : 36;
    const sch = schedule(n);
    let theta, phi, sigma, outPhase;
    if (MESH_SRC) {
      theta = decode16(MESH_SRC.theta_b64, 2 * Math.PI);
      phi = decode16(MESH_SRC.phi_b64, 2 * Math.PI);
      sigma = decode16(MESH_SRC.sigma_b64, 1);
      outPhase = decode16(MESH_SRC.out_phase_b64, 2 * Math.PI);
    } else {
      // Stand-in with the right shape, so the widget mounts where the bundle is
      // not inlined. Never reached on a built page.
      const g = gaussians(4 * sch.nMzi + 4 * n, 20260817);
      theta = new Float64Array(2 * sch.nMzi);
      phi = new Float64Array(2 * sch.nMzi);
      for (let i = 0; i < theta.length; i++) {
        theta[i] = Math.abs(g[i]) % Math.PI;
        phi[i] = Math.abs(g[i + theta.length]) % (2 * Math.PI);
      }
      sigma = new Float64Array(n);
      for (let i = 0; i < n; i++) sigma[i] = 1 - i / n;
      outPhase = new Float64Array(2 * n);
    }
    MESH = {
      n: n, sch: sch, nMzi: sch.nMzi, theta: theta, phi: phi, sigma: sigma,
      outV: outPhase.subarray(0, n), outU: outPhase.subarray(n, 2 * n),
      splitNoise: gaussians(4 * sch.nMzi, 20260817),
    };
    return MESH;
  }

  /** Nearest-neighbour upscale of an n-by-n magnitude map, so cells stay square. */
  function drawMatrix(canvas, mag, n, cap, scale) {
    const px = n * scale;
    canvas.width = px; canvas.height = px;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(px, px);
    const hi = cap || 1;
    for (let y = 0; y < px; y++) {
      const sy = (y / scale) | 0;
      for (let x = 0; x < px; x++) {
        const v = Math.min(1, mag[sy * n + ((x / scale) | 0)] / hi);
        const i = 4 * (y * px + x);
        const c = Math.round(255 * v);
        img.data[i] = c;
        img.data[i + 1] = Math.round(c * 0.55);
        img.data[i + 2] = Math.round(c * 0.30);
        img.data[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  // 7. Coupler imbalance, on the mesh. The one error source a phase mask cannot
  //    have, and the one that turns 72 columns of depth into a liability.
  KINDS.mesh = function (root) {
    const M = loadMesh();
    const n = M.n;
    const ctl = el("div", "ex-row");
    root.appendChild(ctl);
    const s = slider(ctl, "Coupler split error σ", 0, 0.05, 0.002, 0.01);
    const panes = el("div", "ex-panes");
    root.appendChild(panes);
    const a = pane(panes, "as designed");
    const b = pane(panes, "as built");
    const d = pane(panes, "the difference");
    const note = el("p", "ex-note");
    root.appendChild(note);

    const SCALE = 8;                       // 36 modes -> a 288 px bitmap
    const ideal = operatorMag(M, n, M.sigma, null);
    let cap = 0;
    for (let i = 0; i < ideal.length; i++) cap = Math.max(cap, ideal[i]);
    drawMatrix(a, ideal, n, cap, SCALE);

    const splits = new Float64Array(4 * M.nMzi);
    const built = new Float64Array(n * n);
    const diff = new Float64Array(n * n);
    // A fixed scale on the difference panel, so dragging the slider shows the
    // damage growing rather than an auto-scale that hides it. 0.12 of the
    // operator's own peak puts the measured limit (0.01) at about half-bright and
    // saturates well before the slider ends, which is the right way round: the
    // reader should run out of headroom past the limit, not before it.
    const DIFF_CAP = 0.12 * cap;
    function update() {
      const eps = +s.input.value;
      for (let i = 0; i < splits.length; i++) splits[i] = 0.5 + eps * M.splitNoise[i];
      const got = operatorMag(M, n, M.sigma, eps > 0 ? splits : null);
      let num = 0, den = 0;
      for (let i = 0; i < built.length; i++) {
        built[i] = got[i];
        diff[i] = got[i] - ideal[i];
        num += diff[i] * diff[i];
        den += ideal[i] * ideal[i];
      }
      drawMatrix(b, built, n, cap, SCALE);
      drawMatrix(d, diff, n, DIFF_CAP, SCALE);
      const rms = 100 * Math.sqrt(num / (den || 1));
      s.val.textContent = "±" + eps.toFixed(3) + " · " + rms.toFixed(1)
        + "% operator shift";
      const ok = eps <= 0.01;
      note.innerHTML = "Each square is one entry of the 36&times;36 operation the chip "
        + "performs: how much of input mode <i>k</i> arrives at output mode <i>i</i>. "
        + "A coupler that splits 51:49 instead of 50:50 is a small error <b>in one "
        + "interferometer</b> &mdash; but the light crosses <b>72 columns of them in "
        + "series</b>, and every column carries the previous column's error forward. "
        + "That is why the difference panel fills in across the whole matrix rather "
        + "than staying near the couplers that were mis-made. "
        + "<span class='ex-flag " + (ok ? "ok" : "warn") + "'>"
        + (ok ? "Inside" : "Past") + " the measured limit of ±0.01.</span> "
        + "How well a real foundry holds that split is not yet sourced &mdash; the "
        + "limit here is a property of the network, not a verdict on a process.";
    }
    s.input.addEventListener("input", update);
    update();
  };

  function mount(container, opts) {
    opts = opts || {};
    const kind = opts.kind;
    if (!KINDS[kind]) throw new Error("errors.js: unknown kind " + kind);
    injectStyle();
    const root = el("div", "ex-root");
    container.innerHTML = "";
    container.appendChild(root);
    KINDS[kind](root);
  }

  // The chip widget's operator build is physics rather than decoration -- a
  // transposed index or the wrong 2x2 block would draw a confident fiction. So it
  // is exposed for tests/test_mesh_web.py to check against photonn.mzi, rather
  // than re-implemented in the runner, which would only ever test the copy.
  const api = { mount, kinds: Object.keys(KINDS), _mesh: { load: loadMesh, operatorMag } };
  if (typeof window !== "undefined") window.PhotonnErrors = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})();
