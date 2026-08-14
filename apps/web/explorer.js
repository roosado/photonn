/*
 * explorer.js -- live diffraction explorer widget.
 *
 * Mounts a self-contained interactive control panel + canvases that recompute
 * angular-spectrum propagation on every input, using the physics in asm.js
 * (window.ASM). Continuous controls for aperture shape/size, distance, wavelength
 * and grid size, with a live sampling-violation flag (z vs z_crit = n*dx^2/lam).
 *
 * Usage:  window.PhotonnExplorer.mount(containerElement, opts)
 * opts (all optional): { extent, grid, wavelength, apertureSize, distance, shape }
 * SI units in, metres/metres/metres.
 *
 * Depends on window.ASM (asm.js). No external libraries; no network.
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

  // Inferno-style colormap anchors (matplotlib inferno, 9 stops) -> 256 LUT.
  const ANCHORS = [
    [0, 0, 4], [22, 11, 57], [66, 10, 104], [106, 23, 110], [147, 38, 103],
    [188, 55, 84], [221, 81, 58], [243, 120, 25], [252, 255, 164],
  ];
  const LUT = (function () {
    const lut = new Uint8ClampedArray(256 * 3);
    const seg = ANCHORS.length - 1;
    for (let i = 0; i < 256; i++) {
      const t = i / 255 * seg;
      const k = Math.min(seg - 1, Math.floor(t));
      const f = t - k;
      const a = ANCHORS[k], b = ANCHORS[k + 1];
      lut[i * 3] = a[0] + (b[0] - a[0]) * f;
      lut[i * 3 + 1] = a[1] + (b[1] - a[1]) * f;
      lut[i * 3 + 2] = a[2] + (b[2] - a[2]) * f;
    }
    return lut;
  })();

  const STYLE_ID = "pe-style";
  const CSS = `
.pe-root{--pe-fg:#1b1f24;--pe-muted:#5a6472;--pe-panel:#f4f6f9;--pe-border:#d7dde5;
  --pe-accent:#3b6ea5;--pe-ok:#3f8f4e;--pe-warn:#c14a3d;
  color:var(--pe-fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  display:flex;flex-direction:column;gap:14px;max-width:820px;}
@media (prefers-color-scheme:dark){.pe-root{--pe-fg:#e6eaf0;--pe-muted:#9aa6b5;
  --pe-panel:#1c2128;--pe-border:#30363d;--pe-accent:#6ea8e0;--pe-ok:#5cc06e;--pe-warn:#e0705f;}}
.pe-controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
  gap:12px 18px;background:var(--pe-panel);border:1px solid var(--pe-border);
  border-radius:10px;padding:14px 16px;}
.pe-ctl{display:flex;flex-direction:column;gap:4px;}
.pe-ctl > label{font-weight:600;font-size:12px;letter-spacing:.01em;
  display:flex;justify-content:space-between;gap:8px;}
.pe-ctl > label span{color:var(--pe-accent);font-variant-numeric:tabular-nums;font-weight:600;}
.pe-ctl input[type=range]{width:100%;accent-color:var(--pe-accent);}
.pe-seg{display:flex;border:1px solid var(--pe-border);border-radius:7px;overflow:hidden;}
.pe-seg button{flex:1;border:0;background:transparent;color:var(--pe-muted);
  padding:6px 4px;font:inherit;font-size:12px;cursor:pointer;}
.pe-seg button[aria-pressed=true]{background:var(--pe-accent);color:#fff;}
.pe-banner{border-radius:8px;padding:9px 13px;font-weight:600;font-size:13px;
  color:#fff;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;}
.pe-banner.ok{background:var(--pe-ok);}
.pe-banner.warn{background:var(--pe-warn);}
.pe-banner small{font-weight:500;opacity:.92;}
.pe-plots{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start;}
.pe-panelbox{background:var(--pe-panel);border:1px solid var(--pe-border);
  border-radius:10px;padding:10px;}
.pe-panelbox h4{margin:0 0 2px;font-size:12px;color:var(--pe-muted);font-weight:600;
  text-transform:uppercase;letter-spacing:.04em;}
.pe-hint{margin:0 0 8px;font-size:11.5px;line-height:1.4;color:var(--pe-muted);max-width:46ch;}
.pe-field-wrap{display:flex;gap:8px;align-items:stretch;}
.pe-field{width:320px;height:320px;max-width:100%;image-rendering:auto;
  border-radius:6px;display:block;background:#000;}
.pe-colorbar{width:14px;border-radius:4px;
  background:linear-gradient(to top,rgb(0,0,4),rgb(66,10,104),rgb(147,38,103),
    rgb(221,81,58),rgb(252,255,164));}
.pe-cross{width:320px;height:150px;max-width:100%;display:block;}
.pe-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));
  gap:6px 16px;font-size:12px;color:var(--pe-muted);
  border-top:1px solid var(--pe-border);padding-top:10px;}
.pe-meta b{color:var(--pe-fg);font-variant-numeric:tabular-nums;font-weight:600;}
.pe-axis{font-size:11px;color:var(--pe-muted);text-align:center;margin-top:2px;}
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

  function control(label, valueSpanId) {
    const wrap = el("div", "pe-ctl");
    const lab = el("label");
    lab.innerHTML = `${label} <span id="${valueSpanId}"></span>`;
    wrap.appendChild(lab);
    return { wrap, lab };
  }

  function mount(container, opts) {
    opts = opts || {};
    injectStyle();

    const state = {
      extent: opts.extent || 4.0e-3,          // fixed window (m)
      grid: opts.grid || 128,                 // 64 / 128 / 256
      wavelength: opts.wavelength || 633e-9,  // m
      apertureSize: opts.apertureSize || 0.4e-3, // m (full width)
      distance: opts.distance || 0.05,        // m
      shape: opts.shape || "circular",
    };

    const root = el("div", "pe-root");
    container.innerHTML = "";
    container.appendChild(root);

    // ---- controls
    const controls = el("div", "pe-controls");

    // aperture shape (segmented)
    const shapeCtl = el("div", "pe-ctl");
    shapeCtl.appendChild(el("label", null, "Aperture shape"));
    const seg = el("div", "pe-seg");
    const btnCirc = el("button", null, "Circular");
    const btnSq = el("button", null, "Square");
    seg.appendChild(btnCirc); seg.appendChild(btnSq);
    shapeCtl.appendChild(seg);

    // aperture size
    const size = control("Aperture width ⌀", "pe-size-v");
    const sizeIn = el("input");
    sizeIn.type = "range"; sizeIn.min = "0.05"; sizeIn.max = "3.0"; sizeIn.step = "0.01";
    sizeIn.value = (state.apertureSize * 1e3).toString();
    size.wrap.appendChild(sizeIn);

    // distance
    const dist = control("Distance z", "pe-dist-v");
    const distIn = el("input");
    distIn.type = "range"; distIn.min = "1"; distIn.max = "500"; distIn.step = "1";
    distIn.value = (state.distance * 1e3).toString();
    dist.wrap.appendChild(distIn);

    // wavelength
    const wl = control("Wavelength λ", "pe-wl-v");
    const wlIn = el("input");
    wlIn.type = "range"; wlIn.min = "380"; wlIn.max = "1600"; wlIn.step = "1";
    wlIn.value = (state.wavelength * 1e9).toString();
    wl.wrap.appendChild(wlIn);

    // grid size (segmented)
    const gridCtl = el("div", "pe-ctl");
    gridCtl.appendChild(el("label", null, "Grid N (samples/side)"));
    const gseg = el("div", "pe-seg");
    const gbtns = [64, 128, 256].map((g) => {
      const b = el("button", null, String(g));
      b.dataset.g = String(g);
      gseg.appendChild(b);
      return b;
    });
    gridCtl.appendChild(gseg);

    controls.appendChild(shapeCtl);
    controls.appendChild(size.wrap);
    controls.appendChild(dist.wrap);
    controls.appendChild(wl.wrap);
    controls.appendChild(gridCtl);
    root.appendChild(controls);

    // ---- banner
    const banner = el("div", "pe-banner ok");
    root.appendChild(banner);

    // ---- plots
    const plots = el("div", "pe-plots");

    const fieldBox = el("div", "pe-panelbox");
    fieldBox.appendChild(el("h4", null, "Brightness |E|² (scaled to its own peak)"));
    fieldBox.appendChild(el("p", "pe-hint",
      "The beam seen face-on after travelling distance <b>z</b>, as a camera there would "
      + "see it. Bright means more light arriving. Every frame is rescaled to its own "
      + "brightest point, so this shows the <em>shape</em> of the pattern, not how much "
      + "light is left."));
    const fieldWrap = el("div", "pe-field-wrap");
    const fieldCanvas = el("canvas", "pe-field");
    const colorbar = el("div", "pe-colorbar");
    fieldWrap.appendChild(fieldCanvas);
    fieldWrap.appendChild(colorbar);
    fieldBox.appendChild(fieldWrap);
    fieldBox.appendChild(el("div", "pe-axis", "x, y in mm (window fixed)"));

    const crossBox = el("div", "pe-panelbox");
    crossBox.appendChild(el("h4", null, "Horizontal cross-section (mid-row)"));
    crossBox.appendChild(el("p", "pe-hint",
      "One straight line of pixels taken across the middle of the panel on the left, "
      + "plotted as a graph. The panel shows you the pattern; this shows you its "
      + "numbers, so the faint outer rings are actually readable."));
    const crossCanvas = el("canvas", "pe-cross");
    crossBox.appendChild(crossCanvas);
    crossBox.appendChild(el("div", "pe-axis", "x in mm"));

    plots.appendChild(fieldBox);
    plots.appendChild(crossBox);
    root.appendChild(plots);

    // ---- meta readout
    const meta = el("div", "pe-meta");
    root.appendChild(meta);

    // offscreen buffer for the n x n image
    const off = document.createElement("canvas");
    const offCtx = off.getContext("2d");

    function renderField(intensity, n) {
      let peak = 0;
      for (let i = 0; i < intensity.length; i++) if (intensity[i] > peak) peak = intensity[i];
      const inv = peak > 0 ? 1 / peak : 0;
      off.width = n; off.height = n;
      const img = offCtx.createImageData(n, n);
      const d = img.data;
      for (let i = 0; i < n * n; i++) {
        let v = intensity[i] * inv;
        if (v < 0) v = 0; else if (v > 1) v = 1;
        const li = (v * 255) | 0;
        d[i * 4] = LUT[li * 3];
        d[i * 4 + 1] = LUT[li * 3 + 1];
        d[i * 4 + 2] = LUT[li * 3 + 2];
        d[i * 4 + 3] = 255;
      }
      offCtx.putImageData(img, 0, 0);

      const dpr = canvasScale();
      const W = 320;
      fieldCanvas.width = W * dpr; fieldCanvas.height = W * dpr;
      const ctx = fieldCanvas.getContext("2d");
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.clearRect(0, 0, fieldCanvas.width, fieldCanvas.height);
      ctx.drawImage(off, 0, 0, fieldCanvas.width, fieldCanvas.height);
      return peak;
    }

    function renderCross(intensity, n, dx) {
      const dpr = canvasScale();
      const W = 320, H = 150;
      crossCanvas.width = W * dpr; crossCanvas.height = H * dpr;
      const ctx = crossCanvas.getContext("2d");
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, W, H);

      const mid = (n >> 1) * n;
      let peak = 0;
      for (let ix = 0; ix < n; ix++) { const v = intensity[mid + ix]; if (v > peak) peak = v; }
      const inv = peak > 0 ? 1 / peak : 0;

      const style = getComputedStyle(root);
      const accent = style.getPropertyValue("--pe-accent").trim() || "#3b6ea5";
      const border = style.getPropertyValue("--pe-border").trim() || "#ccc";
      const pad = 6, baseY = H - 16;

      // axis baseline
      ctx.strokeStyle = border; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pad, baseY); ctx.lineTo(W - pad, baseY); ctx.stroke();

      // curve
      ctx.strokeStyle = accent; ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (let ix = 0; ix < n; ix++) {
        const x = pad + (ix / (n - 1)) * (W - 2 * pad);
        const y = baseY - intensity[mid + ix] * inv * (baseY - pad);
        if (ix === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // x tick labels (min / 0 / max in mm)
      const half = (n >> 1) * dx * 1e3;
      ctx.fillStyle = style.getPropertyValue("--pe-muted").trim() || "#888";
      ctx.font = "10px system-ui, sans-serif";
      ctx.textBaseline = "top";
      ctx.textAlign = "left"; ctx.fillText((-half).toFixed(1), pad, baseY + 3);
      ctx.textAlign = "center"; ctx.fillText("0", W / 2, baseY + 3);
      ctx.textAlign = "right"; ctx.fillText(half.toFixed(1), W - pad, baseY + 3);
    }

    function compute() {
      const n = state.grid;
      const dx = state.extent / n;
      const lam = state.wavelength;
      const z = state.distance;

      const t0 = performance.now();
      const I = window.ASM.propagateIntensity(n, dx, lam, z, state.shape, state.apertureSize);
      renderField(I, n);
      renderCross(I, n, dx);
      const ms = performance.now() - t0;

      const zc = window.ASM.zCrit(n, dx, lam);
      const zf = window.ASM.zFraunhofer(n, dx, lam);
      const ok = Math.abs(z) <= zc;

      banner.className = "pe-banner " + (ok ? "ok" : "warn");
      banner.innerHTML = ok
        ? `✓ Sampling OK <small>|z| = ${(z * 1e3).toFixed(0)} mm ≤ z_crit = ${(zc * 1e3).toFixed(0)} mm · far field beyond ${(zf * 1e2).toFixed(1)} cm</small>`
        : `⚠ Aliasing risk <small>|z| = ${(z * 1e3).toFixed(0)} mm &gt; z_crit = ${(zc * 1e3).toFixed(0)} mm. The steepest light is being discarded to keep the result honest</small>`;

      meta.innerHTML =
        `<div>λ <b>${(lam * 1e9).toFixed(0)} nm</b></div>` +
        `<div>Aperture ⌀ <b>${(state.apertureSize * 1e3).toFixed(2)} mm</b></div>` +
        `<div>z <b>${(z * 1e3).toFixed(0)} mm</b></div>` +
        `<div>Grid <b>${n}×${n}</b></div>` +
        `<div>dx <b>${(dx * 1e6).toFixed(1)} µm</b></div>` +
        `<div>Window <b>${(state.extent * 1e3).toFixed(1)} mm</b></div>` +
        `<div>z_crit <b>${(zc * 1e3).toFixed(0)} mm</b></div>` +
        `<div>compute <b>${ms.toFixed(1)} ms</b></div>`;
    }

    // rAF-coalesced recompute (one per frame max, keeps 256^2 dragging smooth)
    let scheduled = false;
    function requestCompute() {
      if (scheduled) return;
      scheduled = true;
      requestAnimationFrame(() => { scheduled = false; compute(); });
    }

    function updateValueLabels() {
      document.getElementById("pe-size-v").textContent = (state.apertureSize * 1e3).toFixed(2) + " mm";
      document.getElementById("pe-dist-v").textContent = (state.distance * 1e3).toFixed(0) + " mm";
      document.getElementById("pe-wl-v").textContent = (state.wavelength * 1e9).toFixed(0) + " nm";
    }

    function setShape(sh) {
      state.shape = sh;
      btnCirc.setAttribute("aria-pressed", String(sh === "circular"));
      btnSq.setAttribute("aria-pressed", String(sh === "square"));
      requestCompute();
    }
    function setGrid(g) {
      state.grid = g;
      gbtns.forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.g === g)));
      requestCompute();
    }

    // wiring
    sizeIn.addEventListener("input", () => { state.apertureSize = +sizeIn.value * 1e-3; updateValueLabels(); requestCompute(); });
    distIn.addEventListener("input", () => { state.distance = +distIn.value * 1e-3; updateValueLabels(); requestCompute(); });
    wlIn.addEventListener("input", () => { state.wavelength = +wlIn.value * 1e-9; updateValueLabels(); requestCompute(); });
    btnCirc.addEventListener("click", () => setShape("circular"));
    btnSq.addEventListener("click", () => setShape("square"));
    gbtns.forEach((b) => b.addEventListener("click", () => setGrid(+b.dataset.g)));

    // init
    setShape(state.shape);
    setGrid(state.grid);
    updateValueLabels();
    compute();

    return { recompute: compute, state };
  }

  if (typeof window !== "undefined") window.PhotonnExplorer = { mount };
})();
