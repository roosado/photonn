/*
 * d2nn_demo.js -- the trained diffractive network, as an interactive widget.
 *
 * Pick a frozen MNIST test digit or draw one, and the Phase-2 D2NN classifies it
 * live: the digit is encoded into the entrance field, propagated through the five
 * trained phase masks, and read out by integrating intensity over ten detector
 * boxes. Every panel is the actual computed field -- nothing is precomputed, and
 * nothing is fetched.
 *
 * Usage:  window.PhotonnD2NN.mount(containerElement, opts)
 * opts (all optional): { gallery: <index to select first> }
 *
 * Depends on window.PhotonnD2NN_Net (d2nn.js), which depends on window.ASM
 * (asm.js) and window.D2NN_WEIGHTS (d2nn_weights.js). No libraries, no network.
 *
 * The forward pass is ~12 ms at 128x128, so recompute is rAF-debounced and runs
 * while you draw.
 */
(function () {
  "use strict";

  const NET = (typeof window !== "undefined" && window.PhotonnD2NN_Net)
    ? window.PhotonnD2NN_Net
    : (typeof require !== "undefined" ? require("./d2nn.js") : null);

  // Same inferno LUT the diffraction explorer uses, so both widgets speak one
  // visual language for optical intensity.
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

  // Detector-plane intensity spans several decades; a sqrt display stretch makes
  // the ten regions legible. Stated in the caption -- the numbers are untouched.
  const DISPLAY_GAMMA = 0.5;

  const STYLE_ID = "pd-style";
  const CSS = `
.pd-root{--pe-fg:#1b1f24;--pe-muted:#5a6472;--pe-panel:#f4f6f9;--pe-border:#d7dde5;
  --pe-accent:#3b6ea5;--pe-ok:#3f8f4e;--pe-warn:#c14a3d;
  color:var(--pe-fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  display:flex;flex-direction:column;gap:14px;max-width:820px;}
@media (prefers-color-scheme:dark){.pd-root{--pe-fg:#e6eaf0;--pe-muted:#9aa6b5;
  --pe-panel:#1c2128;--pe-border:#30363d;--pe-accent:#6ea8e0;--pe-ok:#5cc06e;--pe-warn:#e0705f;}}
.pd-box{background:var(--pe-panel);border:1px solid var(--pe-border);border-radius:10px;padding:12px 14px;}
.pd-box h4{margin:0 0 8px;font-size:12px;color:var(--pe-muted);font-weight:600;
  text-transform:uppercase;letter-spacing:.04em;}
.pd-seg{display:flex;border:1px solid var(--pe-border);border-radius:7px;overflow:hidden;
  align-self:flex-start;}
.pd-seg button{border:0;background:transparent;color:var(--pe-muted);padding:7px 14px;
  font:inherit;font-size:12px;font-weight:600;cursor:pointer;}
.pd-seg button[aria-pressed=true]{background:var(--pe-accent);color:#fff;}
.pd-src{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start;}
.pd-gallery{display:grid;grid-template-columns:repeat(8,1fr);gap:6px;flex:1 1 300px;min-width:260px;}
.pd-gallery canvas{width:100%;aspect-ratio:1;display:block;background:#000;border-radius:4px;
  cursor:pointer;border:2px solid transparent;image-rendering:pixelated;}
.pd-gallery canvas[aria-selected=true]{border-color:var(--pe-accent);}
.pd-draw{display:flex;flex-direction:column;gap:8px;}
.pd-pad{width:196px;height:196px;background:#000;border-radius:8px;display:block;
  border:1px solid var(--pe-border);cursor:crosshair;touch-action:none;}
.pd-btn{border:1px solid var(--pe-border);background:transparent;color:var(--pe-fg);
  border-radius:7px;padding:6px 12px;font:inherit;font-size:12px;cursor:pointer;}
.pd-btn:hover{border-color:var(--pe-accent);color:var(--pe-accent);}
.pd-hint{font-size:12px;color:var(--pe-muted);max-width:24ch;margin:0;}
.pd-result{display:flex;gap:16px;flex-wrap:wrap;align-items:stretch;}
.pd-pred{display:flex;flex-direction:column;align-items:center;justify-content:center;
  min-width:140px;gap:2px;}
.pd-pred .big{font-size:64px;line-height:1;font-weight:700;color:var(--pe-accent);
  font-variant-numeric:tabular-nums;}
.pd-pred .verdict{font-size:12px;font-weight:600;}
.pd-pred .verdict.ok{color:var(--pe-ok);}
.pd-pred .verdict.bad{color:var(--pe-warn);}
.pd-pred .sub{font-size:12px;color:var(--pe-muted);}
.pd-bars{flex:1 1 320px;display:flex;align-items:flex-end;gap:5px;height:132px;}
.pd-bar{flex:1;display:flex;flex-direction:column;align-items:center;gap:4px;height:100%;
  justify-content:flex-end;}
.pd-bar .fill{width:100%;background:var(--pe-border);border-radius:3px 3px 0 0;min-height:2px;
  transition:height .12s ease-out;}
.pd-bar.win .fill{background:var(--pe-accent);}
.pd-bar .k{font-size:11px;color:var(--pe-muted);font-variant-numeric:tabular-nums;}
.pd-bar.win .k{color:var(--pe-accent);font-weight:700;}
.pd-optics{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start;}
.pd-plane{display:flex;flex-direction:column;gap:5px;align-items:center;}
.pd-plane canvas{display:block;background:#000;border-radius:6px;image-rendering:auto;}
.pd-plane .lab{font-size:11px;color:var(--pe-muted);text-align:center;}
.pd-film{display:flex;gap:8px;flex-wrap:wrap;}
.pd-note{font-size:12px;color:var(--pe-muted);margin:0;}
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

  /** Peak-normalised inferno render of an n x n map onto a canvas of `size` CSS px. */
  function renderMap(canvas, data, n, size, gamma) {
    let peak = 0;
    for (let i = 0; i < data.length; i++) if (data[i] > peak) peak = data[i];
    const inv = peak > 0 ? 1 / peak : 0;
    const g = gamma || 1;

    const off = document.createElement("canvas");
    off.width = n; off.height = n;
    const octx = off.getContext("2d");
    const img = octx.createImageData(n, n);
    const d = img.data;
    for (let i = 0; i < n * n; i++) {
      let v = data[i] * inv;
      if (v < 0) v = 0; else if (v > 1) v = 1;
      if (g !== 1) v = Math.pow(v, g);
      const li = (v * 255) | 0;
      d[i * 4] = LUT[li * 3];
      d[i * 4 + 1] = LUT[li * 3 + 1];
      d[i * 4 + 2] = LUT[li * 3 + 2];
      d[i * 4 + 3] = 255;
    }
    octx.putImageData(img, 0, 0);

    const dpr = window.devicePixelRatio || 1;
    canvas.style.width = size + "px";
    canvas.style.height = size + "px";
    canvas.width = Math.round(size * dpr);
    canvas.height = Math.round(size * dpr);
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
    return ctx;
  }

  function mount(container, opts) {
    opts = opts || {};
    injectStyle();

    const N = NET.N;
    const DIGIT = NET.DIGIT;
    const regions = NET.weights.regions;

    const state = {
      mode: "gallery",
      gallery: Math.min(opts.gallery || 0, NET.nGallery - 1),
      drawing: false,
    };

    const root = el("div", "pe-root pd-root");
    container.innerHTML = "";
    container.appendChild(root);

    // ---------------------------------------------------------- input source
    const seg = el("div", "pd-seg");
    const btnGallery = el("button", null, "Test digits");
    const btnDraw = el("button", null, "Draw your own");
    seg.appendChild(btnGallery); seg.appendChild(btnDraw);
    root.appendChild(seg);

    const srcBox = el("div", "pd-box");
    const srcHead = el("h4", null, "Input digit");
    srcBox.appendChild(srcHead);
    const src = el("div", "pd-src");
    srcBox.appendChild(src);
    root.appendChild(srcBox);

    // gallery of frozen test digits
    const gallery = el("div", "pd-gallery");
    const thumbs = [];
    for (let k = 0; k < NET.nGallery; k++) {
      const c = el("canvas");
      c.width = DIGIT; c.height = DIGIT;
      c.title = "MNIST test digit, true label " + NET.galleryLabel(k);
      const ctx = c.getContext("2d");
      const img = ctx.createImageData(DIGIT, DIGIT);
      const digit = NET.galleryDigit(k);
      for (let i = 0; i < DIGIT * DIGIT; i++) {
        const v = (digit[i] * 255) | 0;
        img.data[i * 4] = v; img.data[i * 4 + 1] = v; img.data[i * 4 + 2] = v;
        img.data[i * 4 + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
      c.addEventListener("click", () => { state.gallery = k; selectGallery(); requestCompute(); });
      gallery.appendChild(c);
      thumbs.push(c);
    }
    src.appendChild(gallery);

    // paint pad
    const drawWrap = el("div", "pd-draw");
    const PAD = 196;                       // CSS px; the drawing buffer matches 1:1
    const pad = el("canvas", "pd-pad");
    pad.width = PAD; pad.height = PAD;
    const padCtx = pad.getContext("2d", { willReadFrequently: true });
    const clearBtn = el("button", "pd-btn", "Clear");
    drawWrap.appendChild(pad);
    drawWrap.appendChild(clearBtn);
    drawWrap.appendChild(el("p", "pd-hint",
      "Draw a digit. It is re-centred and scaled the way MNIST was built, then sent "
      + "through the optics."));
    src.appendChild(drawWrap);

    function clearPad() {
      padCtx.fillStyle = "#000";
      padCtx.fillRect(0, 0, PAD, PAD);
    }
    clearPad();

    // ------------------------------------------------------------- prediction
    const resBox = el("div", "pd-box");
    resBox.appendChild(el("h4", null, "Detector readout — share of output power per class"));
    const result = el("div", "pd-result");
    const pred = el("div", "pd-pred");
    const predBig = el("div", "big", "–");
    const predVerdict = el("div", "verdict", "");
    const predSub = el("div", "sub", "");
    pred.appendChild(predBig); pred.appendChild(predVerdict); pred.appendChild(predSub);
    const bars = el("div", "pd-bars");
    const barEls = [];
    for (let c = 0; c < 10; c++) {
      const b = el("div", "pd-bar");
      const fill = el("div", "fill");
      const k = el("div", "k", String(c));
      b.appendChild(fill); b.appendChild(k);
      bars.appendChild(b);
      barEls.push({ b, fill });
    }
    result.appendChild(pred); result.appendChild(bars);
    resBox.appendChild(result);
    root.appendChild(resBox);

    // ----------------------------------------------------------- optical path
    const opticsBox = el("div", "pd-box");
    opticsBox.appendChild(el("h4", null, "The light, plane by plane"));
    const optics = el("div", "pd-optics");

    function planePanel(label, size) {
      const wrap = el("div", "pd-plane");
      const c = el("canvas");
      wrap.appendChild(c);
      wrap.appendChild(el("div", "lab", label));
      return { wrap, canvas: c, size };
    }

    const inputPanel = planePanel("Entrance field", 132);
    optics.appendChild(inputPanel.wrap);

    const film = el("div", "pd-film");
    const filmPanels = [];
    for (let L = 0; L < NET.weights.n_layers; L++) {
      const p = planePanel("at mask " + (L + 1), 76);
      film.appendChild(p.wrap);
      filmPanels.push(p);
    }
    const filmWrap = el("div", "pd-plane");
    filmWrap.appendChild(film);
    filmWrap.appendChild(el("div", "lab", "intensity arriving at each phase mask"));
    optics.appendChild(filmWrap);

    const detPanel = planePanel("Detector plane — ten regions", 172);
    optics.appendChild(detPanel.wrap);

    opticsBox.appendChild(optics);
    opticsBox.appendChild(el("p", "pd-note",
      "Square-root intensity scale for legibility; the readout numbers are untouched."));
    root.appendChild(opticsBox);

    const note = el("p", "pd-note", "");
    root.appendChild(note);

    // -------------------------------------------------------------- rendering
    function drawRegions(ctx, canvas, winner) {
      const s = canvas.width / N;
      ctx.lineWidth = Math.max(1, 1.25 * (window.devicePixelRatio || 1));
      ctx.font = `${11 * (window.devicePixelRatio || 1)}px system-ui, sans-serif`;
      ctx.textBaseline = "bottom";
      for (let c = 0; c < regions.length; c++) {
        const r = regions[c];                       // [y0, y1, x0, x1]
        const x = r[2] * s, y = r[0] * s;
        const w = (r[3] - r[2]) * s, h = (r[1] - r[0]) * s;
        const win = c === winner;
        ctx.strokeStyle = win ? "#ffffff" : "rgba(255,255,255,0.42)";
        ctx.lineWidth = (win ? 2.0 : 1.0) * (window.devicePixelRatio || 1);
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = win ? "#ffffff" : "rgba(255,255,255,0.55)";
        ctx.fillText(String(c), x + 1.5, y - 1.5);
      }
    }

    function render(res, trueLabel) {
      // entrance field: the input-plane transmission the optics actually sees
      renderMap(inputPanel.canvas, res.canvas, N, inputPanel.size, 1);
      for (let L = 0; L < filmPanels.length; L++) {
        renderMap(filmPanels[L].canvas, res.planes[L], N, filmPanels[L].size, DISPLAY_GAMMA);
      }
      const dctx = renderMap(detPanel.canvas, res.intensity, N, detPanel.size, DISPLAY_GAMMA);
      drawRegions(dctx, detPanel.canvas, res.pred);

      let top = 0;
      for (let c = 0; c < 10; c++) if (res.fractions[c] > res.fractions[top]) top = c;
      const scale = res.fractions[top] > 0 ? res.fractions[top] : 1;
      for (let c = 0; c < 10; c++) {
        const frac = res.fractions[c] / scale;
        barEls[c].fill.style.height = Math.max(2, frac * 96) + "px";
        barEls[c].b.classList.toggle("win", c === res.pred);
      }

      predBig.textContent = String(res.pred);
      predSub.textContent = (res.fractions[res.pred] * 100).toFixed(1) + "% of output power";
      if (trueLabel == null) {
        predVerdict.textContent = "";
        predVerdict.className = "verdict";
      } else if (trueLabel === res.pred) {
        predVerdict.textContent = "✓ correct — true label " + trueLabel;
        predVerdict.className = "verdict ok";
      } else {
        predVerdict.textContent = "✗ wrong — true label " + trueLabel;
        predVerdict.className = "verdict bad";
      }
    }

    function compute() {
      let digit, trueLabel = null;
      if (state.mode === "gallery") {
        digit = NET.galleryDigit(state.gallery);
        trueLabel = NET.galleryLabel(state.gallery);
      } else {
        const px = padCtx.getImageData(0, 0, PAD, PAD).data;
        const gray = new Float64Array(PAD * PAD);
        for (let i = 0; i < PAD * PAD; i++) gray[i] = px[i * 4] / 255;
        digit = NET.normalizeDrawn(gray, PAD);
      }
      const t0 = (typeof performance !== "undefined") ? performance.now() : 0;
      const res = NET.classify(digit);
      const ms = (typeof performance !== "undefined") ? performance.now() - t0 : 0;
      render(res, trueLabel);
      note.textContent =
        `${NET.weights.n_layers} phase masks, ${NET.weights.n_layers + 1} propagations of `
        + `${(NET.weights.separation * 1e3).toFixed(0)} mm at `
        + `${(NET.weights.wavelength * 1e9).toFixed(0)} nm, ${N}×${N} grid — `
        + `computed in your browser in ${ms.toFixed(0)} ms.`;
    }

    let scheduled = false;
    function requestCompute() {
      if (scheduled) return;
      scheduled = true;
      requestAnimationFrame(() => { scheduled = false; compute(); });
    }

    // ---------------------------------------------------------------- events
    function selectGallery() {
      thumbs.forEach((t, k) => t.setAttribute("aria-selected", String(k === state.gallery)));
    }

    function setMode(mode) {
      state.mode = mode;
      btnGallery.setAttribute("aria-pressed", String(mode === "gallery"));
      btnDraw.setAttribute("aria-pressed", String(mode === "draw"));
      gallery.style.display = mode === "gallery" ? "" : "none";
      drawWrap.style.display = mode === "draw" ? "" : "none";
      srcHead.textContent = mode === "gallery"
        ? "Input digit — frozen MNIST test set"
        : "Input digit — your drawing";
      requestCompute();
    }

    btnGallery.addEventListener("click", () => setMode("gallery"));
    btnDraw.addEventListener("click", () => setMode("draw"));
    clearBtn.addEventListener("click", () => { clearPad(); requestCompute(); });

    function padPos(ev) {
      const r = pad.getBoundingClientRect();
      return [(ev.clientX - r.left) * (PAD / r.width), (ev.clientY - r.top) * (PAD / r.height)];
    }
    function strokeTo(x, y) {
      padCtx.lineTo(x, y);
      padCtx.stroke();
      padCtx.beginPath();
      padCtx.moveTo(x, y);
    }
    pad.addEventListener("pointerdown", (ev) => {
      state.drawing = true;
      pad.setPointerCapture(ev.pointerId);
      // Stroke width is ~10% of the 20 px normalisation box, matching MNIST's pen.
      padCtx.strokeStyle = "#fff";
      padCtx.lineWidth = 16;
      padCtx.lineCap = "round";
      padCtx.lineJoin = "round";
      const p = padPos(ev);
      padCtx.beginPath();
      padCtx.moveTo(p[0], p[1]);
      strokeTo(p[0], p[1]);
      requestCompute();
      ev.preventDefault();
    });
    pad.addEventListener("pointermove", (ev) => {
      if (!state.drawing) return;
      const p = padPos(ev);
      strokeTo(p[0], p[1]);
      requestCompute();
      ev.preventDefault();
    });
    const endStroke = () => { state.drawing = false; requestCompute(); };
    pad.addEventListener("pointerup", endStroke);
    pad.addEventListener("pointercancel", endStroke);
    pad.addEventListener("pointerleave", () => { if (state.drawing) state.drawing = false; });

    selectGallery();
    setMode("gallery");
  }

  const API = { mount };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  if (typeof window !== "undefined") window.PhotonnD2NN = API;
})();
