/*
 * d2nn_compare.js -- the same digit through two trained machines.
 *
 * The optics sweep found that a fixed reach budget spent on more masks beats the
 * same budget spent on distance. This runs both models on one digit so the
 * difference can be watched rather than read: pick a digit, and each column
 * shows that network's detector plane, where it thinks the light landed, and how
 * much of the output power it put in the winning box.
 *
 * The interesting panel is the detector plane. The shipped 5-mask network throws
 * a diffuse interference pattern across the whole plane and reads a weak maximum
 * off it; the 14-mask candidate gathers light into the detector boxes themselves.
 * Same physics, same reach, more masks.
 *
 * Both networks are built by d2nn.js:buildNet from their own weight bundles, and
 * both are cross-checked against torch (< 1e-3 on logits). The candidate is NOT
 * a shipped model -- it comes from the ranking sweep, has never been scored on
 * the frozen test set, and had not converged. The widget prints that next to it
 * rather than leaving its number to be read as an accuracy.
 *
 * Usage:  window.PhotonnD2NNCompare.mount(containerElement, opts)
 * opts (all optional): { gallery: <initial digit index> }
 */
(function () {
  "use strict";

  const NET = (typeof window !== "undefined" && window.PhotonnD2NN_Net)
    ? window.PhotonnD2NN_Net
    : (typeof require !== "undefined" ? require("./d2nn.js") : null);
  const SWEEP_W = (typeof window !== "undefined" && window.D2NN_SWEEP_WEIGHTS)
    ? window.D2NN_SWEEP_WEIGHTS
    : (typeof require !== "undefined" ? require("./d2nn_sweep_weights.js") : null);

  const STYLE_ID = "dc-style";
  const CSS = `
.dc-root{--pe-fg:#1b1f24;--pe-muted:#5a6472;--pe-panel:#f4f6f9;--pe-border:#d7dde5;
  --pe-accent:#3b6ea5;--pe-ok:#3f8f4e;--pe-warn:#c14a3d;--dc-cand:#c9701f;
  color:var(--pe-fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  display:flex;flex-direction:column;gap:16px;}
@media (prefers-color-scheme:dark){.dc-root{--pe-fg:#e6eaf0;--pe-muted:#9aa6b5;
  --pe-panel:#1c2128;--pe-border:#30363d;--pe-accent:#6ea8e0;--pe-ok:#5cc06e;
  --pe-warn:#e0705f;--dc-cand:#f2994a;}}
.dc-box{background:var(--pe-panel);border:1px solid var(--pe-border);border-radius:10px;
  padding:14px 16px;}
.dc-box h4{margin:0 0 3px;font-size:12px;color:var(--pe-muted);font-weight:600;
  text-transform:uppercase;letter-spacing:.04em;}
.dc-box .sub{margin:0 0 12px;font-size:13px;color:var(--pe-muted);max-width:70ch;}
.dc-strip{display:flex;flex-wrap:wrap;gap:6px;margin:2px 0 0;}
.dc-strip canvas{width:44px;height:44px;border:1px solid var(--pe-border);border-radius:5px;
  cursor:pointer;background:#000;transition:border-color .12s,transform .12s;}
.dc-strip canvas:hover{border-color:var(--pe-accent);}
.dc-strip canvas[aria-selected=true]{border-color:var(--pe-accent);border-width:2px;
  transform:translateY(-2px);}
.dc-cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;}
.dc-col{background:var(--pe-panel);border:1px solid var(--pe-border);border-radius:10px;
  padding:13px 15px;display:flex;flex-direction:column;gap:9px;}
.dc-col.cand{border-color:color-mix(in srgb,var(--dc-cand) 45%,var(--pe-border));}
.dc-col .name{font:600 13px ui-monospace,SFMono-Regular,Menlo,monospace;}
.dc-col.ship .name{color:var(--pe-accent);}
.dc-col.cand .name{color:var(--dc-cand);}
.dc-col .geom{font-size:11.5px;color:var(--pe-muted);}
.dc-col canvas.plane{width:100%;image-rendering:pixelated;border-radius:6px;background:#000;
  border:1px solid var(--pe-border);}
.dc-verdict{display:flex;align-items:baseline;gap:9px;}
.dc-verdict .big{font:700 30px ui-monospace,SFMono-Regular,Menlo,monospace;line-height:1;}
.dc-verdict .ok{color:var(--pe-ok);}
.dc-verdict .bad{color:var(--pe-warn);}
.dc-verdict .share{font-size:12px;color:var(--pe-muted);}
.dc-note{font-size:11.5px;color:var(--pe-muted);border-top:1px solid var(--pe-border);
  padding-top:9px;margin:0;}
.dc-note b{color:var(--pe-warn);}
.dc-truth{font-size:13px;color:var(--pe-muted);margin:10px 0 0;}
.dc-truth b{color:var(--pe-fg);}
`;

  function injectStyle() {
    if (typeof document === "undefined" || document.getElementById(STYLE_ID)) return;
    const el = document.createElement("style");
    el.id = STYLE_ID;
    el.textContent = CSS;
    document.head.appendChild(el);
  }

  /** Draw a 28x28 digit into a small canvas. */
  function drawDigit(canvas, digit, size) {
    canvas.width = size; canvas.height = size;
    const ctx = canvas.getContext("2d");
    const im = ctx.createImageData(size, size);
    for (let i = 0; i < size * size; i++) {
      const v = Math.max(0, Math.min(1, digit[i])) * 255;
      im.data[i * 4] = v; im.data[i * 4 + 1] = v; im.data[i * 4 + 2] = v;
      im.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);
  }

  /**
   * Detector plane, gamma-stretched, with the ten readout boxes drawn on.
   *
   * Intensity spans decades, so a linear map shows a single hot pixel; the same
   * 0.5 gamma the 3D stage uses keeps the structure visible. The winning box is
   * outlined so "where the light landed" is literally what you see.
   */
  function drawPlane(canvas, net, res) {
    const n = net.N;
    canvas.width = n; canvas.height = n;
    const ctx = canvas.getContext("2d");
    const im = ctx.createImageData(n, n);

    let peak = 0;
    for (let i = 0; i < res.intensity.length; i++) {
      if (res.intensity[i] > peak) peak = res.intensity[i];
    }
    const inv = peak > 0 ? 1 / peak : 0;
    for (let i = 0; i < n * n; i++) {
      const t = Math.pow(res.intensity[i] * inv, 0.5);
      // magma-ish ramp: black -> violet -> orange -> white
      im.data[i * 4] = Math.min(255, 255 * Math.pow(t, 0.8));
      im.data[i * 4 + 1] = Math.min(255, 255 * Math.pow(Math.max(0, t - 0.25) / 0.75, 1.6));
      im.data[i * 4 + 2] = Math.min(255, 255 * (t < 0.5 ? t * 1.5 : Math.pow((t - 0.5) * 2, 2) * 0.9 + 0.1));
      im.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);

    const regions = net.weights.regions;
    for (let c = 0; c < regions.length; c++) {
      const [y0, y1, x0, x1] = regions[c];
      ctx.lineWidth = c === res.pred ? 1.6 : 0.6;
      ctx.strokeStyle = c === res.pred ? "#ffffff" : "rgba(255,255,255,0.32)";
      ctx.strokeRect(x0 + 0.5, y0 + 0.5, x1 - x0 - 1, y1 - y0 - 1);
    }
  }

  function mount(container, opts) {
    opts = opts || {};
    if (!NET || !SWEEP_W) throw new Error("d2nn_compare.js needs d2nn.js and d2nn_sweep_weights.js");
    injectStyle();

    const shipped = NET;
    const candidate = NET.buildNet(SWEEP_W);
    const prov = SWEEP_W.provenance || {};

    const models = [
      { net: shipped, cls: "ship", name: "Shipped",
        geom: `${shipped.weights.n_layers} masks · `
            + `${(shipped.weights.separation * 1e3).toFixed(0)} mm gaps · `
            + `${(shipped.weights.n_layers * shipped.N * shipped.N / 1000).toFixed(0)}k phases`,
        note: "Test accuracy <b style=\"color:inherit\">0.799</b> on the frozen 2,000-digit test set." },
      { net: candidate, cls: "cand", name: "Candidate",
        geom: `${candidate.weights.n_layers} masks · `
            + `${(candidate.weights.separation * 1e3).toFixed(0)} mm gaps · `
            + `${(candidate.weights.n_layers * candidate.N * candidate.N / 1000).toFixed(0)}k phases`,
        note: `<b>Not shipped.</b> ${(prov.val_acc || 0).toFixed(4)} on a held-out validation `
            + `split after ${prov.protocol ? prov.protocol.epochs : "?"} epochs on `
            + `${prov.protocol ? prov.protocol.n_train.toLocaleString() : "?"} images &mdash; a ranking `
            + `run, never scored on the test set, and not converged. Not comparable to 0.799.` },
    ];

    const root = document.createElement("div");
    root.className = "dc-root";
    root.innerHTML = `
      <div class="dc-box">
        <h4>Pick a digit</h4>
        <p class="sub">Sixteen from the frozen MNIST test set. Six of them the shipped
        network gets <em>wrong</em> &mdash; those are the interesting ones.</p>
        <div class="dc-strip"></div>
        <p class="dc-truth"></p>
      </div>
      <div class="dc-cols"></div>`;
    container.appendChild(root);

    const strip = root.querySelector(".dc-strip");
    const truth = root.querySelector(".dc-truth");
    const cols = root.querySelector(".dc-cols");

    for (const m of models) {
      const el = document.createElement("div");
      el.className = "dc-col " + m.cls;
      el.innerHTML = `
        <div><span class="name">${m.name}</span><br><span class="geom">${m.geom}</span></div>
        <canvas class="plane"></canvas>
        <div class="dc-verdict"><span class="big"></span><span class="share"></span></div>
        <p class="dc-note">${m.note}</p>`;
      cols.appendChild(el);
      m.el = el;
    }

    let selected = Math.min(opts.gallery || 0, shipped.nGallery - 1);

    function render() {
      const digit = shipped.galleryDigit(selected);
      const label = shipped.galleryLabel(selected);
      truth.innerHTML = `True label: <b>${label}</b>`;

      for (const m of models) {
        const res = m.net.classify(digit);
        drawPlane(m.el.querySelector("canvas.plane"), m.net, res);
        const big = m.el.querySelector(".big");
        big.textContent = res.pred;
        big.className = "big " + (res.pred === label ? "ok" : "bad");
        m.el.querySelector(".share").textContent =
          `${(res.fractions[res.pred] * 100).toFixed(1)}% of output power`
          + (res.pred === label ? "" : ` — true ${label} got ${(res.fractions[label] * 100).toFixed(1)}%`);
      }
    }

    for (let k = 0; k < shipped.nGallery; k++) {
      const c = document.createElement("canvas");
      c.title = "MNIST test digit, true label " + shipped.galleryLabel(k);
      drawDigit(c, shipped.galleryDigit(k), shipped.DIGIT);
      c.addEventListener("click", () => {
        selected = k;
        for (const other of strip.children) other.setAttribute("aria-selected", "false");
        c.setAttribute("aria-selected", "true");
        render();
      });
      c.setAttribute("aria-selected", String(k === selected));
      strip.appendChild(c);
    }

    render();
    return { render };
  }

  const API = { mount };
  if (typeof window !== "undefined") window.PhotonnD2NNCompare = API;
  if (typeof module !== "undefined") module.exports = API;
})();
