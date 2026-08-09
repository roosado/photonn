/*
 * d2nn_compare.js -- the same digit through several trained machines.
 *
 * The optics sweep found that a fixed reach budget spent on more masks beats the
 * same budget spent on distance. This runs the models on one digit so the
 * difference can be watched rather than read: pick a digit or draw one, and each
 * column shows that network's detector plane, where it thinks the light landed,
 * and how much of the output power it put in the winning box.
 *
 * The interesting panel is the detector plane. The shipped 5-mask network throws
 * a diffuse interference pattern across the whole plane and reads a weak maximum
 * off it; a deeper candidate gathers light into the detector boxes themselves.
 * Same physics, same reach, more masks.
 *
 * Two structural rules hold this widget together:
 *
 *   1. **One input, N models.** The gallery and the draw pad live in
 *      digit_source.js, not here, so every column is fed the identical digit by
 *      construction rather than by care.
 *   1a. **The input decides its own cadence.** A board with a deep column cannot
 *      classify inside a frame, so following a stroke live would block the main
 *      thread and make the pad itself unusable. The board measures what it costs
 *      and tells digit_source.js to hold emissions until the pen pauses; the
 *      columns dim while that is true. Nothing here picks a delay from a mask
 *      count -- it is measured, so it adapts to the machine.
 *   2. **No model metadata in this file.** Every caption -- the column title,
 *      the accuracy, what it was scored on, the unshipped caveat, even the
 *      orange border -- is rendered from the bundle's own `provenance` block.
 *      Promoting a model is regenerating a bundle; this widget does not change.
 *      There is deliberately no accuracy literal anywhere below, and a test
 *      enforces that.
 *
 * Models are built by d2nn.js:buildNet from their own weight bundles, and each
 * is cross-checked against torch (< 1e-3 on logits) by tests/.
 *
 * Usage:  window.PhotonnD2NNCompare.mount(containerElement, opts)
 * opts (all optional):
 *   models  -- array of weight bundles, in column order. Defaults to the shipped
 *              bundle plus the deep candidate.
 *   gallery -- index of the digit selected on mount
 */
(function () {
  "use strict";

  const NET = (typeof window !== "undefined" && window.PhotonnD2NN_Net)
    ? window.PhotonnD2NN_Net
    : (typeof require !== "undefined" ? require("./d2nn.js") : null);
  const SOURCE = (typeof window !== "undefined" && window.PhotonnDigitSource)
    ? window.PhotonnDigitSource
    : (typeof require !== "undefined" ? require("./digit_source.js") : null);
  const SHIPPED_W = (typeof window !== "undefined" && window.D2NN_WEIGHTS)
    ? window.D2NN_WEIGHTS
    : (typeof require !== "undefined" ? require("./d2nn_weights.js") : null);
  const DEEP_W = (typeof window !== "undefined" && window.D2NN_DEEP_WEIGHTS)
    ? window.D2NN_DEEP_WEIGHTS
    : (typeof require !== "undefined" ? require("./d2nn_deep_weights.js") : null);

  //: One animation frame. A board that cannot classify every model inside one
  //: cannot follow a stroke, however the emission is scheduled -- so this is the
  //: line between "update live" and "update when the pen pauses".
  const FRAME_MS = 16.7;
  //: Hysteresis around it, so a board sitting near the budget does not flip mode
  //: between strokes. Enter settle mode above ENTER, leave below LEAVE.
  const SETTLE_ENTER_MS = 20;
  const SETTLE_LEAVE_MS = 10;
  //: How long the pen must be still before an expensive board classifies. Short
  //: enough not to read as waiting, long enough to swallow a whole stroke: moves
  //: during a stroke are milliseconds apart, so this only ever fires on a pause.
  const SETTLE_MS = 250;

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
.dc-truth{font-size:13px;color:var(--pe-muted);margin:12px 0 0;}
.dc-truth b{color:var(--pe-fg);}
.dc-wait{font-size:12px;color:var(--dc-cand);margin:6px 0 0;min-height:1.2em;
  opacity:0;transition:opacity .12s;}
.dc-root.waiting .dc-wait{opacity:1;}
.dc-root.waiting .dc-col .big{opacity:.4;}
.dc-col .big{transition:opacity .12s;}
`;

  function injectStyle() {
    if (typeof document === "undefined" || document.getElementById(STYLE_ID)) return;
    const el = document.createElement("style");
    el.id = STYLE_ID;
    el.textContent = CSS;
    document.head.appendChild(el);
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

  // ------------------------------------------------------------- provenance
  // Everything below turns a bundle's provenance block into text. Nothing here
  // knows which model it is describing, and no number is written down.

  const fmtAcc = (x) => Number(x).toFixed(4);
  const fmtInt = (x) => Number(x).toLocaleString();

  /** "60,000 images, 40 epochs", or null if the bundle did not record a protocol. */
  function protocolPhrase(prov) {
    const p = prov.protocol;
    if (!p || p.n_train == null || p.epochs == null) return null;
    return `${fmtInt(p.n_train)} images, ${p.epochs} epochs`;
  }

  /**
   * A gap in mm, at whatever precision keeps it a number.
   *
   * Deep stacks spend their reach budget on masks rather than distance, so the
   * gap can be well under a millimetre -- 0.5263 mm rounds to "1 mm" at zero
   * decimals, which is both wrong and the opposite of the point being made.
   */
  function fmtGap(metres) {
    const mm = metres * 1e3;
    if (mm < 1) return mm.toFixed(2);
    return mm.toFixed(Number.isInteger(mm) ? 0 : 1);
  }

  /** "5 masks · 3 mm gaps · 82k phases" -- read off the weights, not the prose. */
  function geomLine(net) {
    const w = net.weights;
    return `${w.n_layers} masks · ${fmtGap(w.separation)} mm gaps · `
      + `${(w.n_layers * net.N * net.N / 1000).toFixed(0)}k phases`;
  }

  /**
   * The caption under a column's verdict.
   *
   * A model a visitor can operate invites its number being read as an accuracy,
   * so an unshipped model states what its number is not. `reference` is the
   * shipped model in the same board, if there is one -- that is where the number
   * it is or is not comparable to comes from, rather than from a literal.
   *
   * Whether the two numbers *are* comparable is a property of the bundles, not
   * of shipping: a sweep candidate ranked on a validation split is not
   * comparable to the headline, while a candidate scored on the same frozen test
   * set under the same protocol is exactly comparable and saying otherwise would
   * throw away the comparison the board exists to make. The rule is that the
   * measurements match and neither bundle declares itself measured elsewhere.
   */
  function comparable(prov, reference) {
    return !!(reference && reference.accuracy != null
      && !prov.not_scored_on && !reference.not_scored_on
      && prov.scored_on && prov.scored_on === reference.scored_on);
  }

  function noteFor(prov, reference) {
    if (!prov || prov.accuracy == null) return "";
    const run = protocolPhrase(prov);
    const on = prov.scored_on ? ` on ${prov.scored_on}` : "";

    if (prov.shipped === false) {
      let s = `<b>Not shipped.</b> ${fmtAcc(prov.accuracy)}${on}`;
      if (run) s += ` (${run})`;
      if (prov.caveat) s += ` &mdash; ${prov.caveat}`;
      s += ".";
      if (comparable(prov, reference)) {
        s += ` Same measurement as the shipped ${fmtAcc(reference.accuracy)}.`;
      } else if (reference && reference.accuracy != null) {
        s += ` Not comparable to ${fmtAcc(reference.accuracy)}.`;
      }
      return s;
    }
    let s = `Accuracy <b style="color:inherit">${fmtAcc(prov.accuracy)}</b>${on}`;
    if (run) s += ` (${run})`;
    return s + ".";
  }

  function mount(container, opts) {
    opts = opts || {};
    if (!NET || !SOURCE) {
      throw new Error("d2nn_compare.js needs d2nn.js and digit_source.js");
    }
    injectStyle();

    const bundles = opts.models || [SHIPPED_W, DEEP_W];
    if (!bundles.length || bundles.some((w) => !w)) {
      throw new Error("d2nn_compare.js needs at least one weights bundle");
    }

    const models = bundles.map((W, i) => {
      // The default export is already built around the shipped bundle; reusing
      // it saves rebuilding a megabyte of precomputed cos/sin for no reason.
      const net = (W === NET.weights) ? NET : NET.buildNet(W);
      const prov = W.provenance || {};
      return {
        net, prov,
        name: prov.label || `Model ${i + 1}`,
        cls: prov.shipped === false ? "cand" : "ship",
      };
    });
    // "Not comparable to X" needs an X: the shipped model on this board, if any.
    const reference = (models.find((m) => m.prov.shipped) || {}).prov || null;
    const lead = models[0].net;

    const root = document.createElement("div");
    root.className = "pe-root dc-root";
    root.innerHTML = `
      <div class="dc-box">
        <h4>Pick a digit</h4>
        <p class="sub">${lead.nGallery} from the frozen MNIST test set, deliberately stocked
        with ones the shipped network gets <em>wrong</em> &mdash; those are the interesting
        ones. Or draw your own and watch both machines follow the stroke.</p>
        <div class="dc-source"></div>
        <p class="dc-truth"></p>
        <p class="dc-wait">Still drawing &mdash; the machines run when you pause.</p>
      </div>
      <div class="dc-cols"></div>`;
    container.appendChild(root);

    const truth = root.querySelector(".dc-truth");
    const cols = root.querySelector(".dc-cols");

    for (const m of models) {
      const el = document.createElement("div");
      el.className = "dc-col " + m.cls;
      el.innerHTML = `
        <div><span class="name">${m.name}</span><br><span class="geom">${geomLine(m.net)}</span></div>
        <canvas class="plane"></canvas>
        <div class="dc-verdict"><span class="big"></span><span class="share"></span></div>
        <p class="dc-note">${noteFor(m.prov, m.prov === reference ? null : reference)}</p>`;
      cols.appendChild(el);
      m.el = el;
      m.plane = el.querySelector("canvas.plane");
      m.big = el.querySelector(".big");
      m.share = el.querySelector(".share");
    }

    /**
     * How long a full board render costs, smoothed over renders.
     *
     * The board cannot know this before it has run: it depends on the models on
     * it and on the machine it is running on, and a 56-mask column costs roughly
     * eight times a 5-mask one on the same laptop. So it is measured here rather
     * than guessed from mask counts, and the digit source is told what to do with
     * it. An exponential average rides out one slow frame; the hysteresis keeps a
     * board near the budget from flipping mode between strokes.
     */
    let source = null;
    let cost = null;
    function updateCadence(elapsed) {
      cost = cost == null ? elapsed : cost * 0.7 + elapsed * 0.3;
      if (!source) return;
      if (cost > SETTLE_ENTER_MS) source.setSettle(SETTLE_MS);
      else if (cost < SETTLE_LEAVE_MS) source.setSettle(0);
    }

    /** Run one digit through every column. `label` is null for a drawing. */
    function render(digit, meta) {
      const t0 = (typeof performance !== "undefined") ? performance.now() : 0;
      const label = meta ? meta.label : null;
      truth.innerHTML = label == null
        ? "Your drawing &mdash; no ground truth to score against."
        : `True label: <b>${label}</b>`;

      for (const m of models) {
        const res = m.net.classify(digit);
        drawPlane(m.plane, m.net, res);
        m.big.textContent = res.pred;
        m.big.className = "big " + (label == null ? "" : (res.pred === label ? "ok" : "bad"));
        m.share.textContent =
          `${(res.fractions[res.pred] * 100).toFixed(1)}% of output power`
          + (label == null || res.pred === label
            ? ""
            : ` — true ${label} got ${(res.fractions[label] * 100).toFixed(1)}%`);
      }
      if (t0) updateCadence(performance.now() - t0);
    }

    source = SOURCE.mount(root.querySelector(".dc-source"), {
      net: lead,
      gallery: opts.gallery,
      // Start expensive boards in settle mode rather than discovering it on the
      // first laggy stroke: the mount render below measures the real cost and
      // corrects this either way, within one emission.
      settleMs: models.length > 1 ? SETTLE_MS : 0,
      hint: "Drawn digits are re-centred and scaled the way MNIST was built, then sent "
        + "through both machines.",
    });
    source.subscribe(render);
    // The columns go dim and the board says so while a stroke is still being
    // drawn, because two frozen digits with no explanation read as a hang.
    source.onPending((p) => root.classList.toggle("waiting", p));

    // `models` is exposed so a page can hang something else off a column's network
    // -- the 3D stage on the optics page draws the deep model from here rather
    // than calling buildNet again, which would duplicate ~15 MB of precomputed
    // cos/sin for a network already built two lines above.
    const api = {
      render: () => { const c = source.current(); if (c) render(c.digit, c.meta); },
      source, models,
      cost: () => cost,
    };
    container.__photonnCompare = api;   // handle for page scripts and browser tests
    return api;
  }

  // noteFor and geomLine are exported so the "captions come from provenance,
  // never from a literal" rule can be tested without a DOM -- there is no test
  // runner with jsdom here, and that rule is the point of the widget.
  const API = { mount, noteFor, geomLine, protocolPhrase, comparable };
  if (typeof window !== "undefined") window.PhotonnD2NNCompare = API;
  if (typeof module !== "undefined") module.exports = API;
})();
