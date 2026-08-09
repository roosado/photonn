/*
 * digit_source.js -- where a digit comes from, independent of what reads it.
 *
 * The gallery of frozen MNIST test digits, the 196x196 draw pad, the mode
 * toggle and Clear, packaged as one mountable widget that emits a normalised
 * 28x28 digit to anyone who subscribes. Nothing here knows about optics: it
 * produces digits, and a network -- or two networks side by side -- consumes
 * them.
 *
 * That split is the point. The comparison board (d2nn_compare.js) runs several
 * trained models on *one* input, so the input cannot live inside any one
 * model's widget. One source, N consumers, no per-consumer input state.
 *
 * Usage:  window.PhotonnDigitSource.mount(container, opts) -> handle
 * opts (all optional):
 *   net      -- the network supplying gallery bytes and normalizeDrawn;
 *               defaults to window.PhotonnD2NN_Net. Only used for its
 *               *preprocessing*, which is model-independent in practice:
 *               normalizeDrawn reads gallery_size (28) and nothing else.
 *   gallery  -- index of the digit selected on mount (default 0)
 *   thumb    -- gallery thumbnail size in CSS px (default 44)
 *   mode     -- "gallery" (default) or "draw"
 *   hint     -- text under the pad; pass "" to suppress it
 *   settleMs -- while drawing, wait this long after the last pointer move
 *               before emitting (default 0: emit every frame, as before)
 * handle: { subscribe(fn), onPending(fn), current(), pending(), setMode(mode),
 *           setSettle(ms), clear(), destroy() }
 *
 * Subscribers are called as fn(digit28, meta) where meta is
 * {mode, index, label} -- label is null while drawing, because a drawing has
 * no ground truth. A late subscriber is replayed the current digit
 * immediately, so mount order does not matter.
 *
 * **Emission cadence is the consumers' problem to declare, and this is where it
 * is solved.** Moves are coalesced to one per animation frame, which is enough
 * while a consumer classifies inside a frame budget. It is not enough for a deep
 * network: two models at ~17 ms and ~129 ms block the main thread for ~146 ms
 * per frame, and since the pad's ink cannot be *painted* until the thread
 * yields, the stroke arrives in 146 ms jumps and the pad feels broken. With
 * `settleMs` set, nothing is emitted while the pen is moving; one emission
 * follows the pause. Consumers that are cheap pass nothing and are unaffected.
 *
 * The extraction is deliberately a *move*, not a rewrite: the pad geometry,
 * the stroke width and the MNIST normalisation are all load-bearing and were
 * verified against the trained model. d2nn_demo.js still carries its own copy
 * and is untouched -- the classifier page is live, and consolidating it is a
 * separate step with its own testing.
 */
(function () {
  "use strict";

  const DEFAULT_NET = (typeof window !== "undefined" && window.PhotonnD2NN_Net)
    ? window.PhotonnD2NN_Net
    : (typeof require !== "undefined" ? require("./d2nn.js") : null);

  const STYLE_ID = "ds-style";
  const CSS = `
.ds-root{--pe-fg:#1b1f24;--pe-muted:#5a6472;--pe-panel:#f4f6f9;--pe-border:#d7dde5;
  --pe-accent:#3b6ea5;
  color:var(--pe-fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  display:flex;flex-direction:column;gap:11px;}
@media (prefers-color-scheme:dark){.ds-root{--pe-fg:#e6eaf0;--pe-muted:#9aa6b5;
  --pe-panel:#1c2128;--pe-border:#30363d;--pe-accent:#6ea8e0;}}
.ds-seg{display:flex;border:1px solid var(--pe-border);border-radius:7px;overflow:hidden;
  align-self:flex-start;}
.ds-seg button{border:0;background:transparent;color:var(--pe-muted);padding:7px 14px;
  font:inherit;font-size:12px;font-weight:600;cursor:pointer;}
.ds-seg button[aria-pressed=true]{background:var(--pe-accent);color:#fff;}
.ds-gallery{display:flex;flex-wrap:wrap;gap:6px;}
.ds-gallery canvas{border:1px solid var(--pe-border);border-radius:5px;cursor:pointer;
  background:#000;image-rendering:pixelated;transition:border-color .12s,transform .12s;}
.ds-gallery canvas:hover{border-color:var(--pe-accent);}
.ds-gallery canvas[aria-selected=true]{border-color:var(--pe-accent);border-width:2px;
  transform:translateY(-2px);}
.ds-draw{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start;}
.ds-pad{width:196px;height:196px;background:#000;border-radius:8px;display:block;
  border:1px solid var(--pe-border);cursor:crosshair;touch-action:none;}
.ds-side{display:flex;flex-direction:column;gap:8px;align-items:flex-start;}
.ds-btn{border:1px solid var(--pe-border);background:transparent;color:var(--pe-fg);
  border-radius:7px;padding:6px 12px;font:inherit;font-size:12px;cursor:pointer;}
.ds-btn:hover{border-color:var(--pe-accent);color:var(--pe-accent);}
.ds-hint{font-size:12px;color:var(--pe-muted);max-width:26ch;margin:0;}
`;

  function injectStyle() {
    if (typeof document === "undefined" || document.getElementById(STYLE_ID)) return;
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

  const DEFAULT_HINT =
    "Draw a digit. It is re-centred and scaled the way MNIST was built, then sent "
    + "through the optics.";

  function mount(container, opts) {
    opts = opts || {};
    const net = opts.net || DEFAULT_NET;
    if (!net) throw new Error("digit_source.js needs a network (opts.net or d2nn.js)");
    injectStyle();

    const DIGIT = net.DIGIT;
    const THUMB = opts.thumb || 44;
    const PAD = 196;                       // CSS px; the drawing buffer matches 1:1

    const state = {
      mode: opts.mode === "draw" ? "draw" : "gallery",
      gallery: Math.min(Math.max(opts.gallery || 0, 0), net.nGallery - 1),
      drawing: false,
    };
    const subscribers = [];
    const pendingSubs = [];
    let last = null;
    let settleMs = Math.max(0, opts.settleMs || 0);
    let settleTimer = 0;
    let pending = false;

    // `pe-root` lets a host page retheme the widget through its own tokens
    // (build_site.py maps --pe-* onto the site palette); the defaults above
    // keep the standalone file:// pages looking right on their own.
    const root = el("div", "pe-root ds-root");
    container.appendChild(root);

    // ------------------------------------------------------------ mode toggle
    const seg = el("div", "ds-seg");
    const btnGallery = el("button", null, "Test digits");
    const btnDraw = el("button", null, "Draw your own");
    seg.appendChild(btnGallery); seg.appendChild(btnDraw);
    root.appendChild(seg);

    // --------------------------------------------------------------- gallery
    const gallery = el("div", "ds-gallery");
    const thumbs = [];
    for (let k = 0; k < net.nGallery; k++) {
      const c = el("canvas");
      c.width = DIGIT; c.height = DIGIT;
      c.style.width = THUMB + "px";
      c.style.height = THUMB + "px";
      c.title = "MNIST test digit, true label " + net.galleryLabel(k);
      const ctx = c.getContext("2d");
      const img = ctx.createImageData(DIGIT, DIGIT);
      const digit = net.galleryDigit(k);
      for (let i = 0; i < DIGIT * DIGIT; i++) {
        const v = (digit[i] * 255) | 0;
        img.data[i * 4] = v; img.data[i * 4 + 1] = v; img.data[i * 4 + 2] = v;
        img.data[i * 4 + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
      c.addEventListener("click", () => { state.gallery = k; selectGallery(); emit(); });
      gallery.appendChild(c);
      thumbs.push(c);
    }
    root.appendChild(gallery);

    // -------------------------------------------------------------- draw pad
    const drawWrap = el("div", "ds-draw");
    const pad = el("canvas", "ds-pad");
    pad.width = PAD; pad.height = PAD;
    const padCtx = pad.getContext("2d", { willReadFrequently: true });
    const side = el("div", "ds-side");
    const clearBtn = el("button", "ds-btn", "Clear");
    side.appendChild(clearBtn);
    const hint = opts.hint === undefined ? DEFAULT_HINT : opts.hint;
    if (hint) side.appendChild(el("p", "ds-hint", hint));
    drawWrap.appendChild(pad);
    drawWrap.appendChild(side);
    root.appendChild(drawWrap);

    function clearPad() {
      padCtx.fillStyle = "#000";
      padCtx.fillRect(0, 0, PAD, PAD);
    }
    clearPad();

    // -------------------------------------------------------------- emission
    function sameDigit(a, b) {
      if (!a || !b || a.length !== b.length) return false;
      for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
      return true;
    }

    /** Read the current input and hand the normalised digit to every subscriber.
     *
     * An emission that carries the same digit as the last one is dropped. That is
     * not a micro-optimisation here: pointerup re-emits after the final
     * pointermove already did, and a settle timer that fires while the pen rests
     * mid-stroke is followed by another emission on lift. Each duplicate costs a
     * consumer a full forward pass -- ~146 ms on the comparison board -- for a
     * result identical to the one already on screen.
     */
    function emit() {
      let digit, label = null, index = null;
      if (state.mode === "gallery") {
        index = state.gallery;
        digit = net.galleryDigit(index);
        label = net.galleryLabel(index);
      } else {
        const px = padCtx.getImageData(0, 0, PAD, PAD).data;
        const gray = new Float64Array(PAD * PAD);
        for (let i = 0; i < PAD * PAD; i++) gray[i] = px[i * 4] / 255;
        digit = net.normalizeDrawn(gray, PAD);
      }
      const meta = { mode: state.mode, index, label };
      const unchanged = last && last.meta.mode === meta.mode
        && last.meta.index === meta.index && sameDigit(last.digit, digit);

      last = { digit, meta };
      if (unchanged) return;
      for (let i = 0; i < subscribers.length; i++) subscribers[i](last.digit, last.meta);
    }

    // Recompute is rAF-gated rather than run per pointer event: a stroke fires
    // moves far faster than the consumers can classify, and coalescing them to
    // one per frame is what makes live recompute while drawing viable.
    let scheduled = false;
    function requestEmit() {
      if (scheduled) return;
      scheduled = true;
      requestAnimationFrame(() => { scheduled = false; emit(); });
    }

    /** True while a stroke is being drawn and its emission is still waiting. */
    function setPending(v) {
      if (v === pending) return;
      pending = v;
      for (let i = 0; i < pendingSubs.length; i++) pendingSubs[i](pending);
    }

    /** Restart the settle wait: emit once the pen has been still for settleMs. */
    function requestSettledEmit() {
      if (settleTimer) clearTimeout(settleTimer);
      setPending(true);
      settleTimer = setTimeout(() => {
        settleTimer = 0;
        setPending(false);
        requestEmit();
      }, settleMs);
    }

    /** Cancel any pending settle. Used when the stroke ends: a lift is a pause. */
    function cancelSettle() {
      if (settleTimer) clearTimeout(settleTimer);
      settleTimer = 0;
      setPending(false);
    }

    function inputChanged() {
      if (settleMs > 0 && state.drawing) requestSettledEmit();
      else requestEmit();
    }

    /** Register a consumer. Replays the current digit, so mount order is free. */
    function subscribe(fn) {
      subscribers.push(fn);
      if (last) fn(last.digit, last.meta);
      return () => {
        const i = subscribers.indexOf(fn);
        if (i >= 0) subscribers.splice(i, 1);
      };
    }

    /** Register a listener for "a stroke is waiting to be classified". */
    function onPending(fn) {
      pendingSubs.push(fn);
      fn(pending);
      return () => {
        const i = pendingSubs.indexOf(fn);
        if (i >= 0) pendingSubs.splice(i, 1);
      };
    }

    /**
     * Change the settle delay after mount.
     *
     * A consumer cannot know what it costs until it has run once, so the board
     * measures itself and calls this rather than the delay being guessed from a
     * model's size here. Dropping to 0 releases anything already waiting.
     */
    function setSettle(ms) {
      settleMs = Math.max(0, ms || 0);
      if (settleMs === 0 && settleTimer) { cancelSettle(); requestEmit(); }
    }

    // ---------------------------------------------------------------- events
    function selectGallery() {
      thumbs.forEach((t, k) => t.setAttribute("aria-selected", String(k === state.gallery)));
    }

    /** Show the controls for the current mode. No emission -- see setMode. */
    function applyMode() {
      btnGallery.setAttribute("aria-pressed", String(state.mode === "gallery"));
      btnDraw.setAttribute("aria-pressed", String(state.mode === "draw"));
      gallery.style.display = state.mode === "gallery" ? "" : "none";
      drawWrap.style.display = state.mode === "draw" ? "" : "none";
    }

    function setMode(mode) {
      state.mode = mode === "draw" ? "draw" : "gallery";
      applyMode();
      requestEmit();
    }

    btnGallery.addEventListener("click", () => setMode("gallery"));
    btnDraw.addEventListener("click", () => setMode("draw"));
    clearBtn.addEventListener("click", () => { clearPad(); cancelSettle(); requestEmit(); });

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
      inputChanged();
      ev.preventDefault();
    });
    pad.addEventListener("pointermove", (ev) => {
      if (!state.drawing) return;
      const p = padPos(ev);
      strokeTo(p[0], p[1]);
      inputChanged();
      ev.preventDefault();
    });
    // Lifting the pen is the clearest pause there is, so it pre-empts the timer
    // rather than waiting out another settleMs.
    const endStroke = () => { state.drawing = false; cancelSettle(); requestEmit(); };
    pad.addEventListener("pointerup", endStroke);
    pad.addEventListener("pointercancel", endStroke);
    pad.addEventListener("pointerleave", () => { if (state.drawing) state.drawing = false; });

    selectGallery();
    applyMode();
    // Emit synchronously on mount so a consumer that subscribes immediately has
    // something to draw without waiting a frame -- and so the first frame does
    // one round of classification rather than two.
    emit();

    return {
      subscribe,
      onPending,
      current: () => last,
      pending: () => pending,
      setMode,
      setSettle,
      clear: () => { clearPad(); cancelSettle(); requestEmit(); },
      destroy: () => {
        cancelSettle();
        subscribers.length = 0;
        pendingSubs.length = 0;
        root.remove();
      },
    };
  }

  const API = { mount };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  if (typeof window !== "undefined") window.PhotonnDigitSource = API;
})();
