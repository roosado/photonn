/*
 * plot.js -- the canvas primitives every drawing widget on this site shares.
 *
 * Nine widgets each carried their own copy of the same five things: the
 * device-pixel-ratio clamp, the guarded canvas resize, the width observer, the
 * palette read, the colour-ramp lookup tables, and the scalar-field rasteriser.
 * Each copy was shallow -- an interface as large as its implementation -- and
 * they had drifted: `fitCanvas` existed in four files in two mutually
 * incompatible shapes, `MAX_DPR` in eight with the same six-line comment pasted
 * verbatim, the inferno LUT in three.
 *
 * The cost of that was not tidiness. Commit 4b1e842 fixed an unguarded bitmap
 * reallocation in two copies of `fitCanvas` and noted "a third copy is the point
 * to reconsider" -- and the third copy already existed, in d2nn_stage.js, under a
 * different name, inside the rAF body that drives orbiting. The one widget on the
 * site that actually animates went on discarding and reallocating a ~1440x860
 * surface every frame. Separately, three widgets learned about theme changes from
 * `matchMedia` alone, which never fires for this site's own toggle, so their
 * charts held the previous theme's ink until something resized them.
 *
 * One module, one guard, one ramp.
 *
 * Load order: this must be evaluated before any widget that uses it, because
 * widgets read it at module scope. apps/build_site.py emits it first; the Node
 * runners require() it directly.
 */
(function () {
  "use strict";

  /* -------------------------------------------------------------- geometry */

  // Backing-store scale, capped at 2x.
  //
  // A dpr-3 phone would otherwise get 2.25x the pixels of a dpr-2 one for a
  // difference nobody can see at arm's length, and the cost is quadratic in the
  // canvas area -- the 3D stage re-rasterises ~64 drawImage calls at
  // imageSmoothingQuality "high" on every orbit frame, so this is the difference
  // between a smooth orbit and a slideshow on exactly the devices least able to
  // afford it.
  const MAX_DPR = 2;

  function scale() {
    const dpr = (typeof window !== "undefined" && window.devicePixelRatio) || 1;
    return Math.min(dpr, MAX_DPR);
  }

  /**
   * Resize a canvas to `pxW` x `pxH` backing pixels, but only if that changed.
   *
   * Assigning `canvas.width` throws the bitmap away and allocates a new one
   * *even when the value is unchanged* -- it is the `canvas.width = canvas.width`
   * clear idiom. On a slider drag or an orbit frame the width never moves, so
   * the unguarded form reallocates the whole surface per event.
   *
   * Nothing may depend on the implicit clear: every caller here repaints the
   * whole surface as its first act, and the ones that need a blank ground call
   * clearRect explicitly.
   *
   * @returns {boolean} whether the backing store was actually resized.
   */
  function resize(canvas, pxW, pxH) {
    if (canvas.width === pxW && canvas.height === pxH) return false;
    canvas.width = pxW;
    canvas.height = pxH;
    return true;
  }

  /**
   * Size a canvas from its own laid-out width; return a ready 2D context.
   *
   * `heightFor(cssWidth)` gives the CSS height. Use this when the height is a
   * function of the width, which is the common case on this site.
   *
   * `setTransform` rather than `scale`: a resize clears the context, so the dpr
   * transform has to be re-established after one, and re-applying the same
   * absolute transform when the guard skipped the resize is a no-op.
   */
  function fit(canvas, heightFor, opts) {
    const o = opts || {};
    const r = scale();
    const min = o.minWidth === undefined ? 240 : o.minWidth;
    const fallback = o.fallbackWidth === undefined ? 420 : o.fallbackWidth;
    const measured = (canvas.getBoundingClientRect && canvas.getBoundingClientRect().width)
      || canvas.clientWidth || fallback;
    const W = Math.max(min, Math.round(measured));
    const H = Math.round(heightFor(W));
    resize(canvas, Math.round(W * r), Math.round(H * r));
    // Written either way: a canvas that has never been sized is already exactly
    // 300x150 and would otherwise be left with no height at all at that one size.
    canvas.style.height = H + "px";
    const ctx = canvas.getContext("2d");
    ctx.setTransform(r, 0, 0, r, 0, 0);
    return { ctx, W, H, dpr: r };
  }

  /**
   * Size a canvas to a CSS box the caller already computed; return its context.
   *
   * The other legitimate shape: the caller knows both dimensions because the
   * layout gave them, rather than deriving height from width.
   */
  function fitTo(canvas, cssW, cssH) {
    const r = scale();
    resize(canvas, Math.round(cssW * r), Math.round(cssH * r));
    canvas.style.height = cssH + "px";
    const ctx = canvas.getContext("2d");
    ctx.setTransform(r, 0, 0, r, 0, 0);
    return { ctx, W: cssW, H: cssH, dpr: r };
  }

  /* --------------------------------------------------------------- reacting */

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
    if (typeof window !== "undefined" && typeof window.ResizeObserver === "function") {
      new window.ResizeObserver(check).observe(target);
    } else if (typeof window !== "undefined") {
      window.addEventListener("resize", check);
    }
  }

  /**
   * Re-run `fn` when the page theme changes, however it changed.
   *
   * Both sources, because there are two. The site's toggle sets `data-theme` on
   * the root element and never touches the OS preference, so a widget listening
   * only on `matchMedia` hears nothing when a reader flips it -- the card around
   * a chart restyles instantly, being CSS, while the chart inside keeps the old
   * ink until something else forces a redraw. A reader on "system" with no
   * toggle interaction is the other case, and only matchMedia sees that one.
   */
  function onThemeChange(fn) {
    if (typeof document !== "undefined" && typeof MutationObserver !== "undefined"
        && document.documentElement) {
      new MutationObserver(fn).observe(document.documentElement,
        { attributes: true, attributeFilter: ["data-theme"] });
    }
    if (typeof window !== "undefined" && window.matchMedia) {
      const mq = window.matchMedia("(prefers-color-scheme:dark)");
      if (mq.addEventListener) mq.addEventListener("change", fn);
      else if (mq.addListener) mq.addListener(fn);
    }
  }

  /* ----------------------------------------------------------------- colour */

  /**
   * Read CSS custom properties off `root`, with a fallback for each.
   *
   * `spec` maps the name you want in the result to `[property, fallback]`. The
   * fallback matters: a canvas is drawn before styles necessarily resolve, and a
   * widget drawing in `""` renders invisibly rather than obviously wrong.
   */
  function readVars(root, spec) {
    const cs = (typeof getComputedStyle === "function")
      ? getComputedStyle(root)
      : { getPropertyValue: () => "" };
    const out = {};
    for (const key of Object.keys(spec)) {
      const [prop, fallback] = spec[key];
      out[key] = String(cs.getPropertyValue(prop) || "").trim() || fallback;
    }
    return out;
  }

  /** The site's ink, read from the CSS custom properties in force at `root`. */
  function palette(root) {
    const cs = (typeof getComputedStyle === "function")
      ? getComputedStyle(root)
      : { getPropertyValue: () => "" };
    const get = (k, f) => (String(cs.getPropertyValue(k) || "").trim() || f);
    return {
      fg: get("--pe-fg", "#1b1f24"),
      muted: get("--pe-muted", "#5a6472"),
      panel: get("--pe-panel", "#f4f6f9"),
      border: get("--pe-border", "#d7dde5"),
      accent: get("--pe-accent", "#3b6ea5"),
      ok: get("--pe-ok", "#3f8f4e"),
      warn: get("--pe-warn", "#c14a3d"),
    };
  }

  // One visual language for optical intensity across the whole site: the 3D
  // stage, the filmstrip and the diffraction explorer all speak inferno.
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

  /** Expand colour anchors into a 256-entry RGB lookup table. */
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

  const LUT_INTENSITY = makeLUT(INFERNO);
  const LUT_PHASE = makeLUT(TWILIGHT);

  /**
   * Rasterise an n-by-n scalar field into an offscreen canvas.
   *
   * @param {ArrayLike<number>} data  n*n samples, row-major.
   * @param {number} n
   * @param {object} [opts]
   * @param {Uint8ClampedArray} [opts.lut]     ramp; defaults to intensity.
   * @param {number} [opts.gamma]  stretch applied to the normalised value.
   *   Detector intensity spans decades, so 0.5 (a sqrt stretch) is the usual
   *   choice; 1 leaves it linear.
   * @param {boolean} [opts.cyclic]  map over [-pi, pi] instead of [0, max].
   *   Phase is cyclic and must not be auto-scaled, or two masks with different
   *   ranges would be drawn as though they matched.
   * @param {number} [opts.lo] [opts.hi]  explicit range, overriding both.
   */
  /**
   * Rasterise into a canvas the caller already has, rather than a fresh one.
   *
   * The reuse matters for a widget that redraws on every slider move: allocating
   * a new offscreen canvas per frame is the same waste as reallocating a backing
   * store per frame, one level up.
   */
  function rasterInto(off, data, n, opts) {
    const o = opts || {};
    const lut = o.lut || LUT_INTENSITY;
    const gamma = o.gamma === undefined ? 1 : o.gamma;

    off.width = n; off.height = n;
    const ctx = off.getContext("2d");
    const img = ctx.createImageData(n, n);
    const d = img.data;

    let lo = 0, hi = 1;
    if (o.lo !== undefined && o.hi !== undefined) { lo = o.lo; hi = o.hi; }
    else if (o.cyclic) { lo = -Math.PI; hi = Math.PI; }
    else {
      hi = 0;
      for (let i = 0; i < data.length; i++) if (data[i] > hi) hi = data[i];
      if (hi <= 0) hi = 1;
    }
    const span = hi - lo || 1;

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

  function raster(data, n, opts) {
    return rasterInto(document.createElement("canvas"), data, n, opts);
  }

  /* -------------------------------------------------------------------- DOM */

  /**
   * Inject a widget's stylesheet once per page.
   *
   * Guarded on the id, which is right for a widget mounted more than once and
   * quietly wrong when two different widgets pick the same id -- the second then
   * finds the id taken and runs with none of its own CSS.
   * tests/test_web_style_ids.py asserts both that ids are unique and that class
   * prefixes are.
   */
  function injectStyle(id, css) {
    if (document.getElementById(id)) return;
    const s = document.createElement("style");
    s.id = id;
    s.textContent = css;
    document.head.appendChild(s);
  }

  /** `document.createElement` with a class and inner markup in one call. */
  function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }

  const API = {
    MAX_DPR, scale, resize, fit, fitTo,
    onWidthChange, onThemeChange,
    palette, readVars, INFERNO, TWILIGHT, makeLUT, LUT_INTENSITY, LUT_PHASE, raster, rasterInto,
    injectStyle, el,
  };

  if (typeof module !== "undefined" && module.exports) module.exports = API;
  if (typeof window !== "undefined") window.PhotonnPlot = API;
})();
