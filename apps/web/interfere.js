/*
 * interfere.js -- two waves, one slider, on the front page.
 *
 * The section "Every panel is real light" opens on the sentence the rest of the
 * site rests on: amplitude is what makes light bright, phase is what decides
 * whether two waves reinforce or cancel, and your eye sees the first and is
 * blind to the second. Everything downstream depends on the reader believing
 * that -- the plates only compute because delays make light add in one detector
 * box and cancel in the other nine, and the whole tolerance study is about how
 * far a delay may slip before the cancelling stops working. Then the page shows
 * a 128^2 field where the effect is present everywhere and visible nowhere.
 *
 * So: two equal waves, a slider that holds the second one back, and a swatch
 * showing what a detector would read. That is the entire widget. It is not a
 * second diffraction explorer -- there is no amplitude control, no wavelength,
 * no second frequency, and no animation.
 *
 * NO requestAnimationFrame. A travelling-wave version would read well and could
 * not be verified here: the driven Chrome tab is always hidden and never fires
 * animation frames, which has already cost this project one silently dead
 * feature. A still frame plus a slider says the same thing and is checkable
 * under Node.
 *
 * The physics is exact and has a closed form, which is what makes this the
 * cheapest honest widget on the site:
 *
 *     cos(kx) + cos(kx - d)  =  2*cos(d/2) * cos(kx - d/2)
 *
 * `samples()` computes the left side pointwise and the drawing consumes it;
 * `envelope()` is the right side's amplitude. tests/test_interference_widget.py
 * checks one against the other, so the picture cannot drift from the identity
 * it is illustrating.
 *
 * Usage:  window.PhotonnInterfere.mount(el)
 */
(function () {
  "use strict";

  const STYLE_ID = "if-style";
  const MAX_DPR = 2;
  function dpr() { return Math.min(window.devicePixelRatio || 1, MAX_DPR); }

  const CSS = `
.if-root{--if-fg:#1b1f24;--if-muted:#5a6472;--if-panel:#f4f6f9;--if-border:#d7dde5;
  --if-accent:#3b6ea5;
  color:var(--if-fg);font:13px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  background:var(--if-panel);border:1px solid var(--if-border);border-radius:10px;padding:13px 15px;}
@media (prefers-color-scheme:dark){.if-root{--if-fg:#e6eaf0;--if-muted:#9aa6b5;
  --if-panel:#1c2128;--if-border:#30363d;--if-accent:#6ea8e0;}}
:root[data-theme="dark"] .if-root{--if-fg:#e6eaf0;--if-muted:#9aa6b5;--if-panel:#1c2128;
  --if-border:#30363d;--if-accent:#6ea8e0;}
:root[data-theme="light"] .if-root{--if-fg:#1b1f24;--if-muted:#5a6472;--if-panel:#f4f6f9;
  --if-border:#d7dde5;--if-accent:#3b6ea5;}
.if-row{display:flex;gap:16px;align-items:flex-end;flex-wrap:wrap;margin-bottom:11px;}
.if-row label{font-size:11px;font-weight:600;color:var(--if-muted);text-transform:uppercase;
  letter-spacing:.04em;display:block;margin-bottom:3px;}
.if-row input[type=range]{width:240px;max-width:100%;accent-color:var(--if-accent);display:block;}
.if-val{font-variant-numeric:tabular-nums;font-weight:600;font-size:14px;white-space:nowrap;}
/* The meter is the point of the widget, not decoration: it is the only thing on
   the page that shows phase turning into something the eye can see. */
.if-meter{display:flex;gap:9px;align-items:center;margin-left:auto;}
.if-swatch{width:36px;height:36px;border-radius:6px;border:1px solid var(--if-border);
  display:block;flex:none;background:#000;}
.if-read{font-size:11px;color:var(--if-muted);text-transform:uppercase;letter-spacing:.04em;
  line-height:1.35;}
.if-read b{display:block;font-size:15px;color:var(--if-fg);text-transform:none;
  letter-spacing:0;font-variant-numeric:tabular-nums;}
/* One wide plot, capped so it does not sprawl into a letterbox on a desktop.
   Its canvas measures this pane before drawing -- see fitCanvas. */
.if-plot{max-width:640px;margin:0 auto;}
.if-plot canvas{width:100%;height:auto;display:block;border-radius:6px;background:#0b0d10;}
.if-cap{font-size:11px;color:var(--if-muted);margin:5px 0 0;text-align:center;line-height:1.35;}
.if-note{font-size:12px;color:var(--if-muted);margin:11px 0 0;line-height:1.5;}
.if-note b{color:var(--if-fg);}
@media (max-width:560px){
  .if-row{gap:10px;margin-bottom:9px;}
  .if-row input[type=range]{width:min(240px,58vw);}
  .if-meter{margin-left:0;}
  .if-cap{font-size:10px;line-height:1.3;}
}
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

  // ------------------------------------------------------------------ physics
  //: wavelengths drawn across the pane. Three is enough to read the shift as a
  //: shift rather than as a different wave, and few enough to stay legible on a
  //: phone at ~280 px.
  const CYCLES = 3;

  /** Amplitude of the sum, in units of one wave: 2 in step, 0 exactly opposed. */
  function envelope(dphi) { return 2 * Math.cos(dphi / 2); }

  /**
   * Brightness of the sum as a fraction of the most the pair can reach.
   *
   * Brightness is amplitude squared -- the one non-linear step in the whole
   * machine (see the front page's |E|^2). Normalised to the in-step case so the
   * readout is 1.00 at 0, exactly 0.50 at a quarter wave, and 0.00 at a half.
   */
  function brightness(dphi) {
    const e = envelope(dphi) / 2;
    return e * e;
  }

  /**
   * The two waves and their sum, sampled across the pane.
   *
   * The second wave is *held back* by `dphi` (cos(kx - d), crest later), which is
   * what a phase mask physically does to the light passing through it, so the
   * slider reads as a delay rather than as an abstract offset.
   */
  function samples(dphi, n) {
    const out = {
      t: new Float64Array(n),
      a: new Float64Array(n),
      b: new Float64Array(n),
      sum: new Float64Array(n),
    };
    for (let i = 0; i < n; i++) {
      const t = n > 1 ? i / (n - 1) : 0;
      const kx = t * CYCLES * 2 * Math.PI;
      out.t[i] = t;
      out.a[i] = Math.cos(kx);
      out.b[i] = Math.cos(kx - dphi);
      out.sum[i] = out.a[i] + out.b[i];
    }
    return out;
  }

  // ------------------------------------------------------------------ drawing
  // The plot stays dark in both themes, as the tolerance page's plots do: the
  // canvas cannot read a CSS variable without getComputedStyle, and one set of
  // stroke colours chosen for one background is worth more than two half-tuned
  // ones. Only the card around it follows the theme.
  const PLOT_BG = "#0b0d10";
  const AXIS = "#39414f";
  const MUTED = "#9aa6b5";
  const WAVE_A = "#6ea8e0";
  const WAVE_B = "#e0a25f";
  const SUM = "#f2f5f9";

  /**
   * Size the canvas to the width it is *actually* laid out at, so one drawing
   * unit is one CSS pixel in both directions.
   *
   * A fixed drawing space scaled by width:100% stretches x without stretching y,
   * which flattens the plot on a desktop and stretches it on a phone, labels and
   * all. errors.js shipped exactly that bug in its two plot widgets.
   *
   * setTransform is right here: resizing a canvas resets its context, so this
   * establishes the dpr scale rather than replacing one already in place.
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
   * Guarded on the measured width because `fn` sets the canvas height, which is
   * itself a resize: an unguarded observer would answer its own callback for
   * ever.
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

  function curve(ctx, s, key, x0, span, mid, unit, colour, width) {
    ctx.strokeStyle = colour;
    ctx.lineWidth = width;
    ctx.beginPath();
    const v = s[key];
    for (let i = 0; i < v.length; i++) {
      const x = x0 + s.t[i] * span;
      const y = mid - v[i] * unit;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  function draw(c, dphi) {
    const { ctx, W, H } = fitCanvas(c, (w) => Math.max(152, Math.min(230, w * 0.42)));
    const padL = 12, padR = 12, padT = 30, padB = 20;
    const span = W - padL - padR;
    const box = H - padT - padB;
    const mid = padT + box / 2;
    // +-2.3 of headroom, so the in-step sum at +-2 clears the frame.
    const unit = box / 2 / 2.3;
    const s = samples(dphi, Math.max(96, Math.round(span)));

    ctx.fillStyle = PLOT_BG;
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = AXIS;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, mid);
    ctx.lineTo(padL + span, mid);
    ctx.stroke();

    curve(ctx, s, "a", padL, span, mid, unit, WAVE_A, 1.3);
    curve(ctx, s, "b", padL, span, mid, unit, WAVE_B, 1.3);
    curve(ctx, s, "sum", padL, span, mid, unit, SUM, 2.4);

    // The delay, drawn where it happens: wave 1 crests at x = 0, wave 2 crests
    // dphi later. The bracket between the two crests is the slider's value made
    // physical, which is the whole reason the second wave is written cos(kx - d).
    const shift = (dphi / (CYCLES * 2 * Math.PI)) * span;
    if (shift > 3) {
      const y = 15;
      ctx.strokeStyle = MUTED;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(padL, y + 3); ctx.lineTo(padL, mid - unit);
      ctx.moveTo(padL + shift, y + 3); ctx.lineTo(padL + shift, mid - unit);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(padL, y); ctx.lineTo(padL + shift, y);
      ctx.stroke();
      if (shift > 46) {
        ctx.fillStyle = MUTED;
        ctx.font = "10px ui-monospace, monospace";
        ctx.textAlign = "center";
        ctx.fillText("delay", padL + shift / 2, y - 4);
      }
    }

    // Legend, in the canvas rather than the DOM: these three colours are chosen
    // against a dark plot and would not survive on a light card.
    ctx.font = "11px ui-monospace, monospace";
    ctx.textAlign = "left";
    const keys = [["wave 1", WAVE_A], ["wave 2", WAVE_B], ["their sum", SUM]];
    let kx = padL;
    for (let i = 0; i < keys.length; i++) {
      ctx.fillStyle = keys[i][1];
      ctx.fillRect(kx, H - 12, 9, 3);
      ctx.fillText(keys[i][0], kx + 13, H - 6);
      kx += 13 + ctx.measureText(keys[i][0]).width + 14;
    }
  }

  // -------------------------------------------------------------------- mount
  function fmt(x, n) { return x.toFixed(n); }

  /**
   * sRGB level for a brightness fraction.
   *
   * The swatch is a light meter, so it has to *emit* the fraction it prints. A
   * display is gamma-encoded, so a pixel value of 128 emits about a fifth of full
   * light, not half -- writing 255*b would make every reading look far darker
   * than the number beside it.
   */
  function swatchColour(b) {
    const v = Math.round(255 * Math.pow(Math.max(0, Math.min(1, b)), 1 / 2.2));
    return "rgb(" + v + "," + v + "," + v + ")";
  }

  function mount(container, opts) {
    opts = opts || {};
    injectStyle();
    const root = el("div", "if-root");
    container.innerHTML = "";
    container.appendChild(root);

    const row = el("div", "if-row");
    root.appendChild(row);

    const ctl = el("div");
    ctl.appendChild(el("label", null, "Hold the second wave back"));
    const input = document.createElement("input");
    input.type = "range";
    input.min = "0";
    input.max = String(2 * Math.PI);
    input.step = "0.01";
    // Opens off zero on purpose: at zero the two waves lie on top of each other
    // and the picture reads as one wave, which is the opposite of the point.
    input.value = String(opts.dphi != null ? opts.dphi : 2);
    input.setAttribute("aria-label", "Delay of the second wave, in radians");
    ctl.appendChild(input);
    row.appendChild(ctl);

    const val = el("span", "if-val", "");
    const valWrap = el("div");
    valWrap.appendChild(val);
    row.appendChild(valWrap);

    const meter = el("div", "if-meter");
    const swatch = el("span", "if-swatch");
    const read = el("span", "if-read", "");
    meter.appendChild(swatch);
    meter.appendChild(read);
    row.appendChild(meter);

    const plot = el("div", "if-plot");
    const canvas = el("canvas");
    plot.appendChild(canvas);
    plot.appendChild(el("p", "if-cap",
      "Two waves of equal brightness, and their sum. Wave 1 crests at the left edge; "
      + "wave 2 crests wherever the slider puts it."));
    root.appendChild(plot);

    const note = el("p", "if-note");
    root.appendChild(note);

    function update() {
      const dphi = +input.value;
      const amp = Math.abs(envelope(dphi));
      const b = brightness(dphi);

      draw(canvas, dphi);

      val.textContent = fmt(dphi / (2 * Math.PI), 2) + " λ · "
        + fmt(dphi, 2) + " rad";
      swatch.style.background = swatchColour(b);
      read.innerHTML = "what a detector reads<b>" + fmt(b, 2) + "</b>";

      let head;
      if (b > 0.98) {
        head = "<b>In step.</b> Crest meets crest, the waves add, and the sum is twice as "
          + "tall as either one: <b>four times</b> the brightness of a single wave.";
      } else if (b < 0.02) {
        head = "<b>Exactly out of step</b>, half a wavelength apart. Every crest meets a "
          + "trough, the sum is flat, and <b>the light is gone</b>. Nothing absorbed it; "
          + "the two waves simply cancel.";
      } else {
        head = "Partly out of step. The sum stands <b>" + fmt(amp, 2) + "×</b> one "
          + "wave, so a detector here collects <b>" + fmt(b, 2) + "</b> of the light it "
          + "would collect with the two in step.";
      }
      note.innerHTML = head + " Neither wave changed brightness, only <em>when</em> the "
        + "second one arrives, and your eye cannot see that difference until the two are "
        + "added together. Setting that delay is the only thing a phase mask does, at "
        + "16,384 points at once, five plates deep.";
    }

    input.addEventListener("input", update);
    onWidthChange(canvas, update);
    update();
  }

  // envelope/brightness/samples are exported because the drawing is only worth
  // trusting if something checks it against the identity it illustrates:
  // tests/test_interference_widget.py holds samples() to 2cos(d/2)cos(kx-d/2).
  const api = { mount, envelope, brightness, samples, CYCLES };
  if (typeof window !== "undefined") window.PhotonnInterfere = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})();
