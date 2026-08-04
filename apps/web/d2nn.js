/*
 * d2nn.js -- the trained diffractive network's forward pass, in the browser.
 *
 * A JavaScript port of photonn/models.py:D2NN plus the input encoding in
 * photonn/train.py:encode_input, running the *already trained* phase masks from
 * d2nn_weights.js on the propagation engine in asm.js. Nothing here is trained
 * or fitted; this only evaluates.
 *
 * The optical path, exactly as in models.py:D2NN.output_field --
 *
 *     for i in 0..n_layers-1:  E = propagate(E, z);  E = E * exp(i * mask_i)
 *     E = propagate(E, z)                            -- n_layers+1 hops total
 *
 * -- then detection: I = |E|^2, integrated over ten fixed detector boxes and
 * normalised by the total output power. The single |E|^2 is the only
 * nonlinearity anywhere in the model.
 *
 * All n_layers+1 propagations cover the same distance, so the angular-spectrum
 * transfer function is built once and reused (see asm.js:propagateField).
 *
 * Pure computation -- no DOM, no network, no libraries. The widget that draws it
 * is d2nn_demo.js. Cross-checked against the torch model by
 * tests/test_d2nn_crosscheck.py: identical predictions, logits within 1e-3.
 *
 * Usage:  window.PhotonnD2NN_Net.classify(img28) -> {logits, pred, ...}
 */
(function () {
  "use strict";

  const ASM = (typeof window !== "undefined" && window.ASM) ? window.ASM
    : (typeof require !== "undefined" ? require("./asm.js") : null);
  const W = (typeof window !== "undefined" && window.D2NN_WEIGHTS) ? window.D2NN_WEIGHTS
    : (typeof require !== "undefined" ? require("./d2nn_weights.js") : null);
  if (!ASM || !W) throw new Error("d2nn.js needs asm.js and d2nn_weights.js loaded first");

  const N = W.n;                                   // field grid (128)
  const WIN = Math.max(1, Math.round(W.input_frac * N));   // digit window (64)
  const OFF = (N - WIN) >> 1;                      // its offset in the canvas (32)
  const DIGIT = W.gallery_size;                    // 28

  // ---------------------------------------------------------------- base64
  // The weights ship as little-endian typed arrays (every platform that runs a
  // browser is little-endian; the exporter writes '<f4'/'u1' explicitly).
  function b64ToBytes(s) {
    if (typeof atob === "function") {
      const bin = atob(s);
      const out = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
      return out;
    }
    return new Uint8Array(Buffer.from(s, "base64"));
  }

  // Copy into a fresh buffer: a Buffer's bytes are not guaranteed 4-byte aligned,
  // which Float32Array's buffer view requires.
  function b64ToFloat32(s) {
    const b = b64ToBytes(s);
    const buf = new ArrayBuffer(b.length);
    new Uint8Array(buf).set(b);
    return new Float32Array(buf);
  }

  // ------------------------------------------------------- decoded weights
  const MASKS = b64ToFloat32(W.masks_b64);         // n_layers * N * N, radians
  const GALLERY = b64ToBytes(W.gallery_b64);       // k * 28 * 28, uint8

  // exp(i*phi) per mask element, precomputed once: the same five masks are
  // applied on every inference, so this trades ~1.3 MB for 5*N*N trig calls
  // per classification.
  const MASK_COS = new Float64Array(MASKS.length);
  const MASK_SIN = new Float64Array(MASKS.length);
  for (let i = 0; i < MASKS.length; i++) {
    MASK_COS[i] = Math.cos(MASKS[i]);
    MASK_SIN[i] = Math.sin(MASKS[i]);
  }

  // One transfer function for every hop (all hops are the same distance).
  const H = ASM.buildTransfer(N, W.dx, W.wavelength, W.separation);

  const N_GALLERY = W.gallery_labels.length;

  // ------------------------------------------------------------- resampling
  // Bilinear resample of the sub-rectangle (x0, y0, w, h) of a sw x sh source
  // into a dw x dh destination, using half-pixel sample centres -- torch's
  // F.interpolate(..., mode="bilinear", align_corners=False) convention, which
  // is what produced the trained model's inputs. Verified against the frozen
  // canvases in tests/fixtures/d2nn_reference.json.
  function resampleRect(src, sw, sh, x0, y0, w, h, dw, dh) {
    const out = new Float64Array(dw * dh);
    const sx = w / dw;
    const sy = h / dh;
    for (let j = 0; j < dh; j++) {
      const fy = y0 + (j + 0.5) * sy - 0.5;
      const iy = Math.floor(fy);
      const ty = fy - iy;
      const ya = Math.min(Math.max(iy, 0), sh - 1);
      const yb = Math.min(Math.max(iy + 1, 0), sh - 1);
      for (let i = 0; i < dw; i++) {
        const fx = x0 + (i + 0.5) * sx - 0.5;
        const ix = Math.floor(fx);
        const tx = fx - ix;
        const xa = Math.min(Math.max(ix, 0), sw - 1);
        const xb = Math.min(Math.max(ix + 1, 0), sw - 1);
        const top = src[ya * sw + xa] * (1 - tx) + src[ya * sw + xb] * tx;
        const bot = src[yb * sw + xa] * (1 - tx) + src[yb * sw + xb] * tx;
        out[j * dw + i] = top * (1 - ty) + bot * ty;
      }
    }
    return out;
  }

  /** Resize a whole square image (the common case). */
  function resize(src, inSize, outSize) {
    return resampleRect(src, inSize, inSize, 0, 0, inSize, inSize, outSize, outSize);
  }

  // ------------------------------------------------------------ preprocessing
  /**
   * 28x28 digit (values in [0,1]) -> the N x N input-plane transmission map.
   * Mirrors photonn/train.py:_embed -- bilinear to a WIN x WIN window, centred
   * in the grid, zero elsewhere, so diffraction has a margin to spread into.
   */
  function preprocess(img28) {
    const small = resize(img28, DIGIT, WIN);
    const canvas = new Float64Array(N * N);
    for (let j = 0; j < WIN; j++) {
      const dst = (OFF + j) * N + OFF;
      for (let i = 0; i < WIN; i++) canvas[dst + i] = small[j * WIN + i];
    }
    return canvas;
  }

  /**
   * Hand-drawn strokes -> a 28x28 digit in MNIST's own normalisation.
   *
   * MNIST digits were built by cropping to the ink, scaling the longer side to
   * 20 px with the aspect ratio preserved, and centring the result in a 28x28
   * frame by centre of mass. A raw drawing does not follow any of that, and the
   * trained masks are sharply tuned to it -- so we apply the same normalisation
   * rather than let a size/position mismatch masquerade as an optical failure.
   * Drawings remain out of distribution in stroke width and style; this only
   * removes the part of the gap that is bookkeeping.
   */
  function normalizeDrawn(gray, size) {
    // 1. ink bounding box
    let x0 = size, y0 = size, x1 = -1, y1 = -1;
    for (let j = 0; j < size; j++) {
      for (let i = 0; i < size; i++) {
        if (gray[j * size + i] > 0.05) {
          if (i < x0) x0 = i;
          if (i > x1) x1 = i;
          if (j < y0) y0 = j;
          if (j > y1) y1 = j;
        }
      }
    }
    if (x1 < 0) return new Float64Array(DIGIT * DIGIT);   // blank canvas

    // 2. scale the longer side to 20 px, aspect preserved
    const w = x1 - x0 + 1;
    const h = y1 - y0 + 1;
    const box = 20;
    const scale = box / Math.max(w, h);
    const dw = Math.max(1, Math.round(w * scale));
    const dh = Math.max(1, Math.round(h * scale));
    const scaled = resampleRect(gray, size, size, x0, y0, w, h, dw, dh);

    // 3. drop it into 28x28, then shift so the centre of mass sits at the centre
    let sum = 0, cx = 0, cy = 0;
    for (let j = 0; j < dh; j++) {
      for (let i = 0; i < dw; i++) {
        const v = scaled[j * dw + i];
        sum += v; cx += v * i; cy += v * j;
      }
    }
    const ox = sum > 0 ? Math.round(DIGIT / 2 - cx / sum - 0.5) : Math.round((DIGIT - dw) / 2);
    const oy = sum > 0 ? Math.round(DIGIT / 2 - cy / sum - 0.5) : Math.round((DIGIT - dh) / 2);

    const out = new Float64Array(DIGIT * DIGIT);
    for (let j = 0; j < dh; j++) {
      const ty = j + oy;
      if (ty < 0 || ty >= DIGIT) continue;
      for (let i = 0; i < dw; i++) {
        const tx = i + ox;
        if (tx < 0 || tx >= DIGIT) continue;
        out[ty * DIGIT + tx] = scaled[j * dw + i];
      }
    }
    return out;
  }

  // ---------------------------------------------------------------- encoding
  /**
   * Input map -> complex entrance field, scheme "both" (train.py:encode_input):
   * the image modulates amplitude *and* phase, |E| = img, arg(E) = phase_scale*img,
   * normalised to unit total intensity. (Full complex modulation needs two
   * cascaded modulators in hardware -- documented in docs/phase2_dnn.md.)
   */
  function encodeInput(canvas) {
    const re = new Float64Array(N * N);
    const im = new Float64Array(N * N);
    let power = 0;
    for (let i = 0; i < canvas.length; i++) power += canvas[i] * canvas[i];
    const norm = 1 / Math.sqrt(Math.max(power, 1e-12));
    for (let i = 0; i < canvas.length; i++) {
      const a = canvas[i] * norm;
      const p = canvas[i] * W.phase_scale;
      re[i] = a * Math.cos(p);
      im[i] = a * Math.sin(p);
    }
    return [re, im];
  }

  // ----------------------------------------------------------------- forward
  /**
   * Propagate an encoded field through the mask stack and detect.
   * Returns the class logits, the per-class share of output power, the
   * prediction, the detector-plane intensity, and the intensity arriving at each
   * mask plane (already computed on the way through, so the filmstrip is free).
   */
  function forward(re, im) {
    const planes = [];
    let a = re, b = im;

    for (let L = 0; L < W.n_layers; L++) {
      const p = ASM.propagateField(a, b, N, H[0], H[1]);
      a = p[0]; b = p[1];

      const I = new Float64Array(N * N);
      for (let i = 0; i < I.length; i++) I[i] = a[i] * a[i] + b[i] * b[i];
      planes.push(I);

      const off = L * N * N;
      for (let i = 0; i < N * N; i++) {
        const c = MASK_COS[off + i], s = MASK_SIN[off + i];
        const x = a[i], y = b[i];
        a[i] = x * c - y * s;
        b[i] = x * s + y * c;
      }
    }
    const out = ASM.propagateField(a, b, N, H[0], H[1]);
    a = out[0]; b = out[1];

    const intensity = new Float64Array(N * N);
    let total = 0;
    for (let i = 0; i < intensity.length; i++) {
      const v = a[i] * a[i] + b[i] * b[i];
      intensity[i] = v;
      total += v;
    }

    // Integrate over the ten detector boxes; normalising by total output power
    // makes the readout scale-invariant (models.py:D2NN.forward).
    const inv = 1 / Math.max(total, 1e-12);
    const logits = new Float64Array(10);
    const fractions = new Float64Array(10);
    let pred = 0;
    for (let c = 0; c < W.regions.length; c++) {
      const r = W.regions[c];
      let s = 0;
      for (let y = r[0]; y < r[1]; y++) {
        for (let x = r[2]; x < r[3]; x++) s += intensity[y * N + x];
      }
      fractions[c] = s * inv;
      logits[c] = fractions[c] * W.readout_gain;
      if (logits[c] > logits[pred]) pred = c;
    }
    return { logits, fractions, pred, intensity, planes };
  }

  /**
   * The same optical path, sampled *between* the mask planes -- display only.
   *
   * Each of the n_layers+1 hops is walked in `subSteps` equal sub-hops, so the
   * returned list is the field's intensity at every intermediate depth as well as
   * at the planes themselves. This is exact, not interpolated: the transfer
   * function composes as H(z1)*H(z2) = H(z1+z2), and the Matsushima band limit --
   * the one z-dependent term -- is inactive while z < z_crit = N*dx^2/lam
   * (15.40 mm here, against 3 mm hops), so it multiplies by 1 either way. See
   * tests/test_propagate.py:test_substep_propagation_equals_one_hop.
   *
   * `classify()` deliberately does NOT use this. The prediction must keep coming
   * from the canonical n_layers+1 propagations that the PyTorch cross-check pins;
   * these slices are for drawing and can never move a class score.
   *
   * Cost is (n_layers+1)*subSteps propagations -- ~4x the forward pass at
   * subSteps=4 -- so callers should run it on digit change, not per pointer move.
   *
   * Returns [{z, I}] ordered by depth, starting at the entrance plane (z=0).
   */
  function sliceForward(re, im, subSteps) {
    const S = Math.max(1, subSteps | 0);
    const dz = W.separation / S;
    const Hs = ASM.buildTransfer(N, W.dx, W.wavelength, dz);

    let a = Float64Array.from(re), b = Float64Array.from(im);
    const intensityOf = (x, y) => {
      const I = new Float64Array(N * N);
      for (let i = 0; i < I.length; i++) I[i] = x[i] * x[i] + y[i] * y[i];
      return I;
    };

    const out = [{ z: 0, I: intensityOf(a, b) }];
    for (let L = 0; L <= W.n_layers; L++) {
      for (let s = 0; s < S; s++) {
        const p = ASM.propagateField(a, b, N, Hs[0], Hs[1]);
        a = p[0]; b = p[1];
        out.push({ z: (L * S + s + 1) * dz, I: intensityOf(a, b) });
      }
      if (L === W.n_layers) break;
      // A pure phase mask leaves |E| alone, so the slice just recorded at this
      // plane is the arriving *and* departing intensity -- no duplicate needed.
      const off = L * N * N;
      for (let i = 0; i < N * N; i++) {
        const c = MASK_COS[off + i], s2 = MASK_SIN[off + i];
        const x = a[i], y = b[i];
        a[i] = x * c - y * s2;
        b[i] = x * s2 + y * c;
      }
    }
    return out;
  }

  /** Trained phase of mask L as an N*N Float32Array in radians (for display). */
  function maskPhase(L) {
    return MASKS.subarray(L * N * N, (L + 1) * N * N);
  }

  /** 28x28 digit in [0,1] -> full result, including the input map for display. */
  function classify(img28) {
    const canvas = preprocess(img28);
    const field = encodeInput(canvas);
    const res = forward(field[0], field[1]);
    res.canvas = canvas;
    return res;
  }

  /** The k-th frozen gallery digit as a 28x28 Float64Array in [0,1]. */
  function galleryDigit(k) {
    const out = new Float64Array(DIGIT * DIGIT);
    const off = k * DIGIT * DIGIT;
    for (let i = 0; i < out.length; i++) out[i] = GALLERY[off + i] / 255;
    return out;
  }

  const NET = {
    N, WIN, OFF, DIGIT, weights: W,
    nGallery: N_GALLERY,
    galleryDigit,
    galleryLabel: (k) => W.gallery_labels[k],
    resize, resampleRect, preprocess, normalizeDrawn, encodeInput, forward, classify,
    sliceForward, maskPhase,
  };

  if (typeof module !== "undefined" && module.exports) module.exports = NET;
  if (typeof window !== "undefined") window.PhotonnD2NN_Net = NET;
})();
