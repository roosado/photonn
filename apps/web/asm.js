/*
 * asm.js -- browser-side angular-spectrum propagation.
 *
 * A faithful JavaScript port of photonn/propagate.py:angular_spectrum, written so
 * the diffraction explorer can recompute scalar diffraction *live* in the browser
 * (every control continuous), rather than flipping between precomputed frames.
 *
 * This is the single source of truth for the browser physics: it is inlined into
 * the standalone explorer and the project explainer page, and it is loaded under
 * Node by tests/test_asm_crosscheck.py, which asserts it matches the NumPy
 * reference to < 1e-6 on a set of configs.
 *
 * Conventions match propagate.py exactly:
 *   - fields are square n x n, stored row-major (index = iy*n + ix), centre-origin
 *     (the origin sample sits at index n//2 along each axis);
 *   - all lengths in metres, wavelength in metres, spatial frequencies in cycles/m;
 *   - grid sizes are powers of two (64/128/256), so fftshift == ifftshift (a roll
 *     by n/2) -- see shift2 below.
 *
 * References: Goodman, Introduction to Fourier Optics, 3rd ed., ch. 3; Matsushima
 * & Shimobaba, IEEE TIP 18(11):2646 (2009) for the band limit; Voelz,
 * Computational Fourier Optics, ch. 5 for the sampling / critical distance.
 */
(function () {
  "use strict";

  // ------------------------------------------------------------------ 1-D FFT
  // Iterative in-place radix-2 Cooley-Tukey. Forward uses exp(-2pi i k n / N)
  // (NumPy's sign convention); inverse uses +2pi and divides by N, so a 2-D
  // inverse divides by n twice = 1/n^2, matching np.fft.ifft2.
  function fft1d(re, im, inverse) {
    const n = re.length;

    // bit-reversal permutation
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        const tr = re[i]; re[i] = re[j]; re[j] = tr;
        const ti = im[i]; im[i] = im[j]; im[j] = ti;
      }
    }

    for (let len = 2; len <= n; len <<= 1) {
      const ang = (inverse ? 2.0 : -2.0) * Math.PI / len;
      const wlenRe = Math.cos(ang);
      const wlenIm = Math.sin(ang);
      const half = len >> 1;
      for (let i = 0; i < n; i += len) {
        let wRe = 1.0;
        let wIm = 0.0;
        for (let k = 0; k < half; k++) {
          const aRe = re[i + k];
          const aIm = im[i + k];
          const bRe = re[i + k + half];
          const bIm = im[i + k + half];
          const vRe = bRe * wRe - bIm * wIm;
          const vIm = bRe * wIm + bIm * wRe;
          re[i + k] = aRe + vRe;
          im[i + k] = aIm + vIm;
          re[i + k + half] = aRe - vRe;
          im[i + k + half] = aIm - vIm;
          const nwRe = wRe * wlenRe - wIm * wlenIm;
          wIm = wRe * wlenIm + wIm * wlenRe;
          wRe = nwRe;
        }
      }
    }

    if (inverse) {
      for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
    }
  }

  // ------------------------------------------------------------------ 2-D FFT
  // Row transforms then column transforms, on flat (n*n) real/imag arrays.
  function fft2(re, im, n, inverse) {
    const tr = new Float64Array(n);
    const ti = new Float64Array(n);

    for (let iy = 0; iy < n; iy++) {
      const off = iy * n;
      for (let ix = 0; ix < n; ix++) { tr[ix] = re[off + ix]; ti[ix] = im[off + ix]; }
      fft1d(tr, ti, inverse);
      for (let ix = 0; ix < n; ix++) { re[off + ix] = tr[ix]; im[off + ix] = ti[ix]; }
    }
    for (let ix = 0; ix < n; ix++) {
      for (let iy = 0; iy < n; iy++) { tr[iy] = re[iy * n + ix]; ti[iy] = im[iy * n + ix]; }
      fft1d(tr, ti, inverse);
      for (let iy = 0; iy < n; iy++) { re[iy * n + ix] = tr[iy]; im[iy * n + ix] = ti[iy]; }
    }
  }

  // fftshift / ifftshift for even n are identical: a circular roll by n/2 on each
  // axis (since 2*(n/2) = n). Returns fresh arrays.
  function shift2(re, im, n) {
    const h = n >> 1;
    const dre = new Float64Array(n * n);
    const dim = new Float64Array(n * n);
    for (let iy = 0; iy < n; iy++) {
      const sy = (iy + h) % n;
      for (let ix = 0; ix < n; ix++) {
        const sx = (ix + h) % n;
        dre[iy * n + ix] = re[sy * n + sx];
        dim[iy * n + ix] = im[sy * n + sx];
      }
    }
    return [dre, dim];
  }

  // -------------------------------------------------------------- frequencies
  // np.fft.fftfreq(n, dx) in cycles/m, unshifted order: [0,1,..,n/2-1,-n/2,..,-1]/(n*dx).
  function fftfreq(n, dx) {
    const f = new Float64Array(n);
    const inv = 1.0 / (n * dx);
    const half = n >> 1;
    for (let i = 0; i < n; i++) f[i] = (i < half ? i : i - n) * inv;
    return f;
  }

  // ------------------------------------------------------------ transfer fn H
  // Mirrors propagate.py:angular_spectrum_transfer (lines 79-87), in unshifted
  // order so it multiplies fft2(ifftshift(E)) directly.
  //   H = exp(i 2pi z sqrt(1/lam^2 - fx^2 - fy^2))   (complex sqrt -> evanescent)
  // then band-limited to |fx|,|fy| <= u_limit = 1/(lam*sqrt((2*du*z)^2 + 1)).
  function buildTransfer(n, dx, lambda, z) {
    const f = fftfreq(n, dx);
    const hre = new Float64Array(n * n);
    const him = new Float64Array(n * n);
    const invLam2 = 1.0 / (lambda * lambda);
    const du = 1.0 / (n * dx);
    const uLimit = 1.0 / (lambda * Math.sqrt((2.0 * du * z) * (2.0 * du * z) + 1.0));
    const twoPiZ = 2.0 * Math.PI * z;

    for (let iy = 0; iy < n; iy++) {
      const fy = f[iy];
      const fy2 = fy * fy;
      const yBand = Math.abs(fy) <= uLimit;
      for (let ix = 0; ix < n; ix++) {
        const idx = iy * n + ix;
        const fx = f[ix];
        if (!yBand || Math.abs(fx) > uLimit) { hre[idx] = 0.0; him[idx] = 0.0; continue; }
        const arg = invLam2 - fx * fx - fy2;
        if (arg >= 0.0) {
          const theta = twoPiZ * Math.sqrt(arg);       // propagating: unit modulus
          hre[idx] = Math.cos(theta);
          him[idx] = Math.sin(theta);
        } else {
          hre[idx] = Math.exp(-twoPiZ * Math.sqrt(-arg)); // evanescent: real decay (z>0)
          him[idx] = 0.0;
        }
      }
    }
    return [hre, him];
  }

  // ---------------------------------------------------------------- aperture
  // Uniform plane wave clipped by a hard aperture, centred at n//2. Mirrors
  // elements.aperture: `size` is the full width (diameter for circular, side for
  // square). Returns [re, im] with im all zero.
  function aperture(n, dx, shape, size) {
    const re = new Float64Array(n * n);
    const im = new Float64Array(n * n);
    const c = n >> 1;
    if (shape === "square") {
      const half = size / 2.0;
      for (let iy = 0; iy < n; iy++) {
        const y = (iy - c) * dx;
        const inY = Math.abs(y) <= half;
        for (let ix = 0; ix < n; ix++) {
          const x = (ix - c) * dx;
          re[iy * n + ix] = (inY && Math.abs(x) <= half) ? 1.0 : 0.0;
        }
      }
    } else { // circular
      const r2 = (size / 2.0) * (size / 2.0);
      for (let iy = 0; iy < n; iy++) {
        const y = (iy - c) * dx;
        const y2 = y * y;
        for (let ix = 0; ix < n; ix++) {
          const x = (ix - c) * dx;
          re[iy * n + ix] = (x * x + y2 <= r2) ? 1.0 : 0.0;
        }
      }
    }
    return [re, im];
  }

  // --------------------------------------------------------------- propagator
  // One propagation step on a complex field: ifftshift -> fft2 -> xH -> ifft2 ->
  // fftshift. Returns fresh [re, im] in centre-origin layout; the inputs are not
  // modified. Mirrors photonn.layers.AngularSpectrumLayer.forward.
  //
  // H is passed in already built rather than derived from (dx, lambda, z) because
  // the caller usually propagates the same distance repeatedly -- the D2NN makes
  // six identical-z hops through its mask stack, so it builds H once and reuses it
  // (see d2nn.js). Use buildTransfer(n, dx, lambda, z) to make one.
  function propagateField(re, im, n, hre, him) {
    let [a, b] = shift2(re, im, n);        // ifftshift
    fft2(a, b, n, false);
    for (let i = 0; i < n * n; i++) {
      const x = a[i], y = b[i], c = hre[i], d = him[i];
      a[i] = x * c - y * d;
      b[i] = x * d + y * c;
    }
    fft2(a, b, n, true);
    return shift2(a, b, n);                // fftshift
  }

  // Full ASM pipeline for a hard-aperture input, then |E|^2. Returns a
  // Float64Array(n*n) of intensity in centre-origin layout, matching
  // photonn.propagate.angular_spectrum(field, z).intensity().
  function propagateIntensity(n, dx, lambda, z, shape, size) {
    const [re0, im0] = aperture(n, dx, shape, size);
    const [hre, him] = buildTransfer(n, dx, lambda, z);
    const [re, im] = propagateField(re0, im0, n, hre, him);
    const I = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) I[i] = re[i] * re[i] + im[i] * im[i];
    return I;
  }

  // ------------------------------------------------------------ sampling info
  // check_sampling criteria (propagate.py:172-188), aperture-independent.
  function zCrit(n, dx, lambda) { return n * dx * dx / lambda; }           // ASM well-sampled if |z| <= this
  function zFraunhofer(n, dx, lambda) {                                     // far field beyond this (D = n*dx)
    const D = n * dx;
    return 2.0 * D * D / lambda;
  }

  const ASM = {
    fft1d, fft2, shift2, fftfreq, buildTransfer, aperture,
    propagateField, propagateIntensity, zCrit, zFraunhofer,
  };

  if (typeof module !== "undefined" && module.exports) module.exports = ASM;
  if (typeof window !== "undefined") window.ASM = ASM;
})();
