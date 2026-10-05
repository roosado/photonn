/*
 * mesh.js -- the MZI mesh, built in the browser. Pure computation, no DOM.
 *
 * Two widgets draw a trained chip: the coupler-imbalance widget on /tolerance
 * (errors.js, KINDS.mesh) and the two-layer activated chip on /activation
 * (activation.js). Both need the same four things -- the 16-bit code decoder, the
 * Clements schedule, the factored 2x2 MZI block, and a mesh built column by
 * column -- and two copies of them would be the pattern ffe6311 spent a commit
 * removing from the canvas code. So they live here, once, like asm.js does for
 * the propagator.
 *
 * Moved out of errors.js unchanged. The extraction was held to the operator the
 * old code built: tests/mesh_operator_runner.js dumped |U.diag(sigma).V| at
 * epsilon = 0 and 0.01 before and after, and the two dumps are byte-identical.
 *
 * Every page that inlines this must inline it before the widget that reads it;
 * build_site.py's script_tags order is the guarantee and test_site_widgets checks
 * the built page.
 *
 * Usage:  window.PhotonnMesh.{decode16, schedule, mziBlock, meshMatrix, ...}
 */
(function () {
  "use strict";

  /** Base64 to bytes, without TextDecoder (absent under the Node runners). */
  function b64ToBytes(b64) {
    const bin = (typeof atob === "function")
      ? atob(b64)
      : Buffer.from(b64, "base64").toString("binary");
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

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

  /**
   * One SVD layer as a single complex matrix: U . diag(sigma) . V, V as stored.
   *
   * The convention photonn.mzi.deep_mesh_forward and +meshmodel hold: no
   * conjugate transpose anywhere. Built once per layer, then applied per input.
   */
  function svdMatrix(n, sch, theta, phi, outV, outU, sigma, off) {
    const v = meshMatrix(n, sch, theta, phi, outV, off, null);
    const u = meshMatrix(n, sch, theta, phi, outU, off + sch.nMzi, null);
    const re = new Float64Array(n * n), im = new Float64Array(n * n);
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
        re[i * n + k] = ar;
        im[i * n + k] = ai;
      }
    }
    return { re: re, im: im };
  }

  /** y = M x for a complex matrix and a complex vector given as (re, im) arrays. */
  function apply(M, n, xr, xi) {
    const yr = new Float64Array(n), yi = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let ar = 0, ai = 0;
      for (let k = 0; k < n; k++) {
        const mr = M.re[i * n + k], mi = M.im[i * n + k];
        ar += mr * xr[k] - mi * xi[k];
        ai += mr * xi[k] + mi * xr[k];
      }
      yr[i] = ar;
      yi[i] = ai;
    }
    return { re: yr, im: yi };
  }

  const api = { b64ToBytes, decode16, schedule, mziBlock, meshMatrix, svdMatrix, apply };
  if (typeof window !== "undefined") window.PhotonnMesh = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})();
