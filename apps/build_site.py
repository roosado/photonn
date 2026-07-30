"""Build the photonn project explainer page (Part C / C2).

One self-contained HTML page telling the whole story -- a section per phase, every
figure embedded as a base64 data URI, and the Phase-1 diffraction explorer embedded
live (the same test-verified physics from apps/web/asm.js). No external requests:
CSP-safe, offline, theme-aware.

Emits two variants from one template:
  - site/index.html          -- a full standalone document (repo / GitHub Pages)
  - site/_artifact_body.html -- body-only (no <!doctype>/<html>/<head>/<body>) for
                                publishing as a claude.ai Artifact, which wraps the
                                file in its own skeleton.

Sections 6-7 (boson sampling, mesh error budget) are "Planned" placeholders: their
code/figures do not exist yet (Parts A/B of the parent plan). Fill them when those land.

Run: python -m apps.build_site
"""
from __future__ import annotations

import base64
import io
import os

from PIL import Image

from apps.d2nn_demo import d2nn_bundle, d2nn_mount
from apps.diffraction_explorer import explorer_bundle, explorer_mount

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(REPO, "site")

# figure key -> path on disk
FIGURES = {
    "phase2_masks": "docs/figures/phase2_masks.png",
    "mesh_topology": "docs/figures/phase3_mesh_topology.png",
    "tol_phase": "photonn-hw/figures/tolerance_phase.png",
    "tol_quant": "photonn-hw/figures/tolerance_quant.png",
    "tol_wavelength": "photonn-hw/figures/tolerance_wavelength.png",
    "tol_crosstalk": "photonn-hw/figures/tolerance_crosstalk.png",
    "tol_detector": "photonn-hw/figures/tolerance_detector.png",
    "tol_loss": "photonn-hw/figures/tolerance_loss.png",
    "confusion": "photonn-hw/figures/confusion_phase035.png",
    "sensitivity": "photonn-hw/figures/sensitivity_map.png",
}

DEFAULT_OPT = {"fmt": "png", "max_w": 1500}
# Per-figure encoding overrides. The two heavy figures are 83% of the page:
#  - the masks plate is a smooth colormap image -> JPEG is ~4x smaller with no
#    visible loss at display size;
#  - the mesh topology is a line/diagram figure -> keep PNG (thin lines + text),
#    just cap the width lower.
# Every other figure is a small line plot and stays PNG at the default width.
FIG_OPTS = {
    "phase2_masks": {"fmt": "jpeg", "max_w": 1400, "quality": 85},
    "mesh_topology": {"fmt": "png", "max_w": 1300},
}


def encode_figure(rel_path: str, fmt: str = "png", max_w: int = 1500, quality: int = 85) -> str:
    """Return a data URI for a figure, downscaled and flattened onto white.

    Matplotlib figures have white backgrounds, so we flatten any alpha onto white
    (keeps them readable inside a light 'plate' on either page theme) and cap the
    width to keep the self-contained page lean. ``fmt`` is ``"png"`` (lossless, for
    line plots and diagrams) or ``"jpeg"`` (for smooth colormap images, far smaller).
    """
    im = Image.open(os.path.join(REPO, rel_path))
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
    else:
        im = im.convert("RGB")
    if im.width > max_w:
        h = round(im.height * max_w / im.width)
        im = im.resize((max_w, h), Image.LANCZOS)
    buf = io.BytesIO()
    if fmt == "jpeg":
        im.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
        mime = "image/jpeg"
    else:
        im.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------- CSS
CSS = r"""
:root{
  color-scheme: light dark;
  --bg:#f4f7fb; --surface:#ffffff; --surface-2:#eef2f8;
  --border:#d8e0ec; --ink:#141b26; --ink-dim:#3f4c60; --muted:#6b7789;
  --beam:#0f9e8f; --fringe:#c9701f; --good:#2f8f52; --bad:#c14a34;
  --beam-soft:rgba(15,158,143,.10);
  --spectral:linear-gradient(90deg,#39d1a0,#4fc6e6,#f2c14e,#ef7d5a,#c05aa8);
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,"Times New Roman",serif;
  --sans:ui-sans-serif,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --mono:ui-monospace,"Cascadia Code","SF Mono",Consolas,"Liberation Mono",Menlo,monospace;
  --measure:68ch;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#0a0d12; --surface:#111826; --surface-2:#0e131d;
    --border:#243149; --ink:#e8edf4; --ink-dim:#b7c2d2; --muted:#7787a0;
    --beam:#2ec9b8; --fringe:#f2994a; --good:#57c07a; --bad:#e0705f;
    --beam-soft:rgba(46,201,184,.14);
  }
}
:root[data-theme="light"]{
  --bg:#f4f7fb; --surface:#ffffff; --surface-2:#eef2f8;
  --border:#d8e0ec; --ink:#141b26; --ink-dim:#3f4c60; --muted:#6b7789;
  --beam:#0f9e8f; --fringe:#c9701f; --good:#2f8f52; --bad:#c14a34; --beam-soft:rgba(15,158,143,.10);
}
:root[data-theme="dark"]{
  --bg:#0a0d12; --surface:#111826; --surface-2:#0e131d;
  --border:#243149; --ink:#e8edf4; --ink-dim:#b7c2d2; --muted:#7787a0;
  --beam:#2ec9b8; --fringe:#f2994a; --good:#57c07a; --bad:#e0705f; --beam-soft:rgba(46,201,184,.14);
}

*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:var(--sans);font-size:17px;line-height:1.65;
  -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility;}
.spectral-rule{height:3px;background:var(--spectral);}

.topbar{position:sticky;top:0;z-index:20;backdrop-filter:blur(8px);
  background:color-mix(in srgb,var(--bg) 82%,transparent);
  border-bottom:1px solid var(--border);}
.topbar-in{max-width:1120px;margin:0 auto;padding:10px 24px;
  display:flex;align-items:center;justify-content:space-between;gap:16px;}
.brand{font-family:var(--mono);font-size:.82rem;letter-spacing:.02em;color:var(--ink-dim);}
.brand b{color:var(--beam);font-weight:600;}
.theme-toggle{font-family:var(--mono);font-size:.74rem;letter-spacing:.04em;
  background:transparent;border:1px solid var(--border);color:var(--muted);
  border-radius:999px;padding:5px 13px;cursor:pointer;transition:color .15s,border-color .15s;}
.theme-toggle:hover{color:var(--ink);border-color:var(--beam);}
.theme-toggle:focus-visible{outline:2px solid var(--beam);outline-offset:2px;}

.wrap{max-width:1120px;margin:0 auto;padding:0 24px;}
.col{max-width:var(--measure);}
.eyebrow{font-family:var(--mono);font-size:.72rem;letter-spacing:.2em;
  text-transform:uppercase;color:var(--beam);margin:0 0 .5rem;}

/* hero */
.hero{padding:76px 0 30px;}
.hero h1{font-family:var(--serif);font-weight:600;text-wrap:balance;
  font-size:clamp(2.1rem,4.6vw,3.35rem);line-height:1.08;letter-spacing:-.01em;
  margin:.2rem 0 .1rem;}
.hero h1 em{font-style:italic;color:var(--fringe);}
.hero .underbar{width:132px;height:3px;background:var(--spectral);margin:20px 0 22px;border-radius:2px;}
.standfirst{font-size:1.16rem;color:var(--ink-dim);max-width:60ch;}
.standfirst b{color:var(--ink);font-weight:600;}

.stat-strip{display:flex;flex-wrap:wrap;gap:14px;margin:34px 0 8px;}
.stat{flex:1 1 150px;background:var(--surface);border:1px solid var(--border);
  border-radius:12px;padding:15px 17px;}
.stat .v{font-family:var(--mono);font-size:1.5rem;font-weight:600;color:var(--ink);
  font-variant-numeric:tabular-nums;display:block;letter-spacing:-.01em;}
.stat .v small{font-size:.9rem;color:var(--muted);font-weight:500;}
.stat .l{font-size:.8rem;color:var(--muted);display:block;margin-top:3px;line-height:1.35;}

/* phase sections */
.phase{padding:52px 0;border-top:1px solid var(--border);}
.phase-head{display:flex;gap:20px;align-items:flex-start;margin-bottom:10px;}
.ph-num{font-family:var(--mono);font-size:.95rem;font-weight:600;color:var(--beam);
  border:1px solid var(--border);border-radius:9px;padding:7px 11px;line-height:1;
  background:var(--beam-soft);white-space:nowrap;margin-top:4px;}
.ph-num.next{color:var(--muted);background:transparent;border-style:dashed;}
.phase-head h2{font-family:var(--serif);font-weight:600;text-wrap:balance;
  font-size:clamp(1.55rem,2.8vw,2.05rem);line-height:1.14;margin:.1rem 0 0;}
.prose{color:var(--ink-dim);}
.prose p{margin:.85rem 0;max-width:var(--measure);}
.prose strong{color:var(--ink);font-weight:600;}
.q{font-family:var(--mono);font-size:.92em;background:var(--surface-2);
  border:1px solid var(--border);border-radius:5px;padding:.05em .38em;color:var(--ink);
  white-space:nowrap;}
a.link{color:var(--beam);text-decoration:none;border-bottom:1px solid color-mix(in srgb,var(--beam) 40%,transparent);}
a.link:hover{border-bottom-color:var(--beam);}

.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:24px 0;}
.stats .s{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:13px 15px;}
.stats .s .v{font-family:var(--mono);font-size:1.15rem;font-weight:600;color:var(--ink);
  font-variant-numeric:tabular-nums;}
.stats .s .l{font-size:.76rem;color:var(--muted);margin-top:2px;line-height:1.35;}

/* figure plates -- always light, since the figures are light-background */
.plate{background:#fff;border:1px solid var(--border);border-radius:12px;
  padding:12px;margin:26px 0;overflow-x:auto;}
.plate img{display:block;width:100%;height:auto;border-radius:6px;}
.plate figcaption{font-family:var(--sans);font-size:.83rem;color:#5b6675;
  margin-top:10px;padding:0 4px;line-height:1.5;}
.plate figcaption .fign{font-family:var(--mono);color:var(--beam);font-weight:600;
  letter-spacing:.02em;margin-right:.5em;}
.plate-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));
  gap:16px;margin:26px 0;}
.plate-grid .plate{margin:0;}

/* finding callout */
.finding{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--bad);
  border-radius:10px;padding:18px 20px;margin:26px 0;}
.finding .tag{font-family:var(--mono);font-size:.7rem;letter-spacing:.16em;text-transform:uppercase;
  color:var(--bad);margin:0 0 .4rem;font-weight:600;}
.finding p.body{margin:0;color:var(--ink);}
.finding p.body strong{color:var(--ink);}

/* explorer host */
.explorer-band{background:var(--surface-2);border:1px solid var(--border);border-radius:16px;
  padding:22px;margin:28px 0;}
.explorer-band .cap{font-family:var(--mono);font-size:.76rem;letter-spacing:.05em;
  color:var(--muted);margin:14px 0 0;}
.pe-host .pe-root{--pe-fg:var(--ink);--pe-muted:var(--muted);--pe-panel:var(--surface);
  --pe-border:var(--border);--pe-accent:var(--beam);--pe-ok:var(--good);--pe-warn:var(--bad);}

/* planned cards */
.planned{opacity:.96;}
.planned .phase-head h2{color:var(--ink-dim);}
.badge-next{font-family:var(--mono);font-size:.68rem;letter-spacing:.14em;text-transform:uppercase;
  color:var(--muted);border:1px dashed var(--border);border-radius:999px;padding:3px 10px;
  display:inline-block;margin-bottom:6px;}

/* footer */
footer{border-top:1px solid var(--border);margin-top:20px;padding:44px 0 70px;}
.foot-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:26px;}
footer h3{font-family:var(--mono);font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;
  color:var(--beam);margin:0 0 .6rem;font-weight:600;}
footer p{margin:.3rem 0;color:var(--ink-dim);font-size:.92rem;max-width:42ch;}
footer .colophon{margin-top:30px;font-family:var(--mono);font-size:.76rem;color:var(--muted);
  border-top:1px solid var(--border);padding-top:18px;}

/* motion */
.reveal{opacity:0;transform:translateY(14px);transition:opacity .6s ease,transform .6s ease;}
.reveal.in{opacity:1;transform:none;}
@media (prefers-reduced-motion:reduce){.reveal{opacity:1;transform:none;transition:none;}}

@media (max-width:640px){
  .phase-head{gap:13px;}
  .hero{padding:52px 0 22px;}
}
"""

# --------------------------------------------------------------------------- BODY
# Placeholder tokens (@@...@@) are substituted in render(); avoids CSS/JS brace escaping.
BODY = r"""
<div class="spectral-rule"></div>
<header class="topbar">
  <div class="topbar-in">
    <span class="brand"><b>photonn</b> · a fabrication-tolerance study of optical neural networks</span>
    <button class="theme-toggle" id="themeToggle" aria-label="Toggle colour theme">◐ theme</button>
  </div>
</header>

<main class="wrap">

  <section class="hero col reveal">
    <p class="eyebrow">Physical photonic neural network simulator</p>
    <h1>How precisely must a photonic chip be built before it stops computing what it was <em>trained</em> to compute?</h1>
    <div class="underbar"></div>
    <p class="standfirst">A neural network can be made of light: a digit is encoded into a coherent
    field, the field is shaped by fabricated optics, and the answer is read as where the light lands.
    This project trains such networks in an <b>ideal</b> simulation, then asks the only question that
    decides whether they could be built &mdash; <b>how much fabrication error they survive.</b></p>
    <div class="stat-strip">
      <div class="stat"><span class="v">0.770</span><span class="l">diffractive net · MNIST test accuracy</span></div>
      <div class="stat"><span class="v">31&times;</span><span class="l">fewer parameters in the interferometer mesh, at 0.736</span></div>
      <div class="stat"><span class="v">&lt;0.35<small> px</small></span><span class="l">crosstalk blur the design tolerates &mdash; the binding constraint</span></div>
    </div>
  </section>

  <section class="phase reveal">
    <div class="phase-head col"><span class="ph-num">01</span>
      <div><p class="eyebrow">Phase one &middot; Wave optics</p>
      <h2>Everything rests on propagating a light field through space</h2></div></div>
    <div class="prose col">
      <p>Before anything learns, we need to move coherent light from one plane to another,
      exactly. The <strong>angular-spectrum method</strong> does it without approximation up to the
      grid&rsquo;s band limit: Fourier-transform the field, multiply by the transfer function
      <span class="q">H = exp(i&thinsp;2&pi;z&thinsp;&radic;(1/&lambda;&sup2; &minus; f&sup2;))</span>,
      transform back. Evanescent components decay; a band limit keeps the result alias-free into
      the far field. A single criterion, <span class="q">z_crit = N&middot;dx&sup2;/&lambda;</span>,
      marks where the grid can no longer represent the propagating field &mdash; and it is enforced
      at runtime, not just in tests. Against an analytic Gaussian beam the method agrees to
      <strong>~2&times;10<sup>&minus;8</sup></strong>.</p>
    </div>
    <div class="explorer-band reveal">
      <div class="pe-host"><div id="explorer"></div></div>
      <p class="cap">Live &mdash; the diffraction runs on a hand-written FFT in your browser, the same
      physics as the Python reference (cross-checked to &lt;10<sup>&minus;6</sup>). Move any control;
      the sampling flag turns amber the moment <span style="white-space:nowrap">z &gt; z_crit</span>.</p>
    </div>
  </section>

  <section class="phase reveal">
    <div class="phase-head col"><span class="ph-num">02</span>
      <div><p class="eyebrow">Phase two &middot; Diffractive network (D&sup2;NN)</p>
      <h2>A stack of phase masks, trained by light</h2></div></div>
    <div class="prose col">
      <p>Place five <strong>trainable phase masks</strong> between propagation steps. Encode a digit
      into the input field&rsquo;s amplitude and phase, let it diffract through the stack, and read the
      class by integrating intensity over ten detector zones. The masks are trained by
      back-propagating through the propagator itself &mdash; the optics <em>is</em> the network.</p>
      <p>It reaches <strong>0.770</strong> on MNIST (chance is 0.10) with 81,920 phase values.
      The optical power budget is real: at 1&nbsp;mW for 1&nbsp;ms, <strong>2.68&times;10<sup>12</sup></strong>
      photons enter and <strong>58%</strong> land inside detector regions &mdash; so shot noise on the
      winning class is ~10<sup>&minus;6</sup>, negligible.</p>
      <p>And its ceiling is honest: the whole stack, masks and all, is <strong>one linear operator
      followed by a single intensity readout.</strong> With no optical nonlinearity, expressivity is
      capped &mdash; a limitation this project characterises rather than engineers around.</p>
    </div>
    <div class="stats col">
      <div class="s"><div class="v">0.770</div><div class="l">test accuracy (chance 0.10)</div></div>
      <div class="s"><div class="v">5 masks</div><div class="l">81,920 trainable phases</div></div>
      <div class="s"><div class="v">532 nm</div><div class="l">N=128, dx=8&micro;m, 3&nbsp;mm gaps</div></div>
      <div class="s"><div class="v">58%</div><div class="l">of input photons captured</div></div>
    </div>
    <figure class="plate reveal">
      <img src="@@FIG_phase2_masks@@" alt="Five trained phase masks and one input-to-output intensity example for the diffractive network" loading="lazy">
      <figcaption><span class="fign">Fig 1</span>The five learned phase masks (top) and one worked
      example: an input digit field diffracting to its detector plane (bottom). Each mask is a fabricated
      surface relief; training only ever adjusted these phase profiles.</figcaption>
    </figure>
    <div class="explorer-band reveal">
      <div class="pe-host"><div id="d2nn"></div></div>
      <p class="cap">Live &mdash; the trained network itself, running in your browser. Pick a digit
      from the frozen test set or draw one; the five masks and the propagation between them are the
      exported parameters, and the ten detector boxes are where the class is read. It is right about
      77% of the time, so the gallery includes digits it gets wrong. A drawing is further out of
      distribution than an MNIST test digit &mdash; expect it to struggle more there.</p>
    </div>
  </section>

  <section class="phase reveal">
    <div class="phase-head col"><span class="ph-num">03</span>
      <div><p class="eyebrow">Phase three &middot; MZI mesh</p>
      <h2>The same linear algebra, built from interferometers</h2></div></div>
    <div class="prose col">
      <p>A different optical computer: a mesh of <strong>Mach&ndash;Zehnder interferometers</strong>,
      two phase shifters each, tiled in a Clements rectangle. Such a mesh can realise <em>any</em>
      unitary; place a diagonal of amplitudes between two meshes and you have any real matrix by its
      singular-value decomposition, <span class="q">U&middot;&Sigma;&middot;V&dagger;</span>. The
      decompositions reconstruct random unitaries to <strong>~10<sup>&minus;15</sup></strong>.</p>
      <p>Trained on the same task it scores <strong>0.736</strong> &mdash; just shy of the diffractive
      net&rsquo;s 0.770, but with <strong>2,628 parameters against 81,920, about 31&times; fewer.</strong>
      The gap traces to input downsampling (MNIST compressed to 6&times;6 = 36 modes), a
      footprint&ndash;versus&ndash;input&#8209;dimensionality trade, not a modelling failure. Several of
      its singular values exceed 1 &mdash; physically that needs gain, a documented finding.</p>
    </div>
    <div class="stats col">
      <div class="s"><div class="v">0.736</div><div class="l">accuracy vs 0.770 (D&sup2;NN)</div></div>
      <div class="s"><div class="v">2,628</div><div class="l">parameters &middot; ~31&times; fewer</div></div>
      <div class="s"><div class="v">36 modes</div><div class="l">72 serial MZI layers</div></div>
      <div class="s"><div class="v">1e&minus;15</div><div class="l">Clements/Reck reconstruction error</div></div>
    </div>
    <figure class="plate reveal">
      <img src="@@FIG_mesh_topology@@" alt="MZI mesh topology and the learned singular-value spectrum" loading="lazy">
      <figcaption><span class="fign">Fig 2</span>Left: the rectangular MZI mesh topology. Right: the
      learned singular-value spectrum &mdash; effectively low-rank, only ~15&ndash;20 of 36 values carry
      weight, which is why so few parameters suffice.</figcaption>
    </figure>
  </section>

  <section class="phase reveal">
    <div class="phase-head col"><span class="ph-num">04</span>
      <div><p class="eyebrow">Phase four &middot; Error budget</p>
      <h2>Now break it on purpose &mdash; the central question, answered</h2></div></div>
    <div class="prose col">
      <p>The trained parameters cross a one-directional handoff into a MATLAB &ldquo;as-built&rdquo;
      model, which re-scores the exact same network under each fabrication imperfection &mdash; phase
      error, DAC quantisation, coupler imbalance, loss, wavelength drift, thermal crosstalk, detector
      noise &mdash; Monte Carlo over realisations, <strong>every magnitude traced to a published
      measurement.</strong> With zero error injected it reproduces the ideal 0.7695 baseline exactly,
      so any drop below is fabrication, nothing else.</p>
    </div>
    <div class="finding col reveal">
      <p class="tag">Binding constraint</p>
      <p class="body"><strong>Thermal / pixel crosstalk sets the tolerance.</strong> Accuracy holds
      only while the phase blur between neighbouring pixels stays below <strong>~0.35&nbsp;px</strong>
      &mdash; and a real ~1&nbsp;px LCoS spatial light modulator does not clear it. The ranking is
      unambiguous: crosstalk &raquo; phase error &gt; detector power &asymp; loss &raquo; wavelength
      &asymp; quantisation. To hold 95% of ideal (accuracy &ge; 0.731): phase &sigma; &lt; 0.35&nbsp;rad
      (&lambda;/18), &ge;3-bit DAC, drift &lt; 15&nbsp;nm, detector &gt; 0.1&nbsp;pW. The number that
      decides feasibility is not precision or bit depth &mdash; it is how sharply each pixel&rsquo;s
      phase is confined against its neighbours.</p>
    </div>
    <div class="plate-grid reveal">
      <figure class="plate"><img src="@@FIG_tol_crosstalk@@" alt="Accuracy vs thermal/pixel crosstalk" loading="lazy"><figcaption><span class="fign">Fig 3</span>Crosstalk &mdash; the binding constraint.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_tol_phase@@" alt="Accuracy vs per-pixel phase error" loading="lazy"><figcaption><span class="fign">Fig 4</span>Per-pixel phase-setting error.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_tol_detector@@" alt="Accuracy vs detector noise / input power" loading="lazy"><figcaption><span class="fign">Fig 5</span>Detector &amp; shot noise vs input power.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_tol_loss@@" alt="Accuracy vs optical insertion loss" loading="lazy"><figcaption><span class="fign">Fig 6</span>Optical insertion loss.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_tol_wavelength@@" alt="Accuracy vs wavelength drift" loading="lazy"><figcaption><span class="fign">Fig 7</span>Laser wavelength drift.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_tol_quant@@" alt="Accuracy vs DAC bit resolution" loading="lazy"><figcaption><span class="fign">Fig 8</span>DAC / SLM bit resolution.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_confusion@@" alt="As-built confusion matrix at phase sigma 0.35 rad" loading="lazy"><figcaption><span class="fign">Fig 9</span>As-built confusion matrix at &sigma;=0.35&nbsp;rad.</figcaption></figure>
    </div>
    <figure class="plate reveal">
      <img src="@@FIG_sensitivity@@" alt="Per-mask spatial sensitivity map" loading="lazy">
      <figcaption><span class="fign">Fig 10</span>Spatial sensitivity &mdash; where on each mask a phase
      error costs the most accuracy. Sensitivity is not uniform, which is what makes a per-pixel
      tolerance meaningful.</figcaption>
    </figure>
  </section>

  <section class="phase planned reveal">
    <div class="phase-head col"><span class="ph-num next">next</span>
      <div><span class="badge-next">Planned &middot; Part B</span>
      <p class="eyebrow">Quantum branch &middot; Boson sampling</p>
      <h2>Same mesh, single photons instead of a beam</h2></div></div>
    <div class="prose col">
      <p>The interferometer mesh has a second life. Send <strong>indistinguishable single photons</strong>
      through the very same trained unitary and the output statistics become permanent-based rather than
      intensity-based &mdash; the regime behind boson sampling. The planned deliverable computes those
      distributions, shows the <strong>Hong&ndash;Ou&ndash;Mandel dip</strong> (two photons on a 50:50
      coupler never leave separately), and contrasts the quantum output with the classical,
      distinguishable-particle case. Only the input state&rsquo;s statistics change; the transfer matrix
      is identical to Phase&nbsp;3.</p>
    </div>
  </section>

  <section class="phase planned reveal">
    <div class="phase-head col"><span class="ph-num next">next</span>
      <div><span class="badge-next">Planned &middot; Part A</span>
      <p class="eyebrow">Error budget &middot; MZI mesh</p>
      <h2>Fabrication tolerance for the interferometer mesh</h2></div></div>
    <div class="prose col">
      <p>Phase&nbsp;4&rsquo;s framework will extend to the mesh, reactivating the two error sources unique
      to it: <strong>coupler imbalance</strong> (deviation from 50:50) and <strong>per-MZI insertion
      loss</strong>, which makes the transfer sub-unitary &mdash; a real effect here, unlike the
      normalised diffractive readout. The interesting comparison is the failure mode: <strong>serial
      phase-error accumulation down 72 MZI layers</strong> versus the diffractive net&rsquo;s
      crosstalk-dominated per-pixel budget. Two optical computers, two different ways to lose the
      computation to fabrication.</p>
    </div>
  </section>

  <footer class="reveal">
    <div class="foot-grid">
      <div><h3>The boundary</h3><p>Python designs the ideal network; a single HDF5 file crosses to
        MATLAB, which models the as-built chip and never writes back. Design versus as-built, enforced
        by a one-directional handoff.</p></div>
      <div><h3>Scope</h3><p>Scalar diffraction only. Idealised components parameterised by
        literature-sourced values. One task &mdash; MNIST &mdash; reused across every phase so the
        results stay comparable.</p></div>
      <div><h3>Built with</h3><p>NumPy &middot; SciPy &middot; PyTorch &middot; Matplotlib on the design
        side; MATLAB App Designer on the as-built side. This page&rsquo;s diffraction runs on a
        ~200-line hand-written FFT &mdash; no libraries, no network.</p></div>
    </div>
    <p class="colophon">photonn &mdash; a portfolio study in optical computing and fabrication tolerance.
    Every physical constant on this page is cited in the source; unsourced values are flagged, never invented.</p>
  </footer>

</main>

<script>
(function(){
  var root=document.documentElement, btn=document.getElementById('themeToggle');
  try{var saved=localStorage.getItem('photonn-theme'); if(saved){root.setAttribute('data-theme',saved);}}catch(e){}
  function cur(){var a=root.getAttribute('data-theme'); if(a) return a;
    return window.matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light';}
  btn.addEventListener('click',function(){var n=cur()==='dark'?'light':'dark';
    root.setAttribute('data-theme',n); try{localStorage.setItem('photonn-theme',n);}catch(e){}});
})();
(function(){
  var els=document.querySelectorAll('.reveal');
  if(!('IntersectionObserver' in window)||window.matchMedia('(prefers-reduced-motion:reduce)').matches){
    els.forEach(function(el){el.classList.add('in');}); return;}
  var io=new IntersectionObserver(function(es){es.forEach(function(e){
    if(e.isIntersecting){e.target.classList.add('in'); io.unobserve(e.target);}});},{rootMargin:'0px 0px -8% 0px'});
  els.forEach(function(el){io.observe(el);});
})();
</script>
@@EXPLORER_BUNDLE@@
@@EXPLORER_MOUNT@@
@@D2NN_BUNDLE@@
@@D2NN_MOUNT@@
"""

HEAD_META = (
    '<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    '<meta name="description" content="How precisely must a photonic neural network be fabricated '
    'before it stops computing what it was trained to compute? An interactive study.">\n'
    "<title>photonn &mdash; photonic neural networks &amp; fabrication tolerance</title>"
)


def render():
    """Return (full_document_html, artifact_body_html)."""
    body = BODY
    for key in FIGURES:
        opt = {**DEFAULT_OPT, **FIG_OPTS.get(key, {})}
        body = body.replace(f"@@FIG_{key}@@", encode_figure(FIGURES[key], **opt))
    body = body.replace("@@EXPLORER_BUNDLE@@", explorer_bundle())
    body = body.replace("@@EXPLORER_MOUNT@@", explorer_mount("explorer"))
    # asm.js is already inlined by the explorer bundle above; the D2NN bundle
    # reuses that same window.ASM rather than shipping a second copy.
    body = body.replace("@@D2NN_BUNDLE@@", d2nn_bundle(include_asm=False))
    body = body.replace("@@D2NN_MOUNT@@", d2nn_mount("d2nn"))

    full = (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        + HEAD_META
        + "\n<style>\n" + CSS + "\n</style>\n</head>\n<body>\n"
        + body
        + "\n</body>\n</html>\n"
    )
    artifact_body = "<style>\n" + CSS + "\n</style>\n" + body
    return full, artifact_body


def main():
    os.makedirs(SITE, exist_ok=True)
    full, artifact_body = render()
    with open(os.path.join(SITE, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(full)
    with open(os.path.join(SITE, "_artifact_body.html"), "w", encoding="utf-8") as fh:
        fh.write(artifact_body)
    print(f"wrote {os.path.join(SITE, 'index.html')} ({len(full)//1024} KB)")
    print(f"wrote {os.path.join(SITE, '_artifact_body.html')} ({len(artifact_body)//1024} KB)")


if __name__ == "__main__":
    main()
