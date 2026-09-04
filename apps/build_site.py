"""Build the photonn site: five self-contained HTML pages.

The site is an explainer of optical neural networks that *lands* on the project's
central question rather than opening with it. A reader meets the working machine
first, learns what it is, and only then is asked how precisely it would have to be
fabricated -- which is the answer to "why don't we already have these".

  site/index.html      -- the machine: the trained network, live, plus what it is
  site/physics.html    -- the wave optics underneath it, plus the diffraction explorer
  site/chip.html       -- the same computation built as an interferometer mesh
  site/tolerance.html  -- the fabrication error budget for both machines: the
                          study's destination, and where they stop being alike
  site/optics.html     -- live work: how much better the optics could still be
  site/_artifact_body.html -- body-only front page for publishing as a claude.ai
                              Artifact, which supplies its own <head>/<body>

Every figure is embedded as a base64 data URI and every widget is inlined, so the
pages make no external requests: CSP-safe, offline, theme-aware, and openable from
``file://``. Page weight is therefore the payload -- ``tests/test_page_budget.py``
holds the ceilings.

Navigation is generated from :data:`PAGES`: the topbar, the sequential "next"
hand-off at the foot of each page, and the relative/absolute link swap the Artifact
needs all read from that one list.

Run: python -m apps.build_site
"""
from __future__ import annotations

import base64
import functools
import html as _html
import io
import json
import os
import re
from typing import NamedTuple

from PIL import Image

from apps.compare_demo import compare_bundle, compare_mount
from apps.d2nn_demo import d2nn_bundle, d2nn_mount
from apps.diffraction_explorer import (
    explorer_bundle,
    explorer_mount,
    mount_queue_bundle,
    mount_script,
    read_web_asset,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(REPO, "site")

#: Where the site is deployed. Only the Artifact variant needs it: it is a single
#: standalone body with no sibling pages, so its links must be absolute.
SITE_URL = "https://roosado.github.io/photonn/"


class Page(NamedTuple):
    """One page of the site, and everything the chrome needs to know about it."""

    key: str      # link token (@@HREF_key@@) and the stem of the filename
    file: str     # name written into site/
    nav: str      # topbar label -- short, five of these share one row
    title: str    # <title>
    head: str     # human name, used by the "next" hand-off card
    desc: str     # <meta name="description">
    blurb: str    # one sentence on the hand-off card
    #: Host container ids this page carries, in document order.
    #:
    #: Declared rather than inferred so the two can be checked against each
    #: other. A widget's presence on a page is otherwise six coordinated edits
    #: -- a host div in a body string, a token pair, a bundle function, a mount
    #: function, and two `.replace()` calls in `render()` -- with nothing tying
    #: them together. `mount_queue.js` deliberately skips a container it cannot
    #: find, so a mistyped id produced a page that silently lacked a widget and
    #: passed every test. See tests/test_site_widgets.py.
    widgets: tuple = ()


#: Reading order. The topbar lists all five; each page hands off to the next.
PAGES = (
    Page(
        "index", "index.html", "The machine",
        "photonn &middot; a neural network made of light",
        "This neural network is made of light",
        "A trained neural network made of light: a digit enters as a beam, crosses five "
        "plates of fabricated glass, and the answer is where the light lands. Runs live "
        "in your browser.",
        "A digit enters as a beam, crosses five plates of fabricated glass, and the answer "
        "is where the light lands. It runs live, in your browser.",
        widgets=("d2nn", "stage", "interfere"),
    ),
    Page(
        "physics", "physics.html", "The physics",
        "photonn &middot; the wave optics underneath",
        "The wave optics underneath",
        "How light is moved across a gap of air exactly, the sampling limit that bounds the "
        "calculation, and the ceiling that linearity puts on the whole idea.",
        "How light is moved from one plate to the next exactly, the point where the simulation "
        "can no longer represent what it is computing, and the ceiling that linear optics puts "
        "on all of this.",
        widgets=("explorer",),
    ),
    Page(
        "chip", "chip.html", "The chip",
        "photonn &middot; the same machine, built two ways",
        "The same machine, built two ways",
        "A mesh of Mach-Zehnder interferometers on silicon computes what a stack of etched "
        "glass computes. The two differ on exactly one number.",
        "A mesh of interferometers on silicon computes what a stack of etched glass computes. "
        "Strip both to their skeletons and they differ on exactly one number -- which is also "
        "why one of them needs building ten times more accurately than the other.",
        widgets=(),
    ),
    Page(
        "tolerance", "tolerance.html", "Tolerance",
        "photonn &middot; how precisely must it be built?",
        "How precisely must it be built?",
        "The fabrication error budget: both trained networks broken on purpose, one imperfection "
        "at a time, until the number that decides feasibility falls out.",
        "The question the whole project exists to answer. Break the trained network on purpose, "
        "one fabrication error at a time, and find the one that decides whether it can be built. "
        "Then do it again to the chip, which fails a different way.",
        widgets=("err-crosstalk", "err-phase", "err-detector", "err-loss",
         "err-wavelength", "err-quant", "err-mesh"),
    ),
    Page(
        "optics", "optics.html", "Going deeper",
        "photonn &middot; how much better could the optics be?",
        "How much better could the optics be?",
        "Live work: what the 5-mask optical design leaves on the table, and what a "
        "fifty-six-mask network costs in fabrication tolerance to collect it.",
        "Live work, not a result: what the 5-mask design leaves on the table, and what a "
        "fifty-six-mask network costs in fabrication tolerance to collect it.",
        widgets=("scaling", "compare", "stage3d"),
    ),
)

PAGE_BY_KEY = {p.key: p for p in PAGES}

# figure key -> path on disk
FIGURES = {
    "phase2_masks": "docs/figures/phase2_masks.png",
    "optics_sweep": "docs/figures/optics_sweep.png",
    "mesh_topology": "docs/figures/phase3_mesh_topology.png",
    "tol_phase": "photonn-hw/figures/tolerance_phase.png",
    "tol_quant": "photonn-hw/figures/tolerance_quant.png",
    "tol_wavelength": "photonn-hw/figures/tolerance_wavelength.png",
    "tol_crosstalk": "photonn-hw/figures/tolerance_crosstalk.png",
    "tol_registration": "photonn-hw/figures/tolerance_registration.png",
    "tol_detector": "photonn-hw/figures/tolerance_detector.png",
    "tol_loss": "photonn-hw/figures/tolerance_loss.png",
    "confusion_ideal": "photonn-hw/figures/confusion_ideal.png",
    "confusion": "photonn-hw/figures/confusion_phase.png",
    "sensitivity": "photonn-hw/figures/sensitivity_map.png",
    # The mesh budget's own sensitivity map. Two panels of 36 columns by 18 rows, so
    # unlike the D2NN's it is near-square and publishable -- see the note beside
    # "cand_" below for the one that is not.
    "mesh_sensitivity": "photonn-hw/figures_mesh/sensitivity_map.png",
    # The same budget re-run against the 56-mask network. These make
    # "depth costs tolerance" showable rather than merely argued -- 91 KB for all
    # seven, because AVIF wins on every one of them.
    #
    # The 56-mask sensitivity map is deliberately absent. It is one panel per
    # mask in a single row, so at 56 masks the PNG is 16139 x 341 -- an aspect
    # ratio of 47:1, which is 15 px tall inside a grid cell and 30 px tall
    # full-width. There is no web size at which it can be read, so the page says
    # that instead of shipping a smear.
    "cand_phase": "photonn-hw/figures_candidate_L56/tolerance_phase.png",
    "cand_quant": "photonn-hw/figures_candidate_L56/tolerance_quant.png",
    "cand_wavelength": "photonn-hw/figures_candidate_L56/tolerance_wavelength.png",
    "cand_crosstalk": "photonn-hw/figures_candidate_L56/tolerance_crosstalk.png",
    "cand_detector": "photonn-hw/figures_candidate_L56/tolerance_detector.png",
    "cand_loss": "photonn-hw/figures_candidate_L56/tolerance_loss.png",
    # Ideal, not stressed, so it answers "what does depth buy?" against the front
    # page's confusion_ideal rather than "what does fabrication cost?" -- which the
    # six tolerance curves beside it already answer.
    "cand_confusion_ideal": "photonn-hw/figures_candidate_L56/confusion_ideal.png",
}

# A figure is encoded at roughly 2x the CSS width it is actually displayed at,
# which is all a 2x-density screen can resolve. Full-width plates sit in a 1120 px
# column (`.wrap`) less padding, so ~1040 px of image; the plates inside
# `.plate-grid` are laid out `minmax(270px, 1fr)` three-up, so they display at
# only ~330 px and were previously encoded at 970 -- about 3x the pixels they show.
#
# Which of the two a figure gets is a property of the token it is written as, not
# of a side table keyed by name: a side table cannot know which container the
# figure landed in, and drifted out of date the moment `360161d` lifted five
# plates out of a grid into full-width figures without touching it.
GRID_MAX_W = 700
PLATE_MAX_W = 1440

# AVIF at q60 is the knee of the quality curve for these figures, measured against
# the lossless downscale: q50->q65 buys 1.7 dB on the mesh diagram (the hardest
# case, thin lines plus text), q65->q80 buys 0.9 dB for 40% more bytes. Past q60
# the extra bits go to smooth background the eye never inspects. Lossless is not
# competitive here -- lossless AVIF/WebP on that same diagram are 249/238 KB
# against 66 KB at q60.
AVIF_QUALITY = 60
WEBP_QUALITY = 85


def _encodings(im: "Image.Image") -> list[tuple[str, bytes]]:
    """Every encoding worth considering for a figure, as (mime, bytes).

    Which one wins is not predictable from the kind of figure, so it is measured
    per figure rather than declared: WebP beats PNG on most of these plots but
    *loses badly* on the dense detector-noise plot (75 KB against 39 KB), and a
    256-colour palette PNG beats plain PNG on every line plot. Encoding all of
    them and keeping the smallest costs a second of build time and removes the
    guesswork.
    """
    out = []

    def add(mime, image, fmt, **kw):
        buf = io.BytesIO()
        image.save(buf, format=fmt, **kw)
        out.append((mime, buf.getvalue()))

    add("image/avif", im, "AVIF", quality=AVIF_QUALITY)
    add("image/webp", im, "WEBP", quality=WEBP_QUALITY, method=6)
    add("image/png", im, "PNG", optimize=True)
    # Matplotlib line art uses few distinct colours; a palette cut is lossless in
    # practice for these and roughly halves the PNG.
    flat = im.convert("P", palette=Image.ADAPTIVE, colors=256)
    add("image/png", flat, "PNG", optimize=True)
    # The same 256 colours, encoded losslessly by WebP instead of PNG. Identical
    # pixels to the line above, so this can only ever win on bytes -- and it does,
    # by 30% on the flat-shaded plates where the lossy encoders lose outright.
    # Those are the figures made of hard-edged blocks (the sensitivity maps), where
    # AVIF spends its bits smoothing edges that are the entire content.
    add("image/webp", flat.convert("RGB"), "WEBP", lossless=True, method=6)
    return out


@functools.lru_cache(maxsize=None)
def encode_figure(rel_path: str, max_w: int = PLATE_MAX_W) -> str:
    """Return a data URI for a figure, downscaled and flattened onto white.

    Matplotlib figures have white backgrounds, so we flatten any alpha onto white
    (keeps them readable inside a light 'plate' on either page theme) and cap the
    width to what the layout actually displays. The format is chosen by encoding
    the figure every way that could win and keeping the smallest result -- see
    :func:`_encodings`.

    AVIF wins for every figure in this project today, which sets the browser floor
    at Safari 16.4 / Chrome 85 / Firefox 93. There is no fallback: these pages are
    self-contained by design, so a fallback would mean shipping two copies of every
    figure and giving back the saving.
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
    mime, payload = min(_encodings(im), key=lambda c: len(c[1]))
    b64 = base64.b64encode(payload).decode("ascii")
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
.topbar-in{max-width:1120px;margin:0 auto;padding:9px 24px;
  display:flex;align-items:center;justify-content:space-between;gap:6px 16px;flex-wrap:wrap;}
.brand{font-family:var(--mono);font-size:.82rem;letter-spacing:.02em;color:var(--ink-dim);}
.brand b{color:var(--beam);font-weight:600;}
.theme-toggle{font-family:var(--mono);font-size:.74rem;letter-spacing:.04em;
  background:transparent;border:1px solid var(--border);color:var(--muted);
  border-radius:999px;padding:5px 13px;cursor:pointer;transition:color .15s,border-color .15s;}
.theme-toggle:hover{color:var(--ink);border-color:var(--beam);}
.theme-toggle:focus-visible{outline:2px solid var(--beam);outline-offset:2px;}
.topbar-right{display:flex;align-items:center;gap:6px 10px;flex-wrap:wrap;}
/* Five pages share one row, so the nav is quieter than a row of pills would be:
   only the page you are on is drawn as one. */
.topbar-nav{display:flex;align-items:center;gap:2px;flex-wrap:wrap;}
.topbar-nav a{font-family:var(--mono);font-size:.74rem;letter-spacing:.03em;text-decoration:none;
  color:var(--muted);border:1px solid transparent;border-radius:999px;padding:5px 11px;
  white-space:nowrap;transition:color .15s,background .15s,border-color .15s;}
.topbar-nav a:hover{color:var(--ink);background:var(--beam-soft);}
.topbar-nav a[aria-current="page"]{color:var(--beam);background:var(--beam-soft);
  border-color:color-mix(in srgb,var(--beam) 45%,transparent);}
.topbar-nav a:focus-visible{outline:2px solid var(--beam);outline-offset:2px;}
@media (max-width:900px){.brand span.brand-tail{display:none;}}
@media (max-width:640px){.topbar-nav a{padding:5px 8px;font-size:.7rem;}}

/* sequential hand-off at the foot of every page */
.pagenext{display:block;text-decoration:none;background:var(--surface);
  border:1px solid var(--border);border-left:3px solid var(--beam);border-radius:12px;
  padding:19px 22px;margin:46px 0 6px;transition:background .15s,border-color .15s;}
.pagenext:hover{background:var(--surface-2);border-color:var(--beam);}
.pagenext:focus-visible{outline:2px solid var(--beam);outline-offset:3px;}
.pagenext .k{display:block;font-family:var(--mono);font-size:.7rem;letter-spacing:.18em;
  text-transform:uppercase;color:var(--muted);}
.pagenext .t{display:block;font-family:var(--serif);font-weight:600;color:var(--ink);
  font-size:clamp(1.12rem,2vw,1.38rem);line-height:1.2;margin:.32rem 0 .28rem;}
.pagenext .b{display:block;color:var(--ink-dim);font-size:.94rem;max-width:64ch;}

/* In-page contents. One element, two presentations: a card in the flow of the page,
   and at >=1500px the same list pinned in the right margin. 1500 is where a 176 px
   rail plus its gap clears the 1120 px wrap on both sides -- narrower than that and
   the content would have to come off centre to make room, which costs more than the
   rail is worth. Below it the card is the whole feature and nothing is missing. */
.toc{background:var(--surface);border:1px solid var(--border);border-radius:12px;
  padding:15px 18px 17px;margin:30px 0 4px;max-width:var(--measure);}
.toc-k{font-family:var(--mono);font-size:.7rem;letter-spacing:.18em;text-transform:uppercase;
  color:var(--muted);margin:0 0 .55rem;}
.toc-list{list-style:none;margin:0;padding:0;}
.toc-list li{margin:1px 0;}
.toc-list a{display:flex;gap:9px;text-decoration:none;color:var(--ink-dim);
  font-size:.88rem;line-height:1.35;border-radius:6px;padding:3px 7px;
  border-left:2px solid transparent;transition:color .15s,background .15s,border-color .15s;}
.toc-list a:hover{color:var(--ink);background:var(--beam-soft);}
.toc-list a:focus-visible{outline:2px solid var(--beam);outline-offset:1px;}
.toc-list a[aria-current="location"]{color:var(--beam);background:var(--beam-soft);
  border-left-color:var(--beam);}
.toc-n{font-family:var(--mono);font-variant-numeric:tabular-nums;color:var(--beam);
  min-width:1.1em;flex:none;}
.toc-g{margin-top:.5rem;}
.toc-s a{padding-left:18px;}
.toc-g a{font-family:var(--mono);font-size:.7rem;letter-spacing:.14em;text-transform:uppercase;
  color:var(--muted);}
.toc-g a:hover{color:var(--ink);}
@media (min-width:1500px){
  .toc{position:fixed;top:78px;left:calc(50% + 580px);width:176px;margin:0;padding:0 0 0 13px;
    max-height:calc(100vh - 120px);overflow:auto;background:transparent;border:0;
    border-left:1px solid var(--border);border-radius:0;}
  .toc-list a{font-size:.82rem;}
}
/* Anchor jumps have to clear the sticky topbar, which is 42-46 px and taller once
   the nav wraps. */
.phase-head h2[id],.phase-head h3[id],.band-h[id]{scroll-margin-top:78px;}

/* A family of error sources, and what the family is. Only /tolerance groups its
   sections; everywhere else a section is a section. */
.band{padding:44px 0 0;}
.band-k{font-family:var(--mono);font-size:.7rem;letter-spacing:.18em;text-transform:uppercase;
  color:var(--muted);margin:0 0 .3rem;}
.band-h{font-family:var(--mono);font-size:1.02rem;font-weight:600;letter-spacing:.06em;
  text-transform:uppercase;color:var(--beam);margin:0;}
.band-b{color:var(--ink-dim);font-size:.95rem;margin:.4rem 0 0;max-width:58ch;}

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
.phase-head h2,.phase-head h3{font-family:var(--serif);font-weight:600;text-wrap:balance;
  font-size:clamp(1.55rem,2.8vw,2.05rem);line-height:1.14;margin:.1rem 0 0;}
.prose{color:var(--ink-dim);}
.prose p{margin:.85rem 0;max-width:var(--measure);}
.prose strong{color:var(--ink);font-weight:600;}
.sub-h{font-family:var(--serif);font-weight:600;color:var(--ink);
  font-size:clamp(1.15rem,2vw,1.42rem);line-height:1.2;margin:2.1rem 0 .2rem;}
.q{font-family:var(--mono);font-size:.92em;background:var(--surface-2);
  border:1px solid var(--border);border-radius:5px;padding:.05em .38em;color:var(--ink);
  white-space:nowrap;}
/* Mathematics is MathML, rendered by the browser itself. Inline expressions sit in
   the run of the sentence; display ones get their own centred line and a rule-free
   band so a long operator product is not competing with the prose around it.

   Identifier styling is left to the browser on purpose: MathML already italicises a
   single-letter <mi> and leaves a multi-letter one upright, which is exactly the
   convention wanted here (M is a matrix, exp is a function name).

   NEVER set `display` on a <math> element. MathML lays out as `display: math` (or
   `block math`), and overriding it with plain `block` drops the element back into CSS
   block layout, which puts every child on its own line -- a nine-term operator product
   becomes nine stacked rows. The scroll container therefore lives on a wrapper. */
math{font-size:1.06em;color:var(--ink);}
.eq{margin:1.15rem 0;text-align:center;overflow-x:auto;overflow-y:hidden;max-width:100%;
  padding:.1rem 0;}
.eq math{font-size:1.2em;}
a.link{color:var(--beam);text-decoration:none;border-bottom:1px solid color-mix(in srgb,var(--beam) 40%,transparent);}
a.link:hover{border-bottom-color:var(--beam);}
a.link:focus-visible{outline:2px solid var(--beam);outline-offset:2px;border-radius:2px;}

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

/* tables -- wide content scrolls inside its own box, never the page */
.tbl-wrap{overflow-x:auto;margin:24px 0;-webkit-overflow-scrolling:touch;}
.tbl{border-collapse:collapse;font-size:.92rem;min-width:min(100%,540px);}
.tbl th,.tbl td{border-bottom:1px solid var(--border);padding:9px 14px 9px 0;
  text-align:left;vertical-align:top;color:var(--ink-dim);}
.tbl th{font-family:var(--mono);font-size:.7rem;letter-spacing:.09em;text-transform:uppercase;
  color:var(--muted);font-weight:600;white-space:nowrap;}
.tbl td strong{color:var(--ink);font-weight:600;}
.tbl tr:last-child td{border-bottom:0;}

/* inline reference list -- the one section sourced from outside this project */
.refs{list-style:none;padding:0;margin:1.6rem 0 0;max-width:var(--measure);
  border-top:1px solid var(--border);padding-top:.9rem;}
.refs li{font-size:.84rem;color:var(--muted);margin:.42rem 0;line-height:1.5;}
.refs li b{font-family:var(--mono);color:var(--beam);font-weight:600;margin-right:.45em;}
.refs a{color:inherit;text-decoration:none;
  border-bottom:1px solid color-mix(in srgb,var(--muted) 40%,transparent);}
.refs a:hover{color:var(--ink);border-bottom-color:var(--beam);}
.refs a:focus-visible{outline:2px solid var(--beam);outline-offset:2px;border-radius:2px;}
sup.r{font-family:var(--mono);font-size:.66em;color:var(--beam);font-weight:600;
  padding-left:.12em;}

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
  --pe-border:var(--border);--pe-accent:var(--beam);--pe-ok:var(--good);--pe-warn:var(--bad);
  --pa-mix:var(--fringe);}

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

# ---------------------------------------------------------------------- navigation
# Every link between pages is written as @@HREF_<key>@@ and resolved once, at the
# end of render(). That is what lets the Artifact -- a lone body with no sibling
# files -- carry absolute URLs while the deployed site carries relative ones,
# without either variant knowing which pages exist.


def href(key: str, absolute: bool = False) -> str:
    """Return the URL of page ``key``, relative to the site or absolute."""
    page = PAGE_BY_KEY[key]
    if absolute:
        return SITE_URL if key == "index" else SITE_URL + page.file
    return "./" if key == "index" else page.file


def resolve_links(html: str, absolute: bool = False) -> str:
    """Substitute every ``@@HREF_<key>@@`` token in ``html``."""
    for page in PAGES:
        html = html.replace(f"@@HREF_{page.key}@@", href(page.key, absolute))
    return html


def topbar(current: str) -> str:
    """Return the shared page chrome, with ``current`` marked as the live page."""
    parts = []
    for p in PAGES:
        mark = ' aria-current="page"' if p.key == current else ""
        parts.append(f'<a href="@@HREF_{p.key}@@"{mark}>{p.nav}</a>')
    links = "".join(parts)
    return (
        '<header class="topbar">\n'
        '  <div class="topbar-in">\n'
        '    <span class="brand"><b>photonn</b><span class="brand-tail">'
        " &middot; a fabrication-tolerance study of optical neural networks</span></span>\n"
        '    <div class="topbar-right">\n'
        f'      <nav class="topbar-nav" aria-label="Sections">{links}</nav>\n'
        '      <button class="theme-toggle" id="themeToggle" aria-label="Toggle colour theme">'
        "◐ theme</button>\n"
        "    </div>\n"
        "  </div>\n"
        "</header>"
    )


def _hand_off(key: str):
    """``(next key, kicker)`` for the page that follows ``key`` in reading order.

    PAGES is documented as the reading order and drove only the topbar; the
    sequential hand-off was five literal pairs written out in ``render()``, so
    reordering PAGES reordered the nav and left the chain pointing where it
    always had. Now there is one order, and the wrap at the end is a rule rather
    than a keyword argument on the last call.
    """
    keys = [p.key for p in PAGES]
    i = keys.index(key)
    if i == len(keys) - 1:
        return keys[0], "Back to the start"
    return keys[i + 1], "Next"


def next_link(key: str, kicker: str = "Next") -> str:
    """Return the hand-off card that closes a page."""
    page = PAGE_BY_KEY[key]
    return (
        f'<a class="pagenext reveal" href="@@HREF_{key}@@">'
        f'<span class="k">{kicker}</span>'
        f'<span class="t">{page.head} &rarr;</span>'
        f'<span class="b">{page.blurb}</span></a>'
    )


# ---------------------------------------------------------------- IN-PAGE INDEX
# Section navigation is generated from the markup, never authored twice. Every
# section heading on every page sits inside a `.phase-head` and is preceded by its
# own `<p class="eyebrow">`, so one scan finds them all, gives each an id derived
# from its text, and returns the list the contents card is rendered from. Adding a
# section to a page therefore adds it to that page's index, with nothing to
# remember -- the same bargain PAGES makes for the topbar.

#: A heading and the eyebrow above it, as every page writes them.
_HEADING = re.compile(
    r'<p class="eyebrow">(?P<eyebrow>.*?)</p>\s*\n\s*'
    r'<(?P<tag>h2|h3)(?P<attrs>[^>]*)>(?P<text>.*?)</(?P=tag)>',
    re.S,
)
#: A band heading: the family a run of sections belongs to (only /tolerance has these).
_BAND = re.compile(r'<h2 class="band-h"(?P<attrs>[^>]*)>(?P<text>.*?)</h2>', re.S)
#: "Source 4 of 6" -> 4. The eyebrows already number the series; the index reuses it
#: rather than counting, so a renumbering in the prose cannot disagree with the card.
_SOURCE_NUM = re.compile(r'\bSource\s+(\d+)\b')
_TOC_ATTR = re.compile(r'\s*data-toc="([^"]*)"')


def strip_tags(markup: str) -> str:
    """Plain text of a fragment: inline tags dropped, entities resolved."""
    return _html.unescape(re.sub(r"<[^>]+>", "", markup)).strip()


def slugify(text: str) -> str:
    """A stable id from heading text."""
    slug = re.sub(r"[^a-z0-9]+", "-", strip_tags(text).lower()).strip("-")
    return slug or "section"


def toc_label(text: str) -> str:
    """The short form of a heading, for a 176 px rail.

    Section headings are written as ``Topic: what it does to you`` wherever there is
    a topic to name, so the part before the colon is already the label. Headings with
    no colon are short enough to use whole, and anything that is neither carries an
    explicit ``data-toc``.
    """
    plain = strip_tags(text)
    head = plain.split(":", 1)[0].strip()
    return head if 0 < len(head) < len(plain) else plain


def section_index(body: str):
    """Give every section heading an id; return ``(html, entries)``.

    ``entries`` is in document order, each ``{level, id, label, num}``. ``level`` is
    ``"band"`` for a family heading, and otherwise the heading's own tag --
    ``"h2"`` or ``"h3"`` -- which is what :func:`toc` dispatches on to nest the
    card by the page's outline rather than by sibling order.
    """
    entries, seen = [], {}

    def unique(slug: str) -> str:
        seen[slug] = seen.get(slug, 0) + 1
        return slug if seen[slug] == 1 else f"{slug}-{seen[slug]}"

    def take(text: str, attrs: str, level: str, num=None):
        """Record one entry and return the attribute string its heading should carry."""
        override = _TOC_ATTR.search(attrs)
        label = strip_tags(override.group(1)) if override else toc_label(text)
        attrs = _TOC_ATTR.sub("", attrs)
        ident = unique(slugify(label))
        entries.append({"level": level, "id": ident, "label": label, "num": num})
        return f' id="{ident}"{attrs}'

    def band(m):
        attrs = take(m.group("text"), m.group("attrs"), "band")
        return f'<h2 class="band-h"{attrs}>{m.group("text")}</h2>'

    def heading(m):
        # `level` is the heading tag, so the card's nesting is the page's own
        # outline: a section written as <h3> belongs to the band above it and is
        # indented under it, and one written as <h2> is not.
        eyebrow, tag = m.group("eyebrow"), m.group("tag")
        num = _SOURCE_NUM.search(strip_tags(eyebrow))
        attrs = take(m.group("text"), m.group("attrs"), tag,
                     num.group(1) if num else None)
        return (f'<p class="eyebrow">{eyebrow}</p>\n      '
                f'<{tag}{attrs}>{m.group("text")}</{tag}>')

    # Bands first: their heading is an <h2> with no eyebrow, so the two patterns
    # cannot both match the same element, and running bands first keeps the entries
    # in document order for the page that has them.
    body = _BAND.sub(band, body)
    body = _HEADING.sub(heading, body)
    entries.sort(key=lambda e: body.index(f'id="{e["id"]}"'))
    return body, entries


def toc(entries) -> str:
    """Render the contents card. Empty for a page with nothing to index."""
    if not entries:
        return ""
    items = []
    for e in entries:
        cls = {"band": ' class="toc-g"', "h3": ' class="toc-s"'}.get(e["level"], "")
        num = f'<span class="toc-n">{e["num"]}</span>' if e["num"] else ""
        items.append(f'<li{cls}><a href="#{e["id"]}">{num}{e["label"]}</a></li>')
    return (
        # No `reveal` here: this is navigation chrome, like the topbar, not
        # content. `.reveal` starts at opacity 0 and waits for an observer, so a
        # reader landing on a #fragment could arrive at an index they cannot see.
        '<nav class="toc" aria-label="On this page">\n'
        '  <p class="toc-k">On this page</p>\n'
        '  <ol class="toc-list">' + "".join(items) + "</ol>\n"
        "</nav>"
    )


#: An empty caption number, and the optional key prose refers to it by.
_FIGN = re.compile(r'<span class="fign"(?P<attrs>[^>]*)></span>')
_FIG_KEY = re.compile(r'data-fig="([^"]*)"')


def figure_index(body: str):
    """Number every caption in document order; return ``(html, refs)``.

    A figure's number is a property of where it sits, not something an author
    types beside it. Numbering by hand survived until `5f1bad6` inserted a whole
    band of geometry sources above an existing plate and numbered the new figure
    9 rather than renumber the one below it, so /tolerance shipped reading
    1..7, 9, 8. Derived in document order, that cannot happen: inserting a
    figure renumbers everything after it, and nothing else has to be touched.

    The one caption prose refers to carries ``data-fig="key"``; ``refs`` maps it
    to its number, for ``@@FIGREF_key@@``. The same bargain as the contents card
    -- see :func:`section_index`.
    """
    refs = {}
    count = 0

    def one(m):
        nonlocal count
        count += 1
        key = _FIG_KEY.search(m.group("attrs"))
        if key:
            refs[key.group(1)] = count
        return f'<span class="fign">Fig {count}</span>'

    return _FIGN.sub(one, body), refs


# --------------------------------------------------------------------------- MATH
# Mathematics is MathML. Every browser in use renders it natively (Chromium 109+,
# Firefox and Safari all ship MathML Core), so real notation -- stacked fractions,
# actual roots, proper sub- and superscripts -- costs a dependency of exactly
# nothing. KaTeX would be ~300 KB to render fifteen short expressions.
#
# What MathML does cost is verbosity: written out by hand it is unreadable at any
# length. So expressions are written below in a compact notation and expanded by
# `mrow`. Tokens are space-separated; `a_b` is a subscript, `a^b` a superscript,
# anything in _MATH_OPS is an operator, a leading digit makes a number, and
# everything else is an identifier. Fractions and roots nest, which a flat token
# stream cannot express, so they take the explicit helpers.
#
# Note this is *only* for mathematics. Code and UI identifiers keep the mono pill
# (`.q`): `check_sampling` is a function name, not a variable, and italicising it
# as though it were one would be wrong.

_MATH_OPS = {
    "=", "+", "&minus;", "&middot;", "&times;", "(", ")", "|", "&dagger;",
    "&#8827;", "&gt;", "&lt;", ",", "&hellip;", "/",
}


def _mtok(token: str) -> str:
    """Expand one compact token into one MathML element."""
    if token in _MATH_OPS:
        return f"<mo>{token}</mo>"
    if "_" in token:
        base, sub = token.split("_", 1)
        return f"<msub>{_mtok(base)}{_mtok(sub)}</msub>"
    if "^" in token:
        base, sup = token.split("^", 1)
        return f"<msup>{_mtok(base)}{_mtok(sup)}</msup>"
    if token[0].isdigit():
        return f"<mn>{token}</mn>"
    return f"<mi>{token}</mi>"


def mrow(spec: str) -> str:
    """Expand a compact expression into MathML, with no <math> wrapper."""
    return "".join(_mtok(t) for t in spec.split())


def mfrac(num: str, den: str) -> str:
    return f"<mfrac>{mrow(num)}{mrow(den)}</mfrac>"


def msqrt(*parts: str) -> str:
    return "<msqrt>" + "".join(parts) + "</msqrt>"


def mathml(body: str, display: bool = False) -> str:
    """Wrap already-expanded MathML in a <math> element.

    ``display=True`` gives the expression its own centred line, which is what the
    long ones need; the default sits inline in the run of the sentence.

    A display expression is wrapped in ``div.eq`` rather than styled directly,
    because the horizontal scroll a wide equation needs on a narrow screen cannot
    go on the <math> element itself: any `display` or `overflow` set there costs
    MathML its own layout mode. The wrapper carries both.
    """
    if not display:
        return f"<math>{body}</math>"
    return f'<div class="eq"><math display="block">{body}</math></div>'


#: name -> rendered MathML, substituted into page bodies as @@MATH_name@@.
MATH = {
    # /physics -- the propagator
    "asm": mathml(
        mrow("H = exp ( i 2 &pi; z")
        + msqrt(mfrac("1", "&lambda;^2") + mrow("&minus; f^2"))
        + mrow(")"),
        display=True,
    ),
    "evanescent": mathml(mrow("f^2 &gt;") + mfrac("1", "&lambda;^2")),
    "zcrit": mathml(mrow("z_crit = N &middot; dx^2 / &lambda;")),
    "additive": mathml(mrow("H ( z_1 ) &middot; H ( z_2 ) = H ( z_1 + z_2 )")),
    # /physics -- the linearity argument
    "linear": mathml(mrow("E_out = M &middot; E_in"), display=True),
    # "L-1" is a whole expression, not an identifier, so the subscript is a row.
    "operator": mathml(
        mrow("M =")
        + "<msub><mi>P</mi><mi>L</mi></msub><msub><mi>D</mi><mi>L</mi></msub>"
        + "<msub><mi>P</mi><mrow><mi>L</mi><mo>&minus;</mo><mn>1</mn></mrow></msub>"
        + "<msub><mi>D</mi><mrow><mi>L</mi><mo>&minus;</mo><mn>1</mn></mrow></msub>"
        + mrow("&hellip; D_1 P_0"),
        display=True,
    ),
    "score": mathml(mrow("s_c = E_in &dagger; A_c E_in"), display=True),
    "psd": mathml(
        mrow("A_c = M &dagger; R_c M &#8827; 0"), display=True
    ),
    "psd_sign": mathml(mrow("&#8827; 0")),
    "intensity": mathml("<msup><mrow><mo>|</mo><mi>E</mi><mo>|</mo></mrow><mn>2</mn></msup>"),
    "encode": mathml(mrow("exp ( i &middot; &pi; &middot; image )")),
    "maskphase": mathml(mrow("exp ( i &phi; ( x , y ) )")),
    # /chip
    "mzi": mathml(mrow("B &middot; P ( &theta; ) &middot; B &middot; P ( &phi; )")),
    "svd": mathml(mrow("U &middot; &Sigma; &middot; V &dagger;")),
    "nmzi": mathml(mrow("n ( n &minus; 1 ) / 2")),
    "sigma": mathml(mrow("&Sigma;")),
    "theta": mathml(mrow("&theta;")),
    "phi": mathml(mrow("&phi;")),
    # bare symbols carried through the prose
    "H": mathml(mrow("H")),
    "M": mathml(mrow("M")),
    "P": mathml(mrow("P")),
    "D": mathml(mrow("D")),
    "z": mathml(mrow("z")),
    "zcrit_sym": mathml(mrow("z_crit")),
    # /optics
    "reach": mathml(mrow("reach = z &middot; &lambda; / ( 2 &middot; dx^2 )")),
    # /tolerance -- the 95%-of-ideal pass mark
    "bar": mathml("<mo>&ge;</mo><mn>0.7591</mn>"),
}


def _math(html: str) -> str:
    """Substitute every @@MATH_name@@ token."""
    for name, rendered in MATH.items():
        html = html.replace(f"@@MATH_{name}@@", rendered)
    return html


def error_mask_bundle() -> str:
    """One real phase mask, for the error widgets on /tolerance.

    Cut out of the committed weights bundle at build time rather than exported to
    a file of its own, so it cannot drift from the model that ships: there is one
    copy of the trained phases in the repo and this is a slice of it. Mask 0 of 5,
    16 KB of 8-bit codes, which decode exactly as ``d2nn.js`` decodes them.

    The widgets need a *real* surface. A synthetic stand-in would blur and
    quantise differently, and the whole point of them is to show what these errors
    do to the thing actually being built.
    """
    from apps.web_bundle import read_bundle

    w = read_bundle(os.path.join(REPO, "apps", "web", "d2nn_weights.js"))
    if w["masks_bits"] != 8:
        raise SystemExit(
            f"error_mask_bundle expects 8-bit codes, bundle has {w['masks_bits']}"
        )
    n = int(w["n"])
    codes = base64.b64decode(w["masks_b64"])[: n * n]
    payload = json.dumps({"n": n, "mask_b64": base64.b64encode(codes).decode("ascii")})
    js = (
        "(function(){var M=" + payload + ";"
        "if(typeof window!=='undefined')window.PHOTONN_ERR_MASK=M;"
        "if(typeof module!=='undefined'&&module.exports)module.exports=M;})();"
    )
    return f"<script>\n{js}\n</script>\n<script>\n{read_web_asset('errors.js')}\n</script>"


def script_tags(*assets: str) -> str:
    """Inline the named ``apps/web`` assets as <script> blocks, in order.

    Order is the argument order, which is the point: it used to be expressed by
    concatenating two bundle functions with a ``+``, and one of those orders is
    load-bearing. ``errors.js`` reads ``window.PHOTONN_MESH`` at module scope, so
    ``mesh_weights.js`` has to precede it; loaded the other way the widget falls
    back to a synthetic stand-in and draws a chip nobody trained, silently.

    Replaces three one-line functions whose interface was as large as their
    implementation (``scaling_bundle``, ``mesh_weights_bundle``,
    ``interference_bundle``): each named a fixed asset list and did nothing else.
    """
    return "\n".join(f"<script>\n{read_web_asset(a)}\n</script>" for a in assets)


def error_mount(container_id: str, kind: str) -> str:
    """Mount one error-mechanism widget, deferred until the reader nears it."""
    return mount_script(
        container_id,
        f"window.PhotonnErrors.mount(el, {{kind: {json.dumps(kind)}}});",
        defer=True,
    )


def interference_mount(container_id: str) -> str:
    """Mount the interference widget. Not deferred: it is cheap and sits high.

    ``mount_queue.js`` already holds every widget until after first paint and
    starts them one at a time, which is the whole reason deferring exists. This
    one is a few hundred sine evaluations and sits within a screen of the top,
    where a reader would meet *Warming up...* rather than a widget.
    """
    return mount_script(container_id, "window.PhotonnInterfere.mount(el);")


# --------------------------------------------------------------------------- BODY
# Placeholder tokens (@@...@@) are substituted in render(); avoids CSS/JS brace escaping.
#: Page prose lives in apps/pages/*.html, not in this module.
#:
#: 1,565 of this file's 2,652 lines were body literals, which is why it
#: appeared in 17 of 30 commits: it is the file every editorial change lands
#: in, and a diff answering "did the generator change?" was dominated by
#: paragraphs about diffraction. The token vocabulary below (@@TOPBAR@@,
#: @@TOC@@, @@NEXT@@, @@FIG_*@@, @@MATH_*@@, @@HREF_*@@) was already the
#: interface between the prose and the generator, so moving the prose across
#: it costs nothing and makes an editorial commit legible as one.
PAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pages")


def page_body(key: str) -> str:
    """The raw body of one page, before any token is substituted."""
    with open(os.path.join(PAGES_DIR, f"{key}.html"), encoding="utf-8") as fh:
        return fh.read()


BODY = page_body("index")

# Shared page chrome: the theme toggle (persisted) and the scroll-reveal observer.
# Both generated pages get the same block so the toggle carries across navigation.
PAGE_SCRIPT = r"""<script>
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
(function(){
  var links=document.querySelectorAll('.toc-list a[href^="#"]');
  if(!links.length) return;
  var map={},order=[];
  links.forEach(function(a){var id=a.getAttribute('href').slice(1),el=document.getElementById(id);
    if(el){map[id]=a; order.push(el);}});
  if(!order.length) return;
  function mark(){
    var cur=order[0].id;
    order.forEach(function(el){if(el.getBoundingClientRect().top<=90) cur=el.id;});
    for(var k in map){if(k===cur){map[k].setAttribute('aria-current','location');}
      else{map[k].removeAttribute('aria-current');}}
  }
  // A throttled scroll listener, not an IntersectionObserver. "Which section am I
  // reading" is a question about every heading at once, and an observer answers only
  // about the one that crossed -- and answers nothing at all when a click on this
  // card jumps the page straight over the crossing, which is the commonest way this
  // card is used. The reveal observer above cannot be reused either: it unobserves
  // on first sight.
  var queued=false;
  function ping(){if(queued) return; queued=true;
    setTimeout(function(){queued=false; mark();},100);}
  addEventListener('scroll',ping,{passive:true});
  addEventListener('resize',ping,{passive:true});
  addEventListener('hashchange',mark);
  mark();
})();
</script>"""

# ------------------------------------------------------------------------ PHYSICS
# The propagator, the sampling limit that bounds it, and the ceiling that linearity
# puts on everything downstream. This is where the front page's "one linear
# operator" assertion is actually argued.
PHYSICS_BODY = page_body("physics")

# --------------------------------------------------------------------------- CHIP
# The mesh, and the correspondence between it and the diffractive stack. The best
# writing on the site is in here ("why they are the same machine"), so it is kept
# whole and the promoted docs material is arranged around it.
CHIP_BODY = page_body("chip")

# ---------------------------------------------------------------------- TOLERANCE
# The destination. The project's central question gets its own page, and the eight
# 56-mask figures put the "depth costs tolerance" trade beside the 5-mask
# curves -- the first time that argument is shown rather than only asserted.
TOLERANCE_BODY = page_body("tolerance")

# ------------------------------------------------------------------------ OPTICS
# The last page in the reading order, and the only one that is *live work*: the
# other four state the machine as built, this one tracks what the optics could
# still be, and the two must not be confused. Everything here is measured against
# a short ranking protocol, so every number on the page says so.
OPTICS_BODY = page_body("optics")

def _document(body: str, page: Page) -> str:
    """Wrap a rendered body in the shared document shell."""
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<meta name="description" content="{page.desc}">\n'
        f"<title>{page.title}</title>"
        "\n<style>\n" + CSS + "\n</style>\n</head>\n<body>\n"
        + body
        + "\n</body>\n</html>\n"
    )


def _figures(html: str) -> str:
    """Inline every figure the page actually references, and no others.

    Two tokens, because a figure is encoded for the container it is placed in:
    ``@@FIG_x@@`` is a full-width plate, ``@@GRIDFIG_x@@`` one of a row inside a
    `.plate-grid`. Moving a figure between the two is then one edit, in the
    place the move actually happens.
    """
    for key, path in FIGURES.items():
        for token, max_w in ((f"@@FIG_{key}@@", PLATE_MAX_W),
                             (f"@@GRIDFIG_{key}@@", GRID_MAX_W)):
            if token in html:
                html = html.replace(token, encode_figure(path, max_w))
    return html


def _chrome(body: str, key: str) -> str:
    """Fill in everything every page shares: nav, hand-off, page script, figures.

    Link tokens are deliberately left in place -- ``resolve_links`` runs last, so
    one rendered body can be emitted twice, once relative and once absolute. The
    in-page index is filled here rather than later because it needs the ids
    :func:`section_index` writes into the headings; its links are ``#fragment``
    and never page tokens, so it is indifferent to that ordering.
    """
    html, entries = section_index(body)
    html = html.replace("@@TOC@@", toc(entries))
    html, figs = figure_index(html)
    for name, num in figs.items():
        html = html.replace(f"@@FIGREF_{name}@@", str(num))
    html = html.replace("@@TOPBAR@@", topbar(key))
    html = html.replace("@@NEXT@@", next_link(*_hand_off(key)))
    # plot.js goes in the page-script slot, which every page body places above
    # every widget bundle. That gives exactly one copy per page, before any
    # widget reads window.PhotonnPlot at module scope. Prepending it to each
    # bundle instead would inline it two or three times on a page and cost
    # /index most of its remaining weight budget.
    #
    # A page with no widgets does not get it at all -- /chip carries no canvas,
    # so shipping the shared canvas module there would be paying for a drawing
    # nobody makes. Page.widgets is what makes that answerable.
    plot = script_tags("plot.js") + "\n" if PAGE_BY_KEY[key].widgets else ""
    html = html.replace("@@PAGE_SCRIPT@@", plot + mount_queue_bundle() + PAGE_SCRIPT)
    return _figures(_math(html))


def render() -> dict:
    """Return ``{filename: html}`` for every file the site is made of."""
    out = {}

    # The front page carries the whole engine: no explorer runs here, so it
    # inlines asm.js itself.
    body = BODY.replace("@@D2NN_BUNDLE@@", d2nn_bundle(include_asm=True))
    body = body.replace("@@D2NN_MOUNT@@", d2nn_mount("d2nn", stage_id="stage"))
    # interfere.js is a separate file from errors.js rather than an eighth
    # `kind` in it: that module is 37 KB, is bound to the tolerance page, and
    # reads the trained mask bundle at module scope. Charging the front page
    # all of that to draw two sine waves would be the wrong trade twice over.
    body = body.replace("@@INTERFERE_BUNDLE@@", script_tags("interfere.js"))
    body = body.replace("@@INTERFERE_MOUNT@@", interference_mount("interfere"))
    body = _chrome(body, "index")
    out["index.html"] = _document(resolve_links(body), PAGE_BY_KEY["index"])
    # The artifact is a standalone body with no sibling pages, so its links must
    # be absolute and it supplies no <head> of its own.
    out["_artifact_body.html"] = ("<style>\n" + CSS + "\n</style>\n"
                                  + resolve_links(body, absolute=True))

    phys = PHYSICS_BODY.replace("@@EXPLORER_BUNDLE@@", explorer_bundle())
    phys = phys.replace("@@EXPLORER_MOUNT@@", explorer_mount("explorer"))
    phys = _chrome(phys, "physics")
    out["physics.html"] = _document(resolve_links(phys), PAGE_BY_KEY["physics"])

    # No widget here any more. The crosswalk table makes the correspondence argument on
    # its own, and dragging the slider added nothing the prose had not already said.
    # analogy.js and analogy_geom.js stay in the repo for apps/analogy_demo.py. The
    # chip's error budget lives on /tolerance rather than here: it is a comparison
    # against the stack's budget, and in reading order that budget is a page away.
    chip = _chrome(CHIP_BODY, "chip")
    out["chip.html"] = _document(resolve_links(chip), PAGE_BY_KEY["chip"])

    # One widget per error source, six against the stack and a seventh against the
    # chip. They share one copy of errors.js and are each deferred until the reader
    # nears it, so seven of them on one page cost one download and never seven
    # simultaneous starts. mesh_weights_bundle goes first -- errors.js reads the mesh
    # off window at module scope.
    tol = TOLERANCE_BODY.replace(
        "@@ERRORS_BUNDLE@@",
        # mesh_weights.js FIRST: errors.js reads window.PHOTONN_MESH at module
        # scope, so loaded the other way round the widget silently falls back
        # to its synthetic stand-in and draws a chip nobody trained. The order
        # is this list, not a `+` between two functions.
        script_tags("mesh_weights.js") + "\n" + error_mask_bundle())
    tol = tol.replace("@@ERRORS_MOUNT@@", "\n".join(
        error_mount(f"err-{kind}", kind) for kind in
        ("crosstalk", "phase", "detector", "loss", "wavelength", "quant", "mesh")
    ))
    tol = _chrome(tol, "tolerance")
    out["tolerance.html"] = _document(resolve_links(tol), PAGE_BY_KEY["tolerance"])

    # The reach widget is gone with the reach argument; this page is about depth
    # now. scaling.js reads the same sweep bundle optics.js used to.
    #
    # optics.js and analogy.js are therefore mounted on no built page. They are
    # kept rather than retired, and the reason is recorded here so the next
    # reader does not have to work it out: both are still reachable from their
    # standalone previews (apps/optics_demo.py, apps/analogy_demo.py), which
    # tests/test_preview_pages.py now builds and guards; optics.js:reachPerHop is
    # driven by tests/optics_geom_runner.js as a live cross-check of the closed
    # form against photonn.propagate.diffraction_reach_px; and analogy_geom.js is
    # load-bearing for the /chip plate and for tests/test_correspondence.py.
    opt = OPTICS_BODY.replace(
        "@@SCALING_BUNDLE@@", script_tags("optics_sweep.js", "scaling.js"))
    opt = opt.replace("@@SCALING_MOUNT@@", mount_script(
        "scaling", "window.PhotonnScaling.mount(el);"))
    opt = opt.replace("@@COMPARE_BUNDLE@@", compare_bundle(stage=True))
    # Open on a digit the 5-mask model gets wrong and the 56-mask one does not.
    # The stage draws the deep column as a machine; it is fed the same digit the
    # board is, and decides for itself when to run the forward pass.
    opt = opt.replace("@@COMPARE_MOUNT@@", compare_mount(
        "compare", gallery=14, stage_id="stage3d", stage_model="deep"))
    # Last page in the reading order: the hand-off loops back to the machine.
    opt = _chrome(opt, "optics")
    out["optics.html"] = _document(resolve_links(opt), PAGE_BY_KEY["optics"])

    return out


def main():
    os.makedirs(SITE, exist_ok=True)
    for name, text in render().items():
        path = os.path.join(SITE, name)
        # newline="\n" so the file written here is byte-for-byte what `render()`
        # returned. Without it Python translates on Windows, the working tree
        # fills with CRLF that .gitattributes normalises straight back out, and
        # the built page cannot be compared against the render that produced it.
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print(f"wrote {path} ({len(text) // 1024} KB)")


if __name__ == "__main__":
    main()
