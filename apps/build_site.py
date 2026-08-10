"""Build the photonn site: five self-contained HTML pages.

The site is an explainer of optical neural networks that *lands* on the project's
central question rather than opening with it. A reader meets the working machine
first, learns what it is, and only then is asked how precisely it would have to be
fabricated -- which is the answer to "why don't we already have these".

  site/index.html      -- the machine: the trained network, live, plus what it is
  site/physics.html    -- the wave optics underneath it, plus the diffraction explorer
  site/chip.html       -- the same computation built as an interferometer mesh
  site/tolerance.html  -- the fabrication error budget: the study's destination
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
import io
import os
from typing import NamedTuple

from PIL import Image

from apps.analogy_demo import analogy_bundle, analogy_mount
from apps.compare_demo import compare_bundle, compare_mount
from apps.d2nn_demo import d2nn_bundle, d2nn_mount
from apps.diffraction_explorer import explorer_bundle, explorer_mount, mount_queue_bundle
from apps.optics_demo import optics_bundle, optics_mount

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


#: Reading order. The topbar lists all five; each page hands off to the next.
PAGES = (
    Page(
        "index", "index.html", "The machine",
        "photonn &mdash; a neural network made of light",
        "This neural network is made of light",
        "A trained neural network made of light: a digit enters as a beam, crosses five "
        "plates of fabricated glass, and the answer is where the light lands. Runs live "
        "in your browser.",
        "A digit enters as a beam, crosses five plates of fabricated glass, and the answer "
        "is where the light lands. It runs live, in your browser.",
    ),
    Page(
        "physics", "physics.html", "The physics",
        "photonn &mdash; the wave optics underneath",
        "The wave optics underneath",
        "The scalar diffraction that moves a light field from one plane to the next, the "
        "sampling limit that bounds it, and the ceiling linearity puts on the whole idea.",
        "How a field is moved from one plane to the next exactly, where the grid stops being "
        "able to represent it, and the ceiling that linear optics puts on all of this.",
    ),
    Page(
        "chip", "chip.html", "The chip",
        "photonn &mdash; the same machine, built two ways",
        "The same machine, built two ways",
        "A mesh of Mach-Zehnder interferometers on silicon computes what a stack of etched "
        "glass computes. The two differ on exactly one number.",
        "A mesh of interferometers on silicon computes what a stack of etched glass computes. "
        "Strip both to their skeletons and they differ on exactly one number.",
    ),
    Page(
        "tolerance", "tolerance.html", "Tolerance",
        "photonn &mdash; how precisely must it be built?",
        "How precisely must it be built?",
        "The fabrication error budget: the trained network broken on purpose, one imperfection "
        "at a time, until the number that decides feasibility falls out.",
        "The question the whole project exists to answer. Break the trained network on purpose, "
        "one fabrication error at a time, and find the one that decides whether it can be built.",
    ),
    Page(
        "optics", "optics.html", "Going deeper",
        "photonn &mdash; how much better could the optics be?",
        "How much better could the optics be?",
        "Live work: what the shipped optical design leaves on the table, and what a "
        "fifty-six-mask network costs in fabrication tolerance to collect it.",
        "Live work, not a result: what the shipped design leaves on the table, and what a "
        "fifty-six-mask network costs in fabrication tolerance to collect it.",
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
    "tol_detector": "photonn-hw/figures/tolerance_detector.png",
    "tol_loss": "photonn-hw/figures/tolerance_loss.png",
    "confusion": "photonn-hw/figures/confusion_phase035.png",
    "sensitivity": "photonn-hw/figures/sensitivity_map.png",
    # The same budget re-run against the unshipped 56-mask candidate. These make
    # "depth costs tolerance" showable rather than merely argued -- 91 KB for all
    # seven, because AVIF wins on every one of them.
    #
    # The candidate's sensitivity map is deliberately absent. It is one panel per
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
    "cand_confusion": "photonn-hw/figures_candidate_L56/confusion_phase035.png",
}

# A figure is encoded at roughly 2x the CSS width it is actually displayed at,
# which is all a 2x-density screen can resolve. Full-width plates sit in a 1120 px
# column (`.wrap`) less padding, so ~1040 px of image; the seven plates inside
# `.plate-grid` are laid out `minmax(270px, 1fr)` three-up, so they display at
# only ~330 px and were previously encoded at 970 -- about 3x the pixels they show.
GRID_MAX_W = 700
PLATE_MAX_W = 1440

DEFAULT_OPT = {"max_w": PLATE_MAX_W}
FIG_OPTS = {key: {"max_w": GRID_MAX_W} for key in (
    "tol_phase", "tol_quant", "tol_wavelength", "tol_crosstalk", "tol_detector",
    "tol_loss", "confusion",
    "cand_phase", "cand_quant", "cand_wavelength", "cand_crosstalk", "cand_detector",
    "cand_loss", "cand_confusion",
)}

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
    add("image/png", im.convert("P", palette=Image.ADAPTIVE, colors=256), "PNG", optimize=True)
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
.sub-h{font-family:var(--serif);font-weight:600;color:var(--ink);
  font-size:clamp(1.15rem,2vw,1.42rem);line-height:1.2;margin:2.1rem 0 .2rem;}
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


def next_link(key: str, kicker: str = "Next") -> str:
    """Return the hand-off card that closes a page."""
    page = PAGE_BY_KEY[key]
    return (
        f'<a class="pagenext reveal" href="@@HREF_{key}@@">'
        f'<span class="k">{kicker}</span>'
        f'<span class="t">{page.head} &rarr;</span>'
        f'<span class="b">{page.blurb}</span></a>'
    )


# --------------------------------------------------------------------------- BODY
# Placeholder tokens (@@...@@) are substituted in render(); avoids CSS/JS brace escaping.
BODY = r"""
<div class="spectral-rule"></div>
@@TOPBAR@@

<main class="wrap">

  <section class="hero col reveal">
    <p class="eyebrow">An optical neural network, running live</p>
    <h1>This neural network is made of <em>light</em></h1>
    <div class="underbar"></div>
    <p class="standfirst">A handwritten digit is written into a beam, the beam crosses five plates
    of fabricated glass, and the answer is <b>where the light lands</b>. There are no transistors in
    that sentence and no activation functions &mdash; the trained parameters <i>are</i> the surfaces.
    It runs below, in your browser. Then this site asks the question that decides whether such a
    thing could ever be built: <b>how much fabrication error does it survive?</b></p>
    <div class="stat-strip">
      <div class="stat"><span class="v">0.799</span><span class="l">of MNIST digits read correctly, by the optics alone &mdash; chance is 0.10</span></div>
      <div class="stat"><span class="v">5<small> plates</small></span><span class="l">of phase-shifting glass are the entire trained network</span></div>
      <div class="stat"><span class="v">60<small> ps</small></span><span class="l">for light to cross it end to end &mdash; 18&nbsp;mm of air and glass</span></div>
    </div>
  </section>

  <div class="explorer-band reveal">
    <div class="pe-host"><div id="d2nn"></div></div>
    <p class="cap">Live &mdash; pick a digit from the frozen MNIST test set, or draw your own. The
    forward pass is computed here, now, on a hand-written FFT: <b>no libraries and no network
    requests</b>. The masks are the exported parameters of the trained model, unchanged.</p>
  </div>

  <div class="explorer-band reveal">
    <div class="pe-host"><div id="stage"></div></div>
    <p class="cap">The same run, drawn as the machine it is &mdash; entrance plane, five phase
    masks, detector plane, along the optical axis. Drag to orbit; hit <b>Sweep</b> to walk one
    wavefront across the stack.</p>
  </div>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">What you just watched</p>
      <h2>Every panel is a real optical field</h2></div></div>
    <div class="prose col">
      <p>The <strong>entrance field</strong> is your digit written into the amplitude <em>and</em> the
      phase of the light entering the stack. The five small frames are the intensity
      <strong>arriving at each phase mask</strong> &mdash; watch the digit dissolve into structured
      speckle that means nothing to the eye and everything to the detectors. The
      <strong>detector plane</strong> is the final intensity with the ten class regions drawn on it,
      and the class is simply <strong>whichever box collects the most power</strong>. That is the
      entire readout: no electronic layer, no learned classifier on top, just ten sums.</p>
      <p>The 3D view is the geometry of the same run: seven parallel planes along the optical axis,
      each carrying the field actually computed on it, with the light <em>between</em> them drawn as
      haze. That haze is not a gradient &mdash; it is the field at intermediate depths, computed the
      same way, and <a class="link" href="@@HREF_physics@@">the physics page shows why splitting a
      hop like that is exact &rarr;</a> Toggle <em>Mask phase</em> to swap the arriving light for the
      fabricated surface that acts on it.</p>
      <p>There is no electronic network anywhere in this. The only nonlinearity in the whole model
      is the <span class="q">|E|&sup2;</span> of detection &mdash; everything before it is one linear
      optical operator. That is the whole computation, and also its ceiling.</p>
    </div>
    <div class="stats col">
      <div class="s"><div class="v">0.799</div><div class="l">test accuracy (chance 0.10)</div></div>
      <div class="s"><div class="v">5 masks</div><div class="l">81,920 trained phases</div></div>
      <div class="s"><div class="v">6 hops</div><div class="l">3&nbsp;mm each, 532&nbsp;nm, N=128</div></div>
      <div class="s"><div class="v">&lt;10<sup>&minus;3</sup></div><div class="l">class-score agreement with PyTorch</div></div>
    </div>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">The motivation, stated honestly</p>
      <h2>Why build a computer out of light?</h2></div></div>
    <div class="prose col">
      <p>Because the expensive part of the arithmetic above is free. In a processor the
      multiply&ndash;add is already cheap: at 45&nbsp;nm a 32-bit floating-point add costs about
      <strong>0.9&nbsp;pJ</strong>, and a fixed-point one roughly a ninth of that. What costs is
      <em>getting the operands to it</em> &mdash; those same 32 bits read out of on-chip SRAM cost
      <strong>5&nbsp;pJ</strong>, and out of DRAM <strong>640&nbsp;pJ</strong>, some three orders of
      magnitude more than the operation they feed.<sup class="r">1</sup></p>
      <p>The stack above never does that. Every pixel of the field influences every detector because
      <em>that is what a wave does</em> over 3&nbsp;mm of air: the all-to-all connection is performed
      by propagation itself, with nothing charged, nothing switched, and no weight fetched from
      anywhere. <strong>The weights are not fetched because the weights are the glass.</strong> And
      the computation finishes in the time light needs to cross 18&nbsp;mm &mdash;
      <strong>60&nbsp;picoseconds</strong>, set by <span class="q">c</span> and nothing else.</p>
      <p>How little light does it need? The error budget measures exactly that: accuracy holds flat
      from 1&nbsp;mW all the way down to <strong>1&nbsp;pW over a 1&nbsp;ms exposure</strong> &mdash;
      about <strong>one femtojoule of light per inference</strong> &mdash; and only below that does it
      fall off a cliff.</p>
      <p>Now the honest part, because this is the number everyone gets wrong. That femtojoule is the
      energy <em>in the light</em>, not the energy to run the machine. The laser, the modulator that
      writes the digit into the beam, the ten detectors and their converters all cost more, and
      <strong>this project models none of them.</strong> The design&rsquo;s nominal operating point is
      1&nbsp;mW for 1&nbsp;ms, which is a microjoule per inference &mdash; <em>worse</em> than a GPU,
      and stated here so that nobody, including us, quotes it as a win. The narrow claim is the only
      one the evidence supports: <strong>the part the optics does is nearly free, and everything
      around it is the engineering problem.</strong> That is roughly where the field itself
      sits.<sup class="r">4</sup></p>
      <p>Two families of machine chase this. Free-space diffractive networks &mdash; a beam through
      trained plates, which is what runs above<sup class="r">2</sup> &mdash; and integrated meshes of
      interferometers etched into silicon.<sup class="r">3</sup> This project builds both, and
      <a class="link" href="@@HREF_chip@@">they turn out to be the same machine &rarr;</a></p>
      <ol class="refs">
        <li><b>1</b>M. Horowitz, &ldquo;Computing&rsquo;s energy problem (and what we can do about
        it),&rdquo; <em>ISSCC</em> 2014, 10&ndash;14.
        <a href="https://www.semanticscholar.org/paper/947620a1854655ed91a86b90d12695e05be85983">semanticscholar.org</a></li>
        <li><b>2</b>X. Lin <em>et al.</em>, &ldquo;All-optical machine learning using diffractive deep
        neural networks,&rdquo; <em>Science</em> <b>361</b>, 1004 (2018).
        <a href="https://www.science.org/doi/10.1126/science.aat8084">doi:10.1126/science.aat8084</a></li>
        <li><b>3</b>Y. Shen <em>et al.</em>, &ldquo;Deep learning with coherent nanophotonic
        circuits,&rdquo; <em>Nature Photonics</em> <b>11</b>, 441 (2017).
        <a href="https://www.nature.com/articles/nphoton.2017.93">doi:10.1038/nphoton.2017.93</a></li>
        <li><b>4</b>G. Wetzstein <em>et al.</em>, &ldquo;Inference in artificial intelligence with deep
        optics and photonics,&rdquo; <em>Nature</em> <b>588</b>, 39 (2020).
        <a href="https://www.nature.com/articles/s41586-020-2973-6">doi:10.1038/s41586-020-2973-6</a></li>
      </ol>
    </div>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">What the trained surfaces are doing</p>
      <h2>How a phase mask computes</h2></div></div>
    <div class="prose col">
      <p>Two operations alternate, and both are linear. Free-space propagation is
      <strong>convolution with the diffraction kernel</strong> &mdash; shift-invariant, the same
      operator applied everywhere. A phase mask is a <strong>pointwise multiplication</strong> by
      <span class="q">exp(i&thinsp;&phi;(x,y))</span> &mdash; per-pixel, and the only thing training
      ever touches. Alternating them builds a cascade of <em>multiply pointwise, then convolve</em>
      steps, which is to say a <strong>learned cascade of holograms</strong>.</p>
      <p>What the masks learn to do with that is <strong>route and focus</strong>. They shape the
      phase front so that after diffraction an input of class <em>c</em> interferes constructively
      onto detector region <em>c</em> and destructively everywhere else. Early masks behave more like
      feature encoders, redistributing energy across the aperture; later ones behave more like
      focusing and steering elements, concentrating the class-dependent energy onto the right patch.
      About 60% of the light that enters ends up inside a detector box.</p>
      <p>That routing picture is <strong>qualitative</strong>, and it is worth saying so plainly. The
      rigorous statement is that the entire stack is one linear operator and the masks are a
      physically realisable parameterisation of it &mdash;
      <a class="link" href="@@HREF_physics@@">which is also where its ceiling comes from &rarr;</a></p>
    </div>
    <figure class="plate reveal">
      <img src="@@FIG_phase2_masks@@" alt="Five trained phase masks and one input-to-output intensity example for the diffractive network" loading="lazy">
      <figcaption><span class="fign">Fig 1</span>The five learned phase masks (top) and one worked
      example: an input digit field diffracting to its detector plane (bottom). Each mask is a fabricated
      surface relief; training only ever adjusted these phase profiles. The fine, high-spatial-frequency
      structure is exactly what makes the design hard to build.</figcaption>
    </figure>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">Choosing the task</p>
      <h2>Why a digit classifier, and only a digit classifier</h2></div></div>
    <div class="prose col">
      <p>MNIST is the smallest honest version of the job. It is a real classification problem with
      a real error rate, it fits inside a 128&sup2; field without contrivance, and &mdash; the part
      that matters here &mdash; it is <strong>weak enough that the optics stays the interesting
      part</strong>. The moment a task needs a serious electronic head to work, the optical network
      stops being the thing under study.</p>
      <p>So the same ten digits are reused at every stage: the diffractive stack, the interferometer
      mesh, and every one of the fabrication-error sweeps. One task, scored the same way on the same
      frozen 2,000-image test set, is what makes those results <strong>comparable to each
      other</strong> &mdash; which is worth far more here than a higher number on a harder benchmark
      would be. The machine-learning content is deliberately minimal and stays that way.</p>
    </div>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">Read the failures, not just the wins</p>
      <h2>It is wrong about one digit in four</h2></div></div>
    <div class="prose col">
      <p>The network scores <strong>0.799</strong> on MNIST, so the gallery deliberately includes
      digits it <strong>gets wrong</strong> &mdash; hiding them would misrepresent the model. Watch the
      power share when it fails: a confident answer takes ~25&ndash;30% of the output power, a wrong one
      usually much less.</p>
      <p>Drawings are harder still. Your strokes are <strong>out of distribution</strong> relative to
      MNIST no matter what we do, so expect more errors there. To keep the comparison fair rather than
      flattering, a drawing is normalised the way MNIST itself was built &mdash; cropped to the ink,
      scaled so its longer side is 20&nbsp;px, and centred by centre of mass &mdash; so a size or
      position mismatch cannot masquerade as an optical failure.</p>
      <p>What the browser computes is not an approximation of the trained model: on frozen test digits
      it reproduces PyTorch&rsquo;s predictions <strong>exactly</strong>, with class scores agreeing to
      better than 10<sup>&minus;3</sup>. The phases it carries are quantised to 8 bits, which is
      what a real modulator offers anyway; that costs <strong>nothing</strong> measurable
      (0.7995 against 0.7990, on 2,000 digits).</p>
    </div>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">Where this goes</p>
      <h2>What it would take to actually build one</h2></div></div>
    <div class="prose col">
      <p>Everything above ran in simulation, and simulation is where optical neural networks look
      easy. The masks are exact. The plates sit at exactly 3&nbsp;mm. Every pixel takes exactly the
      phase it was trained to take, and its neighbour&rsquo;s phase does not leak into it.</p>
      <p>A fabricated one has none of that. Etch depth varies, the modulator that writes a phase has
      finite resolution and a fringing field that smears each pixel into the next, the laser drifts,
      and light is lost at every surface. <strong>None of those are bugs to be fixed &mdash; they are
      the specification.</strong> The only question that decides whether this design could be built is
      how much of each it survives, and that is a number, not an opinion.</p>
      <p>Getting that number is what the rest of this project is. The trained parameters cross a
      one-directional handoff into an independent as-built model, which breaks the network on purpose,
      one imperfection at a time. <strong>One of the six sources fails against real hardware today</strong>
      &mdash; <a class="link" href="@@HREF_tolerance@@">and it is not the one you would guess &rarr;</a></p>
    </div>
  </section>

  @@NEXT@@

  <footer class="reveal">
    <div class="foot-grid">
      <div><h3>The physics</h3><p>Band-limited angular spectrum, ported to dependency-free JavaScript
        and cross-checked against the NumPy reference to &lt;10<sup>&minus;6</sup>. Six propagations and
        five phase masks per classification.</p></div>
      <div><h3>The parameters</h3><p>Exported straight from the trained PyTorch model &mdash; 81,920
        phase values, quantised to the 8&nbsp;bits a real modulator offers. Nothing is retrained or
        tuned for the browser.</p></div>
      <div><h3>Privacy</h3><p>Everything happens on your machine. Nothing you draw is uploaded, stored,
        or sent anywhere &mdash; the page makes no network requests at all.</p></div>
    </div>
    <p class="colophon">photonn &mdash; a portfolio study in optical computing and fabrication tolerance.
    Every physical constant on this page is cited in the source; unsourced values are flagged, never invented.</p>
  </footer>

</main>

@@PAGE_SCRIPT@@
@@D2NN_BUNDLE@@
@@D2NN_MOUNT@@
"""

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
</script>"""

# ------------------------------------------------------------------------ PHYSICS
# The propagator, the sampling limit that bounds it, and the ceiling that linearity
# puts on everything downstream. This is where the front page's "one linear
# operator" assertion is actually argued.
PHYSICS_BODY = r"""
<div class="spectral-rule"></div>
@@TOPBAR@@

<main class="wrap">

  <section class="hero col reveal">
    <p class="eyebrow">Scalar diffraction, and where it stops working</p>
    <h1>Everything rests on moving a field from one plane to the <em>next</em></h1>
    <div class="underbar"></div>
    <p class="standfirst">Before anything can be trained, coherent light has to be moved across
    3&nbsp;mm of air <b>exactly</b> &mdash; and the simulation has to know when it can no longer do
    that. One method does the moving, one criterion marks the edge, and one structural fact caps
    what the whole idea can ever compute.</p>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">The propagator</p>
      <h2>Decompose into plane waves, delay each one, add them back up</h2></div></div>
    <div class="prose col">
      <p>A field is decomposed into plane waves by a Fourier transform, each is advanced by its own
      axial phase, and they are recombined. That is the whole <strong>angular-spectrum method</strong>:
      transform, multiply by
      <span class="q">H = exp(i&thinsp;2&pi;z&thinsp;&radic;(1/&lambda;&sup2; &minus; f&sup2;))</span>,
      transform back. It makes <strong>no paraxial approximation</strong> &mdash; it is exact up to the
      grid&rsquo;s band limit, which is why Fresnel, Fraunhofer and the differentiable PyTorch layer
      are all checked against it rather than the other way round. Against an analytic Gaussian beam
      it agrees to <strong>~2&times;10<sup>&minus;8</sup></strong>.</p>
      <p>Two behaviours fall straight out of that square root. Where
      <span class="q">f&sup2; &gt; 1/&lambda;&sup2;</span> it turns imaginary, and computing it as a
      <em>complex</em> sqrt makes those components <strong>decay</strong> with distance &mdash;
      evanescent waves, for free, with no special case. And for large <span class="q">z</span> the
      transfer function <strong>oscillates faster than the frequency grid can sample</strong>, which
      is aliasing. Zeroing <span class="q">H</span> beyond the local-frequency limit keeps the method
      alias-free into the far field, at the cost of discarding content past the propagating band.</p>
      <p>The edge is a single criterion, <span class="q">z_crit = N&middot;dx&sup2;/&lambda;</span>
      &mdash; 15.4&nbsp;mm at this operating point, against 3&nbsp;mm hops. It is enforced at runtime,
      not only in tests. Note what it does <em>not</em> contain: <span class="q">z_crit</span> depends
      on <span class="q">&lambda;</span>, <span class="q">dx</span> and <span class="q">N</span>
      but <strong>not on the aperture</strong> &mdash; which is why, below, the aperture control does
      not move the sampling threshold no matter how far you drag it.</p>
    </div>
    <div class="explorer-band reveal">
      <div class="pe-host"><div id="explorer"></div></div>
      <p class="cap">Live &mdash; the diffraction runs on a hand-written FFT in your browser, the same
      physics as the Python reference (cross-checked to &lt;10<sup>&minus;6</sup>). Move any control;
      the sampling flag turns amber the moment <span style="white-space:nowrap">z &gt; z_crit</span>.</p>
    </div>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">Drawing it without lying about it</p>
      <h2>Why the 3D stack has haze in it and no rays</h2></div></div>
    <div class="prose col">
      <p>The light drawn <em>between</em> the mask planes on the front page is not decoration. A hop
      can be split into sub-hops and re-run because
      <span class="q">H(z&#8321;)&middot;H(z&#8322;) = H(z&#8321;+z&#8322;)</span>, and the one
      <span class="q">z</span>-dependent term that would break that equality &mdash; the band limit
      above &mdash; is <strong>inactive below <span class="q">z_crit</span></strong>. At 3&nbsp;mm
      against 15.4&nbsp;mm it is inactive, so the intermediate planes are <strong>exact fields</strong>,
      not interpolation. Above <span class="q">z_crit</span> the equality genuinely breaks, and the
      test suite asserts both halves of that.</p>
      <p><strong>No rays are drawn, deliberately.</strong> Scalar diffraction is not ray optics, and
      straight lines from digit to detector would misrepresent the one thing the figure exists to
      show &mdash; that every point of the input is reaching every detector at once. The stack is
      18&nbsp;mm long across a 1.02&nbsp;mm aperture, about 18:1, so drawn to scale it is an
      unreadable needle; the depth axis is compressed and the figure states its own compression
      factor on its face rather than quietly flattering the geometry.</p>
      <p>The drawing also <strong>cannot touch the answer</strong>. The prediction still comes from
      the canonical six propagations, and the cross-check asserts the logits are bit-identical with
      slicing switched on.</p>
    </div>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">The limit, stated precisely</p>
      <h2>Optical depth is not depth in the machine-learning sense</h2></div></div>
    <div class="prose col">
      <p>Propagation is linear in the field. A phase mask is linear in the field. Their composition
      is therefore a <em>single</em> linear operator:</p>
      <p><span class="q">E_out = M &middot; E_in</span>, with
      <span class="q">M = P_L D_L P_{L&minus;1} D_{L&minus;1} &hellip; D&#8321; P&#8320;</span> &mdash;
      each <span class="q">P</span> a propagation, each <span class="q">D</span> a diagonal mask.</p>
      <p>No matter how many mask-and-propagation layers are stacked, the field-to-field map collapses
      to one linear map. Extra layers add parameters and let <span class="q">M</span> better
      approximate a desired operator <em>under the physical constraint</em> that it factorise into
      phase-only masks separated by diffraction &mdash; but they add <strong>no nonlinear
      compositional power whatsoever</strong>. Stacking here is not stacking in the sense the phrase
      normally carries.</p>
      <p>The only nonlinearity is the detection itself. The logit for class <em>c</em> is the
      intensity summed over its detector region, which works out to
      <span class="q">s_c = E_in&dagger; A_c E_in</span> with
      <span class="q">A_c = M&dagger; R_c M &#8827; 0</span> &mdash; a positive-semidefinite
      <strong>quadratic form</strong> in the input field. So the classifier computes one PSD quadratic
      form per class and takes the argmax. That is strictly more than a linear classifier, and it is
      exactly <em>one</em> nonlinear layer: linear optical transform, magnitude-square, linear sum.</p>
      <p>One consequence is worth pulling out, because it is the only place a nonlinearity touches
      the raw pixels: <strong>the input encoding</strong>. Writing the image into phase applies a
      nonlinear <span class="q">exp(i&middot;&pi;&middot;image)</span> map before the optics ever sees
      it, which is why the encoding choice measurably changes achievable accuracy. It is doing
      nonlinear work the linear optics cannot.</p>
      <p>Per the project&rsquo;s scope this limit is <strong>characterised, not engineered around</strong>.
      No physical activation functions, no nonlinear media, no larger electronic head. If a bigger
      electronic head would lift accuracy, that is a finding to report, not a bug to fix.</p>
      <div class="finding">
        <p class="tag">Correction &middot; kept on the record</p>
        <p class="body">An earlier version of this project read the shipped network&rsquo;s
        <strong>0.799 plateau</strong> as this ceiling being reached &mdash; measured rather than
        asserted. <strong>That inference was wrong,</strong> and it is left here rather than quietly
        removed. The operator argument above is untouched; what did not follow was concluding that
        five masks had already exhausted the approximation. They had not, by at least eight points.
        The plateau belonged to <em>that geometry</em>, and
        <a class="link" href="@@HREF_optics@@">sweeping the geometry is a page of its own &rarr;</a></p>
      </div>
    </div>
  </section>

  @@NEXT@@

  <footer class="reveal">
    <div class="foot-grid">
      <div><h3>Verified against analytics</h3><p>Angular spectrum against an analytic Gaussian to
        ~2&times;10<sup>&minus;8</sup>; Fresnel against ASM in its validity range to ~9&times;10<sup>&minus;9</sup>;
        Fraunhofer of a circular aperture against the Airy pattern, first null on the predicted ring.</p></div>
      <div><h3>Enforced, not assumed</h3><p>Sampling is a runtime check, not just a test:
        <span class="q">check_sampling</span> returns a report and the explorer flags the crossing
        live. Beyond <span class="q">z_crit</span> the method does not fail loudly &mdash; which is
        precisely why it is flagged.</p></div>
      <div><h3>Scope</h3><p>Scalar theory only: no vector or polarisation-resolved propagation, and no
        full-wave solver. If real component data is ever wanted it gets imported as measurements, not
        simulated here.</p></div>
    </div>
    <p class="colophon">photonn &mdash; a portfolio study in optical computing and fabrication tolerance.
    Every physical constant on this page is cited in the source; unsourced values are flagged, never invented.</p>
  </footer>

</main>

@@PAGE_SCRIPT@@
@@EXPLORER_BUNDLE@@
@@EXPLORER_MOUNT@@
"""

# --------------------------------------------------------------------------- CHIP
# The mesh, and the correspondence between it and the diffractive stack. The best
# writing on the site is in here ("why they are the same machine"), so it is kept
# whole and the promoted docs material is arranged around it.
CHIP_BODY = r"""
<div class="spectral-rule"></div>
@@TOPBAR@@

<main class="wrap">

  <section class="hero col reveal">
    <p class="eyebrow">Free space and silicon, side by side</p>
    <h1>The same machine, built <em>two ways</em></h1>
    <div class="underbar"></div>
    <p class="standfirst">A stack of etched glass and a chip of waveguides look like unrelated
    devices. Strip both to their skeletons and the <b>same sequence appears</b>: a layer of phases
    you train, a layer of fixed hardware that mixes channels, repeated, closed by a square-law
    detector. They differ on exactly one number &mdash; and that number sets everything else.</p>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">The other optical computer</p>
      <h2>An interferometer is two couplers around a phase</h2></div></div>
    <div class="prose col">
      <p>A <strong>Mach&ndash;Zehnder interferometer</strong> is not exotic hardware. It is two 50:50
      directional couplers enclosing an internal phase <span class="q">&theta;</span>, preceded by an
      external phase <span class="q">&phi;</span> &mdash; the product
      <span class="q">B&middot;P(&theta;)&middot;B&middot;P(&phi;)</span>, which is unitary for every
      setting of the two phases. Split the light, delay one arm, recombine: the interference decides
      how much leaves by each output. Tile those on alternating adjacent pairs of waveguides and you
      have a <strong>Clements rectangle</strong>, <span class="q">n(n&minus;1)/2</span> of them.</p>
      <p>Such a mesh can realise <em>any</em> unitary, and the proof is constructive: you decompose a
      target by <strong>nulling its elements</strong> one at a time with interferometers &mdash; from
      the right only for the triangular Reck mesh, from both sides for the balanced Clements
      rectangle, which is half the optical depth and more robust to loss. Put a diagonal of
      amplitudes between two meshes and you have <em>any</em> real matrix by its singular-value
      decomposition, <span class="q">U&middot;&Sigma;&middot;V&dagger;</span>. Both decompositions
      reconstruct Haar-random unitaries to <strong>~10<sup>&minus;15</sup></strong>, which is this
      side&rsquo;s correctness anchor.</p>
      <p>Trained on the same digits it scores <strong>0.736</strong> &mdash; just shy of the
      diffractive net&rsquo;s 0.799, but with <strong>2,628 parameters against 81,920, about
      31&times; fewer</strong>, and implementing an arbitrary linear map rather than a constrained
      one. The gap is not a modelling failure: the mesh&rsquo;s footprint grows as
      <span class="q">N&sup2;/2</span>, so ingesting a big input is prohibitive and MNIST has to be
      squashed to 6&times;6 = 36 modes. <strong>It starves on input dimensionality, not on
      expressiveness.</strong> That footprint-versus-input trade is the central structural difference
      between the two processors.</p>
      <p>One finding falls out of the physics rather than the training. A passive mesh
      <strong>cannot amplify</strong> &mdash; it is lossless at best &mdash; so a physically
      realisable <span class="q">&Sigma;</span> needs every singular value at or below 1. Several of
      the trained ones exceed it. A real device would need gain, or a global rescale paid for in
      loss.</p>
    </div>

    <div class="stats col">
      <div class="s"><div class="v">0.736</div><div class="l">accuracy vs 0.799 (D&sup2;NN)</div></div>
      <div class="s"><div class="v">2,628</div><div class="l">parameters &middot; ~31&times; fewer</div></div>
      <div class="s"><div class="v">36 modes</div><div class="l">72 serial MZI layers</div></div>
      <div class="s"><div class="v">1e&minus;15</div><div class="l">Clements/Reck reconstruction error</div></div>
    </div>
    <figure class="plate reveal">
      <img src="@@FIG_mesh_topology@@" alt="MZI mesh topology and the learned singular-value spectrum" loading="lazy">
      <figcaption><span class="fign">Fig 1</span>Left: the rectangular MZI mesh topology. Right: the
      learned singular-value spectrum &mdash; effectively low-rank, only ~15&ndash;20 of 36 values carry
      weight, which is why so few parameters suffice.</figcaption>
    </figure>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">The correspondence</p>
      <h2>Why they are the same machine</h2></div></div>
    <div class="prose col">
      <p>A chip of waveguides looks nothing like a stack of etched glass, and everything above reads
      as two unrelated devices. They are not. Strip both to their skeletons and the same sequence
      appears: <strong>a layer of phases you train, a layer of fixed hardware that mixes channels,
      repeated, closed by a square-law detector.</strong> A phase mask <em>is</em> a column of phase
      shifters. A 3&nbsp;mm air gap <em>is</em> a column of couplers. In both machines you only ever
      train phases; in both, the mixing is unprogrammable.</p>
    </div>
    <div class="tbl-wrap col">
      <table class="tbl">
        <thead><tr><th></th><th>D&sup2;NN &mdash; free space</th><th>MZI mesh &mdash; chip</th></tr></thead>
        <tbody>
          <tr><td>a &ldquo;channel&rdquo; is</td><td>one pixel of the field &mdash; 16,384 of them</td><td>one waveguide mode &mdash; 36 of them</td></tr>
          <tr><td>the <strong>trainable</strong> part</td><td>a phase mask, <span class="q">e^{i&phi;}</span> per pixel</td><td>a phase shifter, <span class="q">e^{i&phi;}</span> per MZI arm</td></tr>
          <tr><td>set in hardware by</td><td>etched surface relief, or an SLM pixel</td><td>a thermo-optic heater current</td></tr>
          <tr><td>the <strong>fixed</strong> part</td><td>3&nbsp;mm of air &mdash; diffraction</td><td>a 50:50 directional coupler</td></tr>
          <tr><td>readout</td><td>integrated intensity, 10 detector boxes</td><td>intensity on the first 10 output modes</td></tr>
        </tbody>
      </table>
    </div>
    <div class="prose col">
      <p>This is <strong>a shared abstraction, not shared hardware.</strong> 16,384 pixels and 36
      modes are nowhere near the same scale, and no rearrangement turns one into the other. What
      transfers is the skeleton, not the device.</p>
      <p>They part on exactly one axis &mdash; <strong>how far one mixing layer moves information
      sideways</strong> &mdash; and that single number sets depth, footprint and failure mode.
      Diffraction hands you a wide reach for free, but you cannot <em>choose</em> it: the 12.5&nbsp;px
      per hop is fixed by <span class="q">z</span>, <span class="q">&lambda;</span> and the pixel
      pitch, and it is the same operator for every pixel. A coupler reaches exactly one neighbour, so
      you need as many columns as modes &mdash; and in exchange each of those couplings is steered
      individually, which is what makes any unitary realisable. <strong>Free space &rarr; chip trades
      one wide, fixed, unsteerable mixing operator for N sparse ones you can steer.</strong> The rest
      is packaging.</p>
    </div>

    <div class="explorer-band reveal">
      <div class="pe-host"><div id="analogy"></div></div>
      <p class="cap">Live &mdash; every number here is read from the trained models, not typed into
      the figure. The mesh drawn is the actual Clements schedule; the cone is
      <span style="white-space:nowrap">z&middot;&lambda;/(2&middot;dx&sup2;)</span> per hop.</p>
    </div>

    <div class="finding col reveal">
      <p class="tag">Finding &middot; the D&sup2;NN is connected by 0.8 px</p>
      <p class="body">Six hops of 3&nbsp;mm give each pixel a reach of <strong>74.8&nbsp;px</strong>.
      The worst case the design has to cover &mdash; an input pixel at one edge of the entrance
      window influencing the detector pixel farthest from it &mdash; is <strong>74&nbsp;px</strong>.
      So every input pixel <em>can</em> reach every detector, with <strong>0.81&nbsp;px of margin,
      about 1%</strong>. Shrink the mask separation below <strong>2.967&nbsp;mm</strong> and parts of
      the input become physically invisible to parts of the readout, whatever the masks are set to.
      Nothing in training knew about this bound &mdash; the operating point happens to clear it. The
      mesh has no such fragility: 36 columns for 36 modes is the Clements bound exactly, so full
      connectivity is guaranteed by the topology. <strong>One machine&rsquo;s connectivity is an
      accident that holds by 1%; the other&rsquo;s is a theorem.</strong></p>
    </div>

    <div class="prose col">
      <p>A bound on what <em>can</em> couple is not a claim about how much power actually does: the
      corner of the cone is the Nyquist ray, which carries little energy in practice. It says where
      the design sits relative to a hard limit, and the answer is &ldquo;just inside it, by
      accident&rdquo;.</p>
      <p>The two machines also <strong>share their ceiling</strong>. Everything the
      <a class="link" href="@@HREF_physics@@">physics page argues about linearity &rarr;</a> applies
      to the mesh unchanged: one linear optical transform followed by
      <span class="q">|&middot;|&sup2;</span> detection. Swapping free space for silicon buys
      steerability, footprint and manufacturability &mdash; it buys no expressiveness at all.</p>
      <p>Their <strong>failure modes</strong> differ, and that is the part this project is built to
      measure. The mesh&rsquo;s error is dominated by accumulation down 72 serial interferometers,
      plus coupler imbalance and per-MZI loss making the realised transfer sub-unitary. The
      diffractive net&rsquo;s errors act <em>in parallel</em>, per pixel, and are dominated by
      sub-pixel phase fidelity. <a class="link" href="@@HREF_tolerance@@">Two optical computers, two
      different ways to lose the computation to fabrication &rarr;</a></p>
      <ol class="refs">
        <li><b>1</b>M. Reck, A. Zeilinger, H. J. Bernstein &amp; P. Bertani, &ldquo;Experimental
        realization of any discrete unitary operator,&rdquo; <em>Phys. Rev. Lett.</em> <b>73</b>, 58
        (1994) &mdash; the triangular mesh.</li>
        <li><b>2</b>W. R. Clements <em>et al.</em>, &ldquo;Optimal design for universal multiport
        interferometers,&rdquo; <em>Optica</em> <b>3</b>, 1460 (2016) &mdash; the rectangular mesh and
        the nulling algorithm.
        <a href="https://opg.optica.org/optica/abstract.cfm?uri=optica-3-12-1460">doi:10.1364/OPTICA.3.001460</a></li>
      </ol>
    </div>
  </section>

  <section class="phase planned reveal">
    <div class="phase-head col"><span class="ph-num next">next</span>
      <div><span class="badge-next">Planned</span>
      <p class="eyebrow">Quantum branch &middot; Boson sampling</p>
      <h2>Same mesh, single photons instead of a beam</h2></div></div>
    <div class="prose col">
      <p>The interferometer mesh has a second life. Send <strong>indistinguishable single photons</strong>
      through the very same trained unitary and the output statistics become permanent-based rather than
      intensity-based &mdash; the regime behind boson sampling. The planned deliverable computes those
      distributions, shows the <strong>Hong&ndash;Ou&ndash;Mandel dip</strong> (two photons on a 50:50
      coupler never leave separately), and contrasts the quantum output with the classical,
      distinguishable-particle case. Only the input state&rsquo;s statistics change; the transfer matrix
      is identical.</p>
    </div>
  </section>

  @@NEXT@@

  <footer class="reveal">
    <div class="foot-grid">
      <div><h3>Derived, never typed</h3><p>The correspondence figure reads its reach from the
        propagator, its detector layout from the detector module and its topology from the mesh
        schedule. A test re-derives all of it and fails if the exported geometry drifts.</p></div>
      <div><h3>Verified</h3><p>MZI unitarity across the phase range; Clements and Reck reconstruction
        to ~10<sup>&minus;15</sup>; the SVD real-matrix layer; the torch mesh against the NumPy
        reference.</p></div>
      <div><h3>Still open</h3><p>The mesh&rsquo;s own error budget, which reactivates the two sources
        that have no meaning for phase masks: coupler imbalance and per-MZI insertion loss. Its
        parameter set will need a PDK, not the modulator literature this study uses.</p></div>
    </div>
    <p class="colophon">photonn &mdash; a portfolio study in optical computing and fabrication tolerance.
    Every physical constant on this page is cited in the source; unsourced values are flagged, never invented.</p>
  </footer>

</main>

@@PAGE_SCRIPT@@
@@ANALOGY_BUNDLE@@
@@ANALOGY_MOUNT@@
"""

# ---------------------------------------------------------------------- TOLERANCE
# The destination. The project's central question gets its own page, and the eight
# candidate-L56 figures put the "depth costs tolerance" trade beside the shipped
# curves -- the first time that argument is shown rather than only asserted.
TOLERANCE_BODY = r"""
<div class="spectral-rule"></div>
@@TOPBAR@@

<main class="wrap">

  <section class="hero col reveal">
    <p class="eyebrow">The question the project exists to answer</p>
    <h1>How precisely must a photonic chip be built before it stops computing what it was <em>trained</em> to compute?</h1>
    <div class="underbar"></div>
    <p class="standfirst">Everything so far was ideal. The masks were exact, the planes were exactly
    3&nbsp;mm apart, and every pixel took exactly the phase it was trained to take. Now the trained
    parameters cross into an independent as-built model and get <b>broken on purpose</b>, one
    imperfection at a time, until the number that decides feasibility falls out.</p>
    <div class="stat-strip">
      <div class="stat"><span class="v">0.25<small> px</small></span><span class="l">of phase blur between neighbouring pixels is all it tolerates</span></div>
      <div class="stat"><span class="v">&asymp;1<small> px</small></span><span class="l">is what a real LCoS modulator actually delivers &mdash; it does not clear the bar</span></div>
      <div class="stat"><span class="v">0.7990</span><span class="l">reproduced exactly by the as-built model with zero error injected</span></div>
    </div>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">Method</p>
      <h2>Now break it on purpose</h2></div></div>
    <div class="prose col">
      <p>The trained parameters cross a <strong>one-directional handoff</strong> &mdash; a single HDF5
      file carrying the phase masks, the geometry, the operating point and the frozen test set &mdash;
      into a MATLAB &ldquo;as-built&rdquo; model that never writes back. That boundary is the design;
      it keeps the ideal network and the imperfect one from contaminating each other. Nothing is
      retrained on the far side, and no error model exists on the near side.</p>
      <p>That model then re-scores the exact same network under each fabrication imperfection in turn
      &mdash; per-pixel phase error, DAC quantisation, optical loss, wavelength drift, thermal
      crosstalk, detector and shot noise &mdash; Monte Carlo over realisations with the seeds
      recorded, <strong>every magnitude traced to a published measurement</strong> and cited inline
      at the point of use.</p>
      <p>One control makes it a measurement rather than a demonstration: <strong>with zero error
      injected the as-built model reproduces 0.7990 exactly</strong>, the same number the PyTorch
      model scores on the same 2,000 images. Two independent implementations agreeing to the last
      digit means any drop below it is fabrication and nothing else. That anchor has held through
      every change in the project.</p>
      <p>The bar throughout is <strong>95% of ideal accuracy</strong>, which is
      <span class="q">&ge; 0.7591</span>. Edges are quoted as the bracket the sweep actually resolves
      &mdash; <em>holds at X, fails at Y</em> &mdash; rather than interpolated between grid points, so
      they stay comparable across models.</p>
    </div>
    <div class="finding col reveal">
      <p class="tag">Binding constraint</p>
      <p class="body"><strong>Thermal / pixel crosstalk sets the tolerance, and it is the one that
      fails.</strong> Accuracy holds while the phase blur between neighbouring pixels stays at
      <strong>0.25&nbsp;px</strong> and fails by <strong>0.5&nbsp;px</strong>; three quarters of a
      pixel of blur collapses the network to near chance. A standard LCoS modulator&rsquo;s
      fringing-field crosstalk is about <strong>1&nbsp;px</strong> &mdash; so this design, as
      specified, <strong>would not work on one</strong>. The ranking is unambiguous: crosstalk
      &raquo; phase error &gt; detector power &asymp; loss &raquo; wavelength &asymp; quantisation.
      To hold <span class="q">&ge; 0.7591</span>: phase &sigma; &le; <strong>0.3&nbsp;rad</strong>
      (&lambda;/21), <strong>&ge;3-bit</strong> DAC, drift &le; <strong>10&nbsp;nm</strong>, and
      <strong>&ge;1&nbsp;pW</strong> over 1&nbsp;ms. The number that decides feasibility is not
      precision or bit depth &mdash; it is how sharply each pixel&rsquo;s phase is confined against
      its neighbours.</p>
    </div>
    <div class="prose col">
      <p>Read the ranking rather than the individual numbers. <strong>Phase calibration and bit depth
      are comfortable</strong>: a well-calibrated modulator sits at &lambda;/100 &asymp; 0.05&nbsp;rad,
      an order of magnitude inside the 0.3&nbsp;rad that still holds, and 3-bit control suffices
      against an 8-bit standard. <strong>Wavelength is a non-issue</strong> for any
      temperature-controlled source. <strong>Detection is nine orders of magnitude</strong> from
      shot-noise-limited at the nominal operating point.</p>
      <p><strong>Loss is the interesting one</strong>, because it does not act alone. Uniform
      attenuation cancels exactly in the power-normalised readout, so when photons are plentiful it
      has <em>zero</em> effect on accuracy. It only bites through the photon budget &mdash; which is
      why it is swept at the shot-noise knee, where every 3&nbsp;dB halves the photon count. Loss and
      the power budget have to be reasoned about together or the sweep measures nothing.</p>
      <p>And <strong>a better-trained network was not a more fragile one</strong>, which is worth
      stating because the opposite is a reasonable prior. Retraining on five times more data lifted
      the ideal baseline from 0.7695 to 0.7990 and every edge was re-measured against a
      correspondingly stricter bar. Five of the six landed in the same bracket. Masks fitted to more
      data did not carry finer, more brittle structure &mdash; but that result belongs to
      <em>training</em>, and the next section is what happened when the <em>optics</em> changed
      instead.</p>
    </div>
    <div class="plate-grid reveal">
      <figure class="plate"><img src="@@FIG_tol_crosstalk@@" alt="Accuracy vs thermal/pixel crosstalk" loading="lazy"><figcaption><span class="fign">Fig 1</span>Crosstalk &mdash; the binding constraint.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_tol_phase@@" alt="Accuracy vs per-pixel phase error" loading="lazy"><figcaption><span class="fign">Fig 2</span>Per-pixel phase-setting error.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_tol_detector@@" alt="Accuracy vs detector noise / input power" loading="lazy"><figcaption><span class="fign">Fig 3</span>Detector &amp; shot noise vs input power.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_tol_loss@@" alt="Accuracy vs optical insertion loss" loading="lazy"><figcaption><span class="fign">Fig 4</span>Optical insertion loss, swept at the knee.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_tol_wavelength@@" alt="Accuracy vs wavelength drift" loading="lazy"><figcaption><span class="fign">Fig 5</span>Laser wavelength drift.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_tol_quant@@" alt="Accuracy vs DAC bit resolution" loading="lazy"><figcaption><span class="fign">Fig 6</span>DAC / SLM bit resolution.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_confusion@@" alt="As-built confusion matrix at phase sigma 0.35 rad" loading="lazy"><figcaption><span class="fign">Fig 7</span>As-built confusion matrix at &sigma;=0.35&nbsp;rad.</figcaption></figure>
    </div>
    <figure class="plate reveal">
      <img src="@@FIG_sensitivity@@" alt="Per-mask spatial sensitivity map" loading="lazy">
      <figcaption><span class="fign">Fig 8</span>Spatial sensitivity &mdash; where on each mask a phase
      error costs the most accuracy. Sensitivity is not uniform, which is what makes a per-pixel
      tolerance meaningful.</figcaption>
    </figure>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">The result this project actually has</p>
      <h2>What depth costs</h2></div></div>
    <div class="prose col">
      <p>A deeper network is <strong>a lot</strong> more accurate. Spending the same diffractive reach
      budget on 56 masks instead of 5 reaches <strong>0.9040</strong> on the same frozen test set,
      against 0.7990 &mdash; and it is <a class="link" href="@@HREF_optics@@">running live, beside the
      shipped model &rarr;</a> so the improvement is easy to see and easy to mistake for a
      straightforward upgrade.</p>
      <p>It is not one. The whole budget above was re-run against that candidate, scored against its
      own correspondingly stricter bar of 0.8588, and <strong>+10.5 points of accuracy is priced in
      fabrication tolerance</strong>:</p>
    </div>
    <div class="tbl-wrap col">
      <table class="tbl">
        <thead><tr><th>Error source</th><th>Shipped &middot; 5 masks</th><th>Candidate &middot; 56 masks</th><th>Change</th></tr></thead>
        <tbody>
          <tr><td><strong>Thermal / pixel crosstalk</strong></td><td>holds 0.25&nbsp;px, fails 0.5</td><td>holds <strong>0.25&nbsp;px</strong>, fails 0.5</td><td><strong>unchanged &mdash; still binding</strong></td></tr>
          <tr><td>Per-pixel phase error</td><td>holds 0.3&nbsp;rad, fails 0.5</td><td>holds <strong>0.15&nbsp;rad</strong>, fails 0.2</td><td>2&times; tighter</td></tr>
          <tr><td>DAC / SLM resolution</td><td>holds 3 bits, fails 2</td><td>holds <strong>4 bits</strong>, fails 3</td><td>1 bit tighter</td></tr>
          <tr><td>Optical loss at the knee</td><td>holds 1&nbsp;dB/mask (5&nbsp;dB total)</td><td>holds <strong>0.214&nbsp;dB/mask</strong> (12&nbsp;dB total)</td><td>4.7&times; tighter per mask</td></tr>
          <tr><td>Detector / shot noise</td><td>holds 1&nbsp;pW, fails 0.1</td><td>holds <strong>0.1&nbsp;pW</strong>, fails 0.01</td><td>10&times; looser</td></tr>
          <tr><td>Wavelength drift</td><td>holds 10&nbsp;nm, fails 20</td><td>holds <strong>20&nbsp;nm</strong>, fails 30</td><td>2&times; looser</td></tr>
        </tbody>
      </table>
    </div>
    <div class="prose col">
      <p>Three readings are worth separating. <strong>The binding constraint does not move at
      all</strong> &mdash; crosstalk fails at the same 0.25&nbsp;px edge, and a real modulator&rsquo;s
      ~1&nbsp;px fringing field destroys either design. Depth neither helps nor hurts the thing that
      already made this unbuildable.</p>
      <p><strong>The two sources that loosened both follow from photon capture.</strong> The deep
      stack routes <strong>79.1%</strong> of input photons into the detector boxes against ~60% for
      the shipped design &mdash; the same &ldquo;route rather than scatter&rdquo; mechanism the
      accuracy gain comes from. Better signal at the readout drops the shot-noise knee a decade and
      buys wavelength margin. The design became more robust exactly where it already passed by nine
      orders of magnitude.</p>
      <p><strong>Loss points opposite ways in its two units, and the per-element one governs.</strong>
      In total the candidate tolerates <em>more</em> attenuation &mdash; 12&nbsp;dB against 5 &mdash;
      but that larger budget is divided among eleven times more elements, so the per-mask requirement
      tightens to 0.214&nbsp;dB. A datasheet quotes per element, and 0.214&nbsp;dB/mask sits at the
      optimistic end of the realistic 0.2&ndash;1&nbsp;dB range. Loss moves from comfortable to
      marginal.</p>
      <div class="finding">
        <p class="tag">Finding &middot; depth adds a constraint without relieving one</p>
        <p class="body">Ten and a half points of accuracy cost <strong>2&times; tighter phase
        control, one more DAC bit and 4.7&times; tighter loss per element</strong>, while the source
        that already fails against real hardware <strong>does not move</strong>. This is why the deep
        model is labelled &ldquo;not shipped&rdquo; wherever it appears: its number is real and
        measured exactly like the headline, but what it costs is a build tolerance this project has
        no evidence anyone can hit. <strong>The trade is a more useful result than a clean win would
        have been.</strong></p>
      </div>
      <p>There is also something the budget <em>cannot</em> say. At 56 masks the plane spacing falls
      from 3&nbsp;mm to 0.53&nbsp;mm, so a &plusmn;10&nbsp;&micro;m spacing error goes from 0.33% of
      the gap to 1.9% &mdash; and past roughly forty plates the stack is better described as a
      <strong>volume of glass</strong> than as discrete phase screens, which is a different thing to
      fabricate. This budget covers <strong>device errors only</strong> and has nothing on geometry.
      Alignment and calibration &mdash; plane spacing, lateral registration, systematic phase gain,
      detector offset &mdash; plausibly bind the deep design before anything in the table does.
      <strong>That is flagged, not measured</strong>, and it is the next error sources to be written.</p>
    </div>
    <div class="plate-grid reveal">
      <figure class="plate"><img src="@@FIG_cand_crosstalk@@" alt="Candidate 56-mask network: accuracy vs thermal/pixel crosstalk" loading="lazy"><figcaption><span class="fign">Fig 9</span>Candidate crosstalk &mdash; the same edge as the shipped design.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_cand_phase@@" alt="Candidate 56-mask network: accuracy vs per-pixel phase error" loading="lazy"><figcaption><span class="fign">Fig 10</span>Candidate phase error &mdash; twice as tight.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_cand_detector@@" alt="Candidate 56-mask network: accuracy vs detector noise" loading="lazy"><figcaption><span class="fign">Fig 11</span>Candidate detector noise &mdash; a decade looser.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_cand_loss@@" alt="Candidate 56-mask network: accuracy vs optical loss" loading="lazy"><figcaption><span class="fign">Fig 12</span>Candidate loss, swept as total dB across 56 masks.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_cand_wavelength@@" alt="Candidate 56-mask network: accuracy vs wavelength drift" loading="lazy"><figcaption><span class="fign">Fig 13</span>Candidate wavelength drift &mdash; twice as loose.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_cand_quant@@" alt="Candidate 56-mask network: accuracy vs DAC bit resolution" loading="lazy"><figcaption><span class="fign">Fig 14</span>Candidate bit depth &mdash; one bit tighter.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_cand_confusion@@" alt="Candidate 56-mask network: as-built confusion matrix" loading="lazy"><figcaption><span class="fign">Fig 15</span>Candidate as-built confusion matrix.</figcaption></figure>
    </div>
    <div class="prose col">
      <p>There is no candidate version of Fig&nbsp;8. The sensitivity map is one panel per mask in a
      single row, so at 56 masks it is <strong>16,139&nbsp;pixels wide and 341 tall</strong> &mdash;
      47:1, which is fifteen pixels high in a grid cell and thirty across the full column. There is
      no size at which it can be read, so it is left out rather than shown as a smear. It also costs
      <strong>2,016 evaluations</strong> to compute against 36 for the shipped design, about four
      hours and most of a full run. Both of those are the same fact from different directions: past
      roughly forty plates, per-mask quantities stop being a thing you can look at.</p>
    </div>
  </section>

  <section class="phase planned reveal">
    <div class="phase-head col"><span class="ph-num next">next</span>
      <div><span class="badge-next">Planned</span>
      <p class="eyebrow">Error budget &middot; MZI mesh</p>
      <h2>Fabrication tolerance for the interferometer mesh</h2></div></div>
    <div class="prose col">
      <p>The same framework extends to the mesh, reactivating the two error sources unique
      to it: <strong>coupler imbalance</strong> (deviation from 50:50) and <strong>per-MZI insertion
      loss</strong>, which makes the transfer sub-unitary &mdash; a real effect there, unlike the
      normalised diffractive readout. The interesting comparison is the failure mode: <strong>serial
      phase-error accumulation down 72 MZI layers</strong> versus the diffractive net&rsquo;s
      crosstalk-dominated per-pixel budget. Two optical computers, two different ways to lose the
      computation to fabrication.</p>
    </div>
  </section>

  @@NEXT@@

  <footer class="reveal">
    <div class="foot-grid">
      <div><h3>The boundary</h3><p>Python designs the ideal network; a single HDF5 file crosses to
        MATLAB, which models the as-built device and never writes back. Design versus as-built,
        enforced by a one-directional handoff rather than by discipline.</p></div>
      <div><h3>Every magnitude sourced</h3><p>Phase error, bit depth, drift, read noise, insertion
        loss and fringing-field crosstalk each trace to a published measurement or datasheet, cited
        on the line the constant is defined. Modelling choices are listed as modelling choices.</p></div>
      <div><h3>Reproducible</h3><p>Deterministic given the recorded seeds; the ideal baseline must
        read 0.7990 or the forward model is misaligned with the handoff. A candidate model can be
        scored without touching the shipped one.</p></div>
    </div>
    <p class="colophon">photonn &mdash; a portfolio study in optical computing and fabrication tolerance.
    Every physical constant on this page is cited in the source; unsourced values are flagged, never invented.</p>
  </footer>

</main>

@@PAGE_SCRIPT@@
"""

# ------------------------------------------------------------------------ OPTICS
# The last page in the reading order, and the only one that is *live work*: the
# other four state the machine as built, this one tracks what the optics could
# still be, and the two must not be confused. Everything here is measured against
# a short ranking protocol, so every number on the page says so.
OPTICS_BODY = r"""
<div class="spectral-rule"></div>
@@TOPBAR@@

<main class="wrap">

  <section class="hero col reveal">
    <p class="eyebrow">Live work &middot; not a result</p>
    <h1>How much better could the <em>optics</em> be?</h1>
    <div class="underbar"></div>
    <p class="standfirst">The shipped network scores 0.799 and cannot be pushed further by training:
    after 40 epochs on 60,000 images it still cannot pull ahead on its own training set. So the
    remaining levers are physical &mdash; <b>how far apart the masks sit</b>, and <b>how many there
    are</b>. This page is the running record of measuring them. <b>Nothing here is shipped yet.</b></p>
    <div class="stat-strip">
      <div class="stat"><span class="v">0.771</span><span class="l">shipped geometry, ranking protocol</span></div>
      <div class="stat"><span class="v">0.852</span><span class="l">14 masks, same reach budget</span></div>
      <div class="stat"><span class="v">0.904</span><span class="l">56 masks, full budget, frozen test set</span></div>
    </div>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">The constraint</p>
      <h2>A detector can only be reached by light that gets to it</h2></div></div>
    <div class="prose col">
      <p>One diffractive hop mixes light sideways by a fixed amount,
      <span class="q">reach = z&middot;&lambda;/(2&middot;dx&sup2;)</span> &mdash; about 12.5&nbsp;px per
      3&nbsp;mm gap at this operating point. For the network to be able to compute anything at all,
      every input pixel must be able to influence every detector, and the worst case here is
      <strong>74&nbsp;px</strong>. The shipped design clears it by <strong>0.8&nbsp;px</strong>.</p>
      <p>Below the bound the failure is not statistical, it is <em>geometric</em>: part of the digit
      physically cannot reach the detector that needs it, whatever the masks were trained to. Drag the
      separation and watch the cone fall short.</p>
    </div>

    <div class="explorer-band reveal">
      <div class="pe-host"><div id="optics"></div></div>
      <p class="cap">Live &mdash; the cone is recomputed from the closed form as you drag, the same
      expression as <span class="q">propagate.diffraction_reach_px</span>. The plotted points are
      measured training runs.</p>
    </div>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">The trade</p>
      <h2>Separation and depth spend the same budget</h2></div></div>
    <div class="prose col">
      <p>The grid is finite and the propagator uses an ordinary FFT, which is periodic &mdash; light
      leaving one edge reappears on the other. That wrap turns out to depend only on <em>total</em>
      reach, not on how it is split: 2&nbsp;mm over 9 hops and 3&nbsp;mm over 6 hops both reach
      74.8&nbsp;px and are both wrong by an identical 6.2&times;10<sup>&minus;4</sup>. So distance and
      masks draw on <strong>one shared budget</strong>, about 150&nbsp;px before the simulation stops
      describing free space.</p>
      <p>Which makes the interesting question not &ldquo;more of which?&rdquo; but <strong>how to spend
      a fixed budget</strong>. Holding total reach at 125&nbsp;px and trading distance for masks &mdash;
      every configuration with identical reach <em>and</em> identical wrap error &mdash; accuracy runs
      <strong>0.706 &rarr; 0.790 &rarr; 0.831 &rarr; 0.852</strong> from 2 masks to 14.</p>

      <div class="finding">
        <p class="tag">Finding</p>
        <p class="body"><strong>The plateau belonged to the geometry, not the architecture.</strong>
        This project used to read the shipped network&rsquo;s 0.799 as the ceiling set by being
        <a class="link" href="@@HREF_physics@@">one linear operator and one intensity readout
        &rarr;</a> The operator part is true; the ceiling part was not. Fourteen masks reach
        <strong>0.852</strong> on a third of the data and under a third of the epochs &mdash; and had
        not converged when the run stopped.</p>
      </div>
    </div>

    <figure class="plate reveal">
      <img src="@@FIG_optics_sweep@@" alt="Optics sweep: accuracy against diffractive reach, the reach-budget trade, and detector planes per configuration" loading="lazy">
      <figcaption><span class="fign">Fig 1</span>Left: accuracy against total reach at the shipped
      5 masks, 74&nbsp;px bound marked; ringed points are configurations whose <em>simulation</em> is
      wrap-contaminated, not designs that lost fairly. Right: the same 125&nbsp;px of reach spent on
      different numbers of masks. Below: the diffraction cone and the detector plane for one fixed
      digit, worst to best.</figcaption>
    </figure>

    <div class="prose col">
      <p>The detector planes are the clearest part. At 2 and 5 masks the light arrives as a diffuse
      interference smear; at 9 and 14 it is gathered into <strong>discrete bright squares sitting on
      the detector patches</strong>. Same reach, same physics &mdash; the extra masks buy the ability to
      <em>route</em> light into the readout rather than merely scatter it there.</p>
    </div>

    <h3 class="sub-h">See it for yourself</h3>
    <div class="prose col">
      <p>Since more masks keep paying, the deepest configuration worth the compute was trained
      properly rather than ranked: <strong>56 masks</strong> at 0.53&nbsp;mm gaps, the full 60,000
      images, then scored once on the same frozen test set the shipped model is quoted from. It
      reaches <strong>0.9040</strong> against <strong>0.7990</strong>. Both machines run here, live,
      on whichever digit you pick &mdash; or one you draw &mdash; the same angular-spectrum physics
      as the Python reference, cross-checked against PyTorch to better than
      10<sup>&minus;3</sup>.</p>
      <p>The gallery is the one the front page uses, deliberately stocked with <strong>six
      digits the shipped network gets wrong</strong>. The deep network recovers three of them and
      breaks none of the ten the shipped one already had. The three it still misses, it misses
      <em>the same way</em> &mdash; same wrong class, both machines, which is a hint that those
      digits are hard for the optics rather than for this particular set of masks.</p>
      <p>Both are fed by <em>one</em> input, so the columns compare optics and nothing else. Watch
      the detector plane rather than the answer: the shipped network spreads light across the whole
      plane and reads a weak maximum off it, while the deep stack lands it inside the boxes. That is
      the difference between about 60% and <strong>79%</strong> of the input photons reaching a
      detector, and it is the same routing mechanism the accuracy gain comes from.</p>
      <p>Two trained models is <strong>0.7&nbsp;MB</strong> of phases on one page. The candidate
      carries <strong>eleven times</strong> the parameters &mdash; 918k against 82k &mdash; for under
      six times the download, because its phases are quantised harder: <strong>4&nbsp;bits</strong>
      against 8. Neither model is flattered by that. The error budget measures this design as holding
      accuracy down to <strong>3-bit</strong> phase control, and on the same 2,000 digits the encoded
      models score <strong>0.9015</strong> and <strong>0.7995</strong> against the float 0.9040 and
      0.7990 &mdash; five digits either way, inside the noise of a test set that size. What runs in
      your browser is therefore the same machine the study measured, not a lighter stand-in.</p>
      <p>A forward pass costs roughly <strong>11&nbsp;ms</strong> and <strong>100&nbsp;ms</strong>
      respectively &mdash; the deep machine is 56 diffraction steps of arithmetic, in a browser tab
      &mdash; which is far more than an animation frame can hold. So the board <strong>waits for you
      to pause</strong> rather than classifying mid-stroke, and it measures its own render cost to
      decide that: a board carrying only cheap models still follows the pen. The alternative &mdash;
      keeping the fast column live and deferring the deep one &mdash; was rejected on purpose,
      because it would show two different digits side by side.</p>
    </div>

    <div class="explorer-band reveal">
      <div class="pe-host"><div id="compare"></div></div>
      <p class="cap">Live &mdash; the shipped 5-mask network and the unshipped 56-mask candidate, one
      digit, both forward passes computed here. Every caption in the board is rendered from the
      weight bundles&rsquo; own provenance, so promoting a model is regenerating a bundle rather than
      editing a page.</p>
    </div>

    <h3 class="sub-h">What 56 masks actually looks like</h3>
    <div class="prose col">
      <p>The detector plane says the deep network is better; it does not say what it <em>is</em>.
      Here it is as a machine &mdash; the entrance plane, the phase plates and the detector plane
      drawn along the optical axis, each carrying the light computed on it. Drag to orbit.</p>
      <p>Two things about this figure are honest rather than convenient. It draws
      <strong>6 of the 56 masks</strong>, spread through the stack and each labelled with its real
      index, because fifty-six plates at 0.53&nbsp;mm spacing are fifty-six near-identical
      pictures. And it does not follow your pen: a redraw is a full pass through
      <strong>57 hops</strong>, so it waits for <span class="q">Refresh</span> rather than making
      the whole page stutter for a picture nobody reads mid-stroke.</p>
      <p>The proportions are the point. The shipped design is 18&nbsp;mm of optics across a
      1.02&nbsp;mm aperture; this one is <strong>30&nbsp;mm across the same aperture</strong>, a
      29:1 needle that has to be compressed almost tenfold just to fit on a screen. Past about
      forty plates at this spacing, calling it a stack of discrete masks is already a stretch
      &mdash; it is closer to a volume of glass, which is a different thing to fabricate and one
      the <a class="link" href="@@HREF_tolerance@@">error budget &rarr;</a> does not yet model.</p>
    </div>

    <div class="explorer-band reveal">
      <div class="pe-host"><div id="stage3d"></div></div>
      <p class="cap">Live &mdash; the 56-mask candidate, fed the same digit as the board above.
      Sampled planes, manual refresh; the haze between plates is the field at intermediate depths,
      which is exact here rather than a gradient.</p>
    </div>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">What this does not show</p>
      <h2>Reading it honestly</h2></div></div>
    <div class="prose col">
      <p>The <em>sweep</em> accuracies come from a deliberately <strong>short ranking protocol</strong>
      &mdash; 20,000 images for 12 epochs &mdash; because the question there is which geometry, not
      what final accuracy. They are <strong>not comparable to the 0.799 headline</strong>, which is a
      60,000-image, 40-epoch run. The fair comparison is the shipped geometry under the same short
      protocol: <strong>0.771</strong>. The 56-mask model in the board above is the exception: it was
      trained at the full budget and scored on the frozen test set, so its <strong>0.9040</strong> and
      the headline <strong>0.7990</strong> are the same measurement.</p>
      <p>In a diffractive network <strong>mask count is parameter count</strong> &mdash; 128&sup2;
      phases per mask &mdash; so the sweep on its own cannot separate &ldquo;depth helps&rdquo; from
      &ldquo;more parameters help&rdquo;. Two parameter-matched pairs settle it: 56 masks on a 128&sup2;
      grid and 14 on a 256&sup2; one both carry 917,504 phases and score <strong>0.889 against
      0.856</strong>; the 80/20 pair, both at 1,310,720 phases, gives 0.891 against 0.870. At equal
      parameter count, <strong>depth wins</strong>. Each configuration is one seed; the ordering
      across the range far exceeds run-to-run noise, adjacent points do not.</p>
      <p>Ranking never touched the frozen test set &mdash; that set is exported to the MATLAB
      as-built model and every downstream number is quoted from it, so a disjoint validation split was
      carved out of the training data instead. It was used once, at the end, to score the single
      configuration that had already won.</p>
      <p><strong>Nothing here is shipped, and the deep model is not simply better.</strong> The
      trained model, the <a class="link" href="@@HREF_index@@">browser classifier &rarr;</a> and
      the <a class="link" href="@@HREF_tolerance@@">error budget &rarr;</a> all still describe the
      5-mask, 3&nbsp;mm design. Running the full budget against the 56-mask candidate prices the
      +10.5 points:
      <strong>2&times; tighter per-pixel phase</strong> (0.3 &rarr; 0.15&nbsp;rad), <strong>one more
      DAC bit</strong>, and <strong>4.7&times; tighter loss per mask</strong>. Detector power and
      wavelength margin loosen, for the same photon-capture reason the accuracy rose &mdash; but
      thermal crosstalk, the source that already made this unbuildable on a real SLM, does not move at
      all. Depth adds a second binding constraint without relieving the first, and that trade, rather
      than the accuracy, is the result.</p>
    </div>
  </section>

  @@NEXT@@

  <footer class="reveal">
    <div class="foot-grid">
      <div><h3>The measurement</h3><p>Two sweeps: separation at fixed mask count, then the reach budget
        spent different ways. Wrap error is measured against a zero-padded reference and gates which
        configurations are worth training at all.</p></div>
      <div><h3>Reproducing it</h3><p><span class="q">apps/sweep_optics.py</span> runs the arms and
        checkpoints per configuration; <span class="q">apps/sweep_report.py</span> writes this figure
        and the data this page plots. Fixed seeds throughout.</p></div>
      <div><h3>Status</h3><p>Live work, not a result. The numbers here move as configurations finish;
        the shipped design and everything downstream of it stay where they are until a promotion is
        decided.</p></div>
    </div>
    <p class="colophon">photonn &mdash; a portfolio study in optical computing and fabrication tolerance.
    Every physical constant on this page is cited in the source; unsourced values are flagged, never invented.</p>
  </footer>

</main>

@@PAGE_SCRIPT@@
@@OPTICS_BUNDLE@@
@@OPTICS_MOUNT@@
@@COMPARE_BUNDLE@@
@@COMPARE_MOUNT@@
"""

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
    """Inline every figure the page actually references, and no others."""
    for key, path in FIGURES.items():
        token = f"@@FIG_{key}@@"
        if token in html:
            html = html.replace(token, encode_figure(path, **{**DEFAULT_OPT, **FIG_OPTS.get(key, {})}))
    return html


def _chrome(body: str, key: str, next_key: str, kicker: str = "Next") -> str:
    """Fill in everything every page shares: nav, hand-off, page script, figures.

    Link tokens are deliberately left in place -- ``resolve_links`` runs last, so
    one rendered body can be emitted twice, once relative and once absolute.
    """
    html = body.replace("@@TOPBAR@@", topbar(key))
    html = html.replace("@@NEXT@@", next_link(next_key, kicker))
    html = html.replace("@@PAGE_SCRIPT@@", mount_queue_bundle() + PAGE_SCRIPT)
    return _figures(html)


def render() -> dict:
    """Return ``{filename: html}`` for every file the site is made of."""
    out = {}

    # The front page carries the whole engine: no explorer runs here, so it
    # inlines asm.js itself.
    body = BODY.replace("@@D2NN_BUNDLE@@", d2nn_bundle(include_asm=True))
    body = body.replace("@@D2NN_MOUNT@@", d2nn_mount("d2nn", stage_id="stage"))
    body = _chrome(body, "index", "physics")
    out["index.html"] = _document(resolve_links(body), PAGE_BY_KEY["index"])
    # The artifact is a standalone body with no sibling pages, so its links must
    # be absolute and it supplies no <head> of its own.
    out["_artifact_body.html"] = ("<style>\n" + CSS + "\n</style>\n"
                                  + resolve_links(body, absolute=True))

    phys = PHYSICS_BODY.replace("@@EXPLORER_BUNDLE@@", explorer_bundle())
    phys = phys.replace("@@EXPLORER_MOUNT@@", explorer_mount("explorer"))
    phys = _chrome(phys, "physics", "chip")
    out["physics.html"] = _document(resolve_links(phys), PAGE_BY_KEY["physics"])

    chip = CHIP_BODY.replace("@@ANALOGY_BUNDLE@@", analogy_bundle())
    # Open on the finished machines: the "0.8 px to spare" reading is the point.
    chip = chip.replace("@@ANALOGY_MOUNT@@", analogy_mount("analogy", t=1))
    chip = _chrome(chip, "chip", "tolerance")
    out["chip.html"] = _document(resolve_links(chip), PAGE_BY_KEY["chip"])

    tol = _chrome(TOLERANCE_BODY, "tolerance", "optics")
    out["tolerance.html"] = _document(resolve_links(tol), PAGE_BY_KEY["tolerance"])

    opt = OPTICS_BODY.replace("@@OPTICS_BUNDLE@@", optics_bundle())
    opt = opt.replace("@@OPTICS_MOUNT@@", optics_mount("optics", zMm=3))
    opt = opt.replace("@@COMPARE_BUNDLE@@", compare_bundle(stage=True))
    # Open on a digit the shipped model gets wrong and the candidate does not.
    # The stage draws the deep column as a machine; it is fed the same digit the
    # board is, and decides for itself when to run the forward pass.
    opt = opt.replace("@@COMPARE_MOUNT@@", compare_mount(
        "compare", gallery=14, stage_id="stage3d", stage_model="deep"))
    # Last page in the reading order: the hand-off loops back to the machine.
    opt = _chrome(opt, "optics", "index", kicker="Back to the start")
    out["optics.html"] = _document(resolve_links(opt), PAGE_BY_KEY["optics"])

    return out


def main():
    os.makedirs(SITE, exist_ok=True)
    for name, text in render().items():
        path = os.path.join(SITE, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {path} ({len(text) // 1024} KB)")


if __name__ == "__main__":
    main()
