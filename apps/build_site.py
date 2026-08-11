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
        "photonn &middot; a neural network made of light",
        "This neural network is made of light",
        "A trained neural network made of light: a digit enters as a beam, crosses five "
        "plates of fabricated glass, and the answer is where the light lands. Runs live "
        "in your browser.",
        "A digit enters as a beam, crosses five plates of fabricated glass, and the answer "
        "is where the light lands. It runs live, in your browser.",
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
    ),
    Page(
        "chip", "chip.html", "The chip",
        "photonn &middot; the same machine, built two ways",
        "The same machine, built two ways",
        "A mesh of Mach-Zehnder interferometers on silicon computes what a stack of etched "
        "glass computes. The two differ on exactly one number.",
        "A mesh of interferometers on silicon computes what a stack of etched glass computes. "
        "Strip both to their skeletons and they differ on exactly one number.",
    ),
    Page(
        "tolerance", "tolerance.html", "Tolerance",
        "photonn &middot; how precisely must it be built?",
        "How precisely must it be built?",
        "The fabrication error budget: the trained network broken on purpose, one imperfection "
        "at a time, until the number that decides feasibility falls out.",
        "The question the whole project exists to answer. Break the trained network on purpose, "
        "one fabrication error at a time, and find the one that decides whether it can be built.",
    ),
    Page(
        "optics", "optics.html", "Going deeper",
        "photonn &middot; how much better could the optics be?",
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
    <p class="standfirst">A handwritten digit is written into a beam of light. The beam crosses
    five plates of glass, each with a pattern etched into its surface, and the answer is simply
    <b>where the light lands</b>. Nothing in that sentence is electronic. The etched patterns are
    what this network learned: they hold everything an ordinary neural network would keep as
    numbers in memory. It runs below, in your browser. Then this site asks the question that
    decides whether such a thing could ever be built: <b>how precisely would the glass have to be
    made?</b></p>
    <div class="stat-strip">
      <div class="stat"><span class="v">0.799</span><span class="l">of handwritten digits read correctly by the optics alone. Guessing scores 0.10</span></div>
      <div class="stat"><span class="v">5<small> plates</small></span><span class="l">of patterned glass are the entire trained network</span></div>
      <div class="stat"><span class="v">60<small> ps</small></span><span class="l">for light to cross the whole machine, which is 18&nbsp;mm of air and glass</span></div>
    </div>
  </section>

  <div class="explorer-band reveal">
    <div class="pe-host"><div id="d2nn"></div></div>
    <p class="cap">Live. Pick a digit from the set of test images the model has never been trained
    on, or draw your own. The whole optical calculation runs here, now, in this tab: <b>no
    libraries and no network requests</b>. The patterns are the trained model&rsquo;s own, exported
    unchanged.</p>
  </div>

  <div class="explorer-band reveal">
    <div class="pe-host"><div id="stage"></div></div>
    <p class="cap">The same run, drawn as the machine it is: the entrance, the five plates and the
    detectors, laid out along the path the light travels. Drag to orbit; hit <b>Sweep</b> to walk a
    single sheet of light across the stack.</p>
  </div>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">What you just watched</p>
      <h2>Every panel is real light, not an illustration</h2></div></div>
    <div class="prose col">
      <p>Light is a wave, and a wave gives you two things to adjust. How tall it is, its
      <strong>amplitude</strong>, is what makes light bright or dim. Where it sits in its
      up-and-down cycle, its <strong>phase</strong>, is what decides whether two waves meeting
      each other reinforce or cancel. Your eye sees the first and is completely blind to the
      second. This machine runs on both.</p>
      <p>So the <strong>entrance field</strong> is your digit written into the brightness
      <em>and</em> the phase of the beam. The five small frames are the brightness arriving at each
      plate. Each plate is a <strong>phase mask</strong>, a surface that holds the light back by a
      different amount at every point across it, and that is the name this site uses for them from
      here on. Watch the digit dissolve into a fine scramble that means nothing to the eye and
      everything to the detectors. The <strong>detector plane</strong> is the brightness at the far
      end, with the ten class regions drawn on it, and the answer is simply <strong>whichever box
      collects the most light</strong>. That is the entire readout: no electronic layer, no learned
      classifier on top, just ten sums.</p>
      <p>The 3D view is the same run drawn as a physical object: seven parallel planes strung along
      the path of the beam, each carrying the light actually computed on it, with the light
      <em>between</em> them drawn as haze. That haze is not a shading effect. It is the real light
      at those in-between depths, worked out the same way as everything else, and
      <a class="link" href="@@HREF_physics@@">the physics page shows why that is exact rather than
      an interpolation &rarr;</a> Toggle <em>Mask phase</em> to swap the arriving light for the
      etched surface that acts on it.</p>
      <p>There is no electronic network anywhere in this. Every step the light takes is
      <strong>linear</strong>: double the input and the output doubles, add two inputs together and
      you get the sum of what each would have produced alone. Exactly one step in the machine breaks
      that rule, and it is the detector, which measures brightness. Brightness is the square of the
      wave&rsquo;s height (<span class="q">|E|&sup2;</span>), and squaring is not linear. That is
      the whole computation, and it is also the ceiling on what the computation can do.</p>
    </div>
    <div class="stats col">
      <div class="s"><div class="v">0.799</div><div class="l">test accuracy (chance 0.10)</div></div>
      <div class="s"><div class="v">5 masks</div><div class="l">81,920 trained phases</div></div>
      <div class="s"><div class="v">6 gaps</div><div class="l">3&nbsp;mm each, 532&nbsp;nm green light, 128&times;128 grid</div></div>
      <div class="s"><div class="v">&lt;10<sup>&minus;3</sup></div><div class="l">agreement with the trained PyTorch model</div></div>
    </div>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">The motivation, stated honestly</p>
      <h2>Why build a computer out of light?</h2></div></div>
    <div class="prose col">
      <p>Because the expensive part of the arithmetic above is free. A neural network is mostly one
      operation repeated billions of times: multiply two numbers, add the result to a running total.
      In a processor that operation is already cheap. At 45&nbsp;nm a 32-bit floating-point add
      costs about <strong>0.9&nbsp;pJ</strong>, and a simpler fixed-point one roughly a ninth of
      that. What costs is <em>fetching the two numbers to feed it</em>. Those same 32 bits read out
      of memory on the chip cost <strong>5&nbsp;pJ</strong>, and out of the main memory alongside it
      <strong>640&nbsp;pJ</strong>, roughly a thousand times the operation they
      feed.<sup class="r">1</sup></p>
      <p>The stack above never fetches anything. Every point of the beam influences every detector
      because <em>that is what a wave does</em> over 3&nbsp;mm of air. Connecting everything to
      everything is the expensive part of a neural network layer, and here it is performed by the
      light spreading out on its own, with nothing charged, nothing switched, and no stored number
      read from anywhere. <strong>The trained values are never fetched because the trained values
      are the glass.</strong> And the computation finishes in the time light needs to cross
      18&nbsp;mm, which is <strong>60&nbsp;picoseconds</strong>, set by the speed of light and
      nothing else.</p>
      <p>How little light does it need? The error budget measures exactly that. Accuracy holds flat
      from 1&nbsp;mW all the way down to <strong>1&nbsp;pW over a 1&nbsp;ms exposure</strong>, which
      is about <strong>one femtojoule of light per classification</strong>, and only below that does
      it fall off a cliff.</p>
      <p>Now the honest part, because this is the number everyone gets wrong. That femtojoule is the
      energy <em>in the light</em>, not the energy to run the machine. The laser, the modulator that
      writes the digit into the beam, the ten detectors and their converters all cost more, and
      <strong>this project models none of them.</strong> The design&rsquo;s nominal operating point is
      1&nbsp;mW for 1&nbsp;ms, which is a microjoule per classification. That is <em>worse</em> than
      a GPU, and it is stated here so that nobody, including us, quotes this page as a win. The
      narrow claim is the only one the evidence supports: <strong>the part the optics does is nearly
      free, and everything around it is the engineering problem.</strong> That is roughly where the
      field itself sits.<sup class="r">4</sup></p>
      <p>Two families of machine chase this. One sends a beam through open air and trained plates,
      which is what runs above.<sup class="r">2</sup> The other guides light along channels etched
      into a silicon chip and mixes it in small steps.<sup class="r">3</sup> This project builds
      both, and <a class="link" href="@@HREF_chip@@">they turn out to be the same machine
      &rarr;</a></p>
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
      <p>Two operations alternate, and both are linear. <strong>Crossing the 3&nbsp;mm gap</strong>
      spreads every point of the beam outward into a small disc that overlaps its neighbours, by the
      same amount everywhere on the plate. That blending is what carries information sideways across
      the beam, and it is the step mathematicians call a <em>convolution</em>. <strong>Passing
      through a plate</strong> does the opposite. It touches each point on its own, holding it back
      by the depth etched at that spot and leaving its neighbours alone, which amounts to
      multiplying each point by <span class="q">exp(i&thinsp;&phi;(x,y))</span>, one delay per
      pixel. Those delays are the only thing training ever adjusts. Alternating the two, spread then
      delay, spread then delay, is exactly what a hologram does. This stack is five holograms that
      training wrote.</p>
      <p>What the plates learn to do with that is <strong>steer light and bring it to a focus</strong>.
      Two waves that arrive in step add up and get brighter; two that arrive out of step cancel and
      go dark. The plates set the delays so that after all the spreading, an input of class
      <em>c</em> arrives in step over detector region <em>c</em> and out of step everywhere else.
      Early plates behave more like feature detectors, shuffling light around the whole beam; later
      ones behave more like lenses, gathering the light that identifies a class onto the right
      patch. About 60% of the light that enters ends up inside a detector box.</p>
      <p>That steering picture is <strong>a story, not a proof</strong>, and it is worth saying so
      plainly. What is rigorously true is that the whole stack adds up to a single linear operation,
      and the plates are one way of building that operation out of parts you could actually
      manufacture.
      <a class="link" href="@@HREF_physics@@">That is also where its ceiling comes from &rarr;</a></p>
    </div>
    <figure class="plate reveal">
      <img src="@@FIG_phase2_masks@@" alt="Five trained phase masks and one input-to-output intensity example for the diffractive network" loading="lazy">
      <figcaption><span class="fign">Fig 1</span>The five learned plates (top) and one worked
      example: an input digit spreading out on its way to the detectors (bottom). Each plate is a
      real surface with a relief pattern cut into it, and training only ever adjusted those depths.
      The structure is fine-grained, changing sharply from one pixel to the next, and that is
      exactly what makes the design hard to manufacture.</figcaption>
    </figure>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">Choosing the task</p>
      <h2>Why a digit classifier, and only a digit classifier</h2></div></div>
    <div class="prose col">
      <p>The task is <strong>MNIST</strong>, a collection of 70,000 handwritten digits that
      machine-learning work has used as a first test for decades. It is the smallest honest version
      of the job: a real problem with a real error rate, it fits across the beam without contrivance,
      and, the part that matters here, it is <strong>easy enough that the optics stays the
      interesting part</strong>. The moment a task needs serious electronics bolted on the end to
      work at all, the optical network stops being the thing under study.</p>
      <p>So the same ten digits are reused at every stage: the stack of plates, the silicon chip, and
      every one of the error sweeps that follow. One task, scored the same way on the same fixed
      set of 2,000 images the models never train on, is what makes those results <strong>comparable
      to each other</strong>, and that is worth far more here than a higher score on a harder task
      would be. The machine-learning content is deliberately minimal and stays that way.</p>
    </div>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">Read the failures, not just the wins</p>
      <h2>It is wrong about one digit in four</h2></div></div>
    <div class="prose col">
      <p>The network reads <strong>0.799</strong> of the test digits correctly, so the gallery
      deliberately includes ones it <strong>gets wrong</strong>. Hiding them would misrepresent the
      model. Watch how the light is divided when it fails: a confident answer puts about
      25&ndash;30% of the light in the winning box, a wrong one usually far less.</p>
      <p>Drawings are harder still. Whatever we do, your handwriting will not look quite like the
      handwriting the model was trained on, so expect more errors there. To keep the comparison fair
      rather than flattering, a drawing is cleaned up the same way the MNIST images themselves were
      built: cropped to the ink, scaled so its longer side is 20&nbsp;px, and centred by its balance
      point. That way a digit drawn too large or off to one side cannot be mistaken for a failure of
      the optics.</p>
      <p>What your browser computes is not a simplified stand-in for the trained model. On the test
      digits it reproduces the predictions of the original PyTorch model <strong>exactly</strong>,
      with the ten scores agreeing to better than 10<sup>&minus;3</sup>. Its delays are stored at 8
      bits of precision, 256 possible settings per pixel, which is what a real device offers anyway,
      and that costs <strong>nothing</strong> measurable: 0.7995 against 0.7990 across 2,000
      digits.</p>
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
      <p>A <strong>fabricated</strong> one, meaning one actually cut into real glass, has none of
      that. Etch depths vary. The device that writes the delays can
      only set them so finely, and its electric field spills sideways so that each pixel smears into
      the next. The laser drifts off its colour. Light is lost at every surface.
      <strong>None of those are bugs to be fixed. They are the specification.</strong> The only
      question that decides whether this design could be built is how much of each it survives, and
      that is a number, not an opinion.</p>
      <p>Getting that number is what the rest of this project is. The trained delays are handed over
      to a second, independent model of the machine <em>as it would really be built</em>, which then
      breaks the network on purpose, one imperfection at a time. <strong>One of the six imperfections
      already fails against hardware you can buy today</strong>, and
      <a class="link" href="@@HREF_tolerance@@">it is not the one you would guess &rarr;</a></p>
    </div>
  </section>

  @@NEXT@@

  <footer class="reveal">
    <div class="foot-grid">
      <div><h3>The physics</h3><p>The same diffraction calculation as the Python original, rewritten
        for the browser with no libraries and checked against it to better than
        10<sup>&minus;6</sup>. Six gaps of air and five plates per answer.</p></div>
      <div><h3>The parameters</h3><p>Exported straight from the trained PyTorch model: 81,920
        delays, stored at the 8&nbsp;bits of precision a real device offers. Nothing is retrained or
        tuned to make the browser version look better.</p></div>
      <div><h3>Privacy</h3><p>Everything happens on your machine. Nothing you draw is uploaded,
        stored, or sent anywhere. The page makes no network requests at all.</p></div>
    </div>
    <p class="colophon">photonn &middot; a portfolio study in optical computing and fabrication tolerance.
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
    <p class="eyebrow">How the light is moved, and where the method stops working</p>
    <h1>Everything rests on moving light from one plane to the <em>next</em></h1>
    <div class="underbar"></div>
    <p class="standfirst">Before anything can be trained, the simulation has to move light across
    3&nbsp;mm of air <b>exactly</b>, and it has to know the point at which it can no longer do that.
    One method does the moving, one criterion marks that point, and one structural fact caps what
    the whole idea can ever compute.</p>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">The propagator</p>
      <h2>Decompose into plane waves, delay each one, add them back up</h2></div></div>
    <div class="prose col">
      <p>Any pattern of light, however complicated, can be written as a sum of <strong>plane
      waves</strong>: perfectly flat, evenly spaced wavefronts, each travelling in one particular
      direction. That is useful because a plane wave crossing a gap does something trivial. It stays
      a plane wave and simply falls behind by an amount set by its direction. So the recipe is to
      split the light into plane waves with a Fourier transform, hold each one back by its own
      amount, and add them all back together. That is the whole <strong>angular-spectrum
      method</strong>: transform, multiply by
      <span class="q">H = exp(i&thinsp;2&pi;z&thinsp;&radic;(1/&lambda;&sup2; &minus; f&sup2;))</span>,
      transform back. It makes <strong>no small-angle approximation</strong>, the simplification most
      textbook diffraction formulas lean on and the one usually called <em>paraxial</em>, so it stays
      exact even for steeply travelling light. That is
      why Fresnel, Fraunhofer and the differentiable PyTorch layer are all checked against it rather
      than the other way round. Against a Gaussian beam with a known closed-form answer it agrees to
      <strong>~2&times;10<sup>&minus;8</sup></strong>.</p>
      <p>Two behaviours fall straight out of that square root. Where
      <span class="q">f&sup2; &gt; 1/&lambda;&sup2;</span> the quantity under it goes negative, which
      corresponds to light angled so steeply it cannot actually travel. Computing the root over the
      complex numbers makes those components <strong>die away</strong> with distance instead, which
      is the correct physics: these are <em>evanescent</em> waves, which cling to the surface rather
      than propagating. No special case is needed. And over a long enough distance the multiplier
      <span class="q">H</span> starts <strong>spinning faster than the grid has points to follow
      it</strong>. A grid that cannot keep up with a wave reports a slower one that happens to match
      at the sample points, which is <em>aliasing</em>, and the answer silently becomes wrong.
      Setting <span class="q">H</span> to zero past the point where the grid can still track it
      keeps the method honest at long range, at the cost of throwing away the steepest light.</p>
      <p>The edge is a single criterion,
      <span class="q">z_crit = N&middot;dx&sup2;/&lambda;</span>, which is 15.4&nbsp;mm here against
      3&nbsp;mm gaps. It is checked every time the code runs, not only in the tests. Note what it
      does <em>not</em> contain. It depends on the wavelength, the pixel size and the number of
      pixels, but <strong>not on the width of the opening the light passes through</strong>, which
      is why the aperture control below does not move the sampling threshold no matter how far you
      drag it.</p>
    </div>
    <div class="explorer-band reveal">
      <div class="pe-host"><div id="explorer"></div></div>
      <p class="cap">Live. The diffraction is computed in your browser, the same physics as the
      Python original and checked against it to better than 10<sup>&minus;6</sup>. Move any control;
      the sampling flag turns amber the moment
      <span style="white-space:nowrap">z &gt; z_crit</span>.</p>
    </div>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">Drawing it without lying about it</p>
      <h2>Why the 3D stack has haze in it and no rays</h2></div></div>
    <div class="prose col">
      <p>The light drawn <em>between</em> the mask planes on the front page is not decoration. One
      3&nbsp;mm gap can be cut into several shorter steps and re-run, because propagating a distance
      and then another gives exactly the same answer as propagating the sum in one go:
      <span class="q">H(z&#8321;)&middot;H(z&#8322;) = H(z&#8321;+z&#8322;)</span>. The one thing that
      would break that equality is the band limit described above, and below
      <span class="q">z_crit</span> it is <strong>switched off entirely</strong>. At 3&nbsp;mm
      against 15.4&nbsp;mm it is switched off, so those in-between planes are <strong>the real
      light</strong>, not an interpolation between the planes on either side. Above
      <span class="q">z_crit</span> the equality genuinely does break, and the test suite asserts
      both halves of that.</p>
      <p><strong>No rays are drawn, deliberately.</strong> Light here is treated as a wave, not as
      travelling arrows, and drawing straight lines from digit to detector would misrepresent the one
      thing the figure exists to show: that every point of the input reaches every detector at once.
      The stack is 18&nbsp;mm long across an opening of 1.02&nbsp;mm, about 18:1, so drawn true to
      scale it is an unreadable needle. The depth is therefore squashed, and the figure prints its
      own squash factor on its face rather than quietly flattering the geometry.</p>
      <p>The drawing also <strong>cannot touch the answer</strong>. The prediction still comes from
      the canonical six propagations, and a test asserts the ten class scores are bit-for-bit
      identical whether the extra in-between planes are computed or not.</p>
    </div>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">The limit, stated precisely</p>
      <h2>Optical depth is not depth in the machine-learning sense</h2></div></div>
    <div class="prose col">
      <p>Crossing a gap is linear in the light. Passing through a mask is linear in the light. Doing
      one after the other is therefore still linear, and the whole stack collapses into a
      <em>single</em> multiplication by one big matrix:</p>
      <p><span class="q">E_out = M &middot; E_in</span>, where
      <span class="q">M = P_L D_L P_{L&minus;1} D_{L&minus;1} &hellip; D&#8321; P&#8320;</span>, each
      <span class="q">P</span> a gap and each <span class="q">D</span> a mask.</p>
      <p>This matters more than it sounds. No matter how many masks and gaps are stacked up, the
      map from input light to output light is one matrix. Extra layers add adjustable numbers, and
      they let <span class="q">M</span> come closer to whatever matrix you wanted, <em>subject to
      the physical constraint</em> that it be buildable out of delay-only masks separated by air.
      What they do not add is any of the compounding power that stacking layers gives an ordinary
      neural network, where each layer bends the result before the next one sees it.
      <strong>Depth here is not depth in that sense at all.</strong></p>
      <p>The single exception is the detection. The score for class <em>c</em> is the brightness
      added up over its detector region, and brightness is the square of the wave&rsquo;s height. Work
      that through and the score is
      <span class="q">s_c = E_in&dagger; A_c E_in</span> with
      <span class="q">A_c = M&dagger; R_c M &#8827; 0</span>. In words: each score is a
      <strong>quadratic form</strong>, meaning the inputs enter multiplied together in pairs rather
      than one at a time, and the <span class="q">&#8827; 0</span> says the resulting score can never
      go negative. The classifier computes one of these per class and picks the largest. That is
      strictly more powerful than a plain linear classifier, and it is exactly <em>one</em> nonlinear
      step: linear optics, square the magnitude, add up.</p>
      <p>One consequence is worth pulling out, because it is the only other place anything nonlinear
      touches the raw pixels: <strong>how the image is written into the beam</strong>. Encoding it as
      delays rather than brightness passes the pixel values through
      <span class="q">exp(i&middot;&pi;&middot;image)</span> before the optics ever sees them, and
      that step is not linear. This is why the choice of encoding measurably changes the accuracy
      that can be reached. It is quietly doing work the linear optics downstream cannot do.</p>
      <p>By the project&rsquo;s own scope this limit is <strong>measured and described, not
      engineered around</strong>. There is no attempt to add a nonlinear step in the optics, no
      exotic materials, and no larger electronic network bolted on the end to compensate. If a
      bigger electronic stage would lift the accuracy, that is a result worth reporting, not a bug
      to fix.</p>
      <div class="finding">
        <p class="tag">Correction &middot; kept on the record</p>
        <p class="body">An earlier version of this project read the shipped network&rsquo;s
        <strong>0.799 plateau</strong> as this ceiling being reached, and presented it as measured
        rather than merely asserted. <strong>That inference was wrong,</strong> and it is left here
        rather than quietly removed. The argument above is untouched: the stack really is one matrix.
        What did not follow was concluding that five masks had already wrung everything out of it.
        They had not, by at least eight points. The plateau belonged to <em>that particular
        arrangement of the optics</em>, and
        <a class="link" href="@@HREF_optics@@">sweeping that arrangement is a page of its own
        &rarr;</a></p>
      </div>
    </div>
  </section>

  @@NEXT@@

  <footer class="reveal">
    <div class="foot-grid">
      <div><h3>Checked against known answers</h3><p>Every case here has a textbook solution to check
        against. The angular spectrum matches a Gaussian beam to ~2&times;10<sup>&minus;8</sup>;
        Fresnel matches it inside its valid range to ~9&times;10<sup>&minus;9</sup>; a round hole
        gives the classic bullseye pattern with its first dark ring exactly where theory puts it.</p></div>
      <div><h3>Enforced, not assumed</h3><p>The sampling limit is checked whenever the code runs, not
        only in the tests: <span class="q">check_sampling</span> returns a report, and the explorer
        flags the crossing live. Past <span class="q">z_crit</span> the method does not crash or
        obviously misbehave, which is precisely why it has to be flagged.</p></div>
      <div><h3>Scope</h3><p>Light is treated as a single wave with no direction of vibration, which
        rules out polarisation effects, and there is no full electromagnetic solver anywhere. If real
        component data is ever wanted it gets imported as measurements rather than simulated here.</p></div>
    </div>
    <p class="colophon">photonn &middot; a portfolio study in optical computing and fabrication tolerance.
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
    <p class="standfirst">A stack of etched glass and a silicon chip threaded with light-carrying
    channels look like unrelated devices. Strip both to their skeletons and the <b>same sequence
    appears</b>: a layer of delays you train, then a layer of fixed hardware that blends the
    channels together, repeated, and closed by a detector that measures brightness. They differ on
    exactly one number, and that number sets everything else.</p>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">The other optical computer</p>
      <h2>An interferometer is two splitters around an adjustable delay</h2></div></div>
    <div class="prose col">
      <p>On a chip, light does not spread through open air. It runs along <strong>waveguides</strong>,
      narrow channels of silicon that trap light the way a pipe carries water. Each channel carries
      one stream, called a <strong>mode</strong>, and streams in separate channels cannot mix unless
      the hardware deliberately brings them together. The component that does that is a
      <strong>directional coupler</strong>: run two channels close enough and light leaks between
      them, and at the right length exactly half crosses over.</p>
      <p>A <strong>Mach&ndash;Zehnder interferometer</strong> is not exotic hardware. It is two of
      those 50:50 couplers with an adjustable delay <span class="q">&theta;</span> in between and
      another delay <span class="q">&phi;</span> in front, which written out is
      <span class="q">B&middot;P(&theta;)&middot;B&middot;P(&phi;)</span>. Split the light, hold one
      arm back, recombine: how much the two halves reinforce or cancel decides how much leaves by
      each output. That combination is <strong>unitary</strong> for any setting of the two delays,
      meaning it only ever redistributes light between the outputs and never creates or destroys
      any, which is exactly what passive hardware can do. Tile these across neighbouring pairs of
      channels and you get a <strong>Clements rectangle</strong> of
      <span class="q">n(n&minus;1)/2</span> of them.</p>
      <p>Such a mesh can produce <em>any</em> unitary operation at all, and the proof is a recipe
      rather than an existence argument. You take the target operation and use one interferometer at
      a time to <strong>zero out its entries</strong> one by one, until nothing is left but the
      identity; run the recipe backwards and you have the settings. Attacking it from one side only
      gives the triangular <strong>Reck</strong> mesh; attacking from both sides gives the balanced
      <strong>Clements</strong> rectangle, which is half as deep and loses less light. Put a row of
      simple brightness adjustments between two such meshes and you can build <em>any</em> real
      matrix, not just a unitary one, using the <strong>singular-value decomposition</strong>
      <span class="q">U&middot;&Sigma;&middot;V&dagger;</span>, the standard factorisation that
      splits any matrix into a rotation, a stretch along each axis, and another rotation. Both recipes rebuild randomly drawn target
      operations to <strong>~10<sup>&minus;15</sup></strong>, which is this side&rsquo;s
      correctness anchor.</p>
      <p>Trained on the same digits it scores <strong>0.736</strong>, just shy of the glass
      stack&rsquo;s 0.799, but with <strong>2,628 adjustable numbers against 81,920, about
      31&times; fewer</strong>, and able to implement any linear map at all rather than only the
      ones a stack of masks can reach. The gap is not a modelling failure. The chip&rsquo;s area
      grows as the <em>square</em> of the number of channels, so feeding it a large image is
      prohibitive, and MNIST has to be shrunk to 6&times;6 = 36 channels first. <strong>It starves
      on how much input it can take in, not on what it can compute.</strong> That trade between
      chip area and input size is the central structural difference between the two processors.</p>
      <p>One finding falls out of the physics rather than the training. A mesh with no power source
      <strong>cannot amplify</strong>, since at best it passes all the light through and in practice
      loses some, so a buildable <span class="q">&Sigma;</span> needs every one of its stretch
      factors to be 1 or less. Several of the trained ones come out above 1. A real device would
      need an amplifier, or would have to scale everything down and pay for it in lost light.</p>
    </div>

    <div class="stats col">
      <div class="s"><div class="v">0.736</div><div class="l">accuracy, against 0.799 for the glass stack</div></div>
      <div class="s"><div class="v">2,628</div><div class="l">adjustable numbers &middot; ~31&times; fewer</div></div>
      <div class="s"><div class="v">36 channels</div><div class="l">72 interferometers deep, one after another</div></div>
      <div class="s"><div class="v">1e&minus;15</div><div class="l">error when rebuilding a target operation</div></div>
    </div>
    <figure class="plate reveal">
      <img src="@@FIG_mesh_topology@@" alt="MZI mesh topology and the learned singular-value spectrum" loading="lazy">
      <figcaption><span class="fign">Fig 1</span>Left: how the interferometers are tiled across the
      channels. Right: the size of each of the 36 stretch factors the training settled on. Only about
      15&ndash;20 of them carry any real weight, which makes the operation <em>low-rank</em>: it is
      effectively using far fewer channels than it has, and that is why so few adjustable numbers
      suffice.</figcaption>
    </figure>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">The correspondence</p>
      <h2>Why they are the same machine</h2></div></div>
    <div class="prose col">
      <p>A chip of waveguides looks nothing like a stack of etched glass, and everything above reads
      as two unrelated devices. They are not. Strip both to their skeletons and the same sequence
      appears: <strong>a layer of delays you train, then a layer of fixed hardware that blends the
      channels, repeated, and closed by a detector that measures brightness.</strong> A phase mask
      <em>is</em> a column of delays. A 3&nbsp;mm air gap <em>is</em> a column of couplers. In both
      machines the only thing training ever adjusts is a delay, and in both, the blending step is
      fixed by the hardware and cannot be programmed.</p>
    </div>
    <div class="tbl-wrap col">
      <table class="tbl">
        <thead><tr><th></th><th>Glass stack &middot; free space</th><th>Interferometer mesh &middot; chip</th></tr></thead>
        <tbody>
          <tr><td>a &ldquo;channel&rdquo; is</td><td>one pixel of the beam, 16,384 of them</td><td>one waveguide, 36 of them</td></tr>
          <tr><td>the <strong>trainable</strong> part</td><td>a phase mask, one delay per pixel</td><td>a phase shifter, one delay per arm</td></tr>
          <tr><td>set in hardware by</td><td>how deep the glass is etched at that point</td><td>a tiny heater warming the silicon (thermo-optic)</td></tr>
          <tr><td>the <strong>fixed</strong> part</td><td>3&nbsp;mm of air, where the light spreads</td><td>a coupler that splits light evenly in two</td></tr>
          <tr><td>readout</td><td>brightness added up over 10 detector boxes</td><td>brightness on the first 10 outputs</td></tr>
        </tbody>
      </table>
    </div>
    <div class="prose col">
      <p>This is <strong>a shared abstraction, not shared hardware.</strong> 16,384 pixels and 36
      modes are nowhere near the same scale, and no rearrangement turns one into the other. What
      transfers is the skeleton, not the device.</p>
      <p>They part on exactly one thing, <strong>how far a single blending layer carries information
      sideways</strong>, and that one number sets the depth, the size and the way each machine fails.
      Diffraction hands you a wide reach for nothing, but you cannot <em>choose</em> it: the
      12.5&nbsp;px per gap is fixed by the distance, the wavelength and the pixel size, and it is the
      same everywhere on the plate. A coupler reaches exactly one neighbour, so you need as many
      columns as you have channels, and in exchange every one of those couplings can be set
      individually, which is what lets the mesh reach any operation at all. <strong>Going from free
      space to a chip trades one wide, fixed, unsteerable blending step for many narrow ones you can
      steer.</strong> The rest is packaging.</p>
    </div>

    <div class="explorer-band reveal">
      <div class="pe-host"><div id="analogy"></div></div>
      <p class="cap">Live. Every number here is read out of the trained models, not typed into the
      figure. The mesh drawn is the real tiling the model uses, and the cone is
      <span style="white-space:nowrap">z&middot;&lambda;/(2&middot;dx&sup2;)</span> per gap.</p>
    </div>

    <div class="finding col reveal">
      <p class="tag">Finding &middot; the glass stack is connected by 0.8 px</p>
      <p class="body">Six gaps of 3&nbsp;mm let each pixel&rsquo;s light spread
      <strong>74.8&nbsp;px</strong> sideways in total. The hardest case the design has to cover is a
      pixel at one edge of the entrance influencing the detector pixel farthest from it, and that
      distance is <strong>74&nbsp;px</strong>. So every input pixel <em>can</em> reach every
      detector, with <strong>0.81&nbsp;px to spare, about 1%</strong>. Push the plates closer than
      <strong>2.967&nbsp;mm</strong> apart and parts of the input become physically invisible to
      parts of the readout, no matter what the masks are set to. Nothing in the training knew this
      limit existed; the design happens to clear it. The mesh has no such fragility, because 36
      columns for 36 channels is exactly the bound Clements proves you need, so everything reaching
      everything is guaranteed by the layout. <strong>One machine&rsquo;s connectivity is an
      accident that holds by 1%. The other&rsquo;s is a theorem.</strong></p>
    </div>

    <div class="prose col">
      <p>A limit on what <em>can</em> reach what is not a claim about how much light actually does.
      The very edge of that cone is the steepest ray the grid can represent, and it carries little
      energy in practice. What the bound tells you is where the design sits relative to a hard wall,
      and the answer is &ldquo;just inside it, by accident&rdquo;.</p>
      <p>The two machines also <strong>share their ceiling</strong>. Everything the
      <a class="link" href="@@HREF_physics@@">physics page argues about linearity &rarr;</a> applies
      to the mesh unchanged: one linear optical step, then a detector that squares. Swapping free
      space for silicon buys steerability, a smaller footprint and something you can actually
      manufacture. It buys no extra computational power at all.</p>
      <p>How they <strong>fail</strong> is where they differ, and that is the part this project is
      built to measure. On the chip, small errors pile up as the light passes through 72
      interferometers one after another, and couplers that split slightly unevenly or lose a little
      light make the whole operation leak. In the glass stack the errors act <em>side by side</em>
      instead, one per pixel, and what dominates is whether each pixel&rsquo;s delay stays sharply
      confined to that pixel. <a class="link" href="@@HREF_tolerance@@">Two optical computers, two
      different ways to lose the computation to fabrication &rarr;</a></p>
      <ol class="refs">
        <li><b>1</b>M. Reck, A. Zeilinger, H. J. Bernstein &amp; P. Bertani, &ldquo;Experimental
        realization of any discrete unitary operator,&rdquo; <em>Phys. Rev. Lett.</em> <b>73</b>, 58
        (1994). The triangular mesh.</li>
        <li><b>2</b>W. R. Clements <em>et al.</em>, &ldquo;Optimal design for universal multiport
        interferometers,&rdquo; <em>Optica</em> <b>3</b>, 1460 (2016). The rectangular mesh and the
        recipe for zeroing entries one at a time.
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
      <p>The interferometer mesh has a second life. Everything above treats light as a bright beam.
      Send it through instead <strong>one photon at a time</strong>, with the photons identical
      enough that there is no way even in principle to tell which took which path, and the machine
      stops behaving like a beam splitter and starts behaving quantum mechanically. The odds of each
      possible output pattern are then governed by a quantity called the <em>permanent</em> of the
      matrix, which is famously hard to compute, and that difficulty is the whole point of
      <strong>boson sampling</strong>. The planned deliverable computes those odds, shows the
      <strong>Hong&ndash;Ou&ndash;Mandel dip</strong> (send two identical photons into a 50:50
      coupler and they always leave together, never one each way, which has no classical
      explanation), and contrasts it with what ordinary distinguishable particles would do. Only the
      light going in changes. The mesh itself is identical.</p>
    </div>
  </section>

  @@NEXT@@

  <footer class="reveal">
    <div class="foot-grid">
      <div><h3>Derived, never typed</h3><p>The figure above reads how far light spreads from the
        propagation code, its detector layout from the detector code and its tiling from the mesh
        itself. A test recomputes all of it and fails if the exported design drifts.</p></div>
      <div><h3>Verified</h3><p>That an interferometer conserves light at every setting; that both
        recipes rebuild a target operation to ~10<sup>&minus;15</sup>; that the arbitrary-real-matrix
        layer works; and that the PyTorch mesh matches the plain NumPy one.</p></div>
      <div><h3>Still open</h3><p>The mesh&rsquo;s own error budget, which switches on the two error
        sources that mean nothing for phase masks: couplers that split unevenly, and light lost in
        each interferometer. Its numbers will have to come from a chip foundry&rsquo;s published
        process data, not the display-device literature this study uses.</p></div>
    </div>
    <p class="colophon">photonn &middot; a portfolio study in optical computing and fabrication tolerance.
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
    <p class="standfirst">Everything so far was ideal. The masks were exact, the plates sat exactly
    3&nbsp;mm apart, and every pixel delayed the light by exactly what it was trained to. Now those
    same trained numbers are handed to a separate model of the machine <b>as it would really be
    built</b>, and broken on purpose, one imperfection at a time, until the number that decides
    whether this is buildable falls out.</p>
    <div class="stat-strip">
      <div class="stat"><span class="v">0.25<small> px</small></span><span class="l">is how far one pixel&rsquo;s delay may bleed into its neighbours before accuracy goes</span></div>
      <div class="stat"><span class="v">&asymp;1<small> px</small></span><span class="l">is how far it bleeds on a real display chip. That is four times too much</span></div>
      <div class="stat"><span class="v">0.7990</span><span class="l">reproduced exactly by the second model when no error is injected at all</span></div>
    </div>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">Method</p>
      <h2>Now break it on purpose</h2></div></div>
    <div class="prose col">
      <p>The trained numbers cross a <strong>one-way handover</strong>: a single file carrying the
      phase masks, the layout, the operating conditions and the test images, passed to a completely
      separate MATLAB model of the machine <em>as it would really be built</em>, the
      <strong>as-built</strong> model, which never writes anything back. That one-way boundary is deliberate. It keeps the ideal network and the
      imperfect one from quietly contaminating each other. Nothing is retrained on the far side, and
      no error model exists on the near side.</p>
      <p>That second model then re-scores the very same network under each manufacturing imperfection
      in turn: delays set slightly wrong, delays available only in coarse steps, light lost along the
      way, the laser drifting off its colour, heat from one pixel leaking into the next, and noise in
      the detectors. Each one is run many times over with different random draws, which is what a
      <em>Monte Carlo</em> sweep means, with the random seeds recorded so any run can be reproduced
      exactly, and <strong>every error size traced to a published measurement</strong> cited in the
      code at the point it is used.</p>
      <p>One control turns this from a demonstration into a measurement. <strong>With no error
      injected at all, the second model reproduces 0.7990 exactly</strong>, the same number the
      PyTorch model gets on the same 2,000 images. Two independently written implementations agreeing
      to the last digit means any drop below it is caused by the injected error and nothing else.
      That anchor has held through every change in the project.</p>
      <p>The pass mark throughout is <strong>95% of the ideal accuracy</strong>, which works out to
      <span class="q">&ge; 0.7591</span>. Limits are quoted as the bracket the sweep actually
      resolves, <em>holds at X, fails at Y</em>, rather than interpolating a crossing point between
      two tested values, so they stay comparable from one model to the next.</p>
    </div>
    <div class="finding col reveal">
      <p class="tag">Binding constraint</p>
      <p class="body"><strong>Crosstalk between neighbouring pixels sets the tolerance, and it is
      the one that fails.</strong> Setting one pixel&rsquo;s delay disturbs the pixels around it,
      because heat spreads and electric fields spill sideways, so a pattern meant to be sharp comes
      out blurred. Accuracy holds while that blur stays at <strong>0.25&nbsp;px</strong> and fails by
      <strong>0.5&nbsp;px</strong>; three quarters of a pixel of blur collapses the network to little
      better than guessing. On a standard LCoS device, the liquid-crystal display chip these masks
      would be written to, the blur is about <strong>1&nbsp;px</strong>, so this design as specified
      <strong>would not work on one</strong>. The ranking is unambiguous: crosstalk is far worse than
      delay error, which beats detector power and lost light, which in turn are far worse than
      wavelength drift and coarse delay steps. To stay above
      <span class="q">&ge; 0.7591</span>, the design needs delays accurate to
      <strong>0.3&nbsp;rad</strong>, a twenty-first of a wavelength (&lambda;/21), at least
      <strong>3 bits</strong> of control per pixel, laser drift under <strong>10&nbsp;nm</strong>,
      and at least <strong>1&nbsp;pW</strong> of light over 1&nbsp;ms. What decides whether this is
      buildable is not the precision of each delay or how finely it can be set. It is how sharply
      each pixel&rsquo;s delay can be kept away from its neighbours.</p>
    </div>
    <div class="prose col">
      <p>Read the ranking rather than the individual numbers. <strong>Delay accuracy and bit depth
      are comfortable.</strong> A well-calibrated device sets delays to about a hundredth of a
      wavelength (&lambda;/100, roughly 0.05&nbsp;rad), which is ten times better than the
      0.3&nbsp;rad that still holds, and 3 bits of control is enough against a hardware standard
      of 8.
      <strong>Wavelength drift is a non-issue</strong> for any laser with its temperature held steady.
      <strong>Detection has nine orders of magnitude in hand</strong> at the intended operating
      point, before the graininess of counting individual photons, known as <em>shot noise</em>,
      starts to matter.</p>
      <p><strong>Lost light is the interesting one</strong>, because it never acts alone. Dimming
      everything by the same factor cancels out exactly, since the readout compares the ten boxes
      against each other rather than against an absolute brightness, so when photons are plentiful
      losing light has <em>zero</em> effect on accuracy. It only bites by leaving too few photons to
      count reliably, which is why it is swept right at the point where photon counting starts to
      break down, where every 3&nbsp;dB lost halves the photons arriving. Lost light and the light
      budget have to be reasoned about together or the sweep measures nothing at all.</p>
      <p>And <strong>a better-trained network was not a more fragile one</strong>, which is worth
      stating because the opposite is a reasonable thing to expect. Retraining on five times more
      data lifted the ideal accuracy from 0.7695 to 0.7990, and every limit was re-measured against a
      correspondingly stricter pass mark. Five of the six landed in the same bracket as before. Masks
      fitted to more data did not come out finer and more brittle. But that result belongs to
      <em>training</em>, and the next section is what happened when the <em>optics</em> changed
      instead.</p>
    </div>
    <div class="plate-grid reveal">
      <figure class="plate"><img src="@@FIG_tol_crosstalk@@" alt="Accuracy vs thermal/pixel crosstalk" loading="lazy"><figcaption><span class="fign">Fig 1</span>Crosstalk between pixels, the constraint that binds.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_tol_phase@@" alt="Accuracy vs per-pixel phase error" loading="lazy"><figcaption><span class="fign">Fig 2</span>Each pixel&rsquo;s delay set slightly wrong.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_tol_detector@@" alt="Accuracy vs detector noise / input power" loading="lazy"><figcaption><span class="fign">Fig 3</span>Detector noise as the light is dimmed.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_tol_loss@@" alt="Accuracy vs optical insertion loss" loading="lazy"><figcaption><span class="fign">Fig 4</span>Light lost at each surface, swept where it bites.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_tol_wavelength@@" alt="Accuracy vs wavelength drift" loading="lazy"><figcaption><span class="fign">Fig 5</span>The laser drifting off its colour.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_tol_quant@@" alt="Accuracy vs DAC bit resolution" loading="lazy"><figcaption><span class="fign">Fig 6</span>Delays available only in coarse steps.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_confusion@@" alt="As-built confusion matrix at phase sigma 0.35 rad" loading="lazy"><figcaption><span class="fign">Fig 7</span>Which digits get confused for which, at &sigma;=0.35&nbsp;rad.</figcaption></figure>
    </div>
    <figure class="plate reveal">
      <img src="@@FIG_sensitivity@@" alt="Per-mask spatial sensitivity map" loading="lazy">
      <figcaption><span class="fign">Fig 8</span>Where on each mask an error costs the most accuracy.
      Some regions matter far more than others, and it is that unevenness that makes a per-pixel
      tolerance a meaningful thing to quote.</figcaption>
    </figure>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">The result this project actually has</p>
      <h2>What depth costs</h2></div></div>
    <div class="prose col">
      <p>A deeper network is <strong>a lot</strong> more accurate. Splitting the same total amount
      of light-spreading across 56 masks instead of 5 reaches <strong>0.9040</strong> on the same
      test images, against 0.7990, and it is
      <a class="link" href="@@HREF_optics@@">running live, beside the shipped model &rarr;</a> so the
      improvement is easy to see and easy to mistake for a straightforward upgrade.</p>
      <p>It is not one. The whole budget above was re-run against that deeper candidate, scored
      against its own correspondingly stricter pass mark of 0.8588, and <strong>those extra 10.5
      points are paid for in how precisely the thing would have to be built</strong>:</p>
    </div>
    <div class="tbl-wrap col">
      <table class="tbl">
        <thead><tr><th>Error source</th><th>Shipped &middot; 5 masks</th><th>Candidate &middot; 56 masks</th><th>Change</th></tr></thead>
        <tbody>
          <tr><td><strong>Crosstalk between pixels</strong></td><td>holds 0.25&nbsp;px, fails 0.5</td><td>holds <strong>0.25&nbsp;px</strong>, fails 0.5</td><td><strong>unchanged, still binding</strong></td></tr>
          <tr><td>Each pixel&rsquo;s delay set wrong</td><td>holds 0.3&nbsp;rad, fails 0.5</td><td>holds <strong>0.15&nbsp;rad</strong>, fails 0.2</td><td>2&times; tighter</td></tr>
          <tr><td>How finely a delay can be set</td><td>holds 3 bits, fails 2</td><td>holds <strong>4 bits</strong>, fails 3</td><td>1 bit tighter</td></tr>
          <tr><td>Light lost, where it bites</td><td>holds 1&nbsp;dB/mask (5&nbsp;dB total)</td><td>holds <strong>0.214&nbsp;dB/mask</strong> (12&nbsp;dB total)</td><td>4.7&times; tighter per mask</td></tr>
          <tr><td>Detector noise, dimming the light</td><td>holds 1&nbsp;pW, fails 0.1</td><td>holds <strong>0.1&nbsp;pW</strong>, fails 0.01</td><td>10&times; looser</td></tr>
          <tr><td>Wavelength drift</td><td>holds 10&nbsp;nm, fails 20</td><td>holds <strong>20&nbsp;nm</strong>, fails 30</td><td>2&times; looser</td></tr>
        </tbody>
      </table>
    </div>
    <div class="prose col">
      <p>Three readings are worth separating. <strong>The constraint that binds does not move at
      all.</strong> Crosstalk fails at the same 0.25&nbsp;px, and the roughly 1&nbsp;px of blur a
      real device delivers destroys either design. Depth neither helps nor hurts the one thing that
      already made this unbuildable.</p>
      <p><strong>The two that loosened both come from catching more light.</strong> The deep stack
      lands <strong>79.1%</strong> of the incoming photons inside the detector boxes, against about
      60% for the shipped design, by the same steering rather than scattering that the accuracy gain
      comes from. More light at the readout means the design tolerates ten times more dimming before
      photon counting breaks down, and it gains room on wavelength too. In other words it became
      sturdier precisely where it already had nine orders of magnitude in hand.</p>
      <p><strong>Lost light points in opposite directions depending on how you count it, and the
      per-plate figure is the one that governs.</strong> Added up over the whole machine the deep
      candidate tolerates <em>more</em> loss, 12&nbsp;dB against 5. But that larger allowance is
      divided among eleven times as many surfaces, so the requirement per plate tightens to
      0.214&nbsp;dB. Component datasheets quote loss per surface, and 0.214&nbsp;dB sits at the
      optimistic end of the realistic 0.2&ndash;1&nbsp;dB range. Lost light moves from comfortable to
      marginal.</p>
      <div class="finding">
        <p class="tag">Finding &middot; depth adds a constraint without relieving one</p>
        <p class="body">Ten and a half points of accuracy cost <strong>twice the precision on every
        delay, one more bit of control per pixel, and 4.7&times; less tolerance for lost light at
        each surface</strong>, while the one error that already fails against real hardware
        <strong>does not move at all</strong>. This is why the deep model is labelled &ldquo;not
        shipped&rdquo; wherever it appears. Its number is real, and measured exactly the way the
        headline number was, but what it asks of the manufacturing is something this project has no
        evidence anyone can deliver. <strong>The trade is a more useful result than a clean win
        would have been.</strong></p>
      </div>
      <p>There is also something this budget <em>cannot</em> say. At 56 masks the plates sit
      0.53&nbsp;mm apart instead of 3&nbsp;mm, so being &plusmn;10&nbsp;&micro;m out of position goes
      from 0.33% of the gap to 1.9% of it. Past roughly forty plates the stack is better described as
      a <strong>solid block of glass</strong> than as separate sheets with air between them, and that
      is a different thing to manufacture. This budget covers <strong>errors in the components
      only</strong> and says nothing about getting them into position. Aligning and calibrating the
      machine, meaning plate spacing, sideways registration, delays that all read systematically high
      or low, and detectors reading slightly off zero, plausibly binds the deep design before
      anything in the table above does. <strong>That is flagged, not measured</strong>, and those are
      the next error sources to be written.</p>
    </div>
    <div class="plate-grid reveal">
      <figure class="plate"><img src="@@FIG_cand_crosstalk@@" alt="Candidate 56-mask network: accuracy vs thermal/pixel crosstalk" loading="lazy"><figcaption><span class="fign">Fig 9</span>Deep model, crosstalk. The same limit as the shipped design.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_cand_phase@@" alt="Candidate 56-mask network: accuracy vs per-pixel phase error" loading="lazy"><figcaption><span class="fign">Fig 10</span>Deep model, delay error. Twice as demanding.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_cand_detector@@" alt="Candidate 56-mask network: accuracy vs detector noise" loading="lazy"><figcaption><span class="fign">Fig 11</span>Deep model, detector noise. Ten times more forgiving.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_cand_loss@@" alt="Candidate 56-mask network: accuracy vs optical loss" loading="lazy"><figcaption><span class="fign">Fig 12</span>Deep model, lost light, totalled across all 56 plates.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_cand_wavelength@@" alt="Candidate 56-mask network: accuracy vs wavelength drift" loading="lazy"><figcaption><span class="fign">Fig 13</span>Deep model, wavelength drift. Twice as forgiving.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_cand_quant@@" alt="Candidate 56-mask network: accuracy vs DAC bit resolution" loading="lazy"><figcaption><span class="fign">Fig 14</span>Deep model, delay steps. One bit more demanding.</figcaption></figure>
      <figure class="plate"><img src="@@FIG_cand_confusion@@" alt="Candidate 56-mask network: as-built confusion matrix" loading="lazy"><figcaption><span class="fign">Fig 15</span>Deep model, which digits get confused for which.</figcaption></figure>
    </div>
    <div class="prose col">
      <p>There is no deep-model version of Fig&nbsp;8. That map puts one panel per mask in a single
      row, so at 56 masks it comes out <strong>16,139&nbsp;pixels wide and 341 tall</strong>, a ratio
      of 47:1, which works out to fifteen pixels high in a grid cell and thirty across the full
      column. There is no size at which it can be read, so it is left out rather than shown as a
      smear. It also costs <strong>2,016 separate evaluations</strong> to compute against 36 for the
      shipped design, roughly four hours and most of a full run. Both of those are the same fact seen
      from different sides: past about forty plates, anything quoted per mask stops being something
      you can look at.</p>
    </div>
  </section>

  <section class="phase planned reveal">
    <div class="phase-head col"><span class="ph-num next">next</span>
      <div><span class="badge-next">Planned</span>
      <p class="eyebrow">Error budget &middot; MZI mesh</p>
      <h2>Fabrication tolerance for the interferometer mesh</h2></div></div>
    <div class="prose col">
      <p>The same framework extends to the chip, switching on the two error sources unique to it:
      <strong>couplers that do not split light exactly in half</strong>, deviating from the intended
      50:50, and <strong>light lost in every interferometer</strong>, which leaves the mesh delivering less than it received. That
      second one genuinely changes the answer on a chip, unlike in the glass stack, where dimming
      everything equally cancels out of the readout. The interesting comparison is how each machine
      fails: <strong>errors piling up in sequence down 72 layers of interferometers</strong> against
      the glass stack&rsquo;s crosstalk between neighbouring pixels. Two optical computers, two
      different ways to lose the computation to fabrication.</p>
    </div>
  </section>

  @@NEXT@@

  <footer class="reveal">
    <div class="foot-grid">
      <div><h3>The boundary</h3><p>Python designs the ideal network. A single file crosses to MATLAB,
        which models the device as it would really be built and never writes back. The separation
        between design and reality is enforced by that one-way handover rather than by
        discipline.</p></div>
      <div><h3>Every magnitude sourced</h3><p>Delay error, bit depth, drift, detector noise, lost
        light and pixel crosstalk each trace to a published measurement or a datasheet, cited on the
        line where the number is defined. Anything that is a modelling choice is labelled as
        one.</p></div>
      <div><h3>Reproducible</h3><p>The same seeds give the same answers. The error-free baseline must
        come out at 0.7990 or the two models have drifted apart. A new candidate can be scored
        without disturbing the shipped one.</p></div>
    </div>
    <p class="colophon">photonn &middot; a portfolio study in optical computing and fabrication tolerance.
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
    <p class="standfirst">The shipped network scores 0.799 and cannot be pushed further by training.
    After 40 passes through 60,000 images it still cannot pull ahead even on the images it trained
    on. So the remaining levers are physical: <b>how far apart the masks sit</b>, and <b>how many
    there are</b>. This page is the running record of measuring them. <b>Nothing here is shipped
    yet.</b></p>
    <div class="stat-strip">
      <div class="stat"><span class="v">0.771</span><span class="l">the shipped 5-mask layout, re-run under the quick comparison protocol below</span></div>
      <div class="stat"><span class="v">0.852</span><span class="l">14 masks, same quick protocol, so directly comparable to 0.771</span></div>
      <div class="stat"><span class="v">0.904</span><span class="l">56 masks trained in full, so directly comparable to the 0.799 headline</span></div>
    </div>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">The constraint</p>
      <h2>A detector can only be reached by light that gets to it</h2></div></div>
    <div class="prose col">
      <p>Crossing one gap spreads light sideways by a fixed amount,
      <span class="q">reach = z&middot;&lambda;/(2&middot;dx&sup2;)</span>, which is about
      12.5&nbsp;px per 3&nbsp;mm gap in this design. Call that total spread the <strong>reach</strong>.
      For the network to be able to compute anything at all, every input pixel has to be able to
      influence every detector, and the hardest pair to connect here is <strong>74&nbsp;px</strong>
      apart. The shipped design clears that by <strong>0.8&nbsp;px</strong>.</p>
      <p>Fall below the bound and the failure is not a matter of degree, it is a matter of geometry.
      Part of the digit simply cannot reach the detector that needs to see it, no matter what the
      masks were trained to do. Drag the separation and watch the cone fall short.</p>
    </div>

    <div class="explorer-band reveal">
      <div class="pe-host"><div id="optics"></div></div>
      <p class="cap">Live. The cone is recomputed from the formula as you drag, the same expression
      the simulation itself uses in <span class="q">propagate.diffraction_reach_px</span>. The
      plotted points are real training runs, not a fitted curve.</p>
    </div>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">The trade</p>
      <h2>Separation and depth spend the same budget</h2></div></div>
    <div class="prose col">
      <p>The simulation grid is finite, and the Fourier method underneath it treats that grid as
      repeating forever, so light running off one edge quietly reappears on the opposite one. Call
      that <strong>wrap-around</strong>. It is an artefact of the simulation, not of any real optics,
      and it turns out to depend only on the <em>total</em> reach, not on how that reach is divided
      up: 2&nbsp;mm across 9 gaps and 3&nbsp;mm across 6 gaps both reach 74.8&nbsp;px and are both
      wrong by an identical 6.2&times;10<sup>&minus;4</sup>. So separation and mask count draw on
      <strong>one shared budget</strong>, about 150&nbsp;px, past which the simulation stops
      describing open air.</p>
      <p>Which makes the interesting question not &ldquo;more of which?&rdquo; but <strong>how to
      spend a fixed budget</strong>. Holding the total reach at 125&nbsp;px and trading distance for
      masks, so that every arrangement has identical reach <em>and</em> identical wrap-around error
      and the comparison is genuinely like for like, accuracy runs <strong>0.706 &rarr; 0.790 &rarr;
      0.831 &rarr; 0.852</strong> going from 2 masks to 14.</p>

      <div class="finding">
        <p class="tag">Finding</p>
        <p class="body"><strong>The plateau belonged to this particular arrangement of optics, not
        to the whole idea.</strong> This project used to read the shipped network&rsquo;s 0.799 as
        the ceiling that comes from being
        <a class="link" href="@@HREF_physics@@">one linear operation and one brightness reading
        &rarr;</a> That first part is true. The ceiling part was not. Fourteen masks reach
        <strong>0.852</strong> on a third of the data and under a third of the training, and they
        were still improving when the run stopped.</p>
      </div>
    </div>

    <figure class="plate reveal">
      <img src="@@FIG_optics_sweep@@" alt="Optics sweep: accuracy against diffractive reach, the reach-budget trade, and detector planes per configuration" loading="lazy">
      <figcaption><span class="fign">Fig 1</span>Left: accuracy against total reach at the shipped 5
      masks, with the 74&nbsp;px bound marked. Ringed points are arrangements whose
      <em>simulation</em> was corrupted by wrap-around, not designs that lost a fair fight. Right: the
      same 125&nbsp;px of reach spent on different numbers of masks. Below: the spreading cone and
      the detector plane for one fixed digit, worst to best.</figcaption>
    </figure>

    <div class="prose col">
      <p>The detector planes are the clearest part. At 2 and 5 masks the light arrives as a vague
      smear; at 9 and 14 it is gathered into <strong>distinct bright squares sitting right on the
      detector patches</strong>. Same reach, same physics. What the extra masks buy is the ability to
      <em>steer</em> light into the readout rather than merely scatter it in that direction.</p>
    </div>

    <h3 class="sub-h">See it for yourself</h3>
    <div class="prose col">
      <p>Since more masks keep paying, the deepest arrangement worth the computer time was trained
      properly rather than merely ranked: <strong>56 masks</strong> at 0.53&nbsp;mm gaps, the full
      60,000 images, then scored once against the same held-back test images the shipped model is
      quoted from. It reaches <strong>0.9040</strong> against <strong>0.7990</strong>. Both machines
      run here, live, on whichever digit you pick or one you draw, using the same physics as the
      Python original and checked against PyTorch to better than 10<sup>&minus;3</sup>.</p>
      <p>The gallery is the one the front page uses, deliberately stocked with <strong>six digits the
      shipped network gets wrong</strong>. The deep network gets three of those right and breaks none
      of the ten the shipped one already had. The three it still misses, it misses <em>in the same
      way</em>, calling them the same wrong digit as the shipped model does, which hints that those
      images are hard for optics in general rather than for this particular set of masks.</p>
      <p>Both machines are fed from <em>one</em> input, so the two columns differ in their optics and
      nothing else. Watch the detector plane rather than the answer. The shipped network spreads
      light across the whole plane and picks a weak winner out of it, while the deep stack drops the
      light inside the boxes. That is the difference between about 60% and <strong>79%</strong> of
      the incoming photons reaching a detector at all, and it is the same steering that the accuracy
      gain comes from.</p>
      <p>Two trained models come to <strong>0.7&nbsp;MB</strong> of delays on one page. The deep one
      carries <strong>eleven times</strong> as many numbers, 918k against 82k, for under six times
      the download, because its delays are stored more coarsely: <strong>4&nbsp;bits</strong> each,
      16 possible settings, against 8 bits and 256. Neither model is flattered by that. The error
      budget measures this design as holding its accuracy all the way down to <strong>3
      bits</strong>, and on the same 2,000 digits the stored-down models score
      <strong>0.9015</strong> and <strong>0.7995</strong> against the full-precision 0.9040 and
      0.7990, five thousandths either way, well inside the run-to-run noise of a test set that size.
      What runs in your browser is therefore the machine the study measured, not a lighter
      stand-in.</p>
      <p>One pass through the optics costs roughly <strong>11&nbsp;ms</strong> and
      <strong>100&nbsp;ms</strong> respectively, since the deep machine is 56 diffraction steps of
      arithmetic running in a browser tab, and both are far more than the 17&nbsp;ms an animation
      frame allows. So the board <strong>waits for you to pause</strong> rather than classifying
      mid-stroke, and it decides that by timing itself rather than by counting masks, so a board
      carrying only cheap models still follows the pen. The obvious alternative, keeping the fast
      column live and letting the deep one lag, was rejected on purpose: it would show two different
      digits side by side and invite exactly the wrong comparison.</p>
    </div>

    <div class="explorer-band reveal">
      <div class="pe-host"><div id="compare"></div></div>
      <p class="cap">Live. The shipped 5-mask network and the unshipped 56-mask candidate, one digit,
      both computed here. Every caption in the board is generated from the model files&rsquo; own
      records of how they were trained, so promoting a model means regenerating a file rather than
      editing this page.</p>
    </div>

    <h3 class="sub-h">What 56 masks actually looks like</h3>
    <div class="prose col">
      <p>The detector plane says the deep network is better. It does not say what the thing
      <em>is</em>. Here it is as a physical machine: the entrance, the phase plates and the
      detectors strung along the path of the beam, each carrying the light actually computed on it.
      Drag to orbit.</p>
      <p>Two things about this figure are honest rather than convenient. It draws only
      <strong>6 of the 56 masks</strong>, spread through the stack and each labelled with its real
      position, because fifty-six plates at 0.53&nbsp;mm spacing are fifty-six near-identical
      pictures. And it does not follow your pen: redrawing means a full pass through
      <strong>57 gaps</strong>, so it waits for you to press <span class="q">Refresh</span> rather
      than making the whole page stutter for a picture nobody studies mid-stroke.</p>
      <p>The proportions are the point. The shipped design is 18&nbsp;mm of optics across an opening
      of 1.02&nbsp;mm; this one is <strong>30&nbsp;mm across the same opening</strong>, a 29:1 needle
      that has to be squashed almost tenfold just to fit on a screen. Past about forty plates at this
      spacing, calling it a stack of separate masks is already a stretch. It is closer to a solid
      block of glass, which is a different thing to manufacture and one the
      <a class="link" href="@@HREF_tolerance@@">error budget &rarr;</a> does not model yet.</p>
    </div>

    <div class="explorer-band reveal">
      <div class="pe-host"><div id="stage3d"></div></div>
      <p class="cap">Live. The 56-mask candidate, fed the same digit as the board above. A sample of
      the planes, refreshed on demand; the haze between plates is the real light at those in-between
      depths, computed rather than shaded in.</p>
    </div>
  </section>

  <section class="phase reveal">
    <div class="phase-head col">
      <div><p class="eyebrow">What this does not show</p>
      <h2>Reading it honestly</h2></div></div>
    <div class="prose col">
      <p>The accuracies from the sweep come from a deliberately <strong>short comparison
      protocol</strong>, 20,000 images for 12 passes, because the question being asked there is which
      arrangement wins, not what the final accuracy would be. They are <strong>not comparable to the
      0.799 headline</strong>, which came from 60,000 images and 40 passes. The fair comparison is
      the shipped arrangement re-run under the same short protocol, which gives
      <strong>0.771</strong>. The 56-mask model in the board above is the exception. It was trained
      at the full budget and scored on the held-back test images, so its <strong>0.9040</strong> and
      the headline <strong>0.7990</strong> really are the same measurement.</p>
      <p>There is a confound to rule out. Adding a mask adds 128&sup2; more delays, so
      <strong>more masks also means more adjustable numbers</strong>, and the sweep alone cannot tell
      &ldquo;depth helps&rdquo; apart from &ldquo;more numbers help&rdquo;. Two pairs matched to have
      exactly the same count settle it. 56 masks on a 128&sup2; grid and 14 masks on a 256&sup2; grid
      both carry 917,504 delays, and they score <strong>0.889 against 0.856</strong>; the 80-versus-20
      pair, both at 1,310,720, gives 0.891 against 0.870. With the number of adjustable values held
      equal, <strong>depth wins</strong>. Each arrangement was run once, so trust the trend across the
      whole range, which is far larger than run-to-run noise, and not the gap between neighbours.</p>
      <p>The sweep never touched the held-back test images. That set is handed to the MATLAB model
      and every downstream number is quoted from it, so allowing the sweep to see it would quietly
      contaminate everything. A separate slice was carved out of the training data instead, and it
      was used once, at the very end, to score the single arrangement that had already won.</p>
      <p><strong>Nothing here is shipped, and the deep model is not simply better.</strong> The
      trained model, the <a class="link" href="@@HREF_index@@">browser classifier &rarr;</a> and the
      <a class="link" href="@@HREF_tolerance@@">error budget &rarr;</a> all still describe the
      5-mask, 3&nbsp;mm design. Running the full budget against the 56-mask candidate puts a price on
      those 10.5 points: <strong>twice the precision on every delay</strong> (0.3 down to
      0.15&nbsp;rad), <strong>one more bit of control per pixel</strong>, and <strong>4.7&times; less
      tolerance for light lost at each plate</strong>. Detector power and wavelength drift both get
      easier, for the same photon-catching reason the accuracy rose. But crosstalk between
      neighbouring pixels, the one error that already made this unbuildable on a real display chip,
      does not move at all. Depth adds a second binding constraint without relieving the first, and
      that trade, rather than the accuracy, is the result.</p>
    </div>
  </section>

  @@NEXT@@

  <footer class="reveal">
    <div class="foot-grid">
      <div><h3>The measurement</h3><p>Two sweeps: plate separation at a fixed mask count, then the
        same total reach spent different ways. Wrap-around error is measured against a reference run
        on a padded grid where it cannot occur, and decides which arrangements are worth training at
        all.</p></div>
      <div><h3>Reproducing it</h3><p><span class="q">apps/sweep_optics.py</span> runs both sweeps and
        saves a checkpoint per arrangement; <span class="q">apps/sweep_report.py</span> writes the
        figure and the data this page plots. Fixed random seeds throughout.</p></div>
      <div><h3>Status</h3><p>Live work, not a result. The numbers here move as more arrangements
        finish. The shipped design and everything downstream of it stay exactly where they are until
        a promotion is actually decided.</p></div>
    </div>
    <p class="colophon">photonn &middot; a portfolio study in optical computing and fabrication tolerance.
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
