# photonn — the generated site

Five pages, each a **single self-contained HTML document**: CSS, JavaScript and every figure are
inlined (figures as base64 data URIs), so they make **no external requests** and work offline —
including straight off `file://`.

They read in order, and the topbar lists all five:

1. **`index.html`** — *This neural network is made of light.* The trained D²NN running its
   forward pass in your browser: pick or draw a digit, watch it cross five phase masks onto ten
   detectors, in a **3D view of the optical stack** and as exact per-plane frames. Then what a
   phase mask actually computes, why anyone would build a computer out of light, and why the
   network is wrong about one digit in four — with the **confusion matrix of the ideal model**
   showing which digits it actually trades (5→3, 8→3, 9↔4).
2. **`physics.html`** — the angular-spectrum propagator, the sampling limit `z_crit` that bounds
   it, the **live diffraction explorer** with a reading guide for its controls, and the proof
   that the whole stack collapses to one linear operator followed by a single `|E|²`.
3. **`chip.html`** — the MZI mesh, the crosswalk table, and why a phase mask *is* a column of
   phase shifters. Prose and one figure only; the interactive correspondence widget was removed
   because the table already made its argument. It hands off to `/tolerance` on the one thing
   that separates the two machines: how they fail.
4. **`tolerance.html`** — the fabrication error budget, as **one section per error source**,
   grouped into two families: **Fabrication**, how precisely each part has to be made, and
   **Setup**, where the parts have to sit once they are. Each source carries a purpose-built
   widget showing what that error physically does, its measured tolerance curve, and the one
   number that matters. Then the ranking: five device sources are comfortable and crosstalk fails
   by 4×. **Setup is measured too** (issue #6, 2026-08-17): four more sources, of which lateral
   plate registration at **0.10 px = 0.8 µm** is the tightest number anywhere in the study, a
   33 µm connectivity bound the project had quoted for three phases turns out **not to be a
   tolerance**, and the three stochastic sources **fail when run together** at edges that each
   passed alone. **Closes on the same treatment applied to the interferometer
   chip**, which fails a different way — ten times tighter on delays
   because 72 columns of interferometers in series carry each other's errors forward, plus a
   constraint the stack cannot have at all (couplers that split unevenly, on the slider), and a
   quarter of one mesh that needs no fabrication tolerance whatever. That section sits here and
   not on `/chip` because in reading order the comparison only works once the stack's budget is
   behind you.
5. **`optics.html`** — what scaling buys, and where it stops. The depth-vs-accuracy chart, the
   5-mask model running beside the 56-mask one on one digit, what the
   extra masks cost in tolerance, and then the wall: a mask stack is one linear operator, so
   depth converges rather than compounds, and the way through is a nonlinearity.

Plus **`_artifact_body.html`** — a body-only variant of the front page for publishing as a
claude.ai Artifact (that host supplies its own `<head>`/`<body>`, and needs absolute links since
it has no sibling files). Gitignored; not part of the deployed site.

## Finding your way down a page

Every page carries an **in-page contents card** under its hero, and above 1500 px the same list
pins itself in the right margin as a rail with the section you are reading marked. It is one
element in two presentations, not two implementations: below the breakpoint the rail rules
simply do not apply, and the card stays where it is in the flow. 1500 px is where a 176 px rail
plus its gap clears the 1120 px content column on both sides; anything narrower would mean
pushing the prose off centre, which costs more than the rail is worth.

**The card is generated from the markup, never authored beside it** (`section_index`, `toc` in
`apps/build_site.py`) — the same bargain `PAGES` makes for the topbar. Every section heading sits
inside a `.phase-head` preceded by its own `<p class="eyebrow">`, so one scan finds them all,
gives each an id from its text, and returns the list. **Adding a section to a page adds it to
that page's index, with nothing to remember.** Two conventions feed it:

- The **number** comes from an eyebrow reading `Source N of M`, so the card cannot disagree with
  the prose about which source is which.
- The **label** is the heading text up to its first colon (`Crosstalk: pixels will not stay out
  of each other` → *Crosstalk*), because section headings are written as `Topic: gloss` wherever
  there is a topic to name. Headings that are whole sentences carry an explicit `data-toc`, which
  is stripped before the page is written and so costs nothing in the built file.
- **Nesting follows the outline**, not sibling order. `/tolerance` groups its sources into
  families with a `<h2 class="band-h">` and demotes the sources themselves to `<h3>`, styled
  identically. The closing chip comparison stays an `<h2>` and so is *not* indented under a
  family — which is correct, since it is a comparison rather than a seventh error source.

**The active mark is `aria-current="location"`, never `"page"`.** `tests/test_site_links.py`
asserts exactly one `aria-current="page"` anchor per page and that it is the topbar's own entry;
`location` is both the right ARIA value for an in-page index and what keeps that assertion
meaningful.

**It is driven by a throttled `scroll` listener, not an `IntersectionObserver`, and that is
deliberate.** "Which section am I reading" is a question about every heading at once, and an
observer answers only about the one that crossed — worse, it reports *nothing at all* when a
click on the card jumps the page straight over the crossing, which is the commonest way the card
is used. The reveal observer cannot be reused either: it unobserves on first sight. The first
version of this was observer-based, looked correct in a screenshot, and never updated.

That failure is invisible to the browser here: the driven Chrome tab is always hidden, and while
the page really does scroll (`window.scrollY` changes) **no `scroll` event is ever delivered and
no `IntersectionObserver` callback fires** — measured on this page, not assumed. So the mark is
checked under Node against the *built* page, with scroll position driven by hand:
`tests/toc_spy_runner.js` + `tests/test_toc_spy.py`. `tests/test_site_toc.py` covers the rest —
every fragment link resolving inside its own page, id uniqueness, and the grouping.

Anchor jumps clear the sticky topbar through `scroll-margin-top: 78px` on any heading carrying an
id. On `/tolerance` a jump deep into the page can land on a widget still showing *Warming up…*:
the error widgets are deferred until the reader nears them, and a jump outruns that by a frame or
two. It resolves itself.

**No em-dashes.** A site prose convention since 2026-08-11 that was enforced by hand until it
drifted: four pages held at zero while `/tolerance` accumulated 22, because nothing checked.
`tests/test_site_links.py::test_no_page_uses_an_em_dash` now checks, for both the `&mdash;`
entity and the literal character. En-dashes stay, in numeric ranges and in compound names like
Mach-Zehnder.

> `classifier.html` **was deleted** in the 2026-08-10 redesign. Its reader-facing half became the
> front page and its methodology moved to `physics.html`. `tests/test_site_links.py` fails if
> anything links to it again.

## Nothing here is precomputed, and nothing here is trusted

The browser runs the physics, and every port is held to the Python it came from rather than
to a screenshot of it.

- **Propagation.** `apps/web/asm.js` is a hand-written FFT and a faithful translation of
  `photonn.propagate.angular_spectrum`, in ~200 dependency-free lines.
  `tests/test_asm_crosscheck.py` runs it under Node against the NumPy reference: agreement
  **< 1e-6**. This is what makes every control on the diffraction explorer live, rather than
  the "only distance is live" version it replaced.
- **The classifier.** `tests/test_d2nn_crosscheck.py` runs `d2nn.js` against reference logits
  from PyTorch and asserts **identical predictions**, max class-score error **5.5e-7**. It
  also pins the bilinear resize to torch's `align_corners=False` convention to **1.2e-7** —
  the one place a half-pixel error would silently poison every prediction. Accuracy is
  **0.7995** at 8 bits, so the shipped gallery deliberately includes digits it gets wrong.
- **The chip.** `tests/test_mesh_web.py` + `tests/mesh_operator_runner.js` rebuild the mesh
  operator from `mesh_weights.js` under Node and check it against `photonn.mzi`: **4.5e-5**
  on a peak entry of 0.345, which is 16-bit phase codes compounding through 72 columns.

**The 3D optical stack** (`d2nn_stage.js`) draws the entrance plane, the masks and the
detector plane as parallel panels along the optical axis. An orthographic projection of a
flat plane is affine, so each panel is one `ctx.transform` + `drawImage` — no WebGL, no
library — and parallel non-intersecting panels make back-to-front painting exact. The haze
between them is the field at **intermediate depths, computed not faked**: sub-stepping a hop
is exact because `H(z₁)·H(z₂) = H(z₁+z₂)` while the band limit is inactive below
`z_crit = 15.40 mm`, which `tests/test_propagate.py` asserts. No rays are drawn — scalar
diffraction is not ray optics — and the slices are display-only: the cross-check asserts
`classify()` logits stay **bit-identical** with slicing enabled.

**Runtime came from a lookup table, not from cutting work.** A quantised bundle has only
`2^bits` distinct phases, so `d2nn.js` keeps codes as a `Uint8Array` plus a `2^bits` cos/sin
table rather than two full-length `Float64Array`s. The 56-mask model went **18.4 MB → ~0.9 MB**
retained; the forward pass is 11 ms shipped and 102 ms deep.

> **Canvas transforms must use `ctx.transform`, never `setTransform`.** `draw()` puts a
> devicePixelRatio scale on the context, and replacing the matrix drops it — which split the
> 3D stage apart on high-DPI screens.

## What a model is allowed to claim

Provenance lives **in the bundles, not in the JavaScript**. Each exporter writes `label`,
`accuracy`, `scored_on`, `protocol`, and optionally `caveat` / `not_scored_on`, and
`d2nn_compare.js` renders *every* caption from it. So changing what a model claims is
regenerating a bundle, never a JS edit, and `tests/test_web_contract.py` enforces that.

**A model is named for its depth, never for its standing in this project** — the labels are
`5 masks`, `56 masks`, `14 masks`. The 56-mask network runs live in the browser, and a page
that labelled it "not shipped" while a visitor was operating it was showing an internal
workflow state nobody could act on. What replaced the badge is the *cost*: *"Buying those
points costs 2x tighter phase control and 4.7x lower loss per mask."* Comparability is a
separate sentence keyed on `not_scored_on`, not on status. `test_no_caption_describes_a_model_by_its_status`
scans every committed bundle and fails on any status word.

## Widgets

| File | Page | What it does |
|---|---|---|
| `d2nn.js` + `d2nn_demo.js` | index | the live classifier |
| `d2nn_stage.js` | index, optics | the 3D optical stack |
| `explorer.js` | physics | the diffraction explorer |
| `errors.js` | tolerance | **seven** error-mechanism widgets, one per source |
| `mesh_weights.js` | tolerance | the trained chip's 2,628 settings, 16-bit, for `errors.js` |
| `scaling.js` | optics | accuracy against mask count |
| `d2nn_compare.js` | optics | 5 masks vs 56 masks, one digit |
| `analogy.js` | *(none)* | kept for `apps/analogy_demo.py`; the site no longer mounts it |

`errors.js` is **mechanism only**: it never runs a classifier and computes no accuracy, so
it cannot contradict the measured curve beside it. Both models it draws are the real thing. The
128² phase mask is cut out of `d2nn_weights.js` at build time by
`build_site.error_mask_bundle()` — one copy of the trained phases in the repo, and this is a
slice of it. The chip comes from `mesh_weights.js`, written by `apps.export_mesh_web`, and
`tests/test_mesh_web.py` rebuilds the operator from it under Node and checks it against
`photonn.mzi`, because a transposed index would draw a confident picture of a different chip.

All seven mount on `/tolerance`, in page order `crosstalk, phase, detector, loss, wavelength,
quant, mesh` — the first six against the stack, `mesh` in the closing section against the chip.
The file is inlined once, and is never split, because all seven share a stylesheet and a second
copy would mean a second `STYLE_ID` — a bug this repo has shipped once. They split into two
shapes with **two layout rules that are load-bearing** (both were broken and fixed in
`cd372b1`):

* **Five triptychs** (crosstalk, phase, wavelength, quant, mesh) — three square panels reading
  as-designed / as-built / the difference. They must **never wrap**, because sweeping the
  slider only shows the mechanism if all three are on screen at once. They were
  `flex: 1 1 150px; min-width: 132px`, which needs 420 px of row; a phone column is about
  310 px, so the third panel wrapped, the rest grew to full width, and the reader got one
  enormous picture at a time. Now `flex: 1 1 0; min-width: 0` on a row that cannot wrap, so
  three always fit and simply shrink — about 95 px each at 300 px of column.
* **Two plots** (loss, detector) — one wide pane, capped at 640 px by `.ex-pane.ex-wide`.
  A plot must **measure its pane before it draws**. These used a fixed 420-unit drawing space
  with an inline height and let `width:100%` scale the bitmap to whatever the column really
  was, which scales x without scaling y: flattened on a desktop, stretched tall on a phone,
  labels and all. `fitCanvas()` now measures first and draws one unit per CSS pixel, and
  `onWidthChange()` redraws on a real width change (guarded on the measured width, since the
  redraw sets the canvas height and would otherwise answer its own observer forever).

> **The stylesheet is a JS template literal, so it must contain no backticks.** One anywhere
> inside it — including inside a CSS comment — terminates the literal and breaks the whole
> widget. This has happened once; the Node runner above is what caught it.

Neither rule can be checked in a driven browser — that tab is always hidden, so it never lays
anything out. `tests/error_widget_runner.js` mounts all seven kinds against a DOM stand-in at
300/480/1042 px and device pixel ratios 1 and 2, and `tests/test_error_widgets.py` asserts that
every canvas's **bitmap aspect equals the aspect it is displayed at**, plus the stylesheet rules
above. The runner's mini-flexbox **parses its rules out of `errors.js`'s own CSS** rather than
restating them, so "three panels stay on one row" tests the stylesheet that ships and not the
test file.

## Maths is MathML

Expressions are MathML, which every current browser renders natively, so real notation costs
no library. `apps/build_site.py` holds a compact token notation (`mrow`, `mfrac`, `msqrt`)
expanded into a `MATH` dict and substituted as `@@MATH_*@@`, mirroring `@@FIG_*@@`. Code and
UI identifiers keep the `.q` mono pill; only mathematics becomes MathML.

> **Never set CSS `display` or `overflow` on a `<math>` element.** MathML lays out as
> `display: math`/`block math`, and overriding it drops the element into ordinary CSS block
> layout, which puts every child on its own line — a nine-term operator product renders as
> nine stacked rows. The scroll container lives on the `.eq` wrapper instead.

## Regenerating

These pages are **generated, not hand-edited** — edit `apps/build_site.py`, or the browser
sources under `apps/web/`, and rebuild:

```bash
python -m apps.build_site      # writes all six files above
```

The build is deterministic: running it twice produces byte-identical output.

Navigation is generated from one list. `apps.build_site.PAGES` holds every page's filename, nav
label, title and hand-off blurb; the topbar, the sequential "next" card and the relative →
absolute link swap the Artifact needs are all derived from it. **Adding or renaming a page means
editing `PAGES`, not the markup.** Links between pages are written as `@@HREF_<key>@@` tokens and
resolved last, which is what lets one rendered body be emitted twice.

If the **trained model** changes, regenerate in this order before rebuilding the site:

```bash
python -m apps.export_d2nn_web      # -> apps/web/d2nn_weights.js, tests/fixtures/d2nn_reference.json
python -m apps.export_analogy_web   # -> apps/web/analogy_geom.js
python -m apps.export_mesh_web      # -> apps/web/mesh_weights.js   (needs handoff schema 0.2.0)
python -m apps.analogy_figure       # -> docs/figures/phase3_correspondence.png
python -m apps.build_site
```

If the **optics sweep** is re-run, regenerate its bundle too — the depth-vs-accuracy chart
reads its points from there rather than carrying its own copy:

```bash
python -m apps.sweep_report         # -> apps/web/optics_sweep.js, docs/figures/optics_sweep.png
```

`apps/web/d2nn_weights.js`, `apps/web/analogy_geom.js` and `apps/web/mesh_weights.js` are
**committed on purpose**: `.gitignore` excludes `*.h5`/`*.pt`, so they are the repo's only copies
of what the trained models say, and the only way the site rebuilds from a fresh clone.

The standalone one-widget pages (`apps/d2nn_demo.html`, `apps/diffraction_explorer.html`,
`apps/analogy_demo.html`, `apps/compare_demo.html`) are built by `python -m apps.d2nn_demo`,
`python -m apps.diffraction_explorer`, `python -m apps.analogy_demo` and
`python -m apps.compare_demo` respectively — handy for eyeballing a single widget without
rebuilding the whole site.

## Quoted figures are hand-maintained

**Nothing checks the numbers on these pages against the model.** They are Python string
constants in `apps/build_site.py`, and they have gone stale before — the tolerance edges on the
live site were the pre-retrain values for four months. After any retrain or any re-run of the
error budget, re-grep `build_site.py` for the accuracies, the power budget *and* the tolerance
edges, not just the headline — `/tolerance` now quotes edges for **two** machines, so a mesh
re-run moves numbers there too. `docs/tolerance_d2nn.md`, `docs/tolerance_mesh.md` and
`docs/phase2_dnn.md` are the sources of record.

## Publishing

`.github/workflows/pages.yml` deploys this folder to GitHub Pages on every push to `main`
(GitHub Actions Pages source, so an arbitrary subfolder can be served — the branch-deploy UI
only offers `/` or `/docs`). Live at **https://roosado.github.io/photonn/**.

> **The workflow only publishes `site/`; it never runs a build.** Nothing regenerates these
> pages in CI, so they must be rebuilt locally and committed whenever their sources change, or
> the deployed site will silently lag the repo.
