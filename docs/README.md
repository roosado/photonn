# docs/ — the written record

Each file answers one question and is meant to be read on its own. The root
[`README.md`](../README.md) is the tour; these are the reference.

**Everything measured is stated once, here.** If a number appears on the site or in a
commit message, this is where it comes from — and if the two ever disagree, these files
are right.

## By question

| If you want to know | Read |
|---|---|
| How light is moved across a gap, and where each propagator stops being valid | [`wave_optics.md`](wave_optics.md) |
| What the diffractive network is, what its masks do optically, its power budget, and **why linearity caps it** | [`phase2_dnn.md`](phase2_dnn.md) |
| What depth buys, and why more of it is not the answer | [`phase2_dnn.md` § What the optics can still buy](phase2_dnn.md#what-the-optics-can-still-buy) |
| Whether the light that misses the detectors is recoverable accuracy | [`phase2_dnn.md` § The light that misses the boxes](phase2_dnn.md#the-light-that-misses-the-boxes-is-not-headroom-re-scored-2026-08-17) |
| What the interferometer mesh is, and **why it and the glass stack are the same machine** | [`phase3_mesh.md`](phase3_mesh.md) |
| How precisely the **diffractive network** must be built | [`tolerance_d2nn.md`](tolerance_d2nn.md) |
| How precisely it must be **assembled**, as opposed to fabricated | [`tolerance_d2nn.md` § Geometry](tolerance_d2nn.md#geometry-where-the-parts-sit) |
| How precisely the **chip** must be built, and how the two failure modes differ | [`tolerance_mesh.md`](tolerance_mesh.md) |
| Where every physical constant comes from, and which ones are still unsourced | [`parameter_sources.md`](parameter_sources.md) |
| What crosses the Python → MATLAB boundary, and in what format | [`handoff_schema.md`](handoff_schema.md) |
| What the scaffolding looked like before any physics was trained | [`phase0_baseline.md`](phase0_baseline.md) |

## Two documents that are one study

[`tolerance_d2nn.md`](tolerance_d2nn.md) and [`tolerance_mesh.md`](tolerance_mesh.md) share
a method, a threshold convention and a section structure on purpose, so they can be read
side by side. Both quote limits as the **bracket the sweep resolves** — *holds at X, fails
at Y* — never an interpolated crossing, and both measure against **95 % of their own
model's ideal**, so the edges compare even though the accuracies do not.

The difference between them is the result: the mesh needs its phases **10× more accurate**,
because its error accumulates through 72 MZI columns in series where the stack's acts in
parallel across five masks and averages.

**Only the D²NN has a geometry half.** Where a plate sits is set at assembly; where a
waveguide sits is set by lithography, so the chip has no equivalent sources and its document
has no equivalent section. That asymmetry is a result about the two machines, not a gap in
the mesh study.

They differ in one other way, deliberately. **Only the D²NN's device sources have realistic
values to compare against**, from the display-device literature, which is why "crosstalk fails
by 4×" is the one verdict this project can state. The mesh's column is entirely `UNSOURCED`
(it needs foundry PDK data) and so is the D²NN's geometry column (it needs optomechanical
measurement) — three different literatures, two of them unread. No value is invented to fill
either. Both gaps are tables in [`parameter_sources.md`](parameter_sources.md), not silences.

## Elsewhere in the repo

| | |
|---|---|
| [`../CLAUDE.md`](../CLAUDE.md) | architecture, phase roadmap, scope boundaries, and the decisions that are closed |
| [`../site/README.md`](../site/README.md) | the generated site: what each page argues, the widgets, and how the browser ports are held to the models |
| [`../photonn-hw/`](../photonn-hw/) | the MATLAB as-built side; each package has a `Contents.m` |
| [`figures/`](figures/) | figures referenced from these documents |

**The trained models are not in the repo.** `.gitignore` excludes `*.h5`, `*.pt` and all of
`exports/`, so the handoffs and checkpoints are local only. The repo's record of what the
models say is the committed browser bundles in `apps/web/` (`d2nn_weights.js`,
`d2nn_deep_weights.js`, `d2nn_sweep_weights.js`, `mesh_weights.js`) — quantised, so not
bit-identical to the float models, and the gap is measured in `phase2_dnn.md`. Regenerate
the real files with `python -m apps.train_d2nn` or `python -m apps.train_mesh`.
