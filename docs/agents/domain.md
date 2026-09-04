# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

This is a **single-context** repo: one `CONTEXT.md` and one `docs/adr/`, both at the root.

## Before exploring, read these

- **`CLAUDE.md`** at the repo root: it is the standing project context — architecture, phase objectives, scope boundaries, working conventions.
- **`CONTEXT.md`** at the repo root, if it exists: the glossary.
- **`docs/adr/`**: read ADRs that touch the area you're about to work in.

If `CONTEXT.md` or `docs/adr/` doesn't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The `/domain-modeling` skill (reached via `/grill-with-docs` and `/improve-codebase-architecture`) creates them lazily when terms or decisions actually get resolved.

## Decisions already recorded in CLAUDE.md

`CLAUDE.md` carries an `## Open decisions` section with a `### Resolved` subsection. That subsection is this project's existing decision log — PyTorch over JAX, Phase 4 ordering, parameter-source standardization — and it exists specifically so a later session does not reopen a settled question.

Read it before writing an ADR. Do not create an ADR that restates a decision already recorded there. If a new decision genuinely warrants an ADR, write the ADR in `docs/adr/` and leave a one-line pointer to it in the `### Resolved` list rather than duplicating the reasoning in both places.

## File structure

```
/
├── CLAUDE.md                          ← standing project context + decision log
├── CONTEXT.md                         ← glossary (created lazily)
├── docs/
│   ├── adr/                           ← created lazily
│   │   └── 0001-....md
│   ├── parameter_sources.md           ← physical-constant ledger, not an ADR
│   ├── tolerance_d2nn.md
│   └── tolerance_mesh.md
├── photonn/                           ← Python: physics, models, training
└── photonn-hw/                        ← MATLAB: error modelling
```

Existing files in `docs/` are results and ledgers, not domain docs. Don't treat them as ADRs; don't rewrite them into ADR form.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal: either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders), but worth reopening because…_
