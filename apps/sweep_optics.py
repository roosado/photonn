"""Optics sweep: does changing the *optics* lift the D2NN off its 0.799 plateau?

The Phase-2 retrain established that the network is capacity-limited, not
data-limited -- 40 epochs on the full 60 000 ends at train 0.798 vs val 0.799, so
it cannot pull ahead on its own training set. ``docs/phase2_dnn.md`` names the two
remaining levers as optical: **inter-plane separation** ``z`` (more mixing per
hop) and **mask count** ``L``. This sweeps both and measures what they buy.

Three things this deliberately does *not* do, all CLAUDE.md scope:

* nothing electronic moves -- no LR schedule, no augmentation, no larger head;
  only the optical geometry changes, everything else is held at shipped values;
* model selection never reads the frozen test set (see ``train.split_dataset``);
* the propagator is not modified. ``--geometry`` measures how wrong the finite
  grid is at each ``z`` instead, and the sweep is bounded by that measurement.

Run (from the repo root, in the project venv)::

    python -m apps.sweep_optics --geometry           # free; gates everything below
    python -m apps.sweep_optics --arm z              # ~2.7 h
    python -m apps.sweep_optics --arm layers --z 3   # ~2.1 h, at the winning z

``--grid`` re-measures any of it at another field size. ``dx`` is held fixed, so
a larger grid is a physically larger device rather than a finer sampling of the
same one -- which matters, because the detector layout and the entrance window
are both *fractions* of the grid, so the reach the design requires grows with it
too. The table prints that requirement beside the reach for exactly this reason.

Results accumulate in ``exports/sweep/optics_sweep.json`` (one record per config,
re-runnable arm by arm). ``exports/`` is gitignored, so anything the docs or the
site need is re-emitted to a committed path by ``apps/sweep_report.py``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from photonn.detect import default_regions
from photonn.fields import Field
from photonn.models import D2NN
from photonn.propagate import angular_spectrum, check_sampling, diffraction_reach_px, wraparound_error
from photonn.train import encode_input, evaluate, load_dataset, split_dataset, train

_REPO = Path(__file__).resolve().parent.parent
_OUT = _REPO / "exports" / "sweep" / "optics_sweep.json"

# Shipped Phase-2 operating point; only `separation` and `layers` are swept.
# The grid is a *default*, not a constant: `--grid` re-measures the same table at
# another size, which is how the "has this design run out of grid?" question gets
# an answer without training anything. dx is held fixed, so a larger grid is a
# physically larger device at the same pitch -- see `report_geometry`.
DEFAULT_GRID, DX, WAVELENGTH = 128, 8e-6, 532e-9
BASE_Z_MM, BASE_LAYERS = 3.0, 5

Z_LIST_MM = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0)
LAYER_LIST = (1, 3, 8, 12)

# Iso-reach arm. The z sweep found that wrap error depends on *total* reach and
# not on how it is split, so separation and depth draw on one shared budget --
# which means "does depth help?" cannot be asked by adding masks at fixed z
# (that just spends more budget). Holding total reach at the winning 124.7 px and
# varying the split asks the sharper question instead: given a fixed budget, is
# it better spent on distance or on masks?
#
# Total reach = z * reach_per_mm * (L + 1), so iso-reach means z*(L+1) is
# constant -- 30.0 mm-hops at the winner. Note the arm is deliberately NOT
# parameter-matched: L=2 carries 32 768 phases and L=14 carries 229 376. If the
# deeper, larger models do not pull ahead at equal reach, that is a statement
# about the linearity ceiling, not about reach.
ISO_LAYERS = (2, 5, 9, 14)
ISO_REACH_PX = 5.0 * (WAVELENGTH / (2 * DX**2) * 1e-3) * 6      # the z=5mm, L=5 winner


def config_tag(grid: int, z_mm: float, layers: int) -> str:
    """Identity of one configuration in the results file.

    The default grid keeps the bare ``z2mm_L14`` form: those tags are already in
    ``exports/sweep/optics_sweep.json``, name the ``masks_*.npy`` beside it, and
    are referenced by :mod:`apps.export_sweep_web`. Another grid takes a prefix,
    so re-measuring at 256 cannot silently overwrite a 128 result that a
    published figure is quoting.
    """
    stem = f"z{z_mm:g}mm_L{layers}"
    return stem if grid == DEFAULT_GRID else f"n{grid}_{stem}"


def _mm_list(text: str) -> tuple:
    return tuple(float(v) for v in text.replace(" ", "").split(",") if v)


def _int_list(text: str) -> tuple:
    return tuple(int(v) for v in text.replace(" ", "").split(",") if v)


def parse_args():
    p = argparse.ArgumentParser(description="Sweep D2NN optical geometry.")
    p.add_argument("--arm", choices=["z", "layers", "iso"], help="which sweep to run")
    p.add_argument("--geometry", action="store_true",
                   help="print the geometry + wrap table and exit (no training)")
    p.add_argument("--z", type=float, default=BASE_Z_MM,
                   help="separation in mm; fixed value for the layers arm")
    p.add_argument("--layers", type=int, default=BASE_LAYERS,
                   help="mask count; fixed value for the z arm")
    p.add_argument("--grid", type=int, default=DEFAULT_GRID,
                   help="field grid size; dx is held fixed, so a larger grid is a "
                        "physically larger device (CLAUDE.md caps this at 512)")
    p.add_argument("--z-list", type=_mm_list, default=Z_LIST_MM, dest="z_list",
                   help="comma-separated separations in mm for the z arm")
    p.add_argument("--iso-reach", type=float, default=ISO_REACH_PX,
                   help="total reach in px held constant across the iso arm")
    p.add_argument("--iso-layers", type=_int_list, default=ISO_LAYERS, dest="iso_layers",
                   help="comma-separated mask counts for the iso arm")
    p.add_argument("--epochs", type=int, default=12, help="reduced ranking protocol")
    p.add_argument("--subset-train", type=int, default=20000)
    p.add_argument("--subset-val", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--mask-init-std", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=20260724)
    p.add_argument("--split-seed", type=int, default=20260806)
    p.add_argument("--force", action="store_true",
                   help="re-run configs already present in the results file")
    p.add_argument("--max-wrap", type=float, default=0.02,
                   help="skip configs whose stack wrap mis-states the detector "
                        "readings by more than this fraction (0 disables the cap)")
    p.add_argument("--out", default=str(_OUT))
    return p.parse_args()


# -- geometry: what the grid can and cannot represent ---------------------------
def stack_wraparound(z: float, n_hops: int, fields: np.ndarray, *, grid: int, pad: int = 3):
    """Wrap error accumulated over ``n_hops`` hops, at the plane and at the detectors.

    Single-hop :func:`~photonn.propagate.wraparound_error` understates the D2NN
    badly: the model takes ``n_layers + 1`` hops and the wrapped energy compounds.
    What the classifier actually consumes is not the field but ten
    region-integrated intensities, and those are far more forgiving -- the patches
    sit in the central 75 % of the grid, while wrapped energy arrives at the edges.
    Both are reported because the gap between them *is* the finding.

    Masks are omitted (identity), so this is pure geometry and needs no trained
    model -- which is what lets it gate the sweep instead of following it.
    """
    regions = default_regions(grid, 10)
    lo = (pad * grid - grid) // 2

    def chain(data):
        f = Field(data, DX, WAVELENGTH)
        for _ in range(n_hops):
            f = angular_spectrum(f, z)
        return f.data

    def integrate(intensity):
        return np.array([intensity[r.y0:r.y1, r.x0:r.x1].sum() for r in regions])

    plane, logit, flips = [], [], 0
    for k in range(len(fields)):
        on_grid = chain(fields[k])

        big = np.zeros((pad * grid, pad * grid), dtype=complex)
        big[lo:lo + grid, lo:lo + grid] = fields[k]
        reference = chain(big)[lo:lo + grid, lo:lo + grid]

        plane.append(np.linalg.norm(on_grid - reference) / np.linalg.norm(reference))
        a, b = integrate(np.abs(on_grid) ** 2), integrate(np.abs(reference) ** 2)
        logit.append(np.linalg.norm(a - b) / np.linalg.norm(b))
        flips += int(np.argmax(a) != np.argmax(b))

    return {"plane_error": float(np.mean(plane)), "logit_error": float(np.mean(logit)),
            "argmax_flips": flips, "n_probe": len(fields)}


def geometry_row(z_mm: float, layers: int, fields: np.ndarray, *, grid: int) -> dict:
    z = z_mm * 1e-3
    hops = layers + 1
    reach = diffraction_reach_px(grid, DX, WAVELENGTH, z)
    probe = Field(fields[0], DX, WAVELENGTH)
    row = {
        "grid": grid,
        "z_mm": z_mm, "layers": layers, "hops": hops,
        "reach_px_per_hop": reach, "reach_px_total": hops * reach,
        "required_px": required_reach_px(grid),
        "sampling_ok": bool(check_sampling(probe, z).ok),
        "wrap_one_hop": float(np.mean([wraparound_error(Field(f, DX, WAVELENGTH), z)
                                       for f in fields])),
    }
    row.update(stack_wraparound(z, hops, fields, grid=grid))
    return row


def required_reach_px(grid: int) -> float:
    """Total reach the stack needs before it can compute the mapping at all.

    Worst case, per axis: the input pixel at one edge of the entrance window must
    be able to influence the detector pixel farthest from it. Below this the
    failure is geometric rather than statistical -- part of the digit physically
    cannot reach the detector that needs it, whatever the masks say. Same
    derivation as :mod:`apps.export_analogy_web`; duplicated here rather than
    imported because that module reads the trained handoff and this must run for
    a grid no model has been trained on yet.

    Note it scales with the grid: ``default_regions`` and ``train._embed`` both
    place things as *fractions* of ``n``, so a larger grid is a proportionally
    larger device and needs proportionally more reach. That is exactly why the
    wrap budget alone does not answer the grid question.
    """
    win = max(1, int(round(0.5 * grid)))                # train.embed_input(input_frac=0.5)
    off = (grid - win) // 2
    window = [off, off + win - 1]
    regions = [[r.y0, r.y1, r.x0, r.x1] for r in default_regions(grid, 10)]
    det_x = [min(r[2] for r in regions), max(r[3] - 1 for r in regions)]
    det_y = [min(r[0] for r in regions), max(r[1] - 1 for r in regions)]
    return float(max(window[1] - det_x[0], det_x[1] - window[0],
                     window[1] - det_y[0], det_y[1] - window[0]))


def probe_fields(n: int = 16, *, grid: int) -> np.ndarray:
    """A few encoded test digits -- the actual thing the network propagates."""
    ds = load_dataset("mnist", subset=n, split="test")
    imgs = torch.as_tensor(ds.images[:n], dtype=torch.float32)
    return encode_input(imgs, scheme="both", n=grid).numpy()


def iso_reach_configs(target_px: float = ISO_REACH_PX, layers=ISO_LAYERS):
    """``(z_mm, layers)`` pairs that all reach the same total distance."""
    per_mm = WAVELENGTH / (2 * DX**2) * 1e-3           # px of reach per mm per hop
    return [(round(target_px / (per_mm * (L + 1)), 4), L) for L in layers]


def candidate_geometry(args, fields):
    """Geometry rows for whichever arm is selected.

    Mask count is not a free parameter optically -- ``L`` masks means ``L + 1``
    hops, so raising it raises total reach as well. Sweeping it therefore needs
    the same wrap check the z arm got, over the same probe digits.
    """
    if args.arm == "layers":
        return [geometry_row(args.z, layers, fields, grid=args.grid) for layers in LAYER_LIST]
    if args.arm == "iso":
        return [geometry_row(z, L, fields, grid=args.grid)
                for z, L in iso_reach_configs(args.iso_reach, args.iso_layers)]
    return [geometry_row(z_mm, args.layers, fields, grid=args.grid) for z_mm in args.z_list]


def report_geometry(args, fields=None):
    fields = probe_fields(grid=args.grid) if fields is None else fields
    z_crit = args.grid * DX**2 / WAVELENGTH
    if args.arm == "iso":
        varying, fixed = "the z/L split", f"total reach {args.iso_reach:.1f} px"
    elif args.arm == "layers":
        varying, fixed = "mask count", f"z = {args.z:g} mm"
    else:
        varying, fixed = "separation", f"{args.layers} masks"
    need = required_reach_px(args.grid)
    print(f"grid {args.grid} ({args.grid * DX * 1e3:.3f} mm across), dx {DX*1e6:g} um, "
          f"lambda {WAVELENGTH*1e9:g} nm | varying {varying}, {fixed}")
    print(f"z_crit = {z_crit*1e3:.3f} mm   reach/hop = z x {WAVELENGTH/(2*DX**2):.1f} px/m")
    print(f"required reach = {need:.0f} px  (worst input pixel -> farthest detector pixel)\n")
    print(f"{'z (mm)':>7} {'masks':>6} {'reach/hop':>10} {'total':>8} {'/need':>7} "
          f"{'sampling':>9} {'1-hop wrap':>11} {'stack wrap':>11} {'logit err':>10} {'flips':>7}")

    rows = candidate_geometry(args, fields)
    for r in rows:
        print(f"{r['z_mm']:>7.2f} {r['layers']:>6} {r['reach_px_per_hop']:>10.2f} "
              f"{r['reach_px_total']:>8.1f} {r['reach_px_total'] / need:>6.2f}x "
              f"{'ok' if r['sampling_ok'] else 'ALIAS':>9} "
              f"{r['wrap_one_hop']:>11.2e} {r['plane_error']:>11.2e} {r['logit_error']:>10.2e} "
              f"{r['argmax_flips']:>4}/{r['n_probe']}")

    print("\nThe sampling column is check_sampling (|z| <= z_crit). It passes for every")
    print("row above and says nothing about wrap -- that is what the last three measure.")
    _write(args.out, {"geometry": rows})
    return rows


# -- the sweep ------------------------------------------------------------------
def _write(path, update: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = json.loads(path.read_text()) if path.exists() else {}
    # Both lists accumulate across arms rather than replacing: running the layers
    # arm must not discard the z arm's rows, which the report figure reads.
    # Grid is part of a geometry row's identity, defaulting to 128 for the rows
    # written before --grid existed, so re-measuring at 256 adds rows instead of
    # overwriting the ones the published figure quotes.
    keys = {"runs": lambda r: r["config"],
            "geometry": lambda r: (r.get("grid", DEFAULT_GRID), r["z_mm"], r["layers"])}
    for key, value in update.items():
        if key in keys:
            ident = keys[key]
            existing = {ident(r): r for r in doc.get(key, [])}
            existing.update({ident(r): r for r in value})
            doc[key] = list(existing.values())
        else:
            doc[key] = value
    path.write_text(json.dumps(doc, indent=2))
    print(f"\nwrote {path}")


def _all_runs(path) -> list:
    path = Path(path)
    return json.loads(path.read_text()).get("runs", []) if path.exists() else []


def _completed(path) -> set:
    return {r["config"] for r in _all_runs(path)}


def run_config(z_mm: float, layers: int, train_ds, val_ds, args) -> dict:
    z = z_mm * 1e-3
    tag = config_tag(args.grid, z_mm, layers)
    model = D2NN(n=args.grid, n_layers=layers, dx=DX, wavelength=WAVELENGTH,
                 separation=z, mask_init_std=args.mask_init_std)
    n_params = sum(p.numel() for p in model.parameters())
    reach = diffraction_reach_px(args.grid, DX, WAVELENGTH, z)

    print(f"\n=== {tag} | {n_params} params | reach {reach:.2f} px/hop, "
          f"{(layers + 1) * reach:.1f} px over {layers + 1} hops ===")

    t0 = time.time()
    if layers == 0:
        history = {"note": "untrained: pure diffraction, 0 trainable parameters"}
    else:
        model, history = train(model, train_ds, epochs=args.epochs, seed=args.seed,
                               lr=args.lr, batch_size=args.batch, val_dataset=val_ds)
    val_acc = evaluate(model, val_ds)
    elapsed = time.time() - t0

    # Masks alongside the JSON: the report figure draws each configuration's own
    # detector plane, and retraining a config just to look at it costs ~10 min.
    if layers:
        masks = np.stack([m.phi.detach().cpu().numpy() for m in model.masks])
        mask_path = Path(args.out).with_name(f"masks_{tag}.npy")
        np.save(mask_path, masks.astype("f4"))

    print(f"--- {tag}: val {val_acc:.4f} in {elapsed/60:.1f} min")
    return {
        "config": tag, "grid": args.grid,
        "z_mm": z_mm, "layers": layers, "n_params": n_params,
        "reach_px_per_hop": reach, "reach_px_total": (layers + 1) * reach,
        "val_acc": val_acc, "seconds": elapsed, "history": history,
        "protocol": {"epochs": args.epochs, "n_train": len(train_ds),
                     "n_val": len(val_ds), "seed": args.seed,
                     "split_seed": args.split_seed},
    }


def main():
    args = parse_args()
    if args.geometry or args.arm is None:
        report_geometry(args)
        if args.arm is None:
            print("\n(no --arm given: geometry only)")
        return

    torch.manual_seed(args.seed)
    pool = load_dataset("mnist", subset=args.subset_train + args.subset_val, split="train")
    train_ds, val_ds = split_dataset(pool, n_val=args.subset_val, seed=args.split_seed)
    print(f"train={len(train_ds)}  val={len(val_ds)}  (disjoint, carved from the train "
          f"split; the frozen test set is not touched)")

    # Wrap first, training second: a configuration whose simulation does not
    # describe free space is not worth ten minutes of gradient descent, and
    # scoring it would invite reading a numerical artifact as a design.
    rows = report_geometry(args)
    capped = args.max_wrap > 0
    clean = {(r["z_mm"], r["layers"]) for r in rows
             if not capped or r["logit_error"] <= args.max_wrap}
    dropped = [r for r in rows if capped and r["logit_error"] > args.max_wrap]
    if dropped:
        print(f"\nskipping {len(dropped)} configuration(s) over the "
              f"{args.max_wrap:.0%} wrap budget:")
        for r in dropped:
            print(f"  z={r['z_mm']:g}mm L={r['layers']}: {r['reach_px_total']:.0f} px reach, "
                  f"logit error {r['logit_error']:.1%}")

    if args.arm == "z":
        configs = [(z, args.layers) for z in args.z_list if (z, args.layers) in clean]
        configs.insert(0, (args.z, 0))          # pure-diffraction floor, free
    elif args.arm == "iso":
        configs = [c for c in iso_reach_configs(args.iso_reach, args.iso_layers) if c in clean]
    else:
        configs = [(args.z, L) for L in LAYER_LIST if (args.z, L) in clean]

    done = _completed(args.out)
    for z_mm, layers in configs:
        tag = config_tag(args.grid, z_mm, layers)
        if tag in done and not args.force:
            print(f"\n=== {tag}: already in {Path(args.out).name}, skipping (--force to redo)")
            continue
        # Written per config, not per arm: a config costs ~10 min and an
        # interrupted sweep must not discard the ones that finished.
        _write(args.out, {"runs": [run_config(z_mm, layers, train_ds, val_ds, args)]})

    ranking = sorted(_all_runs(args.out), key=lambda r: -r["val_acc"])
    print("\n=== ranking (validation, reduced protocol -- not final accuracy) ===")
    for r in ranking:
        print(f"  {r['config']:>14}  val {r['val_acc']:.4f}  "
              f"reach {r['reach_px_total']:>6.1f} px  {r['n_params']:>6} params")


if __name__ == "__main__":
    main()
