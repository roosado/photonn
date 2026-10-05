"""Phase-5 gate: does an electro-optic activation make mesh depth real here?

Before the activation enters the library, this script has to reproduce the gain
Williamson, Hughes, Minkov, Bartlett, Pai & Fan report for it ("Reprogrammable
electro-optic nonlinear activation functions for optical neural networks,"
IEEE JSTQE 26(1):7700412 (2020), doi:10.1109/JSTQE.2019.2930455; arXiv:1903.04579).
A first scratch probe did not (plans/12, Step 0). This script tries the named
causes one at a time, then the paper's setup as published, and records every arm's
seed and setting so the write-up can quote them.

It passed (2026-10-04, ``docs/phase5_activation.md``): on the paper's setup the
activated pair beats the linear pair by 8.8 points; on our 6x6 input it does not
beat one mesh, so Phase 5 adopted the paper's Fourier input. The encoder has
since moved to :func:`photonn.train.encode_fourier`, unchanged. The activation
stays here as it ran -- including the trainable gain and bias of cause 4, which
the library's :class:`~photonn.layers.EOActivationLayer` deliberately lacks --
and the library's is tested against the same closed form.

Run (from the repo root, in the project venv)::

    python -m apps.eo_gate --arm ours-2svd-act    # one named arm
    python -m apps.eo_gate --arm all --seeds 20260725 20260726 20260727 --workers 11
    python -m apps.eo_gate --list                 # what the arms are

Results append to ``exports/phase5_gate/results.jsonl`` (gitignored; the numbers
that matter are restated in ``docs/phase5_activation.md``).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from photonn.layers import MZIMeshLayer
from photonn.train import encode_fourier, encode_modes, evaluate, load_dataset, train

_REPO = Path(__file__).resolve().parent.parent
_OUT = _REPO / "exports" / "phase5_gate" / "results.jsonl"


# -- the device ------------------------------------------------------------------
def eo_activation(z: torch.Tensor, alpha, g_phi, phi_b) -> torch.Tensor:
    """Williamson et al. (2020) Eq. (6), element-wise on a complex tensor.

    ``f(z) = j sqrt(1-alpha) exp(-j[g|z|^2 + phi_b]/2) cos([g|z|^2 + phi_b]/2) z``.
    ``|z|^2`` is in whatever units the field carries; here, a unit-L2 input field,
    which is the paper's own normalisation for its MNIST benchmark.
    """
    p = z.real ** 2 + z.imag ** 2
    half = 0.5 * (g_phi * p + phi_b)
    # Not torch.polar(cos(half), -half): the modulus here is *signed* (negative
    # throughout the phi_b = pi tail), and polar's backward takes the sign of the
    # result as the direction of d/d|r|, so it returns the wrong-signed gradient for
    # a negative modulus. Forward agrees to 1e-13 either way; training does not.
    # The first run of this gate sat at chance on every activated arm because of it.
    rot = torch.polar(torch.ones_like(half), -half)
    return 1j * math.sqrt(1.0 - alpha) * torch.cos(half) * rot * z


class EOActivation(nn.Module):
    """One activation per mode, all sharing (alpha, g_phi, phi_b).

    ``trainable`` makes ``g_phi`` and ``phi_b`` one learnable scalar each, per
    layer: the paper's Table III trains ``g`` per layer, and plans/12 cause 4 asks
    for the bias as well.
    """

    def __init__(self, alpha: float, g_phi: float, phi_b: float, *, trainable: bool = False):
        super().__init__()
        self.alpha = float(alpha)
        g = torch.tensor(float(g_phi))
        b = torch.tensor(float(phi_b))
        if trainable:
            self.g_phi, self.phi_b = nn.Parameter(g), nn.Parameter(b)
        else:
            self.register_buffer("g_phi", g)
            self.register_buffer("phi_b", b)

    def forward(self, z):
        return eo_activation(z, self.alpha, self.g_phi, self.phi_b)


# -- the model -----------------------------------------------------------------------
class SVDLayer(nn.Module):
    """``U diag(sigma) V`` with sigma free, exactly as MeshNetwork builds its one layer.

    ``passive`` constrains sigma to (0, 1) through a sigmoid. A free sigma ahead of
    an activation is optical *gain* -- it can lift a mode up the activation curve,
    which no passive chip can -- so plans/12 requires this wherever an activation
    follows. ``mzi.passivize``'s argument that the scale cancels holds only after
    the last activation.
    """

    def __init__(self, n, *, passive=False):
        super().__init__()
        # u before v, as MeshNetwork does, so one layer under the shipped protocol
        # and seed draws the same initial phases and reproduces 0.7355 exactly.
        self.u = MZIMeshLayer(n)
        self.v = MZIMeshLayer(n)
        self.passive = passive
        # sigmoid(4) = 0.982: starts near the free layer's sigma = 1.
        self.sigma = nn.Parameter(torch.full((n,), 4.0) if passive else torch.ones(n))

    def sigma_values(self):
        return torch.sigmoid(self.sigma) if self.passive else self.sigma

    def forward(self, x):
        return self.u(self.v(x) * self.sigma_values().to(x.dtype))


class Stack(nn.Module):
    """``layers`` meshes in series, with the activation where ``placement`` says.

    ``layer`` is ``"svd"`` (U.Sigma.V, the shipped mesh) or ``"unitary"`` (one
    Clements mesh, the paper's). ``placement``: ``"none"`` (linear), ``"between"``
    (one activation bank between consecutive meshes -- Phase 5's model), or
    ``"every"`` (after every mesh, including the last -- the paper's network).

    ``readout``:
    * ``"photonn"`` -- ``10 * I_k / sum_all I`` as logits for a softmax: the
      project's "integrate intensity, softmax", identical to MeshNetwork.
    * ``"paper"`` -- the first 10 intensities normalised by *their* sum and used
      directly as probabilities; returned as their log so cross-entropy on these
      "logits" is the paper's loss exactly (softmax(log p) = p).
    """

    def __init__(self, n, n_layers, *, layer="svd", placement="none", readout="photonn",
                 act=None, n_classes=10, readout_gain=10.0, passive=False):
        super().__init__()
        if placement == "none":
            n_act = 0
        elif placement == "between":
            n_act = n_layers - 1
        elif placement == "every":
            n_act = n_layers
        else:
            raise ValueError(placement)
        if layer == "svd":
            # Passive only where an activation follows the layer.
            self.meshes = nn.ModuleList(SVDLayer(n, passive=passive and i < n_act)
                                        for i in range(n_layers))
        else:
            self.meshes = nn.ModuleList(MZIMeshLayer(n) for _ in range(n_layers))
        act = act or {}
        self.acts = nn.ModuleList(EOActivation(**act) for _ in range(n_act))
        self.placement = placement
        self.readout = readout
        self.n_classes = n_classes
        self.readout_gain = readout_gain

    def forward(self, x, *, taps=None):
        for i, mesh in enumerate(self.meshes):
            x = mesh(x)
            if i < len(self.acts):
                if taps is not None:
                    taps.append(x.detach())
                x = self.acts[i](x)
        intensity = x.real ** 2 + x.imag ** 2
        region = intensity[:, :self.n_classes]
        if self.readout == "photonn":
            total = intensity.sum(dim=1, keepdim=True).clamp_min(1e-30)
            return region / total * self.readout_gain
        p = region / region.sum(dim=1, keepdim=True).clamp_min(1e-30)
        return torch.log(p.clamp_min(1e-30))


# -- the arms --------------------------------------------------------------------------
PAPER_ACT = {"alpha": 0.1, "g_phi": 0.05 * math.pi, "phi_b": math.pi}  # Williamson Fig. 6


@dataclass
class Protocol:
    """How an arm is trained and scored."""

    name: str
    encoder: str          # "6x6" (encode_modes, 36 real modes) or "fourier16"
    n_modes: int
    n_train: int          # 0 = all 60 000
    n_test: int           # 0 = all 10 000
    epochs: int
    batch: int
    lr: float
    readout: str


PROTOCOLS = {
    # The first probe (plans/12): 5 epochs of the shipped mesh's protocol.
    "probe": Protocol("probe", "6x6", 36, 20000, 2000, 5, 128, 2e-2, "photonn"),
    # Cause 1: the shipped mesh's own protocol (apps/train_mesh.py defaults).
    "ours": Protocol("ours", "6x6", 36, 20000, 2000, 20, 128, 2e-2, "photonn"),
    # Cause 3: the paper's input, batch and training set, our readout and lr.
    "fourier": Protocol("fourier", "fourier16", 16, 0, 0, 20, 500, 2e-2, "photonn"),
    # The paper as published, readout included (200 epochs: its Fig. 6b axis).
    "paper": Protocol("paper", "fourier16", 16, 0, 0, 200, 500, 2e-2, "paper"),
}


@dataclass
class Arm:
    name: str
    protocol: str
    n_layers: int
    layer: str
    placement: str
    trainable_act: bool = False
    passive: bool = False
    note: str = ""


ARMS = [
    # The first probe again, to show this script reproduces it before trusting it.
    Arm("probe-1svd", "probe", 1, "svd", "none"),
    Arm("probe-2svd-linear", "probe", 2, "svd", "none"),
    Arm("probe-2svd-act", "probe", 2, "svd", "between"),
    # Cause 1 -- training length, the shipped protocol, three arms.
    Arm("ours-1svd", "ours", 1, "svd", "none", note="the shipped architecture"),
    Arm("ours-2svd-linear", "ours", 2, "svd", "none"),
    Arm("ours-2svd-act", "ours", 2, "svd", "between"),
    # Cause 2 -- the paper's layer: unitary meshes, no Sigma.
    Arm("ours-2u-linear", "ours", 2, "unitary", "none"),
    Arm("ours-2u-act", "ours", 2, "unitary", "between"),
    # Cause 3 -- the paper's input (16 Fourier modes), with a 16-mode one-layer baseline.
    Arm("fourier-1svd", "fourier", 1, "svd", "none"),
    Arm("fourier-2svd-linear", "fourier", 2, "svd", "none"),
    Arm("fourier-2svd-act", "fourier", 2, "svd", "between"),
    Arm("fourier-1u", "fourier", 1, "unitary", "none"),
    Arm("fourier-2u-linear", "fourier", 2, "unitary", "none"),
    Arm("fourier-2u-act", "fourier", 2, "unitary", "between"),
    # Cause 4 -- bias and gain trainable, on our input.
    Arm("ours-2svd-act-train", "ours", 2, "svd", "between", trainable_act=True),
    # The paper as published: unitary meshes, activation after every layer, its readout.
    Arm("paper-1u-linear", "paper", 1, "unitary", "none"),
    Arm("paper-2u-linear", "paper", 2, "unitary", "none"),
    Arm("paper-1u-act", "paper", 1, "unitary", "every"),
    Arm("paper-2u-act", "paper", 2, "unitary", "every"),
    # The physical constraint, not a fifth cause: Sigma ahead of the activation passive.
    Arm("fourier-2svd-act-passive", "fourier", 2, "svd", "between", passive=True,
        note="sigma before the activation in (0, 1)"),
]
ARM_BY_NAME = {a.name: a for a in ARMS}


def _encoder(proto: Protocol):
    if proto.encoder == "6x6":
        return lambda imgs: encode_modes(imgs, n_modes=proto.n_modes)
    return lambda imgs: encode_fourier(imgs, n_modes=proto.n_modes)


def run_arm(arm: Arm, seed: int, *, epochs: int = None, log: bool = False) -> dict:
    proto = PROTOCOLS[arm.protocol]
    epochs = epochs or proto.epochs
    train_ds = load_dataset("mnist", subset=proto.n_train or None, split="train")
    test_ds = load_dataset("mnist", subset=proto.n_test or None, split="test")

    torch.manual_seed(seed)
    act = dict(PAPER_ACT, trainable=arm.trainable_act)
    model = Stack(proto.n_modes, arm.n_layers, layer=arm.layer, placement=arm.placement,
                  readout=proto.readout, act=act, passive=arm.passive)
    n_params = sum(p.numel() for p in model.parameters())
    enc = _encoder(proto)

    t0 = time.perf_counter()
    model, hist = train(model, train_ds, epochs=epochs, seed=seed, lr=proto.lr,
                        batch_size=proto.batch, encoder=enc, log=log)
    test_acc = evaluate(model, test_ds, encoder=enc, batch_size=500)
    train_acc = evaluate(model, train_ds, encoder=enc, batch_size=500)
    elapsed = time.perf_counter() - t0

    acts = [{"g_phi": float(a.g_phi), "phi_b": float(a.phi_b)} for a in model.acts]
    return {
        "arm": arm.name, "seed": seed, "test_acc": test_acc, "train_acc": train_acc,
        "params": n_params, "seconds": round(elapsed, 1),
        "protocol": asdict(proto) | {"epochs": epochs},
        "arm_def": asdict(arm), "act_final": acts,
        "loss_last": hist["loss"][-1],
        "regime": operating_regime(model, test_ds, enc),
    }


@torch.no_grad()
def operating_regime(model: Stack, dataset, encoder) -> list:
    """Where the trained activations actually operate, on the test set.

    Per activation bank: the largest sigma ahead of it (a free sigma > 1 is gain),
    the median and 99th-percentile phase ``g|z|^2`` the light writes on the
    modulator, the fraction of (image, mode) inputs past half-opening -- where
    ``cos^2((g|z|^2 + phi_b)/2)`` has reached half its swing from the dark value
    -- and the fraction of the bank's input power that leaves it.
    """
    model.eval()
    x = encoder(torch.as_tensor(dataset.images, dtype=torch.float32))
    taps = []
    model(x, taps=taps)
    out = []
    for i, (z, act) in enumerate(zip(taps, model.acts)):
        g, pb = float(act.g_phi), float(act.phi_b)
        p = (z.real ** 2 + z.imag ** 2).double()
        phase = g * p
        t = torch.cos(0.5 * (phase + pb)) ** 2
        t0 = math.cos(0.5 * pb) ** 2
        half = (t - t0).abs() >= 0.5 * max(1.0 - t0, t0)
        f = eo_activation(z.to(torch.complex128), act.alpha, g, pb)
        transmitted = float((f.abs() ** 2).sum() / p.sum())
        mesh = model.meshes[i]
        smax = float(mesh.sigma_values().abs().max()) if isinstance(mesh, SVDLayer) else 1.0
        out.append({
            "sigma_max_before": smax,
            "phase_median": float(phase.median()),
            "phase_p99": float(torch.quantile(phase.flatten()[::7], 0.99)),
            "frac_past_half": float(half.double().mean()),
            "power_transmitted": transmitted,
        })
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", nargs="+", default=[], help="arm names, or 'all'")
    p.add_argument("--seeds", type=int, nargs="+", default=[20260725])
    p.add_argument("--epochs", type=int, default=None, help="override the protocol's epochs")
    p.add_argument("--list", action="store_true")
    p.add_argument("--log", action="store_true", help="print per-epoch progress")
    p.add_argument("--out", default=str(_OUT))
    p.add_argument("--workers", type=int, default=1, help="parallel single-thread processes")
    args = p.parse_args()

    if args.list:
        for a in ARMS:
            print(f"{a.name:24s} {a.protocol:8s} L={a.n_layers} {a.layer:8s} {a.placement:8s}"
                  f"{' trainable' if a.trainable_act else ''}  {a.note}")
        return

    names = [a.name for a in ARMS] if args.arm == ["all"] else args.arm
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(ARM_BY_NAME[name], seed) for name in names for seed in args.seeds]

    def record(r):
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")
        print(f"{r['arm']:24s} seed={r['seed']} test={r['test_acc']:.4f} "
              f"train={r['train_acc']:.4f} params={r['params']} {r['seconds']:.0f}s "
              f"act={r['act_final']}", flush=True)

    # The meshes are small enough that torch's intra-op threads only add overhead
    # (77 ms/step at one thread against 83 at four, measured), so parallelism is
    # one single-threaded process per run.
    if args.workers <= 1:
        _one_thread()
        for arm, seed in jobs:
            record(run_arm(arm, seed, epochs=args.epochs, log=args.log))
        return
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(args.workers, initializer=_one_thread) as pool:
        futs = [pool.submit(run_arm, arm, seed, epochs=args.epochs) for arm, seed in jobs]
        for fut in as_completed(futs):
            record(fut.result())


def _one_thread():
    torch.set_num_threads(1)


if __name__ == "__main__":
    main()
