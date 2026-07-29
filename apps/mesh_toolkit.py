"""Phase-3 deliverable: mesh programming toolkit.

Verifies the Clements/Reck decompositions by reconstruction and renders the
rectangular mesh topology with per-MZI phase settings. If a trained mesh
(``exports/mesh_phase3.pt``) is present its learned phases are shown; otherwise a
fresh random mesh illustrates the topology.

    python -m apps.mesh_toolkit
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from scipy.stats import unitary_group

from photonn import mzi
from photonn.layers import MZIMeshLayer
from photonn.models import MeshNetwork

_REPO = Path(__file__).resolve().parent.parent


def verify_decompositions():
    print("Decomposition verification (reconstruction error):")
    rng = np.random.default_rng(0)
    for n in (4, 6, 8, 16):
        u = unitary_group.rvs(n, random_state=rng)
        ec = np.max(np.abs(mzi.reconstruct(mzi.clements_decompose(u)) - u))
        er = np.max(np.abs(mzi.reconstruct(mzi.reck_decompose(u)) - u))
        m = mzi.mzi_matrix(0.7, 1.3)
        print(f"  n={n:2d}: clements {ec:.1e}  reck {er:.1e}  ({n*(n-1)//2} MZIs each)  "
              f"| mzi unitary={mzi.check_unitary(m)}")


def render_topology(layer: MZIMeshLayer, ax, title: str):
    """Draw the rectangular Clements mesh: waveguides + MZIs coloured by theta."""
    n = layer.n
    theta = layer.theta.detach().numpy()
    cmap = plt.cm.viridis
    norm = plt.Normalize(0.0, np.pi)
    for m in range(n):                                   # waveguides
        ax.plot([-0.6, n - 0.4], [m, m], color="0.85", lw=0.8, zorder=0)
    for layer_idx, pairs in enumerate(layer._schedule):
        for mode, idx in pairs:                          # one box per MZI
            ax.add_patch(Rectangle((layer_idx - 0.32, mode - 0.12), 0.64, 1.24,
                                   facecolor=cmap(norm(theta[idx] % np.pi)),
                                   edgecolor="k", lw=0.3, zorder=2))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("mesh layer"); ax.set_ylabel("mode")
    ax.set_xlim(-1, n); ax.set_ylim(-1, n); ax.invert_yaxis()
    ax.set_aspect("equal")
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def load_or_make(ckpt_path: Path):
    """Return (v_mesh, u_mesh, sigma, label) from a trained ckpt or a fresh mesh."""
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        a = ck["args"]
        net = MeshNetwork(a["modes"], a["classes"], use_svd=True)
        net.load_state_dict(ck["state_dict"])
        return net.v, net.u, net.sigma.detach().numpy(), f"trained {a['modes']}-mode mesh"
    torch.manual_seed(0)
    net = MeshNetwork(36, 10, use_svd=True)
    return net.v, net.u, net.sigma.detach().numpy(), "fresh 36-mode mesh (untrained)"


def main():
    verify_decompositions()

    v, u, sigma, label = load_or_make(_REPO / "exports" / "mesh_phase3.pt")
    fig = plt.figure(figsize=(14, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.6])
    try:
        fig.set_facecolor("white")
    except Exception:
        pass

    ax_v = fig.add_subplot(gs[0, 0]); sm = render_topology(v, ax_v, "V mesh (per-MZI theta)")
    ax_u = fig.add_subplot(gs[0, 1]); render_topology(u, ax_u, "U mesh (per-MZI theta)")
    fig.colorbar(sm, ax=[ax_v, ax_u], shrink=0.7, label="theta (rad, mod pi)")

    ax_s = fig.add_subplot(gs[0, 2])
    ax_s.bar(range(len(sigma)), sigma, color="#4c78a8")
    ax_s.axhline(1.0, color="crimson", lw=1, ls="--", label="passivity limit")
    ax_s.set_title("Sigma (singular values)", fontsize=10)
    ax_s.set_xlabel("mode"); ax_s.set_ylabel("transmission"); ax_s.legend(fontsize=8)

    fig.suptitle(f"MZI mesh topology + settings — {label}", fontsize=12)
    out = _REPO / "docs" / "figures" / "phase3_mesh_topology.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
