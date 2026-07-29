"""Training loop, dataset loading, and input encoding.

One classification task, kept simple (MNIST) and reused across all phases so
results stay comparable (CLAUDE.md scope). Training is in-silico only; the
trained parameters are then exported to the as-built model -- there is no
hardware-in-the-loop training.

Random seeds are fixed and recorded for every run (returned in the history dict).

MNIST is fetched from the public S3 mirror with the standard library only (no
torchvision dependency) and cached under ``<repo>/data/mnist``.
"""
from __future__ import annotations

import gzip
import math
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

_MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_MNIST_FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}
_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "mnist"


@dataclass
class Dataset:
    """A loaded classification split: images in ``[0, 1]`` and integer labels."""

    images: np.ndarray  # (N, 28, 28) float32 in [0, 1]
    labels: np.ndarray  # (N,) int64
    name: str
    split: str

    def __len__(self) -> int:
        return int(self.images.shape[0])


# -- dataset loading -----------------------------------------------------------
def _download(fname: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
    req = urllib.request.Request(_MNIST_MIRROR + fname, headers={"User-Agent": "photonn/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
        f.write(r.read())


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        buf = f.read(num * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(num, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _, num = struct.unpack(">II", f.read(8))
        buf = f.read(num)
    return np.frombuffer(buf, dtype=np.uint8)


def load_dataset(name: str = "mnist", subset: int = None, *, split: str = "train",
                 data_dir=None, subset_seed: int = 0) -> Dataset:
    """Load the (single) classification dataset, optionally subsampled.

    ``split`` is ``"train"`` or ``"test"``. ``subset``, if given, draws that many
    samples without replacement using ``subset_seed`` (recorded for
    reproducibility). Files are downloaded to ``data_dir`` (default
    ``<repo>/data/mnist``) on first use and cached thereafter.
    """
    if name != "mnist":
        raise ValueError(f"only 'mnist' is supported (one task, reused across phases); got {name!r}.")
    if split not in _MNIST_FILES:
        raise ValueError(f"split must be 'train' or 'test'; got {split!r}.")

    root = Path(data_dir) if data_dir is not None else _DATA_ROOT
    img_file, lbl_file = _MNIST_FILES[split]
    _download(img_file, root / img_file)
    _download(lbl_file, root / lbl_file)

    images = _read_idx_images(root / img_file).astype(np.float32) / 255.0
    labels = _read_idx_labels(root / lbl_file).astype(np.int64)

    if subset is not None and subset < len(labels):
        rng = np.random.default_rng(subset_seed)
        idx = rng.choice(len(labels), size=subset, replace=False)
        images, labels = images[idx], labels[idx]

    return Dataset(images=images, labels=labels, name=name, split=split)


# -- input encoding ------------------------------------------------------------
def _embed(images, n, input_frac, device):
    """Resize each image into the central ``input_frac`` window of an n x n grid.

    Returns ``(canvas, window)``: ``canvas`` is the raw embedded image map in
    ``[0, 1]`` (no complex encoding, no normalisation); ``window`` is the 0/1
    aperture mask of the occupied region.
    """
    x = torch.as_tensor(images, dtype=torch.float32)
    if x.ndim == 2:
        x = x[None]
    x = x.clamp(0.0, 1.0)
    if device is not None:
        x = x.to(device)

    win = max(1, int(round(input_frac * n)))
    off = (n - win) // 2
    resized = F.interpolate(x[:, None], size=(win, win), mode="bilinear", align_corners=False)[:, 0]

    canvas = torch.zeros(x.shape[0], n, n, device=x.device)
    canvas[:, off:off + win, off:off + win] = resized
    window = torch.zeros(n, n, device=x.device)
    window[off:off + win, off:off + win] = 1.0
    return canvas, window


def embed_input(images, *, n: int = 128, input_frac: float = 0.5, device=None) -> torch.Tensor:
    """Raw embedded input map (``canvas`` in ``[0, 1]``), before complex encoding.

    This is what the handoff stores as the frozen "encoded input": the real
    input-plane transmission the as-built (MATLAB) model re-encodes from. For
    ``scheme="both"`` the complex field is recovered as
    ``E = canvas * exp(i * phase_scale * canvas)`` -- the phase uses the raw
    ``canvas`` values, which is why the raw map (not the unit-normalised one) is
    the thing that must be exported.
    """
    canvas, _ = _embed(images, n, input_frac, device)
    return canvas


def encode_input(images, scheme: str = "both", *, n: int = 128, input_frac: float = 0.5,
                 phase_scale: float = math.pi, dtype: torch.dtype = torch.complex64,
                 device=None) -> torch.Tensor:
    """Encode input images into the optical field ``(B, n, n)`` complex.

    ``scheme`` is one of ``amplitude``, ``phase``, or ``both``. Phase 2 uses
    **both** deliberately: the image modulates both the amplitude and the phase of
    the entrance field, giving the network the richest linear handle on the input.

    Each image is bilinearly resized into a central window of side
    ``input_frac * n`` (leaving a margin for diffraction to spread into) and:

    - ``amplitude`` : ``|E| = image`` in the window, phase 0. A real SLM would
      realise this with an amplitude modulator.
    - ``phase``     : ``|E| = 1`` in the window (plane-wave illumination through an
      aperture), ``arg(E) = phase_scale * image``.
    - ``both``      : ``|E| = image`` and ``arg(E) = phase_scale * image``.

    Physical caveat (documented in ``docs/phase2_dnn.md``): independent amplitude
    **and** phase modulation is full complex modulation, which a single SLM cannot
    do -- it needs two cascaded modulators or a complex-modulation scheme. We use
    it because in-silico we can; the as-built model can revisit that cost.

    Each returned field is normalised to unit total intensity (``sum |E|^2 = 1``).
    The classifier is invariant to this scale (it normalises by total output
    power), and it fixes a clean reference for the photon budget.
    """
    canvas, window = _embed(images, n, input_frac, device)
    b = canvas.shape[0]

    scheme = scheme.lower()
    if scheme == "amplitude":
        amp, phase = canvas, torch.zeros_like(canvas)
    elif scheme == "phase":
        amp, phase = window.unsqueeze(0).expand(b, -1, -1), canvas * phase_scale
    elif scheme == "both":
        amp, phase = canvas, canvas * phase_scale
    else:
        raise ValueError(f"scheme must be 'amplitude', 'phase', or 'both'; got {scheme!r}.")

    field = torch.polar(amp.contiguous(), phase.contiguous())
    norm = field.abs().pow(2).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12).sqrt()
    return (field / norm).to(dtype)


def encode_modes(images, *, n_modes: int = 36, device=None) -> torch.Tensor:
    """Encode images into ``n_modes`` optical amplitudes for an MZI mesh.

    Downsamples each 28x28 image to a ``g x g`` grid (``g = sqrt(n_modes)``),
    flattens, and normalises to unit L2 -- a real amplitude vector carried as
    complex ``(batch, n_modes)``. This is the mesh counterpart to
    :func:`encode_input`; the mesh's footprint (``n_modes*(n_modes-1)/2`` MZIs) is
    why the image must be reduced to a modest number of modes.
    """
    g = int(round(math.sqrt(n_modes)))
    if g * g != n_modes:
        raise ValueError(f"n_modes must be a perfect square for image encoding; got {n_modes}.")
    x = torch.as_tensor(images, dtype=torch.float32)
    if x.ndim == 2:
        x = x[None]
    x = x.clamp(0.0, 1.0)
    if device is not None:
        x = x.to(device)
    ds = F.interpolate(x[:, None], size=(g, g), mode="bilinear", align_corners=False)[:, 0]
    v = ds.reshape(ds.shape[0], -1)
    v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return v.to(torch.complex64)


def _encode(imgs, model, scheme, device, encoder):
    """Encode a batch: use ``encoder`` if given (mesh), else the D²NN field encoder."""
    if encoder is not None:
        return encoder(imgs)
    return encode_input(imgs, scheme=scheme, n=model.n, device=device)


# -- training ------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, dataset: Dataset, *, scheme: str = "both", batch_size: int = 128,
             device: str = "cpu", encoder=None) -> float:
    """Top-1 accuracy of ``model`` on ``dataset``."""
    model.eval()
    images = torch.as_tensor(dataset.images, dtype=torch.float32)
    labels = torch.as_tensor(dataset.labels, dtype=torch.long)
    correct = 0
    for i in range(0, len(dataset), batch_size):
        imgs = images[i:i + batch_size].to(device)
        y = labels[i:i + batch_size].to(device)
        inp = _encode(imgs, model, scheme, device, encoder)
        correct += (model(inp).argmax(dim=1) == y).sum().item()
    return correct / len(dataset)


def train(model, dataset: Dataset, *, epochs: int, seed: int, lr: float = 1e-2,
          batch_size: int = 64, scheme: str = "both", val_dataset: Dataset = None,
          device: str = "cpu", log: bool = True, encoder=None):
    """Train ``model`` on ``dataset``; return ``(model, history)``.

    The ``seed`` is fixed (torch + numpy) and recorded in ``history`` alongside
    every hyperparameter, so a run is fully reproducible. Loss is cross-entropy
    over the detector-region logits; the optimiser is Adam.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    images = torch.as_tensor(dataset.images, dtype=torch.float32)
    labels = torch.as_tensor(dataset.labels, dtype=torch.long)
    n_samples = len(dataset)

    history = {
        "seed": seed, "lr": lr, "epochs": epochs, "batch_size": batch_size,
        "scheme": scheme, "n_samples": n_samples,
        "loss": [], "train_acc": [], "val_acc": [],
    }

    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n_samples)
        running_loss, correct = 0.0, 0
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            imgs = images[idx].to(device)
            y = labels[idx].to(device)
            inp = _encode(imgs, model, scheme, device, encoder)

            logits = model(inp)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item() * len(idx)
            correct += (logits.argmax(dim=1) == y).sum().item()

        history["loss"].append(running_loss / n_samples)
        history["train_acc"].append(correct / n_samples)
        if val_dataset is not None:
            history["val_acc"].append(
                evaluate(model, val_dataset, scheme=scheme, device=device, encoder=encoder)
            )
        if log:
            msg = f"epoch {ep + 1:2d}/{epochs}  loss={history['loss'][-1]:.4f}  train_acc={history['train_acc'][-1]:.3f}"
            if val_dataset is not None:
                msg += f"  val_acc={history['val_acc'][-1]:.3f}"
            print(msg, flush=True)

    return model, history
