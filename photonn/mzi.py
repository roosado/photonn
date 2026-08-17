"""Mach-Zehnder interferometer meshes.

Builds a 2x2 MZI transfer matrix from directional-coupler and phase-shifter
primitives, decomposes an arbitrary unitary into a mesh of MZIs (Clements or
Reck), and runs the forward pass. Unitarity is verified as a runtime invariant,
and every function here gets a corresponding analytic test in
:mod:`photonn.validate` (CLAUDE.md conventions).

Pure and NumPy-native; the differentiable mesh lives in :mod:`photonn.layers`.

References
----------
Reck, Zeilinger, Bernstein & Bertani, "Experimental realization of any discrete
unitary operator," Phys. Rev. Lett. 73:58 (1994) -- triangular mesh. Clements,
Humphreys, Metcalf, Kolthammer & Walmsley, "Optimal design for universal
multiport interferometers," Optica 3(12):1460 (2016) -- rectangular mesh and the
nulling algorithm used here.
"""
from __future__ import annotations

import numpy as np


def beamsplitter(split_ratio: float = 0.5) -> np.ndarray:
    """Return the 2x2 unitary for an ideal directional coupler.

    ``split_ratio`` is the power fraction crossed to the other port (0.5 = 50:50).
    The symmetric coupler ``[[cos k, i sin k], [i sin k, cos k]]`` with
    ``sin^2 k = split_ratio`` (a lossless reciprocal 2-port).
    """
    k = np.arcsin(np.sqrt(split_ratio))
    c, s = np.cos(k), np.sin(k)
    return np.array([[c, 1j * s], [1j * s, c]], dtype=complex)


def phase_shifter(phi: float) -> np.ndarray:
    """Return the 2x2 diagonal matrix applying phase ``phi`` to the top arm."""
    return np.array([[np.exp(1j * phi), 0.0], [0.0, 1.0]], dtype=complex)


def mzi_matrix(theta: float, phi: float) -> np.ndarray:
    """Return the 2x2 MZI transfer matrix built from couplers and phase shifters.

    Two 50:50 couplers enclosing the internal phase ``theta``, preceded by the
    external phase ``phi``: ``B @ P(theta) @ B @ P(phi)``. This evaluates to the
    Clements (2016) convention
    ``i e^{i theta/2} [[e^{i phi} sin(t), cos(t)], [e^{i phi} cos(t), -sin(t)]]``
    with ``t = theta/2``. ``theta`` sets the splitting, ``phi`` the external phase.
    Always unitary; see :func:`check_unitary`.
    """
    b = beamsplitter(0.5)
    return b @ phase_shifter(theta) @ b @ phase_shifter(phi)


def _embed(t2: np.ndarray, m: int, n: int) -> np.ndarray:
    """Embed a 2x2 block ``t2`` acting on modes ``(m, m+1)`` into an ``n x n`` identity."""
    full = np.eye(n, dtype=complex)
    full[m:m + 2, m:m + 2] = t2
    return full


def _right_null(u, i, j):
    """(theta, phi) so that right-multiplying by T^dagger nulls ``u[i, j]``.

    Mixes columns ``(j, j+1)``; ``phi`` is effective for T^dagger (Clements 2016).
    """
    a, b = u[i, j], u[i, j + 1]
    theta = 2.0 * np.arctan2(abs(b), abs(a))
    phi = -np.angle(-b * np.conj(a))
    return theta, phi


def _left_null(u, i, j):
    """(theta, phi) so that left-multiplying by T nulls the bottom element ``u[i, j]``.

    Mixes rows ``(i-1, i)``.
    """
    a, b = u[i - 1, j], u[i, j]
    theta = 2.0 * np.arctan2(abs(a), abs(b))
    phi = np.angle(b * np.conj(a))
    return theta, phi


def clements_decompose(unitary):
    """Decompose an ``N x N`` unitary into a rectangular Clements mesh.

    Nulls sub-diagonal elements alternately from the right (T^dagger on columns)
    and the left (T on rows), per Clements et al. (2016), leaving a residual
    diagonal phase screen. Returns ``settings`` (``n``, ordered ``ops`` list of
    ``(side, mode, theta, phi)``, and ``diag``); :func:`reconstruct` /
    :func:`mesh_forward` rebuild the operator, verified against the input.
    """
    u = np.array(unitary, dtype=complex).copy()
    n = u.shape[0]
    ops = []
    for i in range(1, n):
        if i % 2 == 1:
            for k in range(i):                       # null up the anti-diagonal
                row, col = n - 1 - k, i - 1 - k
                theta, phi = _right_null(u, row, col)
                u = u @ _embed(mzi_matrix(theta, phi).conj().T, col, n)
                ops.append(("R", col, theta, phi))
        else:
            for k in range(1, i + 1):
                row, col = n + k - i - 1, k - 1
                theta, phi = _left_null(u, row, col)
                u = _embed(mzi_matrix(theta, phi), row - 1, n) @ u
                ops.append(("L", row - 1, theta, phi))
    return {"n": n, "ops": ops, "diag": np.diag(u).copy()}


def reck_decompose(unitary):
    """Decompose an ``N x N`` unitary into a triangular Reck mesh.

    Nulls the lower triangle by right-multiplication (T^dagger on columns), per
    Reck et al. (1994). Same ``settings`` format as :func:`clements_decompose`.
    """
    u = np.array(unitary, dtype=complex).copy()
    n = u.shape[0]
    ops = []
    for i in range(n - 1, 0, -1):
        for j in range(i):
            theta, phi = _right_null(u, i, j)
            u = u @ _embed(mzi_matrix(theta, phi).conj().T, j, n)
            ops.append(("R", j, theta, phi))
    return {"n": n, "ops": ops, "diag": np.diag(u).copy()}


def reconstruct(settings) -> np.ndarray:
    """Rebuild the ``N x N`` operator from a decomposition's ``settings``.

    Replays the recorded nulling operations in reverse, inverting each (a right
    op ``U @ T^dagger`` inverts to ``@ T``; a left op ``T @ U`` to ``T^dagger @``),
    starting from the diagonal screen. Exact to numerical precision when the
    settings came from :func:`clements_decompose` / :func:`reck_decompose`.
    """
    n = settings["n"]
    m = np.diag(settings["diag"]).astype(complex)
    for side, mode, theta, phi in reversed(settings["ops"]):
        t = mzi_matrix(theta, phi)
        if side == "R":
            m = m @ _embed(t, mode, n)
        else:
            m = _embed(t.conj().T, mode, n) @ m
    return m


def mesh_forward(settings, x):
    """Run input vector(s) ``x`` through a mesh described by ``settings``.

    ``x`` is an ``(N,)`` or ``(N, batch)`` complex array; returns the mesh output.
    Equivalent to ``reconstruct(settings) @ x``.
    """
    return reconstruct(settings) @ np.asarray(x, dtype=complex)


def check_unitary(matrix, atol: float = 1e-10) -> bool:
    """Return True if ``matrix`` is unitary to tolerance ``atol``.

    Checks ``U^dagger U == I``. Used as a runtime invariant after matrix
    construction and decomposition; :func:`photonn.validate.assert_unitary` is the
    raising wrapper.
    """
    m = np.asarray(matrix)
    n = m.shape[0]
    return np.allclose(m.conj().T @ m, np.eye(n), atol=atol)


def svd_decompose(matrix):
    """Decompose an arbitrary (square) matrix as ``U * Sigma * V^dagger``.

    ``M = U diag(s) V^dagger`` (numpy SVD); ``U`` and ``V`` are unitary, so each is
    realised as a Clements mesh, and ``Sigma`` as a bank of controllable
    attenuators. Returns ``{"u": settings, "s": singular values, "v": settings}``;
    :func:`svd_reconstruct` rebuilds ``M``.

    Passivity note: a lossless mesh cannot amplify, so a *physical* realisation
    needs ``s <= 1`` -- scale ``M`` by ``1 / max(s)`` and track the gain
    externally. The decomposition itself does not enforce this (the reconstruction
    test uses an arbitrary real matrix with ``s`` possibly ``> 1``).
    """
    m = np.asarray(matrix, dtype=complex)
    u, s, vh = np.linalg.svd(m)
    return {"u": clements_decompose(u), "s": s, "v": clements_decompose(vh.conj().T)}


def passivize(sigma, out_phase_v):
    """Rewrite a trained ``Sigma`` as a passive one, without changing what it computes.

    A trained :class:`~photonn.models.MeshNetwork` leaves ``Sigma`` unconstrained, so
    it comes out signed and larger than 1 (the 36-mode mesh runs -0.041 .. 3.907, nine
    values above unity). A lossless mesh cannot amplify, so that is not a device --
    and asking what per-MZI loss costs a model that already contains free gain is not
    a question with an answer. Two rewrites fix it exactly:

    * **Sign.** ``Sigma`` multiplies the V mesh's output element-wise, and V's output
      phase screen is applied immediately before it, so ``sigma_k < 0`` is absorbed as
      ``out_phase_v[k] += pi``. Identical field, everywhere.
    * **Scale.** ``sigma -> sigma / max|sigma|`` is one real scale on the field
      entering U, hence one scale on every output intensity -- which cancels in the
      ``region / total`` readout. Identical logits, not merely identical predictions.

    Returns ``(sigma_p, out_phase_v_p, gain)`` with ``0 <= sigma_p <= 1``, where
    ``gain = max|sigma|`` is the external factor a real device would have to supply.
    Pure representation change, so it lives here rather than in the MATLAB error
    model (CLAUDE.md boundary).
    """
    sigma = np.asarray(sigma, dtype=float)
    out_phase_v = np.asarray(out_phase_v, dtype=float)
    if sigma.shape != out_phase_v.shape:
        raise ValueError(
            f"sigma {sigma.shape} and out_phase_v {out_phase_v.shape} must have the same shape."
        )
    gain = float(np.max(np.abs(sigma)))
    if gain == 0.0:
        raise ValueError("sigma is identically zero; there is nothing to normalise.")
    sigma_p = np.abs(sigma) / gain
    out_phase_p = out_phase_v + np.where(sigma < 0.0, np.pi, 0.0)
    return sigma_p, out_phase_p, gain


def svd_reconstruct(svd_settings) -> np.ndarray:
    """Rebuild ``M = U * Sigma * V^dagger`` from :func:`svd_decompose` output."""
    u = reconstruct(svd_settings["u"])
    v = reconstruct(svd_settings["v"])
    return u @ np.diag(svd_settings["s"]).astype(complex) @ v.conj().T
