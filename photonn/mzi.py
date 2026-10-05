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
    settings = {"n": n, "ops": ops, "diag": np.diag(u).copy()}
    # Once, on the finished decomposition -- the runtime unitarity check
    # CLAUDE.md asks for. Not per MZI: this loop does n(n-1)/2 embeds, and a
    # per-op check would make it quadratically slower to assert something only
    # the whole product can be wrong about.
    assert_decomposition_unitary(settings)
    return settings


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


def assert_decomposition_unitary(settings, atol: float = 1e-8) -> None:
    """Check that a decomposition rebuilds a unitary operator.

    Called once by :func:`clements_decompose` on its own result, not per MZI:
    the decomposition performs ``n(n-1)/2`` embeds and checking each would make
    it quadratically slower to assert something only the whole product can be
    wrong about.

    Honours :data:`photonn.validate.STRICT`, like the propagators.
    """
    from photonn import validate

    if validate.STRICT:
        validate.assert_unitary(reconstruct(settings), atol=atol)


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


# -- Phase 5: the electro-optic activation ----------------------------------------
#
# Williamson, Hughes, Minkov, Bartlett, Pai & Fan, "Reprogrammable electro-optic
# nonlinear activation functions for optical neural networks," IEEE JSTQE 26(1):
# 7700412 (2020), doi:10.1109/JSTQE.2019.2930455 (arXiv:1903.04579). A tap coupler
# sends a fraction alpha of a mode's power to a photodiode; the amplified
# photocurrent drives the internal phase of an MZI that the remaining light crosses.
# It is this module's own MZI with a phase set by the light's own power, which is
# why it lives here rather than in a materials module the project does not have.

def eo_phase_gain(*, alpha: float, tia_gain_ohm: float, responsivity_a_per_w: float,
                  v_pi: float) -> float:
    """Phase written on the modulator per watt of light entering the activation.

    Williamson et al. Eq. (7): ``g_phi = pi * alpha * G * R / V_pi`` (rad/W).
    ``alpha`` is the tapped power fraction, ``G`` the transimpedance gain (ohm),
    ``R`` the photodiode responsivity (A/W) and ``V_pi`` the modulator's half-wave
    voltage. Physical constants enter the activation only through this function and
    :func:`eo_bias_phase`, so each is cited where it is passed.
    """
    return float(np.pi * alpha * tia_gain_ohm * responsivity_a_per_w / v_pi)


def eo_bias_phase(*, v_bias: float, v_pi: float) -> float:
    """Static phase from the bias voltage: Williamson et al. Eq. (5), ``pi V_b / V_pi``."""
    return float(np.pi * v_bias / v_pi)


def eo_activation(z, *, alpha: float, g_phi: float, phi_b: float) -> np.ndarray:
    """Williamson et al. (2020) Eq. (6), element-wise on complex mode amplitudes.

    ``f(z) = j sqrt(1-alpha) exp(-j[g|z|^2 + phi_b]/2) cos([g|z|^2 + phi_b]/2) z``,
    with ``|z|^2`` the mode's power in the units ``g_phi`` is quoted per (watts for
    a physical device, so a field in sqrt(W)).

    **Sign convention.** The paper writes time as ``e^{-j omega t}`` and this
    project's MZI as ``e^{+i ...}``; the two agree with ``f(z) = sqrt(1-alpha) *
    [B P(theta) B]_{10} z`` built from :func:`beamsplitter` and :func:`phase_shifter`
    at ``theta = -(phi_b + g|z|^2)``, element (1, 0) being the cross port: light
    enters the top arm and leaves the bottom one.
    :func:`photonn.validate.eo_activation_reference` builds exactly that, and the
    tests hold the two together to ~1e-15.

    **Where it operates.** Transmission ``|f|^2/|z|^2 = (1-alpha) cos^2(...)``. At
    ``phi_b = pi`` and ``g|z|^2 << 1`` it is a pure cubic,
    ``f ~ -sqrt(1-alpha) (g/2) |z|^2 z``, with no linear term: weak light is not
    passed weakly, it is passed as its own cube.

    Passivity is asserted at runtime (CLAUDE.md "invariants as runtime
    assertions"): an activation that amplifies is a bug, and the only way to get
    one here is an ``alpha`` outside ``[0, 1)``.
    """
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"alpha is a tapped power fraction in [0, 1); got {alpha!r}.")
    z = np.asarray(z, dtype=complex)
    power = np.abs(z) ** 2
    half = 0.5 * (g_phi * power + phi_b)
    out = 1j * np.sqrt(1.0 - alpha) * np.cos(half) * np.exp(-1j * half) * z
    assert_passive(z, out, alpha)
    return out


def clements_schedule(n_modes: int):
    """The rectangular brick the trained meshes use: ``[[(top_mode, mzi_index), ...], ...]``.

    Column ``c`` couples the pairs starting at mode ``c % 2``; MZIs are numbered in
    column order. The same schedule as :class:`photonn.layers.MZIMeshLayer` and
    ``photonn-hw/+meshmodel/schedule.m`` (a test holds the first two together).
    """
    columns, idx = [], 0
    for c in range(n_modes):
        col = []
        for m in range(c % 2, n_modes - 1, 2):
            col.append((m, idx))
            idx += 1
        columns.append(col)
    return columns


def brick_matrix(theta, phi, out_phase) -> np.ndarray:
    """One trained mesh's operator from its stored angles, column by column, in NumPy.

    ``diag(exp(i out_phase)) L_n ... L_1`` with each column built from
    :func:`mzi_matrix`. Mirrors :meth:`photonn.layers.MZIMeshLayer.matrix`.
    """
    out_phase = np.asarray(out_phase, dtype=float)
    n = out_phase.size
    m = np.eye(n, dtype=complex)
    for column in clements_schedule(n):
        layer = np.eye(n, dtype=complex)
        for top, idx in column:
            layer[top:top + 2, top:top + 2] = mzi_matrix(theta[idx], phi[idx])
        m = layer @ m
    return np.diag(np.exp(1j * out_phase)) @ m


def deep_mesh_forward(params: dict, x_unit, *, input_power_w: float, alpha: float,
                      g_phi: float, phi_b: float):
    """The Phase-5 deep mesh in float64, in physical units, from handoff-shaped arrays.

    ``params`` holds ``phase_theta``/``phase_phi`` ``[L, 2 n_mzi]`` (each row
    ``[V, U]``), ``sigma`` ``[L, n]`` and ``out_phase`` ``[L, 2, n]``, exactly as a
    ``deep_mesh`` handoff stores them. ``x_unit`` is the encoder's unit-norm field,
    ``[B, n]``; it enters the chip as ``sqrt(input_power_w) * x_unit``, so every field
    here is in sqrt(W) and ``g_phi`` is in rad/W. Row-vector convention, ``U diag(s) V``
    with V as stored -- the same two conventions ``+meshmodel`` holds.

    Returns ``(out, taps)``: the output field ``[B, n]`` and, per activation bank, the
    per-mode power entering it in watts. No noise of any kind: this is the design
    model, and the reference both the MATLAB as-built model and the browser are held
    to.
    """
    theta = np.asarray(params["phase_theta"], dtype=float)
    phi = np.asarray(params["phase_phi"], dtype=float)
    sigma = np.asarray(params["sigma"], dtype=float)
    out_phase = np.asarray(params["out_phase"], dtype=float)
    n_layers, n = sigma.shape
    n_mzi = n * (n - 1) // 2
    z = np.sqrt(input_power_w) * np.asarray(x_unit, dtype=complex)
    taps = []
    for layer in range(n_layers):
        v = brick_matrix(theta[layer, :n_mzi], phi[layer, :n_mzi], out_phase[layer, 0])
        u = brick_matrix(theta[layer, n_mzi:], phi[layer, n_mzi:], out_phase[layer, 1])
        z = z @ (u @ np.diag(sigma[layer].astype(complex)) @ v).T
        if layer < n_layers - 1:
            taps.append(np.abs(z) ** 2)
            z = eo_activation(z, alpha=alpha, g_phi=g_phi, phi_b=phi_b)
    return z, taps


def assert_passive(z_in, z_out, alpha: float, rtol: float = 1e-12) -> None:
    """Raise if an activation put out more power than ``(1 - alpha)`` of what it took in.

    Element-wise. Honours :data:`photonn.validate.STRICT`, like the other runtime
    invariants.
    """
    from photonn import validate

    if not validate.STRICT:
        return
    p_in = np.abs(np.asarray(z_in)) ** 2
    p_out = np.abs(np.asarray(z_out)) ** 2
    excess = p_out - (1.0 - alpha) * p_in
    if np.any(excess > rtol * np.maximum(p_in, np.finfo(float).tiny)):
        worst = float(np.max(excess))
        raise ValueError(
            f"activation is not passive: output exceeds (1 - alpha) of input power by "
            f"up to {worst:.3e}. An electro-optic activation can only attenuate."
        )
