function c = mesh_coupling_matrix(sched, alpha, decay_um, pitch_um)
%MESH_COUPLING_MATRIX Distance-dependent thermal coupling between MZI heaters.
%   C = ERR.MESH_COUPLING_MATRIX(SCHED, ALPHA, DECAY_UM, PITCH_UM) returns the
%   nMzi-by-nMzi matrix C(i,j) = ALPHA * exp(-d_ij / DECAY_UM), zero on the
%   diagonal, where d_ij is the layout distance in microns between MZI i and MZI j.
%   SCHED comes from meshmodel.schedule; PITCH_UM is [columnPitch modePitch] in
%   microns, converting SCHED.pos from (column, mode) units to a real geometry.
%
%   This is the coupling matrix CLAUDE.md's Phase-4 spec actually asks for. The
%   D2NN could only approximate thermal crosstalk with a shift-invariant Gaussian
%   blur kernel (err.thermal_crosstalk), because a full coupling matrix over a 2-D
%   mask would be N^2-by-N^2 -- 268 million entries at 128^2. A mesh has 630
%   heaters, so the honest form is affordable here: 630-by-630, built once.
%
%   ALPHA is the coupling coefficient extrapolated to zero separation, so the
%   nearest-neighbour figure is ALPHA * exp(-d_nn / DECAY_UM) rather than ALPHA
%   itself. Both ALPHA and DECAY_UM must trace to published thermo-optic
%   crosstalk measurements.  % UNSOURCED
%
%   Modelling choice, recorded rather than buried: the exponential kernel is the
%   standard lumped form for in-plane heat spreading in a thin device layer, not a
%   solution of the heat equation for this geometry. The decay length is doing all
%   the work and it is not sourced yet.
    if nargin < 4 || isempty(pitch_um), pitch_um = [1 1]; end

    xy = sched.pos .* pitch_um(:).';          % nMzi-by-2, microns
    dx = xy(:, 1) - xy(:, 1).';
    dy = xy(:, 2) - xy(:, 2).';
    d = hypot(dx, dy);

    c = alpha * exp(-d / decay_um);
    c(1:size(c, 1) + 1:end) = 0;              % a heater does not couple to itself
end
