function params = mesh_thermal_crosstalk(params, coupling)
%MESH_THERMAL_CROSSTALK Parasitic phase from neighbouring MZI heaters.
%   PARAMS = ERR.MESH_THERMAL_CROSSTALK(PARAMS, COUPLING) adds to every thermo-
%   optic shifter the heat leaking in from every other MZI on the chip. COUPLING is
%   the nMzi-by-nMzi matrix from err.mesh_coupling_matrix; PARAMS.theta and
%   PARAMS.phi are the concatenated [V, U] phase vectors.
%
%   The model, stated so the assumptions are visible:
%     - MZI j dissipates power proportional to its total programmed phase,
%       p_j = theta_j + phi_j. A thermo-optic shifter's phase *is* its applied
%       power times a constant, so programmed phase is the natural proxy.
%     - The temperature rise that reaches MZI i is sum_j COUPLING(i,j) * p_j.
%     - theta_i and phi_i are co-located on the same MZI, so that rise adds the
%       **same** parasitic phase to both.
%
%   The two meshes are treated as thermally independent: V and U are separate
%   rectangles with the Sigma bank between them, so COUPLING is applied blockwise
%   rather than across the whole 1260. That is a layout assumption, not a
%   measurement, and it is the optimistic one -- a real chip that packs the two
%   meshes together would couple them.
%
%   The output phase screen is not perturbed here. It sits outside the rectangle
%   and its layout is not part of the Clements schedule, so it has no distance to
%   its neighbours; err.phase_shifter_error covers it.
%
%   Deterministic (no seed): crosstalk is a fixed property of the layout and the
%   programmed state, not a random draw.
    for f = {'theta', 'phi'}
        if ~isfield(params, f{1}) || isempty(params.(f{1}))
            error("err:mesh_thermal_crosstalk:noPhases", ...
                ["PARAMS has no '%s'. Mesh crosstalk needs the MZI phase vectors; " ...
                 "the D2NN's blur-kernel model is err.thermal_crosstalk."], f{1});
        end
    end

    nMzi = size(coupling, 1);
    theta = params.theta(:);
    phi = params.phi(:);
    if numel(theta) ~= 2 * nMzi
        error("err:mesh_thermal_crosstalk:sizeMismatch", ...
            "COUPLING is %d-by-%d but PARAMS holds %d phases (expected %d).", ...
            nMzi, size(coupling, 2), numel(theta), 2 * nMzi);
    end

    for m = 0:1                                   % mesh 0 = V, mesh 1 = U
        idx = m * nMzi + (1:nMzi);
        dissipated = theta(idx) + phi(idx);
        leaked = coupling * dissipated;
        theta(idx) = theta(idx) + leaked;
        phi(idx) = phi(idx) + leaked;
    end

    params.theta = reshape(theta, size(params.theta));
    params.phi = reshape(phi, size(params.phi));
end
