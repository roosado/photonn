function params = plane_spacing(params, sigma_m, seed)
%PLANE_SPACING Perturb each free-space gap in the stack independently.
%   PARAMS = ERR.PLANE_SPACING(PARAMS, SIGMA_M, SEED) draws an independent
%   N(0, SIGMA_M^2) deviation for every one of the L+1 gaps and writes the result
%   to PARAMS.separations_m, which model.evaluate propagates with in place of the
%   nominal geometry. Requires PARAMS.separations_m to hold the nominal gaps (the
%   MC driver seeds it, the way it seeds wavelength_m for dispersion).
%
%   This is the first *geometry* error in the budget. Every other source here is
%   a device error -- something wrong inside a component. This one is an assembly
%   error: nobody positions five plates with the gaps exactly 3.000 mm.
%
%   The gaps are drawn independently rather than as one common offset, because
%   each plate is mounted separately. The consequence is that total path length
%   errs as sqrt(L+1) x SIGMA_M while each gap errs by SIGMA_M, so a deeper stack
%   built to the same per-gap tolerance holds a *worse* total -- which is the
%   prediction the depth-vs-tolerance question turns on.
%
%   The last gap is the one from the final plate to the detector, so an axial
%   detector misplacement is already in here and is deliberately not modelled
%   again in err.detector_offset (which is lateral only).
%
%   SIGMA_M is a caller-supplied magnitude; realistic values and their sources are
%   in docs/parameter_sources.md.
    if ~isfield(params, 'separations_m') || isempty(params.separations_m)
        error("err:plane_spacing:noSeparations", ...
            "params.separations_m must be set to the nominal gaps before calling.");
    end

    s = RandStream('twister', 'Seed', seed);
    sep = params.separations_m(:);
    sep = sep + sigma_m * randn(s, size(sep));

    % A gap cannot be negative, and a zero gap is not a stack. Clamping instead of
    % erroring keeps a too-wide sweep point meaningful rather than aborting the
    % run; at every magnitude swept here the clamp never fires.
    params.separations_m = max(sep, 0);
end
