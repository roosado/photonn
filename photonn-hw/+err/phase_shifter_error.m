function params = phase_shifter_error(params, sigma_rad, seed)
%PHASE_SHIFTER_ERROR Add Gaussian phase-setting error to each mask phase.
%   PARAMS = ERR.PHASE_SHIFTER_ERROR(PARAMS, SIGMA_RAD, SEED) perturbs every
%   programmed phase-mask value by an independent N(0, SIGMA_RAD^2) draw,
%   modelling per-pixel phase-setting error of the fabricated / addressed phase
%   plate. SEED gives a reproducible draw (recorded by the MC driver).
%
%   SIGMA_RAD is a caller-supplied magnitude; realistic values and their sources
%   are in docs/parameter_sources.md.
    s = RandStream('twister', 'Seed', seed);
    params.phase_masks = params.phase_masks + sigma_rad * randn(s, size(params.phase_masks));
end
