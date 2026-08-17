function params = phase_shifter_error(params, sigma_rad, seed, fields)
%PHASE_SHIFTER_ERROR Add Gaussian phase-setting error to every programmed phase.
%   PARAMS = ERR.PHASE_SHIFTER_ERROR(PARAMS, SIGMA_RAD, SEED) perturbs every
%   programmed phase by an independent N(0, SIGMA_RAD^2) draw. For a D2NN that is
%   each phase-mask pixel (per-pixel setting error of the fabricated / addressed
%   phase plate); for an MZI mesh it is each thermo-optic shifter -- theta, phi and
%   the output phase screen. SEED gives a reproducible draw (recorded by the MC
%   driver).
%
%   PARAMS = ERR.PHASE_SHIFTER_ERROR(PARAMS, SIGMA_RAD, SEED, FIELDS) perturbs only
%   the named fields, e.g. {'theta'} or {'phi'}. theta sets an MZI's splitting and
%   phi only shifts a phase, so which of them binds is a physically different
%   question from the aggregate -- the mesh's version of the D2NN's "spatial phase
%   fidelity is the whole game" finding.
%
%   SIGMA_RAD is a caller-supplied magnitude; realistic values and their sources
%   are in docs/parameter_sources.md.
%
%   One RandStream is drawn from in field order, so a given SEED gives the same
%   perturbation to a given field regardless of what else is in PARAMS.
    if nargin < 4 || isempty(fields)
        fields = phase_fields(params);
    elseif ischar(fields) || isstring(fields)
        fields = cellstr(fields);
    end

    s = RandStream('twister', 'Seed', seed);
    for i = 1:numel(fields)
        f = fields{i};
        if ~isfield(params, f)
            error("err:phase_shifter_error:noField", ...
                "PARAMS has no field '%s' to perturb.", f);
        end
        params.(f) = params.(f) + sigma_rad * randn(s, size(params.(f)));
    end
end
