function params = coupler_imbalance(params, epsilon, seed)
%COUPLER_IMBALANCE Model directional-coupler deviation from a 50:50 split.
%   PARAMS = ERR.COUPLER_IMBALANCE(PARAMS, EPSILON, SEED) perturbs each coupler
%   power split to 0.5 +/- EPSILON, altering the effective MZI transfer matrix.
%   SEED is recorded with the realization.
%
%   EPSILON must trace to a published coupler-fabrication measurement.  % UNSOURCED
    error("err:coupler_imbalance:notImplemented", ...
        "Phase 4: coupler imbalance model not yet implemented.");
end
