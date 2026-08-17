function params = coupler_imbalance(params, epsilon, seed)
%COUPLER_IMBALANCE Model directional-coupler deviation from a 50:50 split.
%   PARAMS = ERR.COUPLER_IMBALANCE(PARAMS, EPSILON, SEED) perturbs each coupler
%   power split to 0.5 + N(0, EPSILON^2), altering the effective MZI transfer
%   matrix. Two couplers per MZI, so PARAMS.splits is (2*n_mzi)-by-2 and every one
%   of them is drawn independently. SEED is recorded with the realization.
%
%   This is the source the D2NN budget could not model at all: a phase mask has no
%   coupler, so the function was a stub through Phase 4 (docs/tolerance_d2nn.md,
%   Caveats). It is the most architecture-specific error the mesh has.
%
%   Why it bites where phase error does not: an MZI's splitting ratio is set by
%   theta *given* 50:50 couplers. Move a coupler off 50:50 and the MZI can no
%   longer reach full extinction at any theta, so the error is not a phase the
%   next stage can partly undo -- it is a floor on how well that MZI can route.
%
%   EPSILON must trace to a published coupler-fabrication measurement.  % UNSOURCED
%   Sweep ranges are set by the driver; see docs/parameter_sources.md for the
%   standing entry recording that the mesh set is not yet PDK-anchored.
    if ~isfield(params, 'splits') || isempty(params.splits)
        error("err:coupler_imbalance:noSplits", ...
            ["PARAMS has no 'splits' field. Coupler imbalance is a mesh-only " ...
             "source; a D2NN parameter set has no couplers to imbalance."]);
    end
    s = RandStream('twister', 'Seed', seed);
    params.splits = params.splits + epsilon * randn(s, size(params.splits));
    % A power fraction outside [0, 1] is not a coupler. meshmodel.beamsplitter
    % clamps too, but clamping here keeps PARAMS itself physical for anything that
    % inspects or records it.
    params.splits = min(max(params.splits, 0), 1);
end
