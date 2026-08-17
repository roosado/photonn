function params = mzi_loss(params, insertion_db, propagation_db_per_cm, mzi_pitch_cm)
%MZI_LOSS Per-MZI insertion loss and waveguide propagation loss in a mesh.
%   PARAMS = ERR.MZI_LOSS(PARAMS, INSERTION_DB) charges every MZI INSERTION_DB of
%   loss, written into PARAMS.lossDb so meshmodel.mzi_matrix can apply it inside
%   each 2x2 block as an amplitude factor.
%
%   PARAMS = ERR.MZI_LOSS(PARAMS, INSERTION_DB, PROPAGATION_DB_PER_CM, MZI_PITCH_CM)
%   adds the waveguide run between columns: each MZI is charged
%   PROPAGATION_DB_PER_CM * MZI_PITCH_CM on top of its insertion loss.
%
%   Why this is a different source from err.loss, rather than the same one renamed.
%   In the D2NN the masks are pure phase, so loss is a near-uniform attenuation
%   that **cancels** in the power-normalised readout and is felt only through the
%   photon budget (see err.loss). In a Clements rectangle it does not cancel: the
%   mesh is a brick, so a mode near the edge passes through fewer MZIs than one in
%   the middle, and the loss is therefore **mode-dependent**. It tilts the realised
%   linear map rather than scaling it, and it is a first-order source here for the
%   first time in this project.
%
%   Deterministic (no seed) -- this models the process mean, not run-to-run spread.
%   Per-MZI variation in loss would be a separate draw and is not modelled.
%
%   INSERTION_DB and PROPAGATION_DB_PER_CM must trace to published integrated-
%   photonics process data.  % UNSOURCED
    if ~isfield(params, 'lossDb') || isempty(params.lossDb)
        error("err:mzi_loss:noLossField", ...
            ["PARAMS has no 'lossDb' field. Per-MZI loss is a mesh-only source; " ...
             "the D2NN's uniform mask loss is err.loss."]);
    end
    if nargin < 3 || isempty(propagation_db_per_cm), propagation_db_per_cm = 0; end
    if nargin < 4 || isempty(mzi_pitch_cm), mzi_pitch_cm = 0; end

    perMzi = insertion_db + propagation_db_per_cm * mzi_pitch_cm;
    params.lossDb = params.lossDb + perMzi;
end
