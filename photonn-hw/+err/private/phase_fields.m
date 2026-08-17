function names = phase_fields(params)
%PHASE_FIELDS Names of the phase-carrying fields PARAMS holds, in a fixed order.
%   NAMES = PHASE_FIELDS(PARAMS) returns a cellstr naming every programmed-phase
%   array in PARAMS: {'phase_masks'} for a D2NN parameter set, or {'theta','phi',
%   'outPhase'} for a mesh one.
%
%   This is what lets err.phase_shifter_error and err.quantize serve both
%   architectures without either growing a model_type switch. It matters that they
%   do: sigma stays in radians and bit depth stays in bits across both, so the two
%   headline tables in docs/tolerance_d2nn.md and docs/tolerance_mesh.md are
%   measuring the same quantity and can be compared directly.
%
%   Private to +err.
    candidates = {'phase_masks', 'theta', 'phi', 'outPhase'};
    names = candidates(cellfun(@(f) isfield(params, f) && ~isempty(params.(f)), candidates));
    if isempty(names)
        error("err:phase_fields:noPhases", ...
            ["PARAMS carries no programmed phases (looked for %s). A D2NN set " ...
             "needs phase_masks; a mesh set needs theta/phi/outPhase."], ...
            strjoin(candidates, ', '));
    end
end
