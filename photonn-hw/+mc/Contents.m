% +MC  Monte Carlo drivers and statistics (as-built side).
%
%   Runs many realizations of the error model over an imported parameter set,
%   evaluates classification accuracy on the frozen test set, and collects the
%   statistics behind the tolerance curves.
%
%   Random seeds are fixed and recorded for every run (CLAUDE.md convention).
%
%   An errorConfig selects sources by field *presence*, so an unrecognised field
%   is indistinguishable from an absent one -- a misspelled key silently runs with
%   that source off and produces a flat tolerance curve. validate_config refuses
%   one before any realization is drawn; error_sources is the recognised set.
%
%   Functions
%     run_montecarlo      - Drive N realizations of a chosen error configuration.
%     run_montecarlo_mesh - The same for the MZI mesh; shares the STATS contract.
%     run_montecarlo_deep - The same for the Phase-5 deep mesh: every mesh source
%                           per layer, plus the activation's own.
%     sweep               - Sweep one source over a range, collecting statistics.
%     pack                - Assemble a results struct for +viz and the .mat file.
%     error_sources       - Every errorConfig field a driver recognises, per arch.
%     validate_config     - Reject any field that is not one of them.
