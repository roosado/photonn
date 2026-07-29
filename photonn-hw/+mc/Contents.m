% +MC  Monte Carlo drivers and statistics (as-built side).
%
%   Runs many realizations of the error model over an imported parameter set,
%   evaluates classification accuracy on the frozen test set, and collects the
%   statistics behind the tolerance curves.
%
%   Random seeds are fixed and recorded for every run (CLAUDE.md convention).
%
%   Functions
%     run_montecarlo  - Drive N realizations of a chosen error configuration.
