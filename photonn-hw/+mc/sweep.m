function acc = sweep(h, cfgs, nReal, baseSeed, mcFn)
%SWEEP Run a Monte Carlo at each error magnitude; return nMag-by-nReal accuracy.
%   ACC = MC.SWEEP(H, CFGS, NREAL, BASESEED) runs mc.run_montecarlo once per
%   config in the cell array CFGS, with NREAL realizations each, and returns their
%   accuracies. Seeds are partitioned BASESEED + 100*i per magnitude, so no two
%   draws in a budget collide.
%
%   ACC = MC.SWEEP(..., MCFN) uses MCFN instead, e.g. @mc.run_montecarlo_mesh.
%   MCFN must take (handoff, errorConfig, nRealizations, baseSeed) and return a
%   struct with .acc and .mean.
%
%   Shared by run_error_budget and run_error_budget_mesh. It was a local function
%   in the former; lifting it here is what stops the two drivers drifting apart on
%   how a sweep is seeded, which is the thing that would quietly make their
%   tolerance tables incomparable.
    if nargin < 5 || isempty(mcFn), mcFn = @mc.run_montecarlo; end
    if ~iscell(cfgs), cfgs = num2cell(cfgs); end

    nMag = numel(cfgs);
    acc = zeros(nMag, nReal);
    for i = 1:nMag
        st = mcFn(h, cfgs{i}, nReal, baseSeed + 100 * i);
        acc(i, :) = st.acc(:)';
        fprintf('  [%2d/%2d] mean acc %.4f\n', i, nMag, st.mean);
    end
end
