function s = pack(mag, acc, thresh)
%PACK Bundle one sweep into the {magnitudes, accMean, accStd, threshold} record.
%   S = MC.PACK(MAG, ACC, THRESH) summarises the nMag-by-nReal accuracy matrix ACC
%   from mc.sweep against the magnitudes MAG. This is the shape every source takes
%   in error_budget_results.mat and the shape viz.tolerance_curve expects.
%
%   Mean and standard deviation only. The tolerance edges quoted in the docs are
%   read off the magnitudes as a bracket -- "holds at X, fails at Y" -- never
%   interpolated from these, so no fitted crossing point is stored.
    s.magnitudes = mag(:)';
    s.accMean = mean(acc, 2)';
    s.accStd = std(acc, 0, 2)';
    s.threshold = thresh;
end
