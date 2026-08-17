function print_summary(r, sources)
%PRINT_SUMMARY Print where each swept source first falls below its threshold.
%   VIZ.PRINT_SUMMARY(R, SOURCES) walks the cellstr SOURCES, looks each up in the
%   results struct R, and reports the first magnitude whose mean accuracy is below
%   the threshold -- or that the sweep never reached it.
%
%   This is an orientation aid, not the published number. The docs quote each edge
%   as a *bracket* ("holds at X, fails at Y") read off the magnitudes, because an
%   interpolated crossing implies a resolution the sweep grid does not have.
    fprintf('\n--- tolerance summary (threshold = %.4f, 95%% of ideal %.4f) ---\n', ...
        r.threshold, r.ideal);
    for i = 1:numel(sources)
        if ~isfield(r, sources{i}), continue; end
        s = r.(sources{i});
        below = find(s.accMean < s.threshold, 1);
        if isempty(below)
            edge = 'not reached in sweep';
        else
            edge = sprintf('crosses at magnitude ~%.4g', s.magnitudes(below));
        end
        fprintf('  %-13s: %s\n', sources{i}, edge);
    end
end
