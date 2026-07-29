function h = tolerance_curve(magnitudes, accuracy, sourceName)
%TOLERANCE_CURVE Plot accuracy versus error magnitude for one source.
%   H = VIZ.TOLERANCE_CURVE(MAGNITUDES, ACCURACY, SOURCENAME) plots the Monte
%   Carlo accuracy (mean, with a +/-1 std shaded band) against the swept error
%   magnitude -- the core deliverable behind the tolerance document.
%
%   MAGNITUDES is a 1-by-M vector; ACCURACY is M-by-NREALIZATIONS (one row per
%   swept magnitude). Returns the figure handle. The caller may set a log x-axis
%   afterwards for magnitudes that span decades (e.g. photon budget).
    magnitudes = magnitudes(:);
    m = mean(accuracy, 2);
    sd = std(accuracy, 0, 2);

    h = figure('Color', 'w');
    try, theme(h, 'light'); catch, end     % force light theme (R2025b defaults to OS theme)
    band = [m - sd; flipud(m + sd)];
    fill([magnitudes; flipud(magnitudes)], band, [0.30 0.45 0.68], ...
        'FaceAlpha', 0.18, 'EdgeColor', 'none'); hold on;
    plot(magnitudes, m, '-o', 'LineWidth', 1.6, 'Color', [0.17 0.30 0.55], ...
        'MarkerFaceColor', [0.17 0.30 0.55], 'MarkerSize', 4);
    grid on; box on;
    ylim([0 1]);
    xlabel(sourceName, 'Interpreter', 'none');
    ylabel('classification accuracy');
    title(sprintf('Tolerance curve: %s', sourceName), 'Interpreter', 'none');
end
