function h = sensitivity_map(sensitivity, layout)
%SENSITIVITY_MAP Plot the spatial sensitivity of accuracy to local phase error.
%   H = VIZ.SENSITIVITY_MAP(SENSITIVITY, LAYOUT) renders, per mask, how strongly
%   accuracy drops when a fixed phase error is applied to each local block of that
%   mask. SENSITIVITY is N-by-N-by-L (accuracy drop per block, one page per mask);
%   LAYOUT is an optional label string for the title. Returns the figure handle.
    if nargin < 2 || isempty(layout), layout = 'phase-mask stack'; end
    L = size(sensitivity, 3);

    h = figure('Color', 'w', 'Position', [100 100 260 * L, 300]);
    try, theme(h, 'light'); catch, end
    clim = [0, max(sensitivity(:)) + eps];
    for k = 1:L
        subplot(1, L, k);
        imagesc(sensitivity(:, :, k), clim);
        axis image off;
        colormap(hot);
        title(sprintf('mask %d', k));
    end
    cb = colorbar('Position', [0.93 0.15 0.015 0.7]);
    cb.Label.String = 'accuracy drop';
    sgtitle(sprintf('Spatial sensitivity map (%s)', layout), 'Interpreter', 'none');
end
