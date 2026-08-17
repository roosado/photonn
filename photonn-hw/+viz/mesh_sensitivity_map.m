function h = mesh_sensitivity_map(sensitivity, sched, names, quantity)
%MESH_SENSITIVITY_MAP Per-MZI sensitivity drawn on the Clements rectangle.
%   H = VIZ.MESH_SENSITIVITY_MAP(SENSITIVITY, SCHED, NAMES, QUANTITY) renders one
%   panel per mesh. SENSITIVITY is nMzi-by-nMeshes (how much the network moves when
%   that one MZI is detuned); SCHED comes from meshmodel.schedule; NAMES is a
%   cellstr of panel titles, e.g. {'V mesh', 'U mesh'}; QUANTITY labels the
%   colorbar (default 'accuracy drop').
%
%   Each MZI is drawn at its layout position -- column across, mode pair down --
%   so the picture is the chip, and a reader can see *where* on the device the
%   sensitivity sits rather than only how much of it there is.
%
%   This exists because viz.sensitivity_map cannot serve a mesh, and the reason is
%   worth recording. That one lays out 1-by-L subplots at 260*L pixels, which for
%   the 56-mask D2NN produced a 16139-by-341 plate at 47:1 -- fifteen pixels tall
%   in a web grid cell, unpublishable at any size. A mesh map is two panels of
%   nModes columns by nModes/2 rows, so it is near-square by construction.
    if nargin < 3 || isempty(names)
        names = arrayfun(@(k) sprintf('mesh %d', k), 1:size(sensitivity, 2), ...
                         'UniformOutput', false);
    end
    if nargin < 4 || isempty(quantity), quantity = 'accuracy drop'; end
    nMeshes = size(sensitivity, 2);

    h = figure('Color', 'w', 'Position', [100 100 460 * nMeshes 420]);
    try, theme(h, 'light'); catch, end

    lim = max(abs(sensitivity(:)));
    if lim == 0, lim = 1; end

    % Three states have to be distinguishable, and the obvious encoding gets two of
    % them backwards: a Clements brick is a checkerboard, so most of the panel is
    % *no device at all*. With hot on a white axes, an insensitive MZI is black and
    % an empty site is white, which reads as the opposite of the truth. So the axes
    % carry a grey "no MZI here" ground and the map runs white -> red -> black, and
    % ink means sensitivity everywhere on the panel.
    for k = 1:nMeshes
        ax = subplot(1, nMeshes, k);
        set(ax, 'Color', [0.90 0.90 0.92]);
        scatter(ax, sched.pos(:, 1), sched.pos(:, 2), 58, sensitivity(:, k), ...
                'filled', 's', 'MarkerEdgeColor', [0.75 0.75 0.78], 'LineWidth', 0.25);
        set(ax, 'YDir', 'reverse');            % mode 1 at the top, as drawn elsewhere
        xlim(ax, [0.5, sched.nModes + 0.5]);
        ylim(ax, [1, sched.nModes + 1]);
        clim(ax, [0 lim]);
        colormap(ax, flipud(hot));
        xlabel(ax, 'mesh column');
        ylabel(ax, 'mode pair');
        title(ax, names{k}, 'Interpreter', 'none');
        box(ax, 'on');
    end

    cb = colorbar('Position', [0.93 0.16 0.015 0.7]);
    cb.Label.String = quantity;
    sgtitle(sprintf('Per-MZI sensitivity (%d modes, %d MZIs per mesh)', ...
                    sched.nModes, sched.nMzi));
end
