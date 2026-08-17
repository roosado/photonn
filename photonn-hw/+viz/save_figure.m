function save_figure(fig, figDir, name)
%SAVE_FIGURE Write a figure to FIGDIR/NAME at the project's fixed resolution.
%   VIZ.SAVE_FIGURE(FIG, FIGDIR, NAME) exports at 130 dpi and closes FIG.
%
%   130 dpi is the figure resolution the whole project ships at; the site's
%   encoder (apps/build_site.encode_figure) downscales from there. Shared by both
%   error-budget drivers so the two studies' plates match.
    exportgraphics(fig, fullfile(figDir, name), 'Resolution', 130);
    close(fig);
end
