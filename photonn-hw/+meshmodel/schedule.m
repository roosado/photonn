function s = schedule(nModes)
%SCHEDULE Clements rectangular mesh topology for NMODES modes.
%   S = MESHMODEL.SCHEDULE(NMODES) returns the pairing schedule that
%   photonn.layers.MZIMeshLayer builds: NMODES columns, column L coupling adjacent
%   mode pairs starting at mod(L,2), for NMODES*(NMODES-1)/2 MZIs in total -- the
%   Clements bound, and the reason 36 columns suffice for 36 modes.
%
%   Fields:
%     nModes   - mode count
%     nMzi     - NMODES*(NMODES-1)/2
%     columns  - 1-by-NMODES cell; each is 2-by-k [topMode; mziIndex], 1-based
%     top      - nMzi-by-1 top mode of each MZI (1-based)
%     column   - nMzi-by-1 column each MZI sits in (1-based)
%     pos      - nMzi-by-2 layout coordinate [x y] in (column, mode) units, with
%                y = top + 0.5 because an MZI sits *between* its two modes
%
%   POS exists for the two things a phase-mask model never needed: a distance-
%   dependent thermal coupling matrix (err.mesh_thermal_crosstalk) and a per-MZI
%   sensitivity map drawn on the physical rectangle (viz.mesh_sensitivity_map).
%
%   Indices are 1-based throughout, unlike the 0-based Python schedule. The
%   pairing order is identical; only the offset moves.
    nModes = double(nModes);
    s.nModes = nModes;
    s.nMzi = nModes * (nModes - 1) / 2;

    s.columns = cell(1, nModes);
    s.top = zeros(s.nMzi, 1);
    s.column = zeros(s.nMzi, 1);
    idx = 0;
    for col = 1:nModes
        offset = mod(col - 1, 2);              % Python: layer % 2, 0-based
        tops = (offset + 1) : 2 : (nModes - 1);   % 1-based top mode of each pair
        k = numel(tops);
        s.columns{col} = [tops; idx + (1:k)];
        s.top(idx + (1:k)) = tops;
        s.column(idx + (1:k)) = col;
        idx = idx + k;
    end

    if idx ~= s.nMzi
        error("meshmodel:schedule:count", ...
            "Built %d MZIs, expected %d for %d modes.", idx, s.nMzi, nModes);
    end
    s.pos = [s.column, s.top + 0.5];
end
