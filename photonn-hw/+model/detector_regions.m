function regions = detector_regions(n, nClasses)
%DETECTOR_REGIONS Fixed detector layout (port of photonn/detect.default_regions).
%   REGIONS = MODEL.DETECTOR_REGIONS(N, NCLASSES) returns a 1-by-NCLASSES struct
%   array with 1-based inclusive index ranges (RLO:RHI rows, CLO:CHI cols) and
%   the 0-based class LABEL. The arithmetic mirrors
%   detect.default_regions(n, n_classes) (field_frac 0.75, patch_frac 0.11) so
%   the as-built readout lands on exactly the same detector patches the model was
%   trained against.
    if nargin < 2, nClasses = 10; end
    cols = ceil(sqrt(nClasses));
    rows = ceil(nClasses / cols);
    patch = max(1, round(0.11 * n));
    span = 0.75 * n;
    origin = (n - span) / 2;

    regions = repmat(struct('label', 0, 'rlo', 1, 'rhi', 1, 'clo', 1, 'chi', 1), 1, nClasses);
    for idx = 0:nClasses - 1
        r = floor(idx / cols);
        c = mod(idx, cols);
        cy = origin + (r + 0.5) * span / rows;
        cx = origin + (c + 0.5) * span / cols;
        y0 = min(max(round(cy - patch / 2), 0), n - patch);
        x0 = min(max(round(cx - patch / 2), 0), n - patch);
        k = idx + 1;
        regions(k).label = idx;
        regions(k).rlo = y0 + 1;  regions(k).rhi = y0 + patch;
        regions(k).clo = x0 + 1;  regions(k).chi = x0 + patch;
    end
end
