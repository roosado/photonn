function regions = regions_from_handoff(raw)
%REGIONS_FROM_HANDOFF Detector layout as carried by the handoff (schema 0.3.0+).
%   REGIONS = MODEL.REGIONS_FROM_HANDOFF(RAW) converts the nClasses-by-4 array
%   read from /geometry/detector_regions into the same struct array
%   MODEL.DETECTOR_REGIONS returns, so callers cannot tell which one produced it.
%
%   RAW rows are (y0, y1, x0, x1) in Python's 0-based half-open convention. The
%   conversion to MATLAB's 1-based inclusive form happens here, once, rather than
%   at each use.
%
%   This exists so the as-built model *reads* where the detectors sit instead of
%   re-deriving it. detect.default_regions is the sole author of that layout;
%   MODEL.DETECTOR_REGIONS remains as the fallback for pre-0.3.0 files, and is
%   the port those files were always read with.

    if isempty(raw)
        error("model:regions_from_handoff:empty", ...
            "No detector_regions in the handoff; caller should fall back to " + ...
            "model.detector_regions for pre-0.3.0 files.");
    end
    if size(raw, 2) ~= 4
        error("model:regions_from_handoff:badShape", ...
            "detector_regions must be nClasses-by-4, got %d-by-%d.", ...
            size(raw, 1), size(raw, 2));
    end

    nClasses = size(raw, 1);
    regions = repmat(struct('label', 0, 'rlo', 1, 'rhi', 1, 'clo', 1, 'chi', 1), 1, nClasses);
    for k = 1:nClasses
        regions(k).label = k - 1;              % 0-based class label, as Python writes
        regions(k).rlo = double(raw(k, 1)) + 1;
        regions(k).rhi = double(raw(k, 2));    % half-open y1 -> inclusive rhi
        regions(k).clo = double(raw(k, 3)) + 1;
        regions(k).chi = double(raw(k, 4));
    end
end
