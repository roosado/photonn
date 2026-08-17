function b = beamsplitter(splitRatio)
%BEAMSPLITTER 2x2 unitary for a directional coupler.
%   B = MESHMODEL.BEAMSPLITTER(SPLITRATIO) returns the symmetric lossless coupler
%   [cos k, i sin k; i sin k, cos k] with sin^2(k) = SPLITRATIO, where SPLITRATIO
%   is the *power* fraction crossed to the other port (0.5 = 50:50).
%
%   Port of photonn.mzi.beamsplitter. This is the only place coupler imbalance can
%   enter the model: err.coupler_imbalance perturbs SPLITRATIO away from 0.5, which
%   is why the split is a parameter here rather than baked into mzi_matrix.
%
%   SPLITRATIO is clamped to [0, 1]; a power fraction outside that is not a coupler,
%   and an imbalance draw in the tail of a Gaussian can land there.
    splitRatio = min(max(splitRatio, 0), 1);
    k = asin(sqrt(splitRatio));
    c = cos(k);
    s = sin(k);
    b = [c, 1i * s; 1i * s, c];
end
