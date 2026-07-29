function [logits, regionIntensity, inputRef] = readout(field, regions, gain, inputField)
%READOUT Detector-region readout (port of the photonn D2NN readout).
%   [LOGITS, REGIONINTENSITY, INPUTREF] = MODEL.READOUT(FIELD, REGIONS, GAIN, INPUTFIELD)
%   integrates intensity |E|^2 over each detector region of the output FIELD
%   (N-by-N-by-B). Returns:
%     LOGITS          - B-by-nClasses, region intensity / total output power * GAIN
%                       (matches the trained model's logits; argmax = class).
%     REGIONINTENSITY - B-by-nClasses raw region sums (unnormalised); argmax over
%                       these gives the same prediction and is what the photon
%                       budget scales for detector noise.
%     INPUTREF        - B-by-1 input power sum|Ein|^2 (if INPUTFIELD given), the
%                       reference for converting region intensity to photon counts.
    intensity = real(field).^2 + imag(field).^2;   % N×N×B
    total = max(sum(sum(intensity, 1), 2), 1e-12);  % 1×1×B
    nClasses = numel(regions);
    B = size(intensity, 3);

    regionIntensity = zeros(B, nClasses);
    for c = 1:nClasses
        rg = regions(c);
        s = sum(sum(intensity(rg.rlo:rg.rhi, rg.clo:rg.chi, :), 1), 2);  % 1×1×B
        regionIntensity(:, c) = s(:);
    end
    logits = regionIntensity ./ total(:) * gain;

    if nargin >= 4 && ~isempty(inputField)
        inRef = sum(sum(real(inputField).^2 + imag(inputField).^2, 1), 2);
        inputRef = inRef(:);
    else
        inputRef = [];
    end
end
