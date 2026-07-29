function field = encode_input(imageMaps, phaseScale, encodingCode, inputFrac)
%ENCODE_INPUT Reconstruct the complex input field from the stored input maps.
%   FIELD = MODEL.ENCODE_INPUT(IMAGEMAPS, PHASESCALE, ENCODINGCODE, INPUTFRAC)
%   mirrors photonn/train.encode_input. IMAGEMAPS is N-by-N-by-B raw input maps
%   in [0,1] (the handoff /test_set/images). ENCODINGCODE selects the scheme:
%   0 = amplitude, 1 = phase, 2 = both. The forward pass is scale-invariant, so
%   no per-sample normalisation is applied here (unlike the Python encoder, whose
%   normalisation cancels in the readout anyway).
    if nargin < 4, inputFrac = 0.5; end
    img = double(imageMaps);
    switch encodingCode
        case 0   % amplitude: |E| = image, phase 0
            field = complex(img);
        case 2   % both: |E| = image, arg(E) = phaseScale * image
            field = img .* exp(1i * phaseScale * img);
        case 1   % phase: plane wave through the input aperture, arg = phaseScale*image
            n = size(img, 1);
            win = max(1, round(inputFrac * n));
            off = floor((n - win) / 2);
            window = zeros(n, n);
            window(off + 1:off + win, off + 1:off + win) = 1;
            field = window .* exp(1i * phaseScale * img);
        otherwise
            error("model:encode_input:badCode", ...
                "Unknown encoding_code %g (expected 0, 1, or 2).", encodingCode);
    end
end
