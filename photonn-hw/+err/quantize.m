function params = quantize(params, n_bits)
%QUANTIZE Round each mask phase to a finite-resolution DAC over [0, 2*pi).
%   PARAMS = ERR.QUANTIZE(PARAMS, N_BITS) wraps each phase to [0, 2*pi) and snaps
%   it to the nearest of 2^N_BITS evenly spaced levels, modelling finite DAC / SLM
%   control resolution. Sweep N_BITS over 6/8/10/12 for the tolerance curve.
%   Deterministic (no seed).
    levels = 2 ^ n_bits;
    step = 2 * pi / levels;
    wrapped = mod(params.phase_masks, 2 * pi);
    params.phase_masks = round(wrapped / step) * step;
end
