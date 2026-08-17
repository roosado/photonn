function params = quantize(params, n_bits)
%QUANTIZE Round every programmed phase to a finite-resolution DAC over [0, 2*pi).
%   PARAMS = ERR.QUANTIZE(PARAMS, N_BITS) wraps each phase to [0, 2*pi) and snaps
%   it to the nearest of 2^N_BITS evenly spaced levels, modelling finite DAC
%   control resolution. For a D2NN that is each phase-mask pixel; for an MZI mesh
%   it is each thermo-optic shifter (theta, phi and the output phase screen).
%   Sweep N_BITS over 6/8/10/12 for the tolerance curve. Deterministic (no seed).
%
%   Note the levels are evenly spaced in *phase*. A thermo-optic shifter is driven
%   in power and phase goes as applied power, so an evenly-spaced-in-code DAC is
%   evenly spaced in phase too; a shifter driven in voltage would not be, and that
%   is a modelling choice recorded in docs/parameter_sources.md rather than a fact.
    levels = 2 ^ n_bits;
    step = 2 * pi / levels;
    fields = phase_fields(params);
    for i = 1:numel(fields)
        f = fields{i};
        params.(f) = round(mod(params.(f), 2 * pi) / step) * step;
    end
end
