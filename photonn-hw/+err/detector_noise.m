function readout = detector_noise(readout, opts, seed)
%DETECTOR_NOISE Add shot, thermal, and ADC-quantization noise to a readout.
%   READOUT = ERR.DETECTOR_NOISE(READOUT, OPTS, SEED) takes per-region photon
%   counts (from the Phase-2 photon budget, computed in model.evaluate) and
%   applies, in order:
%     1. shot noise  - Poisson(count), here the Gaussian approximation
%        N(count, count) which is accurate for the large photon counts of this
%        budget (>~1e10) and, unlike poissrnd, needs no toolbox;
%     2. thermal / read noise - additive N(0, OPTS.read_noise_e^2) electrons;
%     3. ADC quantization - round to 2^OPTS.adc_bits levels over
%        [0, OPTS.full_well_e].
%   SEED gives a reproducible draw (recorded by the MC driver).
%
%   OPTS fields: read_noise_e, adc_bits, and (optional) full_well_e -- the ADC
%   full-scale in the SAME units as READOUT (photons). If full_well_e is omitted,
%   the ADC auto-ranges to the batch maximum, i.e. the detector gain is
%   calibrated to the signal; this is the physically correct default. (A fixed
%   full-scale far above the signal would quantize a low-photon readout to zero,
%   an instrumentation artifact rather than a fundamental limit.)
%   Values/detector sources are in docs/parameter_sources.md.
    s = RandStream('twister', 'Seed', seed);

    % 1. shot noise (Gaussian approximation to Poisson; valid for large counts)
    readout = readout + sqrt(max(readout, 0)) .* randn(s, size(readout));

    % 2. thermal / read noise
    if isfield(opts, 'read_noise_e') && opts.read_noise_e > 0
        readout = readout + opts.read_noise_e * randn(s, size(readout));
    end
    readout = max(readout, 0);

    % 3. ADC quantization; full-scale calibrated to the signal unless specified.
    if isfield(opts, 'adc_bits') && opts.adc_bits > 0
        if isfield(opts, 'full_well_e') && opts.full_well_e > 0
            fullScale = opts.full_well_e;
        else
            fullScale = max(readout(:));   % auto-range: gain set to the signal
        end
        if fullScale > 0
            levels = 2 ^ opts.adc_bits - 1;
            step = fullScale / levels;
            readout = min(readout, fullScale);
            readout = round(readout / step) * step;
        end
    end
end
