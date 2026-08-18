function out = evaluate(handoff, opts)
%EVALUATE Run the as-built D2NN forward pass over the frozen test set.
%   OUT = MODEL.EVALUATE(HANDOFF) reproduces the ideal forward pass from a
%   handoff struct (io.read_handoff) and returns OUT with fields accuracy,
%   predictions (0-based), logits, regionIntensity, and labels.
%
%   OUT = MODEL.EVALUATE(HANDOFF, OPTS) applies error-model overrides:
%     OPTS.params   - struct with .phase_masks (N×N×L, handoff orientation) and
%                     optional .wavelength_m, .separations_m (the L+1 free-space
%                     gaps) and .detector_shift_px ([dRow dCol], sub-pixel);
%                     defaults to the ideal values. This is what the +err models
%                     perturb -- the device ones through the masks and wavelength,
%                     the geometry ones through the last two.
%     OPTS.detector - struct enabling the detector-noise path (fields
%                     input_power_w, integration_time_s, read_noise_e, adc_bits,
%                     full_well_e). When present, predictions come from noisy
%                     per-region photon counts via err.detector_noise.
%     OPTS.seed     - RNG seed for the detector-noise draw.
%
%   The h5 dimension reversal (Python [L,N,N] -> MATLAB N×N×L) is undone here by
%   permute(...,[2 1 3]); a correct baseline accuracy confirms the orientation.
    if nargin < 2, opts = struct(); end

    N          = double(handoff.geometry.grid_size);
    dx         = handoff.operating_point.pixel_pitch_m;
    gain       = handoff.operating_point.readout_gain;
    phaseScale = handoff.operating_point.phase_scale_rad;
    inputFrac  = handoff.operating_point.input_frac;
    encCode    = handoff.operating_point.encoding_code;
    sep        = handoff.geometry.layer_separations_m(:);   % L+1 uniform gaps

    detShift = [0 0];
    if isfield(opts, 'params') && ~isempty(opts.params)
        masks = opts.params.phase_masks;
        if isfield(opts.params, 'wavelength_m') && ~isempty(opts.params.wavelength_m)
            lambda = opts.params.wavelength_m;
        else
            lambda = handoff.operating_point.wavelength_m;
        end
        % Geometry errors arrive the same way device errors do: as a perturbed
        % parameter, never as a flag read here. Absent means nominal.
        if isfield(opts.params, 'separations_m') && ~isempty(opts.params.separations_m)
            sep = opts.params.separations_m(:);
        end
        if isfield(opts.params, 'detector_shift_px') && ~isempty(opts.params.detector_shift_px)
            detShift = opts.params.detector_shift_px;
        end
    else
        masks = handoff.parameters.phase_masks;
        lambda = handoff.operating_point.wavelength_m;
    end

    % Undo h5 dimension reversal -> Python [row, col] plane orientation.
    masks  = permute(double(masks), [2 1 3]);                     % N×N×L
    images = permute(double(handoff.test_set.images), [2 1 3]);   % N×N×B
    labels = double(handoff.test_set.labels(:));

    % Optional test subset (for faster Monte Carlo / sensitivity sweeps).
    if isfield(opts, 'subset') && ~isempty(opts.subset)
        images = images(:, :, opts.subset);
        labels = labels(opts.subset);
    end

    field = model.encode_input(images, phaseScale, encCode, inputFrac);
    inputField = field;

    L = size(masks, 3);
    if numel(sep) ~= L + 1
        error("model:evaluate:gapCount", ...
            "expected %d free-space gaps for %d masks, got %d.", L + 1, L, numel(sep));
    end
    for k = 1:L
        field = model.angular_spectrum(field, dx, lambda, sep(k));
        field = field .* exp(1i * masks(:, :, k));
    end
    field = model.angular_spectrum(field, dx, lambda, sep(L + 1));

    % A laterally misplaced detector array is the output field arriving at the
    % wrong place on it. Translating the field by -shift is exactly equivalent to
    % translating the boxes by +shift, and leaves the detector layout -- which is
    % shared with the design side and with the browser port -- untouched.
    if any(detShift ~= 0)
        field = model.subpixel_shift(field, -detShift(1), -detShift(2));
    end

    regions = model.detector_regions(N, 10);   % MNIST: 10 detector classes
    [logits, regionIntensity, inputRef] = model.readout(field, regions, gain, inputField);

    if isfield(opts, 'detector') && ~isempty(opts.detector)
        det = opts.detector;
        hPlanck = 6.62607015e-34;  cLight = 299792458.0;   % exact SI constants
        ePhoton = hPlanck * cLight / lambda;
        nIn = det.input_power_w * det.integration_time_s / ePhoton;   % photons/inference

        % Optical loss (if err.loss recorded it) only bites here: it attenuates
        % the photon budget, raising shot noise, but cancels in the ideal readout.
        lossPower = 1.0;
        if isfield(opts, 'params') && ~isempty(opts.params) ...
                && isfield(opts.params, 'insertion_db')
            totalDb = L * opts.params.insertion_db ...
                    + sum(sep) * 100 * opts.params.propagation_db_per_cm;
            lossPower = 10 ^ (-totalDb / 10);
        end

        photons = (regionIntensity ./ inputRef) * nIn * lossPower;   % B×nClasses counts
        seed = 0;
        if isfield(opts, 'seed'), seed = opts.seed; end
        noisy = err.detector_noise(photons, det, seed);
        [~, pred] = max(noisy, [], 2);
        out.logits = noisy;
    else
        [~, pred] = max(regionIntensity, [], 2);   % argmax is invariant to gain/normalisation
        out.logits = logits;
    end

    out.predictions = pred - 1;                 % 0-based class labels
    out.labels = labels;
    out.regionIntensity = regionIntensity;
    out.accuracy = mean(out.predictions == labels);
end
