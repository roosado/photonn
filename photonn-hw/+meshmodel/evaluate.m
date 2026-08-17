function out = evaluate(handoff, opts)
%EVALUATE Run the as-built MZI-mesh forward pass over the frozen test set.
%   OUT = MESHMODEL.EVALUATE(HANDOFF) reproduces the ideal forward pass from a
%   mesh handoff struct (io.read_handoff, schema 0.2.0) and returns OUT with
%   fields accuracy, predictions (0-based), logits, regionIntensity and labels --
%   the same contract model.evaluate exposes for the D2NN, which is what lets +mc
%   and +viz serve both architectures.
%
%   OUT = MESHMODEL.EVALUATE(HANDOFF, OPTS) applies error-model overrides:
%     OPTS.params   - struct with .theta, .phi, .sigma, .outPhase (the ideal values
%                     by default) and optionally .splits (2*nMzi-by-2 coupler power
%                     splits) and .lossDb (2*nMzi-by-1 per-MZI insertion loss).
%                     This is what the +err device models perturb.
%     OPTS.detector - struct enabling the detector-noise path (input_power_w,
%                     integration_time_s, read_noise_e, adc_bits, full_well_e).
%     OPTS.seed     - RNG seed for the detector-noise draw.
%     OPTS.subset   - test-set indices, for faster Monte Carlo sweeps.
%
%   The operator is U * diag(sigma) * V, with **no conjugate transpose** -- the
%   prose calls it U*Sigma*V', and since V is a free unitary the model class is the
%   same, but V is used exactly as stored. theta/phi are concatenated [V, U] and
%   outPhase is indexed the same way (handoff.parameters.mesh_order).
%
%   Mode ordering: test images arrive g-by-g-by-B from h5read (dims reversed), and
%   the Python encoder flattened them row-major, so reshape(images, nModes, [])'
%   recovers the mode vector directly. Get this wrong and the accuracy comes out
%   plausible but not equal to 0.7355 -- which is why the gate is exact equality.
    if nargin < 2, opts = struct(); end

    if handoff.parameters.model_type ~= "mesh"
        error("meshmodel:evaluate:wrongModel", ...
            "meshmodel.evaluate needs a mesh handoff; got '%s'. Use model.evaluate.", ...
            handoff.parameters.model_type);
    end

    nModes   = double(handoff.parameters.n_modes);
    nMzi     = double(handoff.parameters.n_mzi);
    nClasses = double(handoff.operating_point.n_classes);
    gain     = handoff.operating_point.readout_gain;
    lambda   = handoff.operating_point.wavelength_m;
    sched    = meshmodel.schedule(nModes);

    % -- parameters: ideal, or whatever +err handed us -------------------
    if isfield(opts, 'params') && ~isempty(opts.params)
        p = opts.params;
    else
        p = struct();
    end
    theta    = defaulted(p, 'theta',    handoff.parameters.phase_theta);
    phi      = defaulted(p, 'phi',      handoff.parameters.phase_phi);
    sigma    = defaulted(p, 'sigma',    handoff.parameters.sigma);
    outPhase = defaulted(p, 'outPhase', handoff.parameters.out_phase);
    splits   = defaulted(p, 'splits',   repmat(0.5, 2 * nMzi, 2));
    lossDb   = defaulted(p, 'lossDb',   zeros(2 * nMzi, 1));

    v = 1:nMzi;              % first half of the concatenation is the V mesh
    u = nMzi + (1:nMzi);
    mV = meshmodel.mesh_matrix(theta(v), phi(v), outPhase(1, :), sched, ...
                               struct('splits', splits(v, :), 'lossDb', lossDb(v)));
    mU = meshmodel.mesh_matrix(theta(u), phi(u), outPhase(2, :), sched, ...
                               struct('splits', splits(u, :), 'lossDb', lossDb(u)));
    operator = mU * diag(complex(sigma(:))) * mV;

    % -- frozen test set -------------------------------------------------
    x = reshape(double(handoff.test_set.images), nModes, []).';   % B-by-nModes
    labels = double(handoff.test_set.labels(:));
    if isfield(opts, 'subset') && ~isempty(opts.subset)
        x = x(opts.subset, :);
        labels = labels(opts.subset);
    end

    % Row-vector convention, matching layers.MZIMeshLayer.forward (x @ M.T).
    y = x * operator.';
    intensity = abs(y) .^ 2;
    regionIntensity = intensity(:, 1:nClasses);
    total = max(sum(intensity, 2), 1e-12);
    logits = regionIntensity ./ total * gain;

    if isfield(opts, 'detector') && ~isempty(opts.detector)
        det = opts.detector;
        hPlanck = 6.62607015e-34;  cLight = 299792458.0;   % exact SI constants
        ePhoton = hPlanck * cLight / lambda;
        nIn = det.input_power_w * det.integration_time_s / ePhoton;   % photons/inference

        % Unlike the D2NN, loss needs no separate scalar here. Per-MZI loss is
        % inside the operator, so it already shows up as missing power in
        % regionIntensity -- and because different modes cross different numbers of
        % MZIs, it does not cancel against inputRef the way a uniform mask loss does.
        inputRef = max(sum(abs(x) .^ 2, 2), 1e-12);
        photons = (regionIntensity ./ inputRef) * nIn;
        seed = 0;
        if isfield(opts, 'seed'), seed = opts.seed; end
        noisy = err.detector_noise(photons, det, seed);
        [~, pred] = max(noisy, [], 2);
        out.logits = noisy;
    else
        [~, pred] = max(regionIntensity, [], 2);   % argmax is invariant to normalisation
        out.logits = logits;
    end

    out.predictions = pred - 1;                 % 0-based class labels
    out.labels = labels;
    out.regionIntensity = regionIntensity;
    out.accuracy = mean(out.predictions == labels);
end


function v = defaulted(s, f, d)
%DEFAULTED Field F of struct S, or D when absent or empty.
    if isfield(s, f) && ~isempty(s.(f))
        v = s.(f);
    else
        v = d;
    end
end
