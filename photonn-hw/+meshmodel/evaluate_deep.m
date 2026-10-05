function out = evaluate_deep(handoff, opts)
%EVALUATE_DEEP As-built forward pass of the Phase-5 deep mesh over the frozen test set.
%   OUT = MESHMODEL.EVALUATE_DEEP(HANDOFF) reproduces the ideal forward pass of a
%   deep_mesh handoff (schema 0.4.0): SVD mesh layers with a bank of Williamson et
%   al. (2020) electro-optic activations between consecutive layers. OUT carries the
%   contract meshmodel.evaluate does -- accuracy, predictions (0-based), logits,
%   regionIntensity, labels -- plus TAPS, the per-mode power entering each
%   activation bank in watts.
%
%   Unlike every earlier model in this project, power is physical here. The stored
%   inputs are unit-norm; the light entering the chip is sqrt(input_power_w) times
%   them, every field below is in sqrt(W), and the activation's phase gain is in
%   rad/W. Input power therefore sets where the activation sits on its curve.
%
%   OUT = MESHMODEL.EVALUATE_DEEP(HANDOFF, OPTS) applies as-built overrides:
%     OPTS.params       - struct from meshmodel.deep_params (the ideal set by
%                         default): .layers(l) is one mesh parameter set, the same
%                         shape meshmodel.evaluate takes, so every mesh +err source
%                         applies per layer unchanged; .eo holds one device per
%                         (mode, bank) -- raw values alpha, tia, resp, vPi, vBias,
%                         plus dPhiB (bias error, rad), gainScale, s1, s2.
%     OPTS.inputPowerW  - override the handoff's input power (power sweeps)
%     OPTS.eoNoise      - struct enabling the activation's own detector noise:
%                         .bandwidth_hz, .tia_noise_a_per_rthz (input-referred
%                         amplifier noise density; 0 for shot noise alone)
%     OPTS.detector     - readout noise, as meshmodel.evaluate (input_power_w is
%                         ignored here: the field already carries watts)
%     OPTS.seed         - RNG seed for both noise draws
%     OPTS.subset       - test-set indices
%
%   The activation's phase noise is drawn once per (image, mode): one input is one
%   symbol, and the loop's bandwidth is the symbol rate, so the noise the modulator
%   sees during a symbol is one draw. The readout integrates that same symbol.
    if nargin < 2, opts = struct(); end
    if handoff.parameters.model_type ~= "deep_mesh"
        error("meshmodel:evaluate_deep:wrongModel", ...
            "evaluate_deep needs a deep_mesh handoff; got '%s'.", handoff.parameters.model_type);
    end

    nModes   = double(handoff.parameters.n_modes);
    nLayers  = double(handoff.parameters.n_layers);
    nClasses = double(handoff.operating_point.n_classes);
    gainRd   = handoff.operating_point.readout_gain;
    lambda   = handoff.operating_point.wavelength_m;
    sched    = meshmodel.schedule(nModes);

    if isfield(opts, 'params') && ~isempty(opts.params)
        p = opts.params;
    else
        p = meshmodel.deep_params(handoff);
    end
    pIn = handoff.operating_point.input_power_w;
    if isfield(opts, 'inputPowerW') && ~isempty(opts.inputPowerW), pIn = opts.inputPowerW; end

    x = handoff.test_set.inputs;
    labels = double(handoff.test_set.labels(:));
    if isfield(opts, 'subset') && ~isempty(opts.subset)
        x = x(opts.subset, :);
        labels = labels(opts.subset);
    end
    seed = 0;
    if isfield(opts, 'seed'), seed = opts.seed; end
    noiseStream = RandStream('twister', 'Seed', seed + 7919);   % distinct from the readout draw

    qe = 1.602176634e-19;              % elementary charge, exact SI
    z = sqrt(pIn) * x;                 % sqrt(W), B-by-nModes
    taps = cell(1, nLayers - 1);
    for l = 1:nLayers
        L = p.layers(l);
        v = 1:sched.nMzi;
        u = sched.nMzi + (1:sched.nMzi);
        mV = meshmodel.mesh_matrix(L.theta(v), L.phi(v), L.outPhase(1, :), sched, ...
                                   struct('splits', L.splits(v, :), 'lossDb', L.lossDb(v)));
        mU = meshmodel.mesh_matrix(L.theta(u), L.phi(u), L.outPhase(2, :), sched, ...
                                   struct('splits', L.splits(u, :), 'lossDb', L.lossDb(u)));
        z = z * (mU * diag(complex(L.sigma(:))) * mV).';
        if l == nLayers, break; end

        % -- the activation bank between layer l and l+1 --------------------
        e = p.eo;
        alpha = e.alpha(:, l).';                                   % 1-by-nModes, per device
        gPhi  = pi * alpha .* e.tia(:, l).' .* e.resp(:, l).' ./ e.vPi(:, l).' ...
                .* e.gainScale(:, l).';                            % rad/W, Eq. (7)
        phiB  = pi * e.vBias(:, l).' ./ e.vPi(:, l).' + e.dPhiB(:, l).';   % rad, Eq. (5)
        power = abs(z) .^ 2;                                       % W into each device
        taps{l} = power;
        if isfield(opts, 'eoNoise') && ~isempty(opts.eoNoise)
            % The photodiode sees alpha*P; its current carries shot noise 2qIB and the
            % amplifier adds its own input-referred density. The modulator turns a
            % current error into a phase error at pi*G/V_pi rad per amp.
            nz = opts.eoNoise;
            current = e.resp(:, l).' .* alpha .* power;            % A
            varI = 2 * qe * current * nz.bandwidth_hz + nz.tia_noise_a_per_rthz ^ 2 * nz.bandwidth_hz;
            sigPhase = pi * e.tia(:, l).' .* e.gainScale(:, l).' ./ e.vPi(:, l).' .* sqrt(varI);
            phiB = phiB + sigPhase .* randn(noiseStream, size(power));
        end
        z = meshmodel.eo_activation(z, alpha, gPhi, phiB, e.s1(:, l).', e.s2(:, l).');
    end

    intensity = abs(z) .^ 2;                       % W at each output
    regionIntensity = intensity(:, 1:nClasses);
    total = max(sum(intensity, 2), realmin);
    logits = regionIntensity ./ total * gainRd;

    if isfield(opts, 'detector') && ~isempty(opts.detector)
        det = opts.detector;
        hPlanck = 6.62607015e-34;  cLight = 299792458.0;   % exact SI constants
        ePhoton = hPlanck * cLight / lambda;
        photons = regionIntensity * det.integration_time_s / ePhoton;
        noisy = err.detector_noise(photons, det, seed);
        [~, pred] = max(noisy, [], 2);
        out.logits = noisy;
    else
        [~, pred] = max(regionIntensity, [], 2);
        out.logits = logits;
    end
    out.predictions = pred - 1;
    out.labels = labels;
    out.regionIntensity = regionIntensity;
    out.taps = taps;
    out.accuracy = mean(out.predictions == labels);
end
