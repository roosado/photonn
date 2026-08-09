function results = run_error_budget(opts)
%RUN_ERROR_BUDGET Scriptable Phase-4 error budget for the trained D2NN.
%   RESULTS = RUN_ERROR_BUDGET() loads the trained-D2NN handoff, verifies the
%   ideal baseline accuracy, sweeps each fabrication/operation error source into a
%   tolerance curve, renders a confusion matrix at a representative degraded
%   operating point, and computes a spatial sensitivity map. All figures are
%   saved to photonn-hw/figures/ and RESULTS is saved alongside as a .mat.
%
%   RESULTS = RUN_ERROR_BUDGET(OPTS) accepts fields:
%     .handoffPath - handoff .h5 (default ../exports/d2nn_phase2.h5)
%     .quick       - true for a fast reduced run (default false)
%     .nReal       - Monte Carlo realizations for stochastic sources
%     .subsetN     - test-subset size for the sweeps (speed)
%     .skipSensitivity - reuse the saved sensitivity map instead of recomputing
%                    it (it costs nMasks x gBlocks^2 evaluations and dominates
%                    the run on a deep stack); valid only when the masks are
%                    unchanged and a sweep is being re-measured
%     .figDir      - output directory for figures
%
%   This is the command-line deliverable; no App Designer GUI is required. Every
%   sweep RANGE below is a design choice; the realistic reference magnitudes and
%   their citations live in docs/parameter_sources.md and docs/tolerance_d2nn.md.
    if nargin < 1, opts = struct(); end
    here = fileparts(mfilename('fullpath'));
    addpath(here);

    handoffPath = getdef(opts, 'handoffPath', fullfile(here, '..', 'exports', 'd2nn_phase2.h5'));
    quick   = getdef(opts, 'quick', false);
    nReal   = getdef(opts, 'nReal', 10);
    subsetN = getdef(opts, 'subsetN', 800);
    gBlocks = getdef(opts, 'gBlocks', 6);
    sensN   = getdef(opts, 'sensN', 300);
    skipSens = getdef(opts, 'skipSensitivity', false);
    figDir  = getdef(opts, 'figDir', fullfile(here, 'figures'));
    if quick, nReal = 4; subsetN = 400; gBlocks = 4; sensN = 200; end
    if ~exist(figDir, 'dir'), mkdir(figDir); end

    h = io.read_handoff(handoffPath);
    lambda0 = h.operating_point.wavelength_m;

    base = model.evaluate(h);
    idealAcc = base.accuracy;
    fprintf('=== D2NN error budget ===\nideal baseline accuracy: %.4f\n', idealAcc);
    thresh = 0.95 * idealAcc;                    % "retain >=95% of ideal"

    rng(20260724);                               % fixed subset for the sweeps
    nTest = numel(h.test_set.labels);
    subset = sort(randperm(nTest, min(subsetN, nTest)));

    results = struct('ideal', idealAcc, 'threshold', thresh, 'handoff', handoffPath);

    % ---------------- 1. phase-shifter error (stochastic) ----------------
    sig = [0 0.02 0.05 0.1 0.15 0.2 0.3 0.5];                 % rad
    acc = sweep(h, arrayfun(@(s) struct('phase_sigma_rad', s, 'subset', subset), sig), nReal, 2000);
    f = viz.tolerance_curve(sig, acc, 'phase-shifter sigma (rad)');
    saveFig(f, figDir, 'tolerance_phase.png');
    results.phase = pack(sig, acc, thresh);

    % ---------------- 2. DAC quantization (deterministic) ----------------
    bits = [12 10 8 6 5 4 3 2];
    acc = sweep(h, arrayfun(@(b) struct('quant_bits', b, 'subset', subset), bits), 1, 3000);
    f = viz.tolerance_curve(bits, acc, 'DAC resolution (bits)');
    set(gca, 'XDir', 'reverse');                 % coarser -> right
    saveFig(f, figDir, 'tolerance_quant.png');
    results.quant = pack(bits, acc, thresh);

    % ---------------- 3. wavelength drift (deterministic) ----------------
    dlam = [0 1 2 5 10 20 30] * 1e-9;            % m
    acc = sweep(h, arrayfun(@(d) struct('delta_lambda_m', d, 'subset', subset), dlam), 1, 4000);
    f = viz.tolerance_curve(dlam * 1e9, acc, 'wavelength drift (nm)');
    saveFig(f, figDir, 'tolerance_wavelength.png');
    results.wavelength = pack(dlam * 1e9, acc, thresh);

    % ---------------- 4. thermal crosstalk (deterministic) ---------------
    blur = [0 0.25 0.5 0.75 1.0 1.5 2.0];        % Gaussian sigma, pixels
    cfgs = cell(1, numel(blur));
    for i = 1:numel(blur)
        c = struct('subset', subset);
        if blur(i) > 0, c.crosstalk_kernel = gaussKernel(blur(i)); end
        cfgs{i} = c;
    end
    acc = sweep(h, cfgs, 1, 5000);
    f = viz.tolerance_curve(blur, acc, 'thermal crosstalk blur (pixels)');
    saveFig(f, figDir, 'tolerance_crosstalk.png');
    results.crosstalk = pack(blur, acc, thresh);

    % ---------------- 5. detector noise: photon budget (stochastic) ------
    powers = [1e-3 1e-6 1e-9 1e-12 1e-13 1e-14 1e-15 1e-16];   % W (1 ms integration)
    det0 = struct('integration_time_s', 1e-3, 'read_noise_e', 2, 'adc_bits', 12);
    cfgs = cell(1, numel(powers));
    for i = 1:numel(powers)
        d = det0; d.input_power_w = powers(i);
        cfgs{i} = struct('detector', d, 'subset', subset);
    end
    acc = sweep(h, cfgs, nReal, 6000);
    f = viz.tolerance_curve(powers, acc, 'input optical power (W)');
    set(gca, 'XScale', 'log', 'XDir', 'reverse');
    saveFig(f, figDir, 'tolerance_detector.png');
    results.detector = pack(powers, acc, thresh);

    % ---------------- 6. optical loss at the shot-noise operating point ---
    % Loss cancels in the ideal readout; it only bites through the photon budget,
    % so it is swept at a low-power (shot-noise-limited) operating point.
    % Loss is *specified* per mask, because that is what a component datasheet
    % states -- but what bites is the TOTAL through the stack, nMasks x per-mask.
    % A range fixed in per-mask units therefore stops measuring anything as soon
    % as the stack gets deep: the old [0 1 2 ... 10] dB/mask range began at 1 dB,
    % which on the 56-mask design is 56 dB total (a factor of 400 000 in power),
    % so every non-zero point sat far below the shot-noise knee and read chance.
    % Span a fixed 0-30 dB of TOTAL loss instead, meaningful at any depth.
    nMasks = double(h.geometry.n_layers);
    lossTotalDb = [0 1 2 3 5 8 12 20 30];
    lossDb = lossTotalDb / nMasks;               % dB per mask (insertion)
    % The operating point must sit just ABOVE the shot-noise knee measured by
    % sweep 5, or the sweep starts below threshold and measures nothing. The knee
    % moves with the model: 0.1 pW for the 12k/15-epoch masks, 1 pW for the
    % 60k/40-epoch 5-mask masks. The 56-mask design captures 79 % of the input
    % photons against ~60 %, which moves its knee back down to 0.1 pW -- but at
    % 0.1 pW the zero-loss baseline is already 0.8851 against a 0.8588 bar, so the
    % sweep would measure the knee rather than the loss. 1 pW remains correct.
    % If the detector sweep's knee moves again, re-check this.
    detLow = det0; detLow.input_power_w = 1e-12;
    cfgs = arrayfun(@(L) struct('loss_insertion_db', L, 'loss_propagation_db_per_cm', 0, ...
        'detector', detLow, 'subset', subset), lossDb, 'UniformOutput', false);
    acc = sweep(h, cfgs, nReal, 7000);
    f = viz.tolerance_curve(lossDb, acc, ...
        sprintf('insertion loss (dB/mask, x%d masks) @ 1 pW', nMasks));
    saveFig(f, figDir, 'tolerance_loss.png');
    results.loss = pack(lossDb, acc, thresh);
    results.loss.totalDb = lossTotalDb;
    results.loss.nMasks = nMasks;

    % ---------------- confusion matrix at a representative degradation ----
    cfgC = struct('phase_sigma_rad', 0.35);
    pC = err.phase_shifter_error(struct('phase_masks', h.parameters.phase_masks, ...
        'wavelength_m', lambda0), 0.35, 1);
    oC = model.evaluate(h, struct('params', pC));
    f = viz.confusion_matrix(oC.labels, oC.predictions);
    saveFig(f, figDir, 'confusion_phase035.png');
    results.confusion.config = cfgC; results.confusion.accuracy = oC.accuracy;

    % ---------------- spatial sensitivity map ----------------------------
    % Cost is nMasks x gBlocks^2 evaluations, so it scales with depth in a way
    % the sweeps above do not: 36 of them on the shipped 5-mask design, but 2016
    % on the 56-mask one -- about four hours, dwarfing every sweep. skipSensitivity
    % carries the previous map forward, which is correct exactly when the masks
    % have not changed and only a sweep is being re-measured.
    outMat = fullfile(figDir, 'error_budget_results.mat');
    if skipSens && exist(outMat, 'file')
        prev = load(outMat, 'results');
        results.sensitivity = prev.results.sensitivity;
        fprintf('sensitivity map: carried forward (skipSensitivity)\n');
    else
        sens = sensitivityMap(h, subset(1:min(sensN, numel(subset))), gBlocks, 0.5, lambda0);
        f = viz.sensitivity_map(permute(sens, [2 1 3]), 'D2NN phase masks');
        saveFig(f, figDir, 'sensitivity_map.png');
        results.sensitivity = sens;
    end

    save(outMat, 'results');
    printSummary(results);
    fprintf('\nfigures + results saved to %s\n', figDir);
end

% ===================== local helpers =====================================
function acc = sweep(h, cfgs, nReal, baseSeed)
%SWEEP Run mc.run_montecarlo for each config; return nMag-by-nReal accuracy.
    if ~iscell(cfgs), cfgs = num2cell(cfgs); end
    nMag = numel(cfgs);
    acc = zeros(nMag, nReal);
    for i = 1:nMag
        st = mc.run_montecarlo(h, cfgs{i}, nReal, baseSeed + 100 * i);
        acc(i, :) = st.acc(:)';
        fprintf('  [%2d/%2d] mean acc %.4f\n', i, nMag, st.mean);
    end
end

function s = pack(mag, acc, thresh)
    s.magnitudes = mag(:)';
    s.accMean = mean(acc, 2)';
    s.accStd = std(acc, 0, 2)';
    s.threshold = thresh;
end

function sens = sensitivityMap(h, subS, g, sig, lambda0)
%SENSITIVITYMAP Accuracy drop when a fixed phase offset hits each mask block.
    N = double(h.geometry.grid_size);
    L = double(h.geometry.n_layers);
    idealSub = model.evaluate(h, struct('subset', subS)).accuracy;
    sens = zeros(N, N, L);
    bs = floor(N / g);
    for k = 1:L
        for bi = 1:g
            for bj = 1:g
                rows = (bi - 1) * bs + 1 : min(bi * bs, N);
                cols = (bj - 1) * bs + 1 : min(bj * bs, N);
                pm = h.parameters.phase_masks;
                pm(rows, cols, k) = pm(rows, cols, k) + sig;
                a = model.evaluate(h, struct('params', ...
                    struct('phase_masks', pm, 'wavelength_m', lambda0), 'subset', subS)).accuracy;
                sens(rows, cols, k) = idealSub - a;
            end
        end
    end
end

function k = gaussKernel(sigmaPx)
    r = max(1, ceil(3 * sigmaPx));
    ax = -r:r;
    [X, Y] = meshgrid(ax, ax);
    k = exp(-(X.^2 + Y.^2) / (2 * sigmaPx^2));
    k = k / sum(k(:));
end

function v = getdef(s, f, d)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end

function saveFig(fig, figDir, name)
    exportgraphics(fig, fullfile(figDir, name), 'Resolution', 130);
    close(fig);
end

function printSummary(r)
    fprintf('\n--- tolerance summary (threshold = %.4f, 95%% of ideal %.4f) ---\n', ...
        r.threshold, r.ideal);
    srcs = {'phase','quant','wavelength','crosstalk','detector','loss'};
    for i = 1:numel(srcs)
        if isfield(r, srcs{i})
            s = r.(srcs{i});
            below = find(s.accMean < s.threshold, 1);
            if isempty(below), edge = 'not reached in sweep';
            else, edge = sprintf('crosses at magnitude ~%.4g', s.magnitudes(below)); end
            fprintf('  %-11s: %s\n', srcs{i}, edge);
        end
    end
end
