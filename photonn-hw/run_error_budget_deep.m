function results = run_error_budget_deep(opts)
%RUN_ERROR_BUDGET_DEEP Phase-5 error budget: the activated two-layer mesh and its baseline.
%   RESULTS = RUN_ERROR_BUDGET_DEEP() scores the trained Phase-5 deep mesh
%   (exports/deep_mesh_phase5.h5: two SVD layers on 16 Fourier modes with Williamson
%   et al.'s 2020 electro-optic activation between them) and its one-layer baseline
%   (exports/mesh16_phase5.h5: one SVD layer, same input, same protocol) against
%   every error source, and writes curves, figures and a plain-text summary to
%   photonn-hw/figures_phase5.
%
%   Same discipline as run_error_budget_mesh: each model against 95% of its own
%   ideal, edges quoted as a bracket ("holds at X, fails at Y"), never interpolated,
%   never an absolute accuracy compared across models. The baseline runs through
%   the very same sweeps, so every "change" docs/phase5_activation.md quotes is one
%   protocol, two machines.
%
%   What is new against the mesh budget:
%     * every mesh source applied to both layers, then to each alone
%     * the activation's own sources: tap spread, bias error, gain calibration,
%       its MZI's coupler imbalance, and its own detector -- shot noise on the
%       tapped light plus the amplifier's input-referred noise
%     * the photon budget swept at two symbol times, because the activation responds
%       to power and the readout to energy, and the two no longer trade one-for-one
%     * the activation's sources at two input powers, because in the cubic tail the
%       signal they compete with grows with power
%
%   OPTS (all optional):
%     .deepPath, .basePath - handoffs (defaults in exports/)
%     .figDir              - output directory (default photonn-hw/figures_phase5)
%     .nReal               - realizations for stochastic sources (10)
%     .subsetN             - test images per sweep (2000, a fixed draw)
%     .sources             - cellstr: run only these, carry the rest forward from
%                            the .mat (see SOURCES below)
%
%   Seeds 29000-50999, in 1000-wide slots, one slot per stochastic sweep -- clear of
%   the D2NN (2000-7000, 15000-19000), the 36-mode mesh (8000-15000) and plan 10's
%   reservation (15000-28000). Plan 12 reserved 29000-40000; the activation's own
%   sources at two powers needed more slots than that, and the extension is recorded
%   rather than squeezed.
%
%   UNSOURCED: as in the mesh budget, every swept magnitude is a range, not a claim
%   about a real process. The activation's device values and their status are in
%   docs/parameter_sources.md, The activation.
    if nargin < 1, opts = struct(); end
    here = fileparts(mfilename('fullpath'));
    addpath(here);
    root = fullfile(here, '..');

    deepPath = getdef(opts, 'deepPath', fullfile(root, 'exports', 'deep_mesh_phase5.h5'));
    basePath = getdef(opts, 'basePath', fullfile(root, 'exports', 'mesh16_phase5.h5'));
    figDir   = getdef(opts, 'figDir', fullfile(here, 'figures_phase5'));
    nReal    = getdef(opts, 'nReal', 10);
    subsetN  = getdef(opts, 'subsetN', 2000);
    SOURCES = {'phase', 'quant', 'coupler', 'loss', 'crosstalk', 'wavelength', ...
               'power', 'tap', 'bias', 'eocoupler', 'gain', 'amp', 'sensitivity'};
    sources  = getdef(opts, 'sources', SOURCES);
    if ~exist(figDir, 'dir'), mkdir(figDir); end
    outMat = fullfile(figDir, 'error_budget_deep_results.mat');

    h = io.read_handoff(deepPath);
    b = io.read_handoff(basePath);
    sched = meshmodel.schedule(double(h.parameters.n_modes));

    % -- the anchor: nothing below means anything unless these are exact --
    ideal = meshmodel.evaluate_deep(h);
    idealB = meshmodel.evaluate(b);
    stated = double(h5readatt(deepPath, '/', 'test_acc'));
    statedB = double(h5readatt(basePath, '/', 'test_acc'));
    if ideal.accuracy ~= stated || idealB.accuracy ~= statedB
        error('run_error_budget_deep:anchor', ...
            'As-built ideal %.4f / %.4f does not reproduce the handoff''s %.4f / %.4f.', ...
            ideal.accuracy, idealB.accuracy, stated, statedB);
    end
    thresh = 0.95 * ideal.accuracy;
    threshB = 0.95 * idealB.accuracy;
    fprintf('=== Phase-5 error budget ===\n');
    fprintf('deep  ideal %.4f (anchor exact) -> bar %.4f\n', ideal.accuracy, thresh);
    fprintf('base  ideal %.4f (anchor exact) -> bar %.4f\n', idealB.accuracy, threshB);

    rng(29000);
    nTest = numel(h.test_set.labels);
    subset = sort(randperm(nTest, min(subsetN, nTest)));
    subIdeal = meshmodel.evaluate_deep(h, struct('subset', subset)).accuracy;
    subIdealB = meshmodel.evaluate(b, struct('subset', subset)).accuracy;
    fprintf('sweep subset: %d images, ideals %.4f / %.4f\n', numel(subset), subIdeal, subIdealB);

    if exist(outMat, 'file')
        prev = load(outMat, 'results');
        results = prev.results;
    else
        results = struct();
    end
    results.ideal = ideal.accuracy;  results.idealBase = idealB.accuracy;
    results.threshold = thresh;      results.thresholdBase = threshB;
    results.subsetIdeal = subIdeal;  results.subsetIdealBase = subIdealB;
    results.subsetN = numel(subset);

    f = viz.confusion_matrix(ideal.labels, ideal.predictions);
    viz.save_figure(f, figDir, 'confusion_ideal.png');

    deepMC = @mc.run_montecarlo_deep;
    baseMC = @mc.run_montecarlo_mesh;
    want = @(name) any(strcmp(sources, name));

    % ============ the mesh's own sources, both machines ===================
    if want('phase')
        fprintf('\n[phase] phase-shifter error (sigma, rad)\n');
        sig = [0 0.005 0.01 0.02 0.03 0.05 0.08 0.12];
        mk = @(s, extra) mergeStruct(struct('phase_sigma_rad', s, 'subset', subset), extra);
        aD  = mc.sweep(h, arrayfun(@(s) mk(s, struct()), sig), nReal, 29000, deepMC);
        aD1 = mc.sweep(h, arrayfun(@(s) mk(s, struct('mesh_layers', 1)), sig), nReal, 30000, deepMC);
        aD2 = mc.sweep(h, arrayfun(@(s) mk(s, struct('mesh_layers', 2)), sig), nReal, 31000, deepMC);
        aB  = mc.sweep(b, arrayfun(@(s) mk(s, struct()), sig), nReal, 33000, baseMC);
        results.phase = pair(sig, aD, aB, thresh, threshB);
        results.phaseLayer1 = mc.pack(sig, aD1, thresh);
        results.phaseLayer2 = mc.pack(sig, aD2, thresh);
        overlay(figDir, 'tolerance_phase.png', sig, {aD, aD1, aD2, aB}, ...
            {'two layers + activation', 'layer 1 only', 'layer 2 only', 'one layer'}, ...
            [thresh thresh thresh threshB], 'phase error sigma (rad)', false);
    end

    if want('quant')
        fprintf('\n[quant] DAC resolution (bits)\n');
        bits = [12 10 8 7 6 5 4 3];
        aD = mc.sweep(h, arrayfun(@(q) struct('quant_bits', q, 'subset', subset), bits), 1, 50000, deepMC);
        aB = mc.sweep(b, arrayfun(@(q) struct('quant_bits', q, 'subset', subset), bits), 1, 50000, baseMC);
        results.quant = pair(bits, aD, aB, thresh, threshB);
        overlay(figDir, 'tolerance_quant.png', bits, {aD, aB}, ...
            {'two layers + activation', 'one layer'}, [thresh threshB], 'DAC resolution (bits)', false, true);
    end

    if want('coupler')
        fprintf('\n[coupler] mesh coupler imbalance (power split sigma)\n');
        eps = [0 0.005 0.01 0.02 0.03 0.05 0.08 0.12];
        aD = mc.sweep(h, arrayfun(@(e) struct('coupler_epsilon', e, 'subset', subset), eps), nReal, 32000, deepMC);
        aB = mc.sweep(b, arrayfun(@(e) struct('coupler_epsilon', e, 'subset', subset), eps), nReal, 34000, baseMC);
        results.coupler = pair(eps, aD, aB, thresh, threshB);
        overlay(figDir, 'tolerance_coupler.png', eps, {aD, aB}, ...
            {'two layers + activation', 'one layer'}, [thresh threshB], 'mesh coupler split error (1-sigma)', false);
    end

    if want('loss')
        % Ideal detector, as in the mesh budget: this is the tilt, not the photons.
        fprintf('\n[loss] per-MZI insertion loss (dB per MZI), by layer\n');
        lossDb = [0 0.05 0.1 0.2 0.3 0.5 0.8 1.2 2.0];
        mk = @(d, extra) mergeStruct(struct('mzi_loss_db', d, 'subset', subset), extra);
        aD  = mc.sweep(h, arrayfun(@(d) mk(d, struct()), lossDb), 1, 50000, deepMC);
        aD1 = mc.sweep(h, arrayfun(@(d) mk(d, struct('mesh_layers', 1)), lossDb), 1, 50000, deepMC);
        aD2 = mc.sweep(h, arrayfun(@(d) mk(d, struct('mesh_layers', 2)), lossDb), 1, 50000, deepMC);
        aB  = mc.sweep(b, arrayfun(@(d) mk(d, struct()), lossDb), 1, 50000, baseMC);
        results.loss = pair(lossDb, aD, aB, thresh, threshB);
        results.lossLayer1 = mc.pack(lossDb, aD1, thresh);
        results.lossLayer2 = mc.pack(lossDb, aD2, thresh);
        overlay(figDir, 'tolerance_loss.png', lossDb, {aD, aD1, aD2, aB}, ...
            {'two layers + activation', 'layer 1 only', 'layer 2 only', 'one layer'}, ...
            [thresh thresh thresh threshB], 'insertion loss (dB per MZI)', false);
    end

    if want('crosstalk')
        fprintf('\n[crosstalk] thermal crosstalk (coupling coefficient)\n');
        DECAY_UM = 50;             % thermal decay length            % UNSOURCED
        PITCH_UM = [80 40];        % [column pitch, mode pitch]      % UNSOURCED
        alpha = [0 0.0005 0.001 0.002 0.005 0.01 0.02];
        cfg = @(a) struct('crosstalk_coupling', err.mesh_coupling_matrix(sched, a, DECAY_UM, PITCH_UM), ...
                          'subset', subset);
        aD = mc.sweep(h, arrayfun(cfg, alpha, 'UniformOutput', false), 1, 50000, deepMC);
        aB = mc.sweep(b, arrayfun(cfg, alpha, 'UniformOutput', false), 1, 50000, baseMC);
        results.crosstalk = pair(alpha, aD, aB, thresh, threshB);
        overlay(figDir, 'tolerance_crosstalk.png', alpha, {aD, aB}, ...
            {'two layers + activation', 'one layer'}, [thresh threshB], 'heater coupling coefficient', false);
    end

    if want('wavelength')
        fprintf('\n[wavelength] drift (nm), mesh and activation\n');
        COUPLER_DISP = 0.002;      % power split per nm              % UNSOURCED
        dlam = [0 1 2 5 10 20 30] * 1e-9;
        cfg = @(d) struct('delta_lambda_m', d, 'coupler_dispersion_per_nm', COUPLER_DISP, 'subset', subset);
        aD = mc.sweep(h, arrayfun(cfg, dlam), 1, 50000, deepMC);
        aB = mc.sweep(b, arrayfun(cfg, dlam), 1, 50000, baseMC);
        results.wavelength = pair(dlam * 1e9, aD, aB, thresh, threshB);
        overlay(figDir, 'tolerance_wavelength.png', dlam * 1e9, {aD, aB}, ...
            {'two layers + activation', 'one layer'}, [thresh threshB], 'wavelength drift (nm)', false);
    end

    % ============ the photon budget: the window ==========================
    if want('power')
        % One input per symbol: the readout integrates T and the activation's loop
        % has bandwidth 1/T. At 1 ms that is the mesh budget's own integration time;
        % at 100 ps it is the activation's design rate (Williamson 2020, Table I).
        powers = [1e-15 1e-14 1e-13 1e-12 1e-11 1e-10 1e-9 1e-8 1e-7 1e-6 1e-5 1e-4 ...
                  3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1];
        T = [1e-3 1e-10];
        slotD = [35000 37000];  slotB = [39000 41000];
        noiseless = arrayfun(@(p) meshmodel.evaluate_deep(h, struct('subset', subset, ...
                             'inputPowerW', p)).accuracy, powers);
        results.powerNoiseless = struct('magnitudes', powers, 'accMean', noiseless);
        curves = {};  labels = {};  bars = [];
        for k = 1:2
            fprintf('\n[power] input power (W), T = %g s\n', T(k));
            det = struct('integration_time_s', T(k), 'read_noise_e', 2, 'adc_bits', 12);
            eo = struct('bandwidth_hz', 1 / T(k), 'tia_noise_a_per_rthz', 0);
            cD = arrayfun(@(p) struct('input_power_w', p, 'detector', det, 'eo_noise', eo, ...
                                      'subset', subset), powers);
            cB = arrayfun(@(p) struct('detector', setfield(det, 'input_power_w', p), ...
                                      'subset', subset), powers, 'UniformOutput', false); %#ok<SFLD>
            aD = mc.sweep(h, cD, nReal, slotD(k), deepMC);
            aB = mc.sweep(b, cB, nReal, slotB(k), baseMC);
            key = sprintf('powerT%d', k);
            results.(key) = pair(powers, aD, aB, thresh, threshB);
            results.(key).T = T(k);
            curves = [curves, {aD, aB}]; %#ok<AGROW>
            labels = [labels, {sprintf('two layers + activation, T = %s', tlabel(T(k))), ...
                               sprintf('one layer, T = %s', tlabel(T(k)))}]; %#ok<AGROW>
            bars = [bars thresh threshB]; %#ok<AGROW>
        end
        curves{end + 1} = noiseless(:);
        labels{end + 1} = 'two layers + activation, no noise';
        bars(end + 1) = thresh;
        overlay(figDir, 'tolerance_power.png', powers, curves, labels, bars, ...
                'input power (W)', true);
    end

    % ============ the activation's own sources =========================
    P = [1e-3 3e-2];          % the design power, and the top of the noiseless window
    if want('tap')
        fprintf('\n[tap] tap-fraction spread (absolute, 1-sigma)\n');
        tap = [0 0.001 0.002 0.005 0.01 0.02 0.05];
        a = mc.sweep(h, arrayfun(@(s) struct('eo_tap_sigma', s, 'subset', subset), tap), nReal, 43000, deepMC);
        results.tap = mc.pack(tap, a, thresh);
        overlay(figDir, 'tolerance_eo_tap.png', tap, {a}, {'two layers + activation'}, thresh, ...
                'activation tap fraction spread (1-sigma)', false);
    end
    if want('bias')
        bias = [0 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 0.1];
        a = cell(1, 2);
        for k = 1:2
            fprintf('\n[bias] bias-phase error (rad) at %g W\n', P(k));
            a{k} = mc.sweep(h, arrayfun(@(s) struct('eo_bias_sigma_rad', s, 'input_power_w', P(k), ...
                            'subset', subset), bias), nReal, 43000 + 1000 * k, deepMC);
            results.(sprintf('biasP%d', k)) = mc.pack(bias, a{k}, thresh);
        end
        overlay(figDir, 'tolerance_eo_bias.png', bias, a, {'at 1 mW', 'at 30 mW'}, [thresh thresh], ...
                'activation bias error sigma (rad)', true);
    end
    if want('eocoupler')
        eps = [0 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2];
        a = cell(1, 2);
        for k = 1:2
            fprintf('\n[eocoupler] activation coupler imbalance at %g W\n', P(k));
            a{k} = mc.sweep(h, arrayfun(@(e) struct('eo_coupler_epsilon', e, 'input_power_w', P(k), ...
                            'subset', subset), eps), nReal, 45000 + 1000 * k, deepMC);
            results.(sprintf('eocouplerP%d', k)) = mc.pack(eps, a{k}, thresh);
        end
        overlay(figDir, 'tolerance_eo_coupler.png', eps, a, {'at 1 mW', 'at 30 mW'}, [thresh thresh], ...
                'activation coupler split error (1-sigma)', true);
    end
    if want('gain')
        fprintf('\n[gain] systematic phase-gain calibration (1 + epsilon)\n');
        g = [-0.9 -0.5 -0.2 0 0.2 0.5 1 3 10 30 100 300 1000];
        a = mc.sweep(h, arrayfun(@(e) struct('eo_gain_epsilon', e, 'subset', subset), g), 1, 50000, deepMC);
        results.gain = mc.pack(1 + g, a, thresh);
        overlay(figDir, 'tolerance_eo_gain.png', 1 + g, {a}, {'two layers + activation'}, thresh, ...
                'phase gain, multiple of design', true);
    end
    if want('amp')
        % At the design rate, readout noiseless: the activation's own detector alone.
        in = [0 0.1 0.3 1 3 10 30 100] * 1e-12;      % A/sqrt(Hz)            % UNSOURCED
        a = cell(1, 2);
        for k = 1:2
            fprintf('\n[amp] amplifier input noise (A/rtHz) at %g W, 10 GHz\n', P(k));
            cfg = @(i) struct('eo_noise', struct('bandwidth_hz', 1e10, 'tia_noise_a_per_rthz', i), ...
                              'input_power_w', P(k), 'subset', subset);
            a{k} = mc.sweep(h, arrayfun(cfg, in), nReal, 47000 + 1000 * k, deepMC);
            results.(sprintf('ampP%d', k)) = mc.pack(in, a{k}, thresh);
        end
        overlay(figDir, 'tolerance_eo_amp.png', in * 1e12, a, {'at 1 mW', 'at 30 mW'}, [thresh thresh], ...
                'amplifier input noise (pA/sqrt(Hz)), 10 GHz', true);
    end

    % ============ the light cone ======================================
    if want('sensitivity')
        fprintf('\nper-MZI sensitivity (theta + 0.5 rad, logit RMS)\n');
        kick = 0.5;
        nMzi = sched.nMzi;
        refD = meshmodel.evaluate_deep(h, struct('subset', subset));
        base = meshmodel.deep_params(h);
        sens = zeros(nMzi, 4);                     % L1 V, L1 U, L2 V, L2 U
        for l = 1:2
            for m = 0:1
                for k = 1:nMzi
                    p = base;
                    p.layers(l).theta(m * nMzi + k) = p.layers(l).theta(m * nMzi + k) + kick;
                    o = meshmodel.evaluate_deep(h, struct('params', p, 'subset', subset));
                    sens(k, 2 * (l - 1) + m + 1) = sqrt(mean((o.logits(:) - refD.logits(:)) .^ 2));
                end
            end
        end
        % The closed-form cone of finding 6 (docs/tolerance_mesh.md), at 16 modes.
        nModes = sched.nModes;  nCls = double(h.operating_point.n_classes);
        outside = sched.top - (nModes - sched.column + 1) > nCls;
        tol = 1e-10 * max(sens(:));
        results.sensitivity = sens;
        results.sensKick = kick;
        results.coneOutside = outside;
        results.zeroCount = sum(sens <= tol, 1);
        results.coneAgrees = isequal(sens(:, 4) <= tol, outside);
        fprintf('  exactly-zero MZIs per mesh [L1V L1U L2V L2U]: %s (closed form for L2 U: %d)\n', ...
                mat2str(results.zeroCount), sum(outside));
        f = viz.mesh_sensitivity_map(sens, sched, {'layer 1, V', 'layer 1, U', 'layer 2, V', 'layer 2, U'}, ...
                                     'logit RMS shift');
        viz.save_figure(f, figDir, 'sensitivity_map.png');
    end

    save(outMat, 'results');
    writeSummary(fullfile(figDir, 'summary.txt'), results);
    fprintf('\nresults -> %s\n', figDir);
end


% ===================== local helpers =====================================
function s = pair(mag, aD, aB, thresh, threshB)
%PAIR One sweep on both machines: .deep and .base in mc.pack's shape.
    s.deep = mc.pack(mag, aD, thresh);
    s.base = mc.pack(mag, aB, threshB);
end

function s = mergeStruct(a, b)
    s = a;
    for f = fieldnames(b)'
        s.(f{1}) = b.(f{1});
    end
end

function t = tlabel(T)
    if T >= 1e-3, t = sprintf('%g ms', T * 1e3); else, t = sprintf('%g ps', T * 1e12); end
end

function overlay(figDir, name, mag, curves, labels, bars, xname, logx, reverseX)
%OVERLAY Several tolerance curves on one axis, each with its own 95% bar.
%   Each model is judged against its own ideal, so each gets its own dashed bar in
%   its own colour -- the one way two curves on one plot can be read honestly.
    if nargin < 9, reverseX = false; end
    colors = [0.17 0.30 0.55; 0.80 0.40 0.10; 0.30 0.60 0.30; 0.45 0.45 0.45; ...
              0.60 0.25 0.60; 0.20 0.55 0.65; 0.10 0.10 0.10];
    f = figure('Color', 'w', 'Position', [100 100 640 420]);
    try, theme(f, 'light'); catch, end
    hold on;
    x = mag(:);
    for i = 1:numel(curves)
        c = colors(1 + mod(i - 1, size(colors, 1)), :);
        m = mean(curves{i}, 2);
        style = '-o';
        if i == numel(curves) && contains(labels{i}, 'no noise'), style = '--'; end
        plot(x, m, style, 'LineWidth', 1.6, 'Color', c, 'MarkerFaceColor', c, 'MarkerSize', 4, ...
             'DisplayName', labels{i});
    end
    [ub, iu] = unique(round(bars, 6), 'stable');
    for j = 1:numel(ub)
        c = colors(1 + mod(iu(j) - 1, size(colors, 1)), :);
        yline(ub(j), ':', 'Color', c, 'LineWidth', 1.2, 'HandleVisibility', 'off');
    end
    if logx, set(gca, 'XScale', 'log'); end
    if reverseX, set(gca, 'XDir', 'reverse'); end
    grid on; box on; ylim([0 1]);
    xlabel(xname, 'Interpreter', 'none');
    ylabel('classification accuracy');
    legend('Location', 'southwest', 'Interpreter', 'none');
    viz.save_figure(f, figDir, name);
end

function writeSummary(path, r)
%WRITESUMMARY Every sweep's mean accuracy and bracket, as plain text.
    fid = fopen(path, 'w');
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, 'Phase-5 error budget (run_error_budget_deep.m)\n');
    fprintf(fid, 'deep ideal %.4f bar %.4f | base ideal %.4f bar %.4f | sweep subset %d (ideals %.4f / %.4f)\n\n', ...
            r.ideal, r.threshold, r.idealBase, r.thresholdBase, r.subsetN, r.subsetIdeal, r.subsetIdealBase);
    names = fieldnames(r);
    for i = 1:numel(names)
        v = r.(names{i});
        if ~isstruct(v), continue; end
        if isfield(v, 'deep')
            bracketLine(fid, [names{i} ' / deep'], v.deep);
            bracketLine(fid, [names{i} ' / base'], v.base);
        elseif isfield(v, 'accMean') && isfield(v, 'threshold')
            bracketLine(fid, names{i}, v);
        elseif isfield(v, 'accMean')
            fprintf(fid, '%-22s mag %s\n%-22s acc %s\n', names{i}, mat2str(v.magnitudes, 4), '', ...
                    mat2str(v.accMean, 4));
        end
    end
    if isfield(r, 'zeroCount')
        fprintf(fid, '\nexactly-zero MZIs [L1V L1U L2V L2U]: %s; closed form (L2 U): %d; agrees: %d\n', ...
                mat2str(r.zeroCount), sum(r.coneOutside), r.coneAgrees);
    end
end

function bracketLine(fid, name, s)
    below = find(s.accMean < s.threshold, 1);
    if isempty(below)
        br = sprintf('holds at every swept value (to %g)', s.magnitudes(end));
    elseif below == 1
        br = sprintf('fails at %g (the first swept value)', s.magnitudes(1));
    else
        br = sprintf('holds at %g, fails at %g', s.magnitudes(below - 1), s.magnitudes(below));
    end
    fprintf(fid, '%-22s %s\n%-22s mag %s\n%-22s acc %s\n', name, br, '', mat2str(s.magnitudes, 4), ...
            '', mat2str(s.accMean, 4));
end

function v = getdef(s, f, d)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
