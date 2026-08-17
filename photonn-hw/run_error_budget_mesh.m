function results = run_error_budget_mesh(opts)
%RUN_ERROR_BUDGET_MESH Phase-4 fabrication error budget for the MZI mesh.
%   RESULTS = RUN_ERROR_BUDGET_MESH() scores the trained Phase-3 mesh
%   (exports/mesh_phase3.h5) against every modelled fabrication error, one source
%   at a time, and writes tolerance curves and figures to photonn-hw/figures_mesh.
%
%   The sibling of run_error_budget, which does the same for the D2NN. Both quote
%   their edges as a bracket -- "holds at X, fails at Y" -- against a threshold of
%   95% of *their own* ideal accuracy, so the two are comparable even though they
%   run on different test sets at different absolute accuracies. Never compare the
%   accuracies themselves; see docs/tolerance_mesh.md.
%
%   OPTS (all optional):
%     .handoffPath      - which handoff to score (default exports/mesh_phase3.h5)
%     .quick            - fast smoke config (nReal 4, subset 400, sensN 200)
%     .nReal            - Monte Carlo realizations for stochastic sources (10)
%     .subsetN          - test-subset size for the sweeps (800)
%     .sensN            - test-subset size inside the sensitivity map (300)
%     .sensKick         - phase offset applied to one MZI in the map, rad (0.5)
%     .skipSensitivity  - carry the map forward from a previous run in .figDir
%     .figDir           - output directory (default photonn-hw/figures_mesh)
%
%   A full run is minutes, not the D2NN's half hour: a mesh evaluation is two
%   36-by-36 matrix builds against 2000 samples, where a D2NN evaluation is 2000
%   angular-spectrum propagations on a 128^2 grid. The sensitivity map costs
%   2*nMzi evaluations (1260 here) and is still the largest single item.
%
%   UNSOURCED: every magnitude swept below is a range, not a claim about any real
%   process. The mesh has no PDK-anchored parameter set yet -- see the mesh block
%   in docs/parameter_sources.md. The edges are measured; what a fabricated device
%   would actually deliver is not yet known, and the tolerance document says so
%   rather than filling the column in.
    if nargin < 1, opts = struct(); end
    here = fileparts(mfilename('fullpath'));
    addpath(here);

    handoffPath = getdef(opts, 'handoffPath', fullfile(here, '..', 'exports', 'mesh_phase3.h5'));
    quick       = getdef(opts, 'quick', false);
    nReal       = getdef(opts, 'nReal', 10);
    subsetN     = getdef(opts, 'subsetN', 800);
    sensN       = getdef(opts, 'sensN', []);   % [] = the whole frozen test set
    sensKick    = getdef(opts, 'sensKick', 0.5);
    skipSens    = getdef(opts, 'skipSensitivity', false);
    figDir      = getdef(opts, 'figDir', fullfile(here, 'figures_mesh'));
    if quick
        nReal = 4; subsetN = 400; sensN = 150;
    end
    if ~exist(figDir, 'dir'), mkdir(figDir); end

    fprintf('=== mesh error budget ===\nhandoff: %s\n', handoffPath);
    h = io.read_handoff(handoffPath);
    nModes = double(h.parameters.n_modes);
    nMzi   = double(h.parameters.n_mzi);
    sched  = meshmodel.schedule(nModes);

    base = meshmodel.evaluate(h);          % full frozen test set, no perturbation
    idealAcc = base.accuracy;
    thresh = 0.95 * idealAcc;              % "retain >=95% of ideal", model-relative
    fprintf('ideal accuracy %.4f  ->  threshold %.4f\n', idealAcc, thresh);
    fprintf('%d modes, %d MZIs per mesh, %d MZIs total\n', nModes, nMzi, 2 * nMzi);

    rng(20260817);                         % fixed subset for the sweeps
    nTest = numel(h.test_set.labels);
    subset = sort(randperm(nTest, min(subsetN, nTest)));

    % The threshold is 95% of the *full-set* ideal, matching run_error_budget so the
    % two studies' edges mean the same thing. The sweeps run on a subset, whose own
    % ideal differs by sampling noise -- record it, because if it falls below the
    % threshold then every source "crosses at zero" and the sweep says nothing.
    idealSubset = meshmodel.evaluate(h, struct('subset', subset)).accuracy;
    fprintf('subset ideal   %.4f  (%d images)\n', idealSubset, numel(subset));
    if idealSubset < thresh
        warning('photonn_hw:run_error_budget_mesh:subsetBelowThreshold', ...
            ['Subset ideal %.4f is already below the %.4f threshold, so every ' ...
             'edge will read as zero. Raise subsetN.'], idealSubset, thresh);
    end

    results = struct('ideal', idealAcc, 'idealSubset', idealSubset, ...
                     'threshold', thresh, 'handoff', handoffPath, ...
                     'nModes', nModes, 'nMzi', nMzi, 'subsetN', numel(subset));

    f = viz.confusion_matrix(base.labels, base.predictions);
    viz.save_figure(f, figDir, 'confusion_ideal.png');

    % Seed partitions 8000-15000, clear of the D2NN budget's 2000-7000 so no draw
    % in either study collides with the other.

    % -- 1. phase-shifter error -----------------------------------------
    % Far finer than the D2NN's [0 .. 0.5] rad grid. A mesh phase error propagates
    % through every subsequent column, so the interesting range is an order of
    % magnitude tighter and a D2NN-shaped grid would have only its first point
    % above the bar.
    fprintf('\n[1/7] phase-shifter error (sigma, rad)\n');
    sig = [0 0.005 0.01 0.02 0.03 0.05 0.08 0.12];
    acc = mc.sweep(h, arrayfun(@(s) struct('phase_sigma_rad', s, 'subset', subset), sig), ...
                   nReal, 8000, @mc.run_montecarlo_mesh);
    f = viz.tolerance_curve(sig, acc, 'phase error sigma (rad)');
    viz.save_figure(f, figDir, 'tolerance_phase.png');
    results.phase = mc.pack(sig, acc, thresh);

    % -- 1b. which shifter binds ----------------------------------------
    % theta sets an MZI's splitting; phi only shifts a phase. Sweeping them apart
    % is cheap and answers a question the aggregate cannot.
    fprintf('\n[1b/7] phase error, theta only\n');
    accT = mc.sweep(h, arrayfun(@(s) struct('phase_sigma_rad', s, 'subset', subset, ...
                    'phase_fields', {{'theta'}}), sig), nReal, 8500, @mc.run_montecarlo_mesh);
    results.phaseTheta = mc.pack(sig, accT, thresh);
    fprintf('\n[1c/7] phase error, phi only\n');
    accP = mc.sweep(h, arrayfun(@(s) struct('phase_sigma_rad', s, 'subset', subset, ...
                    'phase_fields', {{'phi'}}), sig), nReal, 8700, @mc.run_montecarlo_mesh);
    results.phasePhi = mc.pack(sig, accP, thresh);

    % -- 2. DAC quantization --------------------------------------------
    fprintf('\n[2/7] DAC quantization (bits)\n');
    bits = [12 10 8 7 6 5 4 3];
    acc = mc.sweep(h, arrayfun(@(b) struct('quant_bits', b, 'subset', subset), bits), ...
                   1, 9000, @mc.run_montecarlo_mesh);
    f = viz.tolerance_curve(bits, acc, 'DAC resolution (bits)');
    set(gca, 'XDir', 'reverse');
    viz.save_figure(f, figDir, 'tolerance_quant.png');
    results.quant = mc.pack(bits, acc, thresh);

    % -- 3. coupler imbalance -------------------------------------------
    % The source the D2NN budget could not model: a phase mask has no coupler.
    fprintf('\n[3/7] coupler imbalance (epsilon, power split)\n');
    eps = [0 0.005 0.01 0.02 0.03 0.05 0.08 0.12];
    acc = mc.sweep(h, arrayfun(@(e) struct('coupler_epsilon', e, 'subset', subset), eps), ...
                   nReal, 10000, @mc.run_montecarlo_mesh);
    f = viz.tolerance_curve(eps, acc, 'coupler split error (1-sigma, power fraction)');
    viz.save_figure(f, figDir, 'tolerance_coupler.png');
    results.coupler = mc.pack(eps, acc, thresh);

    % -- 4. per-MZI loss -------------------------------------------------
    % Swept without the detector path on purpose. In the D2NN, loss is uniform and
    % cancels in the normalised readout, so it had to be measured at the photon
    % knee to bite at all. Here it is mode-dependent -- a Clements brick routes
    % edge modes through fewer MZIs than middle ones -- so it tilts the realised
    % operator and shows up with an ideal detector.
    fprintf('\n[4/7] per-MZI insertion loss (dB per MZI)\n');
    lossDb = [0 0.05 0.1 0.2 0.3 0.5 0.8 1.2 2.0];
    acc = mc.sweep(h, arrayfun(@(d) struct('mzi_loss_db', d, 'subset', subset), lossDb), ...
                   1, 11000, @mc.run_montecarlo_mesh);
    f = viz.tolerance_curve(lossDb, acc, 'insertion loss (dB per MZI)');
    viz.save_figure(f, figDir, 'tolerance_loss.png');
    results.loss = mc.pack(lossDb, acc, thresh);
    results.loss.totalDbAcrossMesh = lossDb * nModes;   % a mode crosses ~nModes MZIs

    % -- 5. thermal crosstalk --------------------------------------------
    % The distance-dependent coupling matrix CLAUDE.md's Phase-4 spec asks for, and
    % which the D2NN could only approximate with a blur kernel (a full matrix over
    % a 128^2 mask would be 268 million entries; over 630 heaters it is affordable).
    % DECAY_UM and PITCH_UM are fixed here and are UNSOURCED -- the sweep is over
    % the coupling coefficient alone, so the curve's x-axis is the one quantity a
    % measurement would pin down.
    fprintf('\n[5/7] thermal crosstalk (coupling coefficient)\n');
    DECAY_UM = 50;             % thermal decay length            % UNSOURCED
    PITCH_UM = [80 40];        % [column pitch, mode pitch]      % UNSOURCED
    alpha = [0 0.0005 0.001 0.002 0.005 0.01 0.02];
    cfgs = arrayfun(@(a) struct('crosstalk_coupling', ...
        err.mesh_coupling_matrix(sched, a, DECAY_UM, PITCH_UM), 'subset', subset), ...
        alpha, 'UniformOutput', false);
    acc = mc.sweep(h, cfgs, 1, 12000, @mc.run_montecarlo_mesh);
    f = viz.tolerance_curve(alpha, acc, 'heater coupling coefficient');
    viz.save_figure(f, figDir, 'tolerance_crosstalk.png');
    results.crosstalk = mc.pack(alpha, acc, thresh);
    results.crosstalk.decay_um = DECAY_UM;
    results.crosstalk.pitch_um = PITCH_UM;

    % -- 6. wavelength drift ---------------------------------------------
    % Same nm grid as the D2NN so the two are directly comparable. Two effects at
    % once: phases rescale as lambda0/lambda, and every coupler's split drifts --
    % which makes this a *systematic* coupler imbalance and couples it to source 3.
    fprintf('\n[6/7] wavelength drift (nm)\n');
    COUPLER_DISP = 0.002;      % power split per nm              % UNSOURCED
    dlam = [0 1 2 5 10 20 30] * 1e-9;
    acc = mc.sweep(h, arrayfun(@(d) struct('delta_lambda_m', d, 'subset', subset, ...
                   'coupler_dispersion_per_nm', COUPLER_DISP), dlam), ...
                   1, 13000, @mc.run_montecarlo_mesh);
    f = viz.tolerance_curve(dlam * 1e9, acc, 'wavelength drift (nm)');
    viz.save_figure(f, figDir, 'tolerance_wavelength.png');
    results.wavelength = mc.pack(dlam * 1e9, acc, thresh);
    results.wavelength.coupler_dispersion_per_nm = COUPLER_DISP;

    % Phase-only control: how much of the failure above is the couplers, not the
    % shifters. Cheap, and it stops the curve being read as "silicon is dispersive".
    fprintf('\n[6b/7] wavelength drift, phase rescaling only\n');
    accNoC = mc.sweep(h, arrayfun(@(d) struct('delta_lambda_m', d, 'subset', subset), dlam), ...
                      1, 13500, @mc.run_montecarlo_mesh);
    results.wavelengthPhaseOnly = mc.pack(dlam * 1e9, accNoC, thresh);

    % -- 7. detector noise / photon budget --------------------------------
    fprintf('\n[7/7] detector noise (input power, W)\n');
    det0 = struct('integration_time_s', 1e-3, 'read_noise_e', 2, 'adc_bits', 12);
    powers = [1e-3 1e-6 1e-9 1e-12 1e-13 1e-14 1e-15 1e-16];
    cfgs = arrayfun(@(p) struct('detector', setfield(det0, 'input_power_w', p), ...
                    'subset', subset), powers, 'UniformOutput', false);  %#ok<SFLD>
    acc = mc.sweep(h, cfgs, nReal, 14000, @mc.run_montecarlo_mesh);
    f = viz.tolerance_curve(powers, acc, 'input power (W)');
    set(gca, 'XScale', 'log', 'XDir', 'reverse');
    viz.save_figure(f, figDir, 'tolerance_detector.png');
    results.detector = mc.pack(powers, acc, thresh);

    % -- stressed confusion matrix ----------------------------------------
    % Stress point derived from the model's own measured edge, never hardcoded --
    % a constant tuned for one model renders another at chance and reads as a
    % broken figure rather than the fragility result it is.
    CONFUSION_STRESS = 7 / 6;
    holds = results.phase.magnitudes(results.phase.accMean >= thresh);
    if isempty(holds)
        % The sweep subset's own ideal can sit below a threshold taken from the
        % full test set -- a few hundred images carry percent-level sampling
        % noise, which is most of the margin here. Fall back to the smallest
        % non-zero magnitude so the figure is still a stressed model rather than
        % an empty one, and say so.
        nz = results.phase.magnitudes(results.phase.magnitudes > 0);
        holdEdge = min(nz);
        warning('photonn_hw:run_error_budget_mesh:noHoldingMagnitude', ...
            ['No phase magnitude cleared the %.4f threshold on this subset ' ...
             '(subset ideal is below it). Stressing at %.4g rad instead.'], ...
            thresh, holdEdge);
    else
        holdEdge = max(holds);
    end
    sigmaC = CONFUSION_STRESS * holdEdge;
    pC = err.phase_shifter_error(baseParams(h), sigmaC, 1);
    oC = meshmodel.evaluate(h, struct('params', pC));
    f = viz.confusion_matrix(oC.labels, oC.predictions);
    viz.save_figure(f, figDir, 'confusion_phase.png');
    results.confusion = struct('sigma_rad', sigmaC, 'accuracy', oC.accuracy);
    fprintf('\nstressed confusion at sigma = %.4f rad -> %.4f\n', sigmaC, oC.accuracy);

    % -- per-MZI sensitivity map -------------------------------------------
    outMat = fullfile(figDir, 'error_budget_mesh_results.mat');
    if skipSens && exist(outMat, 'file')
        prev = load(outMat, 'results');
        results.sensitivity = prev.results.sensitivity;
        fprintf('sensitivity map: carried forward (skipSensitivity)\n');
    else
        % Run on the whole frozen test set. It costs about a minute here -- a mesh
        % evaluation is two 36-by-36 builds, not 2000 propagations on a 128^2 grid --
        % where the D2NN's map is 80% of its run and had to be subsetted hard.
        if isempty(sensN)
            subS = 1:numel(h.test_set.labels);
        else
            subS = subset(1:min(sensN, numel(subset)));
        end
        fprintf('\nper-MZI sensitivity map (%d evaluations on %d images)\n', ...
                2 * nMzi, numel(subS));
        [sensLogit, sensAcc] = sensitivityMap(h, subS, sensKick, nMzi);
        f = viz.mesh_sensitivity_map(sensLogit, sched, {'V mesh', 'U mesh'}, ...
                                     'logit RMS shift');
        viz.save_figure(f, figDir, 'sensitivity_map.png');
        results.sensitivity = sensLogit;
        results.sensitivityAcc = sensAcc;
        results.sensN = numel(subS);
        results.sensKick = sensKick;
    end
    save(outMat, 'results');

    viz.print_summary(results, {'phase', 'quant', 'coupler', 'loss', ...
                                'crosstalk', 'wavelength', 'detector'});
    fprintf('\nfigures + results saved to %s\n', figDir);
end


% ===================== local helpers =====================================
function p = baseParams(h)
%BASEPARAMS The ideal, unperturbed mesh parameter set.
    nMzi = double(h.parameters.n_mzi);
    p = struct('theta', h.parameters.phase_theta, 'phi', h.parameters.phase_phi, ...
               'sigma', h.parameters.sigma, 'outPhase', h.parameters.out_phase, ...
               'splits', repmat(0.5, 2 * nMzi, 2), 'lossDb', zeros(2 * nMzi, 1), ...
               'wavelength_m', h.operating_point.wavelength_m);
end

function [sensLogit, sensAcc] = sensitivityMap(h, subS, kick, nMzi)
%SENSITIVITYMAP How much one detuned MZI moves the network, per MZI.
%   Perturbs theta by KICK, which spoils that MZI's splitting -- the setting a
%   fabrication error actually gets wrong. Returns two nMzi-by-2 maps (column 1 the
%   V mesh, column 2 the U): the RMS change in the class logits, and the accuracy
%   drop, both against the unperturbed pass.
%
%   The published map is the **logit** one, and the reason is worth stating. The
%   D2NN's map perturbs a 21-by-21 block of a phase mask, which is a large enough
%   kick that accuracy moves well clear of its own resolution. One MZI out of 1260
%   is not: accuracy on the frozen 2000 changes in steps of 1/2000, a single MZI
%   flips a handful of images either way, and measured that way 80% of the mesh
%   reads as exactly zero with a *negative* mean -- quantisation noise, not
%   physics. The logit RMS measures the same disturbance with no floor under it.
%   The accuracy map is kept in the results so the two can be compared.
    ideal = meshmodel.evaluate(h, struct('subset', subS));
    sensLogit = zeros(nMzi, 2);
    sensAcc = zeros(nMzi, 2);
    base = baseParams(h);
    for m = 0:1
        for k = 1:nMzi
            p = base;
            idx = m * nMzi + k;
            p.theta(idx) = p.theta(idx) + kick;
            o = meshmodel.evaluate(h, struct('params', p, 'subset', subS));
            sensLogit(k, m + 1) = sqrt(mean((o.logits(:) - ideal.logits(:)) .^ 2));
            sensAcc(k, m + 1) = ideal.accuracy - o.accuracy;
        end
        fprintf('  mesh %d done\n', m + 1);
    end
end

function v = getdef(s, f, d)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
