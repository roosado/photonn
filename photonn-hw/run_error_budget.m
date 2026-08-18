function results = run_error_budget(opts)
%RUN_ERROR_BUDGET Scriptable Phase-4 error budget for the trained D2NN.
%   RESULTS = RUN_ERROR_BUDGET() loads the trained-D2NN handoff, verifies the
%   ideal baseline accuracy, sweeps each fabrication/operation error source into a
%   tolerance curve, renders two confusion matrices (one ideal, one at a
%   representative degraded operating point), and computes a spatial sensitivity
%   map. All figures are saved to photonn-hw/figures/ and RESULTS is saved
%   alongside as a .mat.
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
%     .sources     - cellstr of sweeps to run (default {} = all). Anything not
%                    named is carried forward from the saved results, so a budget
%                    can be built up over several calls. A full run is ~30 min;
%                    long background MATLAB jobs get killed on this machine and a
%                    foreground call is capped, so chunking is not optional here.
%                    Names: phase, quant, wavelength, crosstalk, detector, loss,
%                    spacing, registration, phaseGain, detectorOffset, joint,
%                    confusion. The last covers both confusion matrices, which are
%                    tracked files and would otherwise be re-rendered -- and so
%                    re-timestamped -- by every chunk of a chunked run.
%
%   Sweeps 1-6 are DEVICE errors -- what is wrong inside a component. Sweeps 7-10
%   are GEOMETRY errors -- where the components sit once they are made. They are
%   reported as two halves because they are two different problems for a builder:
%   a device error is fixed when the part is made, an alignment error is set at
%   assembly and can in principle be calibrated out.
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
    sources = getdef(opts, 'sources', {});
    if ischar(sources) || isstring(sources), sources = cellstr(sources); end
    if quick, nReal = 4; subsetN = 400; gBlocks = 4; sensN = 200; end
    if ~exist(figDir, 'dir'), mkdir(figDir); end
    want = @(name) isempty(sources) || any(strcmp(name, sources));

    h = io.read_handoff(handoffPath);
    lambda0 = h.operating_point.wavelength_m;

    base = model.evaluate(h);
    idealAcc = base.accuracy;
    fprintf('=== D2NN error budget ===\nideal baseline accuracy: %.4f\n', idealAcc);
    thresh = 0.95 * idealAcc;                    % "retain >=95% of ideal"

    % ---------------- confusion matrix, no error applied -----------------
    % The reference every degraded matrix is read against: same model, same
    % figure, nothing perturbed. It is free -- BASE is already the ideal
    % full-test-set pass -- and it is what the site shows for a model's own
    % behaviour, as opposed to what fabrication does to it.
    if want('confusion')
        f = viz.confusion_matrix(base.labels, base.predictions);
        viz.save_figure(f, figDir, 'confusion_ideal.png');
    end

    rng(20260724);                               % fixed subset for the sweeps
    nTest = numel(h.test_set.labels);
    subset = sort(randperm(nTest, min(subsetN, nTest)));

    results = struct('ideal', idealAcc, 'threshold', thresh, 'handoff', handoffPath);

    % Carry forward every sweep this call is not re-running. The baseline above is
    % recomputed every time regardless -- it is one pass and it is the correctness
    % anchor, so a chunked run still fails loudly if the model underneath moved.
    outMat = fullfile(figDir, 'error_budget_results.mat');
    if ~isempty(sources) && exist(outMat, 'file')
        prev = load(outMat, 'results');
        carried = setdiff(fieldnames(prev.results), fieldnames(results));
        for i = 1:numel(carried)
            results.(carried{i}) = prev.results.(carried{i});
        end
        if abs(prev.results.ideal - idealAcc) > 1e-12
            error("runErrorBudget:baselineMoved", ...
                ["carried-forward results were measured against ideal %.6f, this " ...
                 "run's baseline is %.6f -- the model changed, re-run everything."], ...
                prev.results.ideal, idealAcc);
        end
        fprintf('carried forward: %s\n', strjoin(carried', ', '));
    end

    if want('phase')
        % ---------------- 1. phase-shifter error (stochastic) ----------------
        sig = [0 0.02 0.05 0.1 0.15 0.2 0.3 0.5];                 % rad
        acc = mc.sweep(h, arrayfun(@(s) struct('phase_sigma_rad', s, 'subset', subset), sig), nReal, 2000);
        f = viz.tolerance_curve(sig, acc, 'phase-shifter sigma (rad)');
        viz.save_figure(f, figDir, 'tolerance_phase.png');
        results.phase = mc.pack(sig, acc, thresh);
    end

    if want('quant')
        % ---------------- 2. DAC quantization (deterministic) ----------------
        bits = [12 10 8 6 5 4 3 2];
        acc = mc.sweep(h, arrayfun(@(b) struct('quant_bits', b, 'subset', subset), bits), 1, 3000);
        f = viz.tolerance_curve(bits, acc, 'DAC resolution (bits)');
        set(gca, 'XDir', 'reverse');                 % coarser -> right
        viz.save_figure(f, figDir, 'tolerance_quant.png');
        results.quant = mc.pack(bits, acc, thresh);
    end

    if want('wavelength')
        % ---------------- 3. wavelength drift (deterministic) ----------------
        dlam = [0 1 2 5 10 20 30] * 1e-9;            % m
        acc = mc.sweep(h, arrayfun(@(d) struct('delta_lambda_m', d, 'subset', subset), dlam), 1, 4000);
        f = viz.tolerance_curve(dlam * 1e9, acc, 'wavelength drift (nm)');
        viz.save_figure(f, figDir, 'tolerance_wavelength.png');
        results.wavelength = mc.pack(dlam * 1e9, acc, thresh);
    end

    if want('crosstalk')
        % ---------------- 4. thermal crosstalk (deterministic) ---------------
        blur = [0 0.25 0.5 0.75 1.0 1.5 2.0];        % Gaussian sigma, pixels
        cfgs = cell(1, numel(blur));
        for i = 1:numel(blur)
            c = struct('subset', subset);
            if blur(i) > 0, c.crosstalk_kernel = gaussKernel(blur(i)); end
            cfgs{i} = c;
        end
        acc = mc.sweep(h, cfgs, 1, 5000);
        f = viz.tolerance_curve(blur, acc, 'thermal crosstalk blur (pixels)');
        viz.save_figure(f, figDir, 'tolerance_crosstalk.png');
        results.crosstalk = mc.pack(blur, acc, thresh);
    end

    if want('detector')
        % ---------------- 5. detector noise: photon budget (stochastic) ------
        powers = [1e-3 1e-6 1e-9 1e-12 1e-13 1e-14 1e-15 1e-16];   % W (1 ms integration)
        det0 = struct('integration_time_s', 1e-3, 'read_noise_e', 2, 'adc_bits', 12);
        cfgs = cell(1, numel(powers));
        for i = 1:numel(powers)
            d = det0; d.input_power_w = powers(i);
            cfgs{i} = struct('detector', d, 'subset', subset);
        end
        acc = mc.sweep(h, cfgs, nReal, 6000);
        f = viz.tolerance_curve(powers, acc, 'input optical power (W)');
        set(gca, 'XScale', 'log', 'XDir', 'reverse');
        viz.save_figure(f, figDir, 'tolerance_detector.png');
        results.detector = mc.pack(powers, acc, thresh);
    end

    if want('loss')
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
        acc = mc.sweep(h, cfgs, nReal, 7000);
        f = viz.tolerance_curve(lossDb, acc, ...
            sprintf('insertion loss (dB/mask, x%d masks) @ 1 pW', nMasks));
        viz.save_figure(f, figDir, 'tolerance_loss.png');
        results.loss = mc.pack(lossDb, acc, thresh);
        results.loss.totalDb = lossTotalDb;
        results.loss.nMasks = nMasks;
    end

    % =================== GEOMETRY: where the parts sit ====================
    % Everything above is a device error. Nothing above asks whether the parts are
    % in the right place, and for a free-space build that is the hard part: nobody
    % assembles a five-plane stack with the gaps exactly 3.000 mm and every plate
    % registered to its neighbour.

    if want('spacing')
        % ---------------- 7. plane spacing, per-gap jitter (stochastic) -------
        % Each of the L+1 gaps is set independently, so each errs independently. The
        % range is centred on the scale the project already had a reason to care
        % about: docs/phase3_mesh.md derives a connectivity floor at 2.967 mm against
        % a 3.000 mm design, i.e. 33 um of headroom on the MEAN separation.
        spacing = [0 10 25 50 75 100 150 200 300] * 1e-6;         % m, per-gap sigma
        acc = mc.sweep(h, arrayfun(@(s) struct('spacing_sigma_m', s, 'subset', subset), ...
            spacing), nReal, 15000);
        f = viz.tolerance_curve(spacing * 1e6, acc, 'plane spacing sigma (um, per gap)');
        viz.save_figure(f, figDir, 'tolerance_spacing.png');
        results.spacing = mc.pack(spacing * 1e6, acc, thresh);

        % ---------------- 7b. is the spacing limit the connectivity floor? ----
        % A registered prediction, from geometry derived three phases ago with no error
        % model in sight. Per-gap jitter of sigma displaces the MEAN gap by
        % sigma/sqrt(L+1), so at the measured edge the mean gap should be erring by
        % about the 33 um connectivity headroom -- if that is the mechanism. Test it
        % directly by moving every gap the SAME way, which is a pure change of total
        % reach and nothing else: the cliff should sit near -33 um and there should be
        % no matching cliff on the +33 um side, because more reach is not a problem.
        sysOff = [-100 -75 -50 -40 -33 -25 -10 0 10 25 50 75 100] * 1e-6;   % m
        sysAcc = zeros(numel(sysOff), 1);
        for i = 1:numel(sysOff)
            pS = struct('phase_masks', h.parameters.phase_masks, 'wavelength_m', lambda0, ...
                'separations_m', h.geometry.layer_separations_m(:) + sysOff(i));
            sysAcc(i) = model.evaluate(h, struct('params', pS, 'subset', subset)).accuracy;
        end
        f = viz.tolerance_curve(sysOff * 1e6, sysAcc, 'systematic spacing offset (um, all gaps)');
        viz.save_figure(f, figDir, 'tolerance_spacing_systematic.png');
        results.spacingSystematic = mc.pack(sysOff * 1e6, sysAcc, thresh);
        results.spacingSystematic.connectivityFloorUm = -33;   % docs/phase3_mesh.md
        results.spacingSystematic.meanGapSigmaAtEdgeUm = NaN;  % filled below
    end

    if want('registration')
        % ---------------- 8. lateral mask registration (stochastic) -----------
        % Deliberately in the same unit as thermal crosstalk (pixels of the design
        % grid) so the two can be read against each other. They are not the same
        % quantity -- one is a blur width, one is a displacement -- but both are ways
        % of getting fine mask structure into the wrong place, and crosstalk is the
        % source that currently decides the whole budget.
        reg = [0 0.05 0.1 0.15 0.2 0.3 0.5 0.75 1.0];             % px, per-mask sigma
        acc = mc.sweep(h, arrayfun(@(s) struct('registration_sigma_px', s, 'subset', subset), ...
            reg), nReal, 16000);
        f = viz.tolerance_curve(reg, acc, 'mask registration sigma (pixels)');
        viz.save_figure(f, figDir, 'tolerance_registration.png');
        results.registration = mc.pack(reg, acc, thresh);
    end

    if want('phaseGain')
        % ---------------- 9. systematic phase gain (deterministic) ------------
        % A modulator calibrated so that asking for phi delivers k*phi. Swept as a
        % two-sided tolerance -- what a builder needs is "the calibration must be known
        % to within +/- x" -- so each magnitude takes the WORSE of k = 1+d and k = 1-d.
        % They differ: a trained phase near the wrap scaled up lands on the far side of
        % it, and scaled down does not.
        gainDev = [0 0.02 0.05 0.1 0.2 0.3 0.5 0.75];
        gainUp   = mc.sweep(h, arrayfun(@(d) struct('phase_gain', 1 + d, 'subset', subset), ...
            gainDev), 1, 17000);
        gainDown = mc.sweep(h, arrayfun(@(d) struct('phase_gain', 1 - d, 'subset', subset), ...
            gainDev), 1, 17500);
        acc = min(gainUp, gainDown);
        f = viz.tolerance_curve(gainDev * 100, acc, 'phase calibration error (%, worse sign)');
        viz.save_figure(f, figDir, 'tolerance_phase_gain.png');
        results.phaseGain = mc.pack(gainDev * 100, acc, thresh);
        results.phaseGain.accUp = gainUp(:)';
        results.phaseGain.accDown = gainDown(:)';
    end

    if want('detectorOffset')
        % ---------------- 10. lateral detector offset (stochastic) ------------
        % One draw for the whole array -- there is a single detector plane. Axial
        % offset is deliberately absent: it IS the last gap, already swept in 7.
        detOff = [0 0.5 1 1.5 2 3 4 6];                           % px sigma
        acc = mc.sweep(h, arrayfun(@(s) struct('detector_sigma_px', s, 'subset', subset), ...
            detOff), nReal, 18000);
        f = viz.tolerance_curve(detOff, acc, 'detector lateral offset sigma (pixels)');
        viz.save_figure(f, figDir, 'tolerance_detector_offset.png');
        results.detectorOffset = mc.pack(detOff, acc, thresh);
    end

    if want('joint')
        % ---------------- 11. the three geometry sources together -------------
        % Every sweep above moves one thing. A real bench has all of them at once, and
        % the budget has never asked whether its sources add up. Run the three
        % stochastic geometry errors simultaneously, each at the largest magnitude that
        % held on its own, and see whether the combination still does. If it does not,
        % the per-source edges are not a specification a builder can work to.
        gEdge = @(r) max([0, r.magnitudes(r.accMean >= thresh)]);
        jointCfg = struct( ...
            'spacing_sigma_m',       gEdge(results.spacing) * 1e-6, ...
            'registration_sigma_px', gEdge(results.registration), ...
            'detector_sigma_px',     gEdge(results.detectorOffset), ...
            'subset', subset);
        jointAcc = mc.sweep(h, {jointCfg}, nReal, 19000);
        results.geometryJoint.config = jointCfg;
        results.geometryJoint.accMean = mean(jointAcc);
        results.geometryJoint.accStd = std(jointAcc);
        results.geometryJoint.threshold = thresh;
        results.geometryJoint.holds = mean(jointAcc) >= thresh;
        fprintf(['\njoint geometry (spacing %.0f um, registration %.2f px, detector %.1f px): ' ...
                 '%.4f vs %.4f -> %s\n'], jointCfg.spacing_sigma_m * 1e6, ...
            jointCfg.registration_sigma_px, jointCfg.detector_sigma_px, ...
            mean(jointAcc), thresh, string(results.geometryJoint.holds));

        % The mean-gap displacement implied by the per-gap edge, for the connectivity
        % comparison above. sigma_mean = sigma_gap / sqrt(nGaps).
        nGaps = numel(h.geometry.layer_separations_m);
        results.spacingSystematic.meanGapSigmaAtEdgeUm = gEdge(results.spacing) / sqrt(nGaps);
    end

    if want('confusion')
        % ---------------- confusion matrix at a representative degradation ----
        % The stress point is derived from THIS model's own measured phase edge, not
        % fixed, so every model is shown at matched *relative* stress.
        %
        % It used to be a hardcoded 0.35 rad. That was representative for the shipped
        % 5-mask design (holds 0.3, fails 0.5) but is nearly twice past the 56-mask
        % candidate's failure point (holds 0.15, fails 0.2), where the matrix reads
        % 0.108 -- chance -- and shows nothing but total collapse. CONFUSION_STRESS
        % reproduces 0.35 exactly on the shipped design, so that figure is unchanged,
        % and scales to 0.175 rad on the candidate.
        CONFUSION_STRESS = 7 / 6;                    % just past the edge that still holds
        holdEdge = max(results.phase.magnitudes(results.phase.accMean >= thresh));
        sigmaC = CONFUSION_STRESS * holdEdge;
        cfgC = struct('phase_sigma_rad', sigmaC);
        pC = err.phase_shifter_error(struct('phase_masks', h.parameters.phase_masks, ...
            'wavelength_m', lambda0), sigmaC, 1);
        oC = model.evaluate(h, struct('params', pC));
        f = viz.confusion_matrix(oC.labels, oC.predictions);
        viz.save_figure(f, figDir, 'confusion_phase.png');
        results.confusion.config = cfgC; results.confusion.accuracy = oC.accuracy;
        results.confusion.sigma_rad = sigmaC;
    end

    % ---------------- spatial sensitivity map ----------------------------
    % Cost is nMasks x gBlocks^2 evaluations, so it scales with depth in a way
    % the sweeps above do not: 36 of them on the shipped 5-mask design, but 2016
    % on the 56-mask one -- about four hours, dwarfing every sweep. skipSensitivity
    % carries the previous map forward, which is correct exactly when the masks
    % have not changed and only a sweep is being re-measured.
    if skipSens && exist(outMat, 'file')
        prev = load(outMat, 'results');
        results.sensitivity = prev.results.sensitivity;
        fprintf('sensitivity map: carried forward (skipSensitivity)\n');
    else
        sens = sensitivityMap(h, subset(1:min(sensN, numel(subset))), gBlocks, 0.5, lambda0);
        f = viz.sensitivity_map(permute(sens, [2 1 3]), 'D2NN phase masks');
        viz.save_figure(f, figDir, 'sensitivity_map.png');
        results.sensitivity = sens;
    end

    save(outMat, 'results');
    viz.print_summary(results, ...
        {'phase','quant','wavelength','crosstalk','detector','loss', ...
         'spacing','registration','phaseGain','detectorOffset'});
    fprintf('\nfigures + results saved to %s\n', figDir);
end

% ===================== local helpers =====================================


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


