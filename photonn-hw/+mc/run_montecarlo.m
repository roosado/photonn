function stats = run_montecarlo(handoff, errorConfig, nRealizations, baseSeed)
%RUN_MONTECARLO Drive Monte Carlo realizations of the as-built error model.
%   STATS = MC.RUN_MONTECARLO(HANDOFF, ERRORCONFIG, NREALIZATIONS, BASESEED)
%   applies the configured +err perturbations to the ideal parameters in HANDOFF
%   over NREALIZATIONS draws, evaluates accuracy on the frozen test set (via
%   model.evaluate), and returns summary statistics.
%
%   ERRORCONFIG is a struct selecting which sources are active at this operating
%   point (all fields optional; absent = source off):
%     .phase_sigma_rad            - err.phase_shifter_error (stochastic)
%     .quant_bits                 - err.quantize
%     .delta_lambda_m             - err.wavelength_dispersion
%     .crosstalk_kernel           - err.thermal_crosstalk (2-D kernel)
%     .loss_insertion_db          - err.loss (with .loss_propagation_db_per_cm)
%     .detector                   - struct enabling err.detector_noise (stochastic)
%     .subset                     - optional test-set indices (speed)
%
%   Geometry -- where the parts sit, rather than what is wrong inside them:
%     .spacing_sigma_m            - err.plane_spacing (stochastic)
%     .registration_sigma_px      - err.mask_registration (stochastic)
%     .phase_gain                 - err.phase_gain (deterministic)
%     .detector_sigma_px          - err.detector_offset (stochastic)
%
%   Deterministic sources (quantize, wavelength, crosstalk, loss, phase gain) are
%   identical across realizations; the Monte Carlo spread comes from the
%   stochastic ones (phase error, the three geometry displacements, detector
%   noise). BASESEED sets a reproducible per-realization seed stream; the seeds
%   are recorded in STATS (CLAUDE.md convention).
%
%   Each stochastic source draws from its own offset of SEED rather than sharing
%   one stream, so adding a source to a joint configuration cannot change the draw
%   another source gets -- without that, a joint run is not the sum of the
%   independent ones it is supposed to be compared against.
    lambda0 = handoff.operating_point.wavelength_m;
    baseMasks = handoff.parameters.phase_masks;
    baseSep = handoff.geometry.layer_separations_m(:);

    acc = zeros(nRealizations, 1);
    seeds = zeros(nRealizations, 1);

    for i = 1:nRealizations
        seed = baseSeed + i - 1;
        seeds(i) = seed;

        params = struct('phase_masks', baseMasks, 'wavelength_m', lambda0, ...
                        'separations_m', baseSep);

        % -- deterministic device errors --
        if isfield(errorConfig, 'quant_bits') && ~isempty(errorConfig.quant_bits)
            params = err.quantize(params, errorConfig.quant_bits);
        end
        if isfield(errorConfig, 'delta_lambda_m') && ~isempty(errorConfig.delta_lambda_m)
            params = err.wavelength_dispersion(params, errorConfig.delta_lambda_m);
        end
        if isfield(errorConfig, 'phase_gain') && ~isempty(errorConfig.phase_gain)
            params = err.phase_gain(params, errorConfig.phase_gain);
        end
        if isfield(errorConfig, 'crosstalk_kernel') && ~isempty(errorConfig.crosstalk_kernel)
            params = err.thermal_crosstalk(params, errorConfig.crosstalk_kernel);
        end
        if isfield(errorConfig, 'loss_insertion_db') && ~isempty(errorConfig.loss_insertion_db)
            pdb = 0;
            if isfield(errorConfig, 'loss_propagation_db_per_cm')
                pdb = errorConfig.loss_propagation_db_per_cm;
            end
            params = err.loss(params, errorConfig.loss_insertion_db, pdb);
        end

        % -- stochastic phase error --
        if isfield(errorConfig, 'phase_sigma_rad') && ~isempty(errorConfig.phase_sigma_rad) ...
                && errorConfig.phase_sigma_rad > 0
            params = err.phase_shifter_error(params, errorConfig.phase_sigma_rad, seed);
        end

        % -- stochastic geometry errors --
        % Mask registration goes last of the mask-touching sources: a plate is
        % written, then mounted. Blurring an already-displaced plate or displacing
        % an already-blurred one give the same answer for a shift-invariant kernel,
        % but the order should still say which happens when.
        if isfield(errorConfig, 'registration_sigma_px') && ~isempty(errorConfig.registration_sigma_px) ...
                && errorConfig.registration_sigma_px > 0
            params = err.mask_registration(params, errorConfig.registration_sigma_px, seed + 10000);
        end
        if isfield(errorConfig, 'spacing_sigma_m') && ~isempty(errorConfig.spacing_sigma_m) ...
                && errorConfig.spacing_sigma_m > 0
            params = err.plane_spacing(params, errorConfig.spacing_sigma_m, seed + 20000);
        end
        if isfield(errorConfig, 'detector_sigma_px') && ~isempty(errorConfig.detector_sigma_px) ...
                && errorConfig.detector_sigma_px > 0
            params = err.detector_offset(params, errorConfig.detector_sigma_px, seed + 30000);
        end

        % -- evaluate (detector noise applied inside, if configured) --
        evalOpts = struct('params', params);
        if isfield(errorConfig, 'subset') && ~isempty(errorConfig.subset)
            evalOpts.subset = errorConfig.subset;
        end
        if isfield(errorConfig, 'detector') && ~isempty(errorConfig.detector)
            evalOpts.detector = errorConfig.detector;
            evalOpts.seed = seed;
        end

        o = model.evaluate(handoff, evalOpts);
        acc(i) = o.accuracy;
    end

    stats.mean = mean(acc);
    stats.std = std(acc);
    stats.acc = acc;
    stats.seeds = seeds;
    stats.baseSeed = baseSeed;
    stats.nRealizations = nRealizations;
    stats.errorConfig = errorConfig;
end
