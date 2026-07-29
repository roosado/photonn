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
%   Deterministic sources (quantize, wavelength, crosstalk, loss) are identical
%   across realizations; the Monte Carlo spread comes from the stochastic sources
%   (phase error, detector noise). BASESEED sets a reproducible per-realization
%   seed stream; the seeds are recorded in STATS (CLAUDE.md convention).
    lambda0 = handoff.operating_point.wavelength_m;
    baseMasks = handoff.parameters.phase_masks;

    acc = zeros(nRealizations, 1);
    seeds = zeros(nRealizations, 1);

    for i = 1:nRealizations
        seed = baseSeed + i - 1;
        seeds(i) = seed;

        params = struct('phase_masks', baseMasks, 'wavelength_m', lambda0);

        % -- deterministic device errors --
        if isfield(errorConfig, 'quant_bits') && ~isempty(errorConfig.quant_bits)
            params = err.quantize(params, errorConfig.quant_bits);
        end
        if isfield(errorConfig, 'delta_lambda_m') && ~isempty(errorConfig.delta_lambda_m)
            params = err.wavelength_dispersion(params, errorConfig.delta_lambda_m);
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
