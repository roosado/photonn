function stats = run_montecarlo_mesh(handoff, errorConfig, nRealizations, baseSeed)
%RUN_MONTECARLO_MESH Drive Monte Carlo realizations of the as-built MZI mesh.
%   STATS = MC.RUN_MONTECARLO_MESH(HANDOFF, ERRORCONFIG, NREALIZATIONS, BASESEED)
%   applies the configured +err perturbations to the ideal mesh parameters in
%   HANDOFF over NREALIZATIONS draws, evaluates accuracy on the frozen test set
%   (via meshmodel.evaluate), and returns the same STATS contract
%   mc.run_montecarlo returns for the D2NN.
%
%   ERRORCONFIG (all fields optional; absent = source off):
%     .phase_sigma_rad            - err.phase_shifter_error (stochastic)
%     .phase_fields               - restrict that to e.g. {'theta'} or {'phi'}
%     .quant_bits                 - err.quantize
%     .delta_lambda_m             - err.mesh_wavelength_dispersion
%     .coupler_dispersion_per_nm  - split drift accompanying the above
%     .crosstalk_coupling         - err.mesh_thermal_crosstalk (nMzi-by-nMzi)
%     .coupler_epsilon            - err.coupler_imbalance (stochastic)
%     .mzi_loss_db                - err.mzi_loss insertion loss per MZI
%     .propagation_db_per_cm      - waveguide loss, with .mzi_pitch_cm
%     .mzi_pitch_cm               - column pitch, for the above
%     .detector                   - struct enabling err.detector_noise (stochastic)
%     .subset                     - optional test-set indices (speed)
%
%   A sibling of mc.run_montecarlo rather than a model_type switch inside it: that
%   function opens by reading handoff.parameters.phase_masks and every branch
%   mutates it, so generalising it would mean rewriting every line of a shipped,
%   published path. The two share mc.sweep, which is where drift would actually
%   matter.
%
%   Composition order matches the D2NN driver -- deterministic sources first,
%   stochastic last -- so a joint configuration means the same thing in both.
%   BASESEED sets a reproducible per-realization seed stream, recorded in STATS.
%
%   Note the coupler draw and the phase draw take the *same* per-realization seed
%   through two different RandStreams, exactly as the D2NN's phase and detector
%   draws do. They are never swept jointly, so the correlation does not bite; it is
%   recorded here rather than left to be rediscovered.
    lambda0 = handoff.operating_point.wavelength_m;
    nMzi = double(handoff.parameters.n_mzi);

    base = struct( ...
        'theta',    handoff.parameters.phase_theta, ...
        'phi',      handoff.parameters.phase_phi, ...
        'sigma',    handoff.parameters.sigma, ...
        'outPhase', handoff.parameters.out_phase, ...
        'splits',   repmat(0.5, 2 * nMzi, 2), ...
        'lossDb',   zeros(2 * nMzi, 1), ...
        'wavelength_m', lambda0);

    acc = zeros(nRealizations, 1);
    seeds = zeros(nRealizations, 1);

    for i = 1:nRealizations
        seed = baseSeed + i - 1;
        seeds(i) = seed;

        params = base;

        % -- deterministic device errors --
        if has(errorConfig, 'quant_bits')
            params = err.quantize(params, errorConfig.quant_bits);
        end
        if has(errorConfig, 'delta_lambda_m')
            cd = 0;
            if has(errorConfig, 'coupler_dispersion_per_nm')
                cd = errorConfig.coupler_dispersion_per_nm;
            end
            params = err.mesh_wavelength_dispersion(params, errorConfig.delta_lambda_m, cd);
        end
        if has(errorConfig, 'crosstalk_coupling')
            params = err.mesh_thermal_crosstalk(params, errorConfig.crosstalk_coupling);
        end
        if has(errorConfig, 'mzi_loss_db')
            pdb = 0; pitch = 0;
            if has(errorConfig, 'propagation_db_per_cm')
                pdb = errorConfig.propagation_db_per_cm;
            end
            if has(errorConfig, 'mzi_pitch_cm')
                pitch = errorConfig.mzi_pitch_cm;
            end
            params = err.mzi_loss(params, errorConfig.mzi_loss_db, pdb, pitch);
        end

        % -- stochastic device errors --
        if has(errorConfig, 'coupler_epsilon') && errorConfig.coupler_epsilon > 0
            params = err.coupler_imbalance(params, errorConfig.coupler_epsilon, seed);
        end
        if has(errorConfig, 'phase_sigma_rad') && errorConfig.phase_sigma_rad > 0
            fields = [];
            if has(errorConfig, 'phase_fields'), fields = errorConfig.phase_fields; end
            params = err.phase_shifter_error(params, errorConfig.phase_sigma_rad, seed, fields);
        end

        % -- evaluate (detector noise applied inside, if configured) --
        evalOpts = struct('params', params);
        if has(errorConfig, 'subset')
            evalOpts.subset = errorConfig.subset;
        end
        if has(errorConfig, 'detector')
            evalOpts.detector = errorConfig.detector;
            evalOpts.seed = seed;
        end

        o = meshmodel.evaluate(handoff, evalOpts);
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


function t = has(s, f)
%HAS True when struct S has a non-empty field F.
    t = isfield(s, f) && ~isempty(s.(f));
end
