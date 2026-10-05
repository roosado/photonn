function stats = run_montecarlo_deep(handoff, errorConfig, nRealizations, baseSeed)
%RUN_MONTECARLO_DEEP Monte Carlo realizations of the as-built Phase-5 deep mesh.
%   STATS = MC.RUN_MONTECARLO_DEEP(HANDOFF, ERRORCONFIG, NREALIZATIONS, BASESEED)
%   applies the configured +err perturbations to the ideal deep_mesh parameter set
%   (meshmodel.deep_params) over NREALIZATIONS draws, evaluates accuracy with
%   meshmodel.evaluate_deep, and returns the STATS contract every other driver
%   returns, so mc.sweep serves it unchanged.
%
%   ERRORCONFIG (all fields optional; absent = source off):
%     every mesh field mc.run_montecarlo_mesh takes -- applied to each mesh layer,
%       with an independent draw per layer, or only to the layers in
%     .mesh_layers         - e.g. 1 or 2 (default: all)
%     .eo_tap_sigma        - err.eo_tap_spread (stochastic)
%     .eo_bias_sigma_rad   - err.eo_bias_error (stochastic)
%     .eo_gain_epsilon     - err.eo_gain_error (systematic)
%     .eo_coupler_epsilon  - err.eo_coupler_imbalance (stochastic)
%     .eo_noise            - the activation's own detector noise (stochastic), a
%                            struct with .bandwidth_hz and .tia_noise_a_per_rthz
%     .input_power_w       - run the chip at this input power instead
%     .wavelength_reaches_activation - false applies delta_lambda_m to the mesh
%                            layers only: the control that says how much of the
%                            drift's damage is the activation's (default true)
%     .detector            - readout noise (stochastic)
%     .subset              - test-set indices
%
%   Wavelength drift reaches the activation too. Its bias phase is pi*V_b/V_pi and
%   V_pi grows with wavelength, so phi_b and g_phi both rescale as lambda0/lambda,
%   and its two couplers drift with the mesh's. A mesh has no static bias sitting
%   on a dark fringe; this device does, which is why the activation is included
%   rather than left out as "electronics".
%
%   Layer draws are seeded seed + 1e6*(l-1), so layer 2's phase error is not layer
%   1's again. Composition order matches the other drivers: deterministic first,
%   stochastic last.
    mc.validate_config(errorConfig, "deep_mesh");

    nLayers = double(handoff.parameters.n_layers);
    base = meshmodel.deep_params(handoff);
    layers = 1:nLayers;
    if has(errorConfig, 'mesh_layers'), layers = errorConfig.mesh_layers; end

    acc = zeros(nRealizations, 1);
    seeds = zeros(nRealizations, 1);
    for i = 1:nRealizations
        seed = baseSeed + i - 1;
        seeds(i) = seed;
        p = base;

        for l = layers
            L = p.layers(l);
            ls = seed + 1e6 * (l - 1);
            % -- deterministic mesh errors --
            if has(errorConfig, 'quant_bits')
                L = err.quantize(L, errorConfig.quant_bits);
            end
            if has(errorConfig, 'delta_lambda_m')
                L = err.mesh_wavelength_dispersion(L, errorConfig.delta_lambda_m, ...
                                                   getdef(errorConfig, 'coupler_dispersion_per_nm', 0));
            end
            if has(errorConfig, 'crosstalk_coupling')
                L = err.mesh_thermal_crosstalk(L, errorConfig.crosstalk_coupling);
            end
            if has(errorConfig, 'mzi_loss_db')
                L = err.mzi_loss(L, errorConfig.mzi_loss_db, ...
                                 getdef(errorConfig, 'propagation_db_per_cm', 0), ...
                                 getdef(errorConfig, 'mzi_pitch_cm', 0));
            end
            % -- stochastic mesh errors --
            if has(errorConfig, 'coupler_epsilon') && errorConfig.coupler_epsilon > 0
                L = err.coupler_imbalance(L, errorConfig.coupler_epsilon, ls);
            end
            if has(errorConfig, 'phase_sigma_rad') && errorConfig.phase_sigma_rad > 0
                L = err.phase_shifter_error(L, errorConfig.phase_sigma_rad, ls, ...
                                            getdef(errorConfig, 'phase_fields', []));
            end
            p.layers(l) = L;
        end

        % -- wavelength drift, through the activation --
        if has(errorConfig, 'delta_lambda_m') && getdef(errorConfig, 'wavelength_reaches_activation', true)
            scale = handoff.operating_point.wavelength_m / ...
                    (handoff.operating_point.wavelength_m + errorConfig.delta_lambda_m);
            p.eo.vPi = p.eo.vPi / scale;              % V_pi grows with wavelength
            cd = getdef(errorConfig, 'coupler_dispersion_per_nm', 0) * errorConfig.delta_lambda_m * 1e9;
            p.eo.s1 = min(max(p.eo.s1 + cd, 0), 1);
            p.eo.s2 = min(max(p.eo.s2 + cd, 0), 1);
        end
        % -- the activation's own errors --
        if has(errorConfig, 'eo_gain_epsilon')
            p = err.eo_gain_error(p, errorConfig.eo_gain_epsilon);
        end
        if has(errorConfig, 'eo_tap_sigma') && errorConfig.eo_tap_sigma > 0
            p = err.eo_tap_spread(p, errorConfig.eo_tap_sigma, seed + 2e6);
        end
        if has(errorConfig, 'eo_bias_sigma_rad') && errorConfig.eo_bias_sigma_rad > 0
            p = err.eo_bias_error(p, errorConfig.eo_bias_sigma_rad, seed + 3e6);
        end
        if has(errorConfig, 'eo_coupler_epsilon') && errorConfig.eo_coupler_epsilon > 0
            p = err.eo_coupler_imbalance(p, errorConfig.eo_coupler_epsilon, seed + 4e6);
        end

        evalOpts = struct('params', p, 'seed', seed);
        if has(errorConfig, 'subset'), evalOpts.subset = errorConfig.subset; end
        if has(errorConfig, 'input_power_w'), evalOpts.inputPowerW = errorConfig.input_power_w; end
        if has(errorConfig, 'eo_noise'), evalOpts.eoNoise = errorConfig.eo_noise; end
        if has(errorConfig, 'detector'), evalOpts.detector = errorConfig.detector; end
        o = meshmodel.evaluate_deep(handoff, evalOpts);
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
    t = isfield(s, f) && ~isempty(s.(f));
end

function v = getdef(s, f, d)
    if has(s, f), v = s.(f); else, v = d; end
end
