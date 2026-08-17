function params = mesh_wavelength_dispersion(params, delta_lambda_m, coupler_dispersion_per_nm)
%MESH_WAVELENGTH_DISPERSION Retune a mesh off its design wavelength.
%   PARAMS = ERR.MESH_WAVELENGTH_DISPERSION(PARAMS, DELTA_LAMBDA_M) shifts the
%   operating wavelength by DELTA_LAMBDA_M. Drift hits a mesh twice, and the two
%   effects have very different sizes:
%     (1) every phase shifter imprints a fixed optical-path difference, so its
%         phase scales as lambda0/lambda -- theta, phi and the output screen are
%         rescaled. Over a 10 nm drift at 1550 nm that is 0.6%, which is small.
%     (2) a directional coupler's power split is strongly wavelength dependent,
%         and *that* is usually what limits an MZI mesh's optical bandwidth.
%
%   PARAMS = ERR.MESH_WAVELENGTH_DISPERSION(..., COUPLER_DISPERSION_PER_NM) models
%   the second: every coupler's power split moves by
%   COUPLER_DISPERSION_PER_NM * (DELTA_LAMBDA_M in nm), in the same units
%   err.coupler_imbalance uses. Passing 0 measures effect (1) alone, which is worth
%   doing once to show how little of the failure it accounts for.
%
%   That coupling is the reason this source and err.coupler_imbalance are not
%   independent: wavelength drift *is* a systematic coupler imbalance, applied to
%   every coupler in the same direction rather than drawn per device. The
%   findings in docs/tolerance_mesh.md say so rather than presenting six
%   independent sources and implying they are.
%
%   Requires PARAMS.wavelength_m to hold lambda0 (the MC driver seeds it).
%   Deterministic (no seed).
%
%   COUPLER_DISPERSION_PER_NM must trace to a published coupler
%   characterisation.  % UNSOURCED
    if ~isfield(params, 'wavelength_m') || isempty(params.wavelength_m)
        error("err:mesh_wavelength_dispersion:noWavelength", ...
            "params.wavelength_m must be set to lambda0 before calling.");
    end
    if nargin < 3 || isempty(coupler_dispersion_per_nm)
        coupler_dispersion_per_nm = 0;
    end

    lambda0 = params.wavelength_m;
    lambda = lambda0 + delta_lambda_m;
    scale = lambda0 / lambda;

    fields = phase_fields(params);
    for i = 1:numel(fields)
        params.(fields{i}) = params.(fields{i}) * scale;
    end

    if coupler_dispersion_per_nm ~= 0
        if ~isfield(params, 'splits') || isempty(params.splits)
            error("err:mesh_wavelength_dispersion:noSplits", ...
                "Coupler dispersion needs PARAMS.splits; this is a mesh-only source.");
        end
        delta_nm = delta_lambda_m * 1e9;
        params.splits = min(max(params.splits + coupler_dispersion_per_nm * delta_nm, 0), 1);
    end

    params.wavelength_m = lambda;
end
