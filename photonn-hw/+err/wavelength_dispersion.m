function params = wavelength_dispersion(params, delta_lambda_m)
%WAVELENGTH_DISPERSION Retune the operating wavelength and rescale mask phases.
%   PARAMS = ERR.WAVELENGTH_DISPERSION(PARAMS, DELTA_LAMBDA_M) shifts the
%   operating wavelength by DELTA_LAMBDA_M. Wavelength drift hits a diffractive
%   plate twice:
%     (1) a passive plate imprints a fixed optical-path difference, so its phase
%         scales as lambda0/lambda -- the masks are rescaled accordingly;
%     (2) free-space diffraction is wavelength dependent -- PARAMS.wavelength_m is
%         updated so model.evaluate re-propagates at the new wavelength.
%
%   Requires PARAMS.wavelength_m to hold lambda0 (the MC driver seeds it).
%   Deterministic (no seed). Magnitudes/sources: docs/parameter_sources.md.
    if ~isfield(params, 'wavelength_m') || isempty(params.wavelength_m)
        error("err:wavelength_dispersion:noWavelength", ...
            "params.wavelength_m must be set to lambda0 before calling.");
    end
    lambda0 = params.wavelength_m;
    lambda = lambda0 + delta_lambda_m;
    params.phase_masks = params.phase_masks * (lambda0 / lambda);
    params.wavelength_m = lambda;
end
