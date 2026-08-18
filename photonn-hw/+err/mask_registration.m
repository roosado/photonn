function params = mask_registration(params, sigma_px, seed)
%MASK_REGISTRATION Displace each phase plate laterally by a sub-pixel amount.
%   PARAMS = ERR.MASK_REGISTRATION(PARAMS, SIGMA_PX, SEED) draws an independent
%   N(0, SIGMA_PX^2) displacement in x and in y for every mask and translates it
%   by that amount. SIGMA_PX is in pixels of the design grid, the same unit
%   err.thermal_crosstalk uses for blur, so the two can be read against each
%   other -- which is the point, since both are ways of getting fine mask
%   structure into the wrong place.
%
%   The plate is displaced, not its phase array: the transmittance exp(i*phi) is
%   what physically moves, so that is what is translated, and the phase is
%   recovered with angle(). Translating the wrapped phase array directly would
%   ring at every 2*pi wrap, which is an artefact of the storage format and not
%   something a misplaced plate does. A displaced phase-only plate is still
%   phase-only, so discarding the interpolated modulus is correct rather than
%   convenient.
%
%   Each plate is drawn independently because each is mounted independently.
%   A common displacement of the whole stack would be a different (and much more
%   benign) error: it would translate the output, and the detector layout is fixed
%   in the same frame, so most of it would cancel.
%
%   SIGMA_PX is a caller-supplied magnitude; realistic values and their sources are
%   in docs/parameter_sources.md.
    s = RandStream('twister', 'Seed', seed);
    L = size(params.phase_masks, 3);
    offsets = sigma_px * randn(s, L, 2);           % [dRow dCol] per mask

    for k = 1:L
        t = exp(1i * params.phase_masks(:, :, k));
        t = model.subpixel_shift(t, offsets(k, 1), offsets(k, 2));
        params.phase_masks(:, :, k) = angle(t);
    end
    params.registration_offsets_px = offsets;      % recorded, for reporting
end
