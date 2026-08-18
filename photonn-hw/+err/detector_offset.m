function params = detector_offset(params, sigma_px, seed)
%DETECTOR_OFFSET Displace the detector array laterally in its own plane.
%   PARAMS = ERR.DETECTOR_OFFSET(PARAMS, SIGMA_PX, SEED) draws one
%   N(0, SIGMA_PX^2) displacement in x and one in y -- there is a single detector
%   array, so this is one draw for the whole readout, not one per element -- and
%   records it in PARAMS.detector_shift_px = [dRow dCol]. model.evaluate applies
%   it by translating the output field the opposite way before integrating, which
%   is the same thing as moving the boxes and keeps the fixed detector layout
%   shared with the design side untouched.
%
%   Lateral only, on purpose. An *axial* detector misplacement is the last gap in
%   the stack, and err.plane_spacing already perturbs all L+1 gaps including that
%   one. Modelling it here as well would double-count the same displacement in two
%   sources and quietly tighten the joint budget.
%
%   Worth a prediction before the sweep: a detector patch is 14 pixels across on
%   the 128 grid, so a displacement of a pixel or so is a few per cent of a box
%   and ought to be survivable -- unlike err.mask_registration, where the same
%   displacement lands on structure with detail at the pixel scale. If that turns
%   out to be wrong, the interesting quantity is where the light actually sits
%   inside a box rather than the box size.
%
%   SIGMA_PX is a caller-supplied magnitude; realistic values and their sources are
%   in docs/parameter_sources.md.
    s = RandStream('twister', 'Seed', seed);
    params.detector_shift_px = sigma_px * randn(s, 1, 2);
end
