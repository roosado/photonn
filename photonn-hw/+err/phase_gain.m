function params = phase_gain(params, gain)
%PHASE_GAIN Apply a systematic calibration error to every programmed phase.
%   PARAMS = ERR.PHASE_GAIN(PARAMS, GAIN) replaces every programmed phase phi by
%   GAIN*phi: a modulator calibrated so that asking for phi delivers GAIN*phi.
%   GAIN = 1 is perfect calibration. Deterministic (no seed).
%
%   This is a *calibration* error, not a setting error, and the difference is the
%   reason it is worth measuring separately from err.phase_shifter_error. That one
%   is zero-mean and independent per pixel, so its effect averages down across a
%   plate. This one is the same multiplicative bias on every pixel at once, so
%   nothing averages: it rescales the entire learned operator coherently. Two
%   errors of the same nominal size in radians are therefore not comparable, and
%   the budget should not treat them as one source.
%
%   It also has a property no other source in this budget has: it is *correctable*
%   after the fact. A gain is one number, measurable on a test pattern and divided
%   out in software before the masks are written. What the sweep measures is
%   therefore how well the calibration has to be known, not how well the hardware
%   has to behave.
%
%   Note the interaction with phase wrapping: a trained phase near +pi scaled by
%   1.1 lands past pi and wraps to a value near -pi, so a small gain error is not
%   a small perturbation everywhere. That is physical -- a plate can only deliver
%   phase modulo 2*pi -- and it is why the response is not smooth in GAIN.
%
%   GAIN is a caller-supplied magnitude; realistic values and their sources are in
%   docs/parameter_sources.md.
    params.phase_masks = params.phase_masks * gain;
end
