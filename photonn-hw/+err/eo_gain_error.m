function params = eo_gain_error(params, epsilon)
%EO_GAIN_ERROR A systematic calibration error on every activation's phase gain.
%   PARAMS = ERR.EO_GAIN_ERROR(PARAMS, EPSILON) scales every activation's phase gain
%   g_phi by (1 + EPSILON) -- one transimpedance gain, one responsivity or one V_pi
%   off by the same fraction everywhere, which is what a calibration error is. The
%   activation's counterpart of err.phase_gain on the D2NN. Deterministic.
%
%   In the cubic tail a common gain scale multiplies every mode's output by the same
%   factor, and one overall scale cancels in the region / total readout. So this is
%   predicted to be nearly free until the gain is large enough to leave the tail.
%
%   EPSILON must trace to a published amplifier or modulator calibration figure.
%   % UNSOURCED
    params.eo.gainScale = params.eo.gainScale * (1 + epsilon);
end
