function params = eo_tap_spread(params, sigma, seed)
%EO_TAP_SPREAD Spread in the activation's tap fraction, one draw per device.
%   PARAMS = ERR.EO_TAP_SPREAD(PARAMS, SIGMA, SEED) perturbs every activation's
%   tapped power fraction alpha to alpha + N(0, SIGMA^2), independently per (mode,
%   bank). The tap is a directional coupler, so this is err.coupler_imbalance's
%   mechanism -- an absolute error in a power split -- applied to the one coupler
%   the activation adds.
%
%   It moves two things at once, both through the same alpha: the light the
%   activation lets through (sqrt(1 - alpha)) and the phase gain (g_phi is
%   proportional to alpha, Williamson et al. 2020 Eq. 7). In the cubic tail the
%   second dominates: a 10 % relative error in alpha is a 10 % error in that mode's
%   cubic coefficient.
%
%   SIGMA must trace to a published coupler-fabrication spread.   % UNSOURCED
    s = RandStream('twister', 'Seed', seed);
    params.eo.alpha = params.eo.alpha + sigma * randn(s, size(params.eo.alpha));
    params.eo.alpha = min(max(params.eo.alpha, 0), 0.999);   % still a tap
end
