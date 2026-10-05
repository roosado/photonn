function params = eo_bias_error(params, sigma_rad, seed)
%EO_BIAS_ERROR Error in each activation's static bias phase, one draw per device.
%   PARAMS = ERR.EO_BIAS_ERROR(PARAMS, SIGMA_RAD, SEED) adds N(0, SIGMA_RAD^2) to
%   the bias phase phi_b of every activation, independently per (mode, bank). Same
%   quantity as err.phase_shifter_error -- a 1-sigma error in radians on a
%   programmed phase -- so the two edges compare directly.
%
%   Why it should bind. The trained network biases every activation at phi_b = pi,
%   the dark state, and the light then writes only a few milliradians on the
%   modulator (docs/phase5_activation.md, the operating regime). Near pi the
%   transfer is f ~ -sqrt(1-a) * (g|z|^2 + dphi)/2 * z: a bias error dphi adds a
%   *linear* leak of the same size as an equal amount of light-written phase. So the
%   edge is set by the signal, which is milliradians at the design power.
%
%   SIGMA_RAD must trace to a published modulator bias-stability figure. % UNSOURCED
    s = RandStream('twister', 'Seed', seed);
    params.eo.dPhiB = params.eo.dPhiB + sigma_rad * randn(s, size(params.eo.dPhiB));
end
