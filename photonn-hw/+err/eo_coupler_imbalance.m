function params = eo_coupler_imbalance(params, epsilon, seed)
%EO_COUPLER_IMBALANCE Imbalance in the two couplers of each activation's MZI.
%   PARAMS = ERR.EO_COUPLER_IMBALANCE(PARAMS, EPSILON, SEED) perturbs both couplers
%   of every activation MZI to 0.5 + N(0, EPSILON^2), independently. Same quantity
%   as err.coupler_imbalance on the mesh, so the two edges compare directly.
%
%   Why it should bind harder than in the mesh. At phi_b = pi a balanced MZI is dark
%   for weak light; an unbalanced one cannot reach full extinction at any phase, so a
%   linear term of order EPSILON leaks through. The signal it competes with is the
%   cubic term, of order g|z|^2/2 ~ 1e-3 at the design power.
%
%   EPSILON must trace to a published coupler-fabrication spread.   % UNSOURCED
    s = RandStream('twister', 'Seed', seed);
    params.eo.s1 = min(max(params.eo.s1 + epsilon * randn(s, size(params.eo.s1)), 0), 1);
    params.eo.s2 = min(max(params.eo.s2 + epsilon * randn(s, size(params.eo.s2)), 0), 1);
end
