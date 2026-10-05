function f = eo_activation(z, alpha, gPhi, phiB, s1, s2)
%EO_ACTIVATION Williamson et al. (2020) electro-optic activation, element-wise.
%   F = MESHMODEL.EO_ACTIVATION(Z, ALPHA, GPHI, PHIB) applies Eq. (6) of
%   Williamson, Hughes, Minkov, Bartlett, Pai & Fan, IEEE JSTQE 26(1):7700412
%   (2020), to every element of Z (a field in sqrt(W)):
%
%     f(z) = sqrt(1-alpha) * [B(s2) P(theta) B(s1)]_(2,1) * z,
%     theta = -(phiB + gPhi*|z|^2)
%
%   which with 50:50 couplers is the paper's closed form
%   j sqrt(1-alpha) exp(-j[g|z|^2 + phiB]/2) cos([g|z|^2 + phiB]/2) z. Port of
%   photonn.mzi.eo_activation, and built the way photonn.validate builds its
%   reference: the cross port of this project's own MZI, with a phase the light
%   writes on itself.
%
%   ALPHA, GPHI, PHIB may be scalars or arrays the size of Z -- one device per mode
%   is what the +err activation sources perturb.
%
%   F = MESHMODEL.EO_ACTIVATION(..., S1, S2) is the as-built form with the
%   activation MZI's two coupler power splits (default 0.5). Expanded, the cross
%   element is i*(sqrt(s2)*sqrt(1-s1)*exp(i*theta) + sqrt(1-s2)*sqrt(s1)): off 50:50
%   the two arms no longer cancel at theta = pi, so the dark state leaks a linear
%   term. At the trained operating point the light writes ~4 mrad on the modulator,
%   so that leak competes with a cubic signal that is itself ~1e-3 of the field --
%   which is why this source is in the budget.
    if nargin < 5 || isempty(s1), s1 = 0.5; end
    if nargin < 6 || isempty(s2), s2 = 0.5; end
    if any(alpha(:) < 0 | alpha(:) >= 1)
        error("meshmodel:eo_activation:alpha", "alpha is a tapped fraction in [0, 1).");
    end
    s1 = min(max(s1, 0), 1);
    s2 = min(max(s2, 0), 1);

    theta = -(phiB + gPhi .* abs(z) .^ 2);
    cross = 1i * (sqrt(s2) .* sqrt(1 - s1) .* exp(1i * theta) + sqrt(1 - s2) .* sqrt(s1));
    f = sqrt(1 - alpha) .* cross .* z;
end
