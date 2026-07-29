function field = angular_spectrum(field, dx, lambda, z)
%ANGULAR_SPECTRUM Band-limited angular-spectrum propagation of a field batch.
%   FIELD = MODEL.ANGULAR_SPECTRUM(FIELD, DX, LAMBDA, Z) propagates the complex
%   FIELD (N-by-N, or N-by-N-by-B batch) a distance Z (m) on a grid of pitch DX
%   (m) at wavelength LAMBDA (m). Direct port of
%   photonn/propagate.py:angular_spectrum(_transfer); band-limited per
%   Matsushima & Shimobaba, IEEE TIP 18(11):2646 (2009).
%
%   LAMBDA is a free parameter (not baked into a precomputed kernel) so the
%   wavelength-drift error model can re-propagate at a shifted wavelength.
    n = size(field, 1);
    f = [0:floor((n-1)/2), -floor(n/2):-1] / (n * dx);   % np.fft.fftfreq(n, dx)
    [FX, FY] = meshgrid(f, f);                            % 'xy' indexing

    % Transfer function H = exp(i 2pi z sqrt(1/lam^2 - fx^2 - fy^2)); the sqrt of
    % a negative real returns complex in MATLAB -> evanescent decay for z > 0.
    kz = sqrt((1 / lambda)^2 - FX.^2 - FY.^2);
    H = exp(1i * 2 * pi * z * kz);

    % Band limit (Matsushima & Shimobaba 2009): zero content beyond uLimit.
    du = 1 / (n * dx);
    uLimit = 1 / (lambda * sqrt((2 * du * z)^2 + 1));
    H = H .* (abs(FX) <= uLimit & abs(FY) <= uLimit);

    spec = fft2(ifftshift(ifftshift(field, 1), 2));       % shift spatial dims only
    field = fftshift(fftshift(ifft2(spec .* H), 1), 2);   % H broadcasts over pages
end
