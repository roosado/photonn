function a = subpixel_shift(a, dRow, dCol)
%SUBPIXEL_SHIFT Translate a sampled array by a non-integer number of samples.
%   A = MODEL.SUBPIXEL_SHIFT(A, DROW, DCOL) shifts A by DROW rows and DCOL
%   columns using the Fourier shift theorem, so fractional displacements are
%   exact for band-limited content rather than rounded to a sample. A may be
%   real or complex, N-by-N or N-by-N-by-B (pages shift together).
%
%   This is a pure sampling operation with no physics in it, which is why it
%   lives here rather than in +err: both a misregistered phase plate
%   (err.mask_registration) and a laterally misplaced detector plane
%   (err.detector_offset) are the same translation applied to different things.
%
%   The shift is circular -- content leaving one edge returns at the other. That
%   is the same wrap the angular-spectrum propagator already has on this grid,
%   and at the displacements of interest here (a pixel or two, against a 128-pixel
%   grid whose outer quarter is dark) it moves no appreciable energy. Displacing
%   by a large fraction of the grid would not be meaningful anyway.
    if dRow == 0 && dCol == 0, return; end

    n = size(a, 1);
    f = [0:floor((n-1)/2), -floor(n/2):-1] / n;    % cycles per sample
    [FX, FY] = meshgrid(f, f);                      % FX along columns, FY along rows

    ramp = exp(-1i * 2 * pi * (FX * dCol + FY * dRow));
    wasReal = isreal(a);
    a = ifft2(fft2(a) .* ramp);
    if wasReal, a = real(a); end
end
