function params = thermal_crosstalk(params, kernel)
%THERMAL_CROSSTALK Blur each mask phase through a thermal coupling kernel.
%   PARAMS = ERR.THERMAL_CROSSTALK(PARAMS, KERNEL) convolves each phase mask with
%   KERNEL, a small normalised 2-D point-spread modelling heat from one phase
%   shifter leaking into its neighbours: the commanded phase becomes a local
%   weighted average (realised = KERNEL * commanded). KERNEL is built by the
%   caller from a coupling strength and a distance-decay length scale; this
%   shift-invariant kernel stands in for the generic full coupling matrix, which
%   would be N^2-by-N^2 for a 2-D mask. Deterministic (no seed).
%
%   Length scale / strength sources: docs/parameter_sources.md.
    kernel = kernel / sum(kernel(:));            % conserve mean phase
    for k = 1:size(params.phase_masks, 3)
        params.phase_masks(:, :, k) = conv2(params.phase_masks(:, :, k), kernel, 'same');
    end
end
