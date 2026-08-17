% +ERR  Hardware imperfection models (as-built side).
%
%   Each function perturbs an ideal, imported parameter set to model one
%   fabrication/operation imperfection. Modeled independently, then jointly.
%   Every error magnitude must trace to a published measurement, cited inline
%   (see docs/parameter_sources.md); unsourced values are marked % UNSOURCED.
%
%   Functions (CLAUDE.md Phase-4 implementation order)
%
%   Both architectures:
%     phase_shifter_error  - Gaussian phase error, sigma per setting.
%     quantize             - Finite DAC resolution (6/8/10/12-bit).
%     detector_noise       - Shot, thermal, and ADC-quantization noise.
%
%   D2NN (phase masks):
%     loss                 - Uniform attenuation; cancels in the readout, felt
%                            only through the photon budget.
%     wavelength_dispersion- Phase rescaling + re-propagation at the new lambda.
%     thermal_crosstalk    - Shift-invariant blur kernel over the mask.
%
%   MZI mesh:
%     coupler_imbalance    - Directional-coupler deviation from 50:50.
%     mzi_loss             - Per-MZI insertion + propagation loss. Mode-dependent,
%                            so unlike the D2NN's it does NOT cancel.
%     mesh_coupling_matrix - Distance-dependent heater coupling from the layout.
%     mesh_thermal_crosstalk - Parasitic phase from that coupling matrix.
%     mesh_wavelength_dispersion - Phase rescaling + coupler split drift.
%
%   The first three take either parameter set and dispatch on which phase fields
%   are present (private/phase_fields.m), so sigma stays in radians and bit depth
%   in bits across both architectures and the two tolerance tables compare.
%
%   Error modeling lives here, never in Python (boundary is one-directional).
