% +ERR  Hardware imperfection models (as-built side).
%
%   Each function perturbs an ideal, imported parameter set to model one
%   fabrication/operation imperfection. Modeled independently, then jointly.
%   Every error magnitude must trace to a published measurement, cited inline
%   (see docs/parameter_sources.md); unsourced values are marked % UNSOURCED.
%
%   Functions (CLAUDE.md Phase-4 implementation order)
%     phase_shifter_error  - Gaussian phase error, sigma per setting.
%     quantize             - Finite DAC resolution (6/8/10/12-bit).
%     coupler_imbalance    - Directional-coupler deviation from 50:50.
%     loss                 - Insertion loss per MZI, propagation loss per cm.
%     wavelength_dispersion- Wavelength drift and material/waveguide dispersion.
%     thermal_crosstalk    - Distance-dependent thermal coupling matrix.
%     detector_noise       - Shot, thermal, and ADC-quantization noise.
%
%   Error modeling lives here, never in Python (boundary is one-directional).
