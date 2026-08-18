% +ERR  Hardware imperfection models (as-built side).
%
%   Each function perturbs an ideal, imported parameter set to model one
%   fabrication/operation imperfection. Modeled independently, then jointly.
%   Every error magnitude must trace to a published measurement, cited inline
%   (see docs/parameter_sources.md); unsourced values are marked % UNSOURCED.
%
%   The sources split in two. A DEVICE error is something wrong inside a
%   component and is fixed once the part is made. A GEOMETRY error is where the
%   parts sit once they exist: set at assembly, drifting with temperature, and in
%   principle calibratable. They are kept apart because they are different
%   problems for whoever has to build the thing.
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
%   Geometry (D2NN; a chip has no equivalent -- its layout is lithographic):
%     plane_spacing        - Independent per-gap deviation from the nominal z.
%                            The last gap is the detector's axial position, which
%                            is why detector_offset does not model it again.
%     mask_registration    - Sub-pixel lateral displacement of each plate. Shifts
%                            the transmittance exp(i*phi), not the wrapped phase.
%     phase_gain           - Systematic calibration error, phi -> k*phi. Unlike
%                            phase_shifter_error this is not zero-mean, so it does
%                            not average down across a plate.
%     detector_offset      - Lateral displacement of the detector array (one draw
%                            for the whole readout, not one per element).
%
%   The geometry sources reach the forward pass as perturbed PARAMS fields --
%   separations_m and detector_shift_px -- exactly as the device ones reach it as
%   perturbed phase_masks. model.evaluate reads them; no flag is passed to it.
%   mask_registration and detector_offset share model.subpixel_shift, which is a
%   pure sampling operation and therefore lives in +model rather than here.
%
%   The first three take either parameter set and dispatch on which phase fields
%   are present (private/phase_fields.m), so sigma stays in radians and bit depth
%   in bits across both architectures and the two tolerance tables compare.
%
%   Error modeling lives here, never in Python (boundary is one-directional).
