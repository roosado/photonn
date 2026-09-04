function keys = error_sources(arch)
%ERROR_SOURCES Every errorConfig field a Monte Carlo driver recognises.
%   KEYS = MC.ERROR_SOURCES(ARCH) returns the recognised field names for ARCH,
%   which is "d2nn" or "mesh".
%
%   This list exists because "absent = source off" is the drivers' entire
%   selection mechanism, and an unrecognised field is indistinguishable from an
%   absent one. Write .phase_sigma instead of .phase_sigma_rad and the run
%   completes cleanly at every magnitude, produces a flat tolerance curve, and
%   supports the conclusion that the design is insensitive to phase error. For a
%   project whose deliverable is a tolerance document, that is the worst failure
%   available here, and it was one typo away with nothing to catch it -- the
%   valid key set existed only as a comment block, separately, in each driver.
%
%   Adding an error source means adding its key here. See MC.VALIDATE_CONFIG.

    arguments
        arch (1,1) string
    end

    shared = ["quant_bits", "delta_lambda_m", "phase_sigma_rad", "detector", "subset"];

    switch arch
        case "d2nn"
            keys = [shared, ...
                ... % device: what is wrong inside the parts
                "crosstalk_kernel", "loss_insertion_db", "loss_propagation_db_per_cm", ...
                ... % geometry: where the parts sit
                "spacing_sigma_m", "registration_sigma_px", "phase_gain", ...
                "detector_sigma_px"];
        case "mesh"
            keys = [shared, ...
                "phase_fields", "coupler_dispersion_per_nm", "crosstalk_coupling", ...
                "coupler_epsilon", "mzi_loss_db", "propagation_db_per_cm", "mzi_pitch_cm"];
        otherwise
            error("mc:error_sources:badArch", ...
                "Unknown architecture '%s'; expected 'd2nn' or 'mesh'.", arch);
    end
end
