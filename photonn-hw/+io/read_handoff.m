function data = read_handoff(filename)
%READ_HANDOFF Load a photonn design->as-built handoff HDF5 file into a struct.
%   DATA = IO.READ_HANDOFF(FILENAME) reads the one-directional handoff file
%   written by the Python side (photonn/export.py) and returns a struct that
%   mirrors the schema in docs/handoff_schema.md.
%
%   This function only reads. It never writes back into the Python pipeline --
%   the design/as-built boundary is one-directional by design (CLAUDE.md).
%
%   The file's schema version is checked against SUPPORTED_SCHEMAS below; an
%   unknown version is an error so the MATLAB side never silently misreads a file
%   written by a different contract. 0.2.0 completed the mesh parameter set and
%   left the d2nn layout alone, so 0.1.0 files stay readable -- which is what
%   keeps the 131 MB exports/d2nn_phase2.h5 valid without a retrain.
%
%   Note on array order: MATLAB's h5read returns datasets with dimensions
%   reversed relative to the Python (row-major) writer. A Python f32[n, N, N]
%   comes back here as N-by-N-by-n. Downstream code must account for this.

    SUPPORTED_SCHEMAS = ["0.1.0", "0.2.0", "0.3.0"];

    if ~isfile(filename)
        error("io:read_handoff:fileNotFound", "File not found: %s", filename);
    end

    schema = string(h5readatt(filename, "/", "schema_version"));
    if ~ismember(schema, SUPPORTED_SCHEMAS)
        error("io:read_handoff:schemaMismatch", ...
            "Schema version mismatch: file '%s', supported [%s].", ...
            schema, strjoin(SUPPORTED_SCHEMAS, ", "));
    end

    data = struct();
    data.schema_version = schema;
    data.description = string(h5readatt(filename, "/", "description"));

    % -- geometry --------------------------------------------------------
    data.geometry.grid_size          = h5readatt(filename, "/geometry", "grid_size");
    data.geometry.physical_extent_m  = h5readatt(filename, "/geometry", "physical_extent_m");
    data.geometry.n_layers           = h5readatt(filename, "/geometry", "n_layers");
    data.geometry.layer_separations_m = h5read(filename, "/geometry/layer_separations_m");

    % Where the detectors sit, from schema 0.3.0 on. Written as i4[nClasses, 4]
    % (y0, y1, x0, x1), half-open, and returned by h5read with dims reversed.
    % Empty for older files and for meshes, which have no spatial readout;
    % model.detector_regions derives the layout in that case, as it always did.
    try
        raw = h5read(filename, "/geometry/detector_regions");
        data.geometry.detector_regions = double(raw).';
    catch
        data.geometry.detector_regions = [];
    end

    % -- operating point -------------------------------------------------
    % Every one of these is required, and absence is an error rather than a
    % default. It used to be the other way round: a missing attribute silently
    % became NaN, 1.0 or pi. That is the worst failure available on this seam,
    % because it does not look like one -- rename pixel_pitch_m on the Python
    % side and the whole propagation goes NaN with nothing to point at; rename
    % readout_gain and MATLAB quietly substituted 1.0 for 10.0, rescaling every
    % logit by ten into a published tolerance number; rename encoding_code and
    % the input field was reconstructed under a scheme nobody trained.
    %
    % The set below mirrors photonn.export.OPERATING_POINT, which the writer now
    % enforces, so a file that reaches here without one of these did not come
    % from write_handoff and should not be read as though it had.
    op = "/operating_point";
    shared = ["wavelength_m", "readout_gain", "input_power_w", "integration_time_s"];
    perModel = struct( ...
        "d2nn", ["pixel_pitch_m", "phase_scale_rad", "input_frac", "encoding_code"], ...
        "mesh", ["n_modes", "n_classes", "sigma_gain"]);

    modelType = string(h5readatt(filename, "/parameters", "model_type"));
    if ~isfield(perModel, modelType)
        error("io:read_handoff:badModelType", "Unknown model_type '%s'.", modelType);
    end
    for name = [shared, perModel.(modelType)]
        data.operating_point.(name) = requiredAttr(filename, op, name, modelType);
    end

    % -- parameters ------------------------------------------------------
    model_type = string(h5readatt(filename, "/parameters", "model_type"));
    data.parameters.model_type = model_type;
    switch model_type
        case "d2nn"
            data.parameters.phase_masks = h5read(filename, "/parameters/phase_masks");
        case "mesh"
            data.parameters.phase_theta = h5read(filename, "/parameters/phase_theta");
            data.parameters.phase_phi   = h5read(filename, "/parameters/phase_phi");
            if schema == "0.1.0"
                % Readable, but 108 parameters short of the model: no Sigma, no output
                % phases, so the operator cannot be rebuilt and the ideal accuracy
                % cannot be reproduced. Re-export with `python -m apps.train_mesh
                % --export-only` rather than working around it downstream.
                error("io:read_handoff:meshSchemaTooOld", ...
                    ['Mesh handoff ''%s'' is schema 0.1.0, which omits sigma and the ' ...
                     'output phases. Re-export at 0.2.0 (apps.train_mesh --export-only).'], ...
                    filename);
            end
            data.parameters.sigma       = h5read(filename, "/parameters/sigma");
            data.parameters.out_phase   = h5read(filename, "/parameters/out_phase");
            % out_phase is written f64[n_meshes, n_modes] and comes back transposed
            % (h5read reverses dims); undo it so row m is mesh m, as the schema says.
            data.parameters.out_phase   = data.parameters.out_phase.';
            data.parameters.n_modes     = double(h5readatt(filename, "/parameters", "n_modes"));
            data.parameters.n_mzi       = double(h5readatt(filename, "/parameters", "n_mzi_per_mesh"));
            data.parameters.mesh_order  = string(h5readatt(filename, "/parameters", "mesh_order"));
            data.parameters.topology    = string(h5readatt(filename, "/parameters", "topology"));
        otherwise
            error("io:read_handoff:badModelType", ...
                "Unknown model_type '%s'.", model_type);
    end

    % -- frozen test set -------------------------------------------------
    data.test_set.images = h5read(filename, "/test_set/images");
    data.test_set.labels = h5read(filename, "/test_set/labels");
end


function v = requiredAttr(filename, group, name, modelType)
%REQUIREDATTR Read an HDF5 attribute, erroring with a usable message if absent.
%   Deliberately has no default. A default here is indistinguishable from a
%   correct value downstream, which is how a renamed field used to become a
%   plausible wrong answer instead of a stack trace.
    try
        v = h5readatt(filename, group, name);
    catch
        error("io:read_handoff:missingOperatingPoint", ...
            ['Handoff ''%s'' carries no ''%s/%s'', which %s models require.\n' ...
             'See photonn.export.OPERATING_POINT; re-export the model rather ' ...
             'than defaulting it here.'], filename, group, name, modelType);
    end
end
