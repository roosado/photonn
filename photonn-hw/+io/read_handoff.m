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

    SUPPORTED_SCHEMAS = ["0.1.0", "0.2.0"];

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

    % -- operating point -------------------------------------------------
    % wavelength_m is required (schema 0.1.0). The remaining scalars are extra
    % operating constants the Python export adds without a schema bump; read them
    % defensively so minimal handoffs (e.g. the mesh fixtures) still load.
    op = "/operating_point";
    data.operating_point.wavelength_m       = h5readatt(filename, op, "wavelength_m");
    data.operating_point.pixel_pitch_m      = attrOrDefault(filename, op, "pixel_pitch_m", NaN);
    data.operating_point.readout_gain       = attrOrDefault(filename, op, "readout_gain", 1.0);
    data.operating_point.phase_scale_rad    = attrOrDefault(filename, op, "phase_scale_rad", pi);
    data.operating_point.input_frac         = attrOrDefault(filename, op, "input_frac", 0.5);
    data.operating_point.encoding_code      = attrOrDefault(filename, op, "encoding_code", 2);
    data.operating_point.input_power_w      = attrOrDefault(filename, op, "input_power_w", 1e-3);
    data.operating_point.integration_time_s = attrOrDefault(filename, op, "integration_time_s", 1e-3);
    % Mesh-only. n_modes/n_classes are floats in the file: /operating_point holds
    % scalars and the Python writer coerces every one of them with float().
    data.operating_point.n_modes            = attrOrDefault(filename, op, "n_modes", NaN);
    data.operating_point.n_classes          = attrOrDefault(filename, op, "n_classes", 10);
    data.operating_point.sigma_gain         = attrOrDefault(filename, op, "sigma_gain", 1.0);

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
                    ["Mesh handoff '%s' is schema 0.1.0, which omits sigma and the " ...
                     "output phases. Re-export at 0.2.0 (apps.train_mesh --export-only)."], ...
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


function v = attrOrDefault(filename, group, name, default)
%ATTRORDEFAULT Read an HDF5 attribute, returning DEFAULT if it is absent.
    try
        v = h5readatt(filename, group, name);
    catch
        v = default;
    end
end
