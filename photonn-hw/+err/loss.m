function params = loss(params, insertion_db, propagation_db_per_cm)
%LOSS Record per-mask insertion loss and propagation loss.
%   PARAMS = ERR.LOSS(PARAMS, INSERTION_DB, PROPAGATION_DB_PER_CM) attaches an
%   optical-loss budget to PARAMS. The masks are ideally pure phase, so loss is a
%   (near-)uniform amplitude attenuation that cancels in the power-normalised
%   readout: it does NOT change the ideal prediction. Its effect is felt only
%   through the photon budget -- fewer photons reach the detector, so shot noise
%   grows. model.evaluate applies the recorded loss to the photon count in the
%   detector-noise path (total loss = n_layers*INSERTION_DB +
%   path_length*PROPAGATION_DB_PER_CM). This loss -> shot-noise coupling is the
%   whole point of the source and is documented in docs/tolerance_d2nn.md.
%   Deterministic (no seed). Magnitudes/sources: docs/parameter_sources.md.
    params.insertion_db = insertion_db;
    params.propagation_db_per_cm = propagation_db_per_cm;
end
