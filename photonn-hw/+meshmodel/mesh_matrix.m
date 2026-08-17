function m = mesh_matrix(theta, phi, outPhase, sched, opts)
%MESH_MATRIX Compose one mesh operator from its per-MZI settings.
%   M = MESHMODEL.MESH_MATRIX(THETA, PHI, OUTPHASE, SCHED) builds the NMODES-by-
%   NMODES operator diag(exp(i*OUTPHASE)) * L_end * ... * L_1, where each column
%   L_c applies the 2x2 MZI blocks of SCHED.columns{c} on their mode pairs and
%   passes the unpaired modes straight through. Mirrors
%   photonn.layers.MZIMeshLayer.matrix.
%
%   M = MESHMODEL.MESH_MATRIX(..., OPTS) injects per-device error:
%     OPTS.splits - nMzi-by-2 coupler power splits (default 0.5 everywhere)
%     OPTS.lossDb - nMzi-by-1 insertion loss per MZI, dB (default 0)
%
%   The composition is column by column on purpose. The Python side collapses a
%   mesh to a single matrix product (mzi.reconstruct), which is correct for the
%   ideal design and leaves nowhere to put a per-MZI perturbation; that seam is
%   the reason this function exists rather than a transcription of reconstruct.
%
%   With default OPTS the result is unitary to ~1e-15. With either set it is not,
%   which is the point.
    if nargin < 5, opts = struct(); end
    n = sched.nModes;

    splits = defaulted(opts, 'splits', repmat(0.5, sched.nMzi, 2));
    lossDb = defaulted(opts, 'lossDb', zeros(sched.nMzi, 1));

    m = complex(eye(n));
    for c = 1:numel(sched.columns)
        col = sched.columns{c};
        layer = complex(eye(n));
        for j = 1:size(col, 2)
            top = col(1, j);
            k   = col(2, j);
            layer(top:top+1, top:top+1) = ...
                meshmodel.mzi_matrix(theta(k), phi(k), splits(k, :), lossDb(k));
        end
        m = layer * m;
    end
    m = diag(exp(1i * outPhase(:))) * m;
end


function v = defaulted(s, f, d)
%DEFAULTED Field F of struct S, or D when absent or empty.
    if isfield(s, f) && ~isempty(s.(f))
        v = s.(f);
    else
        v = d;
    end
end
