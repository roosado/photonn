function t = mzi_matrix(theta, phi, splits, lossDb)
%MZI_MATRIX 2x2 MZI transfer matrix from coupler and phase-shifter primitives.
%   T = MESHMODEL.MZI_MATRIX(THETA, PHI) returns B*P(THETA)*B*P(PHI) with two ideal
%   50:50 couplers -- the Clements (2016) convention, identical to
%   photonn.mzi.mzi_matrix. THETA is the internal phase (it sets the splitting),
%   PHI the external phase on the top arm.
%
%   T = MESHMODEL.MZI_MATRIX(THETA, PHI, SPLITS, LOSSDB) is the as-built form:
%     SPLITS  - [s1 s2] power split ratios of the two couplers (default [0.5 0.5])
%     LOSSDB  - insertion loss of this MZI in dB (default 0), applied as an
%               amplitude factor 10^(-LOSSDB/20)
%
%   With the defaults it is unitary. With either argument set it is not, by design:
%   an imbalanced coupler makes the splitting wrong, and loss makes the block
%   sub-unitary. Both are what the mesh error budget is measuring, so
%   validate.assert_unitary is the wrong invariant here -- see the Caveats in
%   docs/tolerance_mesh.md.
    if nargin < 3 || isempty(splits), splits = [0.5 0.5]; end
    if nargin < 4 || isempty(lossDb), lossDb = 0; end

    b1 = meshmodel.beamsplitter(splits(1));
    b2 = meshmodel.beamsplitter(splits(2));
    pTheta = [exp(1i * theta), 0; 0, 1];
    pPhi   = [exp(1i * phi),   0; 0, 1];

    t = b2 * pTheta * b1 * pPhi;
    if lossDb ~= 0
        t = t * 10 ^ (-lossDb / 20);   % dB is a power ratio; the field carries half
    end
end
