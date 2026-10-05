% +MESHMODEL  As-built MZI-mesh forward simulator.
%
%   The Phase-3 mesh re-expressed on the MATLAB side so that imperfect, perturbed
%   parameters can be scored on the frozen test set. Mirrors the Python forward
%   pass (photonn/mzi.py, layers.MZIMeshLayer, models.MeshNetwork) closely enough
%   to reproduce the ideal accuracy exactly; that match is the correctness anchor
%   for the mesh error budget, as it is for the D2NN in +model.
%
%   A sibling of +model rather than an extension of it: +model is phase-mask
%   shaped end to end (free-space propagation between planes, 2-D detector boxes)
%   and there is no seam a mesh plugs into. The two packages expose the same
%   evaluate() output contract, which is what lets +mc and +viz serve both.
%
%   The one structural difference from Python: mesh_matrix builds the operator
%   column by column, so a per-MZI perturbation has somewhere to go. The Python
%   side collapses the mesh to a single matrix product (mzi.reconstruct), which is
%   right for the ideal design and useless for an as-built model.
%
%   beamsplitter  - 2x2 directional coupler, split ratio parametric.
%   mzi_matrix    - 2x2 MZI block from couplers and phase shifters, loss-parametric.
%   schedule      - Clements rectangular topology: columns, pairings, layout coords.
%   mesh_matrix   - Compose one n-mode mesh operator from its per-MZI settings.
%   evaluate      - Full forward pass over the test set -> accuracy/predictions.
%
%   Phase 5 (docs/phase5_activation.md): two SVD layers with an electro-optic
%   activation between them. Power is physical here and nowhere earlier.
%   eo_activation - Williamson et al. (2020) Eq. (6), as the cross port of an MZI
%                   whose couplers may be imbalanced.
%   deep_params   - Ideal deep_mesh parameter set: one mesh set per layer, one
%                   activation device per (mode, bank).
%   evaluate_deep - Forward pass of a deep_mesh handoff, with the activation's own
%                   detector noise and the readout's.
