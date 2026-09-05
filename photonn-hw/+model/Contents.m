% +MODEL  As-built D2NN forward simulator.
%
%   The design-side physics re-expressed on the MATLAB side so that imperfect,
%   perturbed parameters can be scored on the frozen test set. Mirrors the Python
%   forward pass (photonn/propagate.py, elements, detect, train.encode_input,
%   models.D2NN) closely enough to reproduce the ideal accuracy exactly; that
%   match is the correctness anchor for the whole error budget.
%
%   These are pure forward-model functions. Fabrication/operation error lives in
%   +err and is applied to the parameters before they reach evaluate -- never
%   here (the design/as-built boundary is one-directional, CLAUDE.md).
%
%   angular_spectrum  - Band-limited angular-spectrum propagation (lambda-parametric).
%   subpixel_shift    - Fourier translation by a non-integer number of samples.
%   encode_input      - Reconstruct the complex input field from stored [0,1] maps.
%   detector_regions  - Fixed detector layout (port of detect.default_regions).
%                       The fallback: from schema 0.3.0 the handoff carries the
%                       layout and evaluate reads it, so the two sides no longer
%                       agree only by two copies of one arithmetic being kept in
%                       step across two languages.
%   regions_from_handoff - That carried layout, in the same struct form.
%   readout           - Region-intensity readout + photon-budget reference.
%   evaluate          - Full forward pass over the test set -> accuracy/predictions.
