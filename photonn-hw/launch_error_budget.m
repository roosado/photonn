function launch_error_budget()
%LAUNCH_ERROR_BUDGET Open the Error Budget dashboard app.
%   LAUNCH_ERROR_BUDGET() adds this folder to the path and starts
%   ErrorBudgetApp. The .mlapp itself is authored interactively in App Designer
%   (see ErrorBudgetApp.README.md); this launcher is the scripted entry point.

    here = fileparts(mfilename("fullpath"));
    addpath(here);

    if exist("ErrorBudgetApp", "class") == 8 || exist("ErrorBudgetApp", "file") == 2
        ErrorBudgetApp();
    else
        error("photonn_hw:launch_error_budget:appMissing", ...
            ["ErrorBudgetApp.mlapp not found. Create it in App Designer " ...
             "(see ErrorBudgetApp.README.md) before launching."]);
    end
end
