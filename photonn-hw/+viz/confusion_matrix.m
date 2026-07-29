function h = confusion_matrix(trueLabels, predLabels)
%CONFUSION_MATRIX Plot a confusion matrix for the as-built classifier.
%   H = VIZ.CONFUSION_MATRIX(TRUELABELS, PREDLABELS) renders the 10-class
%   confusion matrix over the frozen test set (both inputs 0-based label vectors)
%   and returns the figure handle.
    nClasses = 10;
    C = accumarray([trueLabels(:) + 1, predLabels(:) + 1], 1, [nClasses, nClasses]);

    h = figure('Color', 'w');
    try, theme(h, 'light'); catch, end
    imagesc(C);
    colormap(gca, parula); colorbar; axis square;
    set(gca, 'XTick', 1:nClasses, 'XTickLabel', 0:nClasses - 1, ...
             'YTick', 1:nClasses, 'YTickLabel', 0:nClasses - 1);
    xlabel('predicted class'); ylabel('true class');
    acc = sum(diag(C)) / sum(C(:));
    title(sprintf('Confusion matrix (accuracy %.3f)', acc));

    % overlay counts for readability
    maxC = max(C(:));
    for i = 1:nClasses
        for j = 1:nClasses
            if C(i, j) > 0
                if C(i, j) > 0.5 * maxC, col = [0 0 0]; else, col = [0.9 0.9 0.9]; end
                text(j, i, num2str(C(i, j)), 'HorizontalAlignment', 'center', ...
                    'FontSize', 7, 'Color', col);
            end
        end
    end
end
