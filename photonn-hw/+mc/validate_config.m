function validate_config(errorConfig, arch)
%VALIDATE_CONFIG Reject an errorConfig field no driver will act on.
%   MC.VALIDATE_CONFIG(ERRORCONFIG, ARCH) errors if ERRORCONFIG carries a field
%   that is not in MC.ERROR_SOURCES(ARCH).
%
%   The drivers select sources by field presence, so a misspelled field is
%   silently a source that never runs. This turns that into an error at the top
%   of the run, before any realizations are drawn -- which is the only place it
%   can be caught, because every downstream symptom (a clean run, a flat curve)
%   looks exactly like a correct result for a tolerant design.
%
%   Suggests the closest recognised key, since the failure this guards is
%   overwhelmingly a typo rather than an invention.

    arguments
        errorConfig struct
        arch (1,1) string
    end

    known = mc.error_sources(arch);
    present = string(fieldnames(errorConfig)).';
    unknown = setdiff(present, known);

    if isempty(unknown)
        return
    end

    lines = strings(0, 1);
    for u = unknown
        d = arrayfun(@(k) editDistance(u, k), known);
        [best, idx] = min(d);
        if best <= 6
            lines(end+1) = sprintf("  '%s' -- did you mean '%s'?", u, known(idx)); %#ok<AGROW>
        else
            lines(end+1) = sprintf("  '%s'", u); %#ok<AGROW>
        end
    end

    error("mc:validate_config:unknownField", ...
        ['errorConfig has field(s) no %s driver reads, so they would select ' ...
         'nothing and the run would report a clean result at every magnitude:\n' ...
         '%s\nRecognised: %s'], ...
        arch, strjoin(lines, newline), strjoin(sort(known), ", "));
end


function d = editDistance(a, b)
%EDITDISTANCE Levenshtein distance between two strings, for the suggestion above.
    a = char(a); b = char(b);
    m = numel(a); n = numel(b);
    prev = 0:n;
    for i = 1:m
        cur = [i, zeros(1, n)];
        for j = 1:n
            cost = ~isequal(a(i), b(j));
            cur(j + 1) = min([cur(j) + 1, prev(j + 1) + 1, prev(j) + cost]);
        end
        prev = cur;
    end
    d = prev(end);
end
