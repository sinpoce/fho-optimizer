function [Best_pos, Best_score, curve, divcurve, qual] = FHO_clean(SearchAgents, Max_iterations, upperbound, lowerbound, dimension, fitness, opt)

P.mu     = 4.0;    % logistic-map chaos parameter (initialization)
P.beta   = 2.0;    % FCDL fitness-contrast gain (tanh slope)
P.cmin   = 0.15;   % FCDL minimum contrast magnitude
P.w1     = 0.3;    % FCDL elite-attraction weight (x ratio)
P.w2     = 0.1;    % FRDS elite-attraction weight
P.S0     = 0.3;    % FRDS/FRGM differential base scale
P.F      = 0.5;    % FRDS/FRGM rank-dependent differential gain
P.delta  = 0.3;    % time-decay coefficient (x ratio)
P.p0     = 0.4;    % FRGM base perturbation probability
P.p1     = 0.4;    % FRGM rank-dependent perturbation gain
P.pmin   = 0.1;    % FRGM minimum perturbation probability
P.sigma  = 0.001;  % elite worst-slot reseed scale (x range)
P.init   = 'uniform';    % 'uniform' (default) | 'chaosqo' (legacy chaotic+quasi-oppositional, kept ONLY for the E4/E6 justification experiments)
P.chaos  = 'logistic';   % chaos map when init='chaosqo': 'logistic' | 'tent' | 'chebyshev'
P.ablate = 'none';       % ablation switch: 'none'|'fcdl'|'frds'|'frgm'|'elite'
if nargin >= 7 && ~isempty(opt)
    f = fieldnames(opt); for k = 1:numel(f), P.(f{k}) = opt.(f{k}); end
end

if length(lowerbound) == 1
    lb = ones(1, dimension) .* lowerbound;
    ub = ones(1, dimension) .* upperbound;
else
    lb = lowerbound(:)'; ub = upperbound(:)';
end
range_val = ub - lb; N = SearchAgents; d = dimension;

if ~strcmp(P.init,'chaosqo')
    X = lb + rand(N, d) .* range_val; fit = zeros(1, N);
    for i = 1:N, fit(i) = fitness(X(i, :)); end
else
    X = zeros(N, d);
    chaos = 0.1 + 0.8 * rand(1, d);
    for i = 1:N
        for j = 1:d
            chaos(j) = chaos_map(P.chaos, chaos(j), P.mu);
            X(i, j) = lb(j) + chaos(j) * range_val(j);
        end
    end
    X_opp = zeros(N, d); center = (lb + ub) / 2;
    for i = 1:N
        opp = lb + ub - X(i, :);
        X_opp(i, :) = center + (opp - center) .* rand(1, d);
        X_opp(i, :) = max(min(X_opp(i, :), ub), lb);
    end
    X_all = [X; X_opp]; fit_all = zeros(1, 2 * N);
    for i = 1:2*N, fit_all(i) = fitness(X_all(i, :)); end
    [~, sI] = sort(fit_all); X = X_all(sI(1:N), :); fit = fit_all(sI(1:N));
end
[fbest, loc] = min(fit); Xbest = X(loc, :); curve = zeros(1, Max_iterations);
divcurve = zeros(1, Max_iterations);   % population diversity Div(t) for E7
dolog = nargout >= 5;                  % qualitative logging (search history etc.)
if dolog, qual.Xhist = cell(1,Max_iterations); qual.avgfit = zeros(1,Max_iterations); qual.best1d = zeros(1,Max_iterations); qual.bestfit = zeros(1,Max_iterations); end

for t = 1:Max_iterations
    ratio = t / Max_iterations;
    [~, topIdx] = sort(fit); ranks = zeros(1, N); ranks(topIdx) = 1:N;

    if ~strcmp(P.ablate,'fcdl')
    half = round(N / 2);
    for i = half+1:N
        candidates = setdiff(1:N, i);
        sel = candidates(randperm(length(candidates), 2));
        if fit(sel(1)) < fit(sel(2)), ref = sel(1); else, ref = sel(2); end
        contrast = tanh(P.beta * (ranks(i) - ranks(ref)) / N);
        contrast = sign(contrast) * max(abs(contrast), P.cmin);
        X_new = X(i, :) + contrast .* rand(1, d) .* (X(ref, :) - X(i, :));
        X_new = X_new + (P.w1 * ratio) * rand() * (Xbest - X(i, :));
        X_new = max(min(X_new, ub), lb);
        F_new = fitness(X_new);
        if F_new < fit(i), X(i, :) = X_new; fit(i) = F_new; end
    end
    end

    if ~strcmp(P.ablate,'frds')
    for i = 1:N
        j1 = randi(N); while j1 == i, j1 = randi(N); end
        j2 = randi(N); while j2 == i || j2 == j1, j2 = randi(N); end
        if fit(j1) < fit(j2), better = j1; worse = j2; else, better = j2; worse = j1; end
        rank_gap = abs(ranks(worse) - ranks(better)) / N;
        step = (P.S0 + P.F * rank_gap) * (1 - P.delta * ratio);
        X_new = X(i, :) + step .* rand(1, d) .* (X(better, :) - X(worse, :));
        X_new = X_new + P.w2 * rand() * (Xbest - X(i, :));
        X_new = max(min(X_new, ub), lb);
        F_new = fitness(X_new);
        if F_new < fit(i), X(i, :) = X_new; fit(i) = F_new; end
    end
    end

    if ~strcmp(P.ablate,'frgm')
    dim_div = std(X, 0, 1) ./ (range_val + 1e-30);
    for i = 1:N
        cands = setdiff(1:N, i);
        sel = cands(randperm(length(cands), 3));
        [~, sord] = sort(fit(sel));
        best_s = sel(sord(1)); mid_s = sel(sord(2)); worst_s = sel(sord(3));
        base = (2 * X(best_s, :) + X(mid_s, :)) / 3;
        dir_vec = X(best_s, :) - X(worst_s, :);
        F_rank = (P.S0 + P.F * (ranks(i) / N)) * (1 - P.delta * ratio);
        V = base + F_rank .* dir_vec;
        perturb_prob = (1 - dim_div) .* (P.p0 + P.p1 * (ranks(i) / N));
        perturb_prob = max(perturb_prob, P.pmin);
        mask = rand(1, d) < perturb_prob;
        if ~any(mask), mask(randi(d)) = true; end
        X_new = X(i, :); X_new(mask) = V(mask);
        X_new = max(min(X_new, ub), lb);
        F_new = fitness(X_new);
        if F_new < fit(i), X(i, :) = X_new; fit(i) = F_new; end
    end
    end

    [best_t, loc_t] = min(fit);
    if best_t < fbest, fbest = best_t; Xbest = X(loc_t, :); end
    if ~strcmp(P.ablate,'elite')
    [~, wl] = max(fit);
    if fbest < fit(wl)
        X(wl, :) = Xbest + P.sigma * randn(1, d) .* range_val;
        X(wl, :) = max(min(X(wl, :), ub), lb);
        fit(wl) = fitness(X(wl, :));
        if fit(wl) < fbest, fbest = fit(wl); Xbest = X(wl, :); end
    end
    end
    curve(t) = fbest;
    divcurve(t) = mean(mean(abs(X - median(X, 1)), 1));   % mean abs dev from per-dim median
    if dolog, qual.Xhist{t} = X; qual.avgfit(t) = mean(fit); qual.best1d(t) = Xbest(1); qual.bestfit(t) = fbest; end
end
Best_score = fbest; Best_pos = Xbest;
end

function c = chaos_map(name, c, mu)
switch name
    case 'logistic', c = mu * c * (1 - c);
    case 'tent',     if c < 0.5, c = 2*c; else, c = 2*(1-c); end; c = min(max(c,1e-6),1-1e-6);
    case 'chebyshev',c = 0.5*(cos(4*acos(max(min(2*c-1,1),-1)))+1); c = min(max(c,1e-6),1-1e-6);
    otherwise, error('FHO_clean:chaos','unknown map %s',name);
end
end
