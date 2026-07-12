function [Best_pos, Best_score, curve] = FHO(SearchAgents, Max_iterations, upperbound, lowerbound, dimension, fitness)

if length(lowerbound) == 1
    lb = ones(1, dimension) .* lowerbound;
    ub = ones(1, dimension) .* upperbound;
else
    lb = lowerbound(:)';
    ub = upperbound(:)';
end
range_val = ub - lb;
N = SearchAgents;
d = dimension;

X = lb + rand(N, d) .* range_val;
fit = zeros(1, N);
for i = 1:N
    fit(i) = fitness(X(i, :));
end

[fbest, loc] = min(fit);
Xbest = X(loc, :);
curve = zeros(1, Max_iterations);

for t = 1:Max_iterations
    ratio = t / Max_iterations;
    [~, topIdx] = sort(fit);
    ranks = zeros(1, N);
    ranks(topIdx) = 1:N;

    half = round(N / 2);
    for i = half+1:N
        candidates = setdiff(1:N, i);
        sel = candidates(randperm(length(candidates), 2));
        if fit(sel(1)) < fit(sel(2))
            ref = sel(1);
        else
            ref = sel(2);
        end

        contrast = tanh(2.0 * (ranks(i) - ranks(ref)) / N);
        contrast = sign(contrast) * max(abs(contrast), 0.15);

        X_new = X(i, :) + contrast .* rand(1, d) .* (X(ref, :) - X(i, :));
        attract_w = 0.3 * ratio;
        X_new = X_new + attract_w * rand() * (Xbest - X(i, :));
        X_new = max(min(X_new, ub), lb);

        F_new = fitness(X_new);
        if F_new < fit(i)
            X(i, :) = X_new;
            fit(i) = F_new;
        end
    end

    for i = 1:N
        j1 = randi(N); while j1 == i, j1 = randi(N); end
        j2 = randi(N); while j2 == i || j2 == j1, j2 = randi(N); end
        if fit(j1) < fit(j2)
            better = j1; worse = j2;
        else
            better = j2; worse = j1;
        end

        rank_gap = abs(ranks(worse) - ranks(better)) / N;
        step = (0.3 + 0.5 * rank_gap) * (1 - 0.3 * ratio);

        X_new = X(i, :) + step .* rand(1, d) .* (X(better, :) - X(worse, :));
        X_new = X_new + 0.1 * rand() * (Xbest - X(i, :));
        X_new = max(min(X_new, ub), lb);

        F_new = fitness(X_new);
        if F_new < fit(i)
            X(i, :) = X_new;
            fit(i) = F_new;
        end
    end

    dim_div = std(X, 0, 1) ./ (range_val + 1e-30);

    for i = 1:N
        cands = setdiff(1:N, i);
        sel = cands(randperm(length(cands), 3));
        [~, sord] = sort(fit(sel));
        best_s  = sel(sord(1));
        mid_s   = sel(sord(2));
        worst_s = sel(sord(3));

        base = (2 * X(best_s, :) + X(mid_s, :)) / 3;
        dir_vec = X(best_s, :) - X(worst_s, :);
        F_rank = (0.3 + 0.5 * (ranks(i) / N)) * (1 - 0.3 * ratio);
        V = base + F_rank .* dir_vec;

        perturb_prob = (1 - dim_div) .* (0.4 + 0.4 * (ranks(i) / N));
        perturb_prob = max(perturb_prob, 0.1);
        mask = rand(1, d) < perturb_prob;
        if ~any(mask), mask(randi(d)) = true; end

        X_new = X(i, :);
        X_new(mask) = V(mask);
        X_new = max(min(X_new, ub), lb);

        F_new = fitness(X_new);
        if F_new < fit(i)
            X(i, :) = X_new;
            fit(i) = F_new;
        end
    end

    [best_t, loc_t] = min(fit);
    if best_t < fbest
        fbest = best_t;
        Xbest = X(loc_t, :);
    end
    [~, wl] = max(fit);
    if fbest < fit(wl)
        X(wl, :) = Xbest + 0.001 * randn(1, d) .* range_val;
        X(wl, :) = max(min(X(wl, :), ub), lb);
        fit(wl) = fitness(X(wl, :));
        if fit(wl) < fbest
            fbest = fit(wl);
            Xbest = X(wl, :);
        end
    end
    curve(t) = fbest;
end

Best_score = fbest;
Best_pos = Xbest;
end
