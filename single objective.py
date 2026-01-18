# mo_bwr_bmr_bmwr_zdt1_no_metrics.py

# Multi-objective BWR / BMR / BMWR on ZDT1 (n=30).

# Prints ONLY decision variables (x), objective values (f1, f2), and notes constraints (none; bounds only).

# Optionally writes full first-front solutions (variables + objectives) to CSV files.


import numpy as np

import pandas as pd


# =========================

# ZDT1 definition (n objectives = 2, n variables = 30)

# =========================

def zdt1_f(x):

    """

    ZDT1 bi-objective minimization.

    x: (n,) in [0,1]^n

    f1(x) = x1

    g(x)  = 1 + 9/(n-1) * sum_{i=2..n} x_i

    f2(x) = g(x) * (1 - sqrt(f1/g))

    """

    n = x.shape[0]

    f1 = x[0]

    g = 1.0 + 9.0 / (n - 1) * np.sum(x[1:])

    # clamp small numeric issues

    ratio = f1 / g if g > 0 else 0.0

    ratio = min(max(ratio, 0.0), 1.0)

    f2 = g * (1.0 - np.sqrt(ratio))

    return np.array([f1, f2])



# =========================

# NSGA-II style helpers (dominance, sorting, crowding, selection)

# =========================

def fast_non_dominated_sort_vec(F):

    """

    Vectorized fast non-dominated sorting (minimization).

    F: (N, M) objective matrix

    Returns: fronts (list of lists of indices), rank (N,)

    """

    N = F.shape[0]

    # p dominates q if p <= q (all) and p < q (any)

    le = (F[:, None, :] <= F[None, :, :]).all(axis=2)

    lt = (F[:, None, :] <  F[None, :, :]).any(axis=2)

    D = le & lt

    n = D.sum(axis=0)  # number dominating each q

    rank = np.zeros(N, dtype=int)

    fronts = []

    current = np.where(n == 0)[0].tolist()

    used = set()

    while current:

        fronts.append(current)

        used.update(current)

        dominated_by_current = D[current].sum(axis=0)

        n -= dominated_by_current

        D[current, :] = False  # clear rows

        next_front = np.where(n == 0)[0].tolist()

        current = [idx for idx in next_front if idx not in used]

    for r, front in enumerate(fronts):

        for idx in front:

            rank[idx] = r

    return fronts, rank


def crowding_distance(F, idxs):

    if not idxs:

        return {}

    m = F.shape[1]

    d = {i: 0.0 for i in idxs}

    for j in range(m):

        sorted_idx = sorted(idxs, key=lambda i: F[i, j])

        fmin = F[sorted_idx[0], j]

        fmax = F[sorted_idx[-1], j]

        d[sorted_idx[0]] = d[sorted_idx[-1]] = float('inf')

        if fmax == fmin:

            continue

        for k in range(1, len(sorted_idx) - 1):

            i_prev = sorted_idx[k - 1]

            i_next = sorted_idx[k + 1]

            i_curr = sorted_idx[k]

            d[i_curr] += (F[i_next, j] - F[i_prev, j]) / (fmax - fmin + 1e-12)

    return d


def select_by_rank_and_crowding(X, F, pop):

    fronts, _ = fast_non_dominated_sort_vec(F)

    selected = []

    for front in fronts:

        if len(selected) + len(front) <= pop:

            selected.extend(front)

        else:

            cd = crowding_distance(F, front)

            front_sorted = sorted(front, key=lambda i: cd[i], reverse=True)

            selected.extend(front_sorted[:pop - len(selected)])

            break

    return X[selected], F[selected], selected, fronts



# =========================

# Variation operators: BWR / BMR / BMWR

# =========================

def clip_to_bounds(X, L, H):

    return np.minimum(np.maximum(X, L), H)


def make_offspring(name, X, F, rng, L, H):

    """

    name: "BWR" | "BMR" | "BMWR"

    Uses first-front best-by-crowding as Xb, last-front worst-by-crowding as Xw,

    population mean as Xm, and a random peer Xr.

    If r4 > 0.5, apply the update; else sample uniformly in bounds: X' = H - (H-L)*r3.

    """

    N, d = X.shape

    fronts, _ = fast_non_dominated_sort_vec(F)

    first_front = fronts[0]

    last_front = fronts[-1]

    cd_first = crowding_distance(F, first_front)

    cd_last  = crowding_distance(F, last_front)

    best_idx  = max(first_front, key=lambda i: cd_first[i])

    worst_idx = min(last_front,  key=lambda i: cd_last[i])

    x_best = X[best_idx]

    x_worst = X[worst_idx]

    x_mean = np.mean(X, axis=0)


    offspring = np.zeros_like(X)

    for k in range(N):

        r1 = rng.random(d)

        r2 = rng.random(d)

        r3 = rng.random(d)

        r4 = rng.random()

        Ffac = rng.integers(1, 3)  # 1 or 2


        # random peer different from k

        r_idx = k

        if N > 1:

            while r_idx == k:

                r_idx = rng.integers(0, N)

        x_r = X[r_idx]


        if r4 > 0.5:

            if name == "BWR":

                trial = X[k] + r1 * (x_best - Ffac * x_r) - r2 * (x_worst - x_r)

            elif name == "BMR":

                trial = X[k] + r1 * (x_best - Ffac * x_mean) + r2 * (x_best - x_r)

            elif name == "BMWR":

                trial = X[k] + r1 * (x_best - Ffac * x_mean) - r2 * (x_worst - x_r)

            else:

                raise ValueError("Unknown algo name")

        else:

            trial = H - (H - L) * r3  # random reinit within bounds

        offspring[k] = trial

    return clip_to_bounds(offspring, L, H)



# =========================

# MO driver (no metrics)

# =========================

def run_mo_algo(name, n=30, pop=100, iters=1000, seed=0):

    """

    Runs one MO algorithm (BWR/BMR/BMWR) on ZDT1.

    Returns: dict with

        X  : final population (pop, n)

        F  : final objective values (pop, 2)

        F1 : final first-front approximation (k, 2)

        X1 : the corresponding decision vectors (k, n)

    """

    rng = np.random.default_rng(seed)

    L = np.zeros(n)

    H = np.ones(n)


    # Initialize population uniformly in bounds

    X = rng.random((pop, n)) * (H - L) + L

    F = np.array([zdt1_f(x) for x in X])


    for _ in range(iters):

        X_off = make_offspring(name, X, F, rng, L, H)

        F_off = np.array([zdt1_f(x) for x in X_off])

        # Environmental selection (NSGA-II style)

        X_union = np.vstack([X, X_off])

        F_union = np.vstack([F, F_off])

        X, F, _, _ = select_by_rank_and_crowding(X_union, F_union, pop)


    # Extract first front and matching decision vectors

    fronts, _ = fast_non_dominated_sort_vec(F)

    idxs = fronts[0]

    F1 = F[idxs]

    X1 = X[idxs]

    return {"X": X, "F": F, "X1": X1, "F1": F1}



# =========================

# Main: run all three algorithms and PRINT ONLY variables & objectives

# (ZDT1 has no inequality/equality constraints beyond bounds.)

# =========================

if __name__ == "__main__":

    n = 30

    pop = 100

    iters = 1000  # change to 400 if you prefer shorter runs


    seeds = {"MO-BWR": 2025, "MO-BMR": 2026, "MO-BMWR": 2027}

    results = {}


    for name, seed in seeds.items():

        algo = name.split("-")[1]  # "BWR" | "BMR" | "BMWR"

        print("\n" + "="*70)

        print(f"{name} on ZDT1 (n={n}, pop={pop}, iters={iters})")

        print("Constraints: none (only variable bounds 0 ≤ x_i ≤ 1).")

        out = run_mo_algo(algo, n=n, pop=pop, iters=iters, seed=seed)

        results[name] = out


        X1, F1 = out["X1"], out["F1"]

        # Print a compact preview (first 10 Pareto solutions)

        k = min(10, X1.shape[0])

        print(f"First-front solutions shown (first {k} of {X1.shape[0]}):")

        for i in range(k):

            xi = X1[i]

            f1, f2 = F1[i]

            # show x1..x5 for brevity; full vector is saved to CSV below

            x_preview = ", ".join(f"x{j+1}={xi[j]:.4f}" for j in range(min(5, n)))

            print(f"  #{i+1:02d}: {x_preview},  f1={f1:.6f}, f2={f2:.6f}")


        # OPTIONAL: Save full first-front (all variables and objectives) to CSV

        df = pd.DataFrame(X1, columns=[f"x{j+1}" for j in range(n)])

        df["f1"] = F1[:, 0]

        df["f2"] = F1[:, 1]

        csv_name = f"ZDT1_{name.replace('-', '_')}_first_front.csv"

        df.to_csv(csv_name, index=False)

        print(f"Saved full first-front (all variables + f1, f2) to: {csv_name}")


    print("\nDone. No performance metrics were computed or printed—only variables and objectives.")
