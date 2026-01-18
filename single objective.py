# run_bwr_bmr_bmwr_constrained.py

# Complete codes for BWR, BMR, and BMWR used to solve:

# Minimize f(x) = x1^2 + x2^2

# Subject to: g1(x)=x1+x2-1 <= 0, g2(x)=0.2 - x1 <= 0, g3(x)=0.3 - x2 <= 0

# Bounds: 0 <= x1, x2 <= 1

# Settings: pop=20, iters=1000, very high penalty for violations


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

def f_obj(x):

    """Objective: minimize x1^2 + x2^2."""

    return x[0]**2 + x[1]**2


def g1(x):  # <= 0

    return x[0] + x[1] - 1.0


def g2(x):  # <= 0 (feasible if x1 >= 0.2)

    return 0.2 - x[0]


def g3(x):  # <= 0 (feasible if x2 >= 0.3)

    return 0.3 - x[1]


CONSTRAINTS = [g1, g2, g3]

L = np.array([0.0, 0.0])

H = np.array([1.0, 1.0])


# --------------------------

# Utilities

# --------------------------

def clip_to_bounds(x, L, H):

    return np.minimum(np.maximum(x, L), H)


def penalized_objective(x, rho=1e6):

    """

    Penalized objective for minimization:

      M(x) = f(x) + rho * sum(max(g_j(x), 0)^2)

    Returns (M, f, penalty_sum)

    """

    fx = f_obj(x)

    penalty = 0.0

    for g in CONSTRAINTS:

        v = g(x)

        if v > 0:

            penalty += v * v

    return fx + rho * penalty, fx, penalty


def evaluate_population(X, rho=1e6):

    """Evaluate whole population; returns arrays for M, f, penalty, feasibility flag."""

    c = X.shape[0]

    Ms = np.zeros(c)

    fs = np.zeros(c)

    ps = np.zeros(c)

    feas = np.zeros(c, dtype=bool)

    for i in range(c):

        Ms[i], fs[i], ps[i] = penalized_objective(X[i], rho=rho)

        feas[i] = (ps[i] == 0.0)
        print(feas)

    return Ms, fs, ps, feas


def pop_stats(X, fitness):

    """Return (x_best, x_mean, x_worst, idx_best) for given population and fitness (lower is better)."""

    i_best = int(np.argmin(fitness))

    i_worst = int(np.argmax(fitness))

    return X[i_best].copy(), X.mean(axis=0), X[i_worst].copy(), i_best


# --------------------------

# Core solver (BWR, BMR, BMWR)

# --------------------------

def run_algo(name, pop=5, iters=1000, seed=0, rho=1e6):

    """

    name ∈ {"BWR","BMR","BMWR"}

    Implements the update rules:

      BWR:

        if r4>0.5: X' = X + r1*(Xb − F*Xr) − r2*(Xw − Xr)

        else:      X' = H − (H−L)*r3

      BMR:

        if r4>0.5: X' = X + r1*(Xb − F*Xm) + r2*(Xb − Xr)

        else:      same as BWR

      BMWR:

        if r4>0.5: X' = X + r1*(Xb − F*Xm) − r2*(Xw − Xr)

        else:      same as BWR

    """

    rng = np.random.default_rng(seed)

    d = L.shape[0]


    # Initialize population uniformly in bounds

    X = rng.random((pop, d)) * (H - L) + L


    # Evaluate and get population stats

    M, f, p, feas = evaluate_population(X, rho=rho)

    x_best, x_mean, x_worst, i_best = pop_stats(X, M)


    # Track best feasible found so far

    best_feas_f = np.inf

    best_feas_x = None


    history = []

    for it in range(1, iters + 1):

        Xnew = X.copy()

        for k in range(pop):

            r1 = rng.random(d)

            r2 = rng.random(d)

            r3 = rng.random(d)

            r4 = rng.random()

            F = rng.integers(1, 3)  # 1 or 2


            # pick a random peer index != k

            r_idx = k

            if pop > 1:

                while r_idx == k:

                    r_idx = rng.integers(0, pop)

            xr = X[r_idx]


            if name == "BWR":

                if r4 > 0.5:

                    trial = X[k] + r1 * (x_best - F * xr) - r2 * (x_worst - xr)

                else:

                    trial = H - (H - L) * r3

            elif name == "BMR":

                if r4 > 0.5:

                    trial = X[k] + r1 * (x_best - F * x_mean) + r2 * (x_best - xr)

                else:

                    trial = H - (H - L) * r3

            elif name == "BMWR":

                if r4 > 0.5:

                    trial = X[k] + r1 * (x_best - F * x_mean) - r2 * (x_worst - xr)

                else:

                    trial = H - (H - L) * r3

            else:

                raise ValueError("Unknown algorithm: " + str(name))


            # Enforce bounds

            trial = clip_to_bounds(trial, L, H)


            # Greedy selection on penalized objective

            M_trial, _, _ = penalized_objective(trial, rho=rho)

            M_curr, _, _ = penalized_objective(X[k], rho=rho)

            if M_trial < M_curr:

                Xnew[k] = trial


        # Advance generation

        X = Xnew

        M, f, p, feas = evaluate_population(X, rho=rho)

        x_best, x_mean, x_worst, i_best = pop_stats(X, M)


        # Track best feasible solution seen so far

        feas_idxs = np.where(feas)[0]

        if feas_idxs.size > 0:

            f_feas_min = np.min(f[feas_idxs])

            if f_feas_min < best_feas_f:

                j = feas_idxs[np.argmin(f[feas_idxs])]

                best_feas_f = f[j]

                best_feas_x = X[j].copy()


        history.append({

            "iter": it,

            "best_penalized_M": float(np.min(M)),

            "mean_penalized_M": float(np.mean(M)),

            "worst_penalized_M": float(np.max(M)),

            "best_feasible_f": (float(best_feas_f) if np.isfinite(best_feas_f) else None),

            "best_feasible_x1": (float(best_feas_x[0]) if best_feas_x is not None else None),

            "best_feasible_x2": (float(best_feas_x[1]) if best_feas_x is not None else None),

            "num_feasible": int(np.sum(feas))

        })


    # Final best (penalized)

    i_best_final = int(np.argmin(M))

    best_x = X[i_best_final].copy()

    best_M, best_fx, _ = penalized_objective(best_x, rho=rho)


    # Prefer best feasible if available

    if np.isfinite(best_feas_f):

        return pd.DataFrame(history), best_feas_x, best_feas_f, best_feas_f  # M = f if feasible

    else:

        return pd.DataFrame(history), best_x, best_fx, best_M


# --------------------------

# Run all three algorithms

# --------------------------

if __name__ == "__main__":

    seeds = {"BWR": 123, "BMR": 124, "BMWR": 125}

    pop = 5

    iters = 1000

    rho = 1e6


    histories = {}

    results = {}

    for algo in ["BWR", "BMR", "BMWR"]:

        h, bx, bf, bM = run_algo(algo, pop=pop, iters=iters, seed=seeds[algo], rho=rho)

        histories[algo] = h

        results[algo] = {"x": bx, "f": bf, "M": bM}


    # Save CSVs (optional)

    hist_all = pd.concat(

        [histories[a].assign(algorithm=a) for a in ["BWR", "BMR", "BMWR"]],

        ignore_index=True

    )

    hist_all.to_csv("mo_bwr_bmr_bmwr_histories.csv", index=False)


    res_df = pd.DataFrame([

        {"Algorithm": a, "Best x1": results[a]["x"][0], "Best x2": results[a]["x"][1],

         "Best f(x)": results[a]["f"], "Best penalized M(x)": results[a]["M"]}

        for a in ["BWR", "BMR", "BMWR"]

    ])

    res_df.to_csv("mo_bwr_bmr_bmwr_results.csv", index=False)


    # Plots (optional)

    plt.figure()

    for algo in ["BWR", "BMR", "BMWR"]:

        df = histories[algo]

        plt.plot(df["iter"], df["best_feasible_f"], label=algo)

    plt.xlabel("Iteration")

    plt.ylabel("Best feasible f(x) so far")

    plt.title("Constrained minimization: best feasible f(x)")

    plt.legend()

    plt.savefig("convergence_best_feasible_f.png", dpi=150, bbox_inches="tight")
    plt.show()

    plt.close()


    plt.figure()

    for algo in ["BWR", "BMR", "BMWR"]:

        df = histories[algo]

        plt.plot(df["iter"], df["best_penalized_M"], label=algo)

    plt.xlabel("Iteration")

    plt.ylabel("Best penalized M(x)")

    plt.title("Constrained minimization: penalized objective M(x)")

    plt.legend()

    plt.savefig("convergence_best_penalized_M.png", dpi=150, bbox_inches="tight")

    plt.close()
