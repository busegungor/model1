"""
benchmark.py — Deterministic netting benchmark (classical literature).

The classical newsvendor-with-returns literature (Vlachos & Dekker 2003;
Mostard & Teunter 2006; Mostard, de Koster & Teunter 2005) nets returns
DETERMINISTICALLY: it deducts the EXPECTED resalable fraction r*k from
gross demand, giving a single net-demand value per period:

    N_t^det = G_t * (1 - r_hat * k_hat)

There is no Monte Carlo, no stochasticity in the returns, and no
intraperiod sequencing — it is the simplest treatment of returns.

This serves as the LITERATURE REFERENCE POINT against which our stochastic
sequencing rules (S1/S3/S4) and the Bayesian framework are compared. The
gap between this deterministic benchmark and the stochastic rules quantifies
the value of explicitly modeling return stochasticity.

WHERE IT IS USED:
  - SAA side:      deterministic N_t^det fed to the SAA quantile solver.
  - Bayesian side: same deterministic netting, but G_t drawn from the
                   posterior predictive (so the benchmark is comparable
                   within each method). See benchmark_bayesian().

────────────────────────────────────────────────────────────────────
Colab origin: netting_benchmark(), lines 573-578 (SAA side). The Bayesian
              counterpart is new (added for the Step-15 comparison).
────────────────────────────────────────────────────────────────────
"""

import numpy as np

from saa_solver import solve_saa


def netting_benchmark(
    panel,
    r_hat: float,
    k_hat: float,
    c: float,
    s: float,
    pN: float,
    gN: float,
    gurobi_env=None,
    verbose: bool = False,
) -> dict:
    """
    SAA-side deterministic netting benchmark.

    Net demand is the deterministic expected-netting value per period:
        N_t^det = G_t * (1 - r_hat * k_hat)
    These m values (one per period) are fed to the SAA solver as a
    degenerate (m x 1) "matrix" — no replications, since there is no
    stochasticity to sample.

    Parameters
    ----------
    panel : pd.DataFrame
        Category panel with column G_t (from preprocessing).
    r_hat, k_hat : float
        Frequentist return / resalability estimates.
    c, s, pN, gN : float
        Cost parameters.
    gurobi_env : gp.Env or None
    verbose : bool

    Returns
    -------
    dict (same shape as solve_saa output): Q_opt, obj_opt, ESN, alpha, solver.

    Colab origin: lines 573-578.
    """
    # Deterministic net demand: Gt * (1 - r*k), one value per period
    N_det = (panel["G_t"].values * (1.0 - r_hat * k_hat)).reshape(-1, 1)

    result = solve_saa(
        N_det, c=c, s=s, pN=pN, gN=gN,
        gurobi_env=gurobi_env, verbose=verbose,
    )
    result["method"] = "deterministic_netting_SAA"
    return result


def benchmark_bayesian(
    lambda_samples: np.ndarray,
    r_hat: float,
    k_hat: float,
    c: float,
    s: float,
    pN: float,
    gN: float,
    gurobi_env=None,
    verbose: bool = False,
) -> dict:
    """
    Bayesian-side deterministic netting benchmark.

    Same deterministic netting idea, but gross demand is drawn from the
    POSTERIOR PREDICTIVE rather than taken from historical observations:

        G^(j) ~ posterior predictive demand   (lambda_samples → Poisson)
        N^(j) = G^(j) * (1 - r_hat * k_hat)

    This makes the benchmark comparable on the Bayesian side: it isolates
    the deterministic-netting assumption while using the Bayesian demand
    model. (Returns are still netted deterministically via r_hat*k_hat,
    NOT via Beta posteriors — that is what makes it the *benchmark* rather
    than the full Bayesian model.)

    Parameters
    ----------
    lambda_samples : np.ndarray, shape (J,)
        Posterior predictive demand draws G^(j). (In Step 14 these come
        from sampling lambda ~ Gamma posterior, then G ~ Poisson(lambda).
        Here we accept the already-drawn G^(j) array directly.)
    r_hat, k_hat : float
        Point estimates used for deterministic netting.
    c, s, pN, gN : float
        Cost parameters.
    gurobi_env : gp.Env or None
    verbose : bool

    Returns
    -------
    dict (same shape as solve_saa output) with method tag.

    Note
    ----
    This is a thin wrapper: it applies deterministic netting to the
    posterior-predictive demand draws and solves via the same quantile
    SAA core. The (J,) net-demand vector is reshaped to (J, 1) so it
    flattens correctly in solve_saa.
    """
    # Deterministic netting on posterior-predictive demand draws
    N_det = (lambda_samples * (1.0 - r_hat * k_hat)).reshape(-1, 1)

    result = solve_saa(
        N_det, c=c, s=s, pN=pN, gN=gN,
        gurobi_env=gurobi_env, verbose=verbose,
    )
    result["method"] = "deterministic_netting_Bayesian"
    return result


# ════════════════════════════════════════════════════════════════════
# Standalone test (run: python benchmark.py)
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from data_analysis import order_items_clean, returns_clean
    from preprocessing import build_category_datasets
    from parameter_estimation import build_category_params
    from gurobi_env import make_gurobi_env

    datasets = build_category_datasets(order_items_clean, returns_clean, period="W")
    params = build_category_params(datasets, order_items_clean)
    env = make_gurobi_env()

    print(f"\n{'='*70}")
    print("DETERMINISTIC NETTING BENCHMARK — SAA side (W period)")
    print(f"{'='*70}")
    print(f"{'Category':<24} {'alpha':>7} {'Q*':>7} {'Cost':>9} {'ESN':>7}")
    print("-" * 60)

    for cat, ds in datasets.items():
        prm = params[cat]
        res = netting_benchmark(
            ds["panel"], prm["r_hat"], prm["k_hat"],
            c=prm["c"], s=prm["s"], pN=prm["pN"], gN=prm["gN"],
            gurobi_env=env,
        )
        print(f"{cat:<24} {res['alpha']:>7.4f} {res['Q_opt']:>7.2f} "
              f"{res['obj_opt']:>9.4f} {res['ESN']:>7.4f}")