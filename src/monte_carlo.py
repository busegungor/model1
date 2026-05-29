"""
monte_carlo.py — Net-demand simulation for BOTH methods (SAA & Bayesian).

This module builds the net-demand matrix N[t, j] that the newsvendor solver
consumes. Both data-driven methods share the SAME machinery (returns,
resalability, sequencing); they differ ONLY in the demand source:

    SAA      : demand = historical G_t  (fixed panel values; parameters
               r_hat, k_hat are point estimates)  -> ONE uncertainty layer
               (realization only).
    Bayesian : demand = G^(j) ~ NegBin-Beta posterior predictive; r, k drawn
               from their Beta posteriors -> TWO uncertainty layers
               (parameter + realization).

Keeping both in one module (instead of a separate bayesian file) makes the
shared pipeline explicit and removes duplicated sequencing-call logic:
both call the SAME _apply_sequencing helper and the SAME Step-6 rules.

────────────────────────────────────────────────────────────────────
Colab origin (SAA path): build_net_demand_matrix, lines 453-486.
Bayesian path is new (Architecture A: SAA pipeline, posterior demand).
────────────────────────────────────────────────────────────────────
"""

import numpy as np

from sequencing_rules import _net_demand_S1, _net_demand_S3, _net_demand_S4
from bayesian_models import build_posteriors


# ════════════════════════════════════════════════════════════════════
# Shared sequencing dispatcher
def _apply_sequencing(rule, Gt_or_Gj, K_all, at, rt, rng, per_replication):
    """
    Apply a sequencing rule to one period's replications.

    Two calling modes, because the two methods differ in demand structure:

      per_replication=False (SAA):
        A SINGLE gross demand Gt is shared by all J replications.
        Gt_or_Gj is an int.

      per_replication=True (Bayesian):
        Each replication has its OWN demand G^(j). Gt_or_Gj is an array.

    Note on signatures (Step 6): S1 and S3 take only (Gt, K_all) because
    they need no intraperiod timing; S4 additionally needs (arrival,
    return_arrival, rng). The dispatcher passes the right arguments to each.

    Returns
    -------
    np.ndarray, shape (J,) : net demand for this period's J replications.
    """
    if not per_replication:
        # SAA: single Gt, vectorised over K_all
        Gt = int(Gt_or_Gj)
        if rule == "S1":
            return _net_demand_S1(Gt, K_all)
        if rule == "S3":
            return _net_demand_S3(Gt, K_all)
        return _net_demand_S4(Gt, K_all, at, rt, rng)

    # Bayesian: demand varies per replication
    G_j = Gt_or_Gj
    J = len(G_j)
    if rule == "S1":
        return np.maximum(G_j - K_all, 0).astype(float)
    if rule == "S3":
        return G_j.astype(float)
    # S4 — per replication (each draw has its own Gt and single K)
    out = np.empty(J)
    for j in range(J):
        out[j] = _net_demand_S4(int(G_j[j]),
                                np.array([int(K_all[j])]),
                                at, rt, rng)[0]
    return out


# ════════════════════════════════════════════════════════════════════
# 1. SAA net-demand matrix (frequentist; demand = historical G_t)
# ════════════════════════════════════════════════════════════════════
def build_net_demand_matrix(dataset, r_hat, k_hat,
                            rule="S4", J=200, seed=0) -> np.ndarray:
    """
    SAA net-demand matrix N[t, j].

    For each period t:
        U ~ Binomial(Gt, r_hat),  K ~ Binomial(U, k_hat),  then sequencing.
    Gt is the historical panel value (fixed). r_hat, k_hat are point
    estimates (parameter_estimation). One uncertainty layer (realization).

    Colab origin: lines 453-486.
    """
    panel = dataset["panel"]
    arrival = dataset.get("arrival_times", {})
    return_arrival = dataset.get("return_arrival_times", {})
    m = len(panel)
    rng = np.random.default_rng(seed)

    N = np.empty((m, J))
    for idx in range(m):
        period = panel.index[idx]
        at = arrival.get(period, np.array([]))
        rt = return_arrival.get(period, np.array([]))

        Gt = int(panel["G_t"].iloc[idx])
        U_all = rng.binomial(Gt, r_hat, size=J)      # returns
        K_all = rng.binomial(U_all, k_hat)           # resalable returns

        N[idx] = _apply_sequencing(rule, Gt, K_all, at, rt, rng,
                                   per_replication=False)
    return N


# ════════════════════════════════════════════════════════════════════
# 2. Bayesian net-demand matrix (posterior predictive demand)
# ════════════════════════════════════════════════════════════════════
def build_bayesian_net_demand_matrix(dataset, rule="S4", J=200, seed=0,
                                     priors=None, posteriors=None) -> np.ndarray:
    """
    Bayesian net-demand matrix N[t, j] (Architecture A).

    For each period t and replication j:
        Layer 1 (parameter): p^(j), r^(j), k^(j) ~ their posteriors
        Layer 2 (realization): G^(j) ~ NegBin(r_NB, p^(j)),
                               U^(j) ~ Binom(G^(j), r^(j)),
                               K^(j) ~ Binom(U^(j), k^(j))
        then sequencing on (G^(j), K^(j)).

    Same machinery as the SAA matrix (returns, sequencing, m*J scenarios) —
    only the demand source differs, so any Q* difference is due to the
    demand model, not the pipeline.

    Parameters
    ----------
    dataset : dict      (panel + arrival_times + return_arrival_times)
    rule : {"S1","S3","S4"}
    J : int             Monte Carlo replications
    seed : int
    priors : dict, optional      override priors (sensitivity, Step 15)
    posteriors : dict, optional  pre-computed (build_posteriors); if None,
                                 computed from dataset["panel"].

    Returns
    -------
    np.ndarray, shape (m, J)
    """
    panel = dataset["panel"]
    arrival = dataset.get("arrival_times", {})
    return_arrival = dataset.get("return_arrival_times", {})
    m = len(panel)
    rng = np.random.default_rng(seed)

    if posteriors is None:
        posteriors = build_posteriors(panel, priors=priors)
    dem, rp, kp = posteriors["demand"], posteriors["r"], posteriors["k"]
    r_NB = dem["r_NB"]

    N = np.empty((m, J))
    for idx in range(m):
        period = panel.index[idx]
        at = arrival.get(period, np.array([]))
        rt = return_arrival.get(period, np.array([]))

        # Layer 1 — parameter uncertainty (per replication)
        p_j = rng.beta(dem["a_post"], dem["b_post"], size=J)
        r_j = rng.beta(rp["a_post"], rp["b_post"], size=J)
        k_j = rng.beta(kp["a_post"], kp["b_post"], size=J)

        # Layer 2 — realization
        G_j = rng.negative_binomial(r_NB, p_j)
        U_j = rng.binomial(G_j, r_j)
        K_j = rng.binomial(U_j, k_j)

        N[idx] = _apply_sequencing(rule, G_j, K_j, at, rt, rng,
                                   per_replication=True)
    return N


# ════════════════════════════════════════════════════════════════════
# Helper: summary stats for a matrix
# ════════════════════════════════════════════════════════════════════
def summarize_matrix(N):
    """Quick mean/std/min/max of a net-demand matrix (for sanity checks)."""
    flat = N.flatten()
    return {"mean": float(flat.mean()), "std": float(flat.std()),
            "min": float(flat.min()), "max": float(flat.max())}


# ════════════════════════════════════════════════════════════════════
# Standalone test (run: python monte_carlo.py)
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from data_analysis import order_items_clean, returns_clean
    from preprocessing import build_category_datasets
    from parameter_estimation import build_category_params

    datasets = build_category_datasets(order_items_clean, returns_clean, period="W")
    params = build_category_params(datasets, order_items_clean)

    print(f"\n{'='*70}")
    print("Net-demand matrices: SAA vs Bayesian (W, S4) — mean net demand")
    print(f"{'='*70}")
    print(f"{'Category':<22}{'SAA_mean':>10}{'Bay_mean':>10}{'SAA_std':>9}{'Bay_std':>9}")
    print("-" * 60)

    for cat, ds in datasets.items():
        prm = params[cat]
        N_saa = build_net_demand_matrix(ds, prm["r_hat"], prm["k_hat"],
                                        rule="S4", J=200, seed=0)
        N_bay = build_bayesian_net_demand_matrix(ds, rule="S4", J=200, seed=0)
        s_saa = summarize_matrix(N_saa)
        s_bay = summarize_matrix(N_bay)
        print(f"{cat:<22}{s_saa['mean']:>10.2f}{s_bay['mean']:>10.2f}"
              f"{s_saa['std']:>9.2f}{s_bay['std']:>9.2f}")

    print("\nBoth use the same sequencing (_apply_sequencing) and Step-6 rules.")
    print("Bayesian std should be >= SAA std (extra parameter-uncertainty layer).")