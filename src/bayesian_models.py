"""
bayesian_models.py — Conjugate Bayesian updating for demand, return,
                     and resalability parameters.

Implements the parametric (Bayesian) counterpart to the frequentist point
estimates in parameter_estimation.py. Three conjugate structures:

  1. Demand  :  G_t ~ NegBin(r_NB, p) with r_NB KNOWN, p ~ Beta(a0, b0)
        posterior  p ~ Beta(a0 + r_NB*m,  b0 + sum G_t)
        (Berk, Gürler & Levine 2007, Theorem 3.1)
     r_NB (dispersion) is moment-matched per category/period:
        p_mom = mean/var,  r_NB = mean*p_mom/(1 - p_mom).
     RATIONALE: category demand is OVERDISPERSED (VMR ~ 3-6); a Poisson-
     Gamma posterior predictive collapses to VMR~1 and cannot reproduce
     the observed spread. NegBin captures it (verified: pp VMR ~ data VMR).
     AIC analysis (category_distribution_analysis.py) also selects NegBin.

  2. Return prob  r :  u_t ~ Binomial(G_t, r),  r ~ Beta(a0, b0)
        posterior  r ~ Beta(a0 + sum u_t,  b0 + sum (G_t - u_t))

  3. Resalability k :  k_obs_t ~ Binomial(u_t, k),  k ~ Beta(a0, b0)
        posterior  k ~ Beta(a0 + sum k_obs_t,  b0 + sum (u_t - k_obs_t))

CONJUGACY: prior and posterior share the same family — only parameters
update (closed form, no integration). Valid because demand is uncensored
(Berk et al. 2007, Section 2.2).

WEAKLY INFORMATIVE (Philosophy A): defaults from config.BAYESIAN_PRIORS,
deliberately weak so the posterior is data-dominated. All functions accept
overrides for prior-sensitivity analysis (Step 15).

────────────────────────────────────────────────────────────────────
This module is NEW (no Colab origin). Formulas: Berk et al. (2007),
Theorems 3.1; standard Beta-Binomial conjugacy.
────────────────────────────────────────────────────────────────────
"""

import numpy as np
from scipy import optimize, special
from config import BAYESIAN_PRIORS



def estimate_negbin_dispersion(G_values) -> float:
    """
    MLE of the (fixed, known) NegBin dispersion r_NB.

    We maximize the NegBin profile log-likelihood over r. For a fixed r,
    the profile MLE of the success probability is p_hat = r / (r + mean),
    so only r is optimized numerically (scipy handles the derivative).

    r_NB is treated as KNOWN (Berk et al. 2007, Thm 3.1): conjugacy with
    the Beta prior on p requires r fixed (Cook, conjugate tables: NegBin(r,p)
    with r known + Beta → Beta posterior). Estimated PER CATEGORY to avoid
    the downward pooling bias documented by Lloyd-Smith (2007), who studies
    MLE of the NegBin dispersion precisely in the highly-overdispersed regime.

    Guards: if var <= mean (no overdispersion in this slice), return a large
    r_NB so the NegBin approaches the Poisson limit.

    Reference: Lloyd-Smith (2007), PLoS ONE 2(2):e180.

    Returns
    -------
    float : r_NB (> 0)
    """
    G = np.asarray(G_values, dtype=float)
    n = len(G)
    mean = G.mean()
    if mean <= 0:
        return 1.0
    if G.var() <= mean:
        return 1e6  # not overdispersed → Poisson limit

    def neg_loglik(log_r):
        r = np.exp(log_r)            # enforce r > 0 via log-parametrization
        p = r / (r + mean)           # profile MLE of p for this r
        ll = np.sum(
            special.gammaln(G + r) - special.gammaln(r) - special.gammaln(G + 1)
            + r * np.log(p) + G * np.log(1 - p)
        )
        return -ll

    res = optimize.minimize_scalar(neg_loglik, bounds=(-5, 15), method="bounded")
    return float(np.exp(res.x))

def update_negbin_beta(G_values, r_NB=None, a0=None, b0=None) -> dict:
    """
    Conjugate update of the demand posterior (NegBin-Beta).

        G_t ~ NegBin(r_NB, p)  with r_NB KNOWN,   p ~ Beta(a0, b0)
        posterior:  p ~ Beta(a0 + r_NB * m,  b0 + sum_t G_t)

    Parameters
    ----------
    G_values : array-like
        Observed gross demand per (training) period.
    r_NB : float, optional
        Known NegBin dispersion. If None, moment-matched from G_values.
    a0, b0 : float, optional
        Prior Beta parameters for p. Default: config.BAYESIAN_PRIORS.

    Returns
    -------
    dict: {r_NB, a_post, b_post, p_post_mean,
           demand_post_mean, demand_post_var}
        demand_post_mean / var are the NegBin mean/var at p = p_post_mean,
        i.e. the (approximate) posterior-mean demand moments.

    Reference: Berk et al. (2007), Theorem 3.1 (uncensored case).
    """
    if a0 is None:
        a0 = BAYESIAN_PRIORS["demand_p_a0"]
    if b0 is None:
        b0 = BAYESIAN_PRIORS["demand_p_b0"]

    G = np.asarray(G_values, dtype=float)
    m = len(G)
    sum_G = G.sum()

    if r_NB is None:
        r_NB = estimate_negbin_dispersion(G)

    a_post = a0 + r_NB * m
    b_post = b0 + sum_G
    p_post_mean = a_post / (a_post + b_post)

    # NegBin(r_NB, p) moments at the posterior-mean p
    demand_post_mean = r_NB * (1 - p_post_mean) / p_post_mean
    demand_post_var = r_NB * (1 - p_post_mean) / (p_post_mean ** 2)

    return {
        "r_NB": r_NB,
        "a_post": a_post,
        "b_post": b_post,
        "p_post_mean": p_post_mean,
        "demand_post_mean": demand_post_mean,
        "demand_post_var": demand_post_var,
    }


# ════════════════════════════════════════════════════════════════════
# 2. Beta-Bernoulli update (return probability r) — unchanged
# ════════════════════════════════════════════════════════════════════
def update_beta_return(G_values, u_values, a0=None, b0=None) -> dict:
    """
    u_t ~ Binomial(G_t, r),  r ~ Beta(a0, b0)
    posterior:  r ~ Beta(a0 + sum u_t,  b0 + sum (G_t - u_t))
    """
    if a0 is None:
        a0 = BAYESIAN_PRIORS["r_a0"]
    if b0 is None:
        b0 = BAYESIAN_PRIORS["r_b0"]

    G = np.asarray(G_values, dtype=float)
    u = np.asarray(u_values, dtype=float)
    a_post = a0 + u.sum()
    b_post = b0 + (G - u).sum()

    return {
        "a_post": a_post,
        "b_post": b_post,
        "post_mean": a_post / (a_post + b_post),
        "post_var": (a_post * b_post)
                    / ((a_post + b_post) ** 2 * (a_post + b_post + 1)),
    }


# ════════════════════════════════════════════════════════════════════
# 3. Beta-Bernoulli update (resalability k) — unchanged
# ════════════════════════════════════════════════════════════════════
def update_beta_resalable(u_values, k_obs_values, a0=None, b0=None) -> dict:
    """
    k_obs_t ~ Binomial(u_t, k),  k ~ Beta(a0, b0)
    posterior:  k ~ Beta(a0 + sum k_obs_t,  b0 + sum (u_t - k_obs_t))
    Trials are the RETURNS u_t (nested within demand).
    """
    if a0 is None:
        a0 = BAYESIAN_PRIORS["k_a0"]
    if b0 is None:
        b0 = BAYESIAN_PRIORS["k_b0"]

    u = np.asarray(u_values, dtype=float)
    k = np.asarray(k_obs_values, dtype=float)
    a_post = a0 + k.sum()
    b_post = b0 + (u - k).sum()
    denom = a_post + b_post

    return {
        "a_post": a_post,
        "b_post": b_post,
        "post_mean": a_post / denom if denom > 0 else np.nan,
        "post_var": (a_post * b_post) / (denom ** 2 * (denom + 1))
                    if denom > 0 else np.nan,
    }


# ════════════════════════════════════════════════════════════════════
# 4. Posterior samplers
# ════════════════════════════════════════════════════════════════════
def sample_demand_posterior(r_NB, a_post, b_post, J, rng) -> np.ndarray:
    """
    Draw J demand samples from the NegBin-Beta posterior predictive.

      p^(j) ~ Beta(a_post, b_post)        (parameter uncertainty)
      G^(j) ~ NegBin(r_NB, p^(j))         (realization)

    numpy's negative_binomial(n, p) uses n = number of successes (= r_NB)
    and p = success probability, matching our parametrization directly.
    """
    p_j = rng.beta(a_post, b_post, size=J)
    return rng.negative_binomial(r_NB, p_j)


def sample_r_posterior(a_post, b_post, J, rng) -> np.ndarray:
    """Draw J return-probability samples from the Beta posterior."""
    return rng.beta(a_post, b_post, size=J)


def sample_k_posterior(a_post, b_post, J, rng) -> np.ndarray:
    """Draw J resalability samples from the Beta posterior."""
    return rng.beta(a_post, b_post, size=J)


def build_posteriors(panel, priors=None) -> dict:
    """
    Run all three conjugate updates from a category panel.

    For out-of-sample evaluation, pass only the first m training rows.

    Returns
    -------
    dict:
        {
          "demand": {r_NB, a_post, b_post, p_post_mean,
                     demand_post_mean, demand_post_var},
          "r":      {a_post, b_post, post_mean, post_var},
          "k":      {a_post, b_post, post_mean, post_var},
        }
    """
    p = dict(BAYESIAN_PRIORS)
    if priors:
        p.update(priors)

    demand = update_negbin_beta(
        panel["G_t"].values, a0=p["demand_p_a0"], b0=p["demand_p_b0"]
    )
    r = update_beta_return(
        panel["G_t"].values, panel["u_t"].values, p["r_a0"], p["r_b0"]
    )
    k = update_beta_resalable(
        panel["u_t"].values, panel["k_obs_t"].values, p["k_a0"], p["k_b0"]
    )
    return {"demand": demand, "r": r, "k": k}


# ════════════════════════════════════════════════════════════════════
# Standalone test (run: python bayesian_models.py)
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from data_analysis import order_items_clean, returns_clean
    from preprocessing import build_category_datasets
    from parameter_estimation import build_category_params

    datasets = build_category_datasets(order_items_clean, returns_clean, period="W")
    params = build_category_params(datasets, order_items_clean)

    print(f"\n{'='*82}")
    print("POSTERIOR vs FREQUENTIST + overdispersion capture (W period)")
    print(f"{'='*82}")
    print(f"{'Category':<24} {'dem_post':>9} {'dem_freq':>9} {'VMR_pp':>7} "
          f"{'VMR_dat':>7} {'r_post':>7} {'k_post':>7}")
    print("-" * 82)

    rng = np.random.default_rng(0)
    for cat, ds in datasets.items():
        post = build_posteriors(ds["panel"])
        prm = params[cat]
        G = ds["panel"]["G_t"].values
        # posterior predictive VMR
        dem = post["demand"]
        G_pp = sample_demand_posterior(dem["r_NB"], dem["a_post"],
                                       dem["b_post"], 20000, rng)
        vmr_pp = G_pp.var() / G_pp.mean()
        vmr_dat = G.var() / G.mean()
        print(f"{cat:<24} {dem['demand_post_mean']:>9.2f} {G.mean():>9.2f} "
              f"{vmr_pp:>7.2f} {vmr_dat:>7.2f} "
              f"{post['r']['post_mean']:>7.4f} {post['k']['post_mean']:>7.4f}")

    print("\nVMR_pp ≈ VMR_dat → NegBin captures overdispersion (vs Poisson-Gamma VMR~1)")