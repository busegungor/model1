"""
sequencing_rules.py — Intraperiod sequencing rules for net-demand netting.

Given gross demand G_t and resalable returns K_t^(j) per replication, the
EFFECTIVE NET DEMAND N_t^(j) depends on WHEN returns arrive relative to
demand within the period. We characterize the spectrum with three rules:

  S1  Aggregate netting (classical, lower bound):  N = max(G - K, 0)
      All resalable returns offset demand; timing ignored. The assumption
      in Vlachos & Dekker (2003), Mostard & Teunter (2006). Equals S2
      (returns-before-demand).

  S3  Demand-before-returns (worst case, upper bound):  N = G
      All demand arrives before any return; no return offsets within-period
      demand. The economic value of those late resalable returns is captured
      as SALVAGE in the cost parameters (handled in the cost layer, not here).

  S4  Order-of-arrival (empirical, between bounds):  interleaved netting
      Returns/demand merged in empirical arrival order; a resalable return
      offsets only demand arriving AFTER it. The paper's main model.
      Spectrum:  S1 (==S2)  <=  S4  <=  S3.

WHY KEEP ALL THREE: the supervisor asked to characterize sequencing
"systematically ... when does it matter if at all". S1 and S3 are the
BOUNDS of the spectrum; S4 is the empirical point. The gap S3 - S1 measures
how much sequencing can matter; S4's position shows where reality falls.
S4 alone cannot answer "does sequencing matter?" — the bounds are required.

ALL THREE share the same upstream binomial draws (K_all), so differences in
N are attributable purely to sequencing, not to sampling noise.

────────────────────────────────────────────────────────────────────
Colab origin: simulate_net_demand() = rule S4 (lines 370-451). S1/S3 new.
───────────────────────────────────────────────────────────────────


S1 ve S3, period İÇİ zaman (intra-period timing, timestamp'ler) kullanmıyor. 
Ama period'lar arası zamana (her hafta ayrı hesap) ikisi de bağlı.
"""

import numpy as np


# ════════════════════════════════════════════════════════════════════
# S1 — Aggregate netting (closed form)
# ════════════════════════════════════════════════════════════════════
def _net_demand_S1(Gt: int, K_all: np.ndarray) -> np.ndarray:
    """
    Aggregate netting: every resalable return offsets one unit of demand,
    regardless of timing.

        N^(j) = max(Gt - K^(j), 0)

    This is the classical assumption (Vlachos & Dekker 2003) and the LOWER
    bound of the sequencing spectrum (equivalent to returns-before-demand).

    Parameters
    ----------
    Gt : int           Gross demand this period.
    K_all : np.ndarray Resalable return count per replication, shape (J,).

    Returns
    -------
    np.ndarray, shape (J,)  N_t^(j).
    """
    return np.maximum(Gt - K_all, 0).astype(float)


# ════════════════════════════════════════════════════════════════════
# S3 — Demand-before-returns (closed form)
# ════════════════════════════════════════════════════════════════════
def _net_demand_S3(Gt: int, K_all: np.ndarray) -> np.ndarray:
    """
    Demand-before-returns: all demand arrives before any return, so no
    resalable return can offset within-period demand.

        N^(j) = Gt   (for every replication)

    The late resalable returns still have economic value (salvage), but that
    is captured in the cost parameters, NOT in the net demand here. This is
    the UPPER bound of the sequencing spectrum.

    Parameters
    ----------
    Gt : int           Gross demand this period.
    K_all : np.ndarray Shape (J,) — used only to size the output.

    Returns
    -------
    np.ndarray, shape (J,)  All entries equal Gt.
    """
    return np.full(len(K_all), float(Gt))


# ════════════════════════════════════════════════════════════════════
# S4 — Order-of-arrival netting (empirical, vectorised)  [from Colab]
# ════════════════════════════════════════════════════════════════════
def _net_demand_S4(
    Gt: int,
    K_all: np.ndarray,
    arrival_times_t: np.ndarray,
    return_arrival_times_t: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Order-of-arrival netting for one period, J replications, vectorised.

    A resalable return can only offset demand that arrives AFTER it within
    the period. We build a merged timeline, sort by arrival, and track a
    running buffer; demand units that arrive when the buffer is empty must
    be served from ordered stock — those are the net demand.

    Parameters
    ----------
    Gt : int
        Gross demand this period (fixed across replications).
    K_all : np.ndarray, shape (J,)
        Resalable return counts per replication (drawn upstream).
    arrival_times_t : np.ndarray
        Unit-level demand arrival fractions in [0,1) (empirical).
    return_arrival_times_t : np.ndarray
        Unit-level resalable-return arrival fractions in [0,1) (empirical).
    rng : np.random.Generator

    Returns
    -------
    np.ndarray, shape (J,)  N_t^(j).

    Implementation (Colab lines 400-451):
      encode returns in [0,1) and demands in [1,2) so returns win ties;
      merge + argsort; cumsum(+1 return, -1 demand); the magnitude of the
      most-negative cumsum is the count of demand units that hit an empty
      buffer.
    """
    J = len(K_all)
    if Gt == 0:
        return np.zeros(J)

    net_demands = np.empty(J)

    # Demand arrival fracs (fixed for the week; resample if size mismatch)
    if len(arrival_times_t) == Gt:
        d_fracs = arrival_times_t
    elif len(arrival_times_t) > 0:
        d_fracs = np.sort(rng.choice(arrival_times_t, size=Gt, replace=True))
    else:
        d_fracs = np.sort(rng.uniform(0, 1, Gt))

    # Batch replications by identical K value
    for K in np.unique(K_all):
        mask = (K_all == K)
        n_reps = int(mask.sum())

        if K == 0:
            net_demands[mask] = Gt          # no returns → all demand on stock
            continue

        # Return arrival fracs: (n_reps, K)
        if len(return_arrival_times_t) > 0:
            r_fracs = rng.choice(return_arrival_times_t,
                                 size=(n_reps, K), replace=True)
        else:
            r_fracs = rng.uniform(0, 1, size=(n_reps, K))

        # Encode: returns [0,1), demands [1,2) → returns first on ties
        eps = 1e-9
        r_enc = r_fracs
        d_enc = np.broadcast_to(d_fracs, (n_reps, Gt)) + eps

        times = np.concatenate([r_enc, d_enc], axis=1)
        values = np.concatenate([np.ones((n_reps, K)),
                                 -np.ones((n_reps, Gt))], axis=1)
        order = np.argsort(times, axis=1)
        sorted_v = np.take_along_axis(values, order, axis=1)

        cs = np.cumsum(sorted_v, axis=1)
        net_demands[mask] = np.maximum(0, -cs.min(axis=1))

    return net_demands


# ════════════════════════════════════════════════════════════════════
# Unified dispatcher
# ════════════════════════════════════════════════════════════════════
def net_demand_replications(
    Gt: int,
    K_all: np.ndarray,
    rule: str,
    arrival_times_t: np.ndarray = None,
    return_arrival_times_t: np.ndarray = None,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Dispatch to the requested sequencing rule and return N_t^(j), shape (J,).

    Parameters
    ----------
    Gt : int
        Gross demand this period.
    K_all : np.ndarray, shape (J,)
        Resalable return counts per replication. The SAME K_all should be
        passed across rules so that differences in N are attributable purely
        to sequencing (not to sampling noise).
    rule : str
        "S1", "S3", or "S4".
    arrival_times_t, return_arrival_times_t : np.ndarray, optional
        Empirical timestamps; required only for S4.
    rng : np.random.Generator, optional
        Required only for S4.

    Returns
    -------
    np.ndarray, shape (J,)  N_t^(j).
    """
    if Gt == 0:
        return np.zeros(len(K_all))

    if rule == "S1":
        return _net_demand_S1(Gt, K_all)

    elif rule == "S3":
        return _net_demand_S3(Gt, K_all)

    elif rule == "S4":
        if rng is None:
            raise ValueError("S4 requires an rng (np.random.Generator).")
        at = arrival_times_t if arrival_times_t is not None else np.array([])
        rt = return_arrival_times_t if return_arrival_times_t is not None else np.array([])
        return _net_demand_S4(Gt, K_all, at, rt, rng)

    else:
        raise ValueError(f"Unknown sequencing rule: {rule!r}. "
                         f"Choose from 'S1', 'S3', 'S4'.")


# ════════════════════════════════════════════════════════════════════
# Standalone test (run: python sequencing_rules.py)
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    Gt = 10
    K_all = np.array([0, 1, 3, 5, 8])     # same K across rules

    d_times = np.sort(rng.uniform(0, 1, Gt))
    r_times = np.array([0.3, 0.5, 0.7])

    print("Gt =", Gt, " K_all =", K_all.tolist(), "\n")

    results = {}
    for rule in ["S1", "S3", "S4"]:
        N = net_demand_replications(
            Gt, K_all, rule,
            arrival_times_t=d_times,
            return_arrival_times_t=r_times,
            rng=np.random.default_rng(0),
        )
        results[rule] = N
        print(f"  {rule}: N = {N.tolist()}")

    # Verify spectrum S1 <= S4 <= S3 elementwise
    print("\nSpectrum check (S1 <= S4 <= S3 elementwise):")
    ok = np.all(results["S1"] <= results["S4"]) and np.all(results["S4"] <= results["S3"])
    print(f"  {'PASS' if ok else 'FAIL'}")
    print("\nExpected:")
    print("  S1: [10, 9, 7, 5, 2]   (max(Gt-K,0))")
    print("  S3: [10,10,10,10,10]   (all Gt)")
    print("  S4: between S1 and S3")