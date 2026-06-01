"""
sequencing_rules.py — Intraperiod sequencing rules for net-demand netting.

Given gross demand G_t and resalable returns K_t^(j) per replication, the
EFFECTIVE NET DEMAND N_t^(j) depends on WHEN returns arrive relative to
demand within the period. We characterize the spectrum with three rules:

  S1  Aggregate netting (classical, lower bound):  N = max(G - K, 0)
      All resalable returns offset demand; timing ignored. The assumption
      in Vlachos & Dekker (2003), Mostard & Teunter (2006). Equals S2
      (returns-before-demand). BEST case (max offset).

  S3  Demand-before-returns (worst case, upper bound):  N = G
      All demand arrives before any return; no return offsets within-period
      demand. WORST case (zero offset). The economic value of those late
      resalable returns is captured as SALVAGE in the cost parameters.

  S4  Order-of-arrival (empirical, between bounds):  interleaved netting
      Returns/demand merged in empirical arrival order; a resalable return
      offsets only demand arriving AFTER it. The paper's main model =
      AVERAGE/realized case.  Spectrum:  S1 (==S2)  <=  S4  <=  S3.

A given period's REAL timing falls somewhere in this band; classify_period_
timing() below reads WHERE it falls (best / average / worst) directly from
the same S1/S4 netting functions — no new rule, just their ratio.

ALL THREE share the same upstream binomial draws (K_all), so differences in
N are attributable purely to sequencing, not to sampling noise.
"""

import numpy as np


# ════════════════════════════════════════════════════════════════════
# S1 — Aggregate netting (closed form) — BEST case
# ════════════════════════════════════════════════════════════════════
def _net_demand_S1(Gt: int, K_all: np.ndarray) -> np.ndarray:
    """N^(j) = max(Gt - K^(j), 0). Max offset; lower bound of the spectrum."""
    return np.maximum(Gt - K_all, 0).astype(float)


# ════════════════════════════════════════════════════════════════════
# S3 — Demand-before-returns (closed form) — WORST case
# ════════════════════════════════════════════════════════════════════
def _net_demand_S3(Gt: int, K_all: np.ndarray) -> np.ndarray:
    """N^(j) = Gt for every replication. Zero offset; upper bound."""
    return np.full(len(K_all), float(Gt))


# ════════════════════════════════════════════════════════════════════
# S4 — Order-of-arrival netting (empirical, vectorised) — AVERAGE/realized
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
    A resalable return can only offset demand arriving AFTER it.

    encode returns in [0,1) and demands in [1,2) so returns win ties;
    merge + argsort; cumsum(+1 return, -1 demand); the magnitude of the
    most-negative cumsum is the count of demand units that hit an empty
    buffer (= net demand).
    """
    J = len(K_all)
    if Gt == 0:
        return np.zeros(J)

    net_demands = np.empty(J)

    if len(arrival_times_t) == Gt:
        d_fracs = arrival_times_t
    elif len(arrival_times_t) > 0:
        d_fracs = np.sort(rng.choice(arrival_times_t, size=Gt, replace=True))
    else:
        d_fracs = np.sort(rng.uniform(0, 1, Gt))

    for K in np.unique(K_all):
        mask = (K_all == K)
        n_reps = int(mask.sum())

        if K == 0:
            net_demands[mask] = Gt
            continue

        if len(return_arrival_times_t) > 0:
            r_fracs = rng.choice(return_arrival_times_t,
                                 size=(n_reps, K), replace=True)
        else:
            r_fracs = rng.uniform(0, 1, size=(n_reps, K))

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
    """Dispatch to S1/S3/S4 and return N_t^(j), shape (J,)."""
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
        raise ValueError(f"Unknown sequencing rule: {rule!r}. Choose 'S1','S3','S4'.")


# ════════════════════════════════════════════════════════════════════
# Period timing classification  — BEST / AVERAGE / WORST
# ════════════════════════════════════════════════════════════════════
def classify_period_timing(
    Gt: int,
    K_all: np.ndarray,
    arrival_times_t: np.ndarray,
    return_arrival_times_t: np.ndarray,
    rng: np.random.Generator,
    best_thresh: float = 0.66,
    worst_thresh: float = 0.33,
) -> dict:
    """
    Where does THIS period's real intraperiod timing fall in the S1<=S4<=S3
    band? Read directly from the same netting functions — no new rule.

    position = offset_S4 / offset_S1
             = (Gt - N_S4) / (Gt - N_S1)
      offset_S1 = Gt - N_S1   : maximum possible offset (BEST, all returns net)
      offset_S4 = Gt - N_S4   : offset actually achieved under empirical timing
      offset_S3 = Gt - N_S3 = 0 : WORST (no offset)

    position in [0,1]:
      ~1  -> S4 nets almost everything possible  -> BEST  (returns arrive early)
      ~0  -> S4 nets almost nothing              -> WORST (returns arrive late)
      mid -> AVERAGE

    Averaged over the J replications (so one number per period, since a period
    falls in ONE place — best, average, or worst — not several at once).

    Returns
    -------
    dict: {position (float in [0,1] or nan), label (str), offset_S1_mean,
           offset_S4_mean}
    Position is nan when offset_S1 == 0 (no resalable returns drawn at all,
    so timing is undefined / irrelevant for this period).
    """
    if Gt == 0:
        return {"position": np.nan, "label": "n/a",
                "offset_S1_mean": 0.0, "offset_S4_mean": 0.0}

    N_S1 = _net_demand_S1(Gt, K_all)
    N_S4 = _net_demand_S4(Gt, K_all, arrival_times_t,
                          return_arrival_times_t, rng)

    offset_S1 = Gt - N_S1          # per replication, >= 0
    offset_S4 = Gt - N_S4

    # ratio per replication where offset_S1 > 0; average over those
    valid = offset_S1 > 0
    if not np.any(valid):
        return {"position": np.nan, "label": "n/a",
                "offset_S1_mean": 0.0, "offset_S4_mean": float(offset_S4.mean())}

    pos_reps = offset_S4[valid] / offset_S1[valid]
    position = float(np.clip(pos_reps.mean(), 0.0, 1.0))

    if position >= best_thresh:
        label = "best"
    elif position <= worst_thresh:
        label = "worst"
    else:
        label = "average"

    return {"position": position, "label": label,
            "offset_S1_mean": float(offset_S1.mean()),
            "offset_S4_mean": float(offset_S4.mean())}


def classify_panel(dataset, r_hat, k_hat, J=200, seed=0,
                   best_thresh=0.66, worst_thresh=0.33):
    """
    Classify every period of a category panel as best / average / worst by
    its empirical return-demand timing. Returns a DataFrame, one row/period,
    plus the distribution of labels (how many periods fall in each bucket).

    Uses the SAME binomial draws as the SAA net-demand matrix (U~Bin(Gt,r),
    K~Bin(U,k)) so the classification is consistent with the model.
    """
    import pandas as pd
    panel = dataset["panel"]
    arrival = dataset.get("arrival_times", {})
    return_arrival = dataset.get("return_arrival_times", {})
    rng = np.random.default_rng(seed)

    rows = []
    for idx in range(len(panel)):
        period = panel.index[idx]
        Gt = int(panel["G_t"].iloc[idx])
        at = arrival.get(period, np.array([]))
        rt = return_arrival.get(period, np.array([]))
        U_all = rng.binomial(Gt, r_hat, size=J)
        K_all = rng.binomial(U_all, k_hat)
        info = classify_period_timing(Gt, K_all, at, rt, rng,
                                      best_thresh, worst_thresh)
        rows.append({"period": str(period), "Gt": Gt,
                     "position": info["position"], "label": info["label"]})
    df = pd.DataFrame(rows)
    dist = df["label"].value_counts().to_dict()
    return df, dist


# ════════════════════════════════════════════════════════════════════
# Standalone test
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Gt = 10
    K_all = np.array([0, 1, 3, 5, 8])
    d_times = np.sort(rng.uniform(0, 1, Gt))

    print("Spectrum check (S1 <= S4 <= S3):")
    for rule in ["S1", "S3", "S4"]:
        N = net_demand_replications(Gt, K_all, rule,
                                    arrival_times_t=d_times,
                                    return_arrival_times_t=np.array([0.3, 0.5, 0.7]),
                                    rng=np.random.default_rng(0))
        print(f"  {rule}: {N.tolist()}")

    print("\nTiming classification on the same period:")
    # returns EARLY -> should lean 'best'
    early = classify_period_timing(Gt, K_all, d_times,
                                   np.array([0.05, 0.1, 0.15]),
                                   np.random.default_rng(0))
    # returns LATE -> should lean 'worst'
    late = classify_period_timing(Gt, K_all, d_times,
                                  np.array([0.85, 0.9, 0.95]),
                                  np.random.default_rng(0))
    print(f"  returns early: position={early['position']:.3f} -> {early['label']}")
    print(f"  returns late : position={late['position']:.3f} -> {late['label']}")