"""Synthetic, fully observed *potential demand* with transaction-level historical data.

The synthetic DGP is intentionally not a fitted Poisson or Normal demand model.
It is a demonstrative regime-based generator, NOT calibrated empirical evidence.
"""
from dataclasses import dataclass
import numpy as np
from config import ExperimentConfig, Economics
from simulation import PotentialMark, Scenario, SaleRecord, simulate


@dataclass(frozen=True)
class HistoricalSeason:
    season: int
    demand_times: tuple[float, ...]   # fully observed potential demand, even during stockout
    observed_sales: tuple[SaleRecord, ...]  # only fulfilled sales are observed
    order: int
    selling_days: float


@dataclass(frozen=True)
class ResolvedSale:
    returned: bool
    resalable: bool
    return_delay: float
    processing_delay: float


def generate_demand_times(rng: np.random.Generator, cfg: ExperimentConfig) -> tuple[float, ...]:
    """Illustrative non-Poisson event schedule with season-wide low/medium/promo regimes.

    Daily event counts are sampled from a *finite empirical-like discrete library*;
    this is still an explicit synthetic stochastic mechanism.
    """
    regime = str(rng.choice(["slow", "regular", "promotion"], p=[.25, .48, .27]))
    counts: list[int] = []
    days = int(cfg.selling_days)
    for day in range(days):
        weekend = day % 7 in (5, 6)
        if regime == "slow":
            options = [0, 1, 1, 2, 3]
        elif regime == "regular":
            options = [1, 2, 2, 3, 4]
        else:
            options = [2, 3, 4, 6, 8]
        daily = int(rng.choice(options))
        if weekend:
            daily += int(rng.choice([0, 1, 3]))
        counts.append(daily)
    times = [float(day+rng.uniform(0.04, .96)) for day, n in enumerate(counts)
             for _ in range(n)]
    return tuple(sorted(times))


def sample_true_scenario(rng: np.random.Generator, cfg: ExperimentConfig) -> Scenario:
    times = generate_demand_times(rng, cfg)
    marks = []
    for _ in times:
        returned = bool(rng.random() < cfg.true_return_probability)
        good = bool(rng.random() < cfg.true_resalability_probability) if returned else False
        d = float(rng.choice(cfg.return_delays)) if returned else 0.
        b = float(rng.choice(cfg.processing_delays)) if returned and good else 0.
        marks.append(PotentialMark(returned, good, d, b))
    return Scenario(times, tuple(marks), cfg.selling_days)


def generate_history(n: int, rng: np.random.Generator,
                     cfg: ExperimentConfig, ec: Economics) -> list[HistoricalSeason]:
    """Observe n historical seasons at a fixed feasible baseline stock policy.

    The decisions never observe unsold customers' counterfactual return marks.
    Demand is fully observed (by synthetic experimental design).
    """
    result = []
    for t in range(1, n+1):
        sc = sample_true_scenario(rng, cfg)
        realized = simulate(cfg.history_orders, sc, ec)
        result.append(HistoricalSeason(t, sc.demand_times, realized.sales_records,
                                       cfg.history_orders, cfg.selling_days))
    return result


def completed_sale_cohort(history: list[HistoricalSeason], cfg: ExperimentConfig) -> list[ResolvedSale]:
    """Use only sale windows matured by the ordering date following last history season.

    For simplicity, historical seasons are consecutive and equally spaced in global time.
    Transactions not fully mature are excluded from the complete-cohort estimator,
    even if an early return is already known. This conservative exclusion avoids
    look-ahead; all included returns also finish processing by the decision date.
    """
    if not history:
        raise ValueError("Need historical observations")
    decision_time = max(s.season for s in history)*cfg.selling_days
    rows = []
    for s in history:
        start = (s.season-1)*cfg.selling_days
        for record in s.observed_sales:
            elapsed = decision_time - (start + record.sale_time)
            if elapsed < (cfg.return_window_days + max(cfg.processing_delays)):
                continue
            rows.append(ResolvedSale(record.returned, record.resalable,
                        record.return_delay, record.processing_delay))
    return rows


def history_records(history: list[HistoricalSeason], cfg: ExperimentConfig) -> list[dict]:
    """CSV-ready history records with observation states as of next ordering date."""
    decision_time = max(s.season for s in history)*cfg.selling_days
    rows=[]
    for s in history:
        season_start=(s.season-1)*cfg.selling_days
        for tr in s.observed_sales:
            age=decision_time-(season_start+tr.sale_time)
            # Returns observed before cutoff are recorded; some remain pending otherwise.
            if tr.returned and tr.return_time is not None and tr.return_delay <= age:
                if tr.resalable and tr.ready_time is not None and tr.ready_time > tr.sale_time + age:
                    status="returned_awaiting_processing"
                else:
                    status="returned_processed"
            elif age >= cfg.return_window_days:
                status="closed_without_return"
            else:
                status="pending"
            observed_return = status in ("returned_processed", "returned_awaiting_processing")
            rows.append(dict(season=s.season,j=tr.j,sale_time=tr.sale_time,
                             returned_observed=observed_return,status=status,
                             return_time=(tr.return_time if observed_return else None),
                             resalable_observed=(tr.resalable if status=="returned_processed" else None)))
    return rows