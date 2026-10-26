"""Configuration for a single-item, repeated-season newsvendor with customer returns."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Economics:
    price: float = 25.0
    procurement_cost: float = 8.0
    salvage_value: float = 2.0
    return_processing_cost: float = 1.5
    nonresalable_disposal_cost: float = 1.0

    def __post_init__(self):
        if not 0 <= self.salvage_value < self.procurement_cost <= self.price:
            raise ValueError("Require 0 <= salvage < procurement <= price for finite-bound search")
        if self.return_processing_cost < 0 or self.nonresalable_disposal_cost < 0:
            raise ValueError("Processing/disposal costs must be nonnegative")


@dataclass(frozen=True)
class ExperimentConfig:
    selling_days: float = 30.0
    return_window_days: float = 30.0
    # Numerical illustration only -- not calibrated to a retail sector.
    true_return_probability: float = 0.52
    true_resalability_probability: float = 0.70
    return_delays: tuple[float, ...] = (2., 6., 12., 20., 28.)
    processing_delays: tuple[float, ...] = (0.25, 1., 2.5)
    # Discrete delay support for baseline return process and Bayesian prior support.
    saa_scenarios: int = 140
    bayes_scenarios: int = 140
    evaluation_scenarios: int = 250
    reference_scenarios: int = 500
    history_orders: int = 58
    seed: int = 20261026

    def __post_init__(self):
        if not 0 <= self.true_return_probability <= 1 or not 0 <= self.true_resalability_probability <= 1:
            raise ValueError("Probabilities must be in [0,1]")
        if max(self.return_delays) > self.return_window_days:
            raise ValueError("Delays cannot exceed the eligibility window")
        if any(d < 0 for d in (*self.return_delays, *self.processing_delays)):
            raise ValueError("Delays must be nonnegative")