"""Single-SKU seasonal apparel illustration; parameters are explicitly traceable.
All time units are CALENDAR days. Values labeled design choices are NOT estimates.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class Economics:
    # Illustrative joint vector: p,c,s individually lie inside the ranges in
    # Mostard & Teunter (2006); they are NOT an observed retailer-SKU tuple.
    price: float = 49.0
    procurement_cost: float = 20.0
    salvage_value: float = 5.0
    # Asdecker & Karl (2022), Fashion/S2, Table 8: EUR per returned item.
    reverse_transport_cost: float = 1.41
    return_processing_cost: float = 1.22
    nonresalable_salvage_value: float = 0.0  # experimental convention, no repairs
    nonresalable_disposal_cost: float = 0.0   # not sourced; exclude in baseline
    def __post_init__(self):
        if not 0 <= self.nonresalable_salvage_value <= self.salvage_value < self.procurement_cost <= self.price:
            raise ValueError('Require 0 <= nonresalable salvage <= salvage < cost <= price')
        if min(self.reverse_transport_cost,self.return_processing_cost,self.nonresalable_disposal_cost)<0:
            raise ValueError('Negative return cost')

@dataclass(frozen=True)
class ExperimentConfig:
    sku_id: str = 'SEASONAL_APPAREL_SKU_001'
    season_days: int = 126                   # 18 weeks; Hausman & Sides (1973) setting
    first_season_year: int = 2000            # arbitrary synthetic calendar index
    mean_weekly_units: float = 150.0        # DESIGN CHOICE; not an empirical estimate
    weekly_dispersion: float = 12.0         # TRUE DGP design: variance=mu+mu^2/kappa
    bayesian_assumed_dispersion: float = 12.0  # ESTIMATION assumption, independent of true DGP setting
    customer_geometric_p: float = 0.98      # DESIGN CHOICE, not fitted to this SKU
    # No fashion source was found for these precise mean/daily weights; flat weekly profile.
    return_attempt_probability: float = 0.53   # cross-SKU average, Dzyabura et al. (2023)
    return_resalable_probability: float = 0.9421  # Fashion/S2 'sold as new', Asdecker & Karl (2022); proxy
    late_dispatch_given_attempt: float = 0.03  # DESIGN CHOICE, no empirical rate available
    return_window_days: int = 30
    late_dispatch_last_day: int = 45         # DESIGN CHOICE, not empirically observed support
    processing_days: int = 2                # deterministic; sensitivity 1,2,3,5
    # Discrete maximum-entropy tilts on these USER-SELECTED supports; not observed extrema.
    outbound_days: tuple[int,...] = tuple(range(1,11))
    reverse_days: tuple[int,...] = tuple(range(1,6))
    outbound_mean_target: float = 4.93       # Masuch et al. (2024)
    outbound_p_le2_target: float = 0.079      # Masuch et al. (2024)
    reverse_mean_target: float = 2.24        # Asdecker & Karl (2022), Fashion/S2
     # Customer dispatch delay L^C = 1 + NegBin(r, p), truncated to 1..return_window_days.
    # (r, p) fitted to the delivery-to-receipt curve in Difrancesco et al. (2018), Fig. 16,
    # after adding the reverse-transport delay; r = 1 would be the geometric special case.
    dispatch_nb_r: float = 3.495
    dispatch_nb_p: float = 0.2629
    history_stock_fraction: float = 0.58   # DESIGN CHOICE, not observed
    saa_scenarios: int = 20
    bayes_scenarios: int = 20
    evaluation_scenarios: int = 40
    reference_scenarios: int = 60
    seed: int = 20261029
    # Return reason labels are illustrative; no empirical frequencies asserted.
    return_reasons: tuple[str,...] = ('too_small','too_large','not_as_expected',
                                     'no_longer_wanted','defective','wrong_item')
    def __post_init__(self):
        if self.season_days<=0 or self.season_days%7:
            raise ValueError('Season length must be positive whole weeks')
        if self.mean_weekly_units<=0 or self.weekly_dispersion<=0 or self.bayesian_assumed_dispersion<=0: raise ValueError('Positive demand parameters required')
        if not 0<self.customer_geometric_p<=1: raise ValueError('Geometric p in (0,1]')
        for p in (self.return_attempt_probability,self.return_resalable_probability,
                  self.late_dispatch_given_attempt,self.dispatch_nb_p):
            if not 0<=p<=1: raise ValueError('Probability outside [0,1]')
        if self.processing_days<0 or self.return_window_days<1 or self.late_dispatch_last_day<=self.return_window_days:
            raise ValueError('Invalid time limits')
            if self.dispatch_nb_r<=0 or not 0<self.dispatch_nb_p<1: raise ValueError('Invalid dispatch-delay parameters')
