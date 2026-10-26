"""Same return-adjusted loss is used by SAA, Bayesian, oracle, and evaluation."""
from config import Economics
from simulation import Outcome


def equivalent_mismatch_loss(outcome: Outcome, ec: Economics) -> float:
    l, i, k, h, n = (outcome.lost_sales, outcome.leftover,
                      outcome.ontime_resalable_returns, outcome.late_resalable_returns,
                      outcome.nonresalable_returns)
    return ((ec.price-ec.procurement_cost)*l +
            (ec.procurement_cost-ec.salvage_value)*i +
            (ec.price-ec.procurement_cost+ec.return_processing_cost)*k +
            (ec.price-ec.salvage_value+ec.return_processing_cost)*h +
            (ec.price+ec.return_processing_cost+ec.nonresalable_disposal_cost)*n)