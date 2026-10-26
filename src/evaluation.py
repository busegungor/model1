"""Hold-out evaluation; approximate oracle is fitted on separate reference scenarios."""
import numpy as np
from config import Economics
from simulation import Scenario, simulate


def evaluate(q:int, scenarios:list[Scenario], ec:Economics)->dict[str,float]:
    outcomes=[simulate(q,sc,ec) for sc in scenarios]
    return dict(order=q,
                mean_mismatch_cost=float(np.mean([o.mismatch_cost for o in outcomes])),
                mean_net_cash_cost=float(np.mean([o.net_cash_cost for o in outcomes])),
                mean_lost_sales=float(np.mean([o.lost_sales for o in outcomes])),
                mean_sales=float(np.mean([o.sales for o in outcomes])),
                mean_returns=float(np.mean([o.total_returns for o in outcomes])),
                mean_resales_available=float(np.mean([o.ontime_resalable_returns for o in outcomes])),
                mean_late_resalable=float(np.mean([o.late_resalable_returns for o in outcomes])),
                mean_nonresalable=float(np.mean([o.nonresalable_returns for o in outcomes])),
                mean_leftover=float(np.mean([o.leftover for o in outcomes])),
                fill_rate=float(sum(o.sales for o in outcomes)/max(1,sum(o.potential_demand for o in outcomes))))