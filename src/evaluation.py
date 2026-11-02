import numpy as np
from simulation import simulate

def evaluate(q,scenarios,ec):
    outcomes=[simulate(q,sc,ec) for sc in scenarios]
    return dict(order=q,mean_mismatch_cost=float(np.mean([o.mismatch_cost for o in outcomes])),
        mean_net_cash_cost=float(np.mean([o.net_cash_cost for o in outcomes])),
        mean_sales=float(np.mean([o.sales for o in outcomes])),
        mean_lost_sales=float(np.mean([o.lost_sales for o in outcomes])),
        mean_returns=float(np.mean([o.accepted_returns for o in outcomes])),
        mean_rejected_late_attempts=float(np.mean([o.rejected_late_attempts for o in outcomes])),
        mean_resales_actual=float(np.mean([o.actual_resales for o in outcomes])),
        mean_resalable_ready=float(np.mean([o.ontime_resalable_returns for o in outcomes])),
        mean_late_resalable=float(np.mean([o.late_resalable_returns for o in outcomes])),
        mean_nonresalable=float(np.mean([o.nonresalable_returns for o in outcomes])),
        mean_leftover=float(np.mean([o.leftover for o in outcomes])),
        fill_rate=float(sum(o.sales for o in outcomes)/max(1,sum(o.potential_demand for o in outcomes))))
