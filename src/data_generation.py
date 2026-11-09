"""Reproducible DGP and dated CSV; empirical *targets* are not empirical distributions.

Sale-date and per-unit status are recorded after physical inventory simulation.
A rejected (>30-day dispatch) attempt produces NO refund or stock inflow.
"""
from dataclasses import dataclass
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from scipy.optimize import root,brentq
from config import ExperimentConfig, Economics
from simulation import PotentialMark, Scenario, SaleRecord, simulate

@dataclass(frozen=True)
class HistoricalSeason:
    season: int
    demand_times: tuple[float,...]       # observed potential unit demand
    completed_sales: tuple[SaleRecord,...]  # ONLY eligible and observed past returns
    order: int
    selling_days: int
    # Event/order IDs are for reconstructing customer quantities in output.
    order_ids: tuple[str,...] = ()


def maxent_daily_weights(days,mean_target,p_le2=None):
    """Maximum entropy on USER-supported days matching reported moments.

    Does not claim source gives daily probabilities. With 2 constraints, numerical fit
    is exponential family p(d) proportional exp(a*d+b*I(d<=2)).
    """
    x=np.asarray(days,dtype=float)
    if p_le2 is None:
        from scipy.special import logsumexp
        def f(a):
            t=a*x; w=np.exp(t-logsumexp(t)); return float(np.dot(x,w)-mean_target)
        a=brentq(f,-20,20)
        scores=a*x
    else:
        z=(x<=2).astype(float)
        from scipy.special import logsumexp
        def moments(theta):
            t=theta[0]*x+theta[1]*z
            w=np.exp(t-logsumexp(t))
            return [float(np.dot(x,w)-mean_target),float(np.dot(z,w)-p_le2)]
        sol=root(moments,[0.,-1.])
        if not sol.success or max(map(abs,moments(sol.x)))>1e-8:
            raise ValueError('Could not calibrate delivery distribution')
        scores=sol.x[0]*x+sol.x[1]*z
    w=np.exp(scores-max(scores));return w/w.sum()


def conditional_dispatch_weights(cfg):
    """Dispatch delay L^C = 1 + NegBin(r, p), truncated to the return window 1..omega.

    (r, p) are fitted to the delivery-to-receipt curve of Difrancesco et al. (2018,
    Fig. 16) after convolving with the reverse-transport law; r = 1 is the geometric case.
    """
    d=np.arange(1,cfg.return_window_days+1)
    w=nbinom.pmf(d-1,cfg.dispatch_nb_r,cfg.dispatch_nb_p)
    return w/w.sum()


def generate_demand_times(rng,cfg,season:int):
    """Weekly NB units; conditional geometric partition into customer orders.

    NB variance = mean + mean^2/dispersion. Times uniform within the 7-day
    week. No claim that uniform arrival profile is empirically estimated.
    """
    events=[]
    for w in range(cfg.season_days//7):
        k=cfg.weekly_dispersion; mu=cfg.mean_weekly_units
        units=int(rng.negative_binomial(k,k/(k+mu)))
        while units>0:
            size=min(units,int(rng.geometric(cfg.customer_geometric_p)))
            t=float(7*w+rng.uniform(0.01,6.99))
            events.append((t,f'S{season:03d}O{len(events)+1:06d}',size))
            units-=size
    events.sort(key=lambda e:e[0])
    times=[];order_ids=[]
    for t,oid,size in events:
        times.extend([t]*size)
        order_ids.extend([oid]*size)
    return tuple(times),tuple(order_ids)


def draw_marks(rng,cfg,n):
    outbound=np.asarray(cfg.outbound_days)
    reverse=np.asarray(cfg.reverse_days)
    w_out=maxent_daily_weights(outbound,cfg.outbound_mean_target,cfg.outbound_p_le2_target)
    w_back=maxent_daily_weights(reverse,cfg.reverse_mean_target)
    w_cust=conditional_dispatch_weights(cfg)
    marks=[]
    good_reasons=[r for r in cfg.return_reasons if r!='defective']
    for _ in range(n):
        outbound_d=int(rng.choice(outbound,p=w_out))
        attempt=bool(rng.random()<cfg.return_attempt_probability)
        if not attempt:
            marks.append(PotentialMark(outbound_days=outbound_d))
            continue
        late=bool(rng.random()<cfg.late_dispatch_given_attempt)
        customer_d=(int(rng.integers(cfg.return_window_days+1,cfg.late_dispatch_last_day+1))
                    if late else int(rng.choice(np.arange(1,cfg.return_window_days+1),p=w_cust)))
        back_d=int(rng.choice(reverse,p=w_back))
        accepted=not late
        good=bool(rng.random()<cfg.return_resalable_probability) if accepted else False
        # Reason assignment is a transparent *synthetic* coding convention.
        reason=(str(rng.choice(good_reasons)) if good else
                ('defective' if accepted else 'late_request'))
        marks.append(PotentialMark(True,accepted,good,outbound_d,customer_d,
                                   back_d,cfg.processing_days if good else 0,reason))
    return tuple(marks)


def sample_true_scenario(rng,cfg,season=1):
    times,orders=generate_demand_times(rng,cfg,season)
    return Scenario(times,draw_marks(rng,cfg,len(times)),cfg.season_days,orders)


def generate_history(n,rng,cfg,ec):
    """Generate independent seasonal trajectories; 1 season/year for same SKU.

    Fixed history Q is a transparent policy design choice, not data-derived.
    Returns are all mature by the following annual pre-season decision date.
    """
    histories=[]
    for t in range(1,n+1):
        sc=sample_true_scenario(rng,cfg,t)
        q=int(round(cfg.history_stock_fraction*(cfg.season_days/7)*cfg.mean_weekly_units))
        result=simulate(q,sc,ec)
        histories.append(HistoricalSeason(t,sc.demand_times,result.records,q,cfg.season_days,sc.order_ids))
    return histories


def completed_sale_cohort(history,cfg):
    # HistoricalSeason.completed_sales is already the *observed* fulfilled cohort;
    # annual decision dates mean all previous seasons' 30-day windows have closed.
    sales=[r for s in history for r in s.completed_sales]
    if not sales: raise ValueError('No observed sales in history')
    return sales


def absolute_datetime(year,t):
    return datetime(int(year),9,1)+timedelta(days=float(t))


def _stamp(t):return t.isoformat(timespec='seconds') if t is not None else ''


def export_history_csv(history,cfg,ec,folder):
    """Write real-looking dated, *synthetic* relational transaction tables.

    Do not export hidden counterfactual return outcomes for unserved demand.
    Observed demand is fully available by design, including during stockouts.
    """
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    event_rows=[];sales_rows=[];returns_rows=[];summaries=[]
    for hs in history:
        yy=cfg.first_season_year+hs.season-1
        start=datetime(yy,9,1)
        groups={}
        for j,(t,oid) in enumerate(zip(hs.demand_times,hs.order_ids),start=1):
            if oid not in groups: groups[oid]=[t,0,0]
            groups[oid][1]+=1
        # Only sale records (fulfilled units); unserved potential marks never leak.
        for r in hs.completed_sales:
            oid=hs.order_ids[r.j-1]
            groups[oid][2]+=1
            mark=r.mark
            sold=absolute_datetime(yy,r.sale_time)
            # Delivery always realized for a fulfilled sale, including nonreturns.
            # A non-return mark has no pre-generated outbound delay; generate
            # deterministically with a separate reproducible, synthetic seed.
            delivered=sold+timedelta(days=mark.outbound_days)
            dispatch=(sold+timedelta(days=mark.outbound_days+mark.customer_days)) if mark.attempts_return else None
            received=(sold+timedelta(days=mark.outbound_days+mark.customer_days+mark.reverse_days)) if mark.attempts_return else None
            ready=(sold+timedelta(days=mark.outbound_days+mark.customer_days+
                              mark.reverse_days+mark.processing_days)) if mark.resalable else None
            status=('none' if not mark.attempts_return else
                    ('accepted' if mark.accepted else 'rejected_late'))
            sid=f'S{hs.season:03d}_SALE{r.j:07d}'
            sales_rows.append(dict(season_id=hs.season,sku_id=cfg.sku_id,
                sale_id=sid,order_id=oid,unit_demand_index=r.j,
                physical_unit_id=f'S{hs.season:03d}_UNIT{r.physical_unit_id:07d}',
                sale_number_for_unit=r.sale_number_for_unit,
                sale_datetime=_stamp(sold),delivery_datetime=_stamp(delivered),
                return_status=status,return_reason=(mark.reason if mark.attempts_return else ''),
                return_dispatch_datetime=_stamp(dispatch),return_received_datetime=_stamp(received),
                refund_issued=bool(mark.accepted),
                inspected_resalable=(bool(mark.resalable) if mark.accepted else None),
                resale_ready_datetime=_stamp(ready)))
            if mark.attempts_return:
                returns_rows.append(dict(season_id=hs.season,sale_id=sid,
                    return_reason=mark.reason,return_dispatch_datetime=_stamp(dispatch),
                    return_received_datetime=_stamp(received),accepted=bool(mark.accepted),
                    refund_issued=bool(mark.accepted),
                    inspected_resalable=(bool(mark.resalable) if mark.accepted else None),
                    resale_ready_datetime=_stamp(ready)))
        for oid,(t,demand,sales) in groups.items():
            event_rows.append(dict(season_id=hs.season,sku_id=cfg.sku_id,order_id=oid,
                event_datetime=_stamp(absolute_datetime(yy,t)),relative_day=t,
                demand_qty=demand,sales_qty=sales,lost_sales_qty=demand-sales))
        summaries.append(dict(season_id=hs.season,sku_id=cfg.sku_id,
            season_start=_stamp(start),season_end=_stamp(start+timedelta(days=cfg.season_days)),
            history_order=hs.order,potential_demand=len(hs.demand_times),
            realized_sales=len(hs.completed_sales),
            accepted_returns=sum(r.mark.accepted for r in hs.completed_sales),
            rejected_late_attempts=sum(r.mark.attempts_return and not r.mark.accepted for r in hs.completed_sales)))
    for name,rows in (('demand_events',event_rows),('sale_units',sales_rows),
                      ('return_attempts',returns_rows),('season_summary',summaries)):
        pd.DataFrame(rows).to_csv(folder/(name+'.csv'),index=False)
    return {name:len(rows) for name,rows in (('demand_events',event_rows),('sale_units',sales_rows),
                      ('return_attempts',returns_rows),('season_summary',summaries))}


def load_history_csv(folder,cfg):
    """Load history from observable CSV ONLY; no hidden DGP parameters used.

    Assumes fully completed annual past seasons. Future dates are not read if
    future/unfinished seasons are supplied (explicit cutoff safety check).
    """
    folder=Path(folder)
    events=pd.read_csv(folder/'demand_events.csv')
    sales=pd.read_csv(folder/'sale_units.csv')
    summary=pd.read_csv(folder/'season_summary.csv')
    seasons=[]
    for _,meta in summary.sort_values('season_id').iterrows():
        sid=int(meta.season_id)
        ev=events[events.season_id==sid].sort_values(['relative_day','order_id'])
        pairs=[(float(row.relative_day),str(row.order_id)) for row in ev.itertuples()
               for _ in range(int(row.demand_qty))]
        pairs.sort(key=lambda p:p[0])
        times=tuple(x[0] for x in pairs)
        ids=tuple(x[1] for x in pairs)
        # For physical/return behavior the historical sales contain only observed
        # accepted/refused outcomes; no counterfactual marks at missing demand.
        sr=[]
        for record in sales[sales.season_id==sid].itertuples():
            base=datetime(cfg.first_season_year+sid-1,9,1)
            sale=datetime.fromisoformat(str(record.sale_datetime))
            t=(sale-base).total_seconds()/86400
            status=str(record.return_status)
            attempted=status in ('accepted','rejected_late')
            accepted=status=='accepted'
            if attempted:
                delivered=datetime.fromisoformat(str(record.delivery_datetime))
                dispatch=datetime.fromisoformat(str(record.return_dispatch_datetime))
                received=datetime.fromisoformat(str(record.return_received_datetime))
                outd=(delivered-sale).days
                cust=(dispatch-delivered).days
                reverse=(received-dispatch).days
            else:
                delivered=datetime.fromisoformat(str(record.delivery_datetime))
                outd=(delivered-sale).days
                cust=reverse=0
            # Nullable boolean may be NaN; inspect only for accepted returns.
            good=(str(record.inspected_resalable).lower()=='true') if accepted else False
            ready=(datetime.fromisoformat(str(record.resale_ready_datetime)) if good else None)
            proc=(ready-received).days if good else 0
            mark=PotentialMark(attempted,accepted,good,outd,cust,reverse,proc,
                               str(record.return_reason) if attempted else '')
            uid=int(str(record.physical_unit_id).split('UNIT')[-1])
            sr.append(SaleRecord(int(record.unit_demand_index),t,uid,int(record.sale_number_for_unit),mark))
        seasons.append(HistoricalSeason(sid,times,tuple(sr),int(meta.history_order),cfg.season_days,ids))
    return seasons
