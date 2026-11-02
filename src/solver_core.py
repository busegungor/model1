"""Exact finite-grid integer enumeration by vectorizing all Q for each scenario.
No Gurobi required. The event-dependent resale process is recomputed for each Q.
"""
from dataclasses import dataclass
import numpy as np
from config import Economics
from simulation import Scenario,cash_cost

@dataclass(frozen=True)
class Solution:
    method:str
    q:int
    estimated_cost:float
    q_grid:tuple[int,...]
    cost_curve:tuple[float,...]


def _curve_one(sc:Scenario,ec:Economics,cap:int):
    q=np.arange(cap+1,dtype=np.int32)
    inv=q.copy()
    sales=np.zeros(cap+1,dtype=np.int32)
    accepted=np.zeros(cap+1,dtype=np.int32)
    nres=np.zeros(cap+1,dtype=np.int32)
    late_good=np.zeros(cap+1,dtype=np.int32)
    # event kind 0: return-ready; kind 1: demand (ties: inflow first).
    events=[]
    for j,(t,m) in enumerate(zip(sc.demand_times,sc.marks)):
        events.append((t,1,j))
        if m.accepted and m.resalable and m.ready_at(t)<=sc.horizon:
            events.append((m.ready_at(t),0,j))
    events.sort()
    masks={}
    for t,kind,j in events:
        if kind==0:
            inv+=masks.pop(j)
        else:
            served=(inv>0).astype(np.int32)
            inv-=served
            sales+=served
            m=sc.marks[j]
            if m.accepted:
                accepted+=served
                if m.resalable:
                    if m.ready_at(t)<=sc.horizon: masks[j]=served
                    else: late_good+=served
                else: nres+=served
    cash=(ec.procurement_cost*q-ec.price*(sales-accepted)+
         (ec.return_processing_cost+ec.reverse_transport_cost)*accepted+
         ec.nonresalable_disposal_cost*nres-ec.salvage_value*(inv+late_good)-
         ec.nonresalable_salvage_value*nres)
    return cash+(ec.price-ec.procurement_cost)*len(sc.demand_times)


def optimize(method,scenarios,ec):
    if not scenarios: raise ValueError('Need scenarios')
    cap=max(len(sc.demand_times) for sc in scenarios)
    average=np.zeros(cap+1,dtype=float)
    for sc in scenarios: average+=_curve_one(sc,ec,cap)/len(scenarios)
    ix=int(np.argmin(average))
    return Solution(method,ix,float(average[ix]),tuple(range(cap+1)),tuple(average))
