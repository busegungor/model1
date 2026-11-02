"""Event-time stock dynamics with accepted/refused commercial returns.
One potential *unit* event may share an order_id/time with other units.
Actual return outcomes only exist for served sales. Physical unit IDs track resales.
"""
from dataclasses import dataclass
from collections import deque
from heapq import heappush, heappop
from typing import Tuple
from config import Economics

@dataclass(frozen=True)
class PotentialMark:
    attempts_return: bool = False
    accepted: bool = False
    resalable: bool = False
    outbound_days: int = 0
    customer_days: int = 0
    reverse_days: int = 0
    processing_days: int = 0
    reason: str = ''
    def ready_at(self,sale_time:float)->float:
        return sale_time+self.outbound_days+self.customer_days+self.reverse_days+self.processing_days
    def dispatch_at(self,sale_time:float)->float:
        return sale_time+self.outbound_days+self.customer_days
    def received_at(self,sale_time:float)->float:
        return self.dispatch_at(sale_time)+self.reverse_days

@dataclass(frozen=True)
class Scenario:
    demand_times: Tuple[float,...]
    marks: Tuple[PotentialMark,...]
    horizon: float = 126.0
    order_ids: Tuple[str,...] = ()
    def __post_init__(self):
        if len(self.demand_times)!=len(self.marks): raise ValueError('Missing potential marks')
        if self.order_ids and len(self.order_ids)!=len(self.demand_times): raise ValueError('Order ID mismatch')
        if tuple(sorted(self.demand_times)) != self.demand_times: raise ValueError('Demand not time sorted')
        if any(not 0<=t<self.horizon for t in self.demand_times): raise ValueError('Demand outside season')
        if any(m.accepted and not m.attempts_return for m in self.marks): raise ValueError('Cannot accept absent return')
        if any(m.resalable and not m.accepted for m in self.marks): raise ValueError('Cannot restock rejected return')

@dataclass(frozen=True)
class SaleRecord:
    j: int
    sale_time: float
    physical_unit_id: int
    sale_number_for_unit: int
    mark: PotentialMark
    @property
    def returned(self): return self.mark.accepted
    @property
    def resalable(self): return self.mark.resalable
    @property
    def return_time(self): return self.mark.received_at(self.sale_time) if self.mark.attempts_return else None
    @property
    def ready_time(self): return self.mark.ready_at(self.sale_time) if self.mark.resalable else None

@dataclass(frozen=True)
class Outcome:
    q: int
    potential_demand: int
    sales: int
    lost_sales: int
    accepted_returns: int
    rejected_late_attempts: int
    ontime_resalable_returns: int
    late_resalable_returns: int
    nonresalable_returns: int
    actual_resales: int
    leftover: int
    records: Tuple[SaleRecord,...]
    net_cash_cost: float
    mismatch_cost: float
    def check(self,ec:Economics):
        assert self.potential_demand==self.sales+self.lost_sales
        assert self.accepted_returns==(self.ontime_resalable_returns+
                                      self.late_resalable_returns+self.nonresalable_returns)
        assert self.leftover==self.q-self.sales+self.ontime_resalable_returns
        assert self.leftover>=0
        assert self.rejected_late_attempts<=self.sales
        assert abs(self.mismatch_cost-(self.net_cash_cost+(ec.price-ec.procurement_cost)*self.potential_demand))<1e-6

def cash_cost(ec:Economics,q:int,sales:int,accepted:int,nonresalable:int,ending:int,late_good:int)->float:
    """Refund full price only for ACCEPTED returns, including after season closure."""
    return (ec.procurement_cost*q-ec.price*(sales-accepted)+
            (ec.return_processing_cost+ec.reverse_transport_cost)*accepted+
            ec.nonresalable_disposal_cost*nonresalable-
            ec.salvage_value*(ending+late_good)-ec.nonresalable_salvage_value*nonresalable)

def simulate(q:int,sc:Scenario,ec:Economics)->Outcome:
    if not isinstance(q,int) or q<0: raise ValueError('Q must be nonnegative integer')
    available=deque(range(1,q+1))
    inflows=[]
    sale_number={}
    records=[]
    sales=lost=accepted=late_attempt=on_time=late_good=nonres=actual_resales=0
    for j,(t,mark) in enumerate(zip(sc.demand_times,sc.marks),start=1):
        while inflows and inflows[0][0]<=t:
            _,_,uid=heappop(inflows)
            available.append(uid)
            on_time+=1
        if not available:
            lost+=1
            continue
        uid=available.popleft()
        sn=sale_number.get(uid,0)+1
        sale_number[uid]=sn
        actual_resales+=(sn>1)
        sales+=1
        if mark.attempts_return and not mark.accepted:
            late_attempt+=1
        if mark.accepted:
            accepted+=1
            if mark.resalable:
                ready=mark.ready_at(t)
                if ready<=sc.horizon:
                    heappush(inflows,(ready,j,uid))
                else:
                    late_good+=1
            else: nonres+=1
        records.append(SaleRecord(j,t,uid,sn,mark))
    while inflows and inflows[0][0]<=sc.horizon:
        _,_,uid=heappop(inflows)
        available.append(uid)
        on_time+=1
    ending=len(available)
    cash=cash_cost(ec,q,sales,accepted,nonres,ending,late_good)
    loss=cash+(ec.price-ec.procurement_cost)*len(sc.demand_times)
    out=Outcome(q,len(sc.demand_times),sales,lost,accepted,late_attempt,on_time,
                late_good,nonres,actual_resales,ending,tuple(records),cash,loss)
    out.check(ec)
    return out
