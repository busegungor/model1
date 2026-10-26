"""Shared event-time physical process and return-adjusted cost accounting.

j = ordinal index of a *potential unit-demand event*, sorted by timestamp.
There is no third sale-event index: event j owns its eventual return outcome.
Returns from a fulfilled sale may complete processing and be resold again.
"""
from dataclasses import dataclass
from heapq import heappush, heappop
from typing import Tuple
from config import Economics


@dataclass(frozen=True)
class PotentialMark:
    returns: bool = False
    resalable: bool = False
    return_delay: float = 0.0
    processing_delay: float = 0.0


@dataclass(frozen=True)
class Scenario:
    demand_times: Tuple[float, ...]  # potential, fully observed demand within a season
    marks: Tuple[PotentialMark, ...]  # counterfactual marks NOT in decision maker's data
    horizon: float = 30.0

    def __post_init__(self):
        if len(self.demand_times) != len(self.marks):
            raise ValueError("One potential mark required per demand event")
        if any(not 0 <= a < self.horizon for a in self.demand_times):
            raise ValueError("Demand timestamps must be inside season")
        if tuple(sorted(self.demand_times)) != self.demand_times:
            raise ValueError("Demand timestamps must be sorted")
        if any(x.return_delay < 0 or x.processing_delay < 0 for x in self.marks):
            raise ValueError("Negative delays are invalid")


@dataclass(frozen=True)
class SaleRecord:
    j: int
    sale_time: float
    returned: bool
    resalable: bool
    return_delay: float
    processing_delay: float
    return_time: float | None
    ready_time: float | None


@dataclass(frozen=True)
class Outcome:
    q: int
    potential_demand: int
    sales: int
    lost_sales: int
    ontime_resalable_returns: int
    late_resalable_returns: int
    nonresalable_returns: int
    total_returns: int
    leftover: int
    sales_records: Tuple[SaleRecord, ...]
    net_cash_cost: float
    mismatch_cost: float

    def check(self, ec: Economics):
        assert self.potential_demand == self.sales + self.lost_sales
        assert self.leftover == self.q - self.sales + self.ontime_resalable_returns
        assert self.total_returns == (self.ontime_resalable_returns +
                                      self.late_resalable_returns + self.nonresalable_returns)
        assert self.leftover >= 0
        assert abs(self.mismatch_cost - (self.net_cash_cost +
                   (ec.price-ec.procurement_cost)*self.potential_demand)) < 1e-7


def simulate(q: int, sc: Scenario, ec: Economics) -> Outcome:
    if not isinstance(q, int) or q < 0:
        raise ValueError("Order quantity must be a nonnegative integer")
    inventory = q
    inflow_heap: list[float] = []
    sales = lost = k = h = n = r = 0
    transactions: list[SaleRecord] = []
    for j, (arrival, mark) in enumerate(zip(sc.demand_times, sc.marks), start=1):
        # Only returns made resale-ready BEFORE this demand event can satisfy it.
        while inflow_heap and inflow_heap[0] <= arrival:
            heappop(inflow_heap)
            inventory += 1
            k += 1
        if inventory == 0:
            lost += 1
            continue
        inventory -= 1
        sales += 1
        returned = bool(mark.returns)
        good = bool(mark.resalable) if returned else False
        rt = arrival + mark.return_delay if returned else None
        ready = rt + mark.processing_delay if returned and good else None
        if returned:
            r += 1
            if not good:
                n += 1
            elif ready is not None and ready <= sc.horizon:
                heappush(inflow_heap, ready)
            else:
                h += 1
        transactions.append(SaleRecord(j, arrival, returned, good,
                         mark.return_delay if returned else 0.,
                         mark.processing_delay if returned and good else 0.,
                         rt, ready))
    # Returns available by season end after last demand also count as terminal leftover.
    while inflow_heap and inflow_heap[0] <= sc.horizon:
        heappop(inflow_heap)
        inventory += 1
        k += 1
    # Attribution to the originating selling season includes future refunds/costs.
    cash = (ec.procurement_cost*q - ec.price*(sales-r) +
            ec.return_processing_cost*r +
            ec.nonresalable_disposal_cost*n -
            ec.salvage_value*(inventory+h))
    loss = ((ec.price-ec.procurement_cost)*lost +
            (ec.procurement_cost-ec.salvage_value)*inventory +
            (ec.price-ec.procurement_cost+ec.return_processing_cost)*k +
            (ec.price-ec.salvage_value+ec.return_processing_cost)*h +
            (ec.price+ec.return_processing_cost+ec.nonresalable_disposal_cost)*n)
    result = Outcome(q, len(sc.demand_times), sales, lost, k, h, n, r,
                     inventory, tuple(transactions), cash, loss)
    result.check(ec)
    return result