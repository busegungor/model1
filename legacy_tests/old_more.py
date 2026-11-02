import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from config import Economics,ExperimentConfig
from data_generation import HistoricalSeason,history_records,completed_sale_cohort
from simulation import SaleRecord


def test_return_timestamps_respect_decision_time():
    cfg=ExperimentConfig(selling_days=30.,return_window_days=30.)
    # last sale in first season, returned 29 days later, which is visible by day 60.
    # last sale in second season, returned 29 days later, which is NOT visible by day 60.
    record=SaleRecord(j=1,sale_time=29.,returned=True,resalable=True,
                      return_delay=29.,processing_delay=1.,return_time=58.,ready_time=59.)
    hist=[HistoricalSeason(1,(29.,),(record,),1,30.),
          HistoricalSeason(2,(29.,),(record,),1,30.)]
    rows=history_records(hist,cfg)
    assert rows[0]['returned_observed'] is True
    assert rows[1]['returned_observed'] is False
    assert rows[1]['status']=='pending'
    cohort=completed_sale_cohort(hist,cfg)
    assert len(cohort)==0  # conservative window + max processing: 31 days