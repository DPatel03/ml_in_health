import pandas as pd

from build_phase1_model import split_phase1


def test_phase1_cutoff_requires_a_complete_window():
    programs = pd.DataFrame({"drug_id": ["A", "B", "C"], "window_years": [5] * 3,
                             "entry_date": pd.to_datetime(["2010-12-31", "2011-01-01", "2016-01-01"])})
    assert split_phase1(programs).split.tolist() == ["train", "dropped", "test"]
