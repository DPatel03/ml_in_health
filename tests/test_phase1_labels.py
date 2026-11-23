import pandas as pd

from build_phase1_labels import phase1_programs
from label_pipeline.aact import AactTables, started_trials


def test_observed_followup_and_entry_tie():
    trials = pd.DataFrame({
        "nct_id": ["B", "A", "C", "D", "E"],
        "phase": ["PHASE1", "PHASE1", "PHASE2", "PHASE2", "PHASE1/PHASE2"],
        "level": [1, 1, 2, 2, 2], "has_us_site": [True, False, False, True, True],
        "primary_purpose": ["DIAGNOSTIC", None, "TREATMENT", "TREATMENT", "TREATMENT"],
        "overall_status": ["COMPLETED", "COMPLETED", "WITHDRAWN", "NOT_YET_RECRUITING", "COMPLETED"],
        "start_date": ["2010-01-01", "2010-01-01", "2011-01-01", "2028-01-01", "2014-12-31"],
        "study_first_submitted_date": ["2026-10-08"] * 5,
    })
    observed, snapshot = started_trials(trials)
    mesh = pd.DataFrame(columns=["nct_id", "mesh_term"])
    tables = AactTables(observed, pd.DataFrame(), pd.Series(dtype=object), mesh, mesh, snapshot)
    mapping = pd.DataFrame({"nct_id": trials.nct_id, "drug_id": "CHEMBL_A", "drug_name": "a"})
    row = phase1_programs(mapping, tables, pd.Series(dtype="datetime64[ns]")).iloc[0]
    assert row.entry_nct_id == "A" and pd.isna(row.entry_primary_purpose)
    assert row.in_v1_label_set and row.next_date == pd.Timestamp("2014-12-31") and row.label == 1
