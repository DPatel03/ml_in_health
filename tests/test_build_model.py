import numpy as np
import pandas as pd
import pytest

import build_model as bm

D = pd.Timestamp


@pytest.mark.parametrize("raw,key", [
    ("Novartis Pharmaceuticals", "novartis"),
    ("Genentech, Inc.", "genentech"),
    ("Eli Lilly and Company", "eli lilly"),
    ("Novo Nordisk A/S", "novo nordisk"),
    ("Jiangsu HengRui Medicine Co., Ltd.", "jiangsu hengrui medicine"),
    ("Janssen Research & Development, LLC", "janssen research and development"),
    # ownership clauses are cut, not mapped to the parent (they reflect today's owner, not the trial's)
    ("Seagen, a wholly owned subsidiary of Pfizer", "seagen"),
    ("Wyeth is now a wholly owned subsidiary of Pfizer", "wyeth"),
    ("Sierra Oncology LLC - a GSK company", "sierra oncology"),
    ("AbbVie (prior sponsor, Abbott)", "abbvie"),
])
def test_normalize_sponsor(raw, key):
    assert bm.normalize_sponsor(raw) == key


def test_normalize_sponsor_keeps_different_companies_apart():
    assert bm.normalize_sponsor("Merck & Co., Inc.") != bm.normalize_sponsor("Merck KGaA, Darmstadt, Germany")
    assert bm.normalize_sponsor("(Hypothetical) Bio Inc.") != ""


def _programs() -> pd.DataFrame:
    # sponsor S: A succeeds fast (2006) but only counts once its window closes (2010)
    return pd.DataFrame({
        "name": ["A", "B", "F", "C", "E", "T1"],
        "sponsor": ["s", "s", "s", "s", "s", "t"],
        "entry_date": [D("2005-01-01"), D("2006-01-01"), D("2007-01-01"), D("2010-06-01"), D("2011-06-01"),
                       D("2005-01-01")],
        "next_date": [D("2006-01-01"), pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT],
        "label": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "window_years": [5] * 6,
    }).set_index("name", drop=False)


def test_track_record_counts_programs_only_once_their_window_closes():
    df = _programs()
    tr = pd.Series(bm.sponsor_track_record(df, k=0), index=df.index)
    assert np.isnan(tr["A"]) and np.isnan(tr["T1"])
    assert np.isnan(tr["F"])  # A's 2006 success is not counted yet in 2007
    assert tr["C"] == 1.0     # A closed 2010-01; B closes 2011-01
    assert tr["E"] == 0.5     # A and B closed


def test_track_record_shrinks_toward_market_rate_known_at_entry():
    df = _programs()
    tr = pd.Series(bm.sponsor_track_record(df, k=5), index=df.index)
    # C: sponsor 1/1, market (A, T1) 1/2 -> (1 + 2.5) / 6
    assert tr["C"] == pytest.approx(3.5 / 6)
    # E: sponsor 1/2, market (A, B, T1) 1/3 -> (1 + 5/3) / 7
    assert tr["E"] == pytest.approx((1 + 5 / 3) / 7)


def _split_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [  # drug, entry year, in label set, label
        ("X", 2010, True, 1.0), ("X", 2017, True, 0.0),   # older drug: 2017 program must not land in test
        ("Y", 2017, True, 1.0), ("Y", 2020, True, 0.0),
        ("Z", 2009, True, 0.0),
        ("LATE", 2015, True, 0.0), ("Q", 2013, False, 1.0), ("R", 2013, True, np.nan),  # out of the label set / pending
    ]
    transitions = pd.DataFrame({
        "transition": "phase2_to_phase3",
        "drug_id": [r[0] for r in rows],
        "entry_date": [D(f"{r[1]}-03-01") for r in rows],
        "entry_nct_id": [f"NCT{i}" for i in range(len(rows))],
        "next_date": pd.NaT,
        "in_v1_label_set": [r[2] for r in rows],
        "label": [r[3] for r in rows],
        "window_years": 5,
    })
    features = pd.DataFrame({"nct_id": transitions.entry_nct_id, "lead_sponsor_name": "Acme Pharmaceuticals, Inc."})
    return features, transitions


def test_split_keeps_each_drug_on_one_side():
    features, transitions = _split_inputs()
    df = bm.build_program_features(features, transitions, "phase2_to_phase3")
    got = {(d, y): s for d, y, s in zip(df.drug_id, df.entry_date.dt.year, df.split)}
    assert got == {("X", 2010): "train", ("Z", 2009): "train", ("Y", 2017): "test", ("Y", 2020): "test"}
    assert (df.sponsor == "acme").all()


def test_split_embargo_and_test_end():
    features, transitions = _split_inputs()
    df = bm.build_program_features(features, transitions, "phase2_to_phase3", train_end=2009, test_start=2016,
                                   test_end=2018)
    got = {(d, y): s for d, y, s in zip(df.drug_id, df.entry_date.dt.year, df.split)}
    assert got == {("Z", 2009): "train", ("Y", 2017): "test"}


def test_bio_phase3_rate_is_phase3_to_filing_times_filing_to_approval():
    flat, by_area = bm.load_bio_rates(bm.HERE / "external" / "bio" / "bio2021_phase_transitions.csv",
                                      "phase3_to_approval")
    assert flat == pytest.approx(0.578 * 0.906)
    assert by_area["Oncology"] == pytest.approx(0.477 * 0.92)


def test_approval_training_window_must_close_before_test_start():
    df = pd.DataFrame({"drug_id": list("abcd"), "window_years": [7] * 4,
                       "entry_date": pd.to_datetime(["2008-12-31", "2009-01-01", "2015-01-01", "2016-01-01"])})
    split = bm.split_programs(df, 2015, 2016, None)
    assert dict(zip(split.drug_id, split.split)) == {"a": "train", "d": "test"}


def test_watchlist_excludes_already_observed_phase3():
    from build_watchlist import build_pending_features

    features, transitions = _split_inputs()
    transitions["outcome"] = "pending"
    transitions["next_date"] = D("2024-01-01")
    transitions.loc[0, "next_date"] = pd.NaT
    pending = build_pending_features(features, transitions)
    assert pending.entry_nct_id.tolist() == [features.nct_id.iat[0]]
