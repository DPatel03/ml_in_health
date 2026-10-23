import pandas as pd
import pytest

from label_pipeline import areas, bio, chembl, drugs, labels, names
from label_pipeline.aact import AactTables

SNAPSHOT = pd.Timestamp("2026-10-01")
D = pd.Timestamp


# ---------- names ----------

def test_compact_and_salts():
    assert names.compact("MK-3475") == "mk3475"
    assert names.compact("[14C]-Abemaciclib") == "abemaciclib"
    assert names.strip_salt("Naltrexone Hydrochloride") == "Naltrexone"
    assert names.strip_salt("Sodium") == "Sodium"  # never strip a name down to nothing


def test_name_keys_cover_parentheses_doses_and_forms():
    keys = names.name_keys("Pembrolizumab (MK-3475) 200 mg IV infusion")
    assert {"pembrolizumab", "mk3475"} <= keys
    assert not any(k.isdigit() for k in keys)


@pytest.mark.parametrize("name,combo,not_drug", [
    ("Drug A + Drug B", True, False), ("FOLFOX regimen", True, False), ("Placebo", False, True),
    ("Matching placebo tablets", False, True), ("Pembrolizumab", False, False),
])
def test_name_flags(name, combo, not_drug):
    assert names.is_combination(name) is combo
    assert names.is_not_a_drug(name) is not_drug


def test_research_code():
    assert names.is_research_code("ly2189265")
    assert not names.is_research_code("cd20")


# ---------- chembl ----------

def test_synonym_index_maps_salts_to_parent_and_drops_ambiguous_names():
    mols = [
        {"molecule_chembl_id": "C1", "pref_name": "DRUGONE", "molecule_hierarchy": {"parent_chembl_id": "C1"},
         "molecule_synonyms": [{"molecule_synonym": "ABC-123", "synonyms": "ABC-123"},
                               {"molecule_synonym": "SHARED", "synonyms": "SHARED"}]},
        {"molecule_chembl_id": "C1S", "pref_name": "DRUGONE HYDROCHLORIDE", "molecule_hierarchy": {"parent_chembl_id": "C1"},
         "molecule_synonyms": []},
        {"molecule_chembl_id": "C2", "pref_name": "DRUGTWO", "molecule_hierarchy": None,
         "molecule_synonyms": [{"molecule_synonym": "SHARED", "synonyms": "SHARED"}]},
    ]
    index, display = chembl.build_synonym_index(mols)
    assert index["abc123"] == "C1"
    assert index["drugonehydrochloride"] == "C1"
    assert index["drugtwo"] == "C2"
    assert "shared" not in index
    assert display["C1"] == "drugone"


# ---------- bio ----------

BIO_TEXT = "\n".join([
    "All indications 4414 52.0% 4933 28.9% 1928 57.8% 1453 90.6%",
    "Oncology 1628 48.8% 1732 24.6% 495 47.7% 324 92.0%",
    "Oncology 4179 5.3% 2551 10.8% 819 43.9% 324 92.0%",  # later LOA table: must be ignored
    "Gastroenterology* 186 8.3% 141 17.8% 68 51.9% 33 90.9%",
])


def test_bio_rows_keep_first_occurrence_and_strip_footnote_marks():
    rows = pd.DataFrame(bio.parse_rows(BIO_TEXT))
    onc = rows[(rows.disease_area == "Oncology") & (rows.transition == "phase1_to_phase2")]
    assert onc.success_rate.tolist() == [0.488]
    assert "Gastroenterology" in set(rows.disease_area)


def test_bio_benchmark_combines_phase3_and_nda_steps():
    bench = bio.benchmark(pd.DataFrame(bio.parse_rows(BIO_TEXT)))
    row = bench[(bench.disease_area == "All indications") & (bench.transition == "phase3_to_approval")]
    assert row.bio_rate.iat[0] == pytest.approx(0.578 * 0.906)
    assert row.bio_n.iat[0] == 1928


def test_bio_validate_rejects_bad_extraction():
    bad = pd.DataFrame(bio.parse_rows(BIO_TEXT.replace("52.0%", "92.0%")))
    with pytest.raises(ValueError, match="looks wrong"):
        bio.validate(bad)


# ---------- drug resolution ----------

@pytest.fixture
def resolver() -> drugs.Resolver:
    return drugs.Resolver(chembl_index={"pembrolizumab": "CHEMBL_P", "mk3475": "CHEMBL_P", "drugx": "CHEMBL_X"},
                          chembl_names={"CHEMBL_P": "pembrolizumab", "CHEMBL_X": "drugx"},
                          fda_index={"oldgeneric": "FDA:oldgeneric"}, fda_names={"FDA:oldgeneric": "oldgeneric"})


@pytest.mark.parametrize("name,alts,expected", [
    ("MK-3475", [], ("CHEMBL_P", drugs.Source.CHEMBL)),
    ("Study antibody", ["Pembrolizumab"], ("CHEMBL_P", drugs.Source.CHEMBL_ALT)),
    ("Oldgeneric 10 mg tablets", [], ("FDA:oldgeneric", drugs.Source.FDA)),
    ("ZZ-99887", [], ("CODE:zz99887", drugs.Source.CODE)),
    ("Mysterycompound", [], ("NAME:mysterycompound", drugs.Source.NAME)),
])
def test_resolver_order(resolver, name, alts, expected):
    assert resolver.resolve(name, alts) == expected


def test_resolver_refuses_names_pointing_at_two_drugs(resolver):
    assert resolver.resolve("pembrolizumab (drugx)", []) is None


# ---------- labels ----------

def test_classify_outcomes():
    cfg_years = pd.Series([5, 5, 5, 5])
    entry = pd.Series([D("2010-01-01"), D("2010-01-01"), D("2024-01-01"), D("2012-01-01")])
    nxt = pd.Series([D("2013-01-01"), pd.NaT, D("2025-01-01"), D("2011-06-01")])
    out = labels.classify(entry, nxt, cfg_years, SNAPSHOT).tolist()
    assert out == ["advanced", "not_advanced", "pending", "already_past"]


def test_approval_step_gets_extra_years():
    cfg = labels.LabelConfig(snapshot=SNAPSHOT, window_years=5, approval_extra_years=2)
    assert cfg.years_for("phase3_to_approval") == 7
    assert cfg.years_for("phase2_to_phase3") == 5


def _tables() -> AactTables:
    trials = pd.DataFrame({
        "nct_id": ["T1", "T2", "T3", "T4", "T5", "T6"],
        "phase": ["PHASE1", "PHASE2", "PHASE3", "PHASE2", "PHASE2", "PHASE3"],
        "level": [1, 2, 3, 2, 2, 3],
        "start": [D("2008-01-01"), D("2010-01-01"), D("2012-06-01"), D("2009-01-01"), D("2006-01-01"), D("2007-01-01")],
        "has_us_site": [True] * 6,
    })
    conditions = pd.DataFrame({"nct_id": ["T2", "T3", "T4", "T5", "T6"],
                               "mesh_term": ["Breast Neoplasms", "Triple Negative Breast Neoplasms", "Asthma",
                                             "Gout", "Gout"]})
    ancestors = pd.DataFrame({"nct_id": ["T2", "T3", "T3", "T4"],
                              "mesh_term": ["Neoplasms", "Breast Neoplasms", "Neoplasms", "Lung Diseases"]})
    return AactTables(trials=trials, interventions=pd.DataFrame(), other_names=pd.Series(dtype=object),
                      conditions=conditions, ancestors=ancestors, snapshot=SNAPSHOT)


def test_build_transitions_end_to_end():
    trial_drugs = pd.DataFrame({
        "nct_id": ["T1", "T2", "T3", "T4", "T5", "T6"],
        "drug_id": ["CHEMBL_A", "CHEMBL_A", "CHEMBL_A", "CHEMBL_B", "CHEMBL_OLD", "CHEMBL_OLD"],
        "drug_name": ["a", "a", "a", "b", "old", "old"],
    })
    approvals = pd.Series({"CHEMBL_A": D("2016-03-01"), "CHEMBL_OLD": D("1999-01-01")})
    transitions, stats = labels.build_transitions(trial_drugs, _tables(), approvals, first_entry_year=2005)
    labeled = labels.apply_window(transitions, labels.LabelConfig(snapshot=SNAPSHOT))
    got = {(r.transition, r.drug_id): r.label for r in labeled.itertuples()}

    assert stats["drugs_excluded_already_approved"] == 1  # CHEMBL_OLD was approved before its first trial
    assert got[("phase1_to_phase2", "CHEMBL_A")] == 1.0
    # Phase 3 in a narrower term (triple-negative breast cancer) still counts for the breast-cancer program
    assert got[("phase2_to_phase3", "CHEMBL_A")] == 1.0
    assert got[("phase3_to_approval", "CHEMBL_A")] == 1.0
    assert got[("phase2_to_phase3", "CHEMBL_B")] == 0.0
    assert not any(d == "CHEMBL_OLD" for _, d in got)
    assert labeled.in_v1_label_set.all()
    entry = {(r.transition, r.drug_id): r.entry_nct_id for r in labeled.itertuples()}
    assert entry == {("phase1_to_phase2", "CHEMBL_A"): "T1", ("phase2_to_phase3", "CHEMBL_A"): "T2",
                     ("phase3_to_approval", "CHEMBL_A"): "T3", ("phase2_to_phase3", "CHEMBL_B"): "T4"}


def test_entry_trial_is_per_disease_and_ties_break_to_lowest_nct_id():
    trials = pd.DataFrame({
        "nct_id": ["T12", "T10", "T11"],
        "phase": ["PHASE2", "PHASE2", "PHASE1/PHASE2"],
        "level": [2, 2, 2],
        "start": [D("2010-01-01"), D("2010-01-01"), D("2011-01-01")],
        "has_us_site": [True, True, True],
    })
    conditions = pd.DataFrame({"nct_id": ["T12", "T10", "T11"], "mesh_term": ["Psoriasis", "Psoriasis", "Crohn Disease"]})
    tables = AactTables(trials=trials, interventions=pd.DataFrame(), other_names=pd.Series(dtype=object),
                        conditions=conditions, ancestors=pd.DataFrame(columns=["nct_id", "mesh_term"]), snapshot=SNAPSHOT)
    trial_drugs = pd.DataFrame({"nct_id": ["T12", "T10", "T11"], "drug_id": ["CHEMBL_C"] * 3, "drug_name": ["c"] * 3})

    transitions, _ = labels.build_transitions(trial_drugs, tables, pd.Series(dtype="datetime64[ns]"), first_entry_year=2005)
    p2 = transitions[transitions.transition == "phase2_to_phase3"].set_index("mesh_term")

    assert p2.at["Psoriasis", "entry_nct_id"] == "T10"
    assert p2.at["Crohn Disease", "entry_nct_id"] == "T11"
    assert p2.at["Crohn Disease", "entry_date"] == D("2011-01-01")


# ---------- disease areas ----------

@pytest.mark.parametrize("terms,area", [
    ({"Leukemia, Myeloid", "Hematologic Diseases"}, "Oncology"),
    ({"Anemia, Sickle Cell", "Hematologic Diseases"}, "Hematology"),
    ({"Renal Insufficiency, Chronic", "Kidney Diseases", "Urologic Diseases"}, "Others"),
    ({"Prostatic Hyperplasia", "Urologic Diseases"}, "Urology"),
    (set(), "Unknown"),
])
def test_area_for(terms, area):
    assert areas.area_for(terms) == area
