"""Phase-transition labels.

Each row is one program entering a phase. We store when it entered and when (if ever) it reached the next step,
then derive the label for a given success window, so different windows don't need a rebuild.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from .aact import AactTables
from .areas import area_for

TRANSITIONS = ("phase1_to_phase2", "phase2_to_phase3", "phase3_to_approval")
PHASE2_ENTRY = ("PHASE1/PHASE2", "PHASE2")
# BIO tracks therapeutic development programs. Diagnostic / basic-science / supportive trials are left out of
# the v1 label set; missing purpose (common on older registrations) is kept.
V1_PURPOSES = frozenset({"TREATMENT", "PREVENTION"})


class Outcome(str, Enum):
    ADVANCED = "advanced"
    NOT_ADVANCED = "not_advanced"
    PENDING = "pending"            # window hasn't closed yet: no label, these feed the watchlist
    ALREADY_PAST = "already_past"  # next step started on or before the first registered trial of this phase


@dataclass(frozen=True)
class LabelConfig:
    snapshot: pd.Timestamp
    window_years: int = 5
    approval_extra_years: int = 2  # Phase 3 -> approval also needs filing and FDA review
    first_entry_year: int = 2005

    def years_for(self, transition: str) -> int:
        return self.window_years + (self.approval_extra_years if transition == TRANSITIONS[2] else 0)


def classify(entry: pd.Series, nxt: pd.Series, years: pd.Series, snapshot: pd.Timestamp) -> pd.Series:
    deadline = pd.Series([e + pd.DateOffset(years=int(y)) for e, y in zip(entry, years)], index=entry.index)
    # A program whose window is still open gets no label even if it already advanced; labelling only the
    # early successes would bias recent cohorts upward. A next step on the entry date itself is already past:
    # the outcome is known the moment the program enters, so there's nothing to predict.
    out = np.select(
        [nxt.notna() & (nxt <= entry), deadline > snapshot, nxt.notna() & (nxt <= deadline)],
        [Outcome.ALREADY_PAST.value, Outcome.PENDING.value, Outcome.ADVANCED.value],
        default=Outcome.NOT_ADVANCED.value,
    )
    return pd.Series(out, index=entry.index)


def apply_window(transitions: pd.DataFrame, cfg: LabelConfig) -> pd.DataFrame:
    df = transitions.copy()
    df["window_years"] = df.transition.map(cfg.years_for)
    df["outcome"] = classify(df.entry_date, df.next_date, df.window_years, cfg.snapshot)
    df["label"] = df.outcome.map({Outcome.ADVANCED.value: 1.0, Outcome.NOT_ADVANCED.value: 0.0})
    df["cohort"] = pd.cut(df.entry_date.dt.year, [cfg.first_entry_year - 1, 2010, 2015, 2100],
                          labels=[f"{cfg.first_entry_year}-2010", "2011-2015", "2016+"])
    return df


def _trial_terms(aact: AactTables) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    listed = aact.conditions.groupby("nct_id").mesh_term.agg(set).to_dict()
    anc = aact.ancestors.groupby("nct_id").mesh_term.agg(set).to_dict()
    # A term's ancestors are only certain when the trial has that one term, so build the lookup from those trials.
    ancestors_of: dict[str, set[str]] = defaultdict(set)
    for nct, terms in listed.items():
        if len(terms) == 1:
            ancestors_of[next(iter(terms))] |= anc.get(nct, set())
    return listed, anc, ancestors_of


def _entry_trial(trials: pd.DataFrame, keys: list[str]) -> pd.Series:
    """NCT ID of the trial that sets each program's entry date; same-day ties go to the lowest NCT ID."""
    return trials.sort_values(["start", "nct_id"]).groupby(keys).nct_id.first()


def _at_entry(trials: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Trials registered on each program's entry date. Program attributes (US site, disease area) come only from
    these, so later trials can't decide which programs make the label set or what they look like at entry."""
    return trials[trials.start == trials.groupby(keys).start.transform("min")]


def _mode_area(areas: pd.Series) -> str:
    known = areas[areas != "Unknown"]
    return known.mode().iat[0] if len(known) else "Unknown"


def _purpose_ok(purpose: object) -> bool:
    """Missing purpose (older CT.gov rows) is kept; explicit non-therapeutic purposes are not."""
    if purpose is None or (isinstance(purpose, float) and np.isnan(purpose)):
        return True
    if not isinstance(purpose, str) or not purpose.strip():
        return True
    return purpose in V1_PURPOSES


def _collapse_phase2_to_area(mesh_rows: pd.DataFrame) -> pd.DataFrame:
    """One Phase 2 -> 3 program per drug x BIO disease area.

    MeSH-level rows over-count oncology basket trials (one trial -> dozens of tumor terms). Keep the
    earliest entry trial in the area and the earliest Phase 3 match across that area's MeSH rows.
    """
    if mesh_rows.empty:
        return mesh_rows.iloc[0:0].copy()
    rows = mesh_rows.sort_values(["drug_id", "area", "entry_date", "entry_nct_id", "mesh_term"])
    entry = rows.groupby(["drug_id", "area"], as_index=False).first()
    nxt = rows.groupby(["drug_id", "area"], as_index=False).next_mesh.min()
    return entry.drop(columns=["next_mesh"], errors="ignore").merge(nxt, on=["drug_id", "area"])


def build_transitions(trial_drugs: pd.DataFrame, aact: AactTables, approvals: pd.Series,
                      first_entry_year: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """All programs entering Phase 1, 2 or 3, with entry_date and next_date (earliest qualifying next step)."""
    t = trial_drugs[["nct_id", "drug_id", "drug_name"]].merge(aact.trials, on="nct_id")
    listed, anc, ancestors_of = _trial_terms(aact)
    t["area"] = [area_for(listed.get(n, set()) | anc.get(n, set())) for n in t.nct_id]

    first_trial = t.groupby("drug_id").start.min()
    approval = approvals.reindex(first_trial.index)
    lifecycle = set(first_trial.index[approval.notna() & (approval <= first_trial)])
    t = t[~t.drug_id.isin(lifecycle)]
    names = t.groupby("drug_id").drug_name.first()
    approval = approvals.reindex(names.index)
    frames = []

    # Phase 1 -> 2, per drug: Phase 1 is often in healthy volunteers, so it has no reliable disease.
    # Only pure PHASE1 starts here; PHASE1/PHASE2 is a Phase 2 entry (BIO's convention).
    p1 = t[t.phase == "PHASE1"]
    e1 = _at_entry(p1, ["drug_id"]).groupby("drug_id").agg(
        entry_date=("start", "min"), area=("area", _mode_area),
        entry_has_us_site=("has_us_site", "any"), entry_primary_purpose=("primary_purpose", "first"),
    )
    e1["entry_nct_id"] = _entry_trial(p1, ["drug_id"])
    reach2 = t[t.level >= 2].groupby("drug_id").start.min()
    frames.append(e1.assign(transition=TRANSITIONS[0], mesh_term="", next_date=reach2.reindex(e1.index)))

    # Phase 2 -> 3: collapse MeSH rows to drug x BIO disease area. Phase 3 counts if (a) MeSH/ancestor
    # match on any term in the area, or (b) the Phase 3 trial itself maps to the same BIO area.
    # (b) recovers matches that (a) misses because ancestors_of only covers ~57% of listed terms.
    # FDA approval is NOT used: Drugs@FDA has no indication field.
    p2 = t[t.phase.isin(PHASE2_ENTRY)]
    p2_terms = pd.DataFrame(
        [(d, m, n, s, us, pur) for d, n, s, us, pur in
         zip(p2.drug_id, p2.nct_id, p2.start, p2.has_us_site, p2.primary_purpose)
         for m in listed.get(n, ())],
        columns=["drug_id", "mesh_term", "nct_id", "start", "has_us_site", "primary_purpose"])
    e2 = (_at_entry(p2_terms, ["drug_id", "mesh_term"]).groupby(["drug_id", "mesh_term"])
          .agg(entry_date=("start", "min"), entry_has_us_site=("has_us_site", "any"),
               entry_primary_purpose=("primary_purpose", "first")).reset_index())
    e2["entry_nct_id"] = (_entry_trial(p2_terms, ["drug_id", "mesh_term"])
                          .reindex(pd.MultiIndex.from_frame(e2[["drug_id", "mesh_term"]])).to_numpy())
    p3_by_drug: dict[str, list[tuple[pd.Timestamp, set[str], set[str]]]] = defaultdict(list)
    p3 = t[t.level == 3]
    for d, n, s in zip(p3.drug_id, p3.nct_id, p3.start):
        p3_by_drug[d].append((s, listed.get(n, set()), anc.get(n, set())))

    def next_p3_mesh(drug: str, term: str) -> pd.Timestamp:
        broader = ancestors_of.get(term, set())
        starts = [s for s, lst, a in p3_by_drug.get(drug, ()) if term in lst or term in a or lst & broader]
        return min(starts) if starts else pd.NaT

    e2["next_mesh"] = [next_p3_mesh(d, m) for d, m in zip(e2.drug_id, e2.mesh_term)]
    e2["area"] = [area_for({m} | ancestors_of.get(m, set())) for m in e2.mesh_term]
    n_mesh_programs = len(e2)
    e2 = _collapse_phase2_to_area(e2)
    p3_by_area = p3.groupby(["drug_id", "area"]).start.min()
    next_area = pd.Series([p3_by_area.get((d, a), pd.NaT) for d, a in zip(e2.drug_id, e2.area)], index=e2.index)
    e2["next_date"] = pd.concat([e2["next_mesh"], next_area], axis=1).min(axis=1)
    frames.append(e2.drop(columns="next_mesh").set_index("drug_id").assign(transition=TRANSITIONS[1]))

    # Phase 3 -> FDA approval, per drug: Drugs@FDA has no indication field.
    p3e = _at_entry(p3, ["drug_id"]).groupby("drug_id").agg(
        entry_date=("start", "min"), area=("area", _mode_area),
        entry_has_us_site=("has_us_site", "any"), entry_primary_purpose=("primary_purpose", "first"),
    )
    p3e["entry_nct_id"] = _entry_trial(p3, ["drug_id"])
    frames.append(p3e.assign(transition=TRANSITIONS[2], mesh_term="", next_date=approval.reindex(p3e.index)))

    frames = [f for f in frames if len(f)]
    out = pd.concat(frames).rename_axis("drug_id").reset_index()
    out["drug_name"] = out.drug_id.map(names)
    # Drugs known only by a code or bare name can't be followed across trials or found in Drugs@FDA,
    # so their "failures" are mostly linking failures.
    out["identified"] = out.drug_id.str.startswith(("CHEMBL", "FDA:"))
    purpose_ok = out.entry_primary_purpose.map(_purpose_ok)
    out["in_v1_label_set"] = out.identified & out.entry_has_us_site & purpose_ok
    out = out[out.entry_date.dt.year >= first_entry_year]
    # AACT free text occasionally carries NULs / control chars that break the CSV round-trip.
    for col in ("drug_name", "mesh_term", "area", "entry_primary_purpose", "entry_nct_id"):
        if col in out.columns:
            out[col] = out[col].map(lambda x: x.replace("\x00", "").strip() if isinstance(x, str) else x)
    stats = {
        "drugs_in_scope": int(names.size),
        "drugs_excluded_already_approved": len(lifecycle),
        "phase2_mesh_programs_before_collapse": int(n_mesh_programs),
        "phase2_area_programs_after_collapse": int((out.transition == TRANSITIONS[1]).sum()),
    }
    cols = ["transition", "drug_id", "drug_name", "mesh_term", "disease_area", "entry_date", "entry_nct_id",
            "next_date", "entry_has_us_site", "entry_primary_purpose", "identified", "in_v1_label_set"]
    return out.rename(columns={"area": "disease_area"})[cols].reset_index(drop=True), stats


def success_rates(labeled: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    keys = ["transition", *(by or [])]
    df = labeled.dropna(subset=["label"])
    g = df.groupby(keys, observed=True).label.agg(n="size", our_rate="mean").reset_index()
    return g
