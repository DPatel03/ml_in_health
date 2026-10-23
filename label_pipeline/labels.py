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


class Outcome(str, Enum):
    ADVANCED = "advanced"
    NOT_ADVANCED = "not_advanced"
    PENDING = "pending"            # window hasn't closed yet: no label, these feed the watchlist
    ALREADY_PAST = "already_past"  # next step started before the first registered trial of this phase


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
    # early successes would bias recent cohorts upward.
    out = np.select(
        [nxt.notna() & (nxt < entry), deadline > snapshot, nxt.notna() & (nxt <= deadline)],
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


def _mode_area(areas: pd.Series) -> str:
    known = areas[areas != "Unknown"]
    return known.mode().iat[0] if len(known) else "Unknown"


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
    p1 = t[t.phase == "PHASE1"]
    e1 = p1.groupby("drug_id").agg(entry_date=("start", "min"), area=("area", _mode_area),
                                   entry_has_us_site=("has_us_site", "any"))
    e1["entry_nct_id"] = _entry_trial(p1, ["drug_id"])
    reach2 = t[t.level >= 2].groupby("drug_id").start.min()
    frames.append(e1.assign(transition=TRANSITIONS[0], mesh_term="", next_date=reach2.reindex(e1.index)))

    # Phase 2 -> 3, per drug + disease: Phase 3 counts if it shares the MeSH term, or one is the other's ancestor.
    p2 = t[t.phase.isin(PHASE2_ENTRY)]
    p2_terms = pd.DataFrame(
        [(d, m, n, s, us) for d, n, s, us in zip(p2.drug_id, p2.nct_id, p2.start, p2.has_us_site)
         for m in listed.get(n, ())],
        columns=["drug_id", "mesh_term", "nct_id", "start", "has_us_site"])
    e2 = (p2_terms.groupby(["drug_id", "mesh_term"])
          .agg(entry_date=("start", "min"), entry_has_us_site=("has_us_site", "any")).reset_index())
    # per disease, so a drug tested in several diseases gets the right Phase 2 trial on each row
    e2["entry_nct_id"] = (_entry_trial(p2_terms, ["drug_id", "mesh_term"])
                          .reindex(pd.MultiIndex.from_frame(e2[["drug_id", "mesh_term"]])).to_numpy())
    p3_by_drug: dict[str, list[tuple[pd.Timestamp, set[str], set[str]]]] = defaultdict(list)
    p3 = t[t.level == 3]
    for d, n, s in zip(p3.drug_id, p3.nct_id, p3.start):
        p3_by_drug[d].append((s, listed.get(n, set()), anc.get(n, set())))

    def next_p3(drug: str, term: str) -> pd.Timestamp:
        broader = ancestors_of.get(term, set())
        starts = [s for s, lst, a in p3_by_drug.get(drug, ()) if term in lst or term in a or lst & broader]
        return min(starts) if starts else pd.NaT

    e2["next_p3"] = [next_p3(d, m) for d, m in zip(e2.drug_id, e2.mesh_term)]
    # an approval inside the window also counts as getting past Phase 2 (e.g. accelerated approval on Phase 2 data)
    appr = e2.drug_id.map(approval)
    appr = appr.where(appr >= e2.entry_date)
    e2["next_date"] = e2[["next_p3"]].assign(appr=appr).min(axis=1)
    e2["area"] = [area_for({m} | ancestors_of.get(m, set())) for m in e2.mesh_term]
    frames.append(e2.drop(columns="next_p3").set_index("drug_id").assign(transition=TRANSITIONS[1]))

    # Phase 3 -> FDA approval, per drug: Drugs@FDA has no indication field.
    p3e = p3.groupby("drug_id").agg(entry_date=("start", "min"), area=("area", _mode_area),
                                    entry_has_us_site=("has_us_site", "any"))
    p3e["entry_nct_id"] = _entry_trial(p3, ["drug_id"])
    frames.append(p3e.assign(transition=TRANSITIONS[2], mesh_term="", next_date=approval.reindex(p3e.index)))

    out = pd.concat(frames).rename_axis("drug_id").reset_index()
    out["drug_name"] = out.drug_id.map(names)
    # Drugs known only by a code or bare name can't be followed across trials or found in Drugs@FDA,
    # so their "failures" are mostly linking failures.
    out["identified"] = out.drug_id.str.startswith(("CHEMBL", "FDA:"))
    out["in_v1_label_set"] = out.identified & out.entry_has_us_site
    out = out[out.entry_date.dt.year >= first_entry_year]
    stats = {"drugs_in_scope": int(names.size), "drugs_excluded_already_approved": len(lifecycle)}
    cols = ["transition", "drug_id", "drug_name", "mesh_term", "disease_area", "entry_date", "entry_nct_id",
            "next_date", "entry_has_us_site", "identified", "in_v1_label_set"]
    return out.rename(columns={"area": "disease_area"})[cols].reset_index(drop=True), stats


def success_rates(labeled: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    keys = ["transition", *(by or [])]
    df = labeled.dropna(subset=["label"])
    g = df.groupby(keys, observed=True).label.agg(n="size", our_rate="mean").reset_index()
    return g
