"""Score pending Phase 2 -> 3 programs against BIO's benchmark: the proposal's mispricing watchlist.

Refits the model on every labeled program (train + test combined, since there's no more held-out
evaluation to protect), recalibrates on the most recent labeled years, then scores pending programs
and ranks them by how far our estimate diverges from BIO's disease-area rate.

Only scores pending programs with in_v1_label_set: a resolved drug identity, a US site, and a
therapeutic primary purpose -- the same bar used everywhere else (LABEL_RULES.md #7). Pending rows
that fail that bar are mostly unreliable drug IDs and would look like failures for linking reasons.

Usage:
    python3 build_watchlist.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from build_model import (HERE, NUMERIC, CATEGORICAL, build_sponsor_pool, fit_calibrator, join_trial_features,
                         load_bio_rates, log_odds, make_model, sponsor_track_record)
from label_pipeline.labels import TRANSITIONS

TRANSITION = TRANSITIONS[1]


def build_pending_features(features: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    eligible = ((transitions.transition == TRANSITION) & (transitions.outcome == "pending")
                & transitions.in_v1_label_set & transitions.next_date.isna())
    return join_trial_features(features, transitions[eligible])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, default=HERE / "output" / "model" / "trial_design_features.csv")
    p.add_argument("--transitions", type=Path, default=HERE / "output" / "labels" / "transitions.csv")
    p.add_argument("--bio", type=Path, default=HERE / "external" / "bio" / "bio2021_phase_transitions.csv")
    p.add_argument("--genetic-evidence", type=Path,
                    help="optional retrospective genetic scores")
    p.add_argument("--out", type=Path, default=HERE / "output" / "model" / "watchlist_phase2_to_phase3.csv")
    p.add_argument("--top", type=int, default=20)
    a = p.parse_args()

    features = pd.read_csv(a.features, parse_dates=["start"])
    transitions = pd.read_csv(a.transitions, parse_dates=["entry_date", "next_date"], low_memory=False)
    bio_flat, bio_area_rates = load_bio_rates(a.bio, TRANSITION)
    sponsor_pool = build_sponsor_pool(features, transitions)
    labeled = sponsor_pool[sponsor_pool.transition == TRANSITION].copy()
    pending = build_pending_features(features, transitions)
    pending["sponsor_track_record"] = sponsor_track_record(sponsor_pool, targets=pending)
    pending["has_sponsor_history"] = pending.sponsor_track_record.notna().astype(float)
    numeric = list(NUMERIC)
    if a.genetic_evidence:
        scores = pd.read_csv(a.genetic_evidence).set_index("drug_id").genetic_evidence_score
        numeric.append("genetic_evidence_score")
        for frame in (labeled, pending):
            frame["genetic_evidence_score"] = frame.drug_id.map(scores)
    columns = numeric + CATEGORICAL
    model = make_model("lightgbm", numeric).fit(labeled[columns], labeled.label)
    calibrator = fit_calibrator(model, labeled, columns)
    proba = np.array([])
    if len(pending):
        raw = model.predict_proba(pending[columns])[:, 1]
        proba = calibrator.predict_proba(log_odds(raw))[:, 1]
    print(f"Scored {len(pending)} unresolved programs using {len(labeled)} labeled programs.")
    bio_area = pending.disease_area.map(bio_area_rates).fillna(bio_flat)
    watchlist = pd.DataFrame({
        "drug_id": pending.drug_id, "drug_name": pending.drug_name, "mesh_term": pending.mesh_term, "disease_area": pending.disease_area,
        "entry_date": pending.entry_date.dt.date, "entry_nct_id": pending.entry_nct_id,
        "our_probability": proba.round(3), "bio_area_rate": bio_area.round(3).to_numpy(),
        "bio_flat_rate": bio_flat,
    })
    watchlist["divergence_vs_area"] = (watchlist.our_probability - watchlist.bio_area_rate).round(3)
    watchlist["divergence_vs_flat"] = (watchlist.our_probability - bio_flat).round(3)

    our_area_rate = labeled.groupby("disease_area").label.mean()
    label_gap = (our_area_rate - bio_area_rates).reindex(watchlist.disease_area).to_numpy()
    watchlist["known_label_gap"] = np.round(label_gap, 3)
    watchlist["reliable"] = np.abs(label_gap) <= 0.05

    watchlist = watchlist.reindex(watchlist.divergence_vs_area.abs().sort_values(ascending=False).index)
    watchlist.to_csv(a.out, index=False)
    print(f"wrote {len(watchlist):,} rows -> {a.out}")
    n_unreliable = (~watchlist.reliable).sum()
    print(f"{n_unreliable} rows lack an area benchmark or exceed the 5-point label-gap threshold.")
    print(f"Top {a.top} by divergence, RELIABLE disease areas only:")
    reliable = watchlist[watchlist.reliable]
    print(reliable.head(a.top)[["drug_name", "disease_area", "our_probability", "bio_area_rate",
                                 "divergence_vs_area"]].to_string(index=False))
    print(f"\n(excluded areas with a known label gap vs BIO: "
          f"{sorted(watchlist[~watchlist.reliable].disease_area.unique())})")


if __name__ == "__main__":
    main()
