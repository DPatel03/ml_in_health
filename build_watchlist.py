"""Score pending Phase 2 -> 3 programs against BIO's benchmark: the proposal's mispricing watchlist.

Refits the model on every labeled program (train + test combined, since there's no more held-out
evaluation to protect), recalibrates on the most recent labeled years, then scores pending programs
and ranks them by how far our estimate diverges from BIO's disease-area rate.

Only scores pending programs with in_v1_label_set: a resolved drug identity and a US site, the same
bar used everywhere else in this project. Of the 9,537 pending Phase 2 -> 3 programs, only 1,901
clear that bar -- the rest have an unreliable drug identity and would mostly look like failures for
linking reasons, not real ones (LABEL_RULES.md #7).

Usage:
    python3 build_watchlist.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

from build_model import (HERE, TRANSITION, BIO_RATE, NUMERIC, CATEGORICAL, RECENT_TRAIN_YEAR,
                          make_model, load_bio_area_rates)

LOGIT = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def sponsor_rate_asof(history: pd.DataFrame, targets: pd.DataFrame) -> np.ndarray:
    """Same rule as build_model.sponsor_track_record, but scoped to score a separate target set:
    for each target row, that sponsor's label rate among history rows known before its entry_date."""
    known_date = history.next_date.where(
        history.label == 1, history.entry_date + pd.to_timedelta(history.window_years * 365.25, unit="D"))
    by_sponsor: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sponsor, idx in history.groupby("lead_sponsor_name").indices.items():
        order = np.argsort(known_date.iloc[idx].to_numpy())
        by_sponsor[sponsor] = (known_date.iloc[idx].to_numpy()[order],
                                np.cumsum(history.label.iloc[idx].to_numpy()[order]))

    out = np.full(len(targets), np.nan)
    for i, (sponsor, entry) in enumerate(zip(targets.lead_sponsor_name, targets.entry_date)):
        if sponsor not in by_sponsor:
            continue
        known_sorted, cum_labels = by_sponsor[sponsor]
        n_known = np.searchsorted(known_sorted, np.datetime64(entry), side="left")
        if n_known > 0:
            out[i] = cum_labels[n_known - 1] / n_known
    return out


def build_pending_features(features: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    t = transitions[(transitions.transition == TRANSITION) & (transitions.outcome == "pending")
                     & transitions.in_v1_label_set].copy()
    df = t.merge(features, left_on="entry_nct_id", right_on="nct_id", how="inner")
    df["has_us_site"] = df.has_us_site.astype(float)
    return df


def fit_final_model(labeled: pd.DataFrame, model_kind: str = "lightgbm") -> Pipeline:
    model = make_model(model_kind)
    model.fit(labeled[NUMERIC + CATEGORICAL], labeled.label)
    return model


def fit_calibrator(model: Pipeline, labeled: pd.DataFrame) -> LogisticRegression:
    oof = cross_val_predict(clone(model), labeled[NUMERIC + CATEGORICAL], labeled.label,
                             cv=5, method="predict_proba")[:, 1]
    recent = (labeled.entry_date.dt.year >= RECENT_TRAIN_YEAR).to_numpy()
    return LogisticRegression().fit(LOGIT(oof[recent]).reshape(-1, 1), labeled.label.to_numpy()[recent])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, default=HERE / "output" / "model" / "trial_design_features.csv")
    p.add_argument("--transitions", type=Path, default=HERE / "output" / "labels" / "transitions.csv")
    p.add_argument("--bio", type=Path, default=HERE / "external" / "bio" / "bio2021_phase_transitions.csv")
    p.add_argument("--out", type=Path, default=HERE / "output" / "model" / "watchlist_phase2_to_phase3.csv")
    p.add_argument("--top", type=int, default=20)
    a = p.parse_args()

    features = pd.read_csv(a.features, parse_dates=["start"])
    transitions = pd.read_csv(a.transitions, parse_dates=["entry_date", "next_date"])
    bio_area_rates = load_bio_area_rates(a.bio)

    # the same labeled population build_model.py trains and tests on, here combined: no more
    # held-out evaluation to protect once this is the production model for scoring pending programs
    from build_model import build_sponsor_pool
    sponsor_pool = build_sponsor_pool(features, transitions)
    labeled = sponsor_pool[(sponsor_pool.transition == TRANSITION)].reset_index(drop=True)
    print(f"labeled programs (train+test combined): {len(labeled):,}")

    model = fit_final_model(labeled)
    calibrator = fit_calibrator(model, labeled)

    pending = build_pending_features(features, transitions)
    # same broadened, all-transitions history used for the labeled model, not just this transition's
    pending["sponsor_track_record"] = sponsor_rate_asof(sponsor_pool, pending)
    print(f"pending programs scored: {len(pending):,} (of 9,537 total; rest lack a reliable drug ID)")

    raw_proba = model.predict_proba(pending[NUMERIC + CATEGORICAL])[:, 1]
    proba = calibrator.predict_proba(LOGIT(raw_proba).reshape(-1, 1))[:, 1]

    bio_area = pending.disease_area.map(bio_area_rates).fillna(BIO_RATE)
    watchlist = pd.DataFrame({
        "drug_name": pending.drug_name, "mesh_term": pending.mesh_term, "disease_area": pending.disease_area,
        "entry_date": pending.entry_date.dt.date, "entry_nct_id": pending.entry_nct_id,
        "our_probability": proba.round(3), "bio_area_rate": bio_area.round(3).to_numpy(),
        "bio_flat_rate": BIO_RATE,
    })
    watchlist["divergence_vs_area"] = (watchlist.our_probability - watchlist.bio_area_rate).round(3)
    watchlist["divergence_vs_flat"] = (watchlist.our_probability - BIO_RATE).round(3)

    # our own training-years rate runs well below BIO's in exactly the areas Drugs@FDA misses
    # biologics approvals for (README: Phase 3 -> approval gap). Flag those so a "mispricing" in
    # this area isn't read as a market signal when it's really our own known label gap.
    our_area_rate = labeled.groupby("disease_area").label.mean()
    label_gap = (our_area_rate - bio_area_rates).reindex(watchlist.disease_area).to_numpy()
    watchlist["known_label_gap"] = np.round(label_gap, 3)
    watchlist["reliable"] = np.abs(label_gap) <= 0.05

    watchlist = watchlist.reindex(watchlist.divergence_vs_area.abs().sort_values(ascending=False).index)
    watchlist.to_csv(a.out, index=False)
    print(f"wrote {len(watchlist):,} rows -> {a.out}")
    n_unreliable = (~watchlist.reliable).sum()
    print(f"{n_unreliable:,} of those sit in disease areas where our own labels run >5pp off BIO's "
          f"rate (the Drugs@FDA biologics gap) -- their divergence is likely that gap, not mispricing.\n")

    print(f"Top {a.top} by divergence, RELIABLE disease areas only:")
    reliable = watchlist[watchlist.reliable]
    print(reliable.head(a.top)[["drug_name", "disease_area", "our_probability", "bio_area_rate",
                                 "divergence_vs_area"]].to_string(index=False))
    print(f"\n(excluded areas with a known label gap vs BIO: "
          f"{sorted(watchlist[~watchlist.reliable].disease_area.unique())})")


if __name__ == "__main__":
    main()
