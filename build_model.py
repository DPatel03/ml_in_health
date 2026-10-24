"""Phase 2 -> 3 baseline: join trial design features onto the label pipeline's output, train
logistic regression, and plot its calibration against the flat BIO 2021 rate.

Needs output/model/trial_design_features.csv (from build_features.py) and
output/labels/transitions.csv (from build_labels.py). Falls back to a synthetic sample if
transitions.csv is missing, just to check the pipeline runs -- don't report those numbers.

Usage:
    python3 build_model.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
TRANSITION = "phase2_to_phase3"
BIO_RATE = 0.289  # BIO 2021, phase2_to_phase3, all indications -- LABEL_RULES.md

# num_facilities / num_countries dropped: AACT stores the final site/country list, not what was
# registered when Phase 2 started, so they'd leak how the trial actually played out (Deep's note).
NUMERIC = ["log_enrollment", "number_of_arms", "sponsor_track_record", "has_us_site"]
CATEGORICAL = ["allocation", "intervention_model", "masking", "disease_area"]


def make_placeholder_labels(features: pd.DataFrame, n: int = 1200, seed: int = 0) -> pd.DataFrame:
    """Synthetic stand-in for Deep's transitions.csv, same columns, fake values.
    Only for exercising the pipeline before his real file lands -- never report these numbers."""
    rng = np.random.default_rng(seed)
    sample = features.sample(n=min(n, len(features)), random_state=seed).reset_index(drop=True)
    drug_id = [f"CHEMBL{i}" for i in range(len(sample))]
    entry_date = sample.start
    years_since = (pd.Timestamp("2024-01-01") - entry_date).dt.days / 365.25
    window_open = years_since >= 5
    label = np.where(window_open, rng.binomial(1, BIO_RATE, len(sample)).astype(float), np.nan)
    cohort = pd.cut(entry_date.dt.year, [2004, 2010, 2015, 2100], labels=["2005-2010", "2011-2015", "2016+"])
    areas = rng.choice(["Oncology", "Cardiovascular", "Neurology", "Unknown"], len(sample), p=[0.3, 0.2, 0.2, 0.3])
    return pd.DataFrame({
        "transition": TRANSITION, "drug_id": drug_id, "drug_name": drug_id, "mesh_term": "",
        "disease_area": areas, "entry_date": entry_date, "entry_nct_id": sample.nct_id, "next_date": pd.NaT,
        "entry_has_us_site": sample.has_us_site, "identified": True, "in_v1_label_set": sample.has_us_site,
        "window_years": 5, "outcome": np.where(pd.isna(label), "pending", np.where(label == 1, "advanced", "not_advanced")),
        "label": label, "cohort": cohort,
    })


def build_program_features(features: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    t = transitions[transitions.transition == TRANSITION].copy()
    t = t[t.in_v1_label_set & t.label.notna()]

    # exact entry trial per row (drug + disease), from Deep's entry_nct_id
    df = t.merge(features, left_on="entry_nct_id", right_on="nct_id", how="inner")
    df["log_enrollment"] = np.log1p(df.enrollment)
    df["has_us_site"] = df.has_us_site.astype(float)

    # sponsor track record: that sponsor's label rate on programs entered strictly before this one
    # (same transition only, for now -- broaden to all three transitions once real data is in)
    df = df.sort_values("entry_date").reset_index(drop=True)
    track_record = np.full(len(df), np.nan)
    seen: dict[str, list[float]] = {}
    for i, (sponsor, lbl) in enumerate(zip(df.lead_sponsor_name, df.label)):
        hist = seen.get(sponsor, [])
        if hist:
            track_record[i] = float(np.mean(hist))
        seen.setdefault(sponsor, []).append(lbl)
    df["sponsor_track_record"] = track_record

    # keep every program of the same drug on the same side of the split (LABEL_RULES.md #8)
    drug_min_year = df.groupby("drug_id").entry_date.min().dt.year
    df["split"] = np.where(df.drug_id.map(drug_min_year) <= 2015, "train", "test")
    return df


def fit_and_evaluate(df: pd.DataFrame) -> dict:
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])

    train, test = df[df.split == "train"], df[df.split == "test"]
    model.fit(train[NUMERIC + CATEGORICAL], train.label)
    proba = model.predict_proba(test[NUMERIC + CATEGORICAL])[:, 1]

    frac_pos, mean_pred = calibration_curve(test.label, proba, n_bins=5, strategy="quantile")
    bio_proba = np.full(len(test), BIO_RATE)

    # second baseline: our own training-years rate by disease area, not just BIO's one flat number
    area_rate = train.groupby("disease_area").label.mean()
    overall_rate = train.label.mean()
    area_proba = test.disease_area.map(area_rate).fillna(overall_rate).to_numpy()

    return {
        "n_train": len(train), "n_test": len(test),
        "brier_model": brier_score_loss(test.label, proba),
        "brier_bio": brier_score_loss(test.label, bio_proba),
        "brier_area": brier_score_loss(test.label, area_proba),
        "log_loss_model": log_loss(test.label, proba, labels=[0, 1]),
        "log_loss_area": log_loss(test.label, area_proba, labels=[0, 1]),
        "calibration_curve": list(zip(mean_pred.round(3), frac_pos.round(3))),
        "test_positive_rate": test.label.mean(),
        "train_area_rates": area_rate.round(3).to_dict(),
    }


def plot_calibration(results: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfectly calibrated")
    xs, ys = zip(*results["calibration_curve"])
    ax.plot(xs, ys, "o-", label="logistic regression (ours)")
    ax.plot(BIO_RATE, results["test_positive_rate"], "s", color="tab:orange", markersize=9,
            label=f"BIO 2021 flat rate ({BIO_RATE})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed advance rate")
    ax.set_title("Phase 2 → 3: calibration, test cohort 2016+")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved calibration plot -> {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, default=HERE / "output" / "model" / "trial_design_features.csv")
    p.add_argument("--transitions", type=Path, default=HERE / "output" / "labels" / "transitions.csv")
    a = p.parse_args()

    features = pd.read_csv(a.features, parse_dates=["start"])

    if a.transitions.is_file():
        transitions = pd.read_csv(a.transitions, parse_dates=["entry_date", "next_date"])
    else:
        print(f"!! {a.transitions.name} not found -- using synthetic placeholder labels to test "
              f"the pipeline. Do not report these numbers.")
        transitions = make_placeholder_labels(features)

    df = build_program_features(features, transitions)
    print(f"programs with features + label: {len(df):,} "
          f"(train {len(df[df.split == 'train']):,}, test {len(df[df.split == 'test']):,})")

    results = fit_and_evaluate(df)
    print(f"\nn_train={results['n_train']} n_test={results['n_test']} "
          f"test positive rate={results['test_positive_rate']:.3f}")
    print(f"Brier -- model: {results['brier_model']:.4f}  "
          f"BIO flat rate ({BIO_RATE}): {results['brier_bio']:.4f}  "
          f"train-years area rate: {results['brier_area']:.4f}")
    print(f"Log loss -- model: {results['log_loss_model']:.4f}  area baseline: {results['log_loss_area']:.4f}")
    print("Train-years rate by disease area:", results["train_area_rates"])
    print("Calibration curve (mean predicted, observed fraction):", results["calibration_curve"])
    plot_calibration(results, HERE / "output" / "model" / "calibration_phase2_to_phase3.png")


if __name__ == "__main__":
    main()
