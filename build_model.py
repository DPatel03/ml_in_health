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
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
TRANSITION = "phase2_to_phase3"
BIO_RATE = 0.289  # BIO 2021, phase2_to_phase3, all indications -- LABEL_RULES.md

# num_facilities / num_countries dropped: AACT stores the final site/country list, not what was
# registered when Phase 2 started, so they'd leak how the trial actually played out (Deep's note).
# log_enrollment dropped too: it's only kept where enrollment_type is ESTIMATED, which is ~5% of
# training rows but ~16% of test rows, and trials still carrying an estimate advance less often
# (11-14% vs 26-29%). Median-imputing the rest made the few real values extreme outliers, which
# drove the overconfident top fifth (predicted ~50%, actual ~23%).
NUMERIC = ["number_of_arms", "sponsor_track_record", "has_us_site"]
# phase = entry trial's phase. Phase 1/2 trials count as reaching Phase 2 but advance less often
# (21-24% vs 27-29%), and their share doubles from train (24%) to test (47%).
CATEGORICAL = ["phase", "allocation", "intervention_model", "masking", "disease_area", "primary_purpose"]
N_BOOT = 2000


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


def build_sponsor_pool(features: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    """Every labeled program across all three transitions, with sponsor_track_record computed over
    that whole pool -- a sponsor's Phase 1->2 and Phase 3->approval record counts toward its track
    record here too, not just its record on whichever transition is being modeled."""
    pool = transitions[transitions.in_v1_label_set & transitions.label.notna()].copy()
    pool = pool.merge(features, left_on="entry_nct_id", right_on="nct_id", how="inner")
    pool["has_us_site"] = pool.has_us_site.astype(float)
    pool = pool.sort_values("entry_date").reset_index(drop=True)
    pool["sponsor_track_record"] = sponsor_track_record(pool)
    return pool


def build_program_features(features: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    pool = build_sponsor_pool(features, transitions)
    df = pool[pool.transition == TRANSITION].reset_index(drop=True)

    # strict time split: train on entries up to 2015, test on 2016+, and keep every program of the
    # same drug on one side (LABEL_RULES.md #8). An older drug's 2016+ programs fit neither side,
    # so they're dropped rather than letting train see test-period outcomes.
    year = df.entry_date.dt.year
    drug_min_year = df.drug_id.map(df.groupby("drug_id").entry_date.min().dt.year)
    df["split"] = np.select([year <= 2015, drug_min_year >= 2016], ["train", "test"], "dropped")
    return df[df.split != "dropped"].reset_index(drop=True)


def sponsor_track_record(df: pd.DataFrame) -> np.ndarray:
    """That sponsor's label rate on programs whose outcome was already known before this one entered,
    across all three transitions. A success is known on next_date, a failure only once its window
    ran out (entry + window_years), so a program entered in 2014 can't inform a 2016 prediction if
    it failed."""
    known_date = df.next_date.where(df.label == 1, df.entry_date + pd.to_timedelta(df.window_years * 365.25, unit="D"))
    track_record = np.full(len(df), np.nan)
    for _, idx in df.groupby("lead_sponsor_name").indices.items():
        order = np.argsort(known_date.iloc[idx].to_numpy())
        known_sorted = known_date.iloc[idx].to_numpy()[order]
        cum_labels = np.cumsum(df.label.iloc[idx].to_numpy()[order])
        # count of the sponsor's outcomes known strictly before each program's entry date
        n_known = np.searchsorted(known_sorted, df.entry_date.iloc[idx].to_numpy(), side="left")
        has_hist = n_known > 0
        track_record[idx[has_hist]] = cum_labels[n_known[has_hist] - 1] / n_known[has_hist]
    return track_record


def load_bio_area_rates(path: Path) -> pd.Series:
    """BIO 2021 Phase 2 -> 3 rate per disease area: the phase x area table analysts actually use."""
    bio = pd.read_csv(path)
    return bio[bio.transition == TRANSITION].set_index("disease_area").success_rate


def clustered_bootstrap_ci(diff: np.ndarray, groups: np.ndarray, n_boot: int = N_BOOT,
                           seed: int = 0) -> tuple[float, float]:
    """95% CI for the mean of a per-row score difference, resampling whole drugs: one drug's
    programs share most of their information, so resampling rows would make the CI too narrow."""
    rng = np.random.default_rng(seed)
    uniq, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=diff)
    counts = np.bincount(inverse)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        means[b] = (pick * sums).sum() / (pick * counts).sum()
    return tuple(np.percentile(means, [2.5, 97.5]))


RECENT_TRAIN_YEAR = 2011  # recalibrate on the 2011-2015 cohort: the training years closest to test


def recalibrate(model: Pipeline, train: pd.DataFrame, raw_test_proba: np.ndarray) -> np.ndarray:
    """Platt-scale the model's output, fit only on its most recent training years. The audit found
    the model overpredicts because success rates fell after 2015 and training skews older; a
    calibrator fit on train's own tail should track that shift better than one fit on all of train.
    Uses out-of-fold predictions on train, not in-sample ones, so the calibrator isn't just undoing
    overfitting it never saw."""
    oof = cross_val_predict(clone(model), train[NUMERIC + CATEGORICAL], train.label, cv=5, method="predict_proba")[:, 1]
    recent = (train.entry_date.dt.year >= RECENT_TRAIN_YEAR).to_numpy()
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    platt = LogisticRegression().fit(logit(oof[recent]).reshape(-1, 1), train.label.to_numpy()[recent])
    return platt.predict_proba(logit(raw_test_proba).reshape(-1, 1))[:, 1]


def make_model(kind: str) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    # ~4,700 training rows is little data for gradient boosting; num_leaves=3, min_child_samples=40,
    # n_estimators=100 chosen by 5-fold CV grid search on train only (CV Brier 0.1953 vs 0.1996 for
    # an earlier hand-picked num_leaves=7) -- this small a dataset wants very shallow trees
    clf = (LogisticRegression(max_iter=1000) if kind == "logistic" else
           LGBMClassifier(n_estimators=100, num_leaves=3, min_child_samples=40, learning_rate=0.05,
                           reg_lambda=1.0, verbosity=-1))
    return Pipeline([("pre", pre), ("clf", clf)])


def fit_and_evaluate(df: pd.DataFrame, bio_area_rates: pd.Series | None = None, model_kind: str = "logistic") -> dict:
    model = make_model(model_kind)

    train, test = df[df.split == "train"], df[df.split == "test"]
    model.fit(train[NUMERIC + CATEGORICAL], train.label)
    raw_proba = model.predict_proba(test[NUMERIC + CATEGORICAL])[:, 1]
    proba = recalibrate(model, train, raw_proba)

    raw_frac_pos, raw_mean_pred = calibration_curve(test.label, raw_proba, n_bins=5, strategy="quantile")
    frac_pos, mean_pred = calibration_curve(test.label, proba, n_bins=5, strategy="quantile")
    bio_proba = np.full(len(test), BIO_RATE)

    # second baseline: our own training-years rate by disease area, not just BIO's one flat number
    area_rate = train.groupby("disease_area").label.mean()
    overall_rate = train.label.mean()
    area_proba = test.disease_area.map(area_rate).fillna(overall_rate).to_numpy()

    benchmarks = {"BIO flat rate": bio_proba, "train-years area rate": area_proba}
    if bio_area_rates is not None:
        # areas missing from BIO's table (e.g. Unknown) fall back to BIO's all-indications rate
        benchmarks["BIO by area"] = test.disease_area.map(bio_area_rates).fillna(BIO_RATE).to_numpy()
    y = test.label.to_numpy()
    model_sq_err = (proba - y) ** 2
    brier_model = model_sq_err.mean()
    vs = {}
    for name, bench in benchmarks.items():
        brier_bench = brier_score_loss(y, bench)
        lo, hi = clustered_bootstrap_ci(model_sq_err - (bench - y) ** 2, test.drug_id.to_numpy())
        vs[name] = {"brier": brier_bench, "skill": 1 - brier_model / brier_bench, "diff": brier_model - brier_bench,
                    "diff_ci": (lo, hi)}

    return {
        "n_train": len(train), "n_test": len(test),
        "brier_model": brier_model,
        "brier_model_raw": brier_score_loss(y, raw_proba),
        "brier_bio": brier_score_loss(test.label, bio_proba),
        "brier_area": brier_score_loss(test.label, area_proba),
        "vs_benchmarks": vs,
        "log_loss_model": log_loss(test.label, proba, labels=[0, 1]),
        "log_loss_area": log_loss(test.label, area_proba, labels=[0, 1]),
        "calibration_curve": list(zip(mean_pred.round(3), frac_pos.round(3))),
        "calibration_curve_raw": list(zip(raw_mean_pred.round(3), raw_frac_pos.round(3))),
        "test_positive_rate": test.label.mean(),
        "train_area_rates": area_rate.round(3).to_dict(),
    }


def plot_calibration(results: dict, out: Path, model_kind: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfectly calibrated")
    raw_xs, raw_ys = zip(*results["calibration_curve_raw"])
    ax.plot(raw_xs, raw_ys, "o--", color="tab:gray", alpha=0.7, label="before recalibration")
    xs, ys = zip(*results["calibration_curve"])
    ax.plot(xs, ys, "o-", label=f"recalibrated ({model_kind})")
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
    p.add_argument("--bio", type=Path, default=HERE / "external" / "bio" / "bio2021_phase_transitions.csv")
    p.add_argument("--model", choices=["logistic", "lightgbm"], default="lightgbm")
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

    results = fit_and_evaluate(df, load_bio_area_rates(a.bio), model_kind=a.model)
    print(f"\nmodel={a.model} n_train={results['n_train']} n_test={results['n_test']} "
          f"test positive rate={results['test_positive_rate']:.3f}")
    print(f"Brier -- model: {results['brier_model']:.4f} (before recalibration: {results['brier_model_raw']:.4f})")
    for name, v in results["vs_benchmarks"].items():
        lo, hi = v["diff_ci"]
        print(f"  vs {name:22s} {v['brier']:.4f}  skill {v['skill']:+.1%}  "
              f"model minus benchmark {v['diff']:+.4f} (95% CI {lo:+.4f} to {hi:+.4f}, drug bootstrap)")
    print(f"Log loss -- model: {results['log_loss_model']:.4f}  area baseline: {results['log_loss_area']:.4f}")
    print("Train-years rate by disease area:", results["train_area_rates"])
    print("Calibration curve (mean predicted, observed fraction):", results["calibration_curve"])
    plot_calibration(results, HERE / "output" / "model" / "calibration_phase2_to_phase3.png", a.model)


if __name__ == "__main__":
    main()
