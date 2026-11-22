"""Evaluate only Phase 1 to Phase 2 using complete five-year training outcomes."""
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

from build_model import clustered_bootstrap_ci
from label_pipeline import bio

ROOT = Path(__file__).resolve().parent
STEP = "phase1_to_phase2"
NUMERIC = ["number_of_arms", "sponsor_track_record", "has_sponsor_history"]
CATEGORICAL = ["phase", "allocation", "intervention_model", "masking", "disease_area", "primary_purpose"]


def split_phase1(programs):
    if programs.drug_id.duplicated().any() or not programs.window_years.eq(5).all():
        raise ValueError("Expected one Phase 1 entry per drug and a five-year window")
    cutoff = pd.Timestamp("2016-01-01")
    train = programs.entry_date + pd.DateOffset(years=5) < cutoff
    test = programs.entry_date >= cutoff
    return programs.assign(split=np.select([train, test], ["train", "test"], "dropped"))


def fit_and_evaluate(df: pd.DataFrame, bio_flat: float, bio_area: pd.Series) -> dict:
    columns = NUMERIC + CATEGORICAL
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler())])
    pre = ColumnTransformer([("num", numeric, NUMERIC),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
    train, test = df[df.split == "train"], df[df.split == "test"]
    if train.label.nunique() < 2 or test.empty:
        raise ValueError(f"not enough data to fit: {len(train)} train / {len(test)} test rows")
    model.fit(train[columns], train.label)
    proba = model.predict_proba(test[columns])[:, 1]
    frac_pos, mean_pred = calibration_curve(test.label, proba, n_bins=5, strategy="quantile")

    # our own training-years rate by disease area: the fairest baseline, since it's scored on the same
    # label definition the model learns (BIO's table comes from a different program database)
    area_rate = train.groupby("disease_area").label.mean()
    benchmarks = {
        "BIO by area": test.disease_area.map(bio_area).fillna(bio_flat).to_numpy(),  # Unknown -> flat rate
        "BIO flat rate": np.full(len(test), bio_flat),
        "train-years area rate": test.disease_area.map(area_rate).fillna(train.label.mean()).to_numpy(),
    }
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
        "model": "logistic", "n_train": len(train), "n_test": len(test),
        "train_years": (int(train.entry_date.dt.year.min()), int(train.entry_date.dt.year.max())),
        "test_years": (int(test.entry_date.dt.year.min()), int(test.entry_date.dt.year.max())),
        "train_positive_rate": train.label.mean(), "test_positive_rate": test.label.mean(),
        "brier_model": brier_model,
        "vs_benchmarks": vs,
        "log_loss_model": log_loss(y, proba, labels=[0, 1]),
        "log_loss_area": log_loss(y, benchmarks["train-years area rate"], labels=[0, 1]),
        "calibration_curve": list(zip(mean_pred.round(3), frac_pos.round(3))),
        "bio_flat": bio_flat,
    }


def plot_calibration(results: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfectly calibrated")
    xs, ys = zip(*results["calibration_curve"])
    ax.plot(xs, ys, "o-", label=results["model"])
    ax.plot(results["bio_flat"], results["test_positive_rate"], "s", color="tab:orange", markersize=9,
            label=f"BIO 2021 flat rate ({results['bio_flat']:.3f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed advance rate")
    lo, hi = results["test_years"]
    ax.set_title(f"Phase 1 → 2: calibration, test entries {lo}–{hi}")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved calibration plot -> {out}")


def results_rows(r: dict) -> list[dict]:
    base = {"transition": STEP, "model": r["model"], "n_train": r["n_train"], "n_test": r["n_test"],
            "train_years": "-".join(map(str, r["train_years"])), "test_years": "-".join(map(str, r["test_years"])),
            "train_rate": round(r["train_positive_rate"], 4), "test_rate": round(r["test_positive_rate"], 4),
            "model_brier": round(r["brier_model"], 4)}
    return [{**base, "benchmark": name, "benchmark_brier": round(v["brier"], 4), "skill": round(v["skill"], 4),
             "model_minus_benchmark": round(v["diff"], 4), "ci_low": round(v["diff_ci"][0], 4),
             "ci_high": round(v["diff_ci"][1], 4)} for name, v in r["vs_benchmarks"].items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "output/model")
    args = parser.parse_args()
    labels = pd.read_csv(ROOT / "output/labels/phase1_to_phase2.csv", parse_dates=["entry_date"])
    programs = labels[labels.in_v1_label_set & labels.label.notna()].copy()
    features = pd.read_csv(ROOT / "output/model/trial_design_features.csv")
    sponsors = pd.read_csv(ROOT / "output/model/phase1_sponsor_history.csv")
    programs = programs.merge(features, left_on="entry_nct_id", right_on="nct_id", validate="many_to_one",
                              how="left", indicator=True)
    if programs._merge.ne("both").any():
        raise ValueError("An entry trial has no design features")
    programs = programs.drop(columns="_merge").merge(sponsors, on=["drug_id", "entry_nct_id"], how="left",
                                                      validate="one_to_one", indicator=True)
    if programs._merge.ne("both").any():
        raise ValueError("The sponsor-history snapshot does not cover the Phase 1 cohort")
    programs["has_sponsor_history"] = programs.sponsor_track_record.notna().astype(float)
    programs = split_phase1(programs.sort_values("entry_date"))
    rates = bio.benchmark(pd.read_csv(ROOT / "external/bio/bio2021_phase_transitions.csv"))
    by_area = rates[rates.transition == STEP].set_index("disease_area").bio_rate
    results = fit_and_evaluate(programs, by_area["All indications"], by_area)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results_rows(results)).to_csv(args.out_dir / "results_phase1_to_phase2.csv", index=False)
    plot_calibration(results, args.out_dir / "calibration_phase1_to_phase2.png")


if __name__ == "__main__":
    main()
