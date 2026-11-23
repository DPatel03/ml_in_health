"""Train and evaluate separate models for the three clinical development transitions."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from label_pipeline import bio
from label_pipeline.labels import TRANSITIONS, window_end

HERE = Path(__file__).resolve().parent
TITLES = {"phase1_to_phase2": "Phase 1 → 2", "phase2_to_phase3": "Phase 2 → 3",
          "phase3_to_approval": "Phase 3 → approval"}

# Registry site counts and final enrollment can incorporate post-entry updates.
NUMERIC = ["number_of_arms", "sponsor_track_record", "has_sponsor_history"]
CATEGORICAL = ["phase", "allocation", "intervention_model", "masking", "disease_area", "primary_purpose"]
N_BOOT = 2000
# Pseudo-programs pulling a sponsor's track record toward the market-wide rate, so a sponsor with one or two
# closed programs doesn't score 0% or 100%. Validation Brier (train <= 2012, validate 2013-15) is flat for
# k = 0-20, so this is a guard against extremes, not a tuned value.
SHRINK_K = 5

# Ownership clauses ("X, a wholly owned subsidiary of Pfizer", "Wyeth is now ...") reflect who owns the
# company today, not when the trial ran, so they're cut rather than mapped to the parent: mapping would hand
# a 2015 Seagen trial Pfizer's track record because of a 2023 acquisition.
_OWNERSHIP_CUT = re.compile(r",|\(| [-–] | is now | now a | has merged | a wholly | a subsidiary | a member of ", re.I)
_TRAILING_NOISE = {"inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation", "co", "company", "and",
                   "gmbh", "ag", "sa", "sas", "nv", "bv", "kk", "plc", "ab", "as", "spa", "srl", "pty", "lp",
                   "pharmaceuticals", "pharmaceutical", "pharma"}


def normalize_sponsor(name: str) -> str:
    """One key per company: 'Novartis Pharmaceuticals' = 'Novartis', 'Genentech, Inc.' = 'Genentech'.
    Legal-entity suffixes and 'Pharma(ceuticals)' are dropped; different companies stay apart."""
    def words(s: str) -> list[str]:
        return re.sub(r"[^a-z0-9]+", " ", re.sub(r"[./]", "", s.lower().replace("&", " and "))).split()

    # a name that starts with the cut pattern would come out empty and lump unrelated sponsors together
    tokens = words(_OWNERSHIP_CUT.split(name, maxsplit=1)[0]) or words(name)
    stripped = list(tokens)
    while stripped and stripped[-1] in _TRAILING_NOISE:
        stripped.pop()
    return " ".join(stripped or tokens)


def join_trial_features(features: pd.DataFrame, programs: pd.DataFrame) -> pd.DataFrame:
    df = programs.merge(features, left_on="entry_nct_id", right_on="nct_id", how="left",
                        validate="many_to_one", indicator=True)
    if df._merge.ne("both").any():
        raise ValueError("Missing entry trial features. Rebuild features from the same AACT snapshot.")
    df = df.drop(columns="_merge")
    df["sponsor"] = df.lead_sponsor_name.map(normalize_sponsor, na_action="ignore")
    return df.sort_values("entry_date").reset_index(drop=True)


def build_sponsor_pool(features: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    pool = join_trial_features(features, transitions[transitions.in_v1_label_set & transitions.label.notna()])
    pool["sponsor_track_record"] = sponsor_track_record(pool)
    pool["has_sponsor_history"] = pool.sponsor_track_record.notna().astype(float)
    return pool


def build_program_features(features: pd.DataFrame, transitions: pd.DataFrame, transition: str,
                           train_end: int = 2015, test_start: int = 2016, test_end: int | None = None) -> pd.DataFrame:
    pool = build_sponsor_pool(features, transitions)
    return split_programs(pool[pool.transition == transition].copy(), train_end, test_start, test_end)


def split_programs(df: pd.DataFrame, train_end: int, test_start: int, test_end: int | None) -> pd.DataFrame:
    # time split: train on entries up to train_end, test on test_start+, and keep every program of the
    # same drug on one side (LABEL_RULES.md #8). A drug that started before test_start has its later
    # programs dropped rather than split across sides. With train_end < test_start - window, every
    # training outcome was known before the test period began (embargo).
    year = df.entry_date.dt.year
    drug_min_year = df.drug_id.map(df.groupby("drug_id").entry_date.min().dt.year)
    in_test = (drug_min_year >= test_start) & (year <= (test_end if test_end is not None else year.max()))
    mature = window_end(df.entry_date, df.window_years) < pd.Timestamp(test_start, 1, 1)
    df["split"] = np.select([(year <= train_end) & mature, in_test], ["train", "test"], "dropped")
    return df[df.split != "dropped"].reset_index(drop=True)


def _known_before(known: np.ndarray, labels: np.ndarray, at: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Count and number of successes among outcomes known strictly before each date in `at`."""
    order = np.argsort(known)
    cum = np.concatenate([[0.0], np.cumsum(labels[order])])
    n = np.searchsorted(known[order], at, side="left")
    return n, cum[n]


def sponsor_track_record(df: pd.DataFrame, k: float = SHRINK_K, targets: pd.DataFrame | None = None) -> np.ndarray:
    """Pool all transitions, counting successes and failures only after their entire windows close."""
    if targets is None:
        targets = df
    known = window_end(df.entry_date, df.window_years).to_numpy()
    y = df.label.to_numpy()
    entry = targets.entry_date.to_numpy()
    n_all, s_all = _known_before(known, y, entry)
    market = np.divide(s_all, n_all, out=np.full(len(targets), np.nan), where=n_all > 0)
    track_record = np.full(len(targets), np.nan)
    histories = df.groupby("sponsor").indices
    for sponsor, idx in targets.groupby("sponsor").indices.items():
        if sponsor not in histories:
            continue
        history = histories[sponsor]
        n, successes = _known_before(known[history], y[history], entry[idx])
        track_record[idx] = np.divide(successes + k * market[idx], n + k,
                                      out=np.full(len(idx), np.nan), where=n > 0)
    return track_record


def load_bio_rates(path: Path, transition: str) -> tuple[float, pd.Series]:
    """BIO 2021's all-indications rate and its phase x area table (what analysts plug into valuations).
    Phase 3 -> approval is BIO's Phase 3 -> NDA/BLA x NDA/BLA -> approval, the same step our label covers."""
    raw = pd.read_csv(path)
    bio.validate(raw)
    rates = bio.benchmark(raw)
    rates = rates[rates.transition == transition].set_index("disease_area").bio_rate
    return float(rates["All indications"]), rates


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


def make_model(kind: str, numeric: list[str] = NUMERIC) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    clf = (LogisticRegression(max_iter=1000) if kind == "logistic" else
           LGBMClassifier(n_estimators=300, num_leaves=3, min_child_samples=40, learning_rate=0.10,
                           reg_lambda=1.0, verbosity=-1, n_jobs=1, random_state=0))
    return Pipeline([("pre", pre), ("clf", clf)])


def log_odds(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_calibrator(model: Pipeline, train: pd.DataFrame, columns: list[str]) -> LogisticRegression:
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    oof = cross_val_predict(model, train[columns], train.label, groups=train.drug_id,
                            cv=folds, method="predict_proba")[:, 1]
    recent = train.entry_date.dt.year >= train.entry_date.dt.year.max() - 4
    return LogisticRegression().fit(log_odds(oof[recent]), train.label[recent])


def fit_and_evaluate(df: pd.DataFrame, bio_flat: float, bio_area: pd.Series, kind: str = "logistic") -> dict:
    numeric = NUMERIC + (["genetic_evidence_score"] if "genetic_evidence_score" in df else [])
    columns = numeric + CATEGORICAL
    model = make_model(kind, numeric)
    train, test = df[df.split == "train"], df[df.split == "test"]
    if train.label.nunique() < 2 or test.empty:
        raise ValueError(f"not enough data to fit: {len(train)} train / {len(test)} test rows")
    model.fit(train[columns], train.label)
    proba = model.predict_proba(test[columns])[:, 1]
    if kind == "lightgbm":
        proba = fit_calibrator(model, train, columns).predict_proba(log_odds(proba))[:, 1]
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
        "model": kind, "n_train": len(train), "n_test": len(test),
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


def plot_calibration(results: dict, transition: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfectly calibrated")
    xs, ys = zip(*results["calibration_curve"])
    ax.plot(xs, ys, "o-", label=results["model"])
    ax.plot(results["bio_flat"], results["test_positive_rate"], "s", color="tab:orange", markersize=9,
            label=f"BIO 2021 flat rate ({results['bio_flat']:.3f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed advance rate")
    lo, hi = results["test_years"]
    ax.set_title(f"{TITLES[transition]}: calibration, test entries {lo}–{hi}")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved calibration plot -> {out}")


def results_rows(transition: str, r: dict) -> list[dict]:
    base = {"transition": transition, "model": r["model"], "n_train": r["n_train"], "n_test": r["n_test"],
            "train_years": "-".join(map(str, r["train_years"])), "test_years": "-".join(map(str, r["test_years"])),
            "train_rate": round(r["train_positive_rate"], 4), "test_rate": round(r["test_positive_rate"], 4),
            "model_brier": round(r["brier_model"], 4)}
    return [{**base, "benchmark": name, "benchmark_brier": round(v["brier"], 4), "skill": round(v["skill"], 4),
             "model_minus_benchmark": round(v["diff"], 4), "ci_low": round(v["diff_ci"][0], 4),
             "ci_high": round(v["diff_ci"][1], 4)} for name, v in r["vs_benchmarks"].items()]


def print_results(transition: str, r: dict) -> None:
    print(f"\n=== {TITLES[transition]} ===")
    print(f"n_train={r['n_train']} ({r['train_years'][0]}-{r['train_years'][1]}, rate {r['train_positive_rate']:.3f})  "
          f"n_test={r['n_test']} ({r['test_years'][0]}-{r['test_years'][1]}, rate {r['test_positive_rate']:.3f})")
    print(f"Brier -- model: {r['brier_model']:.4f}")
    for name, v in r["vs_benchmarks"].items():
        lo, hi = v["diff_ci"]
        print(f"  vs {name:22s} {v['brier']:.4f}  skill {v['skill']:+.1%}  "
              f"model minus benchmark {v['diff']:+.4f} (95% CI {lo:+.4f} to {hi:+.4f}, drug bootstrap)")
    print(f"Log loss -- model: {r['log_loss_model']:.4f}  area baseline: {r['log_loss_area']:.4f}")
    print("Calibration curve (mean predicted, observed fraction):", r["calibration_curve"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--features", type=Path, default=HERE / "output" / "model" / "trial_design_features.csv")
    p.add_argument("--transitions", type=Path, default=HERE / "output" / "labels" / "transitions.csv")
    p.add_argument("--bio", type=Path, default=HERE / "external" / "bio" / "bio2021_phase_transitions.csv")
    p.add_argument("--model", choices=["logistic", "lightgbm"], help="override the default for the selected steps")
    p.add_argument("--genetic-evidence", type=Path, help="optional retrospective Phase 2 feature CSV")
    p.add_argument("--out-dir", type=Path, default=HERE / "output" / "model")
    p.add_argument("--transition", choices=TRANSITIONS, action="append",
                   help="step to model (repeatable); default: all three")
    p.add_argument("--train-end", type=int, default=2015, help="last entry year in training")
    p.add_argument("--test-start", type=int, default=2016, help="first entry year in test")
    p.add_argument("--test-end", type=int, help="last entry year in test (default: latest labeled)")
    a = p.parse_args(argv)
    if a.train_end >= a.test_start:
        p.error("--train-end must be before --test-start")
    if a.test_end is not None and a.test_end < a.test_start:
        p.error("--test-end must not be before --test-start")
    return a


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    for path, maker in [(a.features, "build_features.py"), (a.transitions, "build_labels.py"), (a.bio, "build_labels.py")]:
        if not path.is_file():
            print(f"error: {path} not found -- run {maker} first", file=sys.stderr)
            return 1
    features = pd.read_csv(a.features, parse_dates=["start"], low_memory=False)
    transitions = pd.read_csv(a.transitions, parse_dates=["entry_date", "next_date"], low_memory=False)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    pool = build_sponsor_pool(features, transitions)
    rows = []
    for transition in dict.fromkeys(a.transition or TRANSITIONS):
        df = split_programs(pool[pool.transition == transition].copy(), a.train_end, a.test_start, a.test_end)
        kind = a.model or ("lightgbm" if transition == TRANSITIONS[1] else "logistic")
        if a.genetic_evidence and transition == TRANSITIONS[1]:
            scores = pd.read_csv(a.genetic_evidence).set_index("drug_id").genetic_evidence_score
            df["genetic_evidence_score"] = df.drug_id.map(scores)
        try:
            results = fit_and_evaluate(df, *load_bio_rates(a.bio, transition), kind=kind)
        except ValueError as exc:
            print(f"error: {TITLES[transition]}: {exc}", file=sys.stderr)
            return 1
        print_results(transition, results)
        plot_calibration(results, transition, a.out_dir / f"calibration_{transition}.png")
        rows += results_rows(transition, results)

    filename = "baseline_results.csv" if not a.transition else "results_" + "_".join(dict.fromkeys(a.transition)) + ".csv"
    out = a.out_dir / filename
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
