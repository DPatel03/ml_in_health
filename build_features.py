"""Pulls one feature row per trial out of the raw AACT export: masking, allocation, enrollment,
sponsor, site counts. Scoped to the same trials as the label pipeline (industry-sponsored,
interventional, Phase 1-3, with a start date) so everything joins cleanly later.

Usage:
    python3 build_features.py --aact aact_export.zip        # extracted folder, Deep's layout
    python3 build_features.py --aact aact_export.zip.zip     # or the zip itself, unextracted
"""
from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PHASE_LEVEL = {"PHASE1": 1, "PHASE1/PHASE2": 2, "PHASE2": 2, "PHASE2/PHASE3": 3, "PHASE3": 3}
_READ = dict(sep="|", dtype=str, quoting=csv.QUOTE_NONE, on_bad_lines="skip")


def read_table(aact_path: Path, name: str, usecols: list[str]) -> pd.DataFrame:
    if aact_path.is_dir():
        return pd.read_csv(aact_path / f"{name}.txt", usecols=usecols, **_READ)
    with zipfile.ZipFile(aact_path) as z, z.open(f"{name}.txt") as f:
        return pd.read_csv(f, usecols=usecols, **_READ)


def build(aact_path: Path) -> pd.DataFrame:
    studies = read_table(aact_path, "studies", ["nct_id", "study_type", "phase", "start_date", "enrollment",
                                                 "enrollment_type", "number_of_arms"])
    studies["start"] = pd.to_datetime(studies.start_date, errors="coerce")
    sponsors = read_table(aact_path, "sponsors", ["nct_id", "agency_class", "lead_or_collaborator", "name"])
    lead = sponsors[sponsors.lead_or_collaborator == "lead"]
    industry_ids = set(lead[lead.agency_class == "INDUSTRY"].nct_id)
    lead_name = lead.drop_duplicates("nct_id").set_index("nct_id").name.rename("lead_sponsor_name")

    trials = studies[(studies.study_type == "INTERVENTIONAL") & studies.phase.isin(PHASE_LEVEL)
                      & studies.nct_id.isin(industry_ids) & studies.start.notna()].copy()
    trials["level"] = trials.phase.map(PHASE_LEVEL)
    ids = set(trials.nct_id)
    print(f"trials in scope: {len(trials):,}")

    designs = read_table(aact_path, "designs", ["nct_id", "allocation", "intervention_model", "masking",
                                                 "primary_purpose"])
    designs = designs[designs.nct_id.isin(ids)].drop_duplicates("nct_id").set_index("nct_id")

    facilities = read_table(aact_path, "facilities", ["nct_id", "country"])
    facilities = facilities[facilities.nct_id.isin(ids)]
    num_facilities = facilities.groupby("nct_id").size().rename("num_facilities")
    has_us_site = (facilities[facilities.country == "United States"].groupby("nct_id").size() > 0)
    has_us_site = has_us_site.reindex(trials.nct_id.unique(), fill_value=False).rename("has_us_site")

    countries = read_table(aact_path, "countries", ["nct_id", "name", "removed"])
    countries = countries[countries.nct_id.isin(ids) & (countries.removed != "t")]
    num_countries = countries.groupby("nct_id").name.nunique().rename("num_countries")

    out = (trials.set_index("nct_id")[["phase", "level", "start", "enrollment", "enrollment_type", "number_of_arms"]]
           .join(designs, how="left")
           .join(num_facilities, how="left")
           .join(num_countries, how="left")
           .join(has_us_site, how="left")
           .join(lead_name, how="left")
           .reset_index())
    out["enrollment"] = pd.to_numeric(out.enrollment, errors="coerce")
    # ACTUAL enrollment is only known once the trial is over, so it leaks the outcome -- keep ESTIMATED only
    out["enrollment"] = out.enrollment.where(out.enrollment_type == "ESTIMATED")
    out["number_of_arms"] = pd.to_numeric(out.number_of_arms, errors="coerce")
    out["num_facilities"] = out.num_facilities.fillna(0).astype(int)
    out["num_countries"] = out.num_countries.fillna(0).astype(int)
    out["has_us_site"] = out.has_us_site.fillna(False)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aact", type=Path, default=HERE / "aact_export.zip")
    p.add_argument("--out", type=Path, default=HERE / "output" / "model" / "trial_design_features.csv")
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df = build(a.aact)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df):,} rows x {len(df.columns)} cols -> {a.out}")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
