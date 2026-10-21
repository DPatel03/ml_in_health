"""Build v1 phase-transition labels (Phase 1 -> 2 -> 3 -> FDA approval) and compare success rates with BIO 2021.

Inputs: AACT export, Drugs@FDA files, ChEMBL clinical-drug names, BIO 2021 report. Rules: LABEL_RULES.md.

Usage:
    python3 build_labels.py                    # 5-year success window
    python3 build_labels.py --window-years 7 --fetch-chembl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from label_pipeline import aact as aact_mod
from label_pipeline import bio, chembl, drugs, fda, labels

log = logging.getLogger("build_labels")
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Args:
    aact_dir: Path
    fda_dir: Path
    chembl_file: Path
    bio_pdf: Path
    bio_csv: Path
    out_dir: Path
    window_years: int
    approval_extra_years: int
    sensitivity: tuple[int, ...]
    first_entry_year: int
    fetch_chembl: bool


def parse_args(argv: list[str] | None = None) -> Args:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--aact-dir", type=Path, default=ROOT / "aact_export.zip")
    p.add_argument("--fda-dir", type=Path, default=ROOT / "external" / "drugsatfda")
    p.add_argument("--chembl-file", type=Path, default=ROOT / "external" / "chembl" / "clinical_molecules.jsonl")
    p.add_argument("--bio-pdf", type=Path, default=ROOT / "external" / "bio" / "ClinicalDevelopmentSuccessRates2011_2020.pdf")
    p.add_argument("--bio-csv", type=Path, default=ROOT / "external" / "bio" / "bio2021_phase_transitions.csv")
    p.add_argument("--out-dir", type=Path, default=ROOT / "output" / "labels")
    p.add_argument("--window-years", type=int, default=5, help="years a program gets to reach its next phase")
    p.add_argument("--approval-extra-years", type=int, default=2, help="extra years for Phase 3 -> approval")
    p.add_argument("--sensitivity", type=int, nargs="*", default=[3, 5, 7], help="other windows to report")
    p.add_argument("--first-entry-year", type=int, default=2005)
    p.add_argument("--fetch-chembl", action="store_true", help="re-download the ChEMBL dictionary")
    a = p.parse_args(argv)
    if a.window_years < 1 or a.approval_extra_years < 0:
        p.error("--window-years must be >= 1 and --approval-extra-years >= 0")
    return Args(a.aact_dir, a.fda_dir, a.chembl_file, a.bio_pdf, a.bio_csv, a.out_dir, a.window_years,
                a.approval_extra_years, tuple(a.sensitivity), a.first_entry_year, a.fetch_chembl)


def _with_bio(ours: pd.DataFrame, on: list[str], ref: pd.DataFrame) -> pd.DataFrame:
    out = ours.merge(ref, on=on, how="left")
    out["gap_pct_points"] = ((out.our_rate - out.bio_rate) * 100).round(1)
    return out.round({"our_rate": 3, "bio_rate": 3})


def compare(labeled: pd.DataFrame, bench: pd.DataFrame) -> dict[str, pd.DataFrame]:
    all_bio = bench[bench.disease_area == "All indications"][["transition", "bio_rate", "bio_n"]]
    v1 = labeled[labeled.in_v1_label_set]
    subsets = {"v1 label set (identified drug, US site)": v1,
               "identified drugs, any country": labeled[labeled.identified],
               "all industry drugs": labeled}
    overall = pd.concat([_with_bio(labels.success_rates(df), ["transition"], all_bio).assign(subset=name)
                         for name, df in subsets.items()])
    known = v1[v1.disease_area != "Unknown"]
    onc = known.assign(disease_area=np.where(known.disease_area == "Oncology", "Oncology", "Non-oncology"))
    return {
        "overall": overall,
        "oncology": _with_bio(labels.success_rates(onc, ["disease_area"]), ["transition", "disease_area"], bench),
        "by_area": _with_bio(labels.success_rates(known, ["disease_area"]), ["transition", "disease_area"], bench),
        "by_cohort": labels.success_rates(v1, ["cohort"]).round({"our_rate": 3}),
    }


def window_sensitivity(transitions: pd.DataFrame, snapshot: pd.Timestamp, a: Args) -> pd.DataFrame:
    """v1-label-set rates for other windows (the approval step always gets the extra years on top)."""
    rows = []
    for w in sorted(set(a.sensitivity) | {a.window_years}):
        cfg = labels.LabelConfig(snapshot=snapshot, window_years=w, approval_extra_years=a.approval_extra_years,
                                 first_entry_year=a.first_entry_year)
        labeled = labels.apply_window(transitions, cfg)
        rows.append(labels.success_rates(labeled[labeled.in_v1_label_set]).assign(window_years=w))
    return pd.concat(rows).round({"our_rate": 3})


def run(a: Args) -> dict[str, object]:
    bench = bio.benchmark(bio.load_or_extract(a.bio_csv, a.bio_pdf))
    if a.fetch_chembl or not a.chembl_file.is_file():
        log.info("Downloading ChEMBL clinical-drug dictionary to %s", a.chembl_file)
        asyncio.run(chembl.fetch_clinical_molecules(a.chembl_file))
    chembl_index, chembl_names = chembl.build_synonym_index(chembl.load_molecules(a.chembl_file))
    approvals_by_key = fda.load_approvals(a.fda_dir)
    resolver = drugs.build_resolver(chembl_index, chembl_names, approvals_by_key)
    approvals = drugs.approval_dates(resolver, approvals_by_key)

    tables = aact_mod.load(a.aact_dir)
    log.info("AACT snapshot %s: %s industry Phase 1-3 trials", tables.snapshot.date(), len(tables.trials))
    trial_drugs, match_stats = drugs.map_trial_drugs(tables, resolver)
    transitions, scope_stats = labels.build_transitions(trial_drugs, tables, approvals, a.first_entry_year)
    cfg = labels.LabelConfig(snapshot=tables.snapshot, window_years=a.window_years,
                             approval_extra_years=a.approval_extra_years, first_entry_year=a.first_entry_year)
    labeled = labels.apply_window(transitions, cfg)
    results = compare(labeled, bench)
    results["window_sensitivity"] = window_sensitivity(transitions, tables.snapshot, a)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    trial_drugs.to_csv(a.out_dir / "trial_drugs.csv", index=False)
    labeled.to_csv(a.out_dir / "transitions.csv", index=False)
    for name, df in results.items():
        df.to_csv(a.out_dir / f"comparison_{name}.csv", index=False)
    summary = {
        "snapshot": str(tables.snapshot.date()), "window_years": a.window_years,
        "approval_window_years": a.window_years + a.approval_extra_years,
        "v1_labeled_rows": labeled[labeled.in_v1_label_set].label.notna().groupby(labeled.transition).sum().to_dict(),
        "industry_trials": len(tables.trials), "trial_drug_rows": len(trial_drugs),
        "drug_matching": match_stats, **scope_stats,
        "outcomes": labeled.groupby(["transition", "outcome"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "overall": results["overall"].to_dict(orient="records"),
    }
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return {"summary": summary, **results}


def report(res: dict[str, object]) -> str:
    s = res["summary"]
    lines = [f"AACT snapshot {s['snapshot']} | {s['industry_trials']:,} industry trials | "
             f"{s['drugs_in_scope']:,} drugs in scope ({s['drugs_excluded_already_approved']:,} excluded: already approved)",
             f"Drug matching: {s['drug_matching']}", "",
             f"Success rates vs BIO ({s['window_years']}-year window, {s['approval_window_years']} for approval):"]
    for name in ("overall", "oncology", "by_cohort", "window_sensitivity"):
        lines += [f"\n[{name}]", res[name].to_string(index=False)]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    a = parse_args(argv)
    try:
        res = run(a)
    except (FileNotFoundError, ValueError, RuntimeError, httpx.HTTPError) as exc:
        log.error("%s", exc)
        return 1
    print(report(res))
    log.info("Outputs written to %s", a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
