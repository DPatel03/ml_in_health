"""Build the Phase 1 to Phase 2 snapshot without updating shared transition files."""
import argparse
from pathlib import Path

import pandas as pd

from label_pipeline import aact, chembl, drugs, fda, labels
from label_pipeline.areas import phase1_area

ROOT = Path(__file__).resolve().parent


def phase1_programs(trial_drugs, tables, approvals):
    trials = trial_drugs.merge(tables.trials, on="nct_id", validate="many_to_one")
    first_trial = trials.groupby("drug_id").start.min()
    already_approved = approvals.reindex(first_trial.index).le(first_trial)
    trials = trials[~trials.drug_id.isin(first_trial.index[already_approved])]
    entries = labels._at_entry(trials[trials.phase == "PHASE1"], ["drug_id"]).sort_values("nct_id").copy()
    terms, ancestors, _ = labels._trial_terms(tables)
    entries["disease_area"] = [phase1_area(terms.get(n, set()), ancestors.get(n, set())) for n in entries.nct_id]
    grouped = entries.groupby("drug_id")
    programs = entries.drop_duplicates("drug_id").set_index("drug_id").rename(columns={
        "start": "entry_date", "nct_id": "entry_nct_id", "primary_purpose": "entry_primary_purpose"})
    programs["disease_area"] = grouped.disease_area.agg(labels._mode_area)
    programs["entry_has_us_site"] = grouped.has_us_site.any()
    programs["next_date"] = trials[trials.level >= 2].groupby("drug_id").start.min()
    programs["identified"] = programs.index.str.startswith(("CHEMBL", "FDA:"))
    programs["in_v1_label_set"] = (programs.identified & programs.entry_has_us_site
                                   & programs.entry_primary_purpose.map(labels._purpose_ok))
    programs["transition"] = "phase1_to_phase2"
    keep = ["transition", "drug_id", "drug_name", "disease_area", "entry_date", "entry_nct_id", "next_date",
            "entry_has_us_site", "entry_primary_purpose", "identified", "in_v1_label_set"]
    programs = programs[programs.entry_date.dt.year >= 2005].reset_index()[keep].sort_values("drug_id")
    return labels.apply_window(programs, labels.LabelConfig(snapshot=tables.snapshot))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "output/labels/phase1_to_phase2.csv")
    args = parser.parse_args()
    external = args.data_root / "external"
    ids, names = chembl.build_synonym_index(chembl.load_molecules(external / "chembl/clinical_molecules.jsonl"))
    fda_names = fda.load_approvals(external / "drugsatfda")
    resolver = drugs.build_phase1_resolver(ids, names, fda_names)
    tables = aact.load(args.data_root / "aact_export.zip", observed_starts_only=True)
    trial_drugs, _ = drugs.map_trial_drugs(tables, resolver)
    approvals = drugs.approval_dates(resolver, fda_names)
    programs = phase1_programs(trial_drugs, tables, approvals[approvals <= tables.snapshot])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    programs.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
