"""Loads the AACT tables the label needs, already limited to industry-sponsored Phase 1-3 drug trials."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PHASE_LEVEL = {"PHASE1": 1, "PHASE1/PHASE2": 2, "PHASE2": 2, "PHASE2/PHASE3": 3, "PHASE3": 3}
DRUG_TYPES = ("DRUG", "BIOLOGICAL", "GENETIC", "COMBINATION_PRODUCT")
_READ = dict(sep="|", dtype=str, quoting=csv.QUOTE_NONE, on_bad_lines="skip")


@dataclass(frozen=True)
class AactTables:
    trials: pd.DataFrame          # nct_id, phase, level, start, has_us_site
    interventions: pd.DataFrame   # id, nct_id, name (experimental-arm drugs only)
    other_names: pd.Series        # intervention_id -> list of alternate names
    conditions: pd.DataFrame      # nct_id, mesh_term   (NLM-assigned terms)
    ancestors: pd.DataFrame       # nct_id, mesh_term   (their MeSH ancestors)
    snapshot: pd.Timestamp


def _read(aact_dir: Path, table: str, cols: list[str]) -> pd.DataFrame:
    path = aact_dir / f"{table}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"AACT table missing: {path}")
    return pd.read_csv(path, usecols=cols, **_READ)


def load(aact_dir: Path) -> AactTables:
    studies = _read(aact_dir, "studies", ["nct_id", "study_type", "phase", "start_date", "study_first_submitted_date"])
    snapshot = pd.to_datetime(studies.study_first_submitted_date, errors="coerce").max()
    studies["start"] = pd.to_datetime(studies.start_date, errors="coerce")
    sponsors = _read(aact_dir, "sponsors", ["nct_id", "agency_class", "lead_or_collaborator"])
    industry = set(sponsors[(sponsors.lead_or_collaborator == "lead") & (sponsors.agency_class == "INDUSTRY")].nct_id)
    trials = studies[(studies.study_type == "INTERVENTIONAL") & studies.phase.isin(PHASE_LEVEL)
                     & studies.nct_id.isin(industry) & studies.start.notna()].copy()
    trials["level"] = trials.phase.map(PHASE_LEVEL)
    ids = set(trials.nct_id)

    fac = _read(aact_dir, "facilities", ["nct_id", "country"])
    us = set(fac[(fac.country == "United States") & fac.nct_id.isin(ids)].nct_id)
    trials["has_us_site"] = trials.nct_id.isin(us)

    iv = _read(aact_dir, "interventions", ["id", "nct_id", "intervention_type", "name"])
    iv = iv[iv.nct_id.isin(ids) & iv.intervention_type.isin(DRUG_TYPES)]
    dgi = _read(aact_dir, "design_group_interventions", ["nct_id", "design_group_id", "intervention_id"])
    dg = _read(aact_dir, "design_groups", ["id", "nct_id", "group_type"]).rename(columns={"id": "design_group_id"})
    arms = dgi[dgi.nct_id.isin(ids)].merge(dg[dg.nct_id.isin(ids)], on=["nct_id", "design_group_id"])
    experimental = set(arms[arms.group_type == "EXPERIMENTAL"].intervention_id)
    # trials registered without arm data keep all their drug interventions
    iv = iv[iv.id.isin(experimental) | ~iv.nct_id.isin(set(arms.nct_id))]

    on = _read(aact_dir, "intervention_other_names", ["intervention_id", "name"])
    other_names = on[on.intervention_id.isin(set(iv.id))].groupby("intervention_id").name.agg(list)

    bc = _read(aact_dir, "browse_conditions", ["nct_id", "mesh_term", "mesh_type"])
    bc = bc[bc.nct_id.isin(ids)].dropna(subset=["mesh_term"])
    return AactTables(
        trials=trials[["nct_id", "phase", "level", "start", "has_us_site"]].reset_index(drop=True),
        interventions=iv[["id", "nct_id", "name"]].reset_index(drop=True),
        other_names=other_names,
        conditions=bc[bc.mesh_type == "mesh-list"][["nct_id", "mesh_term"]].drop_duplicates(),
        ancestors=bc[bc.mesh_type == "mesh-ancestor"][["nct_id", "mesh_term"]].drop_duplicates(),
        snapshot=snapshot,
    )
