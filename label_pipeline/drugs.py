"""Gives every experimental drug in every trial one stable ID, so renamed drugs (MK-3475 = pembrolizumab) link up."""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from .aact import AactTables
from .names import compact, is_combination, is_not_a_drug, is_research_code, name_keys, strip_salt

log = logging.getLogger(__name__)
MAX_ALT_NAMES = 10  # interventions listing more alternate names than this are usually regimens


class Source(str, Enum):
    CHEMBL = "chembl"
    CHEMBL_ALT = "chembl_alt_name"
    FDA = "drugs_at_fda"
    CODE = "research_code"
    NAME = "name_only"


@dataclass(frozen=True)
class Resolver:
    chembl_index: dict[str, str]
    chembl_names: dict[str, str]
    fda_index: dict[str, str]
    fda_names: dict[str, str]

    def resolve(self, name: str, alt_names: list[str]) -> tuple[str, Source] | None:
        """Returns (drug_id, source), or None when the name is ambiguous (points at two different drugs)."""
        own = name_keys(name)
        alt = set().union(*(name_keys(a) for a in alt_names)) if alt_names else set()

        for keys, source in ((own, Source.CHEMBL), (alt, Source.CHEMBL_ALT)):
            hits = {self.chembl_index[k] for k in keys if k in self.chembl_index}
            if len(hits) == 1:
                return hits.pop(), source
            if len(hits) > 1:
                return None
        hits = {self.fda_index[k] for k in own | alt if k in self.fda_index}
        if len(hits) == 1:
            return hits.pop(), Source.FDA
        if len(hits) > 1:
            return None
        codes = sorted(k for k in own | alt if is_research_code(k))
        if codes:
            return f"CODE:{codes[0]}", Source.CODE
        primary = compact(strip_salt(name))
        return (f"NAME:{primary}", Source.NAME) if len(primary) >= 3 else None

    def display_name(self, drug_id: str) -> str:
        if drug_id in self.chembl_names:
            return self.chembl_names[drug_id]
        if drug_id in self.fda_names:
            return self.fda_names[drug_id]
        return drug_id.split(":", 1)[-1]


def build_resolver(chembl_index: dict[str, str], chembl_names: dict[str, str], fda: pd.DataFrame) -> Resolver:
    # FDA keys that ChEMBL already knows are resolved through ChEMBL, so one drug never gets two IDs
    fda_only = fda[~fda.key.isin(chembl_index.keys())]
    return Resolver(chembl_index=chembl_index, chembl_names=chembl_names,
                    fda_index=dict(zip(fda_only.key, fda_only.fda_id)),
                    fda_names=dict(zip(fda_only.fda_id, fda_only.ingredient.str.lower())))


def approval_dates(resolver: Resolver, fda: pd.DataFrame) -> pd.Series:
    """Earliest FDA approval per drug_id (ChEMBL or FDA IDs)."""
    ids = [resolver.chembl_index.get(k) or resolver.fda_index.get(k) for k in fda.key]
    return fda.assign(drug_id=ids).dropna(subset=["drug_id"]).groupby("drug_id").first_approval.min()


def map_trial_drugs(aact: AactTables, resolver: Resolver) -> tuple[pd.DataFrame, dict[str, int]]:
    """One row per (trial, experimental drug). Combinations, placebos and ambiguous names are skipped and counted."""
    rows, skipped = [], Counter()
    for iid, nct, name in zip(aact.interventions.id, aact.interventions.nct_id, aact.interventions.name):
        if is_not_a_drug(name):
            skipped["not_a_drug"] += 1
            continue
        if is_combination(name):
            skipped["combination"] += 1
            continue
        alts = [a for a in aact.other_names.get(iid, [])[:MAX_ALT_NAMES]
                if not is_combination(a) and not is_not_a_drug(a)]
        resolved = resolver.resolve(name, alts)
        if resolved is None:
            skipped["ambiguous_or_empty"] += 1
            continue
        drug_id, source = resolved
        rows.append((nct, iid, name, drug_id, resolver.display_name(drug_id), source.value))
    df = pd.DataFrame(rows, columns=["nct_id", "intervention_id", "intervention_name", "drug_id", "drug_name", "source"])
    df = df.drop_duplicates(["nct_id", "drug_id"])
    stats = {**{f"skipped_{k}": v for k, v in skipped.items()},
             **{f"matched_{k}": int(v) for k, v in df.source.value_counts().items()}}
    log.info("Trial-drug rows: %s; %s", len(df), stats)
    return df, stats
