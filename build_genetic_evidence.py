"""Genetic-evidence feature from Open Targets: does a drug's target have genetic support for any
disease? (King et al.: targets with genetic support are ~2x more likely to reach approval.)

For each drug, pulls its target(s) from Open Targets' mechanism-of-action data, then each target's
best "genetic_association" datatype score across its associated diseases. A drug's feature is the
best score across its own targets -- not matched to the trial's specific indication (Open Targets'
disease ontology and our BIO-style disease areas don't line up directly), so this is a drug-level
signal, not a drug+indication one. 0 for drugs Open Targets has no mechanism data for.

Usage:
    python3 build_genetic_evidence.py
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pandas as pd

HERE = Path(__file__).resolve().parent
API = "https://api.platform.opentargets.org/api/v4/graphql"
MOA_QUERY = """query($id: String!) { drug(chemblId: $id) {
    mechanismsOfAction { rows { targets { id } } } } }"""
DISEASE_QUERY = """query($id: String!) { target(ensemblId: $id) {
    associatedDiseases(page: {index: 0, size: 25}) { rows { datatypeScores { id score } } } } }"""


def post(client: httpx.Client, query: str, var_id: str) -> dict | None:
    for attempt in range(3):
        try:
            r = client.post(API, json={"query": query, "variables": {"id": var_id}}, timeout=20)
            if r.status_code == 200:
                return r.json().get("data")
        except httpx.HTTPError:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def fetch_targets(drug_ids: list[str], workers: int = 20) -> dict[str, list[str]]:
    """drug_id (ChEMBL) -> list of Ensembl target ids."""
    out: dict[str, list[str]] = {}
    with httpx.Client() as client, ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(post, client, MOA_QUERY, d): d for d in drug_ids}
        for i, fut in enumerate(as_completed(futures), 1):
            d = futures[fut]
            data = fut.result()
            rows = (data or {}).get("drug", {}) or {}
            moa = rows.get("mechanismsOfAction", {}) or {}
            targets = {t["id"] for row in moa.get("rows", []) for t in row.get("targets", [])}
            out[d] = sorted(targets)
            if i % 500 == 0:
                print(f"  targets: {i:,}/{len(drug_ids):,}")
    return out


def fetch_genetic_scores(target_ids: list[str], workers: int = 20) -> dict[str, float]:
    """Ensembl target id -> best genetic_association datatype score among its top-scored diseases."""
    out: dict[str, float] = {}
    with httpx.Client() as client, ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(post, client, DISEASE_QUERY, t): t for t in target_ids}
        for i, fut in enumerate(as_completed(futures), 1):
            t = futures[fut]
            data = fut.result()
            rows = ((data or {}).get("target", {}) or {}).get("associatedDiseases", {}) or {}
            best = 0.0
            for row in rows.get("rows", []):
                for ds in row.get("datatypeScores", []):
                    if ds["id"] == "genetic_association":
                        best = max(best, ds["score"])
            out[t] = best
            if i % 200 == 0:
                print(f"  genetic scores: {i:,}/{len(target_ids):,}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--transitions", type=Path, default=HERE / "output" / "labels" / "transitions.csv")
    p.add_argument("--out", type=Path, default=HERE / "external" / "opentargets" / "genetic_evidence.csv")
    p.add_argument("--cache", type=Path, default=HERE / "external" / "opentargets" / "_raw_cache.json")
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    t = pd.read_csv(a.transitions)
    sub = t[(t.transition == "phase2_to_phase3") & t.in_v1_label_set
            & (t.label.notna() | (t.outcome == "pending"))]
    drug_ids = sorted(sub[sub.drug_id.str.startswith("CHEMBL", na=False)].drug_id.unique())
    print(f"drugs needing a lookup: {len(drug_ids):,}")

    if a.cache.is_file():
        cache = json.loads(a.cache.read_text())
    else:
        cache = {}

    if "drug_targets" not in cache:
        print("fetching drug -> target mechanism-of-action data...")
        cache["drug_targets"] = fetch_targets(drug_ids)
        a.cache.write_text(json.dumps(cache))
    drug_targets = cache["drug_targets"]

    all_targets = sorted({tid for ts in drug_targets.values() for tid in ts})
    print(f"unique targets across all drugs: {len(all_targets):,}")

    if "target_scores" not in cache:
        print("fetching target -> genetic evidence scores...")
        cache["target_scores"] = fetch_genetic_scores(all_targets)
        a.cache.write_text(json.dumps(cache))
    target_scores = cache["target_scores"]

    rows = []
    for d in drug_ids:
        targets = drug_targets.get(d, [])
        # no target found at all (no Open Targets mechanism-of-action data) is missing, not zero --
        # it's much more about how well-characterized the drug is than about genetic evidence (drugs
        # with no mechanism data succeed far less often, 14% vs 33%, a confound distinct from the
        # genetic-evidence question), so it's left for the model's own imputer to handle as missing
        score = max((target_scores.get(t, 0.0) for t in targets), default=None) if targets else None
        rows.append({"drug_id": d, "n_targets": len(targets),
                      "genetic_evidence_score": round(score, 4) if score is not None else ""})

    out = pd.DataFrame(rows)
    out["genetic_evidence_score"] = pd.to_numeric(out.genetic_evidence_score, errors="coerce")
    out.to_csv(a.out, index=False)
    print(f"\nwrote {len(out):,} rows -> {a.out}")
    print(f"drugs with a target found: {(out.n_targets > 0).mean():.1%}")
    print(f"drugs with genetic_evidence_score > 0: {(out.genetic_evidence_score > 0).mean():.1%}")


if __name__ == "__main__":
    main()
