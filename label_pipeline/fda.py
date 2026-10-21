"""First FDA approval date for every ingredient and brand name in the Drugs@FDA data files."""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from .names import compact, name_keys, strip_biosimilar_suffix, strip_salt

_READ = dict(sep="\t", dtype=str, encoding="latin-1", quoting=csv.QUOTE_NONE)
REQUIRED = ("Applications.txt", "Products.txt", "Submissions.txt")


def fda_id(ingredient: str) -> str:
    return f"FDA:{compact(strip_salt(ingredient))}"


def load_approvals(fda_dir: Path) -> pd.DataFrame:
    """One row per lookup key: key, fda_id, ingredient, first_approval (earliest original approval, any application)."""
    for f in REQUIRED:
        if not (fda_dir / f).is_file():
            raise FileNotFoundError(f"Drugs@FDA file missing: {fda_dir / f} (unzip the dafdata*.zip there)")
    prods = pd.read_csv(fda_dir / "Products.txt", usecols=["ApplNo", "DrugName", "ActiveIngredient"], **_READ)
    subs = pd.read_csv(fda_dir / "Submissions.txt",
                       usecols=["ApplNo", "SubmissionType", "SubmissionStatus", "SubmissionStatusDate"], **_READ)
    orig = subs[(subs.SubmissionType == "ORIG") & (subs.SubmissionStatus == "AP")]
    first_by_app = pd.to_datetime(orig.SubmissionStatusDate, errors="coerce").groupby(orig.ApplNo).min()
    prods["approval_date"] = prods.ApplNo.map(first_by_app)
    prods = prods.dropna(subset=["approval_date"])

    rows: list[tuple[str, str, str, pd.Timestamp]] = []
    for brand, ingredients, date in zip(prods.DrugName.fillna(""), prods.ActiveIngredient.fillna(""), prods.approval_date):
        parts = [strip_biosimilar_suffix(p) for p in ingredients.split(";") if p.strip()]
        for part in parts:
            ing = strip_salt(part)
            for key in name_keys(part) | {compact(ing)}:
                rows.append((key, fda_id(ing), ing, date))
        if len(parts) == 1 and len(compact(brand)) >= 4:  # a combination product's brand name isn't one drug
            for key in name_keys(brand):
                rows.append((key, fda_id(strip_salt(parts[0])), strip_salt(parts[0]), date))
    df = pd.DataFrame(rows, columns=["key", "fda_id", "ingredient", "approval_date"])
    # a key that names two different ingredients is ambiguous; drop it instead of guessing
    owners = df.groupby("key").fda_id.nunique()
    df = df[df.key.isin(owners[owners == 1].index)]
    return (df.groupby("key").agg(fda_id=("fda_id", "first"), ingredient=("ingredient", "first"),
                                  first_approval=("approval_date", "min")).reset_index())
