"""BIO / Informa / QLS 'Clinical Development Success Rates 2011-2020' benchmark tables."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

TRANSITIONS = ("phase1_to_phase2", "phase2_to_phase3", "phase3_to_nda", "nda_to_approval")
# Figure 2 (page 7): phase transition success by disease area. Figure 6 (page 12): oncology vs non-oncology.
SOURCE_PAGES = {7: "Figure 2", 12: "Figure 6"}
_ROW = re.compile(
    r"^(?P<area>[A-Za-z][A-Za-z \-]*?)\*?"
    + "".join(rf"\s+(?P<n{i}>\d+)\s+(?P<p{i}>\d+(?:\.\d+)?)%" for i in range(1, 5)) + r"\s*$"
)
PUBLISHED_LOA_FROM_PHASE1 = 0.079  # "All indications" LOA in Figure 5b, used to validate the extraction


def parse_rows(text: str) -> list[dict[str, object]]:
    """Parse 'Area n POS% n POS% n POS% n POS%' lines; keeps the first occurrence of each area."""
    rows, seen = [], set()
    for line in text.splitlines():
        m = _ROW.match(line.strip())
        if not m or (area := m["area"].strip()) in seen:
            continue
        seen.add(area)
        for i, transition in enumerate(TRANSITIONS, 1):
            rows.append({"disease_area": area, "transition": transition,
                         "n": int(m[f"n{i}"]), "success_rate": float(m[f"p{i}"]) / 100})
    return rows


def extract_tables(pdf_path: Path) -> pd.DataFrame:
    import pdfplumber  # only needed for extraction

    if not pdf_path.is_file():
        raise FileNotFoundError(f"BIO report not found: {pdf_path}")
    rows: list[dict[str, object]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, figure in SOURCE_PAGES.items():
            for r in parse_rows(pdf.pages[page_no - 1].extract_text() or ""):
                rows.append({**r, "source": f"{figure}, page {page_no}"})
    df = pd.DataFrame(rows).drop_duplicates(["disease_area", "transition"])
    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    allind = df[df.disease_area == "All indications"].set_index("transition").success_rate
    if set(allind.index) != set(TRANSITIONS):
        raise ValueError("BIO extraction is missing the 'All indications' row")
    loa = allind.prod()
    if abs(loa - PUBLISHED_LOA_FROM_PHASE1) > 0.002:
        raise ValueError(f"BIO extraction looks wrong: product of phase rates {loa:.3f} != published LOA 7.9%")
    if df.disease_area.nunique() < 17:  # 14 areas + Others + All indications + Oncology/Non-oncology split
        raise ValueError(f"Only {df.disease_area.nunique()} BIO disease areas parsed; expected 17")


def benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Rates in the shape our labels use: Phase 3 -> approval = Phase 3 -> NDA/BLA x NDA/BLA -> approval."""
    wide = df.pivot(index="disease_area", columns="transition", values="success_rate")
    n = df.pivot(index="disease_area", columns="transition", values="n")
    out = pd.DataFrame({
        "phase1_to_phase2": wide["phase1_to_phase2"],
        "phase2_to_phase3": wide["phase2_to_phase3"],
        "phase3_to_approval": wide["phase3_to_nda"] * wide["nda_to_approval"],
    })
    long = out.reset_index().melt(id_vars="disease_area", var_name="transition", value_name="bio_rate")
    long["bio_n"] = [int(n.at[a, "phase3_to_nda"] if t == "phase3_to_approval" else n.at[a, t])
                     for a, t in zip(long.disease_area, long.transition)]
    return long


def load_or_extract(csv_path: Path, pdf_path: Path) -> pd.DataFrame:
    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        validate(df)
        return df
    df = extract_tables(pdf_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df
