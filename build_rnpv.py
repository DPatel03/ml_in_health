"""Illustrative rNPV: swap BIO's Phase 2 -> 3 rate for our model's, on real pending programs.

rNPV = P(Phase 2 -> 3) x P(Phase 3 -> approval) x V, where V is a hypothetical discounted value if
approved -- the same number in both scenarios, so the comparison isolates exactly what swapping the
Phase 2 -> 3 estimate does. P(Phase 3 -> approval) also stays BIO's own disease-area rate in both
scenarios: we haven't modeled that transition yet (see README, What's Next), so only the one step
we've actually built changes.

V is not a real valuation of these companies -- we have no revenue or pipeline data to ground one --
it is a single illustrative number so the output reads as a dollar swing, not just a ratio.

Usage:
    python3 build_rnpv.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ILLUSTRATIVE_VALUE_IF_APPROVED = 1_000  # $M, hypothetical, same for every program and both scenarios

# real pending programs from our own watchlist -- picked for large, unambiguous public sponsors so
# there's no risk of mismatching a drug to the wrong company
EXAMPLES = ["spartalizumab", "etentamig"]


def phase3_to_approval_rates(bio_path: Path) -> pd.Series:
    bio = pd.read_csv(bio_path)
    p3 = bio[bio.transition == "phase3_to_nda"].set_index("disease_area").success_rate
    nda = bio[bio.transition == "nda_to_approval"].set_index("disease_area").success_rate
    return (p3 * nda).round(3)


def main() -> None:
    watchlist = pd.read_csv(HERE / "output" / "model" / "watchlist_phase2_to_phase3.csv")
    features = pd.read_csv(HERE / "output" / "model" / "trial_design_features.csv")
    watchlist = watchlist.merge(features[["nct_id", "lead_sponsor_name"]],
                                 left_on="entry_nct_id", right_on="nct_id", how="left")
    p3_rates = phase3_to_approval_rates(HERE / "external" / "bio" / "bio2021_phase_transitions.csv")

    reliable = watchlist[watchlist.reliable]
    rows = []
    for drug in EXAMPLES:
        matches = reliable[reliable.drug_name == drug]
        r = matches.loc[matches.divergence_vs_area.abs().idxmax()]
        p3 = p3_rates[r.disease_area]
        bio_cumulative = r.bio_area_rate * p3
        our_cumulative = r.our_probability * p3
        rows.append({
            "drug": r.drug_name, "sponsor": r.lead_sponsor_name, "disease_area": r.disease_area,
            "BIO P(Phase2->3)": r.bio_area_rate, "Our P(Phase2->3)": r.our_probability,
            "P(Phase3->approval), BIO (held constant)": p3,
            "BIO cumulative PoS": round(bio_cumulative, 3), "Our cumulative PoS": round(our_cumulative, 3),
            "BIO rNPV ($M)": round(bio_cumulative * ILLUSTRATIVE_VALUE_IF_APPROVED, 1),
            "Our rNPV ($M)": round(our_cumulative * ILLUSTRATIVE_VALUE_IF_APPROVED, 1),
            "rNPV change": f"{(our_cumulative / bio_cumulative - 1):+.1%}",
        })

    out = pd.DataFrame(rows)
    out.to_csv(HERE / "output" / "model" / "rnpv_illustration.csv", index=False)
    print(f"Illustrative value if approved: ${ILLUSTRATIVE_VALUE_IF_APPROVED}M (hypothetical, same both scenarios)\n")
    for row in rows:
        print(f"{row['drug']} ({row['sponsor']}, {row['disease_area']}):")
        print(f"  BIO:  P2->3 {row['BIO P(Phase2->3)']:.1%} x P3->approval {row['P(Phase3->approval), BIO (held constant)']:.1%} "
              f"= {row['BIO cumulative PoS']:.1%} cumulative -> rNPV ${row['BIO rNPV ($M)']}M")
        print(f"  Ours: P2->3 {row['Our P(Phase2->3)']:.1%} x P3->approval {row['P(Phase3->approval), BIO (held constant)']:.1%} "
              f"= {row['Our cumulative PoS']:.1%} cumulative -> rNPV ${row['Our rNPV ($M)']}M")
        print(f"  Change: {row['rNPV change']}\n")
    print(f"wrote {HERE / 'output' / 'model' / 'rnpv_illustration.csv'}")


if __name__ == "__main__":
    main()
