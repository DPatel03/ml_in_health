# Label rules (v1)

What we predict: for one drug in one disease, does it move to the next phase?
Three steps: **Phase 1 → 2**, **Phase 2 → 3**, **Phase 3 → FDA approval**.

The code that applies these rules is `build_labels.py` (plus the `label_pipeline/` folder). Run it with `python3 build_labels.py`.

## 1. Which trials count
- ClinicalTrials.gov trials from our AACT export (snapshot Oct 8, 2026).
- Interventional, Phase 1, 1/2, 2, 2/3 or 3, with a start date.
- The lead sponsor is a company. BIO only counts company programs too.

## 2. Which drug a trial is testing
- Drugs in the trial's **experimental** arm. If a trial has no arm information, all its drugs.
- Left out: placebo, saline, standard of care, and anything that names two or more drugs at once (combinations, regimens like FOLFOX). BIO leaves out combinations as well.

## 3. One ID per drug
Each drug name is looked up in this order. The first match wins.
1. **ChEMBL** (16,809 drugs that reached clinical trials), using the trial's drug name. ChEMBL knows generic names, brand names and company codes, so MK-3475, Keytruda and pembrolizumab all get one ID. Salt forms map to the same drug.
2. ChEMBL, using the trial's alternate drug names listed in AACT.
3. **Drugs@FDA**, for approved drugs ChEMBL doesn't know.
4. Otherwise the drug keeps its company code (for example `CODE:abc123`) or its cleaned name.

A name that points at two different drugs is skipped, never guessed.

## 4. Disease
- The disease terms NLM assigns to each trial (MeSH terms in `browse_conditions`).
- A Phase 2 → 3 program is one drug + one MeSH term.
- A Phase 3 trial counts for that program if it has the same term, a narrower term (e.g. "Triple Negative Breast Neoplasms" for "Breast Neoplasms"), or a broader one.
- Each program also gets one of BIO's 14 disease areas (plus "Others"), so we can compare with BIO by area.

## 5. Phases
- Phase 1/2 counts as reaching Phase 2. Phase 2/3 counts as reaching Phase 3.
- A program enters a phase on the start date of its first trial in that phase.

## 6. Success and failure
| Step | Unit | Success means | Time allowed |
|---|---|---|---|
| Phase 1 → 2 | drug | any Phase 2 (or later) trial starts | 5 years |
| Phase 2 → 3 | drug + disease | a Phase 3 trial for the same disease starts, or FDA approval | 5 years |
| Phase 3 → approval | drug | first FDA approval (Drugs@FDA) | 7 years (5 + 2 for filing and FDA review) |

- **Failure** = the time ran out with no next step.
- **Pending** = the time hasn't run out yet. These get no label, even if they already moved on, because labeling only the fast successes would make recent years look better than they are. Pending programs are the watchlist.
- **Already past** = the next step started before the first registered trial of this phase. These are dropped, because the real start isn't in the data.
- Phase 1 → 2 and Phase 3 → approval are per drug. Phase 1 trials often enroll healthy volunteers, and Drugs@FDA doesn't say which disease an approval was for.

## 7. Which programs get a label in v1 (`in_v1_label_set`)
- **The drug was identified** through ChEMBL or Drugs@FDA. A drug known only by a code or bare name can't be followed from trial to trial or found in the FDA file, so it would almost always look like a failure.
- **The phase being judged had at least one US trial site.** This is our stand-in for BIO's "FDA registration-enabling" programs.
- Drugs already FDA-approved before their first trial are left out. These are old drugs tested for new uses.
- Program entered the phase in 2005 or later.

## 8. Time split for modeling
- Cohorts by entry year: 2005–2010, 2011–2015, 2016+.
- Proposal: train on entries up to 2015, test on 2016+.
- Keep every program of the same drug on the same side of the split.

## 9. How v1 compares with BIO 2021

| Step | Our v1 rate (n) | BIO 2021 | Gap |
|---|---|---|---|
| Phase 1 → 2 | 56.6% (1,696) | 52.0% | +4.6 |
| Phase 2 → 3 | 27.6% (8,438) | 28.9% | −1.3 |
| Phase 3 → approval | 41.1% (963) | 52.4% | −11.2 |

Phase 3 → approval for oncology matches BIO exactly (43.9%). Most of the remaining gap is in non-oncology:
- Vaccines, blood-clotting factors and other products regulated by FDA's CBER center aren't in Drugs@FDA, so their approvals are missed. Hematology is the clearest case: 18% for us vs 72% for BIO.
- A few prodrug names don't match (the trial says "adefovir", FDA lists "adefovir dipivoxil").

## 10. Known differences from BIO
- BIO marks a program "suspended" from company news. We infer it when no next-phase trial starts within the window.
- BIO's Phase 3 step ends at filing (then 90.6% of filings are approved). We go straight to approval, so we compare with BIO's Phase 3 → approval rate (52.4%).
- BIO includes old drugs tested for new uses. v1 doesn't.
- Phase 1 trials don't have to be registered under US law, so Phase 1 data is incomplete.
- Matching to ChEMBL favors drugs that became well known, which can push success rates up, especially Phase 1 → 2 outside oncology (70% for us vs 54% for BIO).

## Decisions for the team
1. Keep Phase 1 → 2, or start at Phase 2?
2. Are 5 years (7 for approval) the right windows? Rates with 3 / 5 / 7 years are in `output/labels/comparison_window_sensitivity.csv`.
3. Add old drugs tested for new uses (v2) to match BIO more closely?
4. Add FDA's Purple Book so vaccines and blood products count as approved (v2)?
