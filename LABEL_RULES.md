# Label rules (v1)

What we predict: for one drug in one disease area, does it move to the next phase?
Three steps: **Phase 1 → 2**, **Phase 2 → 3**, **Phase 3 → FDA approval**.

The code that applies these rules is `build_labels.py` (plus the `label_pipeline/` folder). Run it with `python3 build_labels.py`.

## 1. Which trials count
- ClinicalTrials.gov trials from our AACT export (snapshot Oct 8, 2026).
- Interventional, Phase 1, 1/2, 2, 2/3 or 3, with a start date on or before the snapshot. Withdrawn and not-yet-recruiting trials are excluded from entries and advancement evidence.
- The lead sponsor is a company. BIO only counts company programs too.

## 2. Which drug a trial is testing
- Drugs in the trial's **experimental** arm. If a trial has no arm information, all its drugs.
- Left out: placebo, saline, standard of care, and anything that names two or more drugs at once (combinations, regimens like FOLFOX). BIO leaves out combinations as well.

## 3. One ID per drug
Each drug name is looked up in this order. The first match wins.
1. **ChEMBL** (16,809 drugs that reached clinical trials), using the trial's drug name. ChEMBL knows generic names, brand names and company codes, so MK-3475, Keytruda and pembrolizumab all get one ID. Salt forms map to the same drug.
2. ChEMBL, using the trial's alternate drug names listed in AACT.
3. **Drugs@FDA**, including brand names that can be linked through an FDA ingredient to a single ChEMBL drug. Approvals dated after the snapshot are excluded.
4. Otherwise the drug keeps its company code (for example `CODE:abc123`) or its cleaned name.

A name that points at two different drugs is skipped, never guessed.

## 4. Disease
- The disease terms NLM assigns to each trial (MeSH terms in `browse_conditions`).
- A Phase 2 → 3 program is one **drug + one BIO disease area** (not one row per MeSH term). Programs without a listed condition cannot be disease-matched and are excluded from this step. MeSH is still used to decide whether a later Phase 3 matches; then rows collapse to the area. A single oncology basket trial that lists 30 tumor terms is one Oncology program, not 30.
- A Phase 3 trial counts if (a) it MeSH-matches any term in the area (same / narrower / broader term), or (b) the Phase 3 trial maps to the same specific BIO disease area, including areas represented in a multi-condition trial. Listed conditions take precedence over their broader ancestors. “Others” and “Unknown” alone cannot establish a match. (b) recovers cases the MeSH-ancestor table misses (~43% of listed terms have no known ancestors).
- An earlier MeSH-level version produced ~8k Phase 2 → 3 labeled rows from ~2.3k drugs (median 2 MeSH rows per drug, max 57). Drugs with more MeSH terms looked more successful because basket breadth is a confound. Collapsing to area removes that.

## 5. Phases
- Phase 1/2 counts as reaching Phase 2 (and as a Phase 2 **entry**, not a Phase 1 entry — BIO's convention). Phase 2/3 counts as reaching Phase 3.
- A program enters a phase on the start date of its first trial in that phase. That trial's ID is in `entry_nct_id`. For Phase 2 → 3 it's the first Phase 2 trial **for that disease area**. If two trials start on the same day, the lower NCT ID is used.
- What we record about a program (its US site, primary purpose, and its disease area for the per-drug steps) comes only from the trials that started on its entry date. Later trials aren't known when the program enters, so they can't decide what it looks like or whether it's in the label set.

## 6. Success and failure
| Step | Unit | Success means | Time allowed |
|---|---|---|---|
| Phase 1 → 2 | drug | any Phase 2 (or later) trial starts | 5 years |
| Phase 2 → 3 | drug + disease area | a Phase 3 trial for the same disease area starts | 5 years |
| Phase 3 → approval | drug | first FDA approval (Drugs@FDA) | 7 years (5 + 2 for filing and FDA review) |

- **Failure** = the time ran out with no next step.
- **Pending** = the time hasn't run out yet. These get no label, even if they already moved on, because labeling only the fast successes would make recent years look better than they are. Only eligible pending programs without an observed next step enter the watchlist.
- **Already past** = the next step started on or before the first registered trial of this phase. These are dropped: the real start isn't in the data, and a same-day next step is known the moment the program enters, so there's nothing to predict.
- Phase 1 → 2 and Phase 3 → approval are per drug. Phase 1 trials often enroll healthy volunteers, and Drugs@FDA doesn't say which disease an approval was for.
- **Phase 2 → 3 does not count FDA approval as success.** An earlier version did (for accelerated approval on Phase 2 data), but Drugs@FDA has no indication field, so one approval was marking every disease row for that drug as advanced — about 26% of previous "successes," many in the wrong area. True accelerated approvals without a Phase 3 are undercounted until we have indication-level approval data.

## 7. Which programs get a label in v1 (`in_v1_label_set`)
- **The drug was identified** through ChEMBL or Drugs@FDA. A drug known only by a code or bare name can't be followed from trial to trial or found in the FDA file, so it would almost always look like a failure.
- **The trial that started the phase had at least one US site** (any trial starting that same day counts too). This is our stand-in for BIO's "FDA registration-enabling" programs. An earlier version counted a US site on any trial in the phase, including ones that started years later. Programs that only got in that way advanced far more often (76% vs 53% for Phase 1 → 2, 44% vs 27% for Phase 2 → 3), because running more trials is itself a sign of progress.
- **Entry trial `primary_purpose` is TREATMENT or PREVENTION** (or missing — common on older registrations). BASIC_SCIENCE, DIAGNOSTIC, SUPPORTIVE_CARE, etc. are left out of v1; they are not the therapeutic development programs BIO tracks.
- Drugs already FDA-approved before their first trial are left out. These are old drugs tested for new uses.
- Program entered the phase in 2005 or later.

## 8. Time split for modeling
- Cohorts by entry year: 2005–2010, 2011–2015, 2016+.
- Hold out entries from 2016 onward. A training label must complete its entire five- or seven-year window before January 1, 2016.
- Keep every program of the same drug on the same side of the split.

## 9. How v1 compares with BIO 2021

| Step | Our v1 rate (n) | BIO 2021 | Gap |
|---|---|---|---|
| Phase 1 → 2 | 51.3% (1,248) | 52.0% | −0.7 |
| Phase 2 → 3 | 22.4% (3,087) | 28.9% | −6.5 |
| Phase 3 → approval | 41.0% (830) | 52.4% | −11.4 |

Phase 1 → 2 still tracks BIO. Phase 2 → 3 is lower than the old MeSH-exploded / approval-polluted figure on purpose. With a 7-year window it is 28.6% (≈ BIO 28.9%); the 5-year default is kept for now so every step uses the same clock except the known +2 years for approval review.

Phase 3 → approval for oncology is close to BIO (41.6% vs 43.9%). Most of the remaining gap is in non-oncology:
- Vaccines, blood-clotting factors and other products regulated by FDA's CBER center aren't in Drugs@FDA, so their approvals are missed. Hematology is the clearest case: 21% for us vs 72% for BIO.
- A few prodrug names don't match (the trial says "adefovir", FDA lists "adefovir dipivoxil").

## 10. Known differences from BIO
- BIO marks a program "suspended" from company news. We infer it when no next-phase trial starts within the window.
- BIO's Phase 3 step ends at filing (then 90.6% of filings are approved). We go straight to approval, so we compare with BIO's Phase 3 → approval rate (52.4%).
- BIO includes old drugs tested for new uses. v1 doesn't.
- Phase 1 trials don't have to be registered under US law, so Phase 1 data is incomplete.
- Matching to ChEMBL favors drugs that became well known, which can push success rates up, especially Phase 1 → 2 outside oncology.
- We require a disease-matched Phase 3 for Phase 2 → 3; BIO's proprietary timeline can mark progress we never see on ClinicalTrials.gov.

## Decisions for the team
1. Keep Phase 1 → 2, or start at Phase 2?
2. Are 5 years (7 for approval) the right windows? The sensitivity table includes both the base window and the effective window for each step. Rates are in `output/labels/comparison_window_sensitivity.csv`. Phase 2 → 3 is much closer to BIO at 7 years after the label cleanup.
3. Add old drugs tested for new uses (v2) to match BIO more closely?
4. Add FDA's Purple Book so vaccines and blood products count as approved (v2)?
5. Add indication-level approvals so accelerated Phase 2 → approval can count again without cross-disease pollution?
