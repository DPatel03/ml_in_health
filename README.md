# Clinical Trial Success Prediction (Team 4, ML in Health)

Investors value biotech drugs using fixed success rates from one industry table (the BIO 2021 report). This project predicts each drug's own chance of passing **Phase 1 → Phase 2 → Phase 3 → FDA approval**, and tests whether that beats the table.

## Status

- **Done:** v1 label built from raw ClinicalTrials.gov data and checked against BIO 2021. Phase 2 → 3 model trained (LightGBM, after a logistic baseline and fixing two lookahead leaks); beats BIO's flat rate and phase × area table, both CIs excluding zero. Recalibrated on the most recent training years, which fixed the overconfident top fifth. Mispricing watchlist and an rNPV illustration, both scored on pending programs.
- **Next:** Purple Book for biologics (would also fix the label gap the watchlist ran into below), match approvals to their disease, de-duplicate disease rows, drug-level features (mechanism, FDA designations), the other two transitions.

## Label v1 vs BIO 2021

| Step | Our label | BIO 2021 | Labeled examples |
|---|---|---|---|
| Phase 1 → 2 | 56.6% | 52.0% | 1,696 |
| Phase 2 → 3 | 27.6% | 28.9% | 8,438 |
| Phase 3 → approval | 41.1% | 52.4% | 963 |

Phase 3 → approval matches BIO for cancer drugs (43.9% vs 43.9%) but is low overall, because vaccines and blood products aren't in Drugs@FDA. More breakdowns are in `output/labels/`.

## Phase 2 → 3 model

`build_features.py` pulls trial-design features from AACT (masking, allocation, enrollment, sponsor,
site counts) and `build_model.py` joins them onto the label table via `entry_nct_id`, then trains a
model (`--model lightgbm`, default, or `--model logistic` for the original baseline). Enrollment and
site/country counts are left out, since AACT stores the final values for all of them, not what was
known when Phase 2 started. (`build_features.py` still keeps enrollment where `enrollment_type` is
`ESTIMATED`, but that covers only ~5% of training rows and those trials skew toward ones that
stalled, so using it made the model badly overconfident.) Sponsor track record only uses that
sponsor's outcomes already known when the program entered Phase 2 (a success on its next-phase date,
a failure only once its window ran out), and now spans **all three transitions**, not just this one
-- a sponsor's Phase 1 → 2 and Phase 3 → approval record count too. `primary_purpose` (treatment,
prevention, diagnostic, ...) is a feature as well; it was sitting unused in the features table.

Train is entries up to 2015, test is entries from 2016 on, with every program of a drug on one side.
An older drug's 2016+ programs are dropped from this split (1,006 rows) so train never sees
test-period outcomes; the final model used for the watchlist below retrains on all 8,438 labeled
programs, since there's no more held-out evaluation to protect at that point.

The entry trial's phase is a feature: Phase 1/2 trials count as reaching Phase 2 but advance less
often, and they're 24% of training programs vs 47% of test programs.

The main benchmark is BIO's **phase × disease-area table**, since that's what analysts plug into
valuations; the flat all-indications rate is kept for reference. 95% CIs are for model minus
benchmark, resampling whole drugs (2,000 draws), since one drug's programs aren't independent.
Skill = 1 − model Brier / benchmark Brier. LightGBM is small and heavily regularized (7 leaves,
min 40 samples per leaf) given ~4,700 training rows; train Brier (0.179) vs test Brier before
recalibration (0.185) is a modest gap, not runaway overfitting.

| Benchmark | Brier | Model skill | Model minus benchmark (95% CI) |
|---|---|---|---|
| **BIO 2021 by disease area** | 0.1911 | **+4.9%** | −0.0094 (−0.0145 to −0.0047) |
| **BIO 2021 flat rate (28.9%)** | 0.1855 | **+2.0%** | −0.0037 (−0.0063 to −0.0010) |
| Our own area-level rate, training years only | 0.1841 | +1.3% | −0.0023 (−0.0056 to +0.0007) |
| *LightGBM, recalibrated (ours)* | *0.1818* | | |

The model now beats both BIO's area table **and its flat rate**, with the CI excluding zero on
each; only the comparison against our own area rates still includes zero. The earlier logistic
baseline (0.1837) only cleared the area-table bar. In robustness checks the model stays ahead in
earlier test periods and when each drug + trial is counted once, but those intervals include zero;
the gain comes from non-oncology programs. BIO's area table does worse than its own flat rate on
2016+ programs, so the area differences it reports don't hold up out of sample. An earlier version
scored 0.169, but nearly all of that edge came from a sponsor track record that used outcomes not
yet known at entry.

Calibration (`output/model/calibration_phase2_to_phase3.png`) was close through the middle but
overconfident in the top fifth (predicted ~40%, actual ~31%), partly because the training years had
a higher success rate than test: 27.6% vs 24.2%. We recalibrated with Platt scaling fit on
out-of-fold predictions from the 2011–2015 training cohort, the years closest to test. That pulled
the top fifth in and tightened the gap across every bin. The table above already reflects the
recalibrated model.

## Mispricing watchlist

`build_watchlist.py` refits the model on every labeled program across all three transitions (8,438
Phase 2 → 3 programs once the train/test split is no longer needed) and scores pending Phase 2 → 3
programs, ranking them by how far our probability diverges from BIO's disease-area rate. Of the
9,537 pending programs, only **1,901** have a resolved drug identity and a US site
(`in_v1_label_set`) -- the same bar used everywhere else -- so only those are scored. Output:
`output/model/watchlist_phase2_to_phase3.csv`.

Before trusting a "mispriced" program, check its `reliable` column. Our own training rate runs more
than 5pp off BIO's in six disease areas: Hematology (−18pp), Others (−14pp), Infectious disease
(−12pp), Gastroenterology (−12pp), Ophthalmology (−10pp) and Urology (+7pp) -- the first five match
the Drugs@FDA biologics gap described above, so a large divergence there is likely that label gap
showing up again, not real mispricing. Those rows are marked unreliable and excluded from the ranked
list. The reliable watchlist shows the model both more optimistic than BIO (oncology, cardiovascular,
respiratory programs, +19 to +21pp) and more pessimistic (metabolic programs, −20 to −28pp) -- a
real, two-directional result, not a one-sided artifact. Fixing the label gap with the Purple Book
should make more of the 9,537 scoreable and may change which disease areas are trustworthy.

## rNPV illustration

`build_rnpv.py` shows what swapping our Phase 2 → 3 estimate for BIO's does to a simple rNPV, on
two real pending programs from the reliable watchlist above: **sonrotoclax (BeiGene, Oncology)**
and **etentamig (AbbVie, Metabolic)**. These are the actual drug and sponsor names from AACT, not
invented examples.

rNPV = P(Phase 2 → 3) × P(Phase 3 → approval) × V. We haven't modeled Phase 3 → approval yet, so
P(Phase 3 → approval) is BIO's own disease-area rate in both scenarios -- only the Phase 2 → 3 term
changes, which isolates exactly what our model adds. **V is a hypothetical $1,000M "value if
approved," the same number in both scenarios** -- we have no real revenue or pipeline data for
these companies, so V is not a valuation claim about BeiGene or AbbVie, only a constant that turns
the probability swing into a dollar one for illustration.

| Drug (sponsor, area) | BIO P(P2→3) | Our P(P2→3) | BIO rNPV | Our rNPV | Change |
|---|---|---|---|---|---|
| sonrotoclax (BeiGene, Oncology) | 24.6% | 36.3% | $108.0M | $159.4M | **+47.6%** |
| etentamig (AbbVie, Metabolic) | 45.0% | 30.0% | $250.2M | $166.8M | **−33.3%** |

One up, one down -- not a one-directional artifact. Output: `output/model/rnpv_illustration.csv`.

## How the label works

Full rules: **[LABEL_RULES.md](LABEL_RULES.md)**. In short:

1. Company-sponsored Phase 1–3 trials from ClinicalTrials.gov (AACT).
2. The drug in each trial's experimental arm, matched to one ID with ChEMBL (MK-3475 = pembrolizumab).
3. A program is one drug + one disease. Success means reaching the next phase within 5 years, or FDA approval within 7 years of Phase 3.
4. Programs whose time hasn't run out are "pending" and get no label.

## Repo layout

| Path | What it is |
|---|---|
| `build_labels.py` | Builds the label and compares it with BIO |
| `label_pipeline/` | Label code: trials, drug names, FDA approvals, disease areas, success/failure, BIO tables |
| `LABEL_RULES.md` | Label rules and open decisions |
| `output/labels/` | Label table (`transitions.csv`), drug IDs per trial (`trial_drugs.csv`), comparison tables and `summary.json` |
| `build_features.py` | Trial-design features from AACT |
| `build_model.py` | Joins features to the labels, trains the baseline model |
| `build_watchlist.py` | Scores pending programs for the mispricing watchlist |
| `build_rnpv.py` | Illustrative rNPV swap on two real pending programs |
| `output/model/` | Feature table and the Phase 2 → 3 calibration plot |
| `external/bio/bio2021_phase_transitions.csv` | BIO 2021 success rates |
| `tests/` | Tests for the label code |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data (not in git)

| Data | Where to get it | Put it in |
|---|---|---|
| AACT (ClinicalTrials.gov) | aact.ctti-clinicaltrials.org → Downloads → pipe-delimited flat files | `aact_export.zip/` (folder of `.txt` files) |
| Drugs@FDA | https://www.fda.gov/media/89850/download (unzip) | `external/drugsatfda/` |
| ChEMBL drug names | `python3 build_labels.py --fetch-chembl` | `external/chembl/` |

## Run

```bash
python3 build_labels.py     # builds output/labels/ (~30 s)
python3 build_features.py   # builds output/model/trial_design_features.csv
python3 build_model.py      # trains the Phase 2 -> 3 baseline, writes output/model/calibration_phase2_to_phase3.png
python3 build_watchlist.py  # scores pending programs, writes output/model/watchlist_phase2_to_phase3.csv
python3 build_rnpv.py       # rNPV illustration, writes output/model/rnpv_illustration.csv
python3 -m pytest           # tests (no data needed)
```

The label table is `output/labels/transitions.csv`. Use rows where `in_v1_label_set` is True and `label` is 0 or 1. To add trial features, join on `entry_nct_id`: the trial that started that program's phase (for Phase 2 → 3, the first Phase 2 trial in that disease).

## Open decisions

1. Keep Phase 1 → 2 (Phase 1 data is incomplete), or start at Phase 2?
2. Are 5 years (7 for approval) the right windows?
3. Add FDA's Purple Book so vaccines and blood products count as approved?

## Data sources

AACT (CTTI / ClinicalTrials.gov); Drugs@FDA (U.S. FDA); ChEMBL (EMBL-EBI, CC BY-SA 3.0); BIO, Informa Pharma Intelligence and QLS Advisors (2021), *Clinical Development Success Rates and Contributing Factors 2011–2020*.
