# Clinical Trial Success Prediction (Team 4, ML in Health)

Investors value biotech drugs using fixed success rates from one industry table (the BIO 2021 report). This project predicts each drug's own chance of passing **Phase 1 → Phase 2 → Phase 3 → FDA approval**, and tests whether that beats the table.

## Status

- **Done:** v1 label built from raw ClinicalTrials.gov data and checked against BIO 2021. First baseline model (Phase 2 → 3) trained; after fixing two lookahead leaks it beats BIO's phase × area table (Brier 0.184 vs 0.191) but only ties BIO's flat rate.
- **Next:** match approvals to their disease, de-duplicate disease rows, drug-level features, LightGBM, the other two transitions.

## Label v1 vs BIO 2021

| Step | Our label | BIO 2021 | Labeled examples |
|---|---|---|---|
| Phase 1 → 2 | 56.6% | 52.0% | 1,696 |
| Phase 2 → 3 | 27.6% | 28.9% | 8,438 |
| Phase 3 → approval | 41.1% | 52.4% | 963 |

Phase 3 → approval matches BIO for cancer drugs (43.9% vs 43.9%) but is low overall, because vaccines and blood products aren't in Drugs@FDA. More breakdowns are in `output/labels/`.

## Phase 2 → 3 baseline model

`build_features.py` pulls trial-design features from AACT (masking, allocation, enrollment, sponsor,
site counts) and `build_model.py` joins them onto the label table via `entry_nct_id`, then trains
logistic regression. Enrollment and site/country counts are left out of the model, since AACT stores
the final values for all of them, not what was known when Phase 2 started. (`build_features.py` still
keeps enrollment where `enrollment_type` is `ESTIMATED`, but that covers only ~5% of training rows and
those trials skew toward ones that stalled, so using it made the model badly overconfident.) Sponsor track record only uses that sponsor's outcomes that were already known when
the program entered Phase 2 (a success on its Phase 3 date, a failure only once its 5-year window ran out).

Train is entries up to 2015, test is entries from 2016 on, with every program of a drug on one side.
An older drug's 2016+ programs are dropped (1,006 rows) so train never sees test-period outcomes.

The entry trial's phase is a feature: Phase 1/2 trials count as reaching Phase 2 but advance less
often, and they're 24% of training programs vs 47% of test programs.

The main benchmark is BIO's **phase × disease-area table**, since that's what analysts plug into
valuations; the flat all-indications rate is kept for reference. 95% CIs are for model minus
benchmark, resampling whole drugs (2,000 draws), since one drug's programs aren't independent.
Skill = 1 − model Brier / benchmark Brier.

| Benchmark | Brier | Model skill | Model minus benchmark (95% CI) |
|---|---|---|---|
| **BIO 2021 by disease area** | 0.1911 | **+3.7%** | −0.0070 (−0.0128 to −0.0010) |
| BIO 2021 flat rate (28.9%) | 0.1855 | +0.7% | −0.0013 (−0.0060 to +0.0040) |
| Our own area-level rate, training years only | 0.1841 | 0.0% | +0.0001 (−0.0042 to +0.0047) |
| *Logistic regression (ours)* | *0.1842* | | |

The model beats BIO's area table, and the CI excludes zero on this split. In robustness checks the
model stays ahead in earlier test periods (2011–12, 2013–15) and when each drug + trial is counted
once, but those intervals include zero; the gain comes from non-oncology programs. It doesn't beat
BIO's flat rate or our own area rates by a meaningful margin. BIO's area table does worse than its own flat rate on 2016+
programs, so the area differences it reports don't hold up out of sample. An earlier version scored
0.169, but nearly all of that edge came from a sponsor track record that used outcomes not yet known
at entry. Calibration (`output/model/calibration_phase2_to_phase3.png`) is close through the middle
and somewhat overconfident in the top fifth (predicts ~40%, actual ~31%). Part of that is a falling
base rate: 27.6% of training programs advanced vs 24.2% in test.

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
python3 -m pytest           # tests (no data needed)
```

The label table is `output/labels/transitions.csv`. Use rows where `in_v1_label_set` is True and `label` is 0 or 1. To add trial features, join on `entry_nct_id`: the trial that started that program's phase (for Phase 2 → 3, the first Phase 2 trial in that disease).

## Open decisions

1. Keep Phase 1 → 2 (Phase 1 data is incomplete), or start at Phase 2?
2. Are 5 years (7 for approval) the right windows?
3. Add FDA's Purple Book so vaccines and blood products count as approved?

## Data sources

AACT (CTTI / ClinicalTrials.gov); Drugs@FDA (U.S. FDA); ChEMBL (EMBL-EBI, CC BY-SA 3.0); BIO, Informa Pharma Intelligence and QLS Advisors (2021), *Clinical Development Success Rates and Contributing Factors 2011–2020*.
