# Clinical Trial Success Prediction (Team 4, ML in Health)

Investors value biotech drugs using fixed success rates from one industry table (the BIO 2021 report). This project predicts each drug's own chance of passing **Phase 1 → Phase 2 → Phase 3 → FDA approval**, and tests whether that beats the table.

## Status

- **Done:** v1 label built from raw ClinicalTrials.gov data and checked against BIO 2021.
- **Next:** features, models (logistic regression, LightGBM), evaluation (Brier score, calibration).

## Label v1 vs BIO 2021

| Step | Our label | BIO 2021 | Labeled examples |
|---|---|---|---|
| Phase 1 → 2 | 56.6% | 52.0% | 1,696 |
| Phase 2 → 3 | 27.6% | 28.9% | 8,438 |
| Phase 3 → approval | 41.1% | 52.4% | 963 |

Phase 3 → approval matches BIO for cancer drugs (43.9% vs 43.9%) but is low overall, because vaccines and blood products aren't in Drugs@FDA. More breakdowns are in `output/labels/`.

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
| `output/labels/` | Comparison tables and `summary.json` |
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
python3 -m pytest           # tests (no data needed)
```

The label table is `output/labels/transitions.csv`. Use rows where `in_v1_label_set` is True and `label` is 0 or 1.

## Open decisions

1. Keep Phase 1 → 2 (Phase 1 data is incomplete), or start at Phase 2?
2. Are 5 years (7 for approval) the right windows?
3. Add FDA's Purple Book so vaccines and blood products count as approved?

## Data sources

AACT (CTTI / ClinicalTrials.gov); Drugs@FDA (U.S. FDA); ChEMBL (EMBL-EBI, CC BY-SA 3.0); BIO, Informa Pharma Intelligence and QLS Advisors (2021), *Clinical Development Success Rates and Contributing Factors 2011–2020*.
