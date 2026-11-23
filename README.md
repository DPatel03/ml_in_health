# Clinical Trial Success Prediction (Team 4, ML in Health)

Investors value biotech drugs using fixed success rates from one industry table (the BIO 2021 report). This project predicts each drug's own chance of passing **Phase 1 → Phase 2 → Phase 3 → FDA approval**, and tests whether that beats the table.

## What the October 8, 2026 snapshot shows

Each model predicts a specific recorded event. A drug can reach Phase 2 or 3 and still never reach approval. The observed rates below use eligible programs with a complete follow-up window. The BIO rates come from a separate industry dataset.

| Step | What counts as success | Labeled programs | Observed rate | BIO 2021 |
|---|---|---:|---:|---:|
| Phase 1 → 2 | A later Phase 2 or higher trial for the same drug starts within 5 years | 1,248 | 51.3% | 52.0% |
| Phase 2 → 3 | A qualifying Phase 3 trial for the drug and disease area starts within 5 years | 3,087 | 22.4% | 28.9% |
| Phase 3 → FDA approval | First recorded Drugs@FDA approval for the drug within 7 years | 830 | 41.0% | 52.4% |

**Phase 1 → 2.** The overall rate is close to BIO's. Predicting which individual drugs advance is harder. The logistic regression performed worse than BIO's disease-area rates on later trials.

**Phase 2 → 3.** Each drug contributes at most one program per BIO disease area. Direct condition names take priority over broad MeSH ancestors. The generic “Others” category requires a matching or related condition. FDA approval alone cannot establish that a trial advanced for the same disease. With a seven-year window the observed rate is 28.6%.

**Phase 3 → approval.** FDA brand and ingredient names share a drug ID when the mapping is unambiguous. This endpoint measures approval of the drug for any indication. In oncology, the recorded rate is 41.6% versus BIO's 43.9%. Approval coverage for biologics and the absence of indication-level matching limit the comparison.

Full eligibility and matching rules are in [LABEL_RULES.md](LABEL_RULES.md).

## Model evaluation

The models use trial design, disease area and the sponsor's history available at entry. Testing begins in 2016. Every training program must have completed its full observation window before January 1, 2016. That makes 2010 the last eligible training entry year for the five-year steps and 2008 for approval. Within each model, a drug appears on only one side of the split. Eligibility uses the available registry snapshot, so these are retrospective evaluations.

| Step | Model | Train / test programs | Model Brier | BIO by area Brier | Relative change in error |
|---|---|---:|---:|---:|---|
| Phase 1 → 2 | Logistic regression | 426 / 440 | 0.2705 | 0.2456 | 10.1% higher |
| Phase 2 → 3 | Recalibrated LightGBM | 1,011 / 1,100 | 0.1748 | 0.1892 | 7.6% lower |
| Phase 3 → approval | Logistic regression | 207 / 261 | 0.2561 | 0.2552 | 0.4% higher |

Lower Brier scores mean that the predicted probabilities were closer to the observed outcomes. The Phase 2 model's error difference against BIO by area was −0.0144 (95% interval −0.0207 to −0.0083, resampling whole drugs). Its advantage over disease-area rates calculated from our own training data was smaller and uncertain. For Phase 3 the interval also spans zero. See [all benchmark results](output/model/baseline_results.csv).

Sponsor history includes all three transitions. Both successes and failures enter the history after the full window closes. The Phase 2 calibrator uses out-of-fold training predictions with each drug kept in one fold. Final enrollment and site counts are excluded as model inputs. The primary evaluation also excludes the modern Open Targets genetic snapshot because its scores were unavailable at historical trial entry. For a separate retrospective comparison:

```bash
python3 build_model.py --transition phase2_to_phase3 --genetic-evidence external/opentargets/genetic_evidence.csv --out-dir /tmp/phase2-genetic
```

## Watchlist and illustrative valuation

Of 639 eligible Phase 2 programs with an open window, 85 had a recorded Phase 3 start by the snapshot date. The watchlist scores the remaining **554** using a model refitted on all 3,087 mature programs. The probabilities are estimates from phase entry. They do not account for how long an individual program has already waited. The `reliable` column marks an area whose observed success rate is within five percentage points of BIO. It cannot validate an individual forecast or establish market mispricing.

The valuation example holds the BIO Phase 3 → approval rate and a hypothetical $1 billion value at approval fixed. Changing just the Phase 2 probability changes the illustrative value as follows.

| Program | BIO Phase 2 → 3 | Model Phase 2 → 3 | BIO value | Model value |
|---|---:|---:|---:|---:|
| Givastomig, oncology | 24.6% | 26.8% | $108.0M | $117.6M |
| Lutikizumab, autoimmune | 31.4% | 25.1% | $192.9M | $154.2M |

The program-level records are in the [watchlist](output/model/watchlist_phase2_to_phase3.csv) and [valuation examples](output/model/rnpv_illustration.csv).

## Repo layout

| Path | What it is |
|---|---|
| `build_labels.py` | Builds the label and compares it with BIO |
| `label_pipeline/` | Label code: trials, drug names, FDA approvals, disease areas, success/failure, BIO tables |
| `LABEL_RULES.md` | Label rules and open decisions |
| `output/labels/` | Label table (`transitions.csv`), drug IDs per trial (`trial_drugs.csv`), comparison tables and `summary.json` |
| `build_features.py` | Trial-design features from AACT |
| `build_model.py` | Trains and evaluates one model per transition |
| `build_watchlist.py` | Scores pending programs for the mispricing watchlist |
| `build_rnpv.py` | Illustrative rNPV swap on two real pending programs |
| `build_genetic_evidence.py` | Open Targets genetic-evidence feature, cached in `external/opentargets/` |
| `output/model/` | Feature table, test results, calibration plots and watchlist |
| `external/bio/bio2021_phase_transitions.csv` | BIO 2021 success rates |
| `tests/` | Regression tests for labels and models |

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
python3 build_model.py      # trains all three models and writes results and calibration plots
python3 build_watchlist.py  # scores pending programs, writes output/model/watchlist_phase2_to_phase3.csv
python3 build_rnpv.py       # rNPV illustration, writes output/model/rnpv_illustration.csv
python3 -m pytest           # tests (no data needed)
```

The label table is `output/labels/transitions.csv`. Use rows where `in_v1_label_set` is True and `label` is 0 or 1. To add trial features, join on `entry_nct_id`: the trial that started that program's phase (for Phase 2 → 3, the first Phase 2 trial in that disease).

The standalone Phase 1 workflow already merged into `main` remains available:

```bash
python3 build_phase1_labels.py
python3 build_phase1_model.py
```

It uses `output/labels/phase1_to_phase2.csv` and the frozen sponsor-history snapshot. The shared workflow above regenerates all three transitions.

## Open decisions

1. Keep Phase 1 → 2 (Phase 1 data is incomplete), or start at Phase 2?
2. Are 5 years (7 for approval) the right windows?
3. Add FDA's Purple Book so vaccines and blood products count as approved?

## Data sources

AACT (CTTI / ClinicalTrials.gov); Drugs@FDA (U.S. FDA); ChEMBL (EMBL-EBI, CC BY-SA 3.0); BIO, Informa Pharma Intelligence and QLS Advisors (2021), *Clinical Development Success Rates and Contributing Factors 2011–2020*.
