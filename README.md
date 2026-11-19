# Clinical Trial Success Prediction (Team 4, ML in Health)

Investors value biotech drugs using fixed success rates from one industry table (the BIO 2021 report). This project predicts each drug's own chance of passing **Phase 1 → Phase 2 → Phase 3 → FDA approval**, and tests whether that beats the table.

## Status

- **Done:** v1 label built from raw ClinicalTrials.gov data and checked against BIO 2021. Logistic-regression baselines for all three steps. None beats our own training-years area rates. Phase 2 → 3 is ahead of BIO's phase × area table only marginally (Brier 0.185 vs 0.190, CI touches zero); Phase 1 → 2 is worse than every benchmark.
- **Next:** match approvals to their disease, de-duplicate disease rows, drug-level features, LightGBM, Phase 1 design features that mean the same thing over time (arm counts don't).

## Label v1 vs BIO 2021

| Step | Our label | BIO 2021 | Labeled examples |
|---|---|---|---|
| Phase 1 → 2 | 51.8% | 52.0% | 1,249 |
| Phase 2 → 3 | 23.0% | 28.9% | 3,087 |
| Phase 3 → approval | 41.1% | 52.4% | 834 |

Phase 2 → 3 is one row per drug × BIO disease area (not per MeSH term), and no longer counts indication-blind FDA approvals as success — that cut the old ~8k-row MeSH explosion and removed ~26% fake successes. Phase 3 → approval is close to BIO for cancer (41.1% vs 43.9%) but low overall because vaccines/blood products aren't in Drugs@FDA. Full rules: `LABEL_RULES.md`.

## Baseline models (all three steps)

`build_features.py` pulls trial-design features from AACT (masking, allocation, enrollment, sponsor,
site counts) and `build_model.py` joins them onto the label table via `entry_nct_id`, then trains one
logistic regression per step. Results are in `output/model/baseline_results.csv` (Brier, skill and CI for
every step × benchmark), with a calibration plot per step.

What goes in:
- **Left out:** enrollment, site/country counts and "has a US site". AACT stores their final values, not
  what was known when the phase started. (`build_features.py` still keeps enrollment where
  `enrollment_type` is `ESTIMATED`, but that covers only ~5% of training rows and those trials skew toward
  ones that stalled, so using it made the model badly overconfident.)
- **Sponsor track record:** the sponsor's advance rate on its programs whose window had closed before this
  one entered, pulled toward the market-wide rate by 5 pseudo-programs, plus a flag for sponsors with no
  closed programs yet. A success only counts once its window closes, same as a failure: counting it as soon
  as it happens is still time-safe, but failures surface 5 years later, so the rate was inflated by however
  much was in flight (before 2010 every sponsor with history scored 100%). Sponsor names are normalized
  (`Novartis Pharmaceuticals` = `Novartis`, `Genentech, Inc.` = `Genentech`). Ownership notes like "a wholly
  owned subsidiary of Pfizer" are cut, not mapped to the parent, because they describe today's owner.
- **Entry trial's phase:** Phase 1/2 trials count as reaching Phase 2 but advance less often, and they're
  24% of training programs vs 47% of test programs. Phase 2/3 vs Phase 3 plays the same role for approval.

Train is entries up to 2015, test is entries from 2016 on (2016–2019 for approval, whose 7-year window
closes later), with every program of a drug on one side. An older drug's 2016+ programs are dropped (939
Phase 2 → 3 rows) so no drug has programs on both sides. Training labels can still resolve after 2015, since
a 2015 entry's window runs to 2020; `--train-end 2010` trains only on outcomes known by 2015.

Benchmarks: BIO's **phase × disease-area table** (what analysts plug into valuations), BIO's flat
all-indications rate, and our own area rates from the training years. Ours is the fairest test of whether
the features add anything, because it's scored on the same label definition the model learns. BIO's numbers
come from a different database, and on Phase 3 → approval our label misses CBER approvals (vaccines, blood
products), which makes BIO look worse than it is.

Model minus benchmark Brier (negative = model better), 95% CI resampling whole drugs (2,000 draws):

| Step | Train / test | Model Brier | vs BIO by area | vs BIO flat | vs our area rates |
|---|---|---|---|---|---|
| Phase 1 → 2 | 900 / 497 | 0.2644 | +0.0204 (+0.0049 to +0.0361) | +0.0136 (−0.0012 to +0.0284) | +0.0218 (+0.0081 to +0.0350) |
| Phase 2 → 3 | 4,403 / 2,652 | 0.1847 | −0.0051 (−0.0105 to +0.0004) | +0.0002 (−0.0042 to +0.0045) | +0.0024 (−0.0019 to +0.0066) |
| Phase 3 → approval | 582 / 264 | 0.2524 | −0.0041 (−0.0267 to +0.0182) | −0.0021 (−0.0242 to +0.0204) | +0.0077 (−0.0141 to +0.0302) |

**Phase 2 → 3** is ahead of BIO's area table by 0.005 (skill +2.7%), and the CI just touches zero. It ties
BIO's flat rate and our own area rates. The gain is all non-oncology (−0.0110, −0.0184 to −0.0033);
oncology is a tie. BIO's area table does worse than its own flat rate on 2016+ programs. The gap stays
negative but includes zero with the embargo (−0.0041), in earlier test periods (2011–12: −0.0014,
2013–15: −0.0047) and with each drug + trial counted once (−0.0043). Calibration is close through the
middle and overconfident in the top fifth (predicts ~39%, actual ~26%); the base rate also fell from 26.1%
in training to 23.9% in test.

**Phase 1 → 2** loses to every benchmark and is overconfident (top fifth predicts 82%, 60% advanced). The
main cause is a change in Phase 1 design: trials with more than 5 arms went from 12% of training programs
to 26% of test programs, and they advanced 71% of the time in training but 53% in test (dose escalation
plus expansion cohorts now register many arms). The base rate also fell, 54% → 49%.

**Phase 3 → approval** has only 264 test drugs, so CIs are about ±0.02 and nothing is distinguishable.

Earlier versions looked better, and each difference was a leak or a bias. 0.169 came from a sponsor track
record that used outcomes not yet known at entry. The −0.0070 vs BIO's area table (CI excluding zero) ran on
a label set that let later trials decide which programs counted (LABEL_RULES.md §7) and counted sponsor
successes as soon as they happened; with that track-record definition the gap would be −0.0079 (−0.0142 to
−0.0016). We use the closed-window version because it means the same thing in every year.

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
| `build_model.py` | Joins features to the labels, trains one baseline per step |
| `output/model/` | Feature table, `baseline_results.csv` and a calibration plot per step |
| `external/bio/bio2021_phase_transitions.csv` | BIO 2021 success rates |
| `tests/` | Tests for the label and model code |

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
python3 build_model.py      # trains all three baselines, writes output/model/baseline_results.csv + calibration plots
python3 -m pytest           # tests (no data needed)
```

Robustness runs (`--train-end`, `--test-start`, `--test-end`) should get their own `--out-dir` so they don't
overwrite the committed results; see the docstring in `build_model.py`.

The label table is `output/labels/transitions.csv`. Use rows where `in_v1_label_set` is True and `label` is 0 or 1. To add trial features, join on `entry_nct_id`: the trial that started that program's phase (for Phase 2 → 3, the first Phase 2 trial in that disease).

## Open decisions

1. Keep Phase 1 → 2 (Phase 1 data is incomplete), or start at Phase 2?
2. Are 5 years (7 for approval) the right windows?
3. Add FDA's Purple Book so vaccines and blood products count as approved?

## Data sources

AACT (CTTI / ClinicalTrials.gov); Drugs@FDA (U.S. FDA); ChEMBL (EMBL-EBI, CC BY-SA 3.0); BIO, Informa Pharma Intelligence and QLS Advisors (2021), *Clinical Development Success Rates and Contributing Factors 2011–2020*.
