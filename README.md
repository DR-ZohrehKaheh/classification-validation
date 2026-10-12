# Classification Validation, Calibration and Batch Scoring

A Python workflow for preparing environmental classification data, comparing validation designs, calibrating model scores and scoring new observations with saved models. The project demonstrates practical model evaluation and reproducible batch inference.

> **Reuse status:** No repository-wide software licence has been granted. See [LICENSE_STATUS.md](LICENSE_STATUS.md). The execution instructions below describe the workflow; they do not grant reuse permission.

**Project supervisor: Dr. Zohreh Kaheh.**  
**Undergraduate collaborator: Ruihan Sun.**

Dr. Kaheh designed the project based on her earlier work, provided the initial design and implementation teaching, supervised the student's initial implementation, and subsequently developed the code further. This contribution description reflects the maintainer's account.

## Technical focus

- Data validation and spatial and temporal feature engineering.
- Comparison of validation designs and baseline classifiers.
- Probability calibration, model persistence and batch scoring.
- Automated checks for output completeness and probability bounds.

## Run

Python 3.12 was used for verification. Create an isolated Python environment, install `requirements.txt`, and run from this directory:

```bash
python -m pip install -r requirements.txt
python run_pipeline.py --data /absolute/path/to/input.csv --output outputs
```

This runs utility tests, the complete analysis workflow, output-integrity checks, saved-model loading and batch scoring. It fails if a subprocess or integrity check fails. Outputs are created locally and must not be committed. The wrapper limits numerical-library threads to one by default; set `PORTFOLIO_NUM_THREADS` to change this without changing the training design.

## Input contract

The CSV must contain `latitude`, `longitude`, `datetime`, `Wildfire`, `pr`, `rmax`, `rmin`, `sph`, `srad`, `tmmn`, `tmmx`, `vs`, `bi`, `fm100`, `fm1000`, `erc`, `etr`, `pet`, and `vpd`. Labels must be `Yes` or `No`; `tmmn` and `tmmx` are in kelvin. This workflow expects the 2020–2022 temporal design and enough spatial blocks for five folds. Input data are not distributed.

After training, score a separate compatible CSV:

```bash
python score_new_data.py --input /absolute/path/to/new_rows.csv --model-dir outputs/models --output outputs/new_scores.csv
```

The scoring CSV needs all columns listed above except `Wildfire`. It must contain valid dates and numeric predictor values. A compatible input CSV is required to run this project; no example dataset is bundled.

## Scope

Execution was checked with the supplied input. That does not resolve its source provenance, label-construction or sampling limitations. Treat the output as a model score for the supplied label, not a validated operational wildfire probability. No research metrics or paper excerpts are published here.

## Verification and contributions

See [VERIFICATION.md](VERIFICATION.md) for the tested execution scope and [ATTRIBUTION.md](ATTRIBUTION.md) for supervision and student contributions. This repository contains code and technical documentation only.

## Citation

The maintainer requests scholarly citation and retention of contributor credit. MIT is the intended future software licence after rights clearance; currently confirmed student-code permission covers public publication only. See [LICENSE_STATUS.md](LICENSE_STATUS.md#maintainer-clarification--12-october-2026). No open-source reuse licence is in force for the repository as a whole.

For authorised use in research, publications or presentations, please cite both contributors:

Kaheh, Z., & Sun, R. (2026). *Classification Validation, Calibration and Batch Scoring* [Computer software]. GitHub. https://github.com/DR-ZohrehKaheh/classification-validation

Machine-readable metadata: [CITATION.cff](CITATION.cff). The current citation identifies code snapshot [`0211200`](https://github.com/DR-ZohrehKaheh/classification-validation/tree/02112008383b253bd7c2f02d914b7d9ef55b3144); cite the exact commit or released version you used. This citation credits the documented collaboration and does not assign copyright or constitute student approval of a licence.

No DOI has been registered by this review. See the [DOI readiness note](LICENSE_STATUS.md#doi-readiness) before archiving a release. No unverified publication title, article DOI or affiliation has been added.
