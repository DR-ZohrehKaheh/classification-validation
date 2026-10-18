# Classification Validation, Calibration and Batch Scoring

[Run and inputs](#run) · [Scope](#scope) · [Citation](#citation) · [Contributions](ATTRIBUTION.md) · [Reuse status](LICENSE_STATUS.md)

A Python workflow for preparing environmental classification data, comparing validation designs, calibrating model scores and scoring new observations with saved models. The project demonstrates practical model evaluation and reproducible batch inference.

**Availability:** Public source code for technical review and citation. See [publication and reuse terms](LICENSE_STATUS.md).

**Project design, supervision and subsequent development: Dr. Zohreh Kaheh.**  
**Initial implementation (undergraduate collaborator): Ruihan Sun.**

The project builds on Dr Kaheh’s earlier research and supervised student collaboration. See [ATTRIBUTION.md](ATTRIBUTION.md) for the contribution details.

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

Please cite the archived software and credit both contributors:

Kaheh, Z., & Sun, R. (2026). *Classification Validation, Calibration and Batch Scoring* (Version snapshot-4bb79b1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23312435

Download [CITATION.cff](CITATION.cff) or [BibTeX](CITATION.bib). The [archived version](https://zenodo.org/records/23312435) is `snapshot-4bb79b1`, from source commit [`4bb79b1`](https://github.com/DR-ZohrehKaheh/classification-validation/tree/4bb79b1124508c87848dcefc192387d55ce9acb6). Cite the version or commit you used; later GitHub documentation may differ from the fixed archive. See [ZENODO_DEPOSIT.md](ZENODO_DEPOSIT.md) for the deposit record and checksums.

For reuse permissions, see [LICENSE_STATUS.md](LICENSE_STATUS.md).

## Questions and corrections

For technical questions or documentation corrections, [open an issue](https://github.com/DR-ZohrehKaheh/classification-validation/issues) with the version or commit, environment, command and a minimal error report. Do not include private datasets, credentials or unpublished results. Discuss the scope and permissions of proposed code contributions with the maintainer first; [CONTRIBUTOR_RIGHTS.md](CONTRIBUTOR_RIGHTS.md) describes the required record.

## Continuous checks

[GitHub Actions](https://github.com/DR-ZohrehKaheh/classification-validation/actions/workflows/checks.yml) runs the existing dataset-independent checks on Python 3.12 and Ubuntu. This covers utility tests or synthetic native/model checks, not the complete data-dependent research pipeline or scientific conclusions. The workflow does not publish generated research outputs.

