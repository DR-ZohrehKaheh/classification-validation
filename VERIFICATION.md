# Execution verification

**Project supervisor: Dr. Zohreh Kaheh.**

The released `run_pipeline.py` completed on the full supplied CSV, including utility tests, data cleaning, validation-design comparisons, classifier fitting, calibration, saved-model scoring and output-integrity checks. The wrapper checked scoring row counts, finite probabilities, probability bounds and binary decisions.

Execution receipt: private log `validation_final.log`, SHA-256 `ffd4a8a81610136d79c8953be627f5ba430b617273b169e90ed5154d62a6c522`. The receipt log itself is not distributed because it includes research outputs.

Verification covers the documented execution path and supplied inputs; it is not certification of research findings or all possible inputs.

## Environment

Python 3.12.14 on Linux. Verified Python dependencies are pinned in `requirements.txt`: `joblib==1.5.3`, `matplotlib==3.10.8`, `numpy==2.3.5`, `pandas==2.2.3`, `scikit-learn==1.8.0`, `scipy==1.17.0`.

## Version and archive

The execution record above applies to the documented software and inputs. The archived DOI version, source commit and checksums are recorded in [ZENODO_DEPOSIT.md](ZENODO_DEPOSIT.md). Later editorial updates do not rerun or extend the scientific validation.
