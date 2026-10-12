# Execution verification

**Project supervisor: Dr. Zohreh Kaheh.**

The released `run_pipeline.py` completed on the full supplied CSV, including utility tests, data cleaning, validation-design comparisons, classifier fitting, calibration, saved-model scoring and output-integrity checks. The wrapper checked scoring row counts, finite probabilities, probability bounds and binary decisions.

Execution receipt: private log `validation_final.log`, SHA-256 `ffd4a8a81610136d79c8953be627f5ba430b617273b169e90ed5154d62a6c522`. The receipt log itself is not distributed because it includes research outputs.

Verification covers the documented execution path and supplied inputs; it is not certification of research findings or all possible inputs.

## Environment

Python 3.12.14 on Linux. Verified Python dependencies are pinned in `requirements.txt`: `joblib==1.5.3`, `matplotlib==3.10.8`, `numpy==2.3.5`, `pandas==2.2.3`, `scikit-learn==1.8.0`, `scipy==1.17.0`.

## Documentation review

Repository documentation was reviewed on 11 October 2026 for consistency with the included source files and recorded execution scope. This editorial review did not change executable code or rerun training. `SHA256SUMS.txt` records the current file contents.

## Rights and citation documentation — 12 October 2026

A documentation-only review added the reuse-status notice, contributor-rights register and software citation, and clarified the contribution account. Executable sources, dependencies and configuration are unchanged from commit `02112008383b253bd7c2f02d914b7d9ef55b3144`. No training or scientific validation was rerun for this update. Existing execution receipts do not establish publication or licensing authority. `SHA256SUMS.txt` was refreshed for the current files, excluding itself.

The subsequent clarification on 12 October 2026 records publication-only permission and the maintainer's acceptance of standard notice retention with a scholarly citation request. MIT remains conditional and was not added. This follow-up changes documentation and its checksums only.
