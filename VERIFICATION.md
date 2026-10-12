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

DOI preparation added a BibTeX export and a Zenodo metadata/rights record on 12 October 2026. At that stage, these were preparation documents and no deposit or DOI had been created. Source code and the existing CFF citation were unchanged; documentation checksums were refreshed.

## DOI publication — 12 October 2026

Archived snapshot `4bb79b1124508c87848dcefc192387d55ce9acb6` was published as `snapshot-4bb79b1` in [Zenodo](https://zenodo.org/records/23312435), DOI `10.5281/zenodo.23312435`. The public record, both creator names, archive checksum and custom rights notice were checked. Citation files and README now identify this archived version. The original archive retains its pre-DOI citation files. Source code, dependency pins and scientific execution scope are unchanged; no training was rerun. See [ZENODO_DEPOSIT.md](ZENODO_DEPOSIT.md). Documentation checksums were refreshed.

## Portfolio presentation review — 12 October 2026

The README navigation, contribution summary, citation guidance and issue-reporting instructions were reviewed for clarity and consistency with the existing rights record. Documentation links and citation identifiers were checked. Source code, dependencies, configuration and the archived DOI version are unchanged; no training or scientific validation was rerun. `SHA256SUMS.txt` was refreshed for the current files.
