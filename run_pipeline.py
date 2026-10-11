#!/usr/bin/env python3
# Project supervisor: Dr. Zohreh Kaheh.
# Undergraduate project collaborator: Ruihan Sun.
# Portfolio execution wrapper prepared with AI assistance.
"""Run data-dependent analysis, integrity checks and scoring."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    parser.add_argument('--random-seed', type=int, default=42)
    args = parser.parse_args()
    threads = int(os.environ.get('PORTFOLIO_NUM_THREADS', '1'))
    if threads < 1:
        raise ValueError('PORTFOLIO_NUM_THREADS must be positive')
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[name] = str(threads)
    source = Path(__file__).resolve().parent
    data, out = args.data.resolve(), args.output.resolve()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(source/'tests'), '-v'], check=True)
    subprocess.run([sys.executable, str(source/'analysis_pipeline.py'), '--data', str(data),
                    '--output', str(out), '--random-seed', str(args.random_seed)], check=True)
    subprocess.run([sys.executable, str(source/'verify_outputs.py'), '--output', str(out)], check=True)
    scored = out/'scored_rows.csv'
    subprocess.run([sys.executable, str(source/'score_new_data.py'), '--input', str(data),
                    '--model-dir', str(out/'models'), '--output', str(scored)], check=True)
    scores = pd.read_csv(scored, usecols=['wildfire_probability', 'wildfire_predicted'])
    if len(scores) != sum(len(chunk) for chunk in pd.read_csv(data, chunksize=100000)):
        raise RuntimeError('Scoring row count mismatch')
    if not np.isfinite(scores.wildfire_probability).all() or not scores.wildfire_probability.between(0, 1).all():
        raise RuntimeError('Invalid scoring probabilities')
    if not scores.wildfire_predicted.isin([0, 1]).all():
        raise RuntimeError('Invalid decision values')
    print('Analysis, saved-model scoring and integrity checks passed.')


if __name__ == '__main__':
    main()
