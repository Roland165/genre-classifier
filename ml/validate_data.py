"""
Check a dataset version against our data contract before training.
Exits with code 1 if something is wrong (so the CT workflow stops).

Run: python -m ml.validate_data --version v2
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from ml import config

EVAL_PATH = "data/eval/eval.csv"


def check_df(df, name, min_per_class):
    errors = []
    expected_cols = [config.ID_COLUMN] + config.FEATURES + [config.LABEL]

    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        return [f"{name}: missing columns {missing_cols}"]

    if df[expected_cols].isna().any().any():
        errors.append(f"{name}: contains missing values")

    for col, (low, high) in config.FEATURE_RANGES.items():
        bad = (~df[col].between(low, high)).sum()
        if bad > 0:
            errors.append(f"{name}: {bad} values of '{col}' outside [{low}, {high}]")

    bad_labels = set(df[config.LABEL].unique()) - set(config.GENRES)
    if bad_labels:
        errors.append(f"{name}: unknown labels {bad_labels}")

    counts = df[config.LABEL].value_counts()
    for genre in config.GENRES:
        if counts.get(genre, 0) < min_per_class:
            errors.append(f"{name}: only {counts.get(genre, 0)} rows for '{genre}' "
                          f"(min {min_per_class})")
    return errors


def validate(version):
    train_path = f"data/{version}/train.csv"
    for path in [train_path, EVAL_PATH]:
        if not Path(path).exists():
            return [f"file not found: {path} (did you run dvc pull?)"]

    train = pd.read_csv(train_path)
    eval_df = pd.read_csv(EVAL_PATH)

    errors = []
    errors += check_df(train, "train", config.MIN_TRAIN_PER_CLASS)
    errors += check_df(eval_df, "eval", config.MIN_EVAL_PER_CLASS)

    # train / eval separation
    if config.ID_COLUMN in train.columns and config.ID_COLUMN in eval_df.columns:
        overlap = set(train[config.ID_COLUMN]) & set(eval_df[config.ID_COLUMN])
        if overlap:
            errors.append(f"{len(overlap)} tracks are both in train and eval")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, choices=config.DATASET_VERSIONS)
    args = parser.parse_args()

    errors = validate(args.version)
    if errors:
        print(f"Data validation FAILED for {args.version}:")
        for e in errors:
            print(" -", e)
        sys.exit(1)
    print(f"Data validation OK for {args.version}")


if __name__ == "__main__":
    main()
