"""
Build the datasets from the Spotify Tracks dataset (Hugging Face).

Outputs:
- data/eval/eval.csv  : fixed held-out set, never used for training
- data/v1/train.csv   : first batch of data (baseline)
- data/v2/train.csv   : first + second batch, cleaned

Run: python -m ml.prepare_data
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml import config

ID = config.ID_COLUMN
LABEL = config.LABEL


def load_data():
    print("Downloading dataset...")
    df = pd.read_csv(config.HF_DATASET_URL)

    # check that all our genres exist in the dataset
    missing = [g for g in config.GENRES if g not in df[LABEL].unique()]
    if missing:
        print("Available genres:", sorted(df[LABEL].unique()))
        raise SystemExit(f"Genres not found: {missing}")

    df = df[df[LABEL].isin(config.GENRES)]
    return df[[ID] + config.FEATURES + [LABEL]]


def remove_invalid_rows(df):
    # same rules as the data contract: no missing values, values in range
    df = df.dropna()
    for col, (low, high) in config.FEATURE_RANGES.items():
        df = df[df[col].between(low, high)]
    return df


def main():
    rng = np.random.default_rng(config.SEED)

    raw = load_data()
    df = remove_invalid_rows(raw)
    print(f"{len(raw)} rows for our genres, {len(df)} after removing invalid rows")

    # some tracks appear under several genres -> label is ambiguous
    nb_genres = df.groupby(ID)[LABEL].nunique()
    clean_ids = set(nb_genres[nb_genres == 1].index)

    # 1. held-out eval set: 20% of the (non ambiguous) tracks of each genre
    eval_ids = set()
    for genre in config.GENRES:
        genre_ids = df[(df[LABEL] == genre) & df[ID].isin(clean_ids)][ID].unique()
        n = int(len(genre_ids) * config.EVAL_FRACTION)
        eval_ids.update(rng.choice(genre_ids, size=n, replace=False))
    eval_df = df[df[ID].isin(eval_ids)].drop_duplicates(subset=[ID])

    # 2. everything else, without any eval track
    rest = df[~df[ID].isin(eval_ids)]

    # 3. split the rest in 2 batches to simulate new data arriving later
    rest_ids = np.array(sorted(rest[ID].unique()))
    rng.shuffle(rest_ids)
    batch1_ids = set(rest_ids[: len(rest_ids) // 2])
    batch1 = rest[rest[ID].isin(batch1_ids)]
    batch2 = rest[~rest[ID].isin(batch1_ids)]

    # v1 = first batch as it is
    v1 = batch1

    # v2 = both batches + cleaning (duplicates and conflicting labels)
    v2 = pd.concat([batch1, batch2])
    size_before = len(v2)
    v2 = v2.drop_duplicates(subset=[ID, LABEL])
    nb_duplicates = size_before - len(v2)

    size_before = len(v2)
    v2 = v2[v2[ID].isin(clean_ids)]
    nb_conflicts = size_before - len(v2)

    # make sure no eval track ended up in a training set
    assert len(set(v1[ID]) & set(eval_df[ID])) == 0, "leak between v1 and eval!"
    assert len(set(v2[ID]) & set(eval_df[ID])) == 0, "leak between v2 and eval!"

    # save (shuffled so rows are not sorted by genre)
    files = {
        "data/eval/eval.csv": eval_df,
        "data/v1/train.csv": v1,
        "data/v2/train.csv": v2,
    }
    for path, data in files.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data.sample(frac=1, random_state=config.SEED).to_csv(path, index=False)
        print(f"saved {path} ({len(data)} rows)")

    counts = pd.DataFrame({
        "eval": eval_df[LABEL].value_counts(),
        "v1": v1[LABEL].value_counts(),
        "v2": v2[LABEL].value_counts(),
    }).fillna(0).astype(int)
    print("\nRows per genre:")
    print(counts)
    print(f"\nv2 cleaning: removed {nb_duplicates} duplicates "
          f"and {nb_conflicts} rows with conflicting labels")

    # keep some stats for ml/README_data.md
    stats = {
        "rows_selected_genres": len(raw),
        "rows_after_filter": len(df),
        "eval_rows": len(eval_df),
        "v1_rows": len(v1),
        "v2_rows": len(v2),
        "v2_removed_duplicates": nb_duplicates,
        "v2_removed_conflicting_labels": nb_conflicts,
        "rows_per_genre": counts.to_dict(),
        "seed": config.SEED,
    }
    with open("ml/data_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("stats saved in ml/data_stats.json")


if __name__ == "__main__":
    main()
