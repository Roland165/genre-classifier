"""
Build the track list used by the "Beat the AI" game.

Takes a few tracks per genre from the held-out eval set (so the model has
never seen them during training) and adds their title and artists from the
original Hugging Face dataset. Only these few example tracks go into the app,
not the datasets.

Output: app/game_tracks.json
Run: python -m ml.make_game_tracks
"""
import json

import pandas as pd

from ml import config

TRACKS_PER_GENRE = 5
OUT_PATH = "app/game_tracks.json"


def main():
    eval_df = pd.read_csv("data/eval/eval.csv")

    print("Downloading track names...")
    raw = pd.read_csv(config.HF_DATASET_URL, usecols=[config.ID_COLUMN, "track_name", "artists"])
    raw = raw.drop_duplicates(subset=[config.ID_COLUMN])

    df = eval_df.merge(raw, on=config.ID_COLUMN, how="left").dropna(subset=["track_name", "artists"])

    picked = (
        df.groupby(config.LABEL, group_keys=False)
        .apply(lambda g: g.sample(n=min(TRACKS_PER_GENRE, len(g)), random_state=config.SEED))
    )

    tracks = []
    for _, row in picked.iterrows():
        tracks.append({
            "track_id": row[config.ID_COLUMN],
            "name": row["track_name"],
            "artists": row["artists"].replace(";", ", "),
            "genre": row[config.LABEL],
            "features": {f: round(float(row[f]), 4) for f in config.FEATURES},
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(tracks, f, indent=2, ensure_ascii=False)
    print(f"saved {len(tracks)} tracks in {OUT_PATH}")


if __name__ == "__main__":
    main()
