# Data

## Source

- **Dataset:** Spotify Tracks Dataset on Hugging Face – https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset
- **File used:** `dataset.csv`
- **Revision (Hugging Face commit):** `635b034f69257814eff850a5c2b3346fe458134f`
- **License:** BSD (as stated on the dataset card)
- **Downloaded on:** 2026-11-24, with `python -m ml.prepare_data`

The dataset has ~114k tracks from 125 genres, collected with the Spotify Web API. Each row has audio features (danceability, energy, tempo, ...) and a genre label.

## What we keep

- **8 genres:** classical, country, hip-hop, jazz, metal, pop, reggae, techno (chosen because they are quite different from each other and easy to understand in the demo)
- **9 features:** danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo
- **Label:** `track_genre`
- `track_id` is only kept to avoid leaks between train and eval

Rows with missing values or values outside the data contract (see `ml/config.py`, e.g. `tempo` must be between 30 and 250) are removed. This removed 1 row (8000 -> 7999).

## Split design

All of this is done in `ml/prepare_data.py` with a fixed seed (42), so it is reproducible.

1. **Held-out eval set (`data/eval/eval.csv`):** 20% of the tracks of each genre, taken only from tracks that have a single genre (some tracks appear under several genres). Duplicates are removed.
2. All eval tracks are removed from the rest of the data, so **no model ever sees an eval track during training**. The champion and the candidate are always compared on this same eval set.
3. The remaining tracks are split in two random batches of 50% (by `track_id`) to simulate new data arriving later.

There is no time dimension in this problem, so we use a random split (no chronological split needed).

Preprocessing (standard scaling) is done inside the sklearn pipeline, so it is fitted on the training data only.

## Versions

| Version | Content | Rows |
|---|---|---|
| eval | fixed held-out set | 1463 |
| v1 | batch 1, as it is (duplicates kept) | 3262 |
| v2 | batch 1 + batch 2, cleaned | 5870 |

**What changed between v1 and v2:**
- **new labeled data:** batch 2 is added (simulates a new data delivery)
- **cleaning:** 67 duplicate rows removed, and 580 rows removed because the same track had conflicting genre labels

Rows per genre:

| Genre | eval | v1 | v2 |
|---|---|---|---|
| classical | 183 | 390 | 733 |
| country | 198 | 428 | 796 |
| hip-hop | 154 | 400 | 620 |
| jazz | 197 | 395 | 790 |
| metal | 197 | 409 | 789 |
| pop | 168 | 395 | 673 |
| reggae | 167 | 417 | 669 |
| techno | 199 | 428 | 800 |

(Full stats in `ml/data_stats.json`.)

## Versioning with DVC

Data is tracked with DVC, remote = Azure Blob Storage (container `dvc-store`). Only the `.dvc` pointer files are in Git, raw data and credentials are not.

| Data | Git tag | Git commit | DVC md5 |
|---|---|---|---|
| eval | `data-v1` | `444d7fc2736a103bcbc845e6f6cce57937c1a86e` | `19a3d355ebe724d6759b128f23c6f235.dir` |
| v1 | `data-v1` | `444d7fc2736a103bcbc845e6f6cce57937c1a86e` | `4bbba8b3ab56e472258cc351205e9d55.dir` |
| v2 | `data-v2` | `033806b8d279a4d569b9942a5135967dbd6420b9` | `a40c98bca023afce377d3cb39745d1d2.dir` |

The DVC md5 and the dataset version are also logged as tags in every MLflow training run (`dvc_md5`, `eval_dvc_md5`, `dataset_version`), so each model can be linked to the exact data it was trained on.

## How to get the data

```bash
dvc remote modify --local azure connection_string "<connection string>"
dvc pull data/eval.dvc data/v1.dvc data/v2.dvc
python -m ml.validate_data --version v2
```

To rebuild everything from scratch: `python -m ml.prepare_data` (same seed -> same files).
