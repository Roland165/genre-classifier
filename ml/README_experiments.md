# Experiments

All runs are tracked in MLflow on DagsHub: https://dagshub.com/Roland165/genre-classifier (Experiments tab).
Models are registered as `genre-classifier`, the production model has the alias `champion`.

## Primary metric: macro F1 (higher is better)

The app shows the most likely genres for a track. We want it to work for **every** genre, not only for the easy or frequent ones. Macro F1 gives the same weight to each genre, so a model that is bad on one genre (for example pop vs hip-hop) gets a lower score even if its accuracy looks fine. Accuracy is also logged but only for information.

All models are evaluated on the same held-out eval set (`data/eval`, 1463 tracks, never used for training).

## Quality gates

Defined in `ml/config.py` **before** any model was evaluated (first commit of the file: `444d7fc`):

| Gate | Value | Meaning |
|---|---|---|
| Quality floor | macro F1 >= 0.45 | a model below this is never promoted |
| Minimum improvement | +0.01 | a candidate must beat the champion by at least 0.01 macro F1 |

Retraining rule (`ml/decide.py`), the CT pipeline retrains when:
- there is no champion yet (first release), or
- the selected dataset version is different from the one the champion was trained on, or
- the champion is below the floor on the eval set, or
- retraining is forced from the workflow input.

## Models

| Name | Model | Notes |
|---|---|---|
| `logreg` | StandardScaler + LogisticRegression (max_iter=1000) | simple baseline |
| `rf` | StandardScaler + RandomForest (300 trees, seed 42) | candidate, can learn non linear relations between features |

## Runs

Each training run logs: git commit SHA, dataset version, DVC md5 of the train and eval data, hyperparameters, macro F1 and accuracy, classification report, confusion matrix and the model. CT runs (`ct-<version>`) log the retraining and promotion decisions with the champion/candidate versions.

| Registry version | Run | Data | Macro F1 | Accuracy | Decision |
|---|---|---|---|---|---|
| 1 | `logreg-v1` | v1 | 0.5782 | 0.5995 | approved, first champion (baseline) |
| 2 | `rf-v2` (+ `ct-v2`) | v2 | 0.7395 | 0.7485 | local test of the CT pipeline: promoted (+0.16 vs champion), then rolled back to v1 to test the rollback |

_TODO: add the runs done by the GitHub CT workflow (v2 promotion, champion kept, rollback) with links._

## Production model choice

_TODO: update after the CT workflow runs._ Expected: the random forest trained on v2, because it passes the floor and improves the champion by about +0.16 macro F1 on the same eval set (0.74 vs 0.58). _TODO: look at the confusion matrix of the champion run and write which genres are still confused._
