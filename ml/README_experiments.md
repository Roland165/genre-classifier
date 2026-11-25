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
| 1 | `logreg-v1` | v1 | 0.5782 | 0.5995 | approved, first champion (baseline, trained and promoted by hand for the initial release) |
| 2 | `rf-v2` (+ `ct-v2`) | v2 | 0.7395 | 0.7485 | local test of the CT pipeline: promoted (+0.16 vs champion), then rolled back to v1 to test the rollback |
| 3 | `rf-v2` (+ `ct-v2`) | v2 | 0.7395 | 0.7485 | **approved, current champion**, promoted by the CT workflow |
| 4 | `logreg-v2` (+ `ct-v2`) | v2 | 0.5777 | see run | rejected (`approved=false`): not enough improvement vs champion 0.7395 |

### CT and rollback workflow runs (GitHub Actions)

| Run | Inputs | What happened | Prod after |
|---|---|---|---|
| [continuous-training #1](https://github.com/Roland165/genre-classifier/actions/runs/45633164693) | v2, rf | new dataset version (v1 -> v2) so retrain; 0.7395 vs 0.5782 -> promoted, champion 1 -> 3, prod refreshed and checked live | version 3 |
| [continuous-training #2](https://github.com/Roland165/genre-classifier/actions/runs/45718152676) | v2, logreg, force_retrain | forced retrain; 0.5777 vs 0.7395 -> rejected (needs +0.01), champion kept, no refresh, prod checked live | version 3 |
| [rollback to 1](https://github.com/Roland165/genre-classifier/actions/runs/45718648343) | model_version 1 | champion 3 -> 1, prod reloaded with `/admin/reload`, checked live | version 1 |
| [rollback to 3](https://github.com/Roland165/genre-classifier/actions/runs/45719224505) | model_version 3 | champion 1 -> 3, prod reloaded, checked live | version 3 |

Each CT run also has its own MLflow run (`ct-v2`) with the retraining and promotion decisions as tags.

## Production model choice

Production serves **version 3**: the random forest trained on v2. It is the only model that passes both gates against the baseline: 0.7395 macro F1 vs 0.5782, so +0.16 on the same held-out set (floor 0.45, minimum improvement 0.01).

Logistic regression gets about the same score on v1 (0.5782) and on v2 (0.5777), so more data does not help a linear model here: most of the gain comes from the random forest, which can learn non linear relations between the audio features. We did not train a random forest on v1, so we cannot split the gain exactly between "more data" and "better model".

Per-genre recall of the champion on the held-out set (from its confusion matrix):

| Genre | Correct / total | Recall |
|---|---|---|
| techno | 175 / 199 | 88% |
| classical | 156 / 183 | 85% |
| metal | 166 / 197 | 84% |
| jazz | 155 / 197 | 79% |
| country | 146 / 198 | 74% |
| reggae | 122 / 167 | 73% |
| hip-hop | 86 / 154 | 56% |
| pop | 89 / 168 | 53% |

The remaining errors are mostly between genres that sound close with these 9 features:
- **hip-hop, reggae and pop** are mixed up together (30 reggae tracks predicted as hip-hop, 29 hip-hop as reggae, 28 pop as reggae, 22 pop as hip-hop). They are all danceable, mid tempo and with vocals.
- **country and metal** (16 and 17 tracks each way): both often loud guitar music.
- **classical and jazz** (12 and 13 tracks each way): both acoustic and instrumental.

This is why we use macro F1 and not accuracy: pop and hip-hop pull the score down even if the other genres are good. Next steps to improve them would be more features (lyrics, artist information) or more training data for these genres.