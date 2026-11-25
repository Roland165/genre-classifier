
# ---- data ----
HF_DATASET_URL = "hf://datasets/maharshipandya/spotify-tracks-dataset/dataset.csv"

GENRES = ["classical", "country", "hip-hop", "jazz", "metal", "pop", "reggae", "techno"]

LABEL = "track_genre"
ID_COLUMN = "track_id"

# feature -> (min, max), used for data validation and for the API input checks
FEATURE_RANGES = {
    "danceability": (0.0, 1.0),
    "energy": (0.0, 1.0),
    "loudness": (-60.0, 5.0),
    "speechiness": (0.0, 1.0),
    "acousticness": (0.0, 1.0),
    "instrumentalness": (0.0, 1.0),
    "liveness": (0.0, 1.0),
    "valence": (0.0, 1.0),
    "tempo": (30.0, 250.0),
}
FEATURES = list(FEATURE_RANGES.keys())

DATASET_VERSIONS = ["v1", "v2"]
EVAL_FRACTION = 0.2
MIN_TRAIN_PER_CLASS = 50
MIN_EVAL_PER_CLASS = 20
SEED = 42

# ---- model / registry ----
MODEL_NAME = "genre-classifier"
CHAMPION_ALIAS = "champion"
PREVIOUS_ALIAS = "previous"
EXPERIMENT_NAME = "genre-classifier"

BASELINE_MODEL = "logreg"   # logistic regression
CANDIDATE_MODEL = "rf"      # random forest

# ---- quality gates (fixed before any evaluation) ----
PRIMARY_METRIC = "macro_f1"   # higher is better
QUALITY_FLOOR = 0.45          # never promote a model below this
MIN_IMPROVEMENT = 0.01        # candidate must beat the champion by at least this

# example input (used by /ready, smoke tests and the web page)
SAMPLE_INPUT = {
    "danceability": 0.62,
    "energy": 0.81,
    "loudness": -5.2,
    "speechiness": 0.05,
    "acousticness": 0.10,
    "instrumentalness": 0.0,
    "liveness": 0.12,
    "valence": 0.55,
    "tempo": 124.0,
}
