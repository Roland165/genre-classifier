# genre-classifier

Prédiction du **genre musical** d'un morceau à partir de ses caractéristiques audio
(danceability, energy, tempo…). L'application web renvoie les **3 genres les plus probables**
et affiche la **version du modèle** chargée depuis le registry MLflow.

- **Production :** https://genre-classifier-qkrm.onrender.com
- **Modèle servi :** `genre-classifier@champion` (registry MLflow sur DagsHub)
- **Données :** [Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) (licence BSD), 8 genres, versionnées avec DVC sur Azure Blob

| Document | Contenu |
| --- | --- |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Twelve-factor, CI/CD, secrets, reproduction d'un déploiement, CT et rollback |
| [ml/README_data.md](ml/README_data.md) | Source, versions v1 / v2 / eval des données, découpage |
| [ml/README_experiments.md](ml/README_experiments.md) | Métrique, quality gates, runs et choix du champion |

## Utiliser l'application

La page d'accueil propose 9 curseurs et 3 exemples (« Piano calme », « Metal », « Rap »).
L'API est aussi utilisable directement :

```bash
curl -X POST https://genre-classifier-qkrm.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"danceability":0.62,"energy":0.81,"loudness":-5.2,"speechiness":0.05,
       "acousticness":0.10,"instrumentalness":0.0,"liveness":0.12,"valence":0.55,"tempo":124.0}'
```

```json
{"top3":[{"genre":"pop","proba":0.41},{"genre":"techno","proba":0.22},{"genre":"hip-hop","proba":0.15}],
 "model_name":"genre-classifier","model_version":"3"}
```

| Feature | Plage valide |
| --- | --- |
| `danceability`, `energy`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence` | 0 à 1 |
| `loudness` (dB) | −60 à 5 |
| `tempo` (BPM) | 30 à 250 |

Une entrée invalide (champ manquant ou en trop, mauvais type, hors plage) renvoie **422**.

| Route | Rôle |
| --- | --- |
| `GET /` | Interface web |
| `POST /predict` | Top 3 des genres + version du modèle |
| `GET /model-info` | Nom, alias, version chargée, date de chargement, `git_sha` de l'image |
| `GET /health` | 200 tant que le processus tourne (liveness) |
| `GET /ready` | 200 seulement quand le modèle approuvé est chargé et prédit, sinon 503 (readiness) |

## Lancer en local

### Avec un environnement virtuel

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# Option 1 : vrai registry (identifiants DagsHub dans un fichier .env, jamais commité)
#   MLFLOW_TRACKING_URI=https://dagshub.com/<user>/genre-classifier.mlflow
#   MLFLOW_TRACKING_USERNAME=<user>
#   MLFLOW_TRACKING_PASSWORD=<token>
set -a && source .env && set +a
uvicorn app.main:app --reload

# Option 2 : modèle de test local, sans réseau
python tests/fixtures/make_test_model.py --out test-model
MODEL_URI=test-model MODEL_VERSION=test uvicorn app.main:app --reload
```

Puis ouvrir http://localhost:8000.

### Avec Docker

```bash
docker build --build-arg GIT_SHA=$(git rev-parse HEAD) -t genre-app .
docker run --rm -p 8000:8000 --env-file .env genre-app
```

### Tests et lint

```bash
ruff check .   # mêmes règles qu'en CI (ruff.toml)
pytest -q      # sans réseau : modèle de test + registry SQLite temporaire
python scripts/smoke_test.py --url http://localhost:8000 --expected-version <version>
```

## Flux de travail Git

```
feature/<prenom>-<sujet> ──PR──▶ dev ──PR──▶ staging ──PR──▶ main
                          checks      checks + build      checks + déploiement
                                      image + smoke test  Render + smoke test live
```

- Chaque tâche part de `dev` sur une branche `feature/…` (ou `fix/…`).
- `dev`, `staging` et `main` sont protégées : PR obligatoire avec **1 approbation**, checks
  `lint`, `test` et `docker-smoke` verts, pas de force-push ni de suppression.
- Les PR de promotion (`dev → staging`, `staging → main`) sont fusionnées avec un *merge commit*
  pour garder le même historique sur les trois branches.

## Workflows GitHub Actions

| Fichier | Déclencheur | Ce qu'il fait |
| --- | --- | --- |
| [`pr.yml`](.github/workflows/pr.yml) | PR vers `dev`, `staging`, `main` | `lint` (ruff), `test` (pytest), `docker-smoke` (build de l'image + smoke test avec le modèle de test) |
| [`staging.yml`](.github/workflows/staging.yml) | push sur `staging` | Build de l'image `ghcr.io/<owner>/genre-classifier:<sha>`, push sur GHCR, lancement dans le runner avec le vrai registry, smoke test (version du champion + SHA) |
| [`deploy.yml`](.github/workflows/deploy.yml) | push sur `main` | Build + push de l'image, déploiement Render (deploy hook + `imgURL`), smoke test live : nouveau SHA, version du champion, vraie prédiction |
| [`ct.yml`](.github/workflows/ct.yml) | manuel | Entraînement continu : DVC pull + validation des données → évaluer, décider, entraîner, comparer, promouvoir → rafraîchir la prod → vérifier la version servie |
| [`rollback.yml`](.github/workflows/rollback.yml) | manuel | Remet l'alias `champion` sur une version déjà approuvée, rafraîchit la prod, vérifie la version servie |

Détails, secrets et procédures : [DEPLOYMENT.md](DEPLOYMENT.md).

## Arborescence

```
app/        application FastAPI (seul code copié dans l'image)
ml/         données, entraînement, décision, CT, promotion, rollback (jamais dans l'image)
data/       pointeurs DVC (eval.dvc, v1.dvc, v2.dvc) ; les données sont sur Azure Blob
tests/      tests pytest + fixture du modèle de test
scripts/    smoke_test.py, refresh_prod.sh (utilisés par les workflows)
.github/    workflows CI/CD/CT
```
