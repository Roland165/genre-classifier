# Déploiement

## Application des douze facteurs (twelve-factor)

Cette section décrit comment l'application web (FastAPI + modèle MLflow) applique chaque facteur.
Les facteurs qui ont peu de sens pour notre architecture sont signalés comme tels.

### Variables d'environnement de l'application

| Variable | Rôle | Valeur par défaut |
| --- | --- | --- |
| `MLFLOW_TRACKING_URI` | URL du serveur MLflow / registry (DagsHub) | aucune (obligatoire en production) |
| `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | Identifiants du registry | aucune (secrets) |
| `MODEL_NAME` | Nom du modèle enregistré | `genre-classifier` |
| `MODEL_ALIAS` | Alias à résoudre en version concrète | `champion` |
| `PORT` | Port d'écoute d'uvicorn | `8000` |
| `ADMIN_TOKEN` | Secret qui protège `POST /admin/reload` (route désactivée s'il est absent) | aucune (secret) |
| `GIT_SHA` | SHA du commit, injecté au build, affiché par `/model-info` | `unknown` |
| `MODEL_URI`, `MODEL_VERSION` | **CI et tests uniquement** : chemin d'un modèle local, qui remplace le registry | non définies |

> `MODEL_URI` ne doit **jamais** être définie en production : si elle l'est, l'application ignore le registry
> et ne sert plus le modèle approuvé par l'alias `champion`.

### Les douze facteurs

| Facteur | Comment le projet l'applique |
| --- | --- |
| **I. Codebase** | Un seul dépôt GitHub contient le code, les tests, la documentation et les workflows. Le même code est déployé en CI, en staging et en production, à travers les branches `dev`, `staging` et `main`. |
| **II. Dépendances** | Toutes les dépendances sont déclarées dans `requirements.txt` (versions épinglées) et installées dans l'image Docker. `requirements-dev.txt` (tests, lint) n'est jamais copié dans l'image de production. Rien ne dépend de paquets installés sur la machine hôte. |
| **III. Configuration** | Aucune configuration n'est écrite dans le code ou dans l'image : le registry, le nom et l'alias du modèle, le port, le token d'administration et le SHA arrivent par variables d'environnement (tableau ci-dessus). Les secrets sont stockés dans GitHub Actions et Render, jamais dans Git (le `.dockerignore` exclut `.env`). |
| **IV. Services externes** | Le registry MLflow (DagsHub) est une ressource attachée par URL et identifiants : on change de registry en modifiant `MLFLOW_TRACKING_URI`, sans toucher au code. Azure Blob Storage (données DVC) n'est utilisé que par le workflow d'entraînement, pas par l'application en ligne. |
| **V. Build, release, run** | *Build* : la CI construit l'image Docker et la tague avec le SHA du commit. *Release* : le déploiement associe cette image à la configuration du service (variables d'environnement Render). *Run* : le conteneur démarre avec `uvicorn`. Une version déployée se reconstitue à partir du SHA, et `/model-info` affiche `git_sha`. |
| **VI. Processus** | L'application ne garde aucun état persistant : pas de base de données ni de fichier local. Le modèle est chargé en mémoire au démarrage (`ModelHolder`) et se recharge à la demande via `/admin/reload`. Les données durables vivent dans MLflow et DVC. |
| **VII. Association de ports** | L'application est autonome : uvicorn embarque le serveur HTTP et écoute sur `0.0.0.0:$PORT`. Render fournit `PORT` et expose le service en HTTPS. |
| **VIII. Concurrence** | *Pertinence limitée.* Le service tourne en une seule instance et un seul processus uvicorn, ce qui suffit pour une démonstration. La montée en charge se ferait en ajoutant des instances (l'application étant sans état) ou des workers uvicorn ; chaque instance chargerait alors son propre modèle en mémoire. |
| **IX. Jetable** | Le démarrage est rapide et ne plante pas si le registry est injoignable : `/health` répond dès que le processus vit, tandis que `/ready` ne passe à 200 qu'après le chargement du modèle **et** une prédiction de contrôle. Tant que le modèle n'est pas prêt, `/predict` répond 503 et un thread relance le chargement toutes les 30 s. Le conteneur lance `exec uvicorn`, qui reçoit `SIGTERM` et s'arrête proprement. |
| **X. Parité dev/prod** | La même image Docker sert en CI (test de fumée avec un modèle de test), en staging et en production. Seule la configuration diffère (modèle local via `MODEL_URI` en CI, registry en production). Les tests utilisent un vrai registry MLflow SQLite temporaire, sans accès réseau. |
| **XI. Logs** | L'application écrit ses logs sur stdout (`logging.basicConfig`) et ne gère aucun fichier de log. Render collecte ce flux. Les chargements de modèle (version chargée) et leurs échecs y sont journalisés. |
| **XII. Processus d'administration** | L'entraînement, la validation des données, la promotion et le rollback sont des jobs ponctuels exécutés par GitHub Actions avec le même code que l'application, et non des routes de l'application. Seule exception, volontairement limitée : `POST /admin/reload` (protégée par `ADMIN_TOKEN`) relit l'alias `champion` sans redéployer l'image. |

## CI/CD et déploiement

### Environnements et branches

| Branche | Workflow | Ce qui tourne | Résultat |
| --- | --- | --- | --- |
| PR vers `dev`, `staging`, `main` | `pr.yml` | `lint` (ruff), `test` (pytest, sans réseau), `docker-smoke` | Checks obligatoires pour fusionner (ruleset) |
| `staging` (merge d'une PR `dev → staging`) | `staging.yml` | Build + push de l'image, conteneur lancé dans le runner avec le **vrai registry**, smoke test | Release candidate validée |
| `main` (merge d'une PR `staging → main`) | `deploy.yml` | Build + push de l'image, déploiement Render, smoke test **sur l'URL de prod** | Production mise à jour |
| — (manuel) | `ct.yml` | Entraînement continu et promotion éventuelle | Nouveau champion servi, ou champion conservé |
| — (manuel) | `rollback.yml` | Retour à une version approuvée | Ancienne version servie |

`dev`, `staging` et `main` sont protégées par un ruleset : PR obligatoire avec 1 approbation,
checks `lint`, `test` et `docker-smoke` verts, force-push et suppression interdits.
Aucun push direct n'est possible sur ces branches.

### Identification des images

- Chaque image est poussée sur GHCR sous la forme `ghcr.io/<owner>/genre-classifier:<sha>`,
  où `<sha>` est le SHA complet du commit construit (nom en minuscules, exigé par GHCR).
- Le SHA est aussi injecté dans l'image au build (`--build-arg GIT_SHA=<sha>`) et renvoyé par
  `GET /model-info` (`git_sha`) : on sait toujours quel code tourne en production.
- La **version du modèle** n'est pas dans l'image : l'application résout l'alias `champion` au
  démarrage. Changer de modèle ne demande donc pas de rebuild (voir CT et rollback).
- Le smoke test de `deploy.yml` attend que la prod renvoie le **nouveau** SHA : il ne peut pas
  valider par erreur l'ancienne version encore en ligne pendant le redéploiement.

### Configuration et secrets à l'exécution

Aucun secret n'est dans Git ni dans l'image : ils sont injectés à l'exécution.

| Nom | Où | Utilisé par |
| --- | --- | --- |
| `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | Secrets GitHub + env Render | App, `staging.yml`, `deploy.yml`, `ct.yml`, `rollback.yml` |
| `AZURE_STORAGE_CONNECTION_STRING` | Secret GitHub | `ct.yml` (`dvc pull`, écrit dans `.dvc/config.local` du runner) |
| `RENDER_DEPLOY_HOOK_URL` | Secret GitHub | `deploy.yml`, et `ct.yml` / `rollback.yml` si `ADMIN_TOKEN` est absent |
| `PROD_URL` | Secret GitHub | Smoke tests live |
| `ADMIN_TOKEN` | Secret GitHub + env Render | `POST /admin/reload` (rafraîchissement du modèle après CT / rollback) |
| `GITHUB_TOKEN` | Fourni par Actions | Push des images sur GHCR (`packages: write`) |
| `MODEL_NAME`, `MODEL_ALIAS` | Env Render | App (`genre-classifier`, `champion`) |
| `PORT` | Fourni par Render | App |

Côté Render, le service est de type *Existing image* (package GHCR public), région Frankfurt,
avec **Health Check Path = `/ready`**.

### Le smoke test (`scripts/smoke_test.py`)

Utilisé par `pr.yml`, `staging.yml`, `deploy.yml`, `ct.yml` et `rollback.yml` :

1. attend `GET /ready` = 200 (le modèle approuvé est chargé) ;
2. attend que `GET /model-info` renvoie la version attendue (et le SHA attendu, si fourni) ;
3. envoie un vrai `POST /predict` et vérifie : HTTP 200, 3 genres connus, probabilités entre 0
   et 1 triées, `model_version` attendue.

Les erreurs de connexion et les 503 sont réessayés jusqu'au `--timeout` (900 s en prod, à cause du
réveil de l'offre gratuite de Render). Code de sortie 1 au premier échec.

### Reproduire un déploiement

- **Redéployer le code d'un commit de `main` :** Actions → `deploy-prod` → ouvrir le run de ce
  commit → *Re-run all jobs*. L'image est reconstruite avec le même SHA, redéployée et vérifiée.
  (*Run workflow* fait la même chose pour le dernier commit de `main`.)
- **Redéployer une image existante sans rebuild :**
  ```bash
  IMAGE="ghcr.io/<owner>/genre-classifier:<sha>"
  ENC=$(python3 -c "import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=''))" "$IMAGE")
  curl -fsS -X POST "$RENDER_DEPLOY_HOOK_URL&imgURL=$ENC"
  python scripts/smoke_test.py --url "$PROD_URL" --expected-version <version> --expected-sha <sha> --timeout 900
  ```
- **Tester une image en local :** `docker run --rm -p 8000:8000 --env-file .env ghcr.io/<owner>/genre-classifier:<sha>`.

### Entraînement continu (`ct.yml`)

Lancement : Actions → `continuous-training` → *Run workflow*, avec `dataset_version` (`v1`, `v2`),
`model_type` (`rf`, `logreg`) et `force_retrain`.

1. **Récupérer la release de données :** `dvc pull data/<version>.dvc data/eval.dvc` depuis Azure Blob.
2. **Valider les données** (`ml.validate_data`) : un échec arrête le workflow **avant** toute
   modification du registry ou de la prod.
3. **Évaluer, décider, entraîner, comparer, promouvoir** (`ml.ct_pipeline`) : décision de
   réentraînement (nouvelle version de données, champion sous le seuil, ou forcé), entraînement du
   candidat, comparaison sur le même jeu d'évaluation, promotion seulement si macro-F1 ≥ 0,45 **et**
   amélioration ≥ 0,01. Toute la décision est loggée dans une run MLflow `ct-<version>`.
4. **Rafraîchir la prod** si un nouveau champion a été promu (`scripts/refresh_prod.sh`).
5. **Vérifier la version servie** par un smoke test live, **même si le candidat est refusé** :
   cela prouve que la prod sert toujours l'ancien champion.

`ct.yml` et `rollback.yml` partagent un verrou (`concurrency: model-registry`) : ils ne peuvent pas
déplacer l'alias `champion` en même temps.

### Rafraîchissement de la prod après un changement de modèle

`scripts/refresh_prod.sh` fait relire l'alias `champion` à l'application :

- si `ADMIN_TOKEN` est défini : `POST /admin/reload` (instantané, même conteneur). Si le nouveau
  modèle ne se charge pas, l'application **garde l'ancien** et répond 503 à l'appel ;
- sinon : deploy hook Render avec l'image **déjà en ligne** (SHA lu dans `/model-info`), donc seul
  le modèle change, pas le code.

### Rollback (`rollback.yml`)

Lancement : Actions → `rollback` → *Run workflow*, avec `model_version` = numéro de version du registry.

1. `ml.rollback` vérifie que la version existe et porte le tag `approved=true` (sinon : refus,
   rien ne change), puis place `champion` sur cette version et `previous` sur l'ancien champion ;
2. rafraîchissement de la prod (`scripts/refresh_prod.sh`) ;
3. smoke test live avec la version cible et une vraie prédiction.

### En cas d'échec de readiness

- **Nouveau déploiement Render :** le health check `/ready` ne passe à 200 qu'une fois le modèle
  chargé et testé. Tant qu'il échoue, Render **garde l'ancienne instance** en service et finit par
  marquer le déploiement en échec : la prod n'est pas coupée. On corrige (configuration, registry)
  puis on relance `deploy-prod`, ou on lance `rollback` vers la version `previous`.
- **Après un CT ou un rollback :** si le modèle ciblé ne se charge pas, `/admin/reload` échoue,
  l'ancien modèle reste servi et le smoke test du workflow échoue (version inattendue) : le workflow
  passe au rouge. On relance alors `rollback` vers la version `previous` (dernier champion connu).
