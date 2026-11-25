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