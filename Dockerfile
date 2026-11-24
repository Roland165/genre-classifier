FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# Dépendances d'abord : cette couche est mise en cache tant que requirements.txt ne change pas
COPY requirements.txt .
RUN pip install -r requirements.txt

# Uniquement le code de l'app (pas de ml/, data/, tests/ : voir .dockerignore)
COPY app/ app/

# GIT_SHA est injecté au build et exposé par /model-info
ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA} \
    PORT=8000

# Pas d'exécution en root
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Le port vient de l'environnement (Render fournit $PORT)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]