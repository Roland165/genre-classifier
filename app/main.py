"""API FastAPI : /, /predict, /model-info, /health, /ready, /admin/reload, /game."""
import json
import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from app.model_loader import ModelHolder
from app.schemas import PredictRequest, PredictResponse

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
GAME_TRACKS_FILE = BASE_DIR / "game_tracks.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    holder = ModelHolder()
    holder.try_load()
    holder.start_retry_thread()
    app.state.holder = holder
    yield


app = FastAPI(title="Genre classifier", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """422 sans renvoyer la valeur fautive : NaN/inf ne sont pas sérialisables en JSON (sinon 500)."""
    errors = [{k: v for k, v in e.items() if k not in ("input", "ctx")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})


def _holder(request: Request) -> ModelHolder:
    return request.app.state.holder


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/game")
def game(request: Request):
    return templates.TemplateResponse(request, "game.html")


@app.get("/game/tracks")
def game_tracks():
    """Example tracks for the game (taken from the held-out eval set by ml/make_game_tracks.py)."""
    try:
        return json.loads(GAME_TRACKS_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request):
    h = _holder(request)
    if h.ready:
        return {"status": "ready", "model_version": h.version}
    return JSONResponse(status_code=503, content={"status": "not ready", "error": h.last_error})


@app.get("/model-info")
def model_info(request: Request):
    h = _holder(request)
    return {
        "model_name": h.name,
        "model_alias": h.alias,
        "model_version": h.version,
        "loaded_at": h.loaded_at,
        "git_sha": os.getenv("GIT_SHA", "unknown"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request):
    h = _holder(request)
    if not h.ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    top3, version = h.predict_top3(payload.model_dump())
    return {"top3": top3, "model_name": h.name, "model_version": version}


@app.post("/admin/reload")
def reload_model(request: Request, x_admin_token: Annotated[str | None, Header()] = None):
    expected = os.getenv("ADMIN_TOKEN")
    if not expected or not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(status_code=403, detail="Forbidden")
    h = _holder(request)
    if not h.try_load():
        raise HTTPException(status_code=503, detail=f"Reload failed: {h.last_error}")
    return {"status": "reloaded", "model_version": h.version}