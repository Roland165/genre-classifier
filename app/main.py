"""API FastAPI : /, /predict, /model-info, /health, /ready, /admin/reload."""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from app.model_loader import ModelHolder
from app.schemas import PredictRequest, PredictResponse

logging.basicConfig(level=logging.INFO)  # logs sur stdout (12-factor XI)

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    holder = ModelHolder()
    holder.try_load()            # ne plante pas si le registry est injoignable
    holder.start_retry_thread()  # nouvelle tentative toutes les 30 s tant que pas prêt
    app.state.holder = holder
    yield


app = FastAPI(title="Genre classifier", lifespan=lifespan)


def _holder(request: Request) -> ModelHolder:
    return request.app.state.holder


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


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
    """Plan B pour le CT : relit l'alias champion sans redéployer l'image."""
    expected = os.getenv("ADMIN_TOKEN")
    if not expected or x_admin_token != expected:
        raise HTTPException(status_code=403, detail="Forbidden")
    h = _holder(request)
    if not h.try_load():
        raise HTTPException(status_code=503, detail=f"Reload failed: {h.last_error}")
    return {"status": "reloaded", "model_version": h.version}