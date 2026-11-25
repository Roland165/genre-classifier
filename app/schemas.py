"""Contrat d'entrée/sortie de l'API (plages alignées sur ml/config.py)."""
from pydantic import BaseModel, ConfigDict, Field

# strict=True : "0.5" (chaîne) et true/false sont refusés ; allow_inf_nan=False : NaN/inf refusés
_F = {"strict": True, "allow_inf_nan": False}

class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # champ en trop -> 422

    danceability: float = Field(ge=0.0, le=1.0, **_F)
    energy: float = Field(ge=0.0, le=1.0, **_F)
    loudness: float = Field(ge=-60.0, le=5.0, **_F)
    speechiness: float = Field(ge=0.0, le=1.0, **_F)
    acousticness: float = Field(ge=0.0, le=1.0, **_F)
    instrumentalness: float = Field(ge=0.0, le=1.0, **_F)
    liveness: float = Field(ge=0.0, le=1.0, **_F)
    valence: float = Field(ge=0.0, le=1.0, **_F)
    tempo: float = Field(ge=30.0, le=250.0, **_F)


class GenreScore(BaseModel):
    genre: str
    proba: float


class PredictResponse(BaseModel):
    top3: list[GenreScore]
    model_name: str
    model_version: str
    