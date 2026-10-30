"""API FastAPI que sirve el modelo champion del MLflow Model Registry."""
import logging
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient

from src.api.schemas import FEATURES, ModelInfo, PredictionInput, PredictionOutput
from src.registry.register import CHAMPION_ALIAS, MODEL_NAME
from src.training.train import TRACKING_URI

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def load_champion(app: FastAPI) -> None:
    """Carga el modelo champion y guarda modelo + metadatos en app.state."""
    mlflow.set_tracking_uri(TRACKING_URI)
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    app.state.model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    app.state.info = ModelInfo(
        model_name=MODEL_NAME,
        model_version=str(version.version),
        alias=CHAMPION_ALIAS,
        run_id=version.run_id,
        metadata=dict(version.tags),
    )
    logger.info("Modelo '%s' v%s cargado (@%s)", MODEL_NAME, version.version, CHAMPION_ALIAS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Se ejecuta una vez al arrancar: carga el modelo en memoria."""
    app.state.model = None
    app.state.info = None
    try:
        load_champion(app)
    except Exception:
        # La API arranca igualmente: /health responde y /predict devolvera 503
        logger.exception("No se pudo cargar el modelo champion")
    yield


app = FastAPI(
    title="Diabetes Regressor API",
    description="Sirve el modelo champion del Model Registry de MLflow.",
    lifespan=lifespan,
)


def require_model():
    """Devuelve el modelo cargado o responde 503 si no esta disponible."""
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible: no hay champion cargado")
    return app.state.model


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    require_model()
    return app.state.info


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput) -> PredictionOutput:
    model = require_model()
    try:
        row = pd.DataFrame([payload.model_dump()], columns=FEATURES)
        prediction = float(model.predict(row)[0])
    except Exception as exc:
        logger.exception("Fallo al predecir")
        raise HTTPException(status_code=500, detail="Error interno al generar la prediccion") from exc
    info: ModelInfo = app.state.info
    return PredictionOutput(prediction=prediction, model_name=info.model_name, model_version=info.model_version)
