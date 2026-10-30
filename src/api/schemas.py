"""Esquemas Pydantic de la API: contrato de entrada y salida."""
from pydantic import BaseModel, ConfigDict, Field


class PredictionInput(BaseModel):
    """10 variables del dataset Diabetes (ya estandarizadas por scikit-learn)."""

    model_config = ConfigDict(extra="forbid")

    age: float = Field(ge=-1, le=1, allow_inf_nan=False, examples=[0.038])
    sex: float = Field(ge=-1, le=1, allow_inf_nan=False, examples=[0.051])
    bmi: float = Field(ge=-1, le=1, allow_inf_nan=False, examples=[0.062])
    bp: float = Field(ge=-1, le=1, allow_inf_nan=False, examples=[0.022])
    s1: float = Field(ge=-1, le=1, allow_inf_nan=False, examples=[-0.044])
    s2: float = Field(ge=-1, le=1, allow_inf_nan=False, examples=[-0.035])
    s3: float = Field(ge=-1, le=1, allow_inf_nan=False, examples=[-0.043])
    s4: float = Field(ge=-1, le=1, allow_inf_nan=False, examples=[-0.003])
    s5: float = Field(ge=-1, le=1, allow_inf_nan=False, examples=[0.020])
    s6: float = Field(ge=-1, le=1, allow_inf_nan=False, examples=[-0.018])


FEATURES = list(PredictionInput.model_fields)


class PredictionOutput(BaseModel):
    prediction: float
    model_name: str
    model_version: str


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    alias: str
    run_id: str
    metadata: dict[str, str]
