"""Validación: contrato de datos con Pydantic. Si algo no cumple, el pipeline se detiene."""
import logging
import sys
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.data.extraction import RAW_PATH

logger = logging.getLogger(__name__)


class DiabetesRecord(BaseModel):
    """Contrato de una fila: columnas obligatorias, tipos y rangos."""

    model_config = ConfigDict(extra="forbid")

    # Las features vienen estandarizadas, por eso su rango razonable es [-1, 1]
    age: float = Field(ge=-1, le=1, allow_inf_nan=False)
    sex: float = Field(ge=-1, le=1, allow_inf_nan=False)
    bmi: float = Field(ge=-1, le=1, allow_inf_nan=False)
    bp: float = Field(ge=-1, le=1, allow_inf_nan=False)
    s1: float = Field(ge=-1, le=1, allow_inf_nan=False)
    s2: float = Field(ge=-1, le=1, allow_inf_nan=False)
    s3: float = Field(ge=-1, le=1, allow_inf_nan=False)
    s4: float = Field(ge=-1, le=1, allow_inf_nan=False)
    s5: float = Field(ge=-1, le=1, allow_inf_nan=False)
    s6: float = Field(ge=-1, le=1, allow_inf_nan=False)
    target: float = Field(gt=0, allow_inf_nan=False)  # progresión de la enfermedad


def validate_data(df: pd.DataFrame) -> None:
    """Lanza ValueError si el DataFrame no cumple el contrato (fail-fast)."""
    if df.empty:
        raise ValueError("El dataset esta vacio")

    expected = set(DiabetesRecord.model_fields)
    missing = expected - set(df.columns)
    extra = set(df.columns) - expected
    if missing or extra:
        raise ValueError(f"Esquema invalido. Faltan: {sorted(missing)}. Sobran: {sorted(extra)}")

    errors = []
    for i, row in enumerate(df.to_dict(orient="records")):
        try:
            DiabetesRecord.model_validate(row)
        except ValidationError as exc:
            errors.append(f"fila {i}: {exc.errors()[0]['loc']} -> {exc.errors()[0]['msg']}")
    if errors:
        raise ValueError(f"{len(errors)} filas invalidas. Primeras: {errors[:3]}")

    logger.info("Validacion superada: %d filas correctas", len(df))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    try:
        validate_data(pd.read_csv(Path(RAW_PATH)))
    except ValueError as exc:
        logger.error("Validacion fallida: %s", exc)
        sys.exit(1)  # codigo de salida distinto de 0: el CI se pondra en rojo
