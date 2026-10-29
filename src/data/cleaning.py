"""Limpieza: duplicados, nulos, tipos y poda por percentil, con control de pérdida de datos."""
import logging
import sys
from pathlib import Path

import pandas as pd

from src.data.extraction import RAW_PATH

logger = logging.getLogger(__name__)

PROCESSED_PATH = Path("data/processed/diabetes_clean.csv")
TARGET_PERCENTILE = 0.99   # se eliminan los valores del target por encima del percentil 99
MAX_DATA_LOSS = 0.10       # si se pierde mas del 10% de filas, abortamos


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    out = df.dropna().drop_duplicates().copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="raise")

    threshold = out["target"].quantile(TARGET_PERCENTILE)
    out = out[out["target"] <= threshold]

    loss = 1 - len(out) / n_before
    logger.info("Limpieza: %d -> %d filas (perdida %.1f%%)", n_before, len(out), loss * 100)
    if loss > MAX_DATA_LOSS:
        raise ValueError(f"Se perdio el {loss:.1%} de los datos (maximo {MAX_DATA_LOSS:.0%})")
    return out.reset_index(drop=True)


def run_cleaning(input_path: Path = RAW_PATH, output_path: Path = PROCESSED_PATH) -> Path:
    cleaned = clean_data(pd.read_csv(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    logger.info("Datos limpios guardados en %s", output_path)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    try:
        run_cleaning()
    except ValueError as exc:
        logger.error("Limpieza fallida: %s", exc)
        sys.exit(1)
