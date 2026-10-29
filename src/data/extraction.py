"""Extracción: descarga el dataset y lo guarda tal cual en data/raw/."""
import logging
from pathlib import Path

from sklearn.datasets import load_diabetes

logger = logging.getLogger(__name__)

RAW_PATH = Path("data/raw/diabetes_raw.csv")


def extract_data(output_path: Path = RAW_PATH) -> Path:
    df = load_diabetes(as_frame=True).frame  # 10 features + columna 'target'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Datos extraidos: %d filas, %d columnas -> %s", *df.shape, output_path)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    extract_data()
