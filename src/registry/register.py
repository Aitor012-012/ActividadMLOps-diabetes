"""Registro y evaluacion: quality gate + Champion vs Challenger + alias @champion."""
import argparse
import logging
import sys

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.training.train import EXPERIMENT_NAME, TRACKING_URI

logger = logging.getLogger(__name__)

MODEL_NAME = "diabetes-regressor"
CHAMPION_ALIAS = "champion"
RMSE_THRESHOLD = 60.0  # predecir siempre la media da RMSE ~72; exigimos mejorarlo claramente


def get_best_run(client: MlflowClient):
    """Devuelve la ejecucion con menor RMSE del experimento."""
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"No existe el experimento '{EXPERIMENT_NAME}'. Ejecuta antes el entrenamiento.")
    runs = client.search_runs([experiment.experiment_id], order_by=["metrics.rmse ASC"], max_results=1)
    if not runs:
        raise ValueError("El experimento no tiene ejecuciones. Ejecuta antes el entrenamiento.")
    return runs[0]


def get_champion(client: MlflowClient):
    """Devuelve la version champion actual, o None si todavia no hay."""
    try:
        return client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException:
        return None


def promote_if_better(threshold: float) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    challenger = get_best_run(client)
    c_rmse = challenger.data.metrics["rmse"]
    c_type = challenger.data.params.get("model_type", "desconocido")
    logger.info("Challenger: run %s (%s) RMSE=%.2f", challenger.info.run_id[:8], c_type, c_rmse)

    # 1) Quality gate
    if c_rmse > threshold:
        raise ValueError(f"Quality gate NO superado: RMSE {c_rmse:.2f} > umbral {threshold:.2f}")
    logger.info("Quality gate superado: RMSE %.2f <= %.2f", c_rmse, threshold)

    # 2) Champion vs Challenger
    champion = get_champion(client)
    if champion is not None:
        champ_rmse = client.get_run(champion.run_id).data.metrics["rmse"]
        if champion.run_id == challenger.info.run_id or c_rmse >= champ_rmse:
            logger.info(
                "Se mantiene el champion actual (v%s, RMSE=%.2f): el challenger no lo mejora",
                champion.version, champ_rmse,
            )
            return

    # 3) Registrar la nueva version y asignar el alias
    logged_models = client.search_logged_models(
        experiment_ids=[challenger.info.experiment_id],
        filter_string=f"source_run_id = '{challenger.info.run_id}'",
    )
    if not logged_models:
        raise ValueError("No se encontro el modelo guardado de la ejecucion ganadora")
    version = mlflow.register_model(f"models:/{logged_models[0].model_id}", MODEL_NAME)
    client.set_model_version_tag(MODEL_NAME, version.version, "rmse", f"{c_rmse:.4f}")
    client.set_model_version_tag(MODEL_NAME, version.version, "model_type", c_type)
    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version.version)
    logger.info("Modelo '%s' v%s registrado y marcado como @%s", MODEL_NAME, version.version, CHAMPION_ALIAS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Registra el mejor modelo y asigna el alias champion")
    parser.add_argument("--rmse-threshold", type=float, default=RMSE_THRESHOLD)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    try:
        promote_if_better(args.rmse_threshold)
    except ValueError as exc:
        logger.error("Registro fallido: %s", exc)
        sys.exit(1)
