"""Entrenamiento: entrena modelos y registra parametros, metricas y artefactos en MLflow."""
import argparse
import logging
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from src.data.cleaning import PROCESSED_PATH

logger = logging.getLogger(__name__)

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "diabetes-regression"
MODEL_NAMES = ["linear_regression", "random_forest"]


def build_model(name: str, args: argparse.Namespace):
    """Devuelve (modelo, hiperparametros) para el nombre indicado."""
    if name == "linear_regression":
        return LinearRegression(), {}
    if name == "random_forest":
        params = {"n_estimators": args.n_estimators, "max_depth": args.max_depth, "random_state": args.seed}
        return RandomForestRegressor(**params), params
    raise ValueError(f"Modelo desconocido: {name}")


def train_and_log(name, args, X_train, X_test, y_train, y_test) -> dict:
    model, params = build_model(name, args)
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model_type", name)
        mlflow.log_params(params)
        mlflow.log_params({"test_size": args.test_size, "seed": args.seed, "n_train": len(X_train)})

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "rmse": root_mean_squared_error(y_test, preds),
            "mae": mean_absolute_error(y_test, preds),
            "r2": r2_score(y_test, preds),
        }
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle",
            signature=infer_signature(X_train, model.predict(X_train)),
            input_example=X_train.head(3),
        )
        logger.info("%s -> RMSE=%.2f MAE=%.2f R2=%.3f", name, metrics["rmse"], metrics["mae"], metrics["r2"])
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena modelos de regresion y los registra en MLflow")
    parser.add_argument("--model", choices=MODEL_NAMES + ["all"], default="all")
    parser.add_argument("--data-path", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_path.exists():
        raise FileNotFoundError(f"No existe {args.data_path}. Ejecuta antes la extraccion, validacion y limpieza.")

    df = pd.read_csv(args.data_path)
    X, y = df.drop(columns=["target"]), df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    names = MODEL_NAMES if args.model == "all" else [args.model]
    for name in names:
        train_and_log(name, args, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Entrenamiento fallido: %s", exc)
        sys.exit(1)
