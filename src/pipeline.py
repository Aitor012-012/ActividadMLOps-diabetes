"""Orquestacion del pipeline MLOps: extraccion -> validacion -> limpieza -> entrenamiento -> registro."""
import logging
import subprocess
import sys
import time

logger = logging.getLogger("pipeline")

# (nombre de la etapa, modulo a ejecutar, argumentos extra)
STEPS = [
    ("1/5 Extraccion", "src.data.extraction", []),
    ("2/5 Validacion", "src.data.validation", []),
    ("3/5 Limpieza", "src.data.cleaning", []),
    ("4/5 Entrenamiento", "src.training.train", ["--model", "all"]),
    ("5/5 Registro y evaluacion", "src.registry.register", []),
]


def run_step(name: str, module: str, extra_args: list[str]) -> None:
    """Ejecuta una etapa como proceso aparte; si falla, detiene todo el pipeline."""
    logger.info("=== %s ===", name)
    start = time.time()
    result = subprocess.run([sys.executable, "-m", module, *extra_args])
    if result.returncode != 0:
        logger.error("La etapa '%s' ha fallado (codigo %s). Pipeline detenido.", name, result.returncode)
        sys.exit(result.returncode)
    logger.info("Etapa '%s' completada en %.1fs", name, time.time() - start)


def run_pipeline() -> None:
    for name, module, extra_args in STEPS:
        run_step(name, module, extra_args)
    logger.info("Pipeline completado correctamente")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_pipeline()
