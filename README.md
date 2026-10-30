# Sistema MLOps de extremo a extremo: predicción de progresión de la diabetes

## 1. Objetivo

Implementar un flujo MLOps básico para un problema de **regresión**: extraer, validar y limpiar datos, entrenar modelos con seguimiento en **MLflow**, registrar y seleccionar un modelo **champion** mediante un *quality gate*, servirlo con una API **FastAPI** y automatizar todo el proceso con **GitHub Actions**.

El proyecto parte de un cuaderno exploratorio (`notebooks/exploration.ipynb`) y migra esa lógica a scripts reutilizables, separando responsabilidades: datos, entrenamiento, registro, API, orquestación y CI/CD.

## 2. Dataset

**Diabetes Dataset de scikit-learn** (`sklearn.datasets.load_diabetes`): 442 pacientes, 10 variables numéricas de partida (`age`, `sex`, `bmi`, `bp`, `s1`–`s6`), ya **estandarizadas** (centradas y escaladas), y un objetivo (`target`) que mide la **progresión de la enfermedad un año después**. El objetivo es predecir ese valor a partir de las 10 variables.

## 3. Estructura del repositorio

```
.
├── .github/workflows/ci.yml     # CI/CD: pipeline + tests en cada push a main/master
├── notebooks/exploration.ipynb  # Análisis exploratorio y modelado inicial
├── src/
│   ├── data/
│   │   ├── extraction.py        # Descarga el dataset y lo guarda en data/raw
│   │   ├── validation.py        # Contrato de datos (Pydantic): columnas, tipos, rangos
│   │   └── cleaning.py          # Limpieza -> data/processed
│   ├── training/train.py        # Entrenamiento + MLflow tracking (params, métricas, artefactos)
│   ├── registry/register.py     # Quality gate + Champion vs Challenger + alias @champion
│   ├── api/
│   │   ├── main.py              # FastAPI: /health, /model-info, /predict
│   │   └── schemas.py           # Esquemas Pydantic de entrada/salida
│   └── pipeline.py              # Orquesta las 5 etapas en orden
├── tests/test_api.py            # Tests de la API (200, 422, 503, 500)
├── requirements.txt             # Dependencias con versiones fijadas
└── README.md
```

`data/`, `mlruns/` y `mlflow.db` no se versionan (están en `.gitignore`): se regeneran ejecutando el pipeline.

## 4. Instalación

Requiere Python 3.14 (el proyecto se ha desarrollado y probado con 3.14).

```bash
git clone https://github.com/Aitor012-012/ActividadMLOps-diabetes.git
cd ActividadMLOps-diabetes
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 5. Ejecutar el pipeline de datos

Cada etapa se ejecuta desde la raíz del repositorio:

```bash
python -m src.data.extraction   # 1. Extracción -> data/raw/diabetes_raw.csv
python -m src.data.validation   # 2. Validación del contrato de datos
python -m src.data.cleaning     # 3. Limpieza -> data/processed/diabetes_clean.csv
```

- **Extracción:** carga el dataset y lo guarda en bruto.
- **Validación:** comprueba con Pydantic que existen todas las columnas y el objetivo, que no hay columnas extra, que los tipos son numéricos, que no hay NaN/infinitos y que los valores están en rango. Si algo falla, el script termina con código de salida 1 (*fail-fast*), lo que pone el CI en rojo.
- **Limpieza:** elimina nulos y duplicados, fuerza tipos numéricos y recorta el objetivo por encima del percentil 99 (valores extremos). Registra cuántas filas se pierden (442 -> 437) y **aborta si se pierde más del 10 %**.

## 6. Entrenar y registrar el modelo

```bash
python -m src.training.train --model all    # entrena linear_regression y random_forest
python -m src.registry.register             # quality gate + registro + alias champion
```

Opciones de entrenamiento: `--model {linear_regression,random_forest,all}`, `--test-size 0.2`, `--seed 42`, `--n-estimators 100`, `--max-depth N`.

**Seguimiento con MLflow** (backend SQLite `mlflow.db`, artefactos en `mlruns/`, experimento `diabetes-regression`). Por cada ejecución se registran:
- **Parámetros:** tipo de modelo, hiperparámetros, `test_size`, `seed`, tamaños de train.
- **Métricas:** `rmse`, `mae`, `r2` sobre el conjunto de test.
- **Artefactos:** el modelo, con su *signature* (columnas de entrada y salida), un ejemplo de entrada y el entorno (`requirements.txt`, `conda.yaml`).

Resultados de referencia (test del 20 %, `seed=42`):

| Modelo | RMSE | MAE | R² |
|---|---|---|---|
| linear_regression | 56.45 | 43.34 | 0.386 |
| random_forest (100 árboles) | 58.97 | 46.04 | 0.330 |

Para explorar las ejecuciones en la interfaz web:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

(En GitHub Codespaces hay que añadir `--allowed-hosts "*.app.github.dev,localhost:*"`.)

## 7. Cómo se elige el champion

`register.py` aplica tres reglas:

1. **Challenger:** se toma la ejecución con **menor RMSE** del experimento.
2. **Quality gate:** el challenger solo avanza si `RMSE <= 60`. Si no, el script falla con código 1 y no se registra nada.
3. **Champion vs Challenger:** si ya existe un champion, el challenger lo sustituye **solo si lo mejora** (RMSE menor). Si no, se mantiene el champion actual y no se crea una versión nueva, de modo que repetir el pipeline no duplica modelos.

Si pasa, el modelo se registra en el **Model Registry** como `diabetes-regressor` y recibe el **alias `champion`** (con tags `rmse` y `model_type`). La API siempre carga `models:/diabetes-regressor@champion`, sin depender del número de versión.

**Por qué el umbral es 60:** un modelo que predijera siempre la media del objetivo obtendría un RMSE de aproximadamente **72.08** en el conjunto de test. Exigir `RMSE <= 60` obliga a mejorar esa referencia en torno a un **17 %**, es decir, a que el modelo aporte información real, sin pedir un rendimiento irreal para un dataset pequeño con un R² máximo en torno a 0.4–0.5. El umbral es configurable: `python -m src.registry.register --rmse-threshold 55`.

## 8. Ejecutar todo con un solo comando

```bash
python -m src.pipeline
```

Ejecuta extracción -> validación -> limpieza -> entrenamiento -> registro, y se detiene en la primera etapa que falle devolviendo su código de error.

## 9. Ejecutar la API

Requiere haber ejecutado antes el pipeline (para que exista el champion):

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Al arrancar carga **una sola vez** el modelo champion. Documentación interactiva (Swagger) en `http://localhost:8000/docs`.

## 10. Probar los endpoints

**GET /health**

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

**GET /model-info**

```bash
curl http://localhost:8000/model-info
# {"model_name":"diabetes-regressor","model_version":"1","alias":"champion","run_id":"...","metadata":{"rmse":"56.4504","model_type":"linear_regression"}}
```

**POST /predict** (10 variables ya estandarizadas, todas obligatorias)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":0.038,"sex":0.051,"bmi":0.062,"bp":0.022,"s1":-0.044,"s2":-0.035,"s3":-0.043,"s4":-0.003,"s5":0.02,"s6":-0.018}'
# {"prediction":206.56...,"model_name":"diabetes-regressor","model_version":"1"}
```

Las variables del dataset de scikit-learn están estandarizadas, por lo que los valores de entrada son pequeños (aprox. entre -0.13 y 0.2).

## 11. Manejo de errores

| Caso | Código | Cuándo ocurre |
|---|---|---|
| Petición válida | **200** | Entrada correcta y modelo cargado |
| Entrada inválida | **422** | Falta un campo, tipo incorrecto, valor fuera de [-1, 1], campo extra o JSON mal formado (validación con Pydantic) |
| Modelo no disponible | **503** | No hay champion cargado (`/predict` y `/model-info`); `/health` sigue respondiendo 200 |
| Fallo al predecir | **500** | El modelo lanza una excepción; el cliente recibe un mensaje genérico y el detalle queda en el log |

## 12. Tests

```bash
python -m pytest -q
```

Los 11 tests de `tests/test_api.py` usan un modelo falso, de modo que no dependen de MLflow ni de haber entrenado. Cubren 200, 422 (campo que falta, tipo, rango, campo extra), 503 y 500.

## 13. CI/CD con GitHub Actions

El workflow `.github/workflows/ci.yml` se ejecuta automáticamente en cada **push a `main` o `master`** (y manualmente con *Run workflow*). En una máquina limpia:

1. Descarga el código e instala Python 3.14 y las dependencias de `requirements.txt` (versiones fijadas para reproducibilidad).
2. Ejecuta, como pasos independientes y en orden: extracción -> validación -> limpieza -> entrenamiento -> registro y evaluación.
3. Ejecuta los tests de la API.

Si cualquier paso falla (datos inválidos, quality gate no superado, test roto), el workflow se detiene y queda en rojo.

## 14. Decisiones de diseño

- **Fail-fast:** validación, entrenamiento y registro terminan con código 1 ante un error, para que el CI lo detecte.
- **Alias en lugar de *stages*:** MLflow 3 recomienda aliases (`@champion`) sobre las etapas clásicas del registro.
- **Formato del modelo:** se guarda con `cloudpickle` (`skops` rechaza el Random Forest en MLflow 3.17). MLflow muestra un aviso de seguridad, inocuo aquí porque los modelos los genera el propio pipeline.
- **Versiones fijadas:** el modelo guardado depende de las versiones de scikit-learn/pandas/numpy; fijarlas en `requirements.txt` evita incompatibilidades entre el entrenamiento y la carga.
