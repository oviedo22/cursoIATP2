# Predicción de Tabaquismo — Clasificación con Machine Learning

Trabajo Práctico 2 — Módulo 2. Proyecto de clasificación binaria para predecir si una persona fuma (`smoking` = 1) o no (`smoking` = 0) a partir de variables biométricas y de exámenes médicos.

---

## Descripción del proyecto y objetivo

El objetivo es construir un modelo de Machine Learning capaz de predecir el hábito de fumar de una persona a partir de un conjunto de características físicas y de laboratorio (presión arterial, colesterol, hemoglobina, enzimas hepáticas, etc.).

El flujo sigue un pipeline completo de ciencia de datos: lectura, análisis exploratorio, preprocesamiento, entrenamiento y optimización de varios modelos, validación, y generación de predicciones sobre un conjunto de datos sin etiquetar.

La **métrica de evaluación** es el **F1-Score sobre la clase 1 (fumadores)**, elegida porque el dataset está desbalanceado (~63% no fumadores, ~37% fumadores) y el accuracy sería engañoso.

---

## Estructura del proyecto

```
smoking-classifier/
│
├── data/
│   ├── raw/                          # Datos originales sin modificar
│   │   ├── smoking_prediction.xlsx           # Datos etiquetados (train/test)
│   │   └── smoking_prediction_entrega.xlsx   # Datos sin etiquetar (predicción)
│   ├── processed/                    # Datos procesados y resultado final
│   │   ├── X_train.parquet, X_test.parquet
│   │   ├── X_train_scaled.parquet, X_test_scaled.parquet
│   │   ├── y_train.parquet, y_test.parquet
│   │   └── smoking_prediction_resultado.xlsx  # ENTREGABLE FINAL
│   └── external/                     # (vacío, sin datos externos)
│
├── models/                           # Modelos y transformadores guardados
│   ├── ohe_encoder.joblib                    # OneHotEncoder ajustado
│   ├── scaler.joblib                         # MinMaxScaler ajustado
│   ├── metadata_preprocesamiento.joblib      # Config de columnas
│   ├── modelo_final_xgboost.joblib           # Modelo final
│   ├── threshold_final.joblib                # Threshold de decisión
│   └── metricas_modelos.joblib               # Métricas de comparación
│
├── notebooks/                        # Notebooks ordenados por etapa
│   ├── 01_lectura_y_discovery.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocesamiento.ipynb
│   ├── 04_entrenamiento_y_optimizacion.ipynb
│   ├── 05_validacion.ipynb
│   └── 06_prediccion.ipynb
│
├── requirements.txt
└── README.md
```

---

## Instrucciones para reproducir el entorno y ejecutar el código

### 1. Crear y activar un entorno virtual

```bash
python -m venv env
source env/bin/activate     # Linux / Mac
env\Scripts\activate        # Windows
```

### 2. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 3. Ejecutar los notebooks en orden

Los notebooks están numerados y deben ejecutarse secuencialmente del 01 al 06, ya que cada uno depende de los archivos que genera el anterior (datos procesados, modelo, threshold). Desde la carpeta `notebooks/`:

```bash
jupyter notebook
```

Ejecutar cada notebook con **Kernel → Restart & Run All**. El notebook 06 genera el archivo final `data/processed/smoking_prediction_resultado.xlsx`.

---

## Explicación de la estructura de archivos

Cada notebook cumple una etapa del pipeline:

- **01 — Lectura y Discovery:** carga ambos datasets, verifica dimensiones, tipos, nulos, duplicados y la distribución del target.
- **02 — EDA:** análisis exploratorio. Identifica columnas inútiles, la variable más predictiva, redundancias y distribuciones.
- **03 — Preprocesamiento:** elimina columnas, define X e y, hace el split 80/20 estratificado, codifica categóricas y escala. Guarda los transformadores.
- **04 — Entrenamiento y Optimización:** entrena 4 modelos, los compara por F1 y overfitting, y optimiza el mejor con RandomizedSearchCV.
- **05 — Validación:** matriz de confusión y ajuste del threshold de clasificación.
- **06 — Predicción:** aplica el pipeline completo a los datos sin etiquetar y exporta el resultado.

La carpeta `models/` guarda los objetos entrenados para que la predicción (notebook 06) use exactamente el mismo pipeline que el entrenamiento, sin re-ajustar nada sobre los datos nuevos.

---

## Resumen de experimentos, pruebas e intentos

### Selección de variables
- Se eliminó `ID` (identificador sin valor predictivo) y `oral` (un único valor `Y` en las 50.000 filas, sin varianza).
- Se identificó `gender` como la **variable más predictiva** (correlación 0.51 con el target). La tasa de tabaquismo masculina es muy superior a la femenina en este dataset.
- `height(cm)` y `hemoglobin` también son fuertes predictores, parcialmente por su correlación con el sexo.
- Se decidió mantener todas las demás features y dejar que los modelos de árboles hicieran la selección implícita.

### Tratamiento de datos
- No se eliminaron outliers porque los modelos finales (basados en árboles) son robustos a ellos.
- Se generaron dos versiones de los datos: sin escalar (árboles) y escalada con MinMaxScaler (KNN).
- Split 80/20 estratificado para mantener la proporción del target.

### Modelos entrenados y resultados (F1 sobre clase 1)

| Modelo | F1 test | Gap (overfit) |
|---|---|---|
| Decision Tree (max_depth=9) | 0.673 | +0.038 |
| KNN (k=200) | 0.653 | +0.014 |
| Random Forest | 0.685 | +0.029 |
| XGBoost (base) | 0.688 | +0.084 |
| **XGBoost (optimizado)** | **0.692** | **+0.044** |
| **XGBoost (optimizado + threshold 0.42)** | **0.716** | **+0.031** |

### Decisiones de optimización
- Se optimizó XGBoost con RandomizedSearchCV (30 combinaciones, cv=4), priorizando regularización (`max_depth` bajo, `reg_alpha`/`reg_lambda`, `min_child_weight` alto) para reducir el overfitting. El gap bajó de 0.084 a 0.044.
- Se ajustó el **threshold de decisión** de 0.50 a 0.42, buscando el óptimo sobre el conjunto de train (sin tocar test). Esto mejoró el F1 en test a 0.716 y redujo aún más el gap, ya que un umbral más bajo compensa el desbalance hacia no-fumadores.

---

## Conclusiones principales

- El **sexo** (`gender`) es el predictor más fuerte del hábito de fumar en este dataset, seguido de variables biométricas asociadas (altura, hemoglobina, peso).
- El modelo final, **XGBoost optimizado con threshold ajustado a 0.42**, alcanza un **F1-Score de ~0.716 sobre la clase fumadores** en el conjunto de test, con un overfitting controlado (gap de 0.031).
- La prioridad metodológica fue **evitar el sobreajuste**: en cada decisión (regularización, elección del modelo, threshold sobre train) se privilegió la generalización por encima del rendimiento puro en test.
- El pipeline es **completamente reproducible**: los transformadores y el modelo se guardan y se reutilizan idénticos en la predicción final, evitando data leakage.

---

## Métrica de éxito

El rendimiento definitivo se calcula sobre las predicciones del conjunto sin etiquetar (`smoking_prediction_resultado.xlsx`), en la columna `smoking_prediction` con valores `0`/`1`, evaluado con **F1-Score para target = 1**.

---

*Trabajo Práctico 2 — Módulo 2 — Curso de IA con Machine Learning*
