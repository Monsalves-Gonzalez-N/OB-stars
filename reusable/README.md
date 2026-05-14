# Reusable templates

Plantillas genéricas para problemas recurrentes. Cada notebook se usa editando **únicamente** la celda `CONFIG`.

| Notebook | Para qué |
|---|---|
| `brf_classifier_template.ipynb` | Clasificación con Balanced Random Forest (grid search + CV con feature importance por fold). |
| `spectral_line_fitting.ipynb` | Detección + ajuste Gaussiano (single / blended / core-emission) + EW de líneas espectrales. |
| `spectral_resolution_degrade.ipynb` | Degradar espectros de alta a baja resolución + igualar ruido a un RMS objetivo. |

---

## `brf_classifier_template.ipynb`

Pipeline completo de Balanced Random Forest a partir de cualquier tabla con clasificación.

**Para usarlo:** abre el notebook, edita **únicamente** la celda `CONFIG` (sección 1) y ejecuta todo de arriba a abajo.

### Qué tienes que editar

```python
CONFIG = {
    'data_path':    'path/to/your_table.csv',  # CSV o parquet
    'target_col':   'label',                   # columna con la clase
    'feature_cols': None,                      # lista de features; None = todas las numéricas
    'output_dir':   'results_run1',
    # ...
}
```

### Qué hace

1. Carga la tabla y descarta filas con NaN o valores inválidos (`-999`, `999`, configurables).
2. Grid search sobre `n_estimators`, `max_features`, `max_depth`, `sampling_strategy` con `StratifiedKFold(5)` y métrica `index_balanced_accuracy(geometric_mean_score)`.
3. CV repetida con `StratifiedShuffleSplit(20)` que en **cada fold** guarda:
   - precision/recall/F1 por clase
   - matriz de confusión
   - **feature importance del estimador completo** + std entre árboles
4. Resúmenes: matriz de confusión media ± std (PDF) y feature importance media ± std (CSV + PDF).
5. Modelo final entrenado sobre 80% y guardado con `joblib`. Predicciones sobre toda la tabla (CSV con `prob_<clase>` por clase y columna `split`).

Soporta clasificación **binaria y multiclase** automáticamente. Para binaria con umbral custom, usa `'binary_threshold': 0.6` (etc.) en `CONFIG`.

### Archivos generados en `output_dir/`

| Archivo | Contenido |
|---|---|
| `best_params.csv` | Mejores hiperparámetros del grid search |
| `gridsearch_full_results.csv` | Resultados completos del grid search |
| `cv_metrics_per_fold.csv` | precision/recall/F1 por fold y por clase |
| `cv_metrics_summary.csv` | Media ± std de métricas por clase |
| `cv_feature_importance_per_fold.csv` | Feature importance por fold (una fila por feature × fold) |
| `cv_feature_importance_summary.csv` | Media ± std de importancia por feature |
| `confusion_matrix.pdf` | Matriz de confusión CV (media ± std) |
| `feature_importance.pdf` | Bar plot top-30 features |
| `model.joblib` | Modelo final entrenado |
| `predictions.csv` | Tabla original + columnas `split`, `prob_<clase>`, `predicted` |

### Dependencias

```
pandas numpy scikit-learn imbalanced-learn joblib matplotlib seaborn ray
```

`ray` es opcional (paraleliza la CV); para desactivarlo pon `'use_ray': False` en `CONFIG`.

---

## `spectral_line_fitting.ipynb`

Adaptado de `notebooks/4_LAMOST.ipynb`. Para cada línea declarada en `CONFIG['lines']`:

1. Refina el continuo local con dos pasos sigma-clipping sobre las ventanas `blue_cont` + `red_cont`.
2. Detecta absorción, emisión o no-detección comparando con `detection_threshold × σ` del continuo.
3. Ajusta tres modelos: **single Gaussian**, **dos Gaussianas blendeadas** (si declaraste `companion`) y **core emission** (absorción + emisión central). Elige por BIC.
4. Calcula EW por integración Newton–Cotes sobre `[λ₀ ± 4σ]`.

### CONFIG mínimo

```python
'spectrum_path': 'mi_espectro.fits',
'reader':        'lamost',       # o 'generic_fits', 'csv', o tu propio reader
'R':             1300,
'lines': [
    {'name': 'HeI 4471', 'lambda': 4471,
     'blue_cont': [4445, 4465], 'red_cont': [4490, 4510],
     'line_window': [4465, 4490], 'companion': 4481},
],
```

### Salida

- `line_measurements.csv` — una fila por línea con `detection`, `model`, `EW`, `EW2`, `lambda_fit`, `sigma_fit`, `bic`, `snr`.
- Un PDF por línea con detección + ajuste.

### Para añadir tu reader

Define una función `(path) -> (wave_AA, flux, ivar_or_None)` y agrégala al diccionario `READERS` en la celda 3.

---

## `spectral_resolution_degrade.ipynb`

Adaptado de `notebooks/4_clasificar_standards.ipynb`. Convierte un espectro de alta resolución (`R_in`) a una resolución más baja (`R_out`):

1. **Convolución Gaussiana** con σ calculado para igualar el FWHM al objetivo. Dos modos:
   - `'fwhm_out_only'` — aproximación del notebook original, válida cuando `R_in ≫ R_out`.
   - `'deconv'` — kernel exacto: `FWHM_kernel² = FWHM_out² − FWHM_in²`.
2. **Re-muestreo conservando flujo** (`specutils.FluxConservingResampler`) a una grilla lineal o log-uniforme configurable.
3. **Inyección de ruido opcional** hasta llegar a un `target_rms` (e.g. el RMS típico de tu instrumento de baja resolución).

### CONFIG mínimo

```python
'spectrum_path': 'mi_espectro_alta_resol.fits',
'reader':        'iacob',     # o 'feros', 'hermes', 'generic_fits', 'csv'
'R_in':          85000,
'R_out':         1300,
'lambda_ref':    4471,
'target_rms':    None,        # o un float si quieres simular ruido
```

### Salida

- `<spectrum>_R<R_out>.csv` con `lambda`, `flux`, `continuo`.
- `degrade_comparison.pdf` con espectro de entrada vs degradado.

### Dependencias adicionales

```
specutils astropy
```

