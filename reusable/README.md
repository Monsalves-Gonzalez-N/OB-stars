# Reusable notebooks

Code templates extracted from the analysis pipeline of:

> **Astrophotometric search for massive stars in the Milky Way.**
> **Confronting Random Forest predictions with available spectroscopy.**
> N. Monsalves, A. Bayo, M. Jaque Arancibia, J. Bodensteiner, A. G. Caneppa, P. Sánchez-Sáez, R. Angeloni (2025).
> [arXiv:2508.21573](https://arxiv.org/abs/2508.21573)

The paper identifies OB massive star candidates in the Milky Way by training a Balanced Random Forest on Gaia DR3 photometry (G, G_BP, G_RP, 2MASS JHKs, parallax) cross-matched with the Skiff spectral-type compilation, and then validates predictions spectroscopically against LAMOST DR10 spectra and high-resolution standards (HERMES/FEROS/IACOB). The three notebooks here cover the three technical pieces that required the most adaptation and are likely to be useful in other contexts.

Each notebook runs end-to-end by editing **only the `CONFIG` cell** (section 1). Section 0 in each notebook contains a ready-to-run quick-start example using the files in `example_data/`.

| Notebook | Used in paper for | Conda env |
|---|---|---|
| `brf_classifier_template.ipynb` | Training the BRF classifier + hyperparameter search + feature importance with CV uncertainty | `RF` |
| `spectral_line_fitting.ipynb` | Automatic spectral typing of LAMOST candidates (Hβ, Hγ, He I 4471, Mg II 4481, He II 4686…) | `standards` |
| `spectral_resolution_degrade.ipynb` | Degrading HERMES/FEROS high-resolution standards (R ≈ 85 000) to LAMOST resolution (R ≈ 1300) for consistent comparison | `standards` |

---

## Quick start

```bash
git clone https://github.com/<user>/OB-stars
cd OB-stars/reusable

# BRF template (~30 s on 200 rows)
conda activate RF
jupyter notebook brf_classifier_template.ipynb
# → run section 0, then run all

# Spectral notebooks
conda activate standards
jupyter notebook spectral_line_fitting.ipynb
jupyter notebook spectral_resolution_degrade.ipynb
```

All three example datasets are included in `example_data/` (6.6 MB total):

| File | Description |
|---|---|
| `brf_sample_200.csv` | 200-row subsample of the paper training set (O/B/A labels, 7 Gaia+2MASS features) |
| `spec-56573-KP185031N425443V01_sp08-241.fits` | LAMOST DR10 spectrum (R ≈ 1300, SNR ≈ 143) used to demonstrate line fitting |
| `00374442_melchiors_spectrum_coadded.fits` | HERMES coadded spectrum from the MELCHIORS library (R ≈ 85 000) used to demonstrate resolution degradation |

To use your own data, skip section 0 (or restart the kernel) and fill in `CONFIG` directly.

---

## `brf_classifier_template.ipynb`

### Context in the paper

The paper trains a `BalancedRandomForestClassifier` on ~34 000 stars with reliable MK spectral types from the Skiff (2014) compilation cross-matched with Gaia DR3. The classifier distinguishes OB massive stars from A/FGK stars using 7 photometric + astrometric features. Hyperparameters were tuned with `GridSearchCV` using `index_balanced_accuracy(geometric_mean_score)` as the scoring metric — robust for the severe class imbalance between massive (rare) and non-massive stars. The CV uncertainty on feature importance is used in the paper to argue which photometric features drive the separation.

### What the template does

1. Loads any CSV/parquet table with a classification label.
2. Grid search over `n_estimators`, `max_features`, `max_depth`, `sampling_strategy` with `StratifiedKFold` and IBA-Gmean scoring.
3. Repeated CV with `StratifiedShuffleSplit(20)` that saves per-fold:
   - precision / recall / F1 per class
   - confusion matrix
   - **feature importance + std across trees within each fold** → gives a distribution of importances across folds
4. Summary plots: confusion matrix mean ± std (PDF) and feature importance mean ± std bar chart (PDF).
5. Final model trained on 80 % of data, saved with `joblib`. Predictions CSV with `prob_<class>` columns.

Supports binary and multiclass classification automatically. For binary with a custom threshold use `'binary_threshold': 0.6` in `CONFIG`.

### Edit only this

```python
CONFIG = {
    'data_path':    'path/to/your_table.csv',  # CSV or parquet
    'target_col':   'label',                   # column with the class
    'feature_cols': None,                      # list of features; None = all numeric columns
    'output_dir':   'results_run1',
    # ...
}
```

### Output files

| File | Content |
|---|---|
| `best_params.csv` | Best hyperparameters from grid search |
| `gridsearch_full_results.csv` | Full grid search results |
| `cv_metrics_per_fold.csv` | precision/recall/F1 per fold per class |
| `cv_metrics_summary.csv` | Mean ± std of metrics per class |
| `cv_feature_importance_per_fold.csv` | Feature importance per fold (one row per feature × fold) |
| `cv_feature_importance_summary.csv` | Mean ± std of importance per feature |
| `confusion_matrix.pdf` | CV confusion matrix (mean ± std) |
| `feature_importance.pdf` | Top-30 feature importance bar chart |
| `model.joblib` | Final trained model |
| `predictions.csv` | Input table + `split`, `prob_<class>`, `predicted` columns |

### Dependencies

```
pandas numpy scikit-learn imbalanced-learn joblib matplotlib seaborn ray
```

`ray` is optional (parallelizes the CV loops); disable with `'use_ray': False` in `CONFIG`.

---

## `spectral_line_fitting.ipynb`

### Context in the paper

LAMOST candidates predicted as OB by the BRF were validated by measuring equivalent widths of temperature-sensitive lines: Hβ (4861 Å), Hγ (4340 Å), He I (4471 Å), Mg II (4481 Å), He II (4686 Å). The presence/absence and depth of these lines constrains the spectral type independently of photometry. The Mg II 4481 / He I 4471 ratio is particularly sensitive to temperature near B2–B3. The pipeline handles blended He I + Mg II automatically via a two-Gaussian model selected by BIC.

### What the template does

For each line declared in `CONFIG['lines']`:

1. Refines the local continuum with two sigma-clipping passes over the `blue_cont` + `red_cont` windows.
2. Detects absorption, emission, or non-detection by comparing against `detection_threshold × σ` of the continuum.
3. Fits three models: **single Gaussian**, **blended (two Gaussians)** if a `companion` line is declared, and **core emission** (absorption + central emission). Selects by BIC.
4. Computes EW by Newton–Cotes integration over `[λ₀ ± 4σ]`.

### Minimum CONFIG

```python
CONFIG = {
    'spectrum_path':  'my_spectrum.fits',
    'reader':         'lamost',       # 'lamost' | 'generic_fits' | 'csv'
    'R':              1300,
    'lines': [
        {'name': 'HeI 4471', 'lambda': 4471,
         'blue_cont': [4445, 4465], 'red_cont': [4490, 4510],
         'line_window': [4465, 4490], 'companion': 4481},
    ],
}
```

### Output

- `line_measurements.csv` — one row per line: `detection`, `model`, `EW`, `EW2`, `lambda_fit`, `sigma_fit`, `bic`, `snr`.
- One PDF per line with continuum fit + Gaussian model overlaid.

### Adding your own reader

Define a function `(path) -> (wave_AA, flux, ivar_or_None)` and add it to the `READERS` dict in cell 3. If you don't have `ivar`, return `None` and the code uses `ones_like(flux)`.

### Dependencies

```
numpy pandas scipy astropy matplotlib scikit-learn
```

---

## `spectral_resolution_degrade.ipynb`

### Context in the paper

The spectral standards used for validation (HERMES, FEROS, IACOB) have resolutions between R ≈ 25 000 and 85 000. To build a consistent comparison sample at LAMOST resolution (R ≈ 1300) — and to simulate the noise level of the faintest LAMOST targets — these spectra were convolved to R ≈ 1300 and resampled to the LAMOST wavelength grid. The MELCHIORS library (Royer et al. 2024) provided 163 coadded HERMES spectra that were processed this way.

### What the template does

1. **Gaussian convolution** with σ calculated to match the target FWHM. Two modes:
   - `'fwhm_out_only'` — approximation valid when R_in ≫ R_out: FWHM_kernel ≈ λ/R_out.
   - `'deconv'` — exact kernel: FWHM_kernel² = FWHM_out² − FWHM_in².
2. **Flux-conserving resampling** (`specutils.FluxConservingResampler`) to a configurable linear or log-uniform grid.
3. **Optional noise injection** to reach a target RMS (e.g. the typical continuum RMS of your low-resolution instrument).

### Minimum CONFIG

```python
CONFIG = {
    'spectrum_path': 'my_high_res_spectrum.fits',
    'reader':        'iacob',     # 'iacob' | 'feros' | 'hermes' | 'melchior' | 'generic_fits' | 'csv'
    'R_in':          85000,
    'R_out':         1300,
    'lambda_ref':    4471,        # Å — reference wavelength for FWHM calculation
    'target_rms':    None,        # float to inject noise, None to skip
}
```

### Output

- `<spectrum_name>_R<R_out>.csv` — columns `lambda`, `flux`, `continuo`.
- `degrade_comparison.pdf` — input vs degraded spectrum overplotted.

### Dependencies

```
numpy pandas astropy specutils matplotlib scikit-learn
```
