# OB-stars

Code for the paper:

> **Astrophotometric search for massive stars in the Milky Way.**
> **Confronting Random Forest predictions with available spectroscopy.**
> N. Monsalves, A. Bayo, M. Jaque Arancibia, J. Bodensteiner, A. G. Caneppa, P. Sánchez-Sáez, R. Angeloni (2025).
> [arXiv:2508.21573](https://arxiv.org/abs/2508.21573)

The paper identifies OB massive star candidates in the Milky Way by training a Balanced Random Forest on Gaia DR3 photometry cross-matched with the Skiff spectral-type compilation, then validates predictions against LAMOST DR10 spectra and high-resolution standards (HERMES/FEROS/IACOB).

---

## Setup

```bash
conda env create -f RF.yml          # BRF training notebooks
conda activate RF

# For spectral notebooks:
# conda activate standards           # install specutils, astropy
```

Large data files are stored on Google Drive (links below). The `data/` directory is gitignored.

---

## Analysis notebooks (`notebooks/`)

| Notebook | Description |
|---|---|
| `1_OBA_GDR3_conditions.ipynb` | Select OBA stars from Gaia DR3; define quality cuts |
| `2_preprocesing_OBAGDR3-V2.ipynb` | Homogenise Skiff labels to the MK system; build training table |
| `3_Train_BRF_V2.ipynb` | Train the Balanced Random Forest; hyperparameter grid search; feature importance |
| `4_clasificar_standards.ipynb` | Degrade high-resolution standards to LAMOST resolution; spectral classification pipeline |
| `4_LAMOST.ipynb` | Automatic spectral typing of LAMOST candidates (line detection + Gaussian fitting) |
| `5_Comparation_other_works.ipynb` | Compare predictions against literature catalogues |

### Data files (Google Drive)

| File | Link |
|---|---|
| `OBADR3_2arcsecSkiff.csv` — Gaia DR3 × Skiff crossmatch | [Drive](https://drive.google.com/file/d/1Vu9lyB-xSGzsQLR1f1jtC5gyHDY9Lttv/view?usp=drive_link) |
| `OBADR3.csv` — Gaia DR3 OBA sample | [Drive](https://drive.google.com/file/d/1etbSm_15a_nWZJkP5XPdzrgGVC7C4bFe/view?usp=drive_link) |
| `SKIFF.csv` — full Skiff database | [Drive](https://drive.google.com/file/d/116K_U1-djnHWbtKFWw15YSTZcclVg2NC/view?usp=drive_link) |
| `skiff_2arcsec_OBAGDR3_prep_V2.csv` — preprocessed training table | [Drive](https://drive.google.com/file/d/1fDT3Fw0Rg3FzksvCyzD5OvUg2IoRXr2U/view?usp=drive_link) |
| BRF results (G mag) | [Drive](https://drive.google.com/drive/folders/1PwqXYQs5sDm5UYA9OAtzLAY12jSaDm_8?usp=drive_link) |

---

## Reusable templates (`reusable/`)

Generic notebooks extracted from the pipeline. Each runs end-to-end by editing **only the `CONFIG` cell**. Section 0 in each notebook contains a ready-to-run quick-start example using the files in `reusable/example_data/` (6.6 MB, included in the repo).

| Notebook | Used in paper for | Conda env |
|---|---|---|
| `brf_classifier_template.ipynb` | BRF training + hyperparameter search + CV feature importance uncertainty | `RF` |
| `spectral_line_fitting.ipynb` | Automatic spectral typing of LAMOST candidates (Hβ, Hγ, He I 4471, Mg II 4481, He II 4686) | `standards` |
| `spectral_resolution_degrade.ipynb` | Degrading HERMES/FEROS standards (R ≈ 85 000) to LAMOST resolution (R ≈ 1 300) | `standards` |

### Quick start

```bash
# BRF template (~30 s on 200 rows)
conda activate RF
cd reusable
jupyter notebook brf_classifier_template.ipynb   # run section 0, then run all

# Spectral notebooks
conda activate standards
jupyter notebook spectral_line_fitting.ipynb
jupyter notebook spectral_resolution_degrade.ipynb
```

---

### `brf_classifier_template.ipynb`

Pipeline for training a `BalancedRandomForestClassifier` on any labelled table.

**What it does:**
1. Grid search over `n_estimators`, `max_features`, `max_depth`, `sampling_strategy` with `StratifiedKFold` and IBA-Gmean scoring (robust for imbalanced classes).
2. Repeated CV with `StratifiedShuffleSplit(20)` saving per-fold: precision/recall/F1, confusion matrix, and **feature importance + std across trees** → distribution of importances over folds.
3. Summary plots: confusion matrix mean ± std and feature importance mean ± std bar chart.
4. Final model trained on 80 % of data, saved with `joblib`. Predictions CSV with `prob_<class>` columns.

Supports binary and multiclass automatically. For binary with a custom threshold use `'binary_threshold': 0.6` in `CONFIG`.

**Output files in `output_dir/`:**

| File | Content |
|---|---|
| `best_params.csv` | Best hyperparameters |
| `gridsearch_full_results.csv` | Full grid search results |
| `cv_metrics_per_fold.csv` | precision/recall/F1 per fold per class |
| `cv_metrics_summary.csv` | Mean ± std of metrics per class |
| `cv_feature_importance_per_fold.csv` | Feature importance per fold (one row per feature × fold) |
| `cv_feature_importance_summary.csv` | Mean ± std of importance per feature |
| `confusion_matrix.pdf` | CV confusion matrix (mean ± std) |
| `feature_importance.pdf` | Top-30 feature importance bar chart |
| `model.joblib` | Trained model |
| `predictions.csv` | Input table + `split`, `prob_<class>`, `predicted` columns |

**Dependencies:** `pandas numpy scikit-learn imbalanced-learn joblib matplotlib seaborn ray`
(`ray` optional; disable with `'use_ray': False`)

---

### `spectral_line_fitting.ipynb`

Detects and characterises spectral lines in any optical spectrum. For each line:

1. Refines local continuum with two sigma-clipping passes over `blue_cont` + `red_cont` windows.
2. Detects absorption, emission, or non-detection against `detection_threshold × σ`.
3. Fits three models — **single Gaussian**, **blended (two Gaussians)**, **core emission** — and selects by BIC.
4. Computes EW by Newton–Cotes integration over `[λ₀ ± 4σ]`.

**Minimum CONFIG:**
```python
CONFIG = {
    'spectrum_path': 'my_spectrum.fits',
    'reader':        'lamost',       # 'lamost' | 'generic_fits' | 'csv'
    'R':             1300,
    'lines': [
        {'name': 'HeI 4471', 'lambda': 4471,
         'blue_cont': [4445, 4465], 'red_cont': [4490, 4510],
         'line_window': [4465, 4490], 'companion': 4481},
    ],
}
```

**Output:** `line_measurements.csv` (one row per line: `detection`, `model`, `EW`, `EW2`, `lambda_fit`, `sigma_fit`, `bic`, `snr`) + one PDF per line.

**Dependencies:** `numpy pandas scipy astropy matplotlib scikit-learn`

---

### `spectral_resolution_degrade.ipynb`

Converts a high-resolution spectrum to a lower resolution with:

1. **Gaussian convolution** — two modes: `'fwhm_out_only'` (approximation, valid when R_in ≫ R_out) or `'deconv'` (exact: FWHM_kernel² = FWHM_out² − FWHM_in²).
2. **Flux-conserving resampling** (`specutils.FluxConservingResampler`) to a configurable linear or log-uniform grid.
3. **Optional noise injection** to reach a target continuum RMS.

**Minimum CONFIG:**
```python
CONFIG = {
    'spectrum_path': 'my_high_res_spectrum.fits',
    'reader':        'iacob',   # 'iacob' | 'feros' | 'hermes' | 'melchior' | 'generic_fits' | 'csv'
    'R_in':          85000,
    'R_out':         1300,
    'lambda_ref':    4471,      # Å — reference wavelength for FWHM
    'target_rms':    None,      # float to inject noise, None to skip
}
```

**Output:** `<name>_R<R_out>.csv` (`lambda`, `flux`, `continuo`) + `degrade_comparison.pdf`.

**Dependencies:** `numpy pandas astropy specutils matplotlib scikit-learn`
