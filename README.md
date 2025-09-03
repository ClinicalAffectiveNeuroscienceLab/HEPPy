# HEPPy — Heartbeat-Evoked Potential (HEP) extraction for EEG

HEPPy is a small, practical pipeline for extracting heartbeat-evoked potentials (HEPs) from EEG recordings that include an ECG channel. It provides a **thin GUI** to:

1. run **QC** to detect R-peaks and generate per-file QRS plots;
2. **review** those plots and mark each file **GOOD** or **BAD** (with optional RR estimate for the bad ones);
3. run **HEP** extraction using your QC decisions.

Outputs are standard MNE objects on disk plus a summary CSV. Optionally, HEPPy can compute **HRV metrics** via NeuroKit2.

---

## Features

* **Simple GUI** wrapper over a robust MNE-Python pipeline.
* **Automatic R-peak detection** (+ manual RR hint when needed).
* **QRS QC plots** for rapid screening.
* **Epoching and amplitude rejection** around R-peaks (configurable).
* **Error logs per file** with full Python tracebacks.
* **Optional HRV metrics** (time/frequency/non-linear) with `do_hrv = True`.
* Works with **EDF** or **FIF** inputs; handles typical montages.

## How to use the GUI (step-by-step)

1. **Config key**: leave as `config` unless you have a different config module name (see **Configuration**).
2. **Select Input Directory…**: choose the folder containing your EEG files.

   * Supported: `*.edf`, `*-raw.fif`, and `*.fif` (customise via “Globs”).
3. *(optional)* **Output dir**: set a different output folder; otherwise HEPPy will use `<input>/heppy_output`.
4. **1) Run QC**

   * Detects R-peaks per file.
   * Writes a QC CSV (default `qc_review.csv`) and QRS PNGs (`qc_plots/*_ecg_qc.png`).
5. **2) Start Review**

   * The right panel shows one QC PNG at a time.
   * Mark **GOOD** or **BAD**.
   * If **BAD**, enter an **RR estimate** (in seconds, e.g. `0.8`).
   * Use **Previous/Next**, and **Save QC CSV** whenever you like.
6. **3) Run HEP**

   * Runs the full HEP extraction using your QC decisions.
   * Shows a determinate progress bar (files processed / total).
   * On completion, a message confirms where results are saved.

---

## Configuration

HEPPy loads a Python module (default key: `config`) into a `HEPConfig`. You’ll find an example `config_hep.py` in the repo; you can use it as your active config by setting **Config key** to `config_hep` in the GUI.

### Important options (with sensible defaults)

| Field          | Type / Example    | Meaning                                                                                  |
| -------------- | ----------------- | ---------------------------------------------------------------------------------------- |
| `save_stem`    | `"hep"`           | Prefix for output files.                                                                 |
| `qc_csv_name`  | `"qc_review.csv"` | Name of the QC decisions CSV.                                                            |
| `qc_dirname`   | `"qc_plots"`      | Folder (inside `output_root`) for PNGs & caches.                                         |
| `target_sfreq` | `256.0` or `None` | If set, preprocessing resamples to this rate. Cached R-peaks are automatically rescaled. |
| `montage_name` | `"standard_1020"` | Montage to apply.                                                                        |
| `tmin`, `tmax` | `-0.2`, `0.8`     | Epoch window in seconds around R-peaks.                                                  |
| `baseline`     | `(-0.2, -0.05)`   | Baseline correction window.                                                              |
| `amp_rej_uv`   | `150.0`           | Drop epochs with peak-to-peak amplitude above this (µV).                                 |
| `amp_window_s` | `(0.1, 0.5)`      | Time window (s) used for amplitude check.                                                |
| `min_rr_s`     | `0.7`             | Only R-peaks with **present RR** ≥ this are considered for HEP.                          |
| `ecg_channel`  | `None` or `"ECG"` | ECG channel name (if `None`, detector searches heuristically).                           |
| `stim_name`    | `"STI 014"`       | Name for the generated stim channel marking R-peaks.                                     |
| `do_hrv`       | `False` / `True`  | **NEW**: if `True`, compute HRV metrics (time/freq/non-linear) and add to summary CSV.   |
| `verbose`      | `True`            | Print progress/info to console.                                                          |

To use the sample config:

1. Copy `config_hep.py` to your project root (or edit it there).
2. In the GUI, set **Config key** to `config_hep`.
3. Adjust parameters as needed.

---

## Input expectations

* Place your EEG files (EDF or FIF) in the input directory you select.
* If you have multiple subfolders, adjust the **Globs** field (semicolon-separated), e.g.:
  `**/*.edf; **/*-raw.fif; **/*.fif` *(default)*

---

## Outputs

All outputs are written under `output_root` (defaults to `<input>/heppy_output`):

```
heppy_output/
  qc_plots/
    <base>_ecg_qc.png
    <base>_rpeaks_cache.json
    <base>_ecg_qc_corrected.png     # when RR correction was applied
  qc_review.csv                      # your QC decisions
  hep_<base>.epo.fif                 # HEP epochs per processed file
  hep_summary.csv                    # per-file summary table (see below)
  logs/
    error_001_log.txt                # full traceback + context (if any file failed)
    error_002_log.txt
    ...
```

### Summary CSV columns (typical)

* `file`, `base`, `action` (`saved|excluded|error`)
* `epochs_fif` (path), `n_hep`, `n_rpeaks_used`, `median_rr_s`
* `rr_corrected` (bool), `used_cache` (bool)
* `error_log` (path to `logs/error_###_log.txt` when `action=error`)
* **If `do_hrv=True`**: many `hrv_*` columns, e.g.
  `hrv_HR`, `hrv_SDNN`, `hrv_RMSSD`, `hrv_pNN50`, `hrv_LF`, `hrv_HF`, `hrv_LFHF`, `hrv_SD1`, `hrv_SD2`, etc.
  (Computed via NeuroKit2’s `hrv_time`, `hrv_frequency`, `hrv_nonlinear`.)

---

## How it works (high level)

1. **QC (`make_qc_checklist`)**:

   * Loads each file (EDF/FIF), performs light normalisation.
   * Detects R-peaks and saves a compact cache (`*_rpeaks_cache.json`) with the sampling rate used for detection.
   * Writes a PNG of the cleaned ECG and an initial QC CSV row.
2. **Review (GUI)**:

   * You mark `qc_status = ok/bad` (or `exclude`/`skip`) and enter `manual_est_rr_s` when needed.
3. **HEP (`apply_qc_and_extract`)**:

   * Full preprocessing (montage, filters, optional re-sampling to `target_sfreq`, artefact handling).
   * Uses cached R-peaks when safe; **automatically rescales** cached indices if the sampling rate changed.
   * Applies manual RR hint for “bad” rows (and redraws a “corrected” ECG PNG).
   * Adds a stim channel at R-peaks; builds HEP epochs; drops epochs by amplitude; saves `.epo.fif`.
   * **If `do_hrv=True`**, computes HRV metrics from R-peaks (at the current sampling rate) and appends to the summary row.
   * On any per-file failure, writes a **full traceback log** in `output_root/logs/` and notes it in the summary.

---

## Troubleshooting

* **The GUI opens but the “Select Input Directory…” button is disabled**
  Fixed: the current GUI keeps top controls enabled at launch. If you still see this, you’re running an older `main.py`. Replace it with the latest version in this repo and run again.

* **“ModuleNotFoundError: mne / neurokit2 / pyprep …”**
  You’re not in the right environment or you skipped a dependency.
  Run `mamba activate heppy` (or `conda activate heppy`) and re-install the missing package(s), e.g. `mamba install mne` or `pip install neurokit2`.

* **PyTorch installation is failing**
  Use the CPU-only command:
  `pip install --index-url https://download.pytorch.org/whl/cpu torch`
  You don’t need GPU acceleration for HEPPy.

* **No ECG channel found**
  Set `ecg_channel` in your config if your ECG has a non-standard name. If left `None`, the detector tries a best-effort search.

* **“R-peak indices are out of bounds”** during HEP
  This indicates a sampling-rate mismatch. Ensure you’re on the latest code: cached R-peaks are now auto-rescaled to the current `raw.info['sfreq']` during Stage 2.

* **HEP runs but summary shows many `error` rows**
  Look in `heppy_output/logs/error_###_log.txt` for each failing file. Each log contains the full traceback, config snapshot, and the offending file path.

## Known issues:
1. ASRPY is no longer supported - sometimes there are indexing errors in it, I think due to edge cases with data arrays - currently not in scope to solve.
2. If you rerun it on the same directory it gets confused.

---

## Development tips

* Keep your environment small and reproducible; prefer `mamba`/`conda` for heavy libs and `pip` for the rest.
* If you add new config fields, extend `HEPConfig` in `heppy/config.py`.
* The GUI is a thin wrapper; the core logic is in `heppy/review.py`. Unit tests should target those functions.

---

## Licence

MIT. See `LICENSE`.

---

## Citation

If you use HEPPy in a paper or product, please cite this repository. Check the repo for an associated preprint/paper if one becomes available.

---

### Appendix: Minimal command list (Windows/macOS/Linux)

```bash
# 1) Install Mambaforge (once) → open a new shell
# 2) Create env
mamba create -n heppy python=3.11 -y
mamba activate heppy

# 3) Install deps
mamba install -y numpy pandas pillow mne
pip install mne-icalabel neurokit2 asrpy pyprep
pip install --index-url https://download.pytorch.org/whl/cpu torch  # CPU-only

# 4) Run GUI
cd /path/to/HEPPy
python -m heppy.main
```

That’s it.


