# heppy/review.py
from __future__ import annotations
from pathlib import Path
import glob
import json
import math
import pandas as pd
import numpy as np
import mne
from config import HEPConfig
from ecg import detect_rpeaks, refine_with_rr_hint, add_rpeak_stim, save_ecg_qc_plot
from epochs import build_hep_epochs, save_hep_epochs

def _iter_inputs(input_glob) -> list[str]:
    if isinstance(input_glob, (list, tuple)):
        files = []
        for g in input_glob:
            files.extend(glob.glob(g, recursive=True))
        return sorted(set(files))
    return sorted(set(glob.glob(input_glob, recursive=True)))

def make_qc_checklist(cfg: HEPConfig) -> pd.DataFrame:
    """
    First pass: run ECG detection only, export a QRS QC .png,
    and write a QC CSV skeleton with columns you can hand-edit.
    """
    files = _iter_inputs(cfg.input_glob)
    rows = []
    qc_dir = cfg.output_root / cfg.qc_dirname
    qc_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        base = Path(f).stem
        try:
            raw = mne.io.read_raw_edf(f, preload=True, verbose=False) if f.lower().endswith(".edf") \
                else mne.io.read_raw_fif(f, preload=True, verbose=False)
            # light montage normalisation (no heavy preproc yet)
            if "eeg" in raw.get_channel_types():
                # best-effort name normalisation
                mapping = {ch: ch.replace("EEG ", "").split("-")[0].upper().replace("Z","z") for ch in raw.ch_names}
                raw.rename_channels(mapping)
            r, rr, ecg_clean = detect_rpeaks(raw, ecg_name=cfg.ecg_channel)
            if ecg_clean.size:
                save_ecg_qc_plot(ecg_clean, raw.info["sfreq"], qc_dir / f"{base}_ecg_qc.png")
            
            # Cache R-peaks data for efficiency (save as JSON)
            rpeak_cache = {
                "r_peaks": r.tolist() if r.size else [],
                "sampling_freq": float(raw.info["sfreq"]),
                "ecg_channel": cfg.ecg_channel
            }
            cache_path = qc_dir / f"{base}_rpeaks_cache.json"
            with open(cache_path, 'w') as f_cache:
                json.dump(rpeak_cache, f_cache)
            
            rows.append({
                "file": f,
                "base": base,
                "n_rpeaks_auto": int(len(r)),
                "median_rr_auto": float(np.median(rr)) if rr.size else math.nan,
                "qc_status": "",             # fill "ok" | "bad" by hand
                "manual_est_rr_s": "",       # optional (e.g., 0.8)
                "notes": "",
                "rpeak_cache_path": str(cache_path),  # For efficiency
            })
        except Exception as e:
            rows.append({"file": f, "base": base, "n_rpeaks_auto": 0, "median_rr_auto": math.nan,
                         "qc_status": "bad", "manual_est_rr_s": "", "notes": f"load/detect err: {e}",
                         "rpeak_cache_path": ""})

    df = pd.DataFrame(rows)
    out_csv = cfg.output_root / cfg.qc_csv_name
    df.to_csv(out_csv, index=False)
    return df

def apply_qc_and_extract(cfg: HEPConfig) -> pd.DataFrame:
    """
    Second pass: read QC CSV. For rows:
      - qc_status == 'ok': process as is
      - qc_status == 'bad':
          - if manual_est_rr_s provided: refine peaks using hint and proceed
          - else: skip (log exclusion)
    Saves ONLY HEP epochs for processed files.
    """
    out_rows = []
    df = pd.read_csv(cfg.output_root / cfg.qc_csv_name)
    outdir = cfg.output_root
    for _, row in df.iterrows():
        f = str(row["file"]); base = str(row["base"])
        status = str(row.get("qc_status", "")).lower().strip()
        rr_hint = row.get("manual_est_rr_s", "")
        rr_hint = float(rr_hint) if str(rr_hint).strip() else math.nan
        if status not in ("ok", "bad", "exclude", "skip", ""):
            status = "ok"  # be liberal

        try:
            raw = mne.io.read_raw_edf(f, preload=True, verbose=False) if f.lower().endswith(".edf") \
                else mne.io.read_raw_fif(f, preload=True, verbose=False)
            # full preprocessing (reuses your stable pipeline)
            from preprocess import apply_montage, clean_raw
            raw = apply_montage(raw, cfg)
            raw, prov = clean_raw(raw, cfg)

            # Try to load cached R-peaks for efficiency (if no manual correction needed)
            rpeak_cache_path = row.get("rpeak_cache_path", "")
            use_cached = (status == "ok" and not np.isfinite(rr_hint) and 
                         rpeak_cache_path and Path(rpeak_cache_path).exists())
            
            if use_cached:
                try:
                    with open(rpeak_cache_path, 'r') as f_cache:
                        cache_data = json.load(f_cache)
                    r = np.array(cache_data["r_peaks"], dtype=int)
                    rr = np.diff(r) / raw.info["sfreq"] if len(r) >= 2 else np.array([])
                    # Still need ECG signal for potential plot regeneration
                    from ecg import detect_rpeaks
                    _, _, ecg_clean = detect_rpeaks(raw, ecg_name=cfg.ecg_channel)
                    if cfg.verbose:
                        print(f"[{base}] Using cached R-peaks ({len(r)} peaks)")
                except Exception as cache_err:
                    if cfg.verbose:
                        print(f"[{base}] Cache load failed, re-detecting: {cache_err}")
                    use_cached = False
            
            if not use_cached:
                from ecg import detect_rpeaks
                r, rr, ecg_clean = detect_rpeaks(raw, ecg_name=cfg.ecg_channel)

            # Handle manual RR correction and regenerate plots if needed
            corrected_r_peaks = False
            if status in ("bad", "exclude", "skip"):
                if status in ("exclude", "skip") and not np.isfinite(rr_hint):
                    out_rows.append({"file": f, "base": base, "action": "excluded"})
                    continue
                if np.isfinite(rr_hint):
                    r_orig = r.copy()  # Keep original for comparison
                    r = refine_with_rr_hint(r, raw.info["sfreq"], rr_hint_s=rr_hint, factor=1.8)
                    rr = np.diff(r) / raw.info["sfreq"]
                    corrected_r_peaks = True
                    
                    # Generate corrected ECG plot for comparison
                    qc_dir = cfg.output_root / cfg.qc_dirname
                    qc_dir.mkdir(parents=True, exist_ok=True)
                    corrected_plot_path = qc_dir / f"{base}_ecg_qc_corrected.png"
                    save_ecg_qc_plot(ecg_clean, raw.info["sfreq"], corrected_plot_path, 
                                    r_peaks=r, title_suffix="(RR Corrected)")
                    
                    if cfg.verbose:
                        print(f"[{base}] Applied RR correction: {len(r_orig)} → {len(r)} R-peaks")
                        print(f"[{base}] Saved corrected ECG plot: {corrected_plot_path}")

            # add a stim channel for convenience (optional, helps viewing)
            add_rpeak_stim(raw, r, name="RPEAK")

            from epochs import build_hep_epochs, save_hep_epochs
            hep = build_hep_epochs(raw, r, rr, cfg)
            path = save_hep_epochs(hep, outdir, cfg.save_stem, base)
            
            # Enhanced output information
            out_info = {
                "file": f, "base": base, "action": "saved", 
                "epochs_fif": str(path), "n_hep": int(len(hep)),
                "n_rpeaks_used": int(len(r)),
                "median_rr_s": float(np.median(rr)) if rr.size else math.nan,
                "rr_corrected": bool(corrected_r_peaks),
                "used_cache": bool(use_cached if 'use_cached' in locals() else False)
            }
            out_rows.append(out_info)
        except Exception as e:
            out_rows.append({"file": f, "base": base, "action": "error", "error": str(e)})

    out = pd.DataFrame(out_rows)
    out.to_csv(cfg.output_root / f"{cfg.save_stem}_summary.csv", index=False)
    return out
