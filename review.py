# heppy/review.py
from __future__ import annotations
from pathlib import Path
import glob
import math
import pandas as pd
import numpy as np
import mne
from .config import HEPConfig
from .ecg import detect_rpeaks, refine_with_rr_hint, add_rpeak_stim, save_ecg_qc_plot
from .epochs import build_hep_epochs, save_hep_epochs

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
            rows.append({
                "file": f,
                "base": base,
                "n_rpeaks_auto": int(len(r)),
                "median_rr_auto": float(np.median(rr)) if rr.size else math.nan,
                "qc_status": "",             # fill "ok" | "bad" by hand
                "manual_est_rr_s": "",       # optional (e.g., 0.8)
                "notes": "",
            })
        except Exception as e:
            rows.append({"file": f, "base": base, "n_rpeaks_auto": 0, "median_rr_auto": math.nan,
                         "qc_status": "bad", "manual_est_rr_s": "", "notes": f"load/detect err: {e}"})

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
            # full preprocessing (reuses your stable pipeline) :contentReference[oaicite:7]{index=7}
            from .preprocess import apply_montage, clean_raw
            raw = apply_montage(raw, cfg)
            raw, prov = clean_raw(raw, cfg)

            from .ecg import detect_rpeaks
            r, rr, ecg_clean = detect_rpeaks(raw, ecg_name=cfg.ecg_channel)

            if status in ("bad", "exclude", "skip"):
                if status in ("exclude", "skip") and not np.isfinite(rr_hint):
                    out_rows.append({"file": f, "base": base, "action": "excluded"})
                    continue
                if np.isfinite(rr_hint):
                    r = refine_with_rr_hint(r, raw.info["sfreq"], rr_hint_s=rr_hint, factor=1.8)
                    rr = np.diff(r) / raw.info["sfreq"]

            # add a stim channel for convenience (optional, helps viewing) :contentReference[oaicite:8]{index=8}
            add_rpeak_stim(raw, r, name="RPEAK")

            from .epochs import build_hep_epochs, save_hep_epochs
            hep = build_hep_epochs(raw, r, rr, cfg)
            path = save_hep_epochs(hep, outdir, cfg.save_stem, base)
            out_rows.append({"file": f, "base": base, "action": "saved", "epochs_fif": str(path),
                             "n_hep": int(len(hep))})
        except Exception as e:
            out_rows.append({"file": f, "base": base, "action": "error", "error": str(e)})

    out = pd.DataFrame(out_rows)
    out.to_csv(cfg.output_root / f"{cfg.save_stem}_summary.csv", index=False)
    return out
