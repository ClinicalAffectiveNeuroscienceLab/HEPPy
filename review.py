# heppy/review.py
from __future__ import annotations
from pathlib import Path
import glob
import json
import math
import re
import traceback
from datetime import datetime
import platform

import pandas as pd
import numpy as np
import mne
import neurokit2 as nk  # used for HRV

from config import HEPConfig
from ecg import detect_rpeaks, refine_with_rr_hint, add_rpeak_stim, save_ecg_qc_plot
from epochs import build_hep_epochs, save_hep_epochs


# ------------------------
# Internal logging helpers
# ------------------------

def _ensure_logs_dir(root: Path) -> Path:
    d = root / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _next_error_id(logs_dir: Path) -> int:
    max_id = 0
    for p in logs_dir.glob("error_*_log.txt"):
        m = re.match(r"error_(\d+)_log\.txt$", p.name)
        if m:
            try:
                max_id = max(max_id, int(m.group(1)))
            except ValueError:
                continue
    return max_id + 1

def _write_error_log(
    logs_dir: Path,
    error_id: int,
    stage: str,
    base: str,
    fpath: str,
    cfg: HEPConfig | None,
    row_like: dict | pd.Series | None,
    exc: Exception,
) -> Path:
    log_path = logs_dir / f"error_{error_id:03d}_log.txt"
    try:
        cfg_info = {
            "save_stem": getattr(cfg, "save_stem", None),
            "ecg_channel": getattr(cfg, "ecg_channel", None),
            "qc_csv_name": getattr(cfg, "qc_csv_name", None),
            "qc_dirname": getattr(cfg, "qc_dirname", None),
            "input_glob": getattr(cfg, "input_glob", None),
            "output_root": str(getattr(cfg, "output_root", "")),
            "verbose": getattr(cfg, "verbose", None),
            "do_hrv": getattr(cfg, "do_hrv", None),
        } if cfg is not None else {}
    except Exception:
        cfg_info = {"_cfg_error": "could not serialise cfg"}

    if isinstance(row_like, pd.Series):
        try:
            row_info = row_like.to_dict()
        except Exception:
            row_info = {"_row_error": "could not serialise row"}
    else:
        row_info = dict(row_like) if isinstance(row_like, dict) else {}

    tb_str = traceback.format_exc()

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mne_version": getattr(mne, "__version__", "unknown"),
        "stage": stage,
        "file": fpath,
        "base": base,
        "exception_type": type(exc).__name__,
        "exception_str": repr(exc),
    }

    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=== HEPPy Error Log ===\n")
            for k, v in meta.items():
                f.write(f"{k}: {v}\n")
            f.write("\n--- Config ---\n")
            f.write(json.dumps(cfg_info, indent=2, default=str))
            f.write("\n\n--- Row Context ---\n")
            f.write(json.dumps(row_info, indent=2, default=str))
            f.write("\n\n--- Traceback ---\n")
            f.write(tb_str)
            f.write("\n")
    except Exception:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("FAILED TO WRITE DETAILED LOG; MINIMAL INFO ONLY\n")
            f.write(f"exception_type: {type(exc).__name__}\n")
            f.write(f"exception_str: {repr(exc)}\n")
            f.write("traceback:\n")
            f.write(tb_str)

    return log_path


# ------------------------
# Public API
# ------------------------

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

    On per-file failure, a full traceback is written to output_root/logs/error_###_log.txt
    and the QC row includes a reference to that log.
    """
    files = _iter_inputs(cfg.input_glob)
    rows = []
    qc_dir = cfg.output_root / cfg.qc_dirname
    qc_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = _ensure_logs_dir(cfg.output_root)

    for f in files:
        base = Path(f).stem
        try:
            raw = (
                mne.io.read_raw_edf(f, preload=True, verbose=False)
                if f.lower().endswith(".edf")
                else mne.io.read_raw_fif(f, preload=True, verbose=False)
            )
            if "eeg" in raw.get_channel_types():
                mapping = {ch: ch.replace("EEG ", "").split("-")[0].upper().replace("Z", "z")
                           for ch in raw.ch_names}
                raw.rename_channels(mapping)

            r, rr, ecg_clean = detect_rpeaks(raw, ecg_name=cfg.ecg_channel)
            if ecg_clean.size:
                save_ecg_qc_plot(ecg_clean, raw.info["sfreq"], qc_dir / f"{base}_ecg_qc.png")

            rpeak_cache = {
                "r_peaks": r.tolist() if r.size else [],
                "sampling_freq": float(raw.info["sfreq"]),
                "ecg_channel": cfg.ecg_channel,
            }
            cache_path = qc_dir / f"{base}_rpeaks_cache.json"
            with open(cache_path, "w", encoding="utf-8") as f_cache:
                json.dump(rpeak_cache, f_cache)

            rows.append({
                "file": f,
                "base": base,
                "n_rpeaks_auto": int(len(r)),
                "median_rr_auto": float(np.median(rr)) if rr.size else math.nan,
                "qc_status": "",
                "manual_est_rr_s": "",
                "notes": "",
                "rpeak_cache_path": str(cache_path),
                "error_log": "",
            })

        except Exception as e:
            err_id = _next_error_id(logs_dir)
            log_path = _write_error_log(
                logs_dir=logs_dir, error_id=err_id, stage="make_qc_checklist",
                base=base, fpath=f, cfg=cfg, row_like=None, exc=e,
            )
            rows.append({
                "file": f, "base": base, "n_rpeaks_auto": 0, "median_rr_auto": math.nan,
                "qc_status": "bad", "manual_est_rr_s": "", "notes": f"load/detect err; see log: {log_path.name}",
                "rpeak_cache_path": "", "error_log": str(log_path),
            })

    df = pd.DataFrame(rows)
    out_csv = cfg.output_root / cfg.qc_csv_name
    df.to_csv(out_csv, index=False)
    return df


def apply_qc_and_extract(cfg: HEPConfig, progress_cb=None) -> pd.DataFrame:
    """
    Second pass: read QC CSV and extract HEP. Optionally compute HRV (cfg.do_hrv).
    """
    out_rows = []
    df = pd.read_csv(cfg.output_root / cfg.qc_csv_name)
    outdir = cfg.output_root
    logs_dir = _ensure_logs_dir(cfg.output_root)

    total = len(df)
    done = 0

    for _, row in df.iterrows():
        f = str(row["file"])
        base = str(row["base"])
        status = str(row.get("qc_status", "")).lower().strip()
        rr_hint = row.get("manual_est_rr_s", "")
        rr_hint = float(rr_hint) if str(rr_hint).strip() else math.nan
        if status not in ("ok", "bad", "exclude", "skip", ""):
            status = "ok"

        try:
            raw = (
                mne.io.read_raw_edf(f, preload=True, verbose=False)
                if f.lower().endswith(".edf")
                else mne.io.read_raw_fif(f, preload=True, verbose=False)
            )

            from preprocess import apply_montage, clean_raw
            raw = apply_montage(raw, cfg)
            raw, prov = clean_raw(raw, cfg)
            new_fs = float(raw.info["sfreq"])

            rpeak_cache_path = row.get("rpeak_cache_path", "")
            use_cached = (
                status == "ok"
                and not np.isfinite(rr_hint)
                and rpeak_cache_path
                and Path(rpeak_cache_path).exists()
            )

            if use_cached:
                try:
                    with open(rpeak_cache_path, "r", encoding="utf-8") as f_cache:
                        cache_data = json.load(f_cache)
                    r = np.array(cache_data.get("r_peaks", []), dtype=float)
                    old_fs = float(cache_data.get("sampling_freq", new_fs))
                    if old_fs > 0 and not math.isclose(old_fs, new_fs, rel_tol=1e-6, abs_tol=1e-9):
                        r = np.round(r * (new_fs / old_fs))
                    r = r.astype(int, copy=False)
                    rr = np.diff(r) / new_fs if len(r) >= 2 else np.array([])
                    _, _, ecg_clean = detect_rpeaks(raw, ecg_name=cfg.ecg_channel)
                    if getattr(cfg, "verbose", False):
                        msg_fs = "" if math.isclose(old_fs, new_fs, rel_tol=1e-6, abs_tol=1e-9) \
                                 else f" (rescaled {old_fs:.3f}→{new_fs:.3f} Hz)"
                        print(f"[{base}] Using cached R-peaks ({len(r)} peaks){msg_fs}")
                except Exception as cache_err:
                    if getattr(cfg, "verbose", False):
                        print(f"[{base}] Cache load failed, re-detecting: {cache_err}")
                    use_cached = False

            if not use_cached:
                r, rr, ecg_clean = detect_rpeaks(raw, ecg_name=cfg.ecg_channel)

            corrected_r_peaks = False
            if status in ("bad", "exclude", "skip"):
                if status in ("exclude", "skip") and not np.isfinite(rr_hint):
                    out_rows.append({"file": f, "base": base, "action": "excluded"})
                    done += 1
                    if callable(progress_cb):
                        progress_cb(done, total, base)
                    continue
                if np.isfinite(rr_hint):
                    r_orig = r.copy()
                    r = refine_with_rr_hint(r, new_fs, rr_hint_s=rr_hint, factor=1.8)
                    rr = np.diff(r) / new_fs
                    corrected_r_peaks = True

                    qc_dir = cfg.output_root / cfg.qc_dirname
                    qc_dir.mkdir(parents=True, exist_ok=True)
                    corrected_plot_path = qc_dir / f"{base}_ecg_qc_corrected.png"
                    save_ecg_qc_plot(ecg_clean, new_fs, corrected_plot_path,
                                     r_peaks=r, title_suffix="(RR Corrected)")

                    if getattr(cfg, "verbose", False):
                        print(f"[{base}] Applied RR correction: {len(r_orig)} → {len(r)} R-peaks")
                        print(f"[{base}] Saved corrected ECG plot: {corrected_plot_path}")

            # Optional HRV (requires >= 2 intervals)
            hrv_dict = {}
            if getattr(cfg, "do_hrv", True) and r.size >= 3:
                try:
                    # NeuroKit expects a 'peaks' dict/dataframe keyed by 'ECG_R_Peaks'
                    peaks_payload = {"ECG_R_Peaks": r.astype(int)}
                    hrv_time = nk.hrv_time(peaks=peaks_payload, sampling_rate=new_fs, show=False)
                    hrv_freq = nk.hrv_frequency(peaks=peaks_payload, sampling_rate=new_fs, show=False)
                    hrv_nlin = nk.hrv_nonlinear(peaks=peaks_payload, sampling_rate=new_fs, show=False)
                    hrv_all = pd.concat([hrv_time, hrv_freq, hrv_nlin], axis=1)
                    # single-row dataframe → dict; prefix to avoid name collisions
                    hrv_dict = {f"{k}": float(v) for k, v in hrv_all.iloc[0].items()}
                except Exception as hrv_err:
                    if getattr(cfg, "verbose", False):
                        print(f"[{base}] HRV computation failed: {hrv_err}")

            add_rpeak_stim(raw, r, name="RPEAK")

            hep = build_hep_epochs(raw, r, rr, cfg)
            path = save_hep_epochs(hep, outdir, cfg.save_stem, base)

            out_info = {
                "file": f, "base": base, "action": "saved",
                "epochs_fif": str(path), "n_hep": int(len(hep)),
                "n_rpeaks_used": int(len(r)),
                "median_rr_s": float(np.median(rr)) if rr.size else math.nan,
                "rr_corrected": bool(corrected_r_peaks),
                "used_cache": bool(use_cached if "use_cached" in locals() else False),
                "error_log": "",
            }
            # merge HRV columns if any
            out_info.update(hrv_dict)
            out_rows.append(out_info)

        except Exception as e:
            err_id = _next_error_id(logs_dir)
            log_path = _write_error_log(
                logs_dir=logs_dir, error_id=err_id, stage="apply_qc_and_extract",
                base=base, fpath=f, cfg=cfg, row_like=row, exc=e,
            )
            out_rows.append(
                {"file": f, "base": base, "action": "error",
                 "error": f"check log file: {log_path.name}", "error_log": str(log_path)}
            )

        done += 1
        if callable(progress_cb):
            try:
                progress_cb(done, total, base)
            except Exception:
                pass

    out = pd.DataFrame(out_rows)
    out.to_csv(cfg.output_root / f"{cfg.save_stem}_summary.csv", index=False)
    return out
