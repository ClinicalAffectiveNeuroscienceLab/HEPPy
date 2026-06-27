#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HEPPy Streamlit UI
"""

from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import mne
import neurokit2 as nk
import logging

from ecg_peak_correction import (
    adaptive_refit_excluded_rpeaks,
    corrected_rpeaks_path,
    delete_excluded_rpeaks,
    intersected_trace_indices,
    load_corrected_rpeaks,
    save_corrected_rpeaks,
    selected_box_ranges,
    selected_peak_indices,
)

# --------- Try to pull sensible defaults from your config.py -------------
def _load_defaults():
    defaults = dict(  # keep same keys
        input_dir="", output_dir="", file_glob="**/*.edf;**/*.bdf;**/*.fif;**/*.vhdr",
        ecg_channel=None, spirometry_channel=None, hp=0.5, lp=45.0,
        run_pyprep=True, run_asr=False, run_ica=True, redo_preprocessing=False,
        remove_cfa=True, remove_cfa_mode="remove",  # ICA cardiac handling
        rr_min_bpm=35, rr_max_bpm=140, rr_outlier_mad=3.5, bpm_smooth_win=5,
        epoch_tmin=-0.2, epoch_tmax=0.6, baseline=None,
        amp_reject_uv=150.0, rr_z_reject=3.5,
        export_fif=True, export_eeglab=False,
    )
    try:
        from _configuration_handler import load_from_config_hep
        # Prefer the new example config name; fall back to old `config_hep` to remain compatible.
        try:
            cfg = load_from_config_hep("example_config")
        except Exception:
            cfg = load_from_config_hep("config_hep")
        defaults["input_dir"]  = ""               # leave blank to choose in UI
        defaults["output_dir"] = str(cfg.output_root)
        # map a few knobs:
        defaults["hp"] = float(getattr(cfg, "high_pass", 0.5))
        defaults["lp"] = float(getattr(cfg, "low_pass", 45.0))
        defaults["rr_min_bpm"] = 35
        defaults["rr_max_bpm"] = 140
        defaults["epoch_tmin"], defaults["epoch_tmax"] = cfg.tmin, cfg.tmax
        defaults["baseline"] = cfg.baseline
        defaults["amp_reject_uv"] = cfg.amp_rej_uv
        # some prep params
        defaults["target_sfreq"] = getattr(cfg, "target_sfreq", None)
        defaults["prep_ransac"] = getattr(cfg, "prep_ransac", True)
        defaults["line_freqs"] = getattr(cfg, "line_freqs", (50.0, 100.0))
        defaults["ref_chs"] = getattr(cfg, "ref_chs", "eeg")
        defaults["reref_chs"] = getattr(cfg, "reref_chs", "eeg")
        # reference electrodes for epoching (optional list of channel names)
        defaults["reference_electrodes"] = getattr(cfg, "reference_electrodes", None)
    except Exception:
        pass
    return defaults


D = _load_defaults()

# --- Helper: load user Python config file and return module + dict ---


# Try to load a default config module (config_hep) and apply it to preprocessing globals
try:
    from _configuration_handler import load_from_config_hep
    # Prefer the new example_config name; fall back to legacy config_hep if present
    try:
        default_cfg = load_from_config_hep("example_config")
    except Exception:
        default_cfg = load_from_config_hep("config_hep")
    import _preprocessing as _preproc
    # apply defaults to _preprocessing (so prep_params exist early)
    try:
        _preproc.set_runtime_config(default_cfg)
        from dataclasses import asdict
        st.session_state.setdefault("runtime_config", asdict(default_cfg))
    except Exception:
        # not fatal; continue with D defaults
        pass
except Exception:
    default_cfg = None

# ----------------- Minimal filesystem helpers ---------------------------
def _split_globs(glob_str: str) -> list[str]:
    return [g.strip() for g in glob_str.split(";") if g.strip()]

def _find_raws(indir: Path, glob_str: str) -> list[Path]:
    files = []
    for pat in _split_globs(glob_str):
        files += list(indir.rglob(pat))
    # hide derivatives
    files = [p for p in files if "_pp_raw.fif" not in p.name and "-epo.fif" not in p.name]
    return sorted(files)

def _read_raw_any(path: Path, *, preload=True, verbose="ERROR") -> mne.io.BaseRaw:
    suffix = path.suffix.lower()
    if suffix == ".bdf":
        return mne.io.read_raw_bdf(str(path), preload=preload, verbose=verbose)
    if suffix == ".edf":
        return mne.io.read_raw_edf(str(path), preload=preload, verbose=verbose)
    if suffix == ".fif":
        return mne.io.read_raw_fif(str(path), preload=preload, verbose=verbose)
    if suffix == ".vhdr":
        return mne.io.read_raw_brainvision(str(path), preload=preload, verbose=verbose)
    raise ValueError(f"Unsupported file type: {path}")

# ----------------- Preprocessing glue -----------------------------------
def _preprocess_one(
    path: Path,
    out_dir: Path,
    qc_dir: Path,
    params: dict,
    ecg_channel: str | None,
    spirometry_channel: str | None,
    hp: float,
    lp: float,
    redo: bool = False,
    remove_cfa_mode: str = "remove",
    flip_ecg: bool = False,
    stim_keep: list[str] | None = None,
) -> dict | None:
    """Preprocess one raw file and write one or two *_pp_raw.fif outputs.

    For remove_cfa_mode == "both", the expensive part of preprocessing (read → resample →
    filtering → PyPREP → ASR → ICA fit + ICLabel) is executed once. Two outputs are then
    produced by applying the fitted ICA with different exclude-sets (with vs without
    the heartbeat-labelled components).
    """
    from _preprocessing import preprocess_edf, preprocess_edf_both

    out_dir.mkdir(parents=True, exist_ok=True)
    base = path.stem
    outputs: dict[str, Path] = {}

    log_path = str(out_dir.parent / "logs" / "preprocessing.tsv")

    mode = str(remove_cfa_mode).lower().strip()
    if mode == "both":
        out_remove = out_dir / f"{base}_pp_raw.fif"
        out_keep = out_dir / f"{base}_pp_raw_keepcfa.fif"
        # run common steps once, split after ICA/ICLabel
        raw_remove, raw_keep = preprocess_edf_both(
            str(path),
            output_path_remove=str(out_remove),
            output_path_keep=str(out_keep),
            redo=bool(redo),
            flip_ecg=bool(flip_ecg),
            ecg_channel=ecg_channel,
            stim_keep=stim_keep,
            spirometry_channel=spirometry_channel,
            spirometry_dir=out_dir.parent / "spirometry",
            logging_path=log_path,
        )
        # QC PNG: use the "remove" variant for determinism
        if not (qc_dir / f"{base}_ecg_qc.png").exists() or bool(redo):
            _save_ecg_qc(raw_remove, base, qc_dir, ecg_channel, params, hp, lp, redo=bool(redo))
        outputs["remove"] = out_remove
        outputs["keep"] = out_keep
        return outputs

    # single-branch behaviour (unchanged)
    remove_flag = (mode != "keep")
    suffix = "" if remove_flag else "_keepcfa"
    out_fif = out_dir / f"{base}_pp_raw{suffix}.fif"
    raw_local = preprocess_edf(
        str(path),
        output_path=str(out_fif),
        redo=bool(redo),
        remove_cfa_override=remove_flag,
        flip_ecg=bool(flip_ecg),
        ecg_channel=ecg_channel,
        stim_keep=stim_keep,
        spirometry_channel=spirometry_channel,
        spirometry_dir=out_dir.parent / "spirometry",
        logging_path=log_path,
    )
    if not (qc_dir / f"{base}_ecg_qc.png").exists() or bool(redo):
        _save_ecg_qc(raw_local, base, qc_dir, ecg_channel, params, hp, lp, redo=bool(redo))
    outputs[mode] = out_fif
    return outputs

# ----------------- ECG / R-peak detection + adaptive correction ---------
def _auto_pick_ecg(raw: mne.io.BaseRaw, preferred: str|None) -> str:
    if preferred and preferred in raw.ch_names:
        return preferred
    idx = mne.pick_types(raw.info, ecg=True)
    if len(idx) > 0:
        return raw.ch_names[idx[0]]
    # try by name
    for name in raw.ch_names:
        if any(tok in name.upper() for tok in ("ECG", "EKG", "CARD")):
            return name
    # last resort: strongest single-channel correlation with detected peaks
    raise RuntimeError("No ECG channel found. Please set 'ECG channel' in the sidebar.")

def _mark_ecg_channels(raw: mne.io.BaseRaw, preferred: str | None = None) -> list[str]:
    names = []
    if preferred and preferred in raw.ch_names:
        names.append(preferred)
    elif preferred:
        wanted = preferred.casefold()
        names.extend(ch for ch in raw.ch_names if ch.casefold() == wanted)
    for ch in raw.ch_names:
        if any(tok in ch.upper() for tok in ("ECG", "EKG", "CARD")) and ch not in names:
            names.append(ch)
    for ch in names:
        try:
            raw.set_channel_types({ch: "ecg"})
        except Exception:
            pass
    return names

def _flip_ecg_channels(raw: mne.io.BaseRaw, preferred: str | None = None) -> list[str]:
    names = _mark_ecg_channels(raw, preferred)
    if not names:
        names = [raw.ch_names[i] for i in mne.pick_types(raw.info, ecg=True)]
    idx = [raw.ch_names.index(ch) for ch in names if ch in raw.ch_names]
    if idx:
        raw._data[idx] *= -1
    return [raw.ch_names[i] for i in idx]

def _detect_rpeaks(sig, sf):
    # neurokit robust pipeline
    ecg_proc = nk.ecg_process(ecg_signal=sig, sampling_rate=sf)
    rpeaks = ecg_proc[1]["ECG_R_Peaks"]
    return np.asarray(rpeaks, dtype=int), ecg_proc

def _adaptive_fix_rpeaks(rpeaks: np.ndarray, sf: float,
                         rr_min_bpm: float, rr_max_bpm: float,
                         rr_outlier_mad: float, bpm_smooth_win: int) -> np.ndarray:
    """Robustly fix missed / spurious peaks by RR constraints + MAD outliers + local interpolation."""
    if rpeaks.size < 3:
        return rpeaks
    rr = np.diff(rpeaks) / sf
    bpm = 60.0 / rr
    # smooth BPM (moving median)
    if bpm_smooth_win > 1 and bpm.size >= bpm_smooth_win:
        bpm_sm = pd.Series(bpm).rolling(bpm_smooth_win, center=True).median().to_numpy()
        bpm = np.where(np.isnan(bpm_sm), bpm, bpm_sm)
    # remove segments with implausible BPM
    mask_ok = (bpm >= rr_min_bpm) & (bpm <= rr_max_bpm)
    # MAD outliers on RR
    rr_med = np.median(rr)
    mad = np.median(np.abs(rr - rr_med)) + 1e-12
    z = 0.6745 * (rr - rr_med) / mad
    mask_ok = mask_ok & (np.abs(z) <= rr_outlier_mad)

    kept = [rpeaks[0]]
    for i, ok in enumerate(mask_ok):
        if ok:
            kept.append(rpeaks[i+1])
        else:
            # interpolate a plausible beat (one or more) between kept[-1] and rpeaks[i+1]
            gap = (rpeaks[i+1] - kept[-1]) / sf
            n_insert = max(1, int(round(gap / rr_med)) - 1)
            for k in range(n_insert):
                kept.append(int(round(kept[-1] + rr_med*sf)))
            kept.append(rpeaks[i+1])
    kept = np.unique(np.asarray(kept, dtype=int))
    # final consistency: enforce monotonicity & min distance
    min_dist = int(round((60.0/rr_max_bpm) * sf))
    dedup = [kept[0]]
    for rp in kept[1:]:
        if rp - dedup[-1] >= min_dist:
            dedup.append(rp)
    return np.asarray(dedup, dtype=int)

def _plot_ecg_panels(signals, info, title="ECG QC"):
    # neurokit's plot already shows multiple panels; we’ll also add a zoomed QRS inset (top-right like you requested).
    nk.ecg_plot(signals, info)
    fig = plt.gcf()
    fig.set_size_inches(12, 10)
    fig.suptitle(title)
    return fig

def _render_ecg_qc(raw, ecg_name, params):
    sf = float(raw.info["sfreq"])
    sig = raw.get_data(picks=ecg_name)[0]
    rpeaks, (signals, info) = None, (None, None)
    try:
        rpeaks, proc = _detect_rpeaks(sig, sf)
        signals, info = proc
    except Exception:
        # fallback
        cleaned = nk.ecg_clean(sig, sampling_rate=sf)
        plt.figure(figsize=(12,3))
        plt.plot(np.arange(cleaned.size)/sf, cleaned, linewidth=0.5)
        plt.title("ECG (cleaned) — fallback QC")
        plt.tight_layout()
        return plt.gcf(), np.array([], dtype=int)

    rpeaks_fixed = _adaptive_fix_rpeaks(
        rpeaks=rpeaks, sf=sf,
        rr_min_bpm=params["rr_min_bpm"], rr_max_bpm=params["rr_max_bpm"],
        rr_outlier_mad=params["rr_outlier_mad"], bpm_smooth_win=params["bpm_smooth_win"]
    )
    fig = _plot_ecg_panels(signals, info, title=f"ECG QC — {ecg_name}")
    # overlay fixed peaks on first panel if present
    try:
        ax0 = fig.axes[0]
        ax0.scatter(rpeaks_fixed/sf, signals["ECG_Clean"][rpeaks_fixed], s=10)
    except Exception:
        pass
    return fig, rpeaks_fixed

def _rpeaks_for_raw(raw: mne.io.BaseRaw, base: str, output_dir: str | Path,
                    ecg_name: str, params: dict) -> tuple[np.ndarray, str]:
    sf = float(raw.info["sfreq"])
    qc_dir = Path(output_dir) / "ecg_qc"
    saved = load_corrected_rpeaks(corrected_rpeaks_path(qc_dir, base), sf, raw.n_times)
    if saved is not None and len(saved) >= 3:
        return saved, "manual"
    sig = raw.get_data(picks=ecg_name)[0]
    rpeaks, _ = _detect_rpeaks(sig, sf)
    return _adaptive_fix_rpeaks(
        rpeaks, sf,
        params["rr_min_bpm"], params["rr_max_bpm"],
        params["rr_outlier_mad"], params["bpm_smooth_win"]
    ), "detected"

def _qrs_review_source(row: pd.Series) -> Path | None:
    for value in (row.get("edf_path", ""), row.get("raw_fif", ""), row.get("raw_fif_keep", "")):
        text = str(value or "").strip()
        if text and Path(text).exists():
            return Path(text)
    return None

def _load_qrs_review_data(src: Path, base: str, output_dir: str | Path,
                          ecg_channel: str | None, params: dict,
                          hp: float, lp: float, flip_ecg: bool = False) -> tuple[np.ndarray, float, np.ndarray, str]:
    raw = raw_filt = None
    try:
        raw = _load_raw_minimal_for_ecg(src, ecg_channel, flip_ecg=flip_ecg)
        ecg_name = _auto_pick_ecg(raw, ecg_channel if ecg_channel else None)
        raw_filt = raw.copy().filter(
            l_freq=hp if hp > 0 else None,
            h_freq=lp,
            picks=[ecg_name],
            verbose="ERROR"
        )
        sig = raw_filt.get_data(picks=ecg_name)[0]
        rpeaks, source = _rpeaks_for_raw(raw_filt, base, output_dir, ecg_name, params)
        return sig, float(raw_filt.info["sfreq"]), rpeaks, source
    finally:
        for obj in (raw_filt, raw):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass

def _qrs_wave_lookup(sig: np.ndarray, sf: float, rpeaks: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    try:
        clean = nk.ecg_clean(sig, sampling_rate=sf)
    except Exception:
        clean = np.asarray(sig, dtype=float)
    left = int(round(-0.2 * sf))
    right = int(round(0.35 * sf))
    x = np.arange(left, right) / float(sf)
    waves = {}
    for idx, peak in enumerate(np.asarray(rpeaks, dtype=int)):
        start, stop = int(peak) + left, int(peak) + right
        if start < 0 or stop > len(clean):
            continue
        waves[idx] = clean[start:stop]
    return x, waves

def _make_qrs_selection_plot(x: np.ndarray, waves: dict[int, np.ndarray],
                             selected: list[int] | None = None):
    import plotly.graph_objects as go

    selected = set(selected or [])
    fig = go.Figure()
    wave_values = []
    for idx, y in waves.items():
        wave_values.append(y)
        picked = idx in selected
        fig.add_trace(go.Scattergl(
            x=x, y=y, mode="lines", showlegend=False, hoverinfo="skip",
            customdata=np.full((len(x), 1), idx),
            line=dict(
                color="rgba(255, 76, 61, 0.55)" if picked else "rgba(80, 80, 80, 0.10)",
                width=1.8 if picked else 0.6,
            ),
        ))
    if wave_values:
        avg = np.nanmean(np.vstack(wave_values), axis=0)
        fig.add_trace(go.Scatter(
            x=x, y=avg, mode="lines", name="Average beat",
            customdata=np.full((len(x), 1), -1),
            line=dict(color="#ff4b3e", width=4),
        ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        height=650,
        margin=dict(l=20, r=20, t=35, b=35),
        dragmode="select",
        selectdirection="any",
        xaxis_title="Time (seconds)",
        yaxis_title="ECG",
        title="QRS correction",
        legend=dict(orientation="h"),
    )
    return fig

def _save_ecg_qc(raw: mne.io.BaseRaw, base: str, qc_dir: Path,
                 ecg_channel: str | None, params: dict,
                 hp: float, lp: float, redo: bool = False) -> Path | None:
    """Filter ECG, render QC, and persist a PNG for quick reloads."""
    qc_dir.mkdir(parents=True, exist_ok=True)
    png_path = qc_dir / f"{base}_ecg_qc.png"
    if png_path.exists() and not redo:
        return png_path

    raw_filt = None
    try:
        ecg_name = _auto_pick_ecg(raw, ecg_channel if ecg_channel else None)
        raw_filt = raw.copy().filter(
            l_freq=hp if hp > 0 else None,
            h_freq=lp,
            picks=[ecg_name],
            verbose="ERROR"
        )
        fig, _ = _render_ecg_qc(raw_filt, ecg_name, params)
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return png_path
    except Exception as e:
        try:
            plt.close("all")
        except Exception:
            pass
        st.warning(f"ECG QC render failed for {base}: {e}")
        return None
    finally:
        try:
            if raw_filt is not None:
                raw_filt.close()
        except Exception:
            pass

def _load_raw_minimal_for_ecg(path: Path, ecg_channel: str | None = None, flip_ecg: bool = False) -> mne.io.BaseRaw:
    """Lightweight loader for ECG QC before EEG preprocessing."""
    raw = _read_raw_any(path)
    _mark_ecg_channels(raw, ecg_channel)
    if flip_ecg:
        _flip_ecg_channels(raw, ecg_channel)
    return raw

def _hrv_metrics_from_rpeaks(rpeaks: np.ndarray, sf: float,
                             want_time=True, want_freq=True, want_nl=False) -> dict:
    out = {}
    if rpeaks is None or len(rpeaks) < 3:
        return out
    try:
        import neurokit2 as nk
        rdict = {"ECG_R_Peaks": np.asarray(rpeaks, dtype=int)}
        if want_time:
            out.update(nk.hrv_time(rpeaks=rdict, sampling_rate=sf, show=False).iloc[0].to_dict())
        if want_freq:
            out.update(nk.hrv_frequency(rpeaks=rdict, sampling_rate=sf, show=False).iloc[0].to_dict())
        if want_nl:
            out.update(nk.hrv_nonlinear(rpeaks=rdict, sampling_rate=sf, show=False).iloc[0].to_dict())
    except Exception:
        rr = np.diff(rpeaks) / float(sf)
        if rr.size >= 2:
            diff = np.diff(rr)
            out.update({
                "HRV_MeanNN": float(np.mean(rr) * 1000.0),
                "HRV_SDNN": float(np.std(rr, ddof=1) * 1000.0),
                "HRV_RMSSD": float(np.sqrt(np.mean(diff**2)) * 1000.0),
                "HRV_pNN50": float(np.mean((np.abs(diff) > 0.05)) * 100.0),
                "HR_Mean": float(60.0 / np.mean(rr)),
            })
    return out

def _write_hrv_rows(rows: list[dict], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows)
    if out_csv.exists():
        df_old = pd.read_csv(out_csv)
        df = pd.concat([df_old, df_new], ignore_index=True)
        if "base" in df.columns:
            df = df.drop_duplicates(subset=["base"], keep="last")
    else:
        df = df_new
    df.to_csv(out_csv, index=False)


# ----------------- QRS Review state / CSV --------------------------------
CSV_NAME = "ecg_qc_review.csv"

REVIEW_COLS = ["base","edf_path","raw_fif","raw_fif_keep","qc_png","viable",
               "true_rr_s","true_hr_bpm","beats_per_gap","notes","flip"]

def _ensure_review_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in REVIEW_COLS:
        if col not in df.columns:
            default_val = False if col == "flip" else ""
            df[col] = default_val
    # enforce types to avoid pandas implicit float columns
    if "viable" in df.columns:
        df["viable"] = df["viable"].fillna("").astype(str)
    if "flip" in df.columns:
        df["flip"] = df["flip"].map(lambda v: str(v).strip().lower() in ("true", "1", "yes"))
    return df

def _resolve_qc_png(row: pd.Series, output_dir: Path) -> Path:
    """Return best-guess QC PNG path for a review row."""
    base = str(row.base)
    candidates = []
    # 1) stored path (absolute or relative)
    if isinstance(row.qc_png, str) and row.qc_png.strip():
        p = Path(row.qc_png)
        candidates.append(p if p.is_absolute() else output_dir / p)
    # 2) conventional location under current output_dir
    candidates.append(output_dir / "ecg_qc" / f"{base}_ecg_qc.png")
    # 3) any matching png under ecg_qc
    candidates.extend(sorted((output_dir / "ecg_qc").glob(f"{base}*ecg_qc*.png")))
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]

def _display_png(path: Path, caption: str):
    """Robustly display a PNG in Streamlit; fall back to byte read to avoid path issues."""
    try:
        import io
        data = path.read_bytes()
        st.image(io.BytesIO(data), caption=caption, use_column_width=True)
    except Exception:
        st.image(str(path), caption=caption, use_column_width=True)

def _seed_review_from_raws(raw_paths: list[str], output_dir: Path, qc_dir: Path) -> pd.DataFrame:
    csv_path = output_dir / CSV_NAME
    df = _ensure_review_columns(_load_review(csv_path))
    have = set(df["base"].astype(str))
    rows = []
    for rp in raw_paths:
        base = Path(rp).stem
        if base not in have:
            rows.append(dict(
                base=base,
                edf_path=str(rp),
                raw_fif=str(output_dir / "raw_fif" / f"{base}_pp_raw.fif"),
                raw_fif_keep=str(output_dir / "raw_fif" / f"{base}_pp_raw_keepcfa.fif"),
                qc_png=str(qc_dir / f"{base}_ecg_qc.png"),
                viable="",
                true_rr_s="",
                true_hr_bpm="",
                beats_per_gap="",
                notes="",
                flip=False,
            ))
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    _save_review(df, csv_path)
    return df

def _load_review(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=REVIEW_COLS)
    return _ensure_review_columns(df)

def _save_review(df: pd.DataFrame, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

def _append_missing_bases(df: pd.DataFrame, raws: list[Path], qc_dir: Path) -> pd.DataFrame:
    df = _ensure_review_columns(df)
    have = set(df["base"].astype(str))
    rows = []
    for rf in raws:
        stem = Path(rf).stem
        is_keep = stem.endswith("_pp_raw_keepcfa") or stem.endswith("_raw_keepcfa")
        base = (
            stem
            .replace("_pp_raw_keepcfa", "")
            .replace("_pp_raw", "")
            .replace("_raw_keepcfa", "")
            .replace("_raw", "")
        )
        if base not in have:
            raw_fif = "" if is_keep else str(rf)
            raw_fif_keep = str(rf) if is_keep else str(Path(rf).with_name(Path(rf).stem + "_keepcfa.fif"))
            rows.append(dict(
                base=base, edf_path="",
                raw_fif=raw_fif,
                raw_fif_keep=raw_fif_keep,
                qc_png=str(qc_dir/ f"{base}_ecg_qc.png"),
                viable="", true_rr_s="", true_hr_bpm="", beats_per_gap="", notes="", flip=False
            ))
        else:
            # fill missing raw_fif paths for existing entries
            ix = df.index[df["base"].astype(str) == base]
            if len(ix):
                target_col = "raw_fif_keep" if is_keep else "raw_fif"
                current = df.at[ix[0], target_col]
                if pd.isna(current) or not str(current).strip():
                    df.at[ix[0], target_col] = str(rf)
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return df

# ----------------- Epoch export ------------------------------------------
def _epoch_from_rpeaks(raw: mne.io.BaseRaw, rpeaks: np.ndarray, tmin: float, tmax: float,
                       baseline, rr_z_reject: float, amp_reject_uv: float,
                       reference_electrodes: list | None,
                       spirometry_channel: str | None = None) -> mne.Epochs:
    sf = raw.info["sfreq"]
    events = np.column_stack([rpeaks, np.zeros_like(rpeaks), np.ones_like(rpeaks, dtype=int)])
    # RR-based rejection
    if rpeaks.size >= 3 and rr_z_reject is not None:
        rr = np.diff(rpeaks)/sf
        z = (rr - np.mean(rr))/ (np.std(rr)+1e-9)
        bad_idx = np.where(np.abs(z) > rr_z_reject)[0]
        # drop epochs whose *onset* RR is bad (index aligns to event i+1)
        keep_mask = np.ones(len(events), dtype=bool)
        keep_mask[bad_idx+1] = False
        events = events[keep_mask]

    event_id = dict(HEP=1)
    metadata = None
    if spirometry_channel:
        try:
            from _preprocessing import spirometry_phase_at_samples
            metadata = spirometry_phase_at_samples(raw, events[:, 0])
            if metadata is not None:
                code_map = {"inspiratory": 1, "expiratory": 2, "unknown": 3}
                labels = metadata["resp_phase_label"].astype(str).to_numpy()
                if len(labels):
                    events[:, 2] = [code_map.get(label, 3) for label in labels]
                    event_id = {f"HEP/{label}": code for label, code in code_map.items() if label in set(labels)}
        except Exception:
            metadata = None
            event_id = dict(HEP=1)
            events[:, 2] = 1

    reject = dict(eeg=amp_reject_uv*1e-6) if amp_reject_uv else None
    picks_eeg = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, ecg=False, stim=False)
    # Apply hard reference if provided
    try:
        if reference_electrodes:
            raw.set_eeg_reference(reference_electrodes, projection=True)
    except Exception:
        # non-fatal; continue with current reference
        pass
    epochs = mne.Epochs(raw, events=events, event_id=event_id,
                        tmin=tmin, tmax=tmax, baseline=baseline, proj=True,
                        picks=picks_eeg, preload=True, reject=reject,
                        metadata=metadata, on_missing="ignore")
    return epochs

def _export_epochs(epochs: mne.Epochs, out_base: Path, export_fif=True, export_eeglab=False):
    saved = []
    if export_fif:
        f = out_base.parent / f"{out_base.name}_epo.fif"
        epochs.save(str(f), overwrite=True)
        saved.append(f)
    if export_eeglab:
        try:
            import eeglabio      # pip install eeglabio
            from eeglabio.utils import export_mne_epochs
            f = out_base.with_suffix(".set")
            export_mne_epochs(epochs, str(f))
            saved.append(f)
        except Exception as e:
            st.warning(f"Could not write EEGLAB .set (install eeglabio). Error: {e}.")
    return saved

def _merge_runtime_config(sidebar_vals: dict | None, loaded_cfg: dict | None, default_cfg_obj=None) -> dict:
    """Merge configs with priority: sidebar_vals > loaded_cfg > default_cfg_obj > D defaults.

    Returns a plain dict suitable for applying to preprocessing.
    """
    merged = {}
    # start with static defaults D
    merged.update(D)
    # default_cfg_obj may be a dataclass (HEPConfig) or module-like
    if default_cfg_obj is not None:
        try:
            from dataclasses import asdict
            merged.update(asdict(default_cfg_obj))
        except Exception:
            # fallback: copy common attributes
            for k in ("output_root","target_sfreq","use_asr","line_freqs","high_pass","low_pass","ref_chs","reref_chs","prep_ransac","montage_name","rename_to_1020"):
                if hasattr(default_cfg_obj, k):
                    merged[k] = getattr(default_cfg_obj, k)
    # then loaded config dict
    if loaded_cfg:
        merged.update(loaded_cfg)
    # finally sidebar overrides
    if sidebar_vals:
        merged.update(sidebar_vals)
    return merged


def _apply_to_preproc(merged: dict):
    """Create a simple namespace with fields expected by _preprocessing.set_runtime_config and call it.

    This binds prep_params, output_dir, target_sfreq etc in the _preprocessing module.
    """
    try:
        import _preprocessing as _preproc
    except Exception:
        st.warning("_preprocessing module not importable; skipping prep_params application.")
        return

    ns = SimpleNamespace()
    # output_root expected as Path in _preprocessing
    out_root = merged.get("output_root", merged.get("output_dir", None))
    ns.output_root = Path(out_root) if out_root is not None else Path(".")
    ns.target_sfreq = merged.get("target_sfreq", None)
    # use_asr / remove_cfa may have different names
    ns.use_asr = merged.get("run_asr", merged.get("use_asr", None))
    ns.remove_cfa = merged.get("remove_cfa", merged.get("remove_cfa_mode", "remove") != "keep")
    ns.remove_cfa_mode = merged.get("remove_cfa_mode", "remove")
    ns.use_pyprep = merged.get("run_pyprep", merged.get("use_pyprep", True))
    ns.use_ica = merged.get("run_ica", merged.get("use_ica", True))
    ns.log_file = merged.get("log_file", None)
    ns.ref_chs = merged.get("ref_chs", merged.get("reref_chs", "eeg"))
    ns.reref_chs = merged.get("reref_chs", merged.get("ref_chs", "eeg"))
    ns.high_pass = merged.get("hp", merged.get("high_pass", D.get("hp")))
    ns.low_pass = merged.get("lp", merged.get("low_pass", D.get("lp")))
    ns.prep_ransac = merged.get("prep_ransac", True)
    ns.line_freqs = merged.get("line_freqs", (50.0, 100.0))
    ns.montage_name = merged.get("montage_name", None)
    ns.rename_to_1020 = merged.get("rename_to_1020", True)
    ns.spirometry_channel = merged.get("spirometry_channel", None)

    try:
        _preproc.set_runtime_config(ns)
        st.session_state["preproc_runtime_applied"] = True
    except Exception as e:
        st.error(f"Applying runtime config to preprocessing failed: {e}")

# ============================= UI ========================================
st.set_page_config(page_title="HEPPy — ECG QC → Preprocess → Epochs", layout="wide")

st.title("HEPPy: ECG QC → EEG preprocessing → HEP epochs")

with st.sidebar:
    st.header("Defaults")
    st.caption("Defaults come from config.py if available; you can adjust here and it will not overwrite files.")

    # ---- Optional: load previous run options (options_used.json) ----
    options_json_path = st.text_input(
        "Options JSON path (optional, options_used.json)",
        value="",
        help="Point to a previous run's options_used.json (or a directory containing it) to prefill the GUI."
    )
    if options_json_path:
        try:
            import json
            p = Path(options_json_path)
            if p.exists() and p.is_dir():
                p = p / "options_used.json"
            opts_data = json.loads(p.read_text(encoding="utf-8"))

            # Support newer files that wrap settings in {"runtime_config": {...}}
            if isinstance(opts_data, dict) and isinstance(opts_data.get("runtime_config", None), dict):
                rc = opts_data["runtime_config"]
                # If rc already looks like a flat config (contains hp/lp etc.), use it directly.
                if any(k in rc for k in ("hp", "lp", "line_freqs", "remove_cfa_mode", "cfa_mode")):
                    opts_data = rc
                else:
                    # Otherwise treat rc as the root for sectioned configs.
                    opts_data = rc

            mapped: dict = {}

            filt = opts_data.get("filters", opts_data.get("filter", {})) or {}
            if "hp" in filt: mapped["hp"] = filt.get("hp")
            if "lp" in filt: mapped["lp"] = filt.get("lp")
            lf = filt.get("line_freqs", filt.get("line_freq", None))
            if lf is not None:
                mapped["line_freqs"] = tuple(lf) if isinstance(lf, (list, tuple)) else lf

            mont = opts_data.get("montage", {}) or {}
            if "name" in mont: mapped["montage_name"] = mont.get("name")
            if "rename_to_1020" in mont: mapped["rename_to_1020"] = bool(mont.get("rename_to_1020", True))

            mapped["run_pyprep"] = bool(opts_data.get("pyprep", opts_data.get("run_pyprep", True)))
            mapped["run_asr"] = bool(opts_data.get("asr", opts_data.get("run_asr", False)))
            mapped["run_ica"] = bool(opts_data.get("ica_iclabel", opts_data.get("run_ica", True)))

            cfa_mode = opts_data.get("remove_cfa_mode", opts_data.get("cfa_mode", opts_data.get("cfaMode", "remove")))
            mapped["remove_cfa_mode"] = str(cfa_mode).lower().strip()

            rr = opts_data.get("adaptive_rr", opts_data.get("rr_params", {})) or {}
            if "rr_min_bpm" in rr or "min_bpm" in rr: mapped["rr_min_bpm"] = rr.get("rr_min_bpm", rr.get("min_bpm"))
            if "rr_max_bpm" in rr or "max_bpm" in rr: mapped["rr_max_bpm"] = rr.get("rr_max_bpm", rr.get("max_bpm"))
            if "rr_outlier_mad" in rr: mapped["rr_outlier_mad"] = rr.get("rr_outlier_mad")
            if "bpm_smooth_win" in rr: mapped["bpm_smooth_win"] = rr.get("bpm_smooth_win")

            epoch = opts_data.get("epoch", {}) or {}
            if "tmin" in epoch: mapped["epoch_tmin"] = epoch.get("tmin")
            if "tmax" in epoch: mapped["epoch_tmax"] = epoch.get("tmax")
            bl = epoch.get("baseline", None)
            if isinstance(bl, (list, tuple)) and len(bl) == 2:
                mapped["baseline_tmin"], mapped["baseline_tmax"] = bl[0], bl[1]
            if "amp_reject_uv" in epoch: mapped["amp_reject_uv"] = epoch.get("amp_reject_uv")
            if "rr_z_reject" in epoch: mapped["rr_z_reject"] = epoch.get("rr_z_reject")

            export = opts_data.get("export", {}) or {}
            if "fif" in export: mapped["export_fif"] = bool(export.get("fif", True))
            if "eeglab" in export: mapped["export_eeglab"] = bool(export.get("eeglab", False))

            hrv = opts_data.get("hrv", {}) or {}
            if "compute" in hrv: mapped["compute_hrv"] = bool(hrv.get("compute", True))
            if "time_domain" in hrv: mapped["hrv_time"] = bool(hrv.get("time_domain", True))
            if "frequency_domain" in hrv: mapped["hrv_freq"] = bool(hrv.get("frequency_domain", True))
            if "nonlinear" in hrv: mapped["hrv_nonlinear"] = bool(hrv.get("nonlinear", False))

            ecg_opts = opts_data.get("ecg", {}) or {}
            if "flip_on_load" in ecg_opts:
                mapped["flip_ecg_on_load"] = bool(ecg_opts.get("flip_on_load", False))
            if "flip_ecg_on_load" in opts_data:
                mapped["flip_ecg_on_load"] = bool(opts_data.get("flip_ecg_on_load", False))

            spiro = opts_data.get("spirometry", {}) or {}
            if "channel" in spiro:
                mapped["spirometry_channel"] = spiro.get("channel")
            if "spirometry_channel" in opts_data:
                mapped["spirometry_channel"] = opts_data.get("spirometry_channel")

            if "stim_keep" in opts_data:
                mapped["stim_keep"] = opts_data.get("stim_keep")
            elif "stim_keep_list" in opts_data:
                mapped["stim_keep"] = opts_data.get("stim_keep_list")

            st.session_state["runtime_config"] = mapped
            st.success(f"Loaded options from {p}")
            logging.info(f"Loaded options JSON from {p}")
        except Exception as e:
            st.error(f"Failed to load options JSON: {e}")
            logging.warning(f"Failed to load options JSON from {options_json_path}: {e}")

    # ---- Sidebar values (prefill from runtime_config if present) ----
    _sidebar_cfg = st.session_state.get("runtime_config", {}) or {}

    input_dir = st.text_input("Input folder", _sidebar_cfg.get("input_dir", D["input_dir"]))
    output_dir = st.text_input("Output folder", _sidebar_cfg.get("output_dir", D["output_dir"]))
    file_glob = st.text_input("File glob(s) (semicolon-separated)", _sidebar_cfg.get("file_glob", D["file_glob"]))

    st.subheader("ECG / filtering")
    ecg_channel = st.text_input("ECG channel name (optional)", value=str(_sidebar_cfg.get("ecg_channel") or "")) or None
    flip_ecg_on_load = st.checkbox(
        "Flip ECG on load",
        value=bool(_sidebar_cfg.get("flip_ecg_on_load", False)),
        help="Multiplies detected ECG data by -1 for QC and preprocessing. Force redo preprocessing if saved FIFs already exist."
    )
    spirometry_channel = st.text_input("Spirometry lead name (optional)", value=str(_sidebar_cfg.get("spirometry_channel") or "")) or None
    hp = st.number_input("High-pass (Hz)", value=float(_sidebar_cfg.get("hp", D["hp"])), step=0.1, min_value=0.0)
    lp = st.number_input("Low-pass (Hz)", value=float(_sidebar_cfg.get("lp", D["lp"])), step=0.5, min_value=0.0)

    # Notch / line noise frequencies (Hz) - comma-separated
    _lf_default = _sidebar_cfg.get("line_freqs", D.get("line_freqs", (50.0, 100.0)))
    if isinstance(_lf_default, (list, tuple)):
        _lf_default_str = ", ".join([str(float(x)) for x in _lf_default])
    else:
        _lf_default_str = str(_lf_default) if _lf_default is not None else ""
    line_freqs_str = st.text_input("Line/noise freqs (Hz, comma-separated)", value=_lf_default_str, help="Typical UK: 50, 100. Leave blank to skip notch.")
    try:
        line_freqs = tuple([float(x.strip()) for x in line_freqs_str.split(",") if x.strip()]) if line_freqs_str.strip() else ()
    except Exception:
        st.warning("Could not parse line/noise freqs; using empty list.")
        line_freqs = ()

    # Target sampling rate (Hz). Set to 0 to skip resampling.
    _ts_default = _sidebar_cfg.get("target_sfreq", D.get("target_sfreq", None))
    _ts_default_num = float(_ts_default) if _ts_default not in (None, "", "None") else 0.0
    target_sfreq_num = st.number_input("Target sfreq (Hz, 0 = no resample)", value=float(_ts_default_num), step=1.0, min_value=0.0)
    target_sfreq = None if float(target_sfreq_num) <= 0 else float(target_sfreq_num)

    st.subheader("Montage / referencing")
    montage_name_ui = st.text_input("Montage name (MNE)", value=str(_sidebar_cfg.get("montage_name", D.get("montage_name", "standard_1020"))))
    rename_to_1020_ui = st.checkbox("Rename channels to 10-20 where possible", value=bool(_sidebar_cfg.get("rename_to_1020", D.get("rename_to_1020", True))))
    ref_chs_ui = st.text_input("Reference channel selection (e.g. 'eeg')", value=str(_sidebar_cfg.get("ref_chs", D.get("ref_chs", "eeg"))))
    reref_chs_ui = st.text_input("Re-reference channel selection (e.g. 'eeg')", value=str(_sidebar_cfg.get("reref_chs", D.get("reref_chs", "eeg"))))
    _re_default = _sidebar_cfg.get("reference_electrodes", D.get("reference_electrodes", None))
    reference_electrodes_str = st.text_input("Reference electrodes (comma-separated names, optional)", value="" if _re_default in (None, "", "None") else ", ".join(map(str, _re_default)) )
    reference_electrodes = [x.strip() for x in reference_electrodes_str.split(",") if x.strip()] if reference_electrodes_str.strip() else None

    st.subheader("Stimulus event filtering (optional)")
    stim_keep_str = st.text_input("Stim events to keep (comma-separated, optional)", value=", ".join(_sidebar_cfg.get("stim_keep", D.get("stim_keep", [])) or []))
    stim_keep_list = [x.strip() for x in stim_keep_str.split(",") if x.strip()]
    prep_ransac_ui = st.checkbox("PyPREP RANSAC bad-channel detection", value=bool(_sidebar_cfg.get("prep_ransac", D.get("prep_ransac", True))))

    st.subheader("Preprocessing steps")
    run_pyprep = st.checkbox("Run PyPREP", value=bool(_sidebar_cfg.get("run_pyprep", D["run_pyprep"])))
    run_asr = st.checkbox("Run ASR (experimental)", value=bool(_sidebar_cfg.get("run_asr", D["run_asr"])))
    run_ica = st.checkbox("Run ICA + ICLabel", value=bool(_sidebar_cfg.get("run_ica", D["run_ica"])))
    redo_preprocessing = st.checkbox("Force redo preprocessing", value=bool(_sidebar_cfg.get("redo_preprocessing", D["redo_preprocessing"])))

    st.subheader("Cardiac field artefact (ICA)")
    remove_cfa_mode = st.selectbox(
        "CFA handling",
        options=("remove", "keep", "both"),
        index=("remove", "keep", "both").index(str(_sidebar_cfg.get("remove_cfa_mode", D["remove_cfa_mode"])).lower()),
        help="'remove' excludes ICLabel heart-beat ICs; 'keep' retains them; 'both' writes both outputs from one ICA fit."
    )

    st.subheader("Adaptive RR/BPM")
    rr_min_bpm = st.number_input("Min BPM", value=int(_sidebar_cfg.get("rr_min_bpm", D["rr_min_bpm"])), min_value=20, max_value=200)
    rr_max_bpm = st.number_input("Max BPM", value=int(_sidebar_cfg.get("rr_max_bpm", D["rr_max_bpm"])), min_value=20, max_value=220)
    rr_outlier_mad = st.number_input("RR MAD z-threshold", value=float(_sidebar_cfg.get("rr_outlier_mad", D["rr_outlier_mad"])), step=0.5, min_value=1.0)
    bpm_smooth_win = st.number_input("BPM smoothing (beats)", value=int(_sidebar_cfg.get("bpm_smooth_win", D["bpm_smooth_win"])), min_value=1, max_value=21)

    st.subheader("Epoching")
    epoch_tmin = st.number_input("tmin (s)", value=float(_sidebar_cfg.get("epoch_tmin", D["epoch_tmin"])), step=0.05)
    epoch_tmax = st.number_input("tmax (s)", value=float(_sidebar_cfg.get("epoch_tmax", D["epoch_tmax"])), step=0.05)

    _baseline_sidebar = _sidebar_cfg.get("baseline", D["baseline"])
    use_baseline = st.checkbox("Apply baseline", value=_baseline_sidebar is not None)
    default_bmin, default_bmax = -0.15, -0.05
    if isinstance(_baseline_sidebar, (list, tuple)) and len(_baseline_sidebar) == 2:
        default_bmin, default_bmax = float(_baseline_sidebar[0]), float(_baseline_sidebar[1])
    baseline_tmin = st.number_input("Baseline start (s)", value=float(_sidebar_cfg.get("baseline_tmin", default_bmin)), step=0.01, disabled=not use_baseline)
    baseline_tmax = st.number_input("Baseline end (s)", value=float(_sidebar_cfg.get("baseline_tmax", default_bmax)), step=0.01, disabled=not use_baseline)

    amp_reject_uv = st.number_input("Amplitude reject (µV, peak-to-peak)", value=float(_sidebar_cfg.get("amp_reject_uv", D["amp_reject_uv"])), step=10.0, min_value=0.0)
    rr_z_reject = st.number_input("RR z-threshold for epoch reject", value=float(_sidebar_cfg.get("rr_z_reject", D["rr_z_reject"])), step=0.5, min_value=0.0)

    st.subheader("Export")
    export_fif = st.checkbox("MNE .fif epochs", value=bool(_sidebar_cfg.get("export_fif", D["export_fif"])))
    export_eeglab = st.checkbox("EEGLAB .set (needs eeglabio)", value=bool(_sidebar_cfg.get("export_eeglab", D["export_eeglab"])))


# persist config into session for pipeline calls
# Build runtime params using optional loaded config (session), falling back to sidebar values.
_runtime_cfg = st.session_state.get("runtime_config", {}) or {}
params = dict(
    rr_min_bpm=float(_runtime_cfg.get("rr_min_bpm", rr_min_bpm)),
    rr_max_bpm=float(_runtime_cfg.get("rr_max_bpm", rr_max_bpm)),
    rr_outlier_mad=float(_runtime_cfg.get("rr_outlier_mad", rr_outlier_mad)),
    bpm_smooth_win=int(_runtime_cfg.get("bpm_smooth_win", bpm_smooth_win)),
)

# --------- Step 0: Scan input files ---------
st.header("0) Scan input folder")
cscan1, cscan2 = st.columns([1,1], gap="large")
with cscan1:
    if st.button("Scan input folder"):
        if not input_dir:
            st.error("Please set an input folder.")
        else:
            files = _find_raws(Path(input_dir), file_glob)
            st.session_state["raw_files"] = [str(p) for p in files]
            st.success(f"Found {len(files)} file(s).")
with cscan2:
    st.write(f"Current selection: {len(st.session_state.get('raw_files', []))} file(s)")

# --------- Step 1: ECG detection & QC (do this first) ---------
st.header("1) ECG R-peak detection + QC (first)")
col_ecg1, col_ecg2 = st.columns([1,1], gap="large")
with col_ecg1:
    if st.button("Generate ECG QC", type="primary"):
        files = st.session_state.get("raw_files", [])
        if not files:
            st.warning("Nothing to process. Click 'Scan input folder' first.")
        elif not output_dir:
            st.error("Please set an output folder.")
        else:
            out_qc_dir = Path(output_dir) / "ecg_qc"
            out_qc_dir.mkdir(parents=True, exist_ok=True)
            _seed_review_from_raws(files, Path(output_dir), out_qc_dir)
            progress = st.progress(0.0, text="ECG QC...")
            ok, fail = 0, 0
            for i, f in enumerate(files, 1):
                base = Path(f).stem
                png_path = out_qc_dir / f"{base}_ecg_qc.png"
                if png_path.exists() and not redo_preprocessing and not flip_ecg_on_load:
                    ok += 1
                    progress.progress(i/len(files), text=f"[{i}/{len(files)}] {Path(f).name} (cached)")
                    continue
                try:
                    raw = _load_raw_minimal_for_ecg(Path(f), ecg_channel, flip_ecg=flip_ecg_on_load)
                    _save_ecg_qc(raw, base, out_qc_dir, ecg_channel, params, hp, lp, redo=bool(redo_preprocessing or flip_ecg_on_load))
                    ok += 1
                except Exception as e:
                    st.error(f"[{base}] ECG QC failed: {e}")
                    fail += 1
                progress.progress(i/len(files), text=f"[{i}/{len(files)}] {Path(f).name}")
            progress.empty()
            st.success(f"ECG QC complete. {ok} succeeded, {fail} failed.")
            st.session_state["ecg_qc_done"] = True
with col_ecg2:
    st.caption("Run this before EEG preprocessing so QRS sense-check happens first.")

# --------- Step 2: QRS Review (manual sense check) ----------
st.header("2) QRS checking with adaptive correction")
csv_path = Path(output_dir)/CSV_NAME
df_review = _load_review(csv_path)
if df_review.empty:
    st.caption("No review rows yet. Run 'Generate ECG QC' first.")
else:
    remaining = int((df_review["viable"].astype(str).str.len()==0).sum())
    st.caption(f"Remaining to assess: **{remaining}** / {len(df_review)}")

    idx_options = [f"{i:03d} | {row.base}" for i, row in df_review.iterrows()]
    if "force_sel_next" in st.session_state:
        default_sel = int(st.session_state.pop("force_sel_next"))
    else:
        default_sel = int(st.session_state.get("sel_key", 0)) if str(st.session_state.get("sel_key","")).isdigit() else 0
    cur_ix = st.selectbox(
        "Choose item",
        options=list(range(len(df_review))),
        format_func=lambda i: idx_options[i],
        index=default_sel,
        key="sel_key"
    )

    row = df_review.iloc[cur_ix]
    base = str(row.base)
    row_flip = str(row.flip).lower() in ("true","1","yes")
    flip_this_load = bool(flip_ecg_on_load or row_flip)
    raw_fif = Path(str(row.raw_fif)) if str(row.raw_fif) else Path("")
    qc_png = _resolve_qc_png(row, Path(output_dir))
    if qc_png.exists() and not flip_this_load:
        _display_png(qc_png, caption=f"{base} ECG QC")
        rpfixed = np.array([], dtype=int)
    else:
        raw = None
        try:
            src = raw_fif if raw_fif.exists() else Path(row.edf_path) if str(row.edf_path) else None
            if src is None or (not src.exists()):
                st.error(f"Missing file for QC render: {raw_fif}")
            else:
                with st.spinner("Loading raw..."):
                    raw = _load_raw_minimal_for_ecg(src, ecg_channel, flip_ecg=flip_this_load)
                png_path = _save_ecg_qc(
                    raw, base, qc_png.parent, ecg_channel, params, hp, lp, redo=flip_this_load
                )
                if png_path and png_path.exists():
                    _display_png(png_path, caption=f"{base} ECG QC")
                    # update stored path so future loads find it
                    df_review.at[cur_ix, "qc_png"] = str(png_path)
                    _save_review(df_review, csv_path)
                else:
                    st.error("ECG QC could not be generated.")
            rpfixed = np.array([], dtype=int)
        except Exception as e:
            st.error(f"ECG plotting failed: {e}")
            rpfixed = np.array([], dtype=int)
        finally:
            try:
                if raw is not None:
                    raw.close()
            except Exception:
                pass

    qrs_open_key = f"qrs_editor_open_{base}"
    if st.button("Graphically exclude/refit peaks", key=f"open_qrs_editor_{cur_ix}"):
        st.session_state[qrs_open_key] = True
    if st.session_state.get(qrs_open_key, False):
        src = _qrs_review_source(row)
        if src is None:
            st.error("Missing raw file for graphical QRS correction.")
        else:
            try:
                with st.spinner("Loading QRS editor..."):
                    sig_qrs, sf_qrs, rpeaks_qrs, rpeak_source = _load_qrs_review_data(
                        src, base, output_dir, ecg_channel, params, hp, lp, flip_ecg=flip_this_load
                    )
                selected_key = f"qrs_selected_{base}"
                selected_now = [
                    i for i in st.session_state.get(selected_key, [])
                    if isinstance(i, (int, np.integer)) and 0 <= int(i) < len(rpeaks_qrs)
                ]
                pending_key = f"qrs_pending_{base}"
                pending_now = [
                    i for i in st.session_state.get(pending_key, [])
                    if isinstance(i, (int, np.integer)) and 0 <= int(i) < len(rpeaks_qrs)
                ]
                x_qrs, waves_qrs = _qrs_wave_lookup(sig_qrs, sf_qrs, rpeaks_qrs)
                event = st.plotly_chart(
                    _make_qrs_selection_plot(x_qrs, waves_qrs, selected_now),
                    use_container_width=True,
                    key=f"qrs_selection_plot_{base}",
                    on_select="rerun",
                    selection_mode="box",
                    config={"displaylogo": False},
                )
                box_ranges = selected_box_ranges(event)
                if box_ranges:
                    pending_now = intersected_trace_indices(event, x_qrs, waves_qrs)
                    st.session_state[pending_key] = pending_now
                else:
                    clicked = [i for i in selected_peak_indices(event) if 0 <= i < len(rpeaks_qrs)]
                    if clicked:
                        pending_now = clicked
                        st.session_state[pending_key] = pending_now
                st.caption(
                    f"{len(rpeaks_qrs)} peaks shown from {rpeak_source}; "
                    f"{len(pending_now)} boxed, {len(selected_now)} labelled."
                )

                sidecar = corrected_rpeaks_path(Path(output_dir) / "ecg_qc", base)
                clabel, cfix, cdelete = st.columns(3)
                with clabel:
                    if st.button("Label boxed peaks", disabled=not pending_now, key=f"qrs_label_{base}"):
                        selected_now = sorted(set(selected_now) | set(pending_now))
                        st.session_state[selected_key] = selected_now
                        st.session_state[pending_key] = []
                        st.rerun()
                with cfix:
                    if st.button("Exclude selected + refit", disabled=not selected_now, key=f"qrs_refit_{base}"):
                        fixed = adaptive_refit_excluded_rpeaks(sig_qrs, rpeaks_qrs, sf_qrs, selected_now)
                        save_corrected_rpeaks(sidecar, fixed, sf_qrs)
                        st.session_state[selected_key] = []
                        st.session_state[pending_key] = []
                        st.success(f"Saved {len(fixed)} corrected peaks.")
                        st.rerun()
                with cdelete:
                    if st.button("Exclude selected + delete", disabled=not selected_now, key=f"qrs_delete_{base}"):
                        fixed = delete_excluded_rpeaks(rpeaks_qrs, selected_now)
                        save_corrected_rpeaks(sidecar, fixed, sf_qrs)
                        st.session_state[selected_key] = []
                        st.session_state[pending_key] = []
                        st.success(f"Deleted {len(rpeaks_qrs) - len(fixed)} peaks; saved {len(fixed)} remaining.")
                        st.rerun()
                caccept, cclear, cclose = st.columns(3)
                with caccept:
                    if st.button("Accept current peaks", key=f"qrs_accept_{base}"):
                        save_corrected_rpeaks(sidecar, rpeaks_qrs, sf_qrs)
                        st.session_state[selected_key] = []
                        st.session_state[pending_key] = []
                        st.success("Current peaks accepted.")
                        st.rerun()
                with cclear:
                    if st.button("Clear corrections", disabled=not sidecar.exists(), key=f"qrs_clear_{base}"):
                        sidecar.unlink()
                        st.session_state[selected_key] = []
                        st.session_state[pending_key] = []
                        st.success("Manual corrections cleared.")
                        st.rerun()
                with cclose:
                    if st.button("Close editor", key=f"qrs_close_{base}"):
                        st.session_state[qrs_open_key] = False
                        st.session_state[selected_key] = []
                        st.session_state[pending_key] = []
                        st.rerun()
            except Exception as e:
                st.error(f"QRS correction failed: {e}")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("Approve as viable ✅"):
            df_review.at[cur_ix, "viable"] = "yes"
            _save_review(df_review, csv_path)
            st.success("Marked viable.")
    with col2:
        if st.button("Mark NOT viable ❌"):
            df_review.at[cur_ix, "viable"] = "no"
            _save_review(df_review, csv_path)
            st.warning("Marked not viable.")
    with col3:
        note_value = "" if pd.isna(row.notes) else str(row.notes)
        note = st.text_input("Notes", value=note_value)
        flip_val = st.checkbox("Flip ECG polarity (apply next steps)", value=row_flip, key=f"flip_{cur_ix}")
        if st.button("Save note / flip"):
            df_review.at[cur_ix, "notes"] = note
            df_review.at[cur_ix, "flip"] = bool(flip_val)
            _save_review(df_review, csv_path)
            st.success("Saved.")
    with col4:
        if st.button("Next unassessed ➡️"):
            empties = df_review.index[df_review["viable"].fillna("").astype(str).str.len()==0].tolist()
            if len(empties)>0:
                next_ix = int(empties[0])
                try:
                    st.query_params["sel"] = str(next_ix)
                except Exception:
                    pass
                st.session_state["force_sel_next"] = next_ix
                st.rerun()
    with col5:
        st.caption("Use flip if ECG peaks are inverted; applied during preprocessing.")

# --------- Step 3: Preprocess EEG ---------
st.header("3) Preprocess EEG (after QRS review)")
colP1, colP2 = st.columns([1,1], gap="large")
with colP1:
    if st.button("Preprocess approved files", type="primary"):
        df_review = _load_review(csv_path)
        remaining = int((df_review["viable"].astype(str).str.len()==0).sum())
        if remaining > 0:
            st.error(f"Finish QRS review before preprocessing ({remaining} file(s) still unassessed).")
        else:
            files = df_review[df_review["viable"].astype(str).str.lower()=="yes"]["edf_path"].dropna().tolist()
            if not files:
                st.warning("No files marked viable. Approve at least one in QRS review.")
            else:
                sidebar_overrides = {
                    "output_dir": output_dir,
                    "flip_ecg_on_load": bool(flip_ecg_on_load),
                    "spirometry_channel": spirometry_channel,
                    "hp": hp, "lp": lp,
                    "target_sfreq": target_sfreq,
                    "prep_ransac": bool(prep_ransac_ui),
                    "line_freqs": line_freqs,
                    "ref_chs": ref_chs_ui,
                    "reref_chs": reref_chs_ui,
                    "reference_electrodes": reference_electrodes,
                    "montage_name": montage_name_ui,
                    "rename_to_1020": rename_to_1020_ui,
                    "remove_cfa_mode": remove_cfa_mode,
                    "remove_cfa": remove_cfa_mode != "keep",
                    "stim_keep": stim_keep_list,
                    "run_pyprep": run_pyprep,
                    "run_asr": run_asr,
                    "run_ica": run_ica,
                }
                loaded_cfg = st.session_state.get("runtime_config", {}) or {}
                merged = _merge_runtime_config(sidebar_overrides, loaded_cfg, default_cfg)
                _apply_to_preproc(merged)

                out_raw_dir = Path(output_dir) / "raw_fif"
                out_qc_dir  = Path(output_dir) / "ecg_qc"
                out_qc_dir.mkdir(parents=True, exist_ok=True)
                if (not run_ica) and str(remove_cfa_mode).lower() == "both":
                    st.error("Cardiac field artefact mode 'both' requires ICA + ICLabel. Enable ICA or choose 'keep'/'remove'.")
                    st.stop()
                progress = st.progress(0.0, text="Preprocessing...")
                ok, fail = 0, 0
                processed_rows = {}
                for i, f in enumerate(files, 1):
                    flip_flag = bool(flip_ecg_on_load)
                    if not df_review.empty:
                        rows_match = df_review[df_review["edf_path"] == f]
                        if not rows_match.empty:
                            flip_flag = flip_flag or bool(rows_match.iloc[0]["flip"])
                    try:
                        out = _preprocess_one(
                            Path(f),
                            out_raw_dir,
                            out_qc_dir,
                            params,
                            ecg_channel if ecg_channel else None,
                            spirometry_channel if spirometry_channel else None,
                            hp,
                            lp,
                            redo=redo_preprocessing,
                            remove_cfa_mode=remove_cfa_mode,
                            flip_ecg=flip_flag,
                            stim_keep=stim_keep_list,
                        )
                    except Exception as e:
                        logging.exception("Preprocessing failed for %s: %s", f, e)
                        out = None
                    if out is None:
                        fail += 1
                    else:
                        ok += len(out)
                        processed_rows[Path(f).stem] = out
                    progress.progress(i/len(files), text=f"[{i}/{len(files)}] {Path(f).name}")
                progress.empty()
                st.success(f"Done. {ok} output(s) succeeded, {fail} file(s) failed.")
                df = _load_review(csv_path)
                df = _append_missing_bases(df, sorted(out_raw_dir.glob("*_pp_raw*.fif")), out_qc_dir)
                for base, outs in processed_rows.items():
                    ix = df.index[df["base"].astype(str) == base]
                    if len(ix):
                        if "remove" in outs:
                            df.at[ix[0], "raw_fif"] = str(outs["remove"])
                        only_key = next(iter(outs.keys()))
                        if only_key == "keep":
                            df.at[ix[0], "raw_fif"] = str(outs["keep"])
                        if "keep" in outs:
                            df.at[ix[0], "raw_fif_keep"] = str(outs["keep"])
                _save_review(df, csv_path)
                st.info(f"Review table: {csv_path}")
                if spirometry_channel:
                    try:
                        log_df = pd.read_csv(Path(output_dir) / "logs" / "preprocessing.tsv", sep="\t")
                        if {"input", "spirometry_status"} <= set(log_df.columns):
                            latest = log_df[log_df["input"].isin(files)].groupby("input", sort=False).tail(1)
                            statuses = latest["spirometry_status"].fillna("").astype(str).str.lower()
                            ok_spiro = int((statuses == "ok").sum())
                            bad = latest[statuses != "ok"]
                            if len(bad):
                                names = ", ".join(Path(p).stem for p in bad["input"].astype(str).tolist()[:6])
                                more = "" if len(bad) <= 6 else f", +{len(bad) - 6} more"
                                st.warning(f"Spirometry ok for {ok_spiro}/{len(latest)} file(s). Not found/usable for: {names}{more}.")
                            else:
                                st.success(f"Spirometry ok for {ok_spiro}/{len(latest)} file(s).")
                    except Exception:
                        pass
with colP2:
    st.caption("Preprocessing unlocks only after all QRS rows reviewed; uses files marked viable = yes.")

# HRV (optional) - after preprocessing
st.divider()
st.subheader("HRV (optional)")
_runtime_cfg = st.session_state.get("runtime_config", {}) or {}
compute_hrv = st.checkbox("Compute HRV metrics", value=bool(_runtime_cfg.get("compute_hrv", True)))
hrv_time = st.checkbox("Time-domain (RMSSD, SDNN, pNN50...)", value=bool(_runtime_cfg.get("hrv_time", True)))
hrv_freq = st.checkbox("Frequency-domain (LF, HF, LF/HF...)", value=bool(_runtime_cfg.get("hrv_freq", True)))
hrv_nonlinear = st.checkbox("Nonlinear (SD1/SD2, SampEn...)", value=bool(_runtime_cfg.get("hrv_nonlinear", False)))
# HRV compute uses preprocessed files (prefers cardiac-removed when available)
if compute_hrv:
    if st.button("Compute HRV for approved"):
        df = _load_review(csv_path)
        todo = df[df["viable"].astype(str).str.lower() == "yes"].to_dict(orient="records")
        if not todo:
            st.info("No approved rows to compute HRV for.")
        else:
            out_hrv_csv = Path(output_dir) / "hrv" / "hrv_metrics.csv"
            progress = st.progress(0.0, text="HRV metrics...")
            rows = []
            for i, r in enumerate(todo, 1):
                base = r["base"]
                try:
                    raw_path = Path(r.get("raw_fif",""))
                    if not raw_path.exists() and r.get("raw_fif_keep"):
                        alt = Path(r["raw_fif_keep"])
                        raw_path = alt if alt.exists() else raw_path
                    if not raw_path.exists():
                        raise FileNotFoundError(f"Missing raw FIF for HRV: {raw_path}")
                    raw = mne.io.read_raw_fif(str(raw_path), preload=True, verbose="ERROR")
                    sf = float(raw.info["sfreq"])
                    ecg_name = _auto_pick_ecg(raw, ecg_channel if ecg_channel else None)
                    rpeaks, rpeak_source = _rpeaks_for_raw(raw, base, output_dir, ecg_name, params)
                    metrics = _hrv_metrics_from_rpeaks(
                        rpeaks, sf,
                        want_time=hrv_time, want_freq=hrv_freq, want_nl=hrv_nonlinear
                    )
                    row = {
                        "base": base,
                        "raw_fif": str(raw_path),
                        "n_beats": int(len(rpeaks)),
                        "rpeak_source": rpeak_source,
                        "duration_s": float((raw.times[-1] - raw.times[0]) if len(raw.times) else 0.0),
                    }
                    row.update(metrics)
                    rows.append(row)
                except Exception as e:
                    st.error(f"[{base}] HRV failed: {e}")
                progress.progress(i/len(todo), text=f"[{i}/{len(todo)}] {base}")
            progress.empty()
            if rows:
                _write_hrv_rows(rows, out_hrv_csv)
                st.success(f"HRV metrics saved -> {out_hrv_csv}")
# --------- Step 4: Epoch generation ----------
st.header("4) Generate epochs")
c1, c2 = st.columns([1,1], gap="large")
with c1:
    if st.button("Generate", type="primary"):
        df = _load_review(csv_path)
        if df.empty:
            st.warning("No review table found.")
        else:
            out_epo_dir = Path(output_dir) / "epochs"
            out_epo_dir.mkdir(parents=True, exist_ok=True)
            rows = df.to_dict(orient="records")
            todo = [r for r in rows if str(r.get("viable","")).lower()=="yes"]
            if not todo:
                st.info("No rows marked viable yet.")
            else:
                progress = st.progress(0.0, text="Epoching...")
                done = 0
                for i, r in enumerate(todo, 1):
                    base = r["base"]
                    try:
                        def _paths_for_row(row):
                            mode = str(remove_cfa_mode).lower()
                            paths = []
                            if mode == "both":
                                if row.get("raw_fif") and Path(row["raw_fif"]).exists():
                                    paths.append(("remove", Path(row["raw_fif"])))
                                if row.get("raw_fif_keep") and Path(row["raw_fif_keep"]).exists():
                                    paths.append(("keep", Path(row["raw_fif_keep"])))
                            elif mode == "keep":
                                p = Path(row.get("raw_fif_keep") or row.get("raw_fif", ""))
                                if p.exists():
                                    paths.append(("keep", p))
                            else:
                                p = Path(row.get("raw_fif", ""))
                                if not p.exists() and row.get("raw_fif_keep"):
                                    alt = Path(row["raw_fif_keep"])
                                    if alt.exists():
                                        p = alt
                                if p.exists():
                                    paths.append(("remove", p))
                            return paths

                        targets = _paths_for_row(r)
                        if not targets:
                            raise FileNotFoundError("No preprocessed raw file found for this row.")

                        for label, raw_path in targets:
                            raw = mne.io.read_raw_fif(str(raw_path), preload=True, verbose="ERROR")
                            ecg_name = _auto_pick_ecg(raw, ecg_channel if ecg_channel else None)
                            rpeaks, rpeak_source = _rpeaks_for_raw(raw, base, output_dir, ecg_name, params)
                            baseline_tuple = (baseline_tmin, baseline_tmax) if use_baseline else None
                            epochs = _epoch_from_rpeaks(
                                raw, rpeaks, epoch_tmin, epoch_tmax, baseline_tuple,
                                rr_z_reject, amp_reject_uv, reference_electrodes,
                                spirometry_channel if spirometry_channel else None,
                            )
                            out_base = out_epo_dir / (f"{base}_keepcfa" if label == "keep" and str(remove_cfa_mode).lower() == "both" else base)
                            _export_epochs(epochs, out_base,
                                           export_fif=export_fif, export_eeglab=export_eeglab)
                            # log per-variant epoch stats
                            import csv
                            log_line = {
                                "base": base,
                                "variant": label,
                                "raw_path": str(raw_path),
                                "epochs_saved": len(epochs),
                                "events_detected": len(rpeaks),
                                "rpeak_source": rpeak_source,
                                "epochs_dropped": len(rpeaks) - len(epochs),
                                "amp_reject_uv": amp_reject_uv,
                                "rr_z_reject": rr_z_reject,
                                "baseline": baseline_tuple,
                                "remove_cfa_mode": remove_cfa_mode,
                                "flip_ecg": bool(df_review[df_review["base"].astype(str)==base]["flip"].iloc[0]) if "flip" in df_review else False,
                            }
                            log_csv = out_epo_dir / "epoch_log.csv"
                            write_header = not log_csv.exists()
                            with open(log_csv, "a", newline="") as f:
                                w = csv.DictWriter(f, fieldnames=log_line.keys())
                                if write_header:
                                    w.writeheader()
                                w.writerow(log_line)
                            done += 1
                    except Exception as e:
                        st.error(f"[{base}] Epoching failed: {e}")
                    finally:
                        try:
                            plt.close('all')
                        except Exception:
                            pass
                    progress.progress(i/len(todo), text=f"[{i}/{len(todo)}] {base}")
                progress.empty()
                st.success(f"Epoching complete. {done} set(s) exported from {len(todo)} base file(s).")

with c2:
    st.markdown("**Options used**")
    options_used = {
        "filters": {"hp": hp, "lp": lp, "line_freqs": list(line_freqs)},
        "ecg": {"channel": ecg_channel or "", "flip_on_load": bool(flip_ecg_on_load)},
        "montage": {"name": montage_name_ui, "rename_to_1020": rename_to_1020_ui},
        "pyprep": run_pyprep, "asr": run_asr, "ica_iclabel": run_ica, "cfa_mode": remove_cfa_mode,
        "adaptive_rr": params,
        "epoch": {"tmin": epoch_tmin, "tmax": epoch_tmax, "baseline": (baseline_tmin, baseline_tmax) if use_baseline else None,
                  "amp_reject_uv": amp_reject_uv, "rr_z_reject": rr_z_reject},
        "export": {"fif": export_fif, "eeglab": export_eeglab},
        "hrv": {"compute": compute_hrv, "time_domain": hrv_time, "frequency_domain": hrv_freq,
                "nonlinear": hrv_nonlinear},
        "spirometry": {"channel": spirometry_channel or ""},
        "stim_keep": stim_keep_list,
        "remove_cfa_mode": remove_cfa_mode,
        "flip_flags": True if df_review["flip"].any() else False,
    }
    st.json(options_used)
    try:
        import json
        opt_path = Path(output_dir) / "options_used.json"
        opt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(opt_path, "w", encoding="utf-8") as f:
            json.dump(options_used, f, indent=2)
        st.caption(f"Saved options → {opt_path}")
    except Exception as e:
        st.warning(f"Could not save options JSON: {e}")


# --------- Step 5: Statistical analysis ----------
st.header("5) Statistical analysis")

def _guess_participant_condition(base_name: str) -> tuple[str, str]:
    stem = Path(base_name).stem.replace("_epo", "").replace("-epo", "")
    for token in ("_condition_", "_cond_", "_state_", "_task_"):
        if token in stem:
            left, right = stem.split(token, 1)
            return left, right
    parts = re.split(r"[_-]+", stem)
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return stem, "condition"

def _bpm_from_hrv_row(row: pd.Series):
    for col in ("bpm", "BPM", "HR_Mean", "heart_rate", "HeartRate"):
        if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
            try:
                return float(row[col])
            except Exception:
                pass
    if "HRV_MeanNN" in row and pd.notna(row["HRV_MeanNN"]):
        try:
            mean_nn_ms = float(row["HRV_MeanNN"])
            if mean_nn_ms > 0:
                return 60000.0 / mean_nn_ms
        except Exception:
            pass
    return ""

def _load_pipeline_sources(output_root: Path) -> dict[str, dict]:
    """Map HEPPy base names to HRV BPM and raw paths from existing pipeline CSVs."""
    sources: dict[str, dict] = {}
    review_path = output_root / CSV_NAME
    if review_path.exists():
        try:
            review = _load_review(review_path)
            for _, row in review.iterrows():
                base = str(row.get("base", "")).strip()
                if not base:
                    continue
                rec = sources.setdefault(base, {})
                raw_fif = str(row.get("raw_fif", "") or "").strip()
                raw_keep = str(row.get("raw_fif_keep", "") or "").strip()
                edf_path = str(row.get("edf_path", "") or "").strip()
                for candidate in (raw_fif, raw_keep, edf_path):
                    if candidate and Path(candidate).exists():
                        rec["raw_file_path"] = candidate
                        break
        except Exception:
            pass
    hrv_path = output_root / "hrv" / "hrv_metrics.csv"
    if hrv_path.exists():
        try:
            hrv = pd.read_csv(hrv_path)
            for _, row in hrv.iterrows():
                base = str(row.get("base", "")).strip()
                if not base:
                    continue
                rec = sources.setdefault(base, {})
                bpm_val = _bpm_from_hrv_row(row)
                if bpm_val != "":
                    rec["bpm"] = bpm_val
                raw_fif = str(row.get("raw_fif", "") or "").strip()
                if raw_fif and Path(raw_fif).exists():
                    rec.setdefault("raw_file_path", raw_fif)
        except Exception:
            pass
    return sources

def _match_pipeline_source(participant: str, file_name: str, sources: dict[str, dict]) -> dict:
    candidates = [
        participant,
        Path(file_name).stem.replace("_epo", "").replace("-epo", ""),
    ]
    candidates.extend(k for k in sources if str(candidates[1]).startswith(str(k)))
    for key in candidates:
        if key in sources:
            return sources[key]
    return {}

def _make_analysis_manifest(epochs_dir: Path, out_csv: Path, output_root: Path, enrich_from_pipeline: bool = True) -> Path:
    rows = []
    sources = _load_pipeline_sources(output_root) if enrich_from_pipeline else {}
    for f in sorted(epochs_dir.glob("*epo.fif")) + sorted(epochs_dir.glob("*-epo.fif")):
        participant, condition = _guess_participant_condition(f.name)
        source = _match_pipeline_source(participant, f.name, sources)
        rows.append(
            {
                "participant": participant,
                "group": "all",
                "condition": condition,
                "file_path": str(f),
                "bpm": source.get("bpm", ""),
                "raw_file_path": source.get("raw_file_path", ""),
                "ecg_channel": ecg_channel or "",
            }
        )
    if not rows:
        raise FileNotFoundError(f"No Epochs FIF files found in {epochs_dir}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop_duplicates("file_path").to_csv(out_csv, index=False)
    return out_csv

def _enrich_analysis_manifest(csv_path: Path, output_root: Path) -> Path:
    df = pd.read_csv(csv_path)
    sources = _load_pipeline_sources(output_root)
    if "participant" not in df.columns or "file_path" not in df.columns:
        raise ValueError("Manifest needs participant and file_path columns to auto-fill HRV/raw-path fields.")
    if "bpm" not in df.columns:
        df["bpm"] = ""
    if "raw_file_path" not in df.columns:
        df["raw_file_path"] = ""
    if "ecg_channel" not in df.columns:
        df["ecg_channel"] = ecg_channel or ""
    for ix, row in df.iterrows():
        source = _match_pipeline_source(str(row["participant"]), str(row["file_path"]), sources)
        if source.get("bpm", "") != "" and (pd.isna(row.get("bpm", "")) or str(row.get("bpm", "")).strip() == ""):
            df.at[ix, "bpm"] = source["bpm"]
        if source.get("raw_file_path", "") and (pd.isna(row.get("raw_file_path", "")) or str(row.get("raw_file_path", "")).strip() == ""):
            df.at[ix, "raw_file_path"] = source["raw_file_path"]
        if ecg_channel and (pd.isna(row.get("ecg_channel", "")) or str(row.get("ecg_channel", "")).strip() == ""):
            df.at[ix, "ecg_channel"] = ecg_channel
    df.to_csv(csv_path, index=False)
    return csv_path

st.caption("Use a participant-condition manifest for inference. Required columns: participant, condition, file_path. Optional: group, bpm, raw_file_path, ecg_channel, and per-trial metadata covariates.")

ana_left, ana_right = st.columns([1, 1], gap="large")
with ana_left:
    analysis_manifest_default = str(Path(output_dir) / "analysis_manifest_input.csv") if output_dir else ""
    analysis_manifest_path = st.text_input("Analysis manifest CSV", value=analysis_manifest_default)
    analysis_output_dir = st.text_input("Analysis output folder", value=str(Path(output_dir) / "statistics") if output_dir else "")
    enrich_manifest = st.checkbox("Auto-fill BPM and raw paths from HRV/review outputs", value=True)
    if st.button("Create starter manifest from exported epochs"):
        try:
            made = _make_analysis_manifest(Path(output_dir) / "epochs", Path(analysis_manifest_path), Path(output_dir), enrich_manifest)
            st.success(f"Starter manifest saved -> {made}")
            st.dataframe(pd.read_csv(made), use_container_width=True)
        except Exception as e:
            st.error(f"Could not create manifest: {e}")
    if st.button("Enrich existing manifest with HRV/raw paths"):
        try:
            enriched = _enrich_analysis_manifest(Path(analysis_manifest_path), Path(output_dir))
            st.success(f"Manifest enriched -> {enriched}")
            st.dataframe(pd.read_csv(enriched), use_container_width=True)
        except Exception as e:
            st.error(f"Could not enrich manifest: {e}")

    conditions_text = st.text_input(
        "Canonical conditions, comma-separated",
        value="condition_a,condition_b",
        help="Edit to match the condition labels in the manifest. Three or more enables omnibus condition tests."
    )
    aliases_json = st.text_area(
        "Condition aliases as JSON (optional)",
        value="{}",
        help='Example: {"dreaming": ["D", "2"], "non_dreaming": ["ND", "0"]}'
    )
    cluster_method = st.selectbox("Cluster method", ["TFCE", "RFT", "CLUSTER_SUM"], index=0)
    control_mode = st.selectbox(
        "Validation/control mode",
        ["observed", "pseudotrial", "surrogate_heartbeat"],
        help="Pseudotrial uses repeated equal-count trial subsampling. Surrogate heartbeat relocks raw data to shifted pseudo R-peaks and requires raw_file_path."
    )

with ana_right:
    stat_tmin = st.number_input("Analysis tmin (s)", value=0.10, step=0.01)
    stat_tmax = st.number_input("Analysis tmax (s)", value=0.60, step=0.01)
    stat_alpha = st.number_input("Corrected alpha", value=0.05, step=0.01, min_value=0.001, max_value=0.5)
    stat_perms = st.number_input("Permutations", value=1024, min_value=16, step=128)
    stat_min_subjects = st.number_input("Minimum complete participants", value=5, min_value=2, step=1)
    use_bpm_correction = st.checkbox("Correct for heartrate/BPM with Freedman-Lane", value=True)
    run_omnibus = st.checkbox("Run omnibus condition effect", value=True)
    run_pairwise = st.checkbox("Run pairwise condition comparisons", value=True)
    run_within_groups = st.checkbox("Also run within-group analyses", value=False)
    stimulus_lead_column = st.text_input(
        "Stimulus lead metadata column (optional)",
        value="",
        help="For per-trial overlap correction, enter the Epochs metadata column containing stimulus-to-heartbeat lead/lag."
    ) or None
    extra_trial_covs = st.text_input("Extra per-trial covariates (comma-separated, optional)", value="")
    montage_for_stats = st.text_input("Stats montage name if files lack locations", value=str(montage_name_ui or ""))

if st.button("Run statistical analysis", type="primary"):
    try:
        from heppy_stats import AnalysisConfig, run_analysis_from_manifest

        conds = [c.strip() for c in conditions_text.split(",") if c.strip()]
        if len(conds) < 2:
            st.error("Please provide at least two canonical conditions.")
        elif not analysis_manifest_path or not Path(analysis_manifest_path).exists():
            st.error("Please provide an existing analysis manifest CSV, or create the starter manifest first.")
        elif not analysis_output_dir:
            st.error("Please set an analysis output folder.")
        else:
            try:
                condition_aliases = json.loads(aliases_json or "{}")
                if not isinstance(condition_aliases, dict):
                    raise ValueError("Aliases JSON must be an object/dict.")
            except Exception as e:
                raise ValueError(f"Condition aliases JSON is invalid: {e}") from e
            trial_covariates = [c.strip() for c in extra_trial_covs.split(",") if c.strip()]
            cfg = AnalysisConfig(
                manifest_csv=Path(analysis_manifest_path),
                output_dir=Path(analysis_output_dir),
                conditions=conds,
                condition_aliases={str(k): list(v) for k, v in condition_aliases.items()},
                tmin=float(stat_tmin),
                tmax=float(stat_tmax),
                alpha=float(stat_alpha),
                n_permutations=int(stat_perms),
                cluster_method=str(cluster_method),
                run_omnibus=bool(run_omnibus),
                run_pairwise=bool(run_pairwise),
                run_within_groups=bool(run_within_groups),
                use_bpm=bool(use_bpm_correction),
                min_subjects=int(stat_min_subjects),
                montage_name=montage_for_stats or None,
                control_mode=str(control_mode),
                stimulus_lead_column=stimulus_lead_column,
                trial_covariate_columns=trial_covariates,
            )
            with st.spinner("Running cluster statistics..."):
                summary = run_analysis_from_manifest(cfg)
            st.success(f"Analysis complete -> {summary['output_dir']}")
            st.dataframe(pd.DataFrame(summary["summaries"]), use_container_width=True)
            st.json(summary)
    except Exception as e:
        st.error(f"Statistical analysis failed: {e}")
        st.exception(e)
