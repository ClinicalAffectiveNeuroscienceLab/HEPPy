#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HEPPy Streamlit UI
"""

from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import importlib.util

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import mne
import neurokit2 as nk

# --------- Try to pull sensible defaults from your config.py -------------
def _load_defaults():
    defaults = dict(  # keep same keys
        input_dir="", output_dir="", file_glob="**/*.edf;**/*.bdf;**/*.fif",
        ecg_channel=None, hp=0.5, lp=45.0, notch=50.0,
        run_pyprep=True, run_asr=False, run_ica=True, redo_preprocessing=False,
        rr_min_bpm=35, rr_max_bpm=140, rr_outlier_mad=3.5, bpm_smooth_win=5,
        epoch_tmin=-0.2, epoch_tmax=0.6, baseline=None,
        amp_reject_uv=150.0, rr_z_reject=3.5,
        export_fif=True, export_eeglab=False, export_brainvision=True,
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
def _load_config_from_path(path: str) -> tuple[object | None, dict | None, object | None]:
    """Import a Python file at path as a module and extract a simple dict of useful runtime keys.

    Returns (module, dict) or (None, None) on failure.
    """
    p = Path(path)
    if not p.exists():
        return None, None, None
    spec = importlib.util.spec_from_file_location("user_config", str(p))
    if spec is None or spec.loader is None:
        st.error(f"Could not load spec for config file: {p}")
        return None, None, None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore
    except Exception as e:
        st.error(f"Error executing config file {p}: {e}")
        return None, None, None

    # If module exposes set_runtime_config(), prefer calling it (may return a dict or config object)
    cfg_obj = None
    try:
        if hasattr(mod, "set_runtime_config") and callable(mod.set_runtime_config):
            try:
                cfg_obj = mod.set_runtime_config()
            except Exception as e:
                st.warning(f"Calling set_runtime_config() in config file raised: {e}")
    except Exception:
        cfg_obj = None

    # Build a small dict of known parameters to seed the UI
    out = {}
    for k in ("input_dir","output_dir","file_glob","ecg_channel",
              "hp","lp","notch","run_pyprep","run_asr","run_ica","redo_preprocessing",
              "rr_min_bpm","rr_max_bpm","rr_outlier_mad","bpm_smooth_win",
              "epoch_tmin","epoch_tmax","baseline","amp_reject_uv","rr_z_reject",
              "export_fif","export_eeglab","export_brainvision",
              "target_sfreq","prep_ransac","line_freqs","ref_chs","reref_chs",
              "compute_hrv","hrv_time","hrv_freq","hrv_nonlinear"):
        if hasattr(mod, k):
            out[k] = getattr(mod, k)
    return mod, (out or None), cfg_obj

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
        st.session_state.setdefault("runtime_config_module", default_cfg)
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

# ----------------- Preprocessing glue -----------------------------------
def _preprocess_one(path: Path, out_dir: Path, redo=False) -> Path | None:
    """Call your existing helper to produce <base>_pp_raw.fif"""
    from _preprocessing import preprocess_edf
    out_dir.mkdir(parents=True, exist_ok=True)
    base = path.stem
    out_fif = out_dir / f"{base}_pp_raw.fif"
    try:
        raw = preprocess_edf(str(path), output_path=str(out_fif), redo=bool(redo))
    except Exception as e:
        st.error(f"Preprocessing failed for {base}: {e}")
        return None
    return out_fif

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

def _load_review(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        return pd.read_csv(csv_path)
    cols = ["base","edf_path","raw_fif","qc_png","viable","true_rr_s","true_hr_bpm","beats_per_gap","notes"]
    return pd.DataFrame(columns=cols)

def _save_review(df: pd.DataFrame, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

def _append_missing_bases(df: pd.DataFrame, raws: list[Path], qc_dir: Path) -> pd.DataFrame:
    have = set(df["base"].astype(str))
    rows = []
    for rf in raws:
        base = Path(rf).stem.replace("_pp_raw","").replace("_raw","")
        if base not in have:
            rows.append(dict(
                base=base, edf_path="", raw_fif=str(rf),
                qc_png=str(qc_dir/ f"{base}_ecg_qc.png"),
                viable="", true_rr_s="", true_hr_bpm="", beats_per_gap="", notes=""
            ))
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return df

# ----------------- Epoch export ------------------------------------------
def _epoch_from_rpeaks(raw: mne.io.BaseRaw, rpeaks: np.ndarray, tmin: float, tmax: float,
                       baseline, rr_z_reject: float, amp_reject_uv: float,
                       reference_electrodes: list | None) -> mne.Epochs:
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

    reject = dict(eeg=amp_reject_uv*1e-6) if amp_reject_uv else None
    picks_eeg = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, ecg=False, stim=False)
    # Apply hard reference if provided
    try:
        if reference_electrodes:
            raw.set_eeg_reference(reference_electrodes, projection=True)
    except Exception:
        # non-fatal; continue with current reference
        pass
    epochs = mne.Epochs(raw, events=events, event_id=dict(HEP=1),
                        tmin=tmin, tmax=tmax, baseline=baseline, proj=True,
                        picks=picks_eeg, preload=True, reject=reject, on_missing="ignore")
    return epochs

def _export_epochs(epochs: mne.Epochs, out_base: Path, export_fif=True, export_eeglab=False, export_brainvision=True):
    saved = []
    if export_fif:
        f = out_base.with_suffix("_epo.fif")
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
            st.warning(f"Could not write EEGLAB .set (install eeglabio). Error: {e}. Falling back to BrainVision.")
            export_brainvision = True
    if export_brainvision:
        f = out_base.with_suffix(".vhdr")
        mne.export.export_epochs(str(f), epochs=epochs, fmt='brainvision')  # requires MNE>=1.6
        saved.append(f)
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
            for k in ("output_root","target_sfreq","use_asr","line_freqs","high_pass","low_pass","ref_chs","reref_chs","prep_ransac"):
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
    ns.remove_cfa = merged.get("remove_cfa", False)
    ns.log_file = merged.get("log_file", None)
    ns.ref_chs = merged.get("ref_chs", merged.get("reref_chs", "eeg"))
    ns.reref_chs = merged.get("reref_chs", merged.get("ref_chs", "eeg"))
    ns.high_pass = merged.get("hp", merged.get("high_pass", D.get("hp")))
    ns.low_pass = merged.get("lp", merged.get("low_pass", D.get("lp")))
    ns.prep_ransac = merged.get("prep_ransac", True)
    ns.line_freqs = merged.get("line_freqs", (50.0, 100.0))

    try:
        _preproc.set_runtime_config(ns)
        st.session_state["preproc_runtime_applied"] = True
    except Exception as e:
        st.error(f"Applying runtime config to preprocessing failed: {e}")

# ============================= UI ========================================
st.set_page_config(page_title="HEPPy — Preprocess → QRS → Epochs", layout="wide")

st.title("HEPPy: EEG → Adaptive QRS → HEP Epochs")

with st.sidebar:
    st.header("Defaults")
    st.caption("Loaded from config.py if present; you can adjust here and it won't overwrite the file.")
    # allow user to choose an optional Python config file that can set runtime parameters
    config_path = st.text_input("Config file path (optional, python)", value="")
    if config_path:
        mod, cfgdict, cfgobj = _load_config_from_path(config_path)
        if mod is not None:
            # store both module, dict and object (if returned by set_runtime_config)
            st.session_state["runtime_config_module"] = mod
            st.session_state["runtime_config"] = cfgdict or {}
            if cfgobj is not None:
                st.session_state["runtime_config_obj"] = cfgobj
            st.success("Loaded runtime configuration from config file.")
            # apply to preprocessing immediately so prep_params are available
            try:
                loaded_cfg = st.session_state.get("runtime_config", {}) or {}
                default_module = st.session_state.get("runtime_config_module", default_cfg if 'default_cfg' in globals() else None)
                sidebar_overrides = None
                merged = _merge_runtime_config(sidebar_overrides, loaded_cfg, default_module if cfgobj is None else cfgobj)
                _apply_to_preproc(merged)
            except Exception:
                pass
        else:
             if Path(config_path).exists():
                 st.warning("Config file executed but did not expose a usable dict; falling back to sidebar values.")
             else:
                 st.info("Config file not found; using sidebar/default values.")

    # allow runtime config values to populate sidebar defaults
    _sidebar_cfg = st.session_state.get("runtime_config", {}) or {}

    input_dir = st.text_input("Input folder", _sidebar_cfg.get("input_dir", D["input_dir"]))
    output_dir = st.text_input("Output folder", _sidebar_cfg.get("output_dir", D["output_dir"]))
    file_glob = st.text_input("File patterns (;-separated)", _sidebar_cfg.get("file_glob", D["file_glob"]))
    st.divider()
    st.subheader("Preprocessing")
    ecg_channel = st.text_input("ECG channel (blank=auto)", _sidebar_cfg.get("ecg_channel", D["ecg_channel"] or "") or "")
    hp = st.number_input("High-pass (Hz)", value=float(_sidebar_cfg.get("hp", D["hp"])), step=0.1, min_value=0.0)
    lp = st.number_input("Low-pass (Hz)", value=float(_sidebar_cfg.get("lp", D["lp"])), step=1.0, min_value=1.0)
    notch = st.number_input("Notch (Hz)", value=float(_sidebar_cfg.get("notch", D["notch"])), step=1.0, min_value=0.0)
    run_pyprep = st.checkbox("PyPREP", value=bool(_sidebar_cfg.get("run_pyprep", D["run_pyprep"])))
    run_asr = st.checkbox("ASR", value=bool(_sidebar_cfg.get("run_asr", D["run_asr"])))
    run_ica = st.checkbox("ICA + ICLabel", value=bool(_sidebar_cfg.get("run_ica", D["run_ica"])))
    redo_pre = st.checkbox("Redo preprocessing if outputs exist", value=bool(_sidebar_cfg.get("redo_preprocessing", D["redo_preprocessing"])))
    # Advanced prep options (allow overriding defaults loaded from config)
    st.markdown("**Preprocessing advanced**")
    # target_sfreq may be None in configs; streamlit requires a numeric 'value' if min_value/step are numeric
    _cfg_target = _sidebar_cfg.get("target_sfreq", D.get("target_sfreq"))
    try:
        input_default = float(_cfg_target) if _cfg_target is not None else 0.0
    except Exception:
        input_default = 0.0
    target_sfreq_val = st.number_input("Target sampling freq (Hz)", value=input_default, min_value=0.0, step=1.0)
    # interpret 0 or empty as None for downstream behavior
    target_sfreq = None if (target_sfreq_val == 0.0) else float(target_sfreq_val)
    prep_ransac_ui = st.checkbox("Use RANSAC interpolation in PyPREP", value=bool(_sidebar_cfg.get("prep_ransac", D.get("prep_ransac", True))))
    line_freqs_raw = st.text_input("Line frequencies (comma separated)", value=",".join(str(x) for x in _sidebar_cfg.get("line_freqs", D.get("line_freqs", (50.0,100.0)))))
    # parse line freqs string into tuple
    try:
        line_freqs = tuple(float(x.strip()) for x in line_freqs_raw.split(",") if x.strip())
    except Exception:
        line_freqs = D.get("line_freqs", (50.0,100.0))
    ref_chs_ui = st.text_input("PyPrep ref (e.g. 'eeg' or comma list)", value=str(_sidebar_cfg.get("ref_chs", D.get("ref_chs", "eeg"))))
    reref_chs_ui = st.text_input("PyPrep reref (e.g. 'eeg' or comma list)", value=str(_sidebar_cfg.get("reref_chs", D.get("reref_chs", "eeg"))))
    # Optional hard reference to apply before epoching
    reference_electrodes_raw = st.text_input("Final Montage Reference electrodes (comma list, optional)", value=",").join([]) if False else st.text_input("Reference electrodes (comma list, optional)", value=",".join((_sidebar_cfg.get("reference_electrodes") or [])) if isinstance(_sidebar_cfg.get("reference_electrodes"), (list, tuple)) else str(_sidebar_cfg.get("reference_electrodes", "")))
    # parse into list or None
    reference_electrodes = None
    try:
        toks = [t.strip() for t in reference_electrodes_raw.split(",") if t.strip()]
        reference_electrodes = toks if toks else None
    except Exception:
        reference_electrodes = None

    st.divider()
    st.subheader("Adaptive RR/BPM")
    rr_min_bpm = st.number_input("Min BPM", value=int(_sidebar_cfg.get("rr_min_bpm", D["rr_min_bpm"])), min_value=20, max_value=200)
    rr_max_bpm = st.number_input("Max BPM", value=int(_sidebar_cfg.get("rr_max_bpm", D["rr_max_bpm"])), min_value=20, max_value=220)
    rr_outlier_mad = st.number_input("RR MAD z-threshold", value=float(_sidebar_cfg.get("rr_outlier_mad", D["rr_outlier_mad"])), step=0.5, min_value=1.0)
    bpm_smooth_win = st.number_input("BPM smoothing (beats)", value=int(_sidebar_cfg.get("bpm_smooth_win", D["bpm_smooth_win"])), min_value=1, max_value=21)
    st.divider()
    st.subheader("Epoching")
    epoch_tmin = st.number_input("tmin (s)", value=float(_sidebar_cfg.get("epoch_tmin", D["epoch_tmin"])), step=0.05)
    epoch_tmax = st.number_input("tmax (s)", value=float(_sidebar_cfg.get("epoch_tmax", D["epoch_tmax"])), step=0.05)
    use_baseline = st.checkbox("Apply baseline", value=_sidebar_cfg.get("baseline", D["baseline"]) is not None)
    baseline_tmin = st.number_input("Baseline start (s)", value=float(_sidebar_cfg.get("baseline_tmin", -0.15)), step=0.01)
    baseline_tmax = st.number_input("Baseline end (s)", value=float(_sidebar_cfg.get("baseline_tmax", -0.05)), step=0.01)
    amp_reject_uv = st.number_input("Amplitude reject (µV, peak-to-peak)", value=float(_sidebar_cfg.get("amp_reject_uv", D["amp_reject_uv"])), step=10.0, min_value=0.0)
    rr_z_reject = st.number_input("RR z-threshold for epoch reject", value=float(_sidebar_cfg.get("rr_z_reject", D["rr_z_reject"])), step=0.5, min_value=0.0)
    st.divider()
    st.subheader("Export")
    export_fif = st.checkbox("MNE .fif epochs", value=bool(_sidebar_cfg.get("export_fif", D["export_fif"])))
    export_eeglab = st.checkbox("EEGLAB .set (needs eeglabio)", value=bool(_sidebar_cfg.get("export_eeglab", D["export_eeglab"])))
    export_brainvision = st.checkbox("BrainVision .vhdr (EEGLAB-friendly)", value=bool(_sidebar_cfg.get("export_brainvision", D["export_brainvision"])))

# persist config into session for pipeline calls
# Build runtime params using optional loaded config (session), falling back to sidebar values.
_runtime_cfg = st.session_state.get("runtime_config", {}) or {}
params = dict(
    rr_min_bpm=float(_runtime_cfg.get("rr_min_bpm", rr_min_bpm)),
    rr_max_bpm=float(_runtime_cfg.get("rr_max_bpm", rr_max_bpm)),
    rr_outlier_mad=float(_runtime_cfg.get("rr_outlier_mad", rr_outlier_mad)),
    bpm_smooth_win=int(_runtime_cfg.get("bpm_smooth_win", bpm_smooth_win)),
)

# --------- Step 1: Preprocess ---------
st.header("1) Preprocess raw files")
colA, colB = st.columns([1,1], gap="large")
with colA:
    if st.button("Scan input folder"):
        if not input_dir:
            st.error("Please set an input folder.")
        else:
            files = _find_raws(Path(input_dir), file_glob)
            st.session_state["raw_files"] = [str(p) for p in files]
            st.success(f"Found {len(files)} file(s).")
with colB:
    if st.button("Preprocess", type="primary"):
        files = st.session_state.get("raw_files", [])
        if not files:
            st.warning("Nothing to preprocess. Click 'Scan input folder' first.")
        else:
            # Before running preprocessing, ensure _preprocessing has current runtime config
            # Build sidebar overrides dict
            sidebar_overrides = {
                "output_dir": output_dir,
                "hp": hp, "lp": lp,
                "target_sfreq": target_sfreq,
                "prep_ransac": bool(prep_ransac_ui),
                "line_freqs": line_freqs,
                "ref_chs": ref_chs_ui,
                "reref_chs": reref_chs_ui,
                "reference_electrodes": reference_electrodes,
            }
            loaded_cfg = st.session_state.get("runtime_config", {}) or {}
            default_module = st.session_state.get("runtime_config_module", default_cfg if 'default_cfg' in globals() else None)
            merged = _merge_runtime_config(sidebar_overrides, loaded_cfg, default_module)
            _apply_to_preproc(merged)

            out_raw_dir = Path(output_dir) / "raw_fif"
            out_qc_dir  = Path(output_dir) / "ecg_qc"
            out_qc_dir.mkdir(parents=True, exist_ok=True)
            progress = st.progress(0.0, text="Preprocessing…")
            ok, fail = 0, 0
            for i, f in enumerate(files, 1):
                out = _preprocess_one(Path(f), out_raw_dir, redo=redo_pre)
                if out is None:
                    fail += 1
                else:
                    ok += 1
                progress.progress(i/len(files), text=f"[{i}/{len(files)}] {Path(f).name}")
            progress.empty()
            st.success(f"Done. {ok} succeeded, {fail} failed.")
            # initialize review table
            raw_fifs = sorted(out_raw_dir.glob("*_pp_raw.fif"))
            csv_path = Path(output_dir)/CSV_NAME
            df = _append_missing_bases(_load_review(csv_path), raw_fifs, out_qc_dir)
            _save_review(df, csv_path)
            st.info(f"Review table: {csv_path}")

# Move HRV controls above QRS review so they can be referenced during review actions
st.divider()
st.subheader("HRV (optional)")
# use runtime cfg defaults if provided
_runtime_cfg = st.session_state.get("runtime_config", {}) or {}
compute_hrv = st.checkbox("Compute HRV metrics", value=bool(_runtime_cfg.get("compute_hrv", True)))
hrv_time = st.checkbox("Time-domain (RMSSD, SDNN, pNN50…)", value=bool(_runtime_cfg.get("hrv_time", True)))
hrv_freq = st.checkbox("Frequency-domain (LF, HF, LF/HF…)", value=bool(_runtime_cfg.get("hrv_freq", True)))
hrv_nonlinear = st.checkbox("Nonlinear (SD1/SD2, SampEn…)", value=bool(_runtime_cfg.get("hrv_nonlinear", False)))

# --------- Step 2: QRS Review ----------
st.header("2) QRS checking with adaptive correction")
csv_path = Path(output_dir)/CSV_NAME
df_review = _load_review(csv_path)
if df_review.empty:
    st.caption("No review rows yet. Preprocess first.")
else:
    # compute remaining
    remaining = int((df_review["viable"].astype(str).str.len()==0).sum())
    st.caption(f"Remaining to assess: **{remaining}** / {len(df_review)}")

    idx_options = [f"{i:03d} | {row.base}" for i, row in df_review.iterrows()]
    cur_ix = st.selectbox("Choose item", options=list(range(len(df_review))),
                          format_func=lambda i: idx_options[i], index=0)

    row = df_review.iloc[cur_ix]
    raw_fif = Path(str(row.raw_fif))
    if not raw_fif.exists():
        st.error(f"Missing file: {raw_fif}")
    else:
        with st.spinner("Loading raw…"):
            raw = mne.io.read_raw_fif(str(raw_fif), preload=True, verbose="ERROR")
            # apply user-set filters *for visualization* only (does not touch saved raw)
            raw_filt = raw.copy().filter(l_freq=hp if hp>0 else None, h_freq=lp, picks="ecg", verbose="ERROR")
        try:
            ecg_name = _auto_pick_ecg(raw_filt, ecg_channel if ecg_channel else None)
            fig, rpfixed = _render_ecg_qc(raw_filt, ecg_name, params)
            st.pyplot(fig, clear_figure=True, use_container_width=True)
        except Exception as e:
            st.error(f"ECG plotting failed: {e}")
            rpfixed = np.array([], dtype=int)
        finally:
            try:
                raw_filt.close()
            except Exception:
                pass

        col1, col2, col3, col4 = st.columns(4)
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
            note = st.text_input("Notes", value=str(row.notes or ""))
            if st.button("Save note"):
                df_review.at[cur_ix, "notes"] = note
                _save_review(df_review, csv_path)
                st.success("Saved.")
        with col4:
            if st.button("Next unassessed ➡️"):
                # jump to next with empty viable
                empties = df_review.index[df_review["viable"].astype(str).str.len()==0].tolist()
                if len(empties)>0:
                    st.experimental_set_query_params(sel=str(empties[0]))
                st.rerun()

        st.markdown("---")
        if compute_hrv:
            if st.button("Compute HRV for approved 🫀"):
                df = _load_review(csv_path)
                todo = df[df["viable"].astype(str).str.lower() == "yes"].to_dict(orient="records")
                if not todo:
                    st.info("No approved rows to compute HRV for.")
                else:
                    out_hrv_csv = Path(output_dir) / "hrv" / "hrv_metrics.csv"
                    progress = st.progress(0.0, text="HRV metrics…")
                    rows = []
                    for i, r in enumerate(todo, 1):
                        base = r["base"]
                        try:
                            raw = mne.io.read_raw_fif(r["raw_fif"], preload=True, verbose="ERROR")
                            sf = float(raw.info["sfreq"])
                            ecg_name = _auto_pick_ecg(raw, ecg_channel if ecg_channel else None)
                            sig = raw.get_data(picks=ecg_name)[0]
                            rpeaks, _ = _detect_rpeaks(sig, sf)
                            rpeaks = _adaptive_fix_rpeaks(
                                rpeaks, sf,
                                rr_min_bpm, rr_max_bpm, rr_outlier_mad, bpm_smooth_win
                            )
                            metrics = _hrv_metrics_from_rpeaks(
                                rpeaks, sf,
                                want_time=hrv_time, want_freq=hrv_freq, want_nl=hrv_nonlinear
                            )
                            row = {
                                "base": base,
                                "raw_fif": r["raw_fif"],
                                "n_beats": int(len(rpeaks)),
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
                        st.success(f"HRV metrics saved → {out_hrv_csv}")


# --------- Step 3: Epoch generation ----------
st.header("3) Generate epochs")
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
                progress = st.progress(0.0, text="Epoching…")
                done = 0
                for i, r in enumerate(todo, 1):
                    base = r["base"]
                    try:
                        raw = mne.io.read_raw_fif(r["raw_fif"], preload=True, verbose="ERROR")
                        ecg_name = _auto_pick_ecg(raw, ecg_channel if ecg_channel else None)
                        sig = raw.get_data(picks=ecg_name)[0]
                        sf = float(raw.info["sfreq"])
                        rpeaks, _ = _detect_rpeaks(sig, sf)
                        rpeaks = _adaptive_fix_rpeaks(rpeaks, sf, rr_min_bpm, rr_max_bpm, rr_outlier_mad, bpm_smooth_win)
                        baseline_tuple = (baseline_tmin, baseline_tmax) if use_baseline else None
                        epochs = _epoch_from_rpeaks(raw, rpeaks, epoch_tmin, epoch_tmax, baseline_tuple, rr_z_reject, amp_reject_uv, reference_electrodes)
                        out_base = (out_epo_dir / base)
                        saved = _export_epochs(epochs, out_base,
                                               export_fif=export_fif, export_eeglab=export_eeglab, export_brainvision=export_brainvision)
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
                st.success(f"Epoching complete. {done}/{len(todo)} exported.")

with c2:
    st.markdown("**Options used**")
    st.json({
        "filters": {"hp": hp, "lp": lp, "notch": notch},
        "pyprep": run_pyprep, "asr": run_asr, "ica_iclabel": run_ica,
        "adaptive_rr": params,
        "epoch": {"tmin": epoch_tmin, "tmax": epoch_tmax, "baseline": (baseline_tmin, baseline_tmax) if use_baseline else None,
                  "amp_reject_uv": amp_reject_uv, "rr_z_reject": rr_z_reject},
        "export": {"fif": export_fif, "eeglab": export_eeglab, "brainvision": export_brainvision},
        "hrv": {"compute": compute_hrv, "time_domain": hrv_time, "frequency_domain": hrv_freq,
                "nonlinear": hrv_nonlinear},

    })
