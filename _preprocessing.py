#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG preprocessing with robust handling of non-10–20 aux channels:

- Standardise names (incl. P7→T5, P8→T6; T7→T3, T8→T4)
- Classify aux channels (ROC/LOC→eog, EMG→emg, PHOTIC→stim, IBI/BURSTS/SUPPR→misc, unknown non-10–20→misc)
- Apply montage without dropping channels
- Run PyPREP on EEG-only with a NaN-free trimmed montage
- RANSAC interpolation + reset bads
- Optional ASR; ICA + ICLabel pruning (optionally keep/strip cardiac)
- ECG R-peak events (with gap interpolation)
- Save and TSV log
"""

import os
import csv
import gc
import inspect
import logging
import re
from datetime import datetime
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

import mne
import numpy as np
import neurokit2 as nk
import pandas as pd
from mne.preprocessing import ICA

try:
    import psutil
except Exception:  # optional; used only for native-memory breadcrumbs
    psutil = None

try:
    import asrpy
except Exception:  # optional; only required when ASR is enabled
    asrpy = None

try:
    import pyprep
except Exception:  # optional; only required when PyPREP is enabled
    pyprep = None

_PYPREP_SAFE_QUANTILE_PATCHED = False

try:
    from mne_icalabel import label_components
except Exception:  # optional; only required when ICA+ICLabel is enabled
    label_components = None

try:
    from mne_icalabel.iclabel import iclabel_label_components
except Exception:  # older mne-icalabel versions do not expose backend selection
    iclabel_label_components = None

# ------------------------- Logging helpers ------------------------- #

def _now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _stage(name: str) -> None:
    # The parent process records stdout.  This breadcrumb survives ordinary
    # exceptions and identifies the native call active immediately before a
    # Windows access violation.  RSS is more useful than tracemalloc here
    # because NumPy, BLAS, ONNX and PyTorch allocate outside Python's heap.
    rss = ""
    if psutil is not None:
        try:
            rss = f"\trss_gib={psutil.Process().memory_info().rss / 2**30:.3f}"
        except Exception:
            pass
    print(f"HEPPY_STAGE\t{name}{rss}", flush=True)
    logging.info("HEPPY stage: %s%s", name, rss)

# Fixed TSV schema so all runs (success/fail) align.
_PREPROC_TSV_FIELDS = [
    "utc", "status", "stage",
    "input", "output",
    "n_channels", "n_events", "bad_channels",
    "remove_cfa", "remove_cfa_mode", "flip_ecg",
    "asr_threshold", "n_comp", "stim_keep",
    "montage_name", "ica_path", "ica_bads", "ica_bad_labels",
    "spirometry_channel", "spirometry_status", "spirometry_peaks",
    "spirometry_troughs", "spirometry_qc_png", "spirometry_signals_csv",
    "spirometry_summary_csv",
    "error_type", "error",
]

def _append_preproc_tsv(logging_path: str | None, rec: dict) -> None:
    """Append a preprocessing record (success or failure) to a TSV with stable columns."""
    if not logging_path:
        return
    try:
        logp = Path(logging_path)
        if str(logp.parent) not in ("", "."):
            logp.parent.mkdir(parents=True, exist_ok=True)
        file_exists = logp.exists() and logp.stat().st_size > 0
        row = {k: "" for k in _PREPROC_TSV_FIELDS}
        for k, v in (rec or {}).items():
            if k in row:
                row[k] = "" if v is None else str(v)
        with open(str(logp), "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_PREPROC_TSV_FIELDS, delimiter="\t")
            if not file_exists:
                w.writeheader()
            w.writerow(row)
    except Exception:
        logging.exception("Failed to append preprocessing TSV record to %s", logging_path)


# ------------------------- Naming & typing helpers ------------------------- #

TEN_TWENTY_SET = {
    'Fp1','Fp2','F7','F3','Fz','F4','F8',
    'T3','C3','Cz','C4','T4',
    'T5','P3','Pz','P4','T6',
    'O1','O2','A1','A2'
}


from dataclasses import dataclass
from pathlib import Path

# Light wrapper so we can keep names below unchanged
@dataclass
class _RuntimeCfg:
    output_root: Path
    target_sfreq: float | None
    use_asr: float | None
    use_pyprep: bool
    remove_cfa: bool
    remove_cfa_mode: str
    log_file: str | None
    ref_chs: str | list
    reref_chs: str | list
    high_pass: float
    low_pass: float
    prep_ransac: bool
    line_freqs: tuple | list
    montage_name: str | None = None
    rename_to_1020: bool = True
    use_ica: bool = True
    spirometry_channel: str | None = None
    pyprep_channel_wise: bool = False
    pyprep_max_chunk_size: int | None = 8
    pyprep_by_segment: bool = False
    pyprep_safe_quantile_patch: bool = False
    ica_fit_hz: float = 100.0
    iclabel_backend: str | None = "auto"

_CFG_RT: _RuntimeCfg | None = None
# fallback defaults in case set_runtime_config is not called
prep_params = {
    "line_freqs": (50.0, 100.0),
    "ref_chs": "eeg",
    "reref_chs": "eeg",
    "l_freq": 1.0,
    "h_freq": 100.0,
    "ransac": True,
}

def set_runtime_config(cfg) -> None:
    """Call this once from the CLI or notebook to bind the active config."""
    global _CFG_RT, output_dir, target_sfreq, do_asr, do_pyprep, remove_cfa, remove_cfa_mode, log_file, prep_params
    # derive ICA cardiac strategy
    mode = getattr(cfg, "remove_cfa_mode", None)
    legacy_flag = getattr(cfg, "remove_cfa", None)
    if mode is None:
        # backward compatibility: infer mode from legacy bool
        if legacy_flag is None:
            mode = "remove"
            legacy_flag = True
        else:
            mode = "remove" if bool(legacy_flag) else "keep"
    if legacy_flag is None:
        legacy_flag = True if str(mode).lower() != "keep" else False
    _CFG_RT = _RuntimeCfg(
        output_root=Path(cfg.output_root),
        target_sfreq=getattr(cfg, "target_sfreq", None),
        use_asr=getattr(cfg, "use_asr", None),
        use_pyprep=bool(getattr(cfg, "use_pyprep", getattr(cfg, "run_pyprep", True))),
        remove_cfa=bool(legacy_flag),
        remove_cfa_mode=str(mode).lower(),
        log_file=getattr(cfg, "log_file", None),
        ref_chs=getattr(cfg, "ref_chs", "eeg"),
        reref_chs=getattr(cfg, "reref_chs", "eeg"),
        high_pass=getattr(cfg, "high_pass", 1.0),
        low_pass=getattr(cfg, "low_pass", 100.0),
        prep_ransac=getattr(cfg, "prep_ransac", True),
        line_freqs=getattr(cfg, "line_freqs", (50.0, 100.0)),
        montage_name=getattr(cfg, "montage_name", None),
        rename_to_1020=getattr(cfg, "rename_to_1020", True),
        use_ica=bool(getattr(cfg, "use_ica", getattr(cfg, "run_ica", True))),
        spirometry_channel=getattr(cfg, "spirometry_channel", None),
        pyprep_channel_wise=bool(getattr(cfg, "pyprep_channel_wise", False)),
        pyprep_max_chunk_size=getattr(cfg, "pyprep_max_chunk_size", 8),
        pyprep_by_segment=bool(getattr(cfg, "pyprep_by_segment", False)),
        pyprep_safe_quantile_patch=bool(getattr(cfg, "pyprep_safe_quantile_patch", False)),
        ica_fit_hz=float(getattr(cfg, "ica_fit_hz", 100.0)),
        iclabel_backend=getattr(cfg, "iclabel_backend", "auto"),
    )

    # keep legacy names used in the rest of this module
    output_dir   = str(_CFG_RT.output_root)
    target_sfreq = float(_CFG_RT.target_sfreq) if _CFG_RT.target_sfreq else None
    do_asr       = _CFG_RT.use_asr
    do_pyprep    = _CFG_RT.use_pyprep
    remove_cfa   = _CFG_RT.remove_cfa
    remove_cfa_mode = _CFG_RT.remove_cfa_mode
    log_file     = _CFG_RT.log_file or ""

    prep_params = {
        "line_freqs": _CFG_RT.line_freqs,
        "ref_chs":   _CFG_RT.ref_chs,
        "reref_chs": _CFG_RT.reref_chs,
        "l_freq":    _CFG_RT.high_pass,
        "h_freq":    _CFG_RT.low_pass,
        "ransac":    _CFG_RT.prep_ransac,
    }

def _canonicalise_name(ch: str) -> str:
    nm = ch.strip()
    nm = re.sub(r'^\s*EEG\s+', '', nm, flags=re.IGNORECASE)
    parts = [p.strip() for p in nm.split('-') if p.strip()]
    if len(parts) > 1:
        nm = parts[1] if parts[0].isdigit() else parts[0]
    nm = nm.upper()
    nm = nm.replace('FP', 'Fp').replace('Z', 'z')
    repl = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}
    return repl.get(nm, nm)

def _canonical_name_mapping(ch_names: list[str]) -> dict[str, str]:
    mapping, used, counts = {}, set(), {}
    for ch in ch_names:
        base = _canonicalise_name(ch)
        name = base
        while name in used:
            counts[base] = counts.get(base, 0) + 1
            name = f"{base}-{counts[base]}"
        mapping[ch] = name
        used.add(name)
    return mapping

AUX_EOG = {'ROC', 'LOC'}
AUX_STIM = {'PHOTIC', 'MKR', 'MKR+', 'MARKER'}
AUX_MISC = {'IBI', 'BURSTS', 'SUPPR'}
RSP_PEAK_DESC = "RSP_Peak"
RSP_TROUGH_DESC = "RSP_Trough"
ECG_TOKENS = ("ECG", "EKG", "CARD", "EXG")

def _optional_name(value: str | None) -> str | None:
    text = str(value or "").strip()
    return text or None

def _find_channel(raw: mne.io.BaseRaw, preferred: str | None) -> str | None:
    preferred = _optional_name(preferred)
    if not preferred:
        return None
    if preferred in raw.ch_names:
        return preferred
    wanted = preferred.casefold()
    for ch in raw.ch_names:
        if ch.casefold() == wanted:
            return ch
    canonical = _canonicalise_name(preferred)
    if canonical in raw.ch_names:
        return canonical
    wanted = canonical.casefold()
    for ch in raw.ch_names:
        if ch.casefold() == wanted:
            return ch
        if _canonicalise_name(ch).casefold() == wanted:
            return ch
    return None

def _active_spirometry_channel(spirometry_channel: str | None = None) -> str | None:
    return _optional_name(spirometry_channel) or (_optional_name(_CFG_RT.spirometry_channel) if _CFG_RT else None)

def _mark_spirometry_channel(raw: mne.io.BaseRaw, spirometry_channel: str | None) -> str | None:
    ch = _find_channel(raw, _active_spirometry_channel(spirometry_channel))
    if ch:
        raw.set_channel_types({ch: "resp"})
    return ch

def _mark_ecg_channels(raw: mne.io.BaseRaw, ecg_channel: str | None = None) -> list[str]:
    names = []
    preferred = _find_channel(raw, ecg_channel)
    if preferred:
        names.append(preferred)
    for ch in raw.ch_names:
        if any(tok in ch.upper() for tok in ECG_TOKENS) and ch not in names:
            names.append(ch)
    for ch in names:
        try:
            raw.set_channel_types({ch: "ecg"})
        except Exception:
            pass
    return names

def _flip_ecg_channels(raw: mne.io.BaseRaw, ecg_channel: str | None = None) -> list[str]:
    names = _mark_ecg_channels(raw, ecg_channel)
    if not names:
        names = [raw.ch_names[i] for i in mne.pick_types(raw.info, ecg=True)]
    idx = [raw.ch_names.index(ch) for ch in names if ch in raw.ch_names]
    if idx:
        raw._data[idx] *= -1
    return [raw.ch_names[i] for i in idx]

def _preproc_base_from_output(path: str | Path) -> str:
    stem = Path(path).stem
    for suffix in ("_pp_raw_keepcfa", "_pp_raw", "_raw_keepcfa", "_raw"):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem

def _has_spirometry_annotations(raw: mne.io.BaseRaw) -> bool:
    if raw.annotations is None or len(raw.annotations) == 0:
        return False
    descriptions = set(np.asarray(raw.annotations.description).astype(str))
    return bool({RSP_PEAK_DESC, RSP_TROUGH_DESC} & descriptions)

def _add_spirometry_annotations(raw: mne.io.BaseRaw, peaks: np.ndarray, troughs: np.ndarray) -> None:
    if raw.annotations is not None and len(raw.annotations):
        old = [i for i, desc in enumerate(raw.annotations.description) if desc in (RSP_PEAK_DESC, RSP_TROUGH_DESC)]
        if old:
            annotations = raw.annotations.copy()
            annotations.delete(old)
            raw.set_annotations(annotations)
    samples = np.concatenate([peaks, troughs]).astype(int)
    if samples.size == 0:
        return
    descriptions = [RSP_PEAK_DESC] * len(peaks) + [RSP_TROUGH_DESC] * len(troughs)
    order = np.argsort(samples)
    annotations = mne.Annotations(
        onset=samples[order] / float(raw.info["sfreq"]),
        duration=np.zeros(samples.size),
        description=[descriptions[i] for i in order],
        orig_time=raw.annotations.orig_time,
    )
    raw.set_annotations(raw.annotations + annotations)

def save_spirometry_analysis(
    raw: mne.io.BaseRaw,
    spirometry_channel: str | None,
    output_dir: str | Path | None,
    base: str,
    redo: bool = False,
) -> dict:
    requested = _active_spirometry_channel(spirometry_channel)
    if not requested:
        return {"spirometry_status": "off"}
    ch = _find_channel(raw, requested)
    if not ch:
        return {"spirometry_channel": requested, "spirometry_status": "missing"}

    sfreq = float(raw.info["sfreq"])
    signals, info = nk.rsp_process(raw.get_data(picks=ch)[0], sampling_rate=sfreq)
    peaks = np.asarray(info.get("RSP_Peaks", []), dtype=int)
    troughs = np.asarray(info.get("RSP_Troughs", []), dtype=int)
    _add_spirometry_annotations(raw, peaks, troughs)

    rec = {
        "spirometry_channel": ch,
        "spirometry_status": "ok",
        "spirometry_peaks": str(len(peaks)),
        "spirometry_troughs": str(len(troughs)),
    }
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        signals_csv = out_dir / f"{base}_rsp_signals.csv"
        summary_csv = out_dir / f"{base}_rsp_summary.csv"
        events_csv = out_dir / f"{base}_rsp_events.csv"
        qc_png = out_dir / f"{base}_rsp_qc.png"

        if redo or not signals_csv.exists():
            signals.to_csv(signals_csv, index=False)
        try:
            summary = nk.rsp_analyze(signals, sampling_rate=sfreq, method="interval-related")
        except Exception as exc:
            summary = pd.DataFrame([{"error": str(exc)}])
        if redo or not summary_csv.exists():
            summary.to_csv(summary_csv, index=False)
        if redo or not events_csv.exists():
            event_rows = (
                [{"sample": int(s), "time_s": float(s / sfreq), "kind": "peak"} for s in peaks]
                + [{"sample": int(s), "time_s": float(s / sfreq), "kind": "trough"} for s in troughs]
            )
            pd.DataFrame(event_rows, columns=["sample", "time_s", "kind"]).sort_values("sample").to_csv(events_csv, index=False)
        if redo or not qc_png.exists():
            import matplotlib.pyplot as plt
            nk.rsp_plot(signals, info, static=True)
            fig = plt.gcf()
            fig.savefig(qc_png, dpi=150, bbox_inches="tight")
            plt.close(fig)

        rec.update({
            "spirometry_qc_png": str(qc_png),
            "spirometry_signals_csv": str(signals_csv),
            "spirometry_summary_csv": str(summary_csv),
        })
    return rec

def spirometry_phase_at_samples(raw: mne.io.BaseRaw, samples: np.ndarray) -> pd.DataFrame | None:
    if raw.annotations is None or len(raw.annotations) == 0:
        return None
    sfreq = float(raw.info["sfreq"])
    desc = np.asarray(raw.annotations.description).astype(str)
    ann_samples = np.rint(np.asarray(raw.annotations.onset) * sfreq).astype(int)
    peaks = ann_samples[desc == RSP_PEAK_DESC]
    troughs = ann_samples[desc == RSP_TROUGH_DESC]
    if len(peaks) == 0 or len(troughs) == 0:
        return None
    phase = nk.rsp_phase(peaks=peaks, troughs=troughs, desired_length=raw.n_times)
    idx = np.clip(np.asarray(samples, dtype=int), 0, raw.n_times - 1)
    rows = phase.iloc[idx].reset_index(drop=True).rename(columns={
        "RSP_Phase": "resp_phase",
        "RSP_Phase_Completion": "resp_phase_completion",
    })
    vals = pd.to_numeric(rows["resp_phase"], errors="coerce").to_numpy()
    rows["resp_phase_label"] = np.where(
        np.isfinite(vals),
        np.where(vals == 1, "inspiratory", "expiratory"),
        "unknown",
    )
    return rows

def _classify_aux_channels(raw: mne.io.BaseRaw) -> None:
    """Classify obvious auxiliary channels but preserve all EEG leads.

    Previous logic demoted any EEG channel not in the 10-20 set to 'misc', causing
    large arrays (e.g., dense caps, BioSemi, EGI) to be dropped from EEG processing.
    We now only re-type clearly non-EEG auxiliary channels (EOG/stim/misc/ECG/EMG) and
    leave all remaining channels as EEG.
    """
    for ch in list(raw.ch_names):
        up = ch.upper()
        if up in AUX_EOG:
            raw.set_channel_types({ch: 'eog'})
        elif up in AUX_STIM:
            raw.set_channel_types({ch: 'stim'})
        elif up in AUX_MISC:
            raw.set_channel_types({ch: 'misc'})
        elif any(tok in up for tok in ECG_TOKENS):
            raw.set_channel_types({ch: 'ecg'})
        elif up == 'EMG':
            raw.set_channel_types({ch: 'emg'})
    # Do not demote remaining EEG channels; keep full set intact.

_DEF_MONTAGES = [
    "standard_1020", "standard_1005", "biosemi32", "biosemi64", "biosemi128",
    "GSN-HydroCel-32", "GSN-HydroCel-64", "GSN-HydroCel-128", "GSN-HydroCel-256",
    "easycap-M1", "easycap-M10", "egi256"
]

def _auto_detect_montage(raw: mne.io.BaseRaw, candidates: list[str] | None = None) -> str | None:
    """Optional montage auto-detection: choose built-in montage with highest channel name coverage."""
    if candidates is None:
        candidates = _DEF_MONTAGES
    ch_set = {
        _canonicalise_name(ch)
        for ch, kind in zip(raw.ch_names, raw.get_channel_types())
        if kind == "eeg" and not any(tok in _canonicalise_name(ch).upper() for tok in ECG_TOKENS)
    }
    best_name, best_cov = None, 0.0
    for name in candidates:
        try:
            mont = mne.channels.make_standard_montage(name)
        except Exception:
            continue
        mont_chs = set(mont.ch_names)
        inter = ch_set & mont_chs
        if not ch_set:
            continue
        cov = len(inter) / max(1, len(ch_set))
        if cov > best_cov:
            best_cov, best_name = cov, name
    # require minimal coverage threshold
    if best_cov >= 0.5:
        return best_name
    return None

def standardise_and_montage(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Rename channels, apply montage early, then classify auxiliaries.

    Applying the montage before auxiliary classification avoids losing EEG leads
    when dense arrays are used. All EEG channels are preserved.
    """
    # Canonicalise channel names first to maximize montage matching
    raw.rename_channels(_canonical_name_mapping(raw.ch_names))
    # Decide montage: explicit, auto-detect, or fallback
    m_name = _CFG_RT.montage_name if _CFG_RT else None
    if m_name is None or str(m_name).lower() in ("", "auto", "none"):
        detected = _auto_detect_montage(raw)
        m_name = detected or "standard_1020"
    mont = mne.channels.make_standard_montage(m_name)
    raw.set_montage(mont, match_case=False, on_missing='ignore')
    logging.info(f"Applied montage: {m_name}")
    # Classify auxiliaries AFTER montage so EEG leads remain EEG
    eeg_before = sum(1 for t in raw.get_channel_types() if t == 'eeg')
    _classify_aux_channels(raw)
    eeg_after = sum(1 for t in raw.get_channel_types() if t == 'eeg')
    if eeg_after < eeg_before:
        logging.warning(f"EEG channel count decreased from {eeg_before} to {eeg_after}. Check classification rules.")

    # Drop EEG channels not present in montage (post-application)
    mont_chs = set(mont.ch_names)
    drop_eeg = [ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eeg" and ch not in mont_chs]
    eeg_chs = [ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eeg"]
    if drop_eeg and len(drop_eeg) == len(eeg_chs):
        logging.warning("No EEG channels matched montage %s; preserving unmatched EEG names.", m_name)
    elif drop_eeg:
        raw.drop_channels(drop_eeg)

    # force the lead name types to be str (to avoid issues with numpy.str_)
    raw.rename_channels({ch: str(ch) for ch in raw.ch_names})
    
    return raw

def _finite_ch_pos(ch_pos: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ok = {}
    for k, v in (ch_pos or {}).items():
        arr = np.asarray(v, float)
        if arr.shape and np.all(np.isfinite(arr)):
            ok[k] = arr
    return ok

def _trim_eeg_montage_no_nan(raw_eeg: mne.io.BaseRaw) -> mne.channels.DigMontage | None:
    mont = raw_eeg.get_montage()
    if mont is None:
        return None
    pos = mont.get_positions()
    ch_pos = _finite_ch_pos(pos.get('ch_pos', {}))
    present = [ch for ch in raw_eeg.ch_names if ch in ch_pos]
    if not present:
        return None
    return mne.channels.make_dig_montage(
        ch_pos={k: ch_pos[k] for k in present},
        nasion=pos.get('nasion'), lpa=pos.get('lpa'), rpa=pos.get('rpa'),
        hpi=pos.get('hpi'), coord_frame=pos.get('coord_frame', 'head'),
    )

def _prune_prep_params_for_raw(params: dict, raw: mne.io.BaseRaw) -> dict:
    import copy
    pruned = copy.deepcopy(params)
    present = set(raw.ch_names)
    keys = {
        "ref_chs","reref_chs","eog_chs","corr_chs",
        "ransac_channel_picks","interpolation_channel_picks",
        "exclude","include","target_channels","bad_channel_prior"
    }
    def _walk(d):
        for k, v in list(d.items()):
            if isinstance(v, dict):
                _walk(v)
            elif isinstance(v, (list, tuple)) and any(isinstance(x, str) for x in v):
                d[k] = [x for x in v if isinstance(x, str) and x in present]
            if k in keys:
                vv = d.get(k, None)
                if isinstance(vv, (list, tuple)):
                    d[k] = [x for x in vv if isinstance(x, str) and x in present]
        return d
    return _walk(pruned)

def _add_channels_by_name(
    target: mne.io.BaseRaw,
    source: mne.io.BaseRaw,
    channel_names: list[str],
) -> mne.io.BaseRaw:
    channel_names = [ch for ch in channel_names if ch in source.ch_names and ch not in target.ch_names]
    if not channel_names:
        return target
    picks = mne.pick_channels(source.ch_names, include=channel_names, ordered=True)
    data = source.get_data(picks=picks)
    info = mne.pick_info(source.info, picks, copy=True)
    selected = mne.io.RawArray(
        data,
        info,
        first_samp=source.first_samp,
        copy="auto",
        verbose=False,
    )
    target.add_channels([selected], force_update_info=True)
    return target


def _add_non_eeg_channels(target: mne.io.BaseRaw, source: mne.io.BaseRaw) -> mne.io.BaseRaw:
    non_eeg_chs = [
        ch for ch, kind in zip(source.ch_names, source.get_channel_types()) if kind != "eeg"
    ]
    return _add_channels_by_name(target, source, non_eeg_chs)

def _snip_sample_bounds_for_input(input_path: str, raw: mne.io.BaseRaw) -> list[tuple[int, int]] | None:
    manifest = Path(input_path).with_name("snip_keyfile.csv")
    if not manifest.exists():
        return None
    sfreq = float(raw.info["sfreq"])
    target = Path(input_path).name
    bounds: list[tuple[int, int]] = []
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if Path(str(row.get("output_edf", ""))).name != target:
                continue
            try:
                start = float(row["output_start_seconds"])
                end = float(row["output_end_seconds"])
            except Exception:
                continue
            if not (np.isfinite(start) and np.isfinite(end) and end > start):
                continue
            start_samp = max(0, min(raw.n_times, int(round(start * sfreq))))
            end_samp = max(0, min(raw.n_times, int(round(end * sfreq))))
            if end_samp > start_samp:
                bounds.append((start_samp, end_samp))
    if len(bounds) <= 1:
        return None
    bounds.sort()
    tol = max(2, int(round(0.01 * sfreq)))
    expected = 0
    for start, stop in bounds:
        if abs(start - expected) > tol:
            logging.warning("Snip manifest for %s has a gap/overlap near sample %s; using whole-record PyPREP.", target, expected)
            return None
        expected = stop
    if abs(expected - raw.n_times) > tol:
        logging.warning("Snip manifest for %s covers %s/%s samples; using whole-record PyPREP.", target, expected, raw.n_times)
        return None
    return bounds

def _safe_mat_quantile_1d(values, q: float) -> float:
    vals = np.asarray(values, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    n = vals.size
    if n == 0:
        return float("nan")
    if n == 1:
        return float(vals[0])
    q_adj = ((float(q) - 0.5) * n / (n - 1)) + 0.5
    exact_idx = (n - 1) * np.clip(q_adj, 0, 1)
    pre_idx = int(np.floor(exact_idx))
    post_idx = int(np.ceil(exact_idx))
    vals.partition((pre_idx, post_idx))
    pre = vals[pre_idx]
    post = vals[post_idx]
    return float(pre + (post - pre) * (exact_idx - pre_idx))

def _safe_mat_quantile(arr, q, axis=None):
    data = np.asarray(arr)
    if data.size == 0:
        return np.nan
    if data.ndim > 2:
        raise ValueError(f"Only 1D and 2D arrays are supported (input has {data.ndim} dimensions)")
    if axis is None or data.ndim == 1:
        return _safe_mat_quantile_1d(data, q)
    if axis < 0:
        axis += data.ndim
    if axis == 0:
        return np.asarray([_safe_mat_quantile_1d(data[:, idx], q) for idx in range(data.shape[1])])
    if axis == 1:
        return np.asarray([_safe_mat_quantile_1d(data[idx, :], q) for idx in range(data.shape[0])])
    raise ValueError(f"axis {axis} is out of bounds for array of dimension {data.ndim}")

def _safe_mat_iqr(arr, axis=None):
    return _safe_mat_quantile(arr, 0.75, axis) - _safe_mat_quantile(arr, 0.25, axis)

def _install_pyprep_safe_quantile_patch() -> None:
    global _PYPREP_SAFE_QUANTILE_PATCHED
    if _PYPREP_SAFE_QUANTILE_PATCHED or pyprep is None:
        return
    try:
        import pyprep.find_noisy_channels as noisy_channels
        import pyprep.utils as utils
    except Exception:
        return
    utils._mat_quantile = _safe_mat_quantile
    utils._mat_iqr = _safe_mat_iqr
    noisy_channels._mat_quantile = _safe_mat_quantile
    noisy_channels._mat_iqr = _safe_mat_iqr
    _PYPREP_SAFE_QUANTILE_PATCHED = True
    logging.info("Patched PyPREP quantile helper for bounded-memory channel IQR.")

# ------------------------------- Core steps -------------------------------- #

def run_pyprep(
    raw: mne.io.BaseRaw,
    param_dict: dict | None = None,
    random_seed: int = 42,
    ransac: bool = True,
    channel_wise: bool | None = None,
    max_chunk_size: int | None = None,
    segment_bounds: list[tuple[int, int]] | None = None,
) -> mne.io.BaseRaw:
    if pyprep is None:
        raise RuntimeError(
            "PyPREP is enabled but the 'pyprep' package is not installed. "
            "Disable 'Run PyPREP' or install pyprep."
        )
    if _CFG_RT is not None and _CFG_RT.pyprep_safe_quantile_patch:
        _install_pyprep_safe_quantile_patch()
    if param_dict is None:
        param_dict = prep_params
    if channel_wise is None:
        channel_wise = bool(getattr(_CFG_RT, "pyprep_channel_wise", False))
    if max_chunk_size is None and _CFG_RT is not None:
        max_chunk_size = _CFG_RT.pyprep_max_chunk_size

    # Segment-wise PyPREP is retained as an opt-in compatibility mode.  It is
    # substantially slower and keeps several complete Raw objects alive, so the
    # stable default is one fit over the patient recording.
    if segment_bounds and len(segment_bounds) > 1:
        sfreq = float(raw.info["sfreq"])
        chunks = []
        logging.info("Running PyPREP over %d snip segments.", len(segment_bounds))
        for idx, (start, stop) in enumerate(segment_bounds, start=1):
            start = max(0, min(raw.n_times, int(start)))
            stop = max(0, min(raw.n_times, int(stop)))
            if stop <= start:
                continue
            chunk = raw.copy().crop(
                tmin=start / sfreq,
                tmax=(stop - 1) / sfreq,
                include_tmax=True,
            )
            try:
                chunks.append(
                    run_pyprep(
                        chunk,
                        param_dict,
                        random_seed=random_seed + idx,
                        ransac=ransac,
                        channel_wise=channel_wise,
                        max_chunk_size=max_chunk_size,
                        segment_bounds=None,
                    )
                )
            except Exception as exc:
                logging.warning(
                    "PyPREP failed on segment %d/%d (%d:%d); keeping raw segment: %s",
                    idx,
                    len(segment_bounds),
                    start,
                    stop,
                    exc,
                )
                chunks.append(chunk)
        if chunks:
            return mne.concatenate_raws(chunks, verbose=False)

    raw_eeg = raw.copy().pick("eeg")
    safe_params = _prune_prep_params_for_raw(param_dict, raw_eeg)
    trimmed_mont = _trim_eeg_montage_no_nan(raw_eeg)
    if trimmed_mont is None:
        logging.warning("PyPREP skipped; no EEG channels have usable montage positions.")
        _add_non_eeg_channels(raw_eeg, raw)
        return raw_eeg

    safe_params.setdefault("line_freqs", ())
    safe_params.setdefault("l_freq", None)
    safe_params.setdefault("h_freq", None)
    nyquist = float(raw_eeg.info["sfreq"]) / 2.0
    safe_params["line_freqs"] = tuple(
        float(freq)
        for freq in safe_params.get("line_freqs", ())
        if 0.0 < float(freq) < nyquist
    )

    def _fit_prep(use_ransac: bool):
        kwargs = {
            "raw": raw_eeg,
            "montage": trimmed_mont,
            "prep_params": safe_params,
            "random_state": random_seed,
            "ransac": use_ransac,
            "channel_wise": bool(channel_wise),
        }
        try:
            parameters = inspect.signature(pyprep.PrepPipeline).parameters
        except Exception:
            parameters = {}
        if "max_chunk_size" in parameters:
            kwargs["max_chunk_size"] = max_chunk_size
        if "reject_by_annotation" in parameters:
            kwargs["reject_by_annotation"] = "omit"
        prep = pyprep.PrepPipeline(**kwargs)
        _stage(
            f"pyprep_fit:r={int(use_ransac)}:channel_wise={int(bool(channel_wise))}"
        )
        with open(os.devnull, "w") as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
            return prep.fit()

    result = None
    try:
        result = _fit_prep(ransac)
    except Exception as exc:
        failure = exc
        msg = str(failure)
        ransac_exhausted = ransac and (
            "Too few channels in the original data to reliably perform RANSAC" in msg
            or "Too many noisy channels in the data to reliably perform RANSAC" in msg
            or "arrays used as indices must be of integer (or boolean) type" in msg
        )
        if ransac_exhausted:
            logging.warning("PyPREP RANSAC skipped after failure: %s", failure)
            try:
                result = _fit_prep(False)
            except Exception as fallback_exc:
                failure = fallback_exc
            else:
                failure = None
        if failure is not None and "RobustReference:TooManyBad" in str(failure):
            logging.warning(
                "PyPREP skipped; too many bad EEG channels for robust reference: %s",
                failure,
            )
            failure = None
        if failure is not None:
            print("EEG picks:", sorted(raw_eeg.ch_names))
            print(
                "trimmed EEG montage ch_pos:",
                sorted(
                    _finite_ch_pos(
                        trimmed_mont.get_positions().get("ch_pos", {})
                    ).keys()
                ),
            )
            print(
                "params (ref/reref/eog/ransac/interp/corr):",
                {
                    key: safe_params.get(key)
                    for key in (
                        "ref_chs",
                        "reref_chs",
                        "eog_chs",
                        "ransac_channel_picks",
                        "interpolation_channel_picks",
                        "corr_chs",
                    )
                },
            )
            raise RuntimeError(f"PyPREP failed: {failure}") from failure

    cleaned_eeg = result.raw_eeg if result is not None else raw_eeg
    if cleaned_eeg.info.get("bads"):
        try:
            cleaned_eeg.interpolate_bads(reset_bads=True)
        except Exception as exc:
            raise RuntimeError("PyPREP bad channel interpolation failed.") from exc
    cleaned_eeg.info["bads"] = []
    _add_non_eeg_channels(cleaned_eeg, raw)
    return cleaned_eeg


def _effective_filter_bounds(
    raw: mne.io.BaseRaw,
    l_freq: float | None,
    h_freq: float | None,
) -> tuple[float | None, float | None]:
    nyquist = float(raw.info["sfreq"]) / 2.0
    low = float(l_freq) if l_freq is not None else None
    high = float(h_freq) if h_freq is not None else None
    if low is not None and low <= 0:
        low = None
    if high is not None and high >= nyquist:
        # Sampling already imposes this upper bound; asking MNE to design a
        # filter exactly at Nyquist is invalid.
        high = None
    if low is not None and low >= nyquist:
        raise ValueError(f"High-pass {low} Hz is not below Nyquist {nyquist} Hz")
    if low is not None and high is not None and low >= high:
        raise ValueError(f"Invalid EEG band-pass: {low}–{high} Hz")
    return low, high


def _filter_eeg_once(
    raw: mne.io.BaseRaw,
    l_freq: float | None,
    h_freq: float | None,
) -> mne.io.BaseRaw:
    low, high = _effective_filter_bounds(raw, l_freq, h_freq)

    # MNE records the effective high- and low-pass limits in info.  The old
    # implementation compared these fields to the wrong requested cut-offs,
    # so already-filtered data were filtered again.
    existing_low = float(raw.info.get("highpass", 0.0) or 0.0)
    existing_high = float(raw.info.get("lowpass", np.inf) or np.inf)
    tolerance = 1e-6
    if low is not None and existing_low >= low - tolerance:
        if existing_low > low + tolerance:
            logging.warning(
                "Input is already high-passed at %.3f Hz, above requested %.3f Hz; "
                "the removed lower frequencies cannot be restored.",
                existing_low,
                low,
            )
        low = None
    if high is not None and existing_high <= high + tolerance:
        if existing_high < high - tolerance:
            logging.warning(
                "Input is already low-passed at %.3f Hz, below requested %.3f Hz; "
                "the removed higher frequencies cannot be restored.",
                existing_high,
                high,
            )
        high = None

    if low is None and high is None:
        return raw
    _stage(f"eeg_filter:{low}:{high}")
    raw.filter(
        l_freq=low,
        h_freq=high,
        picks="eeg",
        fir_design="firwin",
        n_jobs=1,
        verbose=False,
    )
    return raw


def _normalise_asr_cutoff(value) -> float | None:
    if value is None or value is False:
        return None
    if value is True:
        return 20.0
    try:
        cutoff = float(value)
    except (TypeError, ValueError):
        return None
    return cutoff if cutoff > 0 else None


def _ica_decimation(raw: mne.io.BaseRaw) -> int:
    target_hz = float(getattr(_CFG_RT, "ica_fit_hz", 100.0) or 100.0)
    return max(1, int(np.ceil(float(raw.info["sfreq"]) / target_hz)))


_ICLABEL_CLASSES = (
    "brain",
    "muscle artifact",
    "eye blink",
    "heart beat",
    "line noise",
    "channel noise",
    "other",
)


def _normalise_iclabel_label(label: str) -> str:
    text = str(label).strip().lower().replace("_", " ")
    aliases = {
        "heart": "heart beat",
        "cardiac": "heart beat",
        "eye": "eye blink",
        "muscle": "muscle artifact",
    }
    return aliases.get(text, text)


def _label_ica_components(inst: mne.io.BaseRaw, ica: ICA) -> list[str]:
    backend = str(getattr(_CFG_RT, "iclabel_backend", "auto") or "").strip().lower()
    if backend in {"", "auto", "none"}:
        backend = ""

    if backend and iclabel_label_components is not None:
        _stage(f"iclabel:{backend}")
        try:
            probabilities = iclabel_label_components(
                inst, ica, inplace=True, backend=backend
            )
        except Exception as exc:
            raise RuntimeError(
                f"ICLabel backend {backend!r} failed. Install a compatible "
                f"backend or set preprocessing_tuning.iclabel_backend: {exc}"
            ) from exc
        probabilities = np.asarray(probabilities)
        if probabilities.ndim != 2 or probabilities.shape[1] != len(_ICLABEL_CLASSES):
            raise RuntimeError(
                f"Unexpected ICLabel probability shape: {probabilities.shape}"
            )
        return [
            _ICLABEL_CLASSES[int(index)]
            for index in np.argmax(probabilities, axis=1)
        ]

    if label_components is None:
        raise RuntimeError(
            "ICA+ICLabel is enabled but the 'mne_icalabel' package is not installed. "
            "Disable 'Run ICA + ICLabel' or install mne-icalabel."
        )
    _stage("iclabel:auto")
    result = label_components(inst, ica, method="iclabel")
    labels = [_normalise_iclabel_label(value) for value in result.get("labels", [])]
    if not labels:
        raise RuntimeError("ICLabel returned no component labels")
    return labels


def _iclabel_instance(raw_eeg: mne.io.BaseRaw) -> mne.io.BaseRaw:
    highpass = float(raw_eeg.info.get("highpass", 0.0) or 0.0)
    lowpass = float(raw_eeg.info.get("lowpass", np.inf) or np.inf)
    required_high = min(100.0, np.nextafter(float(raw_eeg.info["sfreq"]) / 2.0, 0.0))
    if highpass >= 0.95 and lowpass <= required_high + 0.5:
        return raw_eeg
    inst = raw_eeg.copy()
    return _filter_eeg_once(inst, 1.0, required_high)


def _max_eeg_time_std(inst: mne.io.BaseRaw) -> float:
    picks = mne.pick_types(inst.info, eeg=True, exclude=[])
    maximum = 0.0
    for pick in picks:
        if inst.preload:
            values = inst._data[pick]
        else:
            values = inst.get_data(picks=[pick])[0]
        maximum = max(maximum, float(np.nanstd(values)))
    return maximum


def _apply_ica_excludes(
    inst: mne.io.BaseRaw,
    fitted_ica: ICA,
    exclude: list[int],
) -> mne.io.BaseRaw:
    fitted_ica.exclude = list(exclude)
    if not fitted_ica.exclude:
        return inst.copy()
    before = _max_eeg_time_std(inst)
    _stage(f"ica_apply:n_exclude={len(fitted_ica.exclude)}")
    cleaned = fitted_ica.apply(inst.copy(), verbose=False)
    if before and _max_eeg_time_std(cleaned) <= before * 1e-6:
        logging.warning("ICA application flattened EEG; keeping pre-ICA data.")
        fitted_ica.exclude = []
        return inst.copy()
    return cleaned


def _apply_asr(
    raw_eeg: mne.io.BaseRaw,
    asr_thresh: float | int | bool | None,
) -> mne.io.BaseRaw:
    cutoff = _normalise_asr_cutoff(asr_thresh)
    if cutoff is None:
        return raw_eeg
    if asrpy is None:
        raise RuntimeError(
            "ASR is enabled but the 'asrpy' package is not installed. "
            "Disable 'Run ASR' or install asrpy."
        )
    # ASRpy binds the model to the sampling rate supplied at construction.
    # Fitting at 100 Hz and then transforming a higher-rate Raw is not a valid
    # speed shortcut and can fail inside NumPy/BLAS code.  Keep fit and
    # transform at exactly the same sampling rate.
    sfreq = float(raw_eeg.info["sfreq"])
    _stage(f"asr_fit:cutoff={cutoff}:sfreq={sfreq}")
    asr = asrpy.ASR(sfreq=sfreq, cutoff=cutoff)
    asr.fit(raw_eeg)
    _stage("asr_transform")
    return asr.transform(raw_eeg)


def _fit_ica_and_labels(
    raw: mne.io.BaseRaw,
    asr_thresh: float | int | bool | None,
    random_seed: int,
    n_comp: int | None,
):
    raw_eeg = raw.copy().pick("eeg")
    raw_eeg, _ = mne.set_eeg_reference(raw_eeg, ref_channels="average")
    raw_eeg = _apply_asr(raw_eeg, asr_thresh)

    n_channels = len(raw_eeg.ch_names) - len(raw_eeg.info.get("bads", []))
    if n_channels < 3:
        raise RuntimeError(f"ICA requires at least 3 usable EEG channels; found {n_channels}")
    if n_comp is None:
        n_comp = min(n_channels - 1, 48)
    if not 1 < int(n_comp) < n_channels:
        raise ValueError(
            f"ICA n_components must be between 2 and {n_channels - 1}; got {n_comp}"
        )

    ica = ICA(
        n_components=int(n_comp),
        method="infomax",
        random_state=random_seed,
        fit_params={"extended": True},
        max_iter="auto",
    )
    decim = _ica_decimation(raw_eeg)
    _stage(f"ica_fit:n={n_comp}:decim={decim}")
    ica.fit(raw_eeg, decim=decim, reject_by_annotation=True, verbose=False)

    inst_for_iclabel = _iclabel_instance(raw_eeg)
    labels = _label_ica_components(inst_for_iclabel, ica)
    if inst_for_iclabel is not raw_eeg:
        del inst_for_iclabel
        gc.collect()

    eog_inds: list[int] = []
    eog_chs = [
        ch for ch, kind in zip(raw.ch_names, raw.get_channel_types()) if kind == "eog"
    ]
    if eog_chs:
        try:
            eog_inst = raw_eeg.copy()
            _add_channels_by_name(eog_inst, raw, eog_chs)
            eog_inds, _ = ica.find_bads_eog(eog_inst, verbose=False)
            del eog_inst
        except Exception:
            eog_inds = []

    return raw_eeg, ica, labels, list(eog_inds)


def run_asr_ica(
    raw: mne.io.BaseRaw,
    asr_thresh: float | int | bool | None = 20,
    random_seed: int = 420,
    n_comp: int | None = None,
    remove_cfa_flag: bool = False,
) -> tuple[mne.io.BaseRaw, ICA, dict[int, str]]:
    raw_eeg, ica, labels, eog_inds = _fit_ica_and_labels(
        raw, asr_thresh, random_seed, n_comp
    )

    keep_labels = {"brain", "other"}
    if not remove_cfa_flag:
        keep_labels.add("heart beat")
    bads = [
        index
        for index, label in enumerate(labels)
        if _normalise_iclabel_label(label) not in keep_labels
    ]
    bads = sorted(set(bads) | set(eog_inds))
    bad_label_dict = {
        index: ("eog" if index in eog_inds else labels[index])
        for index in bads
        if index < len(labels)
    }
    logging.info(
        "ICLabel-based ICA pruning (%s cardiac): %s",
        "removing" if remove_cfa_flag else "keeping",
        bad_label_dict,
    )

    eeg_clean = _apply_ica_excludes(raw_eeg, ica, bads)
    if not ica.exclude:
        bad_label_dict = {}
    if eeg_clean.info.get("bads"):
        eeg_clean.interpolate_bads(reset_bads=True)
    _add_non_eeg_channels(eeg_clean, raw)
    return eeg_clean, ica, bad_label_dict


def run_asr_only(
    raw: mne.io.BaseRaw,
    asr_thresh: float | int | bool | None = 20,
) -> mne.io.BaseRaw:
    """Apply ASR without ICA/ICLabel, preserving non-EEG channels."""
    if _normalise_asr_cutoff(asr_thresh) is None:
        return raw
    raw_eeg = raw.copy().pick("eeg")
    raw_eeg, _ = mne.set_eeg_reference(raw_eeg, ref_channels="average")
    eeg_clean = _apply_asr(raw_eeg, asr_thresh)
    _add_non_eeg_channels(eeg_clean, raw)
    return eeg_clean


def detect_r_peaks(
    raw: mne.io.BaseRaw,
    stim_channel: str = "STI 014",
    gap_threshold_factor: float = 2.0,
):
    sfreq = float(raw.info["sfreq"])
    ecg_chs = [
        ch for ch, kind in zip(raw.ch_names, raw.get_channel_types()) if kind == "ecg"
    ]
    if not ecg_chs:
        return raw, []
    if stim_channel in raw.ch_names:
        logging.warning(
            "Stim channel %s already exists; leaving it unchanged and skipping ECG events.",
            stim_channel,
        )
        return raw, []

    _stage("ecg_r_peaks")
    ecg = raw.get_data(picks=ecg_chs[0])[0]
    ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=sfreq)
    _signals, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sfreq)
    peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    if peaks.size < 2:
        return raw, []

    rr_sec = np.diff(peaks) / sfreq
    median_rr = float(np.median(rr_sec))
    if not np.isfinite(median_rr) or median_rr <= 0:
        return raw, []
    max_gap = gap_threshold_factor * median_rr
    all_peaks = list(peaks)
    for idx, interval in enumerate(rr_sec):
        if interval > max_gap:
            n_missing = max(0, int(np.round(interval / median_rr)) - 1)
            all_peaks.extend(
                peaks[idx] + int(j * median_rr * sfreq)
                for j in range(1, n_missing + 1)
            )
    all_peaks = np.unique(all_peaks).astype(int)
    all_peaks = all_peaks[(all_peaks >= 0) & (all_peaks < raw.n_times)]
    events = [(int(sample), 0, 1) for sample in all_peaks]

    stim_data = np.zeros((1, raw.n_times), dtype=np.float64)
    stim_data[0, all_peaks] = 1.0
    info_stim = mne.create_info([stim_channel], sfreq, ch_types=["stim"])
    stim_raw = mne.io.RawArray(stim_data, info_stim, verbose=False)
    raw.add_channels([stim_raw], force_update_info=True)
    return raw, events


def _read_raw_any(input_path: str, *, preload=True, verbose=False) -> mne.io.BaseRaw:
    suffix = Path(input_path).suffix.lower()
    if suffix == ".bdf":
        return mne.io.read_raw_bdf(input_path, preload=preload, verbose=verbose)
    if suffix == ".edf":
        return mne.io.read_raw_edf(input_path, preload=preload, verbose=verbose)
    if suffix == ".fif":
        return mne.io.read_raw_fif(input_path, preload=preload, verbose=verbose)
    if suffix == ".vhdr":
        return mne.io.read_raw_brainvision(input_path, preload=preload, verbose=verbose)
    raise ValueError(f"Unsupported input format: {input_path}")

def _part_fif_path(path: Path, kind: str) -> Path:
    path = Path(path)
    token = f".part-{os.getpid()}"
    name = path.name
    if kind == "raw":
        if name.endswith("_raw.fif"):
            name = name[:-8] + token + "_raw.fif"
        else:
            name = path.stem + token + "_raw.fif"
    elif kind == "ica":
        if name.endswith("_ica.fif"):
            name = name[:-8] + token + "_ica.fif"
        elif name.endswith("-ica.fif"):
            name = name[:-8] + token + "-ica.fif"
        else:
            name = path.stem + token + "_ica.fif"
    else:
        raise ValueError(f"Unknown FIF kind: {kind}")
    return path.with_name(name)


def _atomic_save_raw(raw: mne.io.BaseRaw, path: str | Path) -> Path:
    final = Path(path)
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = _part_fif_path(final, "raw")
    temporary.unlink(missing_ok=True)
    _stage(f"save_raw:{final.name}")
    try:
        raw.save(str(temporary), overwrite=True, verbose=False)
        os.replace(temporary, final)
    finally:
        temporary.unlink(missing_ok=True)
    return final


def _atomic_save_ica(ica: ICA, path: str | Path) -> Path:
    final = Path(path)
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = _part_fif_path(final, "ica")
    temporary.unlink(missing_ok=True)
    _stage(f"save_ica:{final.name}")
    try:
        ica.save(str(temporary), overwrite=True, verbose=False)
        os.replace(temporary, final)
    finally:
        temporary.unlink(missing_ok=True)
    return final


def _pyprep_segment_bounds(input_path: str, raw: mne.io.BaseRaw):
    if _CFG_RT is None or not _CFG_RT.pyprep_by_segment:
        return None
    return _snip_sample_bounds_for_input(input_path, raw)


def preprocess_edf(
    input_path: str,
    output_path: str | None = None,
    redo: bool = False,
    pyprep_dict: dict | None = None,
    asr_threshold=None,
    random_seed: int = 42,
    n_comp: int | None = None,
    logging_path: str | None = None,
    remove_cfa_override: bool | None = None,
    flip_ecg: bool = False,
    ecg_channel: str | None = None,
    stim_keep: list | None = None,
    spirometry_channel: str | None = None,
    spirometry_dir: str | Path | None = None,
):
    if pyprep_dict is None:
        pyprep_dict = dict(prep_params)
    if asr_threshold is None:
        asr_threshold = (
            getattr(_CFG_RT, "asr_cutoff", getattr(_CFG_RT, "use_asr", None))
            if _CFG_RT is not None
            else globals().get("do_asr")
        )

    if output_path is None:
        base = Path(input_path).stem + "_pp_raw.fif"
        output_path = str(Path(output_dir) / base)
    if logging_path is None or str(logging_path).strip() == "":
        logging_path = str(Path(output_dir) / "logs" / "preprocessing.tsv")

    outp = Path(output_path)
    if not redo and outp.exists():
        try:
            raw_cached = mne.io.read_raw_fif(outp, preload=True, verbose=False)
        except Exception as exc:
            logging.warning("Existing output is unreadable and will be replaced: %s", exc)
            outp.unlink(missing_ok=True)
        else:
            logging.info("Output exists and is readable, skipping: %s", outp)
            had_rsp = _has_spirometry_annotations(raw_cached)
            spirometry_rec = save_spirometry_analysis(
                raw_cached,
                spirometry_channel,
                spirometry_dir or (Path(output_dir) / "spirometry"),
                _preproc_base_from_output(outp),
                redo=False,
            )
            if spirometry_rec.get("spirometry_status") == "ok" and not had_rsp:
                _atomic_save_raw(raw_cached, outp)
            rec = {
                "utc": _now_utc_iso(),
                "status": "OK",
                "stage": "cached",
                "input": input_path,
                "output": str(outp),
                "n_channels": len(raw_cached.ch_names),
                "remove_cfa": str(remove_cfa_override),
                "remove_cfa_mode": str(getattr(_CFG_RT, "remove_cfa_mode", "")),
                "flip_ecg": str(bool(flip_ecg)),
            }
            rec.update(spirometry_rec)
            _append_preproc_tsv(logging_path, rec)
            return raw_cached

    try:
        _stage("read_raw")
        raw = _read_raw_any(input_path, preload=True, verbose=False)

        aux_like = ["IBI", "BURSTS", "SUPPR", "T1", "T2", "26", "27", "28", "29", "30"]
        present_aux = [ch for ch in aux_like if ch in raw.ch_names]
        if present_aux:
            raw.set_channel_types({ch: "misc" for ch in present_aux})
        _mark_spirometry_channel(raw, spirometry_channel)
        _mark_ecg_channels(raw, ecg_channel)
        if flip_ecg:
            _flip_ecg_channels(raw, ecg_channel)

        if target_sfreq and not np.isclose(raw.info["sfreq"], float(target_sfreq)):
            _stage(f"resample:{raw.info['sfreq']}->{target_sfreq}")
            raw.resample(float(target_sfreq), npad="auto", n_jobs=1)

        if stim_keep:
            for ch in stim_keep:
                if ch in raw.ch_names:
                    raw.set_channel_types({ch: "stim"})
        _mark_ecg_channels(raw, ecg_channel)
        _stage("montage")
        raw = standardise_and_montage(raw)
        if sum(kind == "eeg" for kind in raw.get_channel_types()) == 0:
            raise ValueError(
                "No EEG channels remain after montage/drop; check montage name and channel labels."
            )

        do_pyprep_flag = bool(
            getattr(_CFG_RT, "use_pyprep", globals().get("do_pyprep", True))
        )
        if do_pyprep_flag:
            raw = run_pyprep(
                raw,
                pyprep_dict,
                random_seed=random_seed,
                ransac=bool(
                    getattr(_CFG_RT, "prep_ransac", pyprep_dict.get("ransac", False))
                ),
                channel_wise=bool(getattr(_CFG_RT, "pyprep_channel_wise", False)),
                max_chunk_size=getattr(_CFG_RT, "pyprep_max_chunk_size", 8),
                segment_bounds=_pyprep_segment_bounds(input_path, raw),
            )

        # Filtering is independent of PyPREP.  In the previous implementation,
        # disabling PyPREP also silently disabled the configured band-pass.
        raw = _filter_eeg_once(
            raw,
            pyprep_dict.get("l_freq", getattr(_CFG_RT, "high_pass", 1.0)),
            pyprep_dict.get("h_freq", getattr(_CFG_RT, "low_pass", 100.0)),
        )

        remove_flag = (
            bool(remove_cfa)
            if remove_cfa_override is None
            else bool(remove_cfa_override)
        )
        use_ica_flag = bool(getattr(_CFG_RT, "use_ica", True))
        if use_ica_flag:
            raw, ica_model, bad_label_dict = run_asr_ica(
                raw,
                asr_thresh=asr_threshold,
                random_seed=random_seed,
                n_comp=n_comp,
                remove_cfa_flag=remove_flag,
            )
        else:
            raw = run_asr_only(raw, asr_threshold)
            ica_model = None
            bad_label_dict = {}

        raw, events = detect_r_peaks(raw)
        _stage("spirometry")
        spirometry_rec = save_spirometry_analysis(
            raw,
            spirometry_channel,
            spirometry_dir or (Path(output_dir) / "spirometry"),
            _preproc_base_from_output(outp),
            redo=redo,
        )
        raw.info["bads"] = []
        raw.set_meas_date(1)
        _atomic_save_raw(raw, outp)

        ica_path = ""
        if ica_model is not None:
            stem_no_pp = outp.stem.replace("_pp_raw", "")
            ica_path_obj = outp.with_name(stem_no_pp + "_ica.fif")
            _atomic_save_ica(ica_model, ica_path_obj)
            ica_path = str(ica_path_obj)

        rec = {
            "utc": _now_utc_iso(),
            "status": "OK",
            "stage": "save",
            "input": input_path,
            "output": str(outp),
            "n_channels": len(raw.ch_names),
            "n_events": len(events),
            "bad_channels": ";".join(raw.info.get("bads", [])),
            "remove_cfa": str(remove_flag),
            "remove_cfa_mode": str(getattr(_CFG_RT, "remove_cfa_mode", "")),
            "flip_ecg": str(bool(flip_ecg)),
            "asr_threshold": str(_normalise_asr_cutoff(asr_threshold) or ""),
            "n_comp": str(n_comp or ""),
            "stim_keep": ";".join(stim_keep) if stim_keep else "",
            "montage_name": str(getattr(_CFG_RT, "montage_name", "auto")),
            "ica_path": ica_path,
            "ica_bads": ";".join(str(key) for key in sorted(bad_label_dict)),
            "ica_bad_labels": ";".join(
                str(bad_label_dict[key]) for key in sorted(bad_label_dict)
            ),
        }
        rec.update(spirometry_rec)
        _append_preproc_tsv(logging_path, rec)
        return raw

    except Exception as exc:
        _append_preproc_tsv(
            logging_path,
            {
                "utc": _now_utc_iso(),
                "status": "FAIL",
                "stage": "preprocess_edf",
                "input": input_path,
                "output": str(outp),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise



def _fit_asr_ica_iclabel_once(
    raw: mne.io.BaseRaw,
    asr_thresh: float | int | bool | None = 20,
    random_seed: int = 420,
    n_comp: int | None = None,
):
    """Run optional ASR, fit ICA once, and derive both ICLabel rules."""
    raw_eeg, ica, labels, eog_inds = _fit_ica_and_labels(
        raw, asr_thresh, random_seed, n_comp
    )
    labels = [_normalise_iclabel_label(label) for label in labels]

    bads_remove = [
        index for index, label in enumerate(labels) if label not in {"brain", "other"}
    ]
    bads_keep = [
        index
        for index, label in enumerate(labels)
        if label not in {"brain", "heart beat", "other"}
    ]
    bads_remove = sorted(set(bads_remove) | set(eog_inds))
    bads_keep = sorted(set(bads_keep) | set(eog_inds))

    bad_map_remove = {
        index: ("eog" if index in eog_inds else labels[index])
        for index in bads_remove
        if index < len(labels)
    }
    bad_map_keep = {
        index: ("eog" if index in eog_inds else labels[index])
        for index in bads_keep
        if index < len(labels)
    }
    return (
        raw_eeg,
        ica,
        labels,
        bads_remove,
        bads_keep,
        bad_map_remove,
        bad_map_keep,
        eog_inds,
    )


def preprocess_edf_both(
    input_path: str,
    output_path_remove: str,
    output_path_keep: str,
    redo: bool = False,
    pyprep_dict: dict | None = None,
    asr_threshold=None,
    random_seed: int = 42,
    n_comp: int | None = None,
    logging_path: str | None = None,
    flip_ecg: bool = False,
    ecg_channel: str | None = None,
    stim_keep: list | None = None,
    spirometry_channel: str | None = None,
    spirometry_dir: str | Path | None = None,
):
    """Preprocess once, then save cardiac-removed and cardiac-retained variants."""
    if logging_path is None or str(logging_path).strip() == "":
        logging_path = str(Path(output_dir) / "logs" / "preprocessing.tsv")
    if pyprep_dict is None:
        pyprep_dict = dict(prep_params)
    if asr_threshold is None:
        asr_threshold = (
            getattr(_CFG_RT, "asr_cutoff", getattr(_CFG_RT, "use_asr", None))
            if _CFG_RT is not None
            else globals().get("do_asr")
        )

    output_remove = Path(output_path_remove)
    output_keep = Path(output_path_keep)
    if not redo and output_remove.exists() and output_keep.exists():
        readable = True
        for out_path in (output_remove, output_keep):
            try:
                cached = mne.io.read_raw_fif(out_path, preload=True, verbose=False)
            except Exception as exc:
                logging.warning("Unreadable cached output %s: %s", out_path, exc)
                out_path.unlink(missing_ok=True)
                readable = False
                continue
            had_rsp = _has_spirometry_annotations(cached)
            spirometry_rec = save_spirometry_analysis(
                cached,
                spirometry_channel,
                spirometry_dir or (Path(output_dir) / "spirometry"),
                _preproc_base_from_output(out_path),
                redo=False,
            )
            if spirometry_rec.get("spirometry_status") == "ok" and not had_rsp:
                _atomic_save_raw(cached, out_path)
            rec = {
                "utc": _now_utc_iso(),
                "status": "OK",
                "stage": "cached",
                "input": input_path,
                "output": str(out_path),
                "n_channels": len(cached.ch_names),
                "remove_cfa_mode": "both",
                "flip_ecg": str(bool(flip_ecg)),
            }
            rec.update(spirometry_rec)
            _append_preproc_tsv(logging_path, rec)
            del cached
        if readable:
            return (
                mne.io.read_raw_fif(output_remove, preload=False, verbose=False),
                mne.io.read_raw_fif(output_keep, preload=False, verbose=False),
            )

    try:
        _stage("read_raw")
        raw = _read_raw_any(input_path, preload=True, verbose=False)

        aux_like = ["IBI", "BURSTS", "SUPPR", "T1", "T2", "26", "27", "28", "29", "30"]
        present_aux = [ch for ch in aux_like if ch in raw.ch_names]
        if present_aux:
            raw.set_channel_types({ch: "misc" for ch in present_aux})
        _mark_spirometry_channel(raw, spirometry_channel)
        _mark_ecg_channels(raw, ecg_channel)
        if flip_ecg:
            _flip_ecg_channels(raw, ecg_channel)

        if target_sfreq and not np.isclose(raw.info["sfreq"], float(target_sfreq)):
            _stage(f"resample:{raw.info['sfreq']}->{target_sfreq}")
            raw.resample(float(target_sfreq), npad="auto", n_jobs=1)

        if stim_keep:
            for ch in stim_keep:
                if ch in raw.ch_names:
                    raw.set_channel_types({ch: "stim"})
        _mark_ecg_channels(raw, ecg_channel)
        _stage("montage")
        raw = standardise_and_montage(raw)
        if sum(kind == "eeg" for kind in raw.get_channel_types()) == 0:
            raise ValueError(
                "No EEG channels remain after montage/drop; check montage name and channel labels."
            )

        if bool(getattr(_CFG_RT, "use_pyprep", globals().get("do_pyprep", True))):
            raw = run_pyprep(
                raw,
                pyprep_dict,
                random_seed=random_seed,
                ransac=bool(
                    getattr(_CFG_RT, "prep_ransac", pyprep_dict.get("ransac", False))
                ),
                channel_wise=bool(getattr(_CFG_RT, "pyprep_channel_wise", False)),
                max_chunk_size=getattr(_CFG_RT, "pyprep_max_chunk_size", 8),
                segment_bounds=_pyprep_segment_bounds(input_path, raw),
            )

        raw = _filter_eeg_once(
            raw,
            pyprep_dict.get("l_freq", getattr(_CFG_RT, "high_pass", 1.0)),
            pyprep_dict.get("h_freq", getattr(_CFG_RT, "low_pass", 100.0)),
        )

        # ECG and respiration depend only on auxiliary channels, so calculate
        # them once rather than once for each ICA exclusion rule.
        raw, events = detect_r_peaks(raw)
        _stage("spirometry")
        spirometry_rec = save_spirometry_analysis(
            raw,
            spirometry_channel,
            spirometry_dir or (Path(output_dir) / "spirometry"),
            _preproc_base_from_output(output_remove),
            redo=redo,
        )

        use_ica_flag = bool(getattr(_CFG_RT, "use_ica", True))
        if use_ica_flag:
            (
                raw_eeg_proc,
                ica,
                _labels,
                bads_remove,
                bads_keep,
                bad_map_remove,
                bad_map_keep,
                _eog_inds,
            ) = _fit_asr_ica_iclabel_once(
                raw,
                asr_thresh=asr_threshold,
                random_seed=random_seed,
                n_comp=n_comp,
            )

            variants = (
                (output_remove, "remove", bads_remove, bad_map_remove),
                (output_keep, "keep", bads_keep, bad_map_keep),
            )
            for outp, mode, excludes, label_map in variants:
                ica_local = ica.copy()
                raw_out = _apply_ica_excludes(raw_eeg_proc, ica_local, list(excludes))
                if not ica_local.exclude:
                    label_map = {}
                if raw_out.info.get("bads"):
                    raw_out.interpolate_bads(reset_bads=True)
                _add_non_eeg_channels(raw_out, raw)
                raw_out.info["bads"] = []
                raw_out.set_meas_date(1)
                _atomic_save_raw(raw_out, outp)

                stem_no_pp = outp.stem.replace("_pp_raw", "")
                ica_path = outp.with_name(stem_no_pp + "_ica.fif")
                _atomic_save_ica(ica_local, ica_path)

                rec = {
                    "utc": _now_utc_iso(),
                    "status": "OK",
                    "stage": "save",
                    "input": input_path,
                    "output": str(outp),
                    "n_channels": len(raw_out.ch_names),
                    "n_events": len(events),
                    "bad_channels": "",
                    "remove_cfa": str(mode == "remove"),
                    "remove_cfa_mode": "both",
                    "flip_ecg": str(bool(flip_ecg)),
                    "asr_threshold": str(_normalise_asr_cutoff(asr_threshold) or ""),
                    "n_comp": str(n_comp or ""),
                    "stim_keep": ";".join(stim_keep) if stim_keep else "",
                    "montage_name": str(getattr(_CFG_RT, "montage_name", "auto")),
                    "ica_path": str(ica_path),
                    "ica_bads": ";".join(str(key) for key in sorted(label_map)),
                    "ica_bad_labels": ";".join(
                        str(label_map[key]) for key in sorted(label_map)
                    ),
                }
                rec.update(spirometry_rec)
                _append_preproc_tsv(logging_path, rec)
                del raw_out, ica_local
                gc.collect()
        else:
            processed = run_asr_only(raw, asr_threshold)
            processed.info["bads"] = []
            processed.set_meas_date(1)
            for outp, mode in ((output_remove, "remove"), (output_keep, "keep")):
                _atomic_save_raw(processed, outp)
                rec = {
                    "utc": _now_utc_iso(),
                    "status": "OK",
                    "stage": "save_no_ica",
                    "input": input_path,
                    "output": str(outp),
                    "n_channels": len(processed.ch_names),
                    "n_events": len(events),
                    "bad_channels": "",
                    "remove_cfa": str(mode == "remove"),
                    "remove_cfa_mode": "both",
                    "flip_ecg": str(bool(flip_ecg)),
                    "asr_threshold": str(_normalise_asr_cutoff(asr_threshold) or ""),
                    "n_comp": "",
                    "stim_keep": ";".join(stim_keep) if stim_keep else "",
                    "montage_name": str(getattr(_CFG_RT, "montage_name", "auto")),
                }
                rec.update(spirometry_rec)
                _append_preproc_tsv(logging_path, rec)

        return (
            mne.io.read_raw_fif(output_remove, preload=False, verbose=False),
            mne.io.read_raw_fif(output_keep, preload=False, verbose=False),
        )

    except Exception as exc:
        logging.exception("preprocess_edf_both failed for %s", input_path)
        _append_preproc_tsv(
            logging_path,
            {
                "utc": _now_utc_iso(),
                "status": "FAIL",
                "stage": "preprocess_edf_both",
                "input": input_path,
                "output": f"{output_remove} | {output_keep}",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


# ----------------------------------- CLI (optional) ----------------------- #
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EDF→FIF EEG preprocessing")
    parser.add_argument("edf", help="Path to input .edf file")
    parser.add_argument("--out", help="Output .fif path (default: output_dir/<name>_pp_raw.fif)")
    parser.add_argument("--redo", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--n_comp", type=int, default=None, help="ICA components (default: auto)")
    parser.add_argument("--asr", type=float, default=None, help="ASR cutoff (default: config.do_asr)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    preprocess_edf(
        args.edf,
        output_path=args.out,
        redo=args.redo,
        n_comp=args.n_comp,
        asr_threshold=args.asr,
    )
