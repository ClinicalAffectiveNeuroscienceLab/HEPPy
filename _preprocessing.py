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
import logging
from datetime import datetime
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import asrpy

import mne
import numpy as np
import neurokit2 as nk
import pyprep
from mne.preprocessing import ICA
from mne_icalabel import label_components

# ------------------------- Logging helpers ------------------------- #

def _now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

# Fixed TSV schema so all runs (success/fail) align.
_PREPROC_TSV_FIELDS = [
    "utc", "status", "stage",
    "input", "output",
    "n_channels", "n_events", "bad_channels",
    "remove_cfa", "remove_cfa_mode", "flip_ecg",
    "asr_threshold", "n_comp", "stim_keep",
    "montage_name", "ica_path", "ica_bads", "ica_bad_labels",
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
    global _CFG_RT, output_dir, target_sfreq, do_asr, remove_cfa, remove_cfa_mode, log_file, prep_params
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
    nm = nm.upper().replace('EEG ', '')
    nm = nm.split('-')[0]
    nm = nm.replace('FP', 'Fp').replace('Z', 'z')
    repl = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}
    return repl.get(nm, nm)

AUX_EOG = {'ROC', 'LOC'}
AUX_STIM = {'PHOTIC'}
AUX_MISC = {'IBI', 'BURSTS', 'SUPPR'}

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
        elif 'ECG' in up or 'EKG' in up:
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
    ch_set = set([_canonicalise_name(c) for c in raw.ch_names])
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
    raw.rename_channels({ch: _canonicalise_name(ch) for ch in raw.ch_names})
    # Decide montage: explicit, auto-detect, or fallback
    m_name = _CFG_RT.montage_name if _CFG_RT else None
    if m_name is None or str(m_name).lower() in ("", "auto"):
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
    if drop_eeg:
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

# ------------------------------- Core steps -------------------------------- #

def run_pyprep(
    raw: mne.io.BaseRaw,
    param_dict: dict | None = None,
    random_seed: int = 42,
    ransac: bool = True,
    channel_wise: bool = True,
) -> mne.io.BaseRaw:
    if param_dict is None:
        param_dict = prep_params
    raw_eeg = raw.copy().pick('eeg')
    safe_params = _prune_prep_params_for_raw(param_dict, raw_eeg)
    trimmed_mont = _trim_eeg_montage_no_nan(raw_eeg)

    # Ensure keys PyPREP expects are present to avoid KeyError
    safe_params.setdefault("line_freqs", ())
    safe_params.setdefault("l_freq", None)
    safe_params.setdefault("h_freq", None)

    prep = pyprep.PrepPipeline(
        raw=raw_eeg, montage=trimmed_mont, prep_params=safe_params,
        random_state=random_seed, ransac=ransac, channel_wise=channel_wise,
    )
    try:
        with open(os.devnull, 'w') as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
            result = prep.fit()
    except Exception as e:
        print("→ EEG picks:", sorted(raw_eeg.ch_names))
        if trimmed_mont is not None:
            print("→ trimmed EEG montage ch_pos:",
                  sorted((_finite_ch_pos(trimmed_mont.get_positions().get('ch_pos', {}))).keys()))
        print("→ params (ref/reref/eog/ransac/interp/corr):",
              {k: safe_params.get(k) for k in ("ref_chs","reref_chs","eog_chs",
                                               "ransac_channel_picks","interpolation_channel_picks","corr_chs")})
        raise RuntimeError(f"PyPREP failed: {e}") from e

    cleaned_eeg = result.raw_eeg
    try:
        cleaned_eeg.interpolate_bads(reset_bads=False)
    except Exception:
        raise RuntimeError("PyPREP bad channel interpolation failed.")
    
    cleaned_eeg.info['bads'] = []

    l_freq = safe_params.get('l_freq', None)
    h_freq = safe_params.get('h_freq', None)

    # if the raw is already filtered with h_freq = h_freq, skip filtering
    if h_freq is not None and raw_eeg.info.get('highpass', None) is not None:
        if np.isclose(raw_eeg.info['highpass'], float(h_freq)):
            h_freq = None
    # if the raw is already filtered with l_freq = l_freq, skip filtering
    if l_freq is not None and raw_eeg.info.get('lowpass', None) is not None:
        if np.isclose(raw_eeg.info['lowpass'], float(l_freq)):
            l_freq = None

    if l_freq is not None or h_freq is not None:
        cleaned_eeg.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')

    non_eeg = raw.copy().pick([ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t != 'eeg'])
    if len(non_eeg.ch_names):
        cleaned_eeg.add_channels([non_eeg], force_update_info=True)

    return cleaned_eeg

def run_asr_ica(
    raw: mne.io.BaseRaw,
    asr_thresh: float | int | bool = 20,
    random_seed: int = 420,
    n_comp: int | None = None,
    remove_cfa_flag: bool = False,
) -> tuple[mne.io.BaseRaw, ICA]:
    raw_eeg = raw.copy().pick('eeg')
    raw_eeg, _ = mne.set_eeg_reference(raw_eeg, ref_channels='average')

    if asr_thresh:
        asr = asrpy.ASR(sfreq=raw_eeg.info['sfreq'], cutoff=asr_thresh)
        asr.fit(raw_eeg.copy().resample(100))
        raw_eeg = asr.transform(raw_eeg)

    n_channels = len(raw_eeg.ch_names) - len(raw_eeg.info.get('bads', []))
    if n_comp is None:
        n_comp = max(2, min(n_channels - 1, 48))
    ica = ICA(n_components=n_comp, method='infomax', random_state=random_seed, fit_params={"extended": True})
    ica.fit(raw_eeg, decim=3)

    iclabel_result = label_components(raw_eeg, ica, method="iclabel")
    labels = iclabel_result.get('labels', [])
    if remove_cfa_flag:
        bads = [i for i, lab in enumerate(labels) if lab not in ('brain', 'other')]
        bad_label_dict = {i: label for i, label in enumerate(labels) if i in bads}
        logging.info(f"ICLabel-based ICA pruning (removing cardiac): {bad_label_dict}")
    else:
        bads = [i for i, lab in enumerate(labels) if lab not in ('brain', 'heart', 'other')]
        bad_label_dict = {i: label for i, label in enumerate(labels) if i in bads}
        logging.info(f"ICLabel-based ICA pruning (not removing cardiac): {bad_label_dict}")

    # see if there are any EOG leads to help identify EOG components
    if any(t == 'eog' for t in raw.get_channel_types()):
        try:
            eog_inds, _ = ica.find_bads_eog(raw_eeg)
            bads = sorted(set(bads) | set(eog_inds))
        except Exception:
            pass

    ica.exclude = bads
    eeg_clean = ica.apply(raw_eeg)
    eeg_clean.interpolate_bads(reset_bads=True)

    non_eeg = raw.copy().pick([ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t != 'eeg'])
    if len(non_eeg.ch_names):
        eeg_clean.add_channels([non_eeg], force_update_info=True)

    return eeg_clean, ica, bad_label_dict

def detect_r_peaks(raw: mne.io.BaseRaw, stim_channel: str = 'STI 014', gap_threshold_factor: float = 2.0):
    sfreq = raw.info['sfreq']
    ecg_chs = [ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t == 'ecg']
    if not ecg_chs:
        return raw, []
    ecg_channel = ecg_chs[0]
    ecg = raw.get_data(picks=ecg_channel)[0]
    ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=sfreq)
    signals, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sfreq)
    peaks = info.get('ECG_R_Peaks', None)
    if peaks is None or len(peaks) < 2:
        return raw, []
    rr_samples = np.diff(peaks)
    rr_sec = rr_samples / sfreq
    median_rr = float(np.median(rr_sec))
    max_gap = gap_threshold_factor * median_rr
    all_peaks = list(peaks)
    for idx, interval in enumerate(rr_sec):
        if interval > max_gap:
            n_missing = int(np.round(interval / median_rr)) - 1
            for j in range(1, n_missing + 1):
                new_sample = peaks[idx] + int(j * median_rr * sfreq)
                all_peaks.append(new_sample)
    all_peaks = np.unique(all_peaks).astype(int)
    events = [(int(s), 0, 1) for s in all_peaks if 0 <= int(s) < raw.n_times]
    stim_data = np.zeros((1, raw.n_times), dtype=int)
    for sample, _, _ in events:
        stim_data[0, sample] = 1
    info_stim = mne.create_info([stim_channel], sfreq, ch_types=['stim'])
    stim_raw = mne.io.RawArray(stim_data, info_stim)
    raw.add_channels([stim_raw], force_update_info=True)
    return raw, events

# ------------------------------- Orchestration ---------------------------- #

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
    stim_keep: list | None = None,
):
    if pyprep_dict is None:
        try:
            pyprep_dict = prep_params
        except Exception:
            if _CFG_RT is not None:
                pyprep_dict = {
                    "line_freqs": _CFG_RT.line_freqs,
                    "ref_chs": _CFG_RT.ref_chs,
                    "reref_chs": _CFG_RT.reref_chs,
                    "l_freq": _CFG_RT.high_pass,
                    "h_freq": _CFG_RT.low_pass,
                    "ransac": _CFG_RT.prep_ransac,
                }
            else:
                pyprep_dict = {
                    "line_freqs": (50.0, 100.0),
                    "ref_chs": "eeg",
                    "reref_chs": "eeg",
                    "l_freq": 1.0,
                    "h_freq": 100.0,
                    "ransac": True,
                }
    if asr_threshold is None:
        asr_threshold = do_asr

    # resolve output path
    if output_path is None:
        base = Path(input_path).stem + "_pp_raw.fif"
        output_path = str(Path(output_dir) / base)

    # resolve log path (safe default if config.log_file is empty)
    if logging_path is None or str(logging_path).strip() == "":
        logging_path = str(Path(output_dir) / "logs" / "preprocessing.tsv")

    # idempotency
    if not redo and Path(output_path).exists():
        logging.info(f"Output exists, skipping: {output_path}")
        return mne.io.read_raw_fif(output_path, preload=True)

    # 1) Read EDF (preload)
    if Path(input_path).suffix.lower() == ".bdf":
        raw = mne.io.read_raw_bdf(input_path, preload=True, verbose=False)
    elif Path(input_path).suffix.lower() == ".edf":
        raw = mne.io.read_raw_edf(input_path, preload=True, verbose=False)
    elif Path(input_path).suffix.lower() == ".fif":
        raw = mne.io.read_raw_fif(input_path, preload=True, verbose=False)
    else:
        raise ValueError(f"Unsupported input format: {input_path}")

    # 2) Type auxiliaries early (no n_times change)
    aux_like = ["IBI", "BURSTS", "SUPPR", "T1", "T2", "26", "27", "28", "29", "30"]
    present_aux = [ch for ch in aux_like if ch in raw.ch_names]
    if present_aux:
        raw.set_channel_types({ch: "misc" for ch in present_aux})

    # Optional ECG polarity flip
    if flip_ecg:
        ecg_idx = mne.pick_types(raw.info, ecg=True)
        if len(ecg_idx):
            data = raw.get_data(picks=ecg_idx)
            raw._data[ecg_idx] = -data

    # 3) Global resample once (for identical n_times across all channels)
    if target_sfreq and not np.isclose(raw.info["sfreq"], float(target_sfreq)):
        raw.resample(float(target_sfreq), npad="auto")

    # 4) Rename / classify / montage
    
    # set stim leads to stim type before montage
    if stim_keep:
        for ch in stim_keep:
            if ch in raw.ch_names:
                raw.set_channel_types({ch: "stim"})

    # do the same for ECG leads (leads with ecg or ekg in name.lower())
    for ch in raw.ch_names:
        if 'ecg' in ch.lower() or 'ekg' in ch.lower():
            raw.set_channel_types({ch: "ecg"})

    raw = standardise_and_montage(raw)

    # Sanity: ensure EEG remains
    if sum(1 for t in raw.get_channel_types() if t == "eeg") == 0:
        raise ValueError("No EEG channels remain after montage/drop; check montage name and channel labels.")

    # 5) PyPREP on EEG-only; recombine with non-EEG
    try:
        do_pyprep_flag = do_pyprep  # global from set_runtime_config
    except NameError:
        do_pyprep_flag = getattr(_CFG_RT, "use_pyprep", True)
    if do_pyprep_flag:
        raw = run_pyprep(raw, pyprep_dict, random_seed=random_seed, ransac=bool(_CFG_RT.prep_ransac))

    # 6) ASR + ICA (+/- remove cardiac via ICLabel)
    if remove_cfa_override is None:
        remove_flag = bool(remove_cfa)
    else:
        remove_flag = bool(remove_cfa_override)
    raw, ica_model, bad_label_dict = run_asr_ica(
        raw,
        asr_thresh=asr_threshold,
        random_seed=random_seed,
        n_comp=n_comp,
        remove_cfa_flag=remove_flag,
    )

    # 7) ECG → R-peaks → stim events
    raw, events = detect_r_peaks(raw)

    raw.info["bads"] = []  # clear any stragglers

    # 8) Save (guard empty parents)
    outp = Path(output_path)
    if str(outp.parent) not in ("", "."):
        outp.parent.mkdir(parents=True, exist_ok=True)
    raw.set_meas_date(1)
    raw.save(str(outp), overwrite=True)

    # save ICA separately (fail fast if it cannot be written)
    ica_path = ""
    if ica_model is not None:
        stem_no_pp = outp.stem.replace("_pp_raw", "")
        ica_path = outp.with_name(stem_no_pp + "_ica.fif")
        ica_path.parent.mkdir(parents=True, exist_ok=True)
        ica_model.save(str(ica_path), overwrite=True)

    # 9) Log (guard empty parents)
    rec = {
        'input': input_path,
        'output': str(outp),
        'n_channels': len(raw.ch_names),
        'n_events': len(events),
        'bad_channels': ';'.join(raw.info.get('bads', [])) if raw.info.get('bads') else '',
        'remove_cfa': str(remove_flag),
        'remove_cfa_mode': str(getattr(_CFG_RT, "remove_cfa_mode", "")),
        'flip_ecg': str(bool(flip_ecg)),
        'asr_threshold': str(asr_threshold),
        'n_comp': str(n_comp),
        'stim_keep': ";".join(stim_keep) if stim_keep else "",
        'ica_path': ica_path,
        'channels_dropped': str(len(drop_candidates)) if 'drop_candidates' in locals() else "0",
        'montage_name': str(getattr(_CFG_RT, "montage_name", "auto")),
        'ica_bads': ';'.join(str(key) for key in bad_label_dict.keys()),
        'ica_bad_labels': ';'.join(str(val) for val in bad_label_dict.values()),
    }
    logp = Path(logging_path)
    if str(logp.parent) not in ("", "."):
        logp.parent.mkdir(parents=True, exist_ok=True)
    file_exists = logp.exists() and logp.stat().st_size > 0
    with open(str(logp), 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rec.keys(), delimiter='\t')
        if not file_exists:
            w.writeheader()
        w.writerow(rec)

    return raw




def _fit_asr_ica_iclabel_once(
    raw: mne.io.BaseRaw,
    asr_thresh: float | int | bool = 20,
    random_seed: int = 420,
    n_comp: int | None = None,
):
    """Run (optional) ASR, fit ICA once, and run ICLabel.

    Returns
    -------
    raw_eeg_proc : mne.io.BaseRaw
        EEG-only data after referencing and optional ASR, used for ICA fitting.
    ica : mne.preprocessing.ICA
        Fitted ICA model.
    labels : list[str]
        ICLabel class label per component.
    bad_label_dict : dict[int, str]
        Mapping from excluded component index to ICLabel label for the *remove-CFA* rule
        (i.e. excluding anything not in ('brain','other')).
    bad_label_dict_keep : dict[int, str]
        Mapping for the *keep-CFA* rule (i.e. excluding anything not in ('brain','heart','other')).
    eog_inds : list[int]
        Components additionally flagged by find_bads_eog (may be empty).
    """
    raw_eeg = raw.copy().pick('eeg')
    raw_eeg, _ = mne.set_eeg_reference(raw_eeg, ref_channels='average')

    # Optional ASR (fit at 100 Hz to reduce cost, consistent with existing code)
    if asr_thresh:
        asr = asrpy.ASR(sfreq=raw_eeg.info['sfreq'], cutoff=asr_thresh)
        asr.fit(raw_eeg.copy().resample(100))
        raw_eeg = asr.transform(raw_eeg)

    n_channels = len(raw_eeg.ch_names) - len(raw_eeg.info.get('bads', []))
    if n_comp is None:
        n_comp = max(2, min(n_channels - 1, 48))

    ica = ICA(
        n_components=n_comp,
        method='infomax',
        random_state=random_seed,
        fit_params={"extended": True},
    )
    ica.fit(raw_eeg, decim=3)

    # ICLabel expects 1–100 Hz bandpass; use a filtered *copy* for feature extraction.
    inst_for_iclabel = raw_eeg.copy()
    try:
        inst_for_iclabel.filter(l_freq=1.0, h_freq=100.0, fir_design='firwin', verbose=False)
    except Exception:
        # If filtering fails for some reason, proceed (mne_icalabel will likely warn / fail).
        pass

    iclabel_result = label_components(inst_for_iclabel, ica, method="iclabel")
    labels = list(iclabel_result.get('labels', []))

    bads_remove = [i for i, lab in enumerate(labels) if lab not in ('brain', 'other')]
    bads_keep = [i for i, lab in enumerate(labels) if lab not in ('brain', 'heart', 'other')]

    bad_label_dict_remove = {i: labels[i] for i in bads_remove if i < len(labels)}
    bad_label_dict_keep = {i: labels[i] for i in bads_keep if i < len(labels)}

    # EOG-based IC detection (optional)
    eog_inds = []
    if any(t == 'eog' for t in raw.get_channel_types()):
        try:
            eog_inds, _ = ica.find_bads_eog(raw_eeg)
        except Exception:
            eog_inds = []

    if eog_inds:
        # EOG inds should be excluded in both variants
        bads_remove = sorted(set(bads_remove) | set(eog_inds))
        bads_keep = sorted(set(bads_keep) | set(eog_inds))
        for i in eog_inds:
            if i < len(labels):
                bad_label_dict_remove.setdefault(i, "eog")
                bad_label_dict_keep.setdefault(i, "eog")

    return raw_eeg, ica, labels, bads_remove, bads_keep, bad_label_dict_remove, bad_label_dict_keep, eog_inds


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
    stim_keep: list | None = None,
):
    """Preprocess once up to ICA+ICLabel, then write *two* outputs.

    - **remove**: excludes ICLabel components not in ('brain','other')
    - **keep**: excludes ICLabel components not in ('brain','heart','other')

    This avoids running read/resample/montage/PyPREP/ASR/ICA/ICLabel twice.
    """
    # Resolve log path
    if logging_path is None or str(logging_path).strip() == "":
        logging_path = str(Path(output_dir) / "logs" / "preprocessing.tsv")

    # Idempotency
    if (not redo) and Path(output_path_remove).exists() and Path(output_path_keep).exists():
        logging.info("Both outputs exist, skipping: %s AND %s", output_path_remove, output_path_keep)
        return (
            mne.io.read_raw_fif(output_path_remove, preload=True),
            mne.io.read_raw_fif(output_path_keep, preload=True),
        )

    # Shared steps: read, resample, montage, PyPREP
    try:
        # (1) Read
        suf = Path(input_path).suffix.lower()
        if suf == ".bdf":
            raw = mne.io.read_raw_bdf(input_path, preload=True, verbose=False)
        elif suf == ".edf":
            raw = mne.io.read_raw_edf(input_path, preload=True, verbose=False)
        elif suf == ".fif":
            raw = mne.io.read_raw_fif(input_path, preload=True, verbose=False)
        else:
            raise ValueError(f"Unsupported input format: {input_path}")

        # (2) Re-type obvious aux channels (keep consistent with preprocess_edf)
        aux_like = ["IBI", "BURSTS", "SUPPR", "T1", "T2", "26", "27", "28", "29", "30"]
        present_aux = [ch for ch in aux_like if ch in raw.ch_names]
        if present_aux:
            raw.set_channel_types({ch: "misc" for ch in present_aux})

        # Flip ECG if requested
        if flip_ecg:
            ecg_idx = mne.pick_types(raw.info, ecg=True)
            if len(ecg_idx):
                data = raw.get_data(picks=ecg_idx)
                raw._data[ecg_idx] = -data

        # (3) Resample
        if target_sfreq and not np.isclose(raw.info["sfreq"], float(target_sfreq)):
            raw.resample(float(target_sfreq), npad="auto")

        # (4) Mark stim + ecg by name before montage
        if stim_keep:
            for ch in stim_keep:
                if ch in raw.ch_names:
                    raw.set_channel_types({ch: "stim"})
        for ch in raw.ch_names:
            if 'ecg' in ch.lower() or 'ekg' in ch.lower():
                try:
                    raw.set_channel_types({ch: "ecg"})
                except Exception:
                    pass

        # (5) Standardise and apply montage (drops EEG channels not in montage)
        raw = standardise_and_montage(raw)

        if sum(1 for t in raw.get_channel_types() if t == "eeg") == 0:
            raise ValueError("No EEG channels remain after montage/drop; check montage name and channel labels.")

        # (6) PyPREP
        if pyprep_dict is None:
            pyprep_dict = prep_params
        do_pyprep_flag = True
        try:
            do_pyprep_flag = bool(do_pyprep)
        except Exception:
            if _CFG_RT is not None:
                do_pyprep_flag = bool(_CFG_RT.use_pyprep)
        if do_pyprep_flag:
            raw = run_pyprep(raw, pyprep_dict, random_seed=random_seed, ransac=bool(_CFG_RT.prep_ransac))

        # (7) ASR + ICA + ICLabel once
        if asr_threshold is None:
            asr_threshold = do_asr

        raw_eeg_proc, ica, labels, bads_remove, bads_keep, bad_map_remove, bad_map_keep, eog_inds = _fit_asr_ica_iclabel_once(
            raw,
            asr_thresh=asr_threshold,
            random_seed=420 + int(random_seed),
            n_comp=n_comp,
        )

        non_eeg = raw.copy().pick([ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t != 'eeg'])

        def _apply_and_recombine(exclude, variant_label_dict):
            ica_local = ica.copy()
            ica_local.exclude = list(exclude)
            eeg_clean = ica_local.apply(raw_eeg_proc.copy())
            try:
                eeg_clean.interpolate_bads(reset_bads=True)
            except Exception:
                pass
            if len(non_eeg.ch_names):
                eeg_clean.add_channels([non_eeg], force_update_info=True)
            # R-peak events
            eeg_clean, events = detect_r_peaks(eeg_clean)
            eeg_clean.info["bads"] = []
            return eeg_clean, events, variant_label_dict, ica_local

        raw_remove, events_remove, bad_labels_remove, ica_remove = _apply_and_recombine(bads_remove, bad_map_remove)
        raw_keep, events_keep, bad_labels_keep, ica_keep = _apply_and_recombine(bads_keep, bad_map_keep)

        # (8) Save both raws + ICA models
        for outp_str, raw_out, ica_out, bad_labels, events, mode in [
            (output_path_remove, raw_remove, ica_remove, bad_labels_remove, events_remove, "remove"),
            (output_path_keep, raw_keep, ica_keep, bad_labels_keep, events_keep, "keep"),
        ]:
            outp = Path(outp_str)
            outp.parent.mkdir(parents=True, exist_ok=True)
            raw_out.set_meas_date(1)
            raw_out.save(str(outp), overwrite=True)

            ica_path = ""
            try:
                stem_no_pp = outp.stem.replace("_pp_raw", "")
                ica_path = str(outp.with_name(stem_no_pp + "_ica.fif"))
                Path(ica_path).parent.mkdir(parents=True, exist_ok=True)
                ica_out.save(ica_path, overwrite=True)
            except Exception:
                ica_path = ""

            _append_preproc_tsv(logging_path, {
                "utc": _now_utc_iso(),
                "status": "OK",
                "stage": "save",
                "input": input_path,
                "output": str(outp),
                "n_channels": len(raw_out.ch_names),
                "n_events": len(events),
                "bad_channels": ';'.join(raw_out.info.get('bads', [])) if raw_out.info.get('bads') else '',
                "remove_cfa": str(mode == "remove"),
                "remove_cfa_mode": "both",
                "flip_ecg": str(bool(flip_ecg)),
                "asr_threshold": str(asr_threshold),
                "n_comp": str(n_comp),
                "stim_keep": ";".join(stim_keep) if stim_keep else "",
                "montage_name": str(getattr(_CFG_RT, "montage_name", "auto")),
                "ica_path": ica_path,
                "ica_bads": ';'.join(str(k) for k in sorted(bad_labels.keys())),
                "ica_bad_labels": ';'.join(str(bad_labels[k]) for k in sorted(bad_labels.keys())),
            })

        return raw_remove, raw_keep

    except Exception as e:
        logging.exception("preprocess_edf_both failed for %s", input_path)
        _append_preproc_tsv(logging_path, {
            "utc": _now_utc_iso(),
            "status": "FAIL",
            "stage": "preprocess_edf_both",
            "input": input_path,
            "output": f"{output_path_remove} | {output_path_keep}",
            "error_type": type(e).__name__,
            "error": str(e),
        })
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
