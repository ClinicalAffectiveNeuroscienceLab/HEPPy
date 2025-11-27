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
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import asrpy

import mne
import numpy as np
import neurokit2 as nk
import pyprep
from mne.preprocessing import ICA
from mne_icalabel import label_components

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
    remove_cfa: bool
    log_file: str | None
    ref_chs: str | list
    reref_chs: str | list
    high_pass: float
    low_pass: float
    prep_ransac: bool
    line_freqs: tuple | list

_CFG_RT: _RuntimeCfg | None = None

def set_runtime_config(cfg) -> None:
    """Call this once from the CLI or notebook to bind the active config."""
    global _CFG_RT, output_dir, target_sfreq, do_asr, remove_cfa, log_file, prep_params
    _CFG_RT = _RuntimeCfg(
        output_root=Path(cfg.output_root),
        target_sfreq=getattr(cfg, "target_sfreq", None),
        use_asr=getattr(cfg, "use_asr", None),
        remove_cfa=getattr(cfg, "remove_cfa", False),
        log_file=getattr(cfg, "log_file", None),
        ref_chs=getattr(cfg, "ref_chs", "eeg"),
        reref_chs=getattr(cfg, "reref_chs", "eeg"),
        high_pass=getattr(cfg, "high_pass", 1.0),
        low_pass=getattr(cfg, "low_pass", 100.0),
        prep_ransac=getattr(cfg, "prep_ransac", True),
        line_freqs=getattr(cfg, "line_freqs", (50.0, 100.0))
    )

    # keep legacy names used in the rest of this module
    output_dir   = str(_CFG_RT.output_root)
    target_sfreq = float(_CFG_RT.target_sfreq) if _CFG_RT.target_sfreq else None
    do_asr       = _CFG_RT.use_asr
    remove_cfa   = _CFG_RT.remove_cfa
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
        else:
            if raw.get_channel_types(picks=[ch])[0] == 'eeg':
                if _canonicalise_name(ch) not in TEN_TWENTY_SET:
                    raw.set_channel_types({ch: 'misc'})

def standardise_and_montage(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    raw.rename_channels({ch: _canonicalise_name(ch) for ch in raw.ch_names})
    _classify_aux_channels(raw)
    mont = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(mont, match_case=False, on_missing='ignore')
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
        pass
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
) -> mne.io.BaseRaw:
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

    labels = label_components(raw_eeg, ica, method="iclabel")['labels']
    if remove_cfa_flag:
        bads = [i for i, lab in enumerate(labels) if lab not in ('brain', 'other')]
    else:
        bads = [i for i, lab in enumerate(labels) if lab not in ('brain', 'heart', 'other')]

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

    return eeg_clean

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
):
    if pyprep_dict is None:
        pyprep_dict = prep_params
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

    # 3) Global resample once (for identical n_times across all channels)
    if target_sfreq and not np.isclose(raw.info["sfreq"], float(target_sfreq)):
        raw.resample(float(target_sfreq), npad="auto")

    # 4) Rename / classify / montage
    raw = standardise_and_montage(raw)

    # 5) PyPREP on EEG-only; recombine with non-EEG
    raw = run_pyprep(raw, pyprep_dict, random_seed=random_seed, ransac=True)

    # 6) ASR + ICA (+/- remove cardiac via ICLabel)
    raw = run_asr_ica(raw, asr_thresh=asr_threshold, random_seed=random_seed,
                      n_comp=n_comp, remove_cfa_flag=bool(remove_cfa))

    # 7) ECG → R-peaks → stim events
    raw, events = detect_r_peaks(raw)

    raw.info["bads"] = []  # clear any stragglers

    # 8) Save (guard empty parents)
    outp = Path(output_path)
    if str(outp.parent) not in ("", "."):
        outp.parent.mkdir(parents=True, exist_ok=True)
    raw.set_meas_date(1)
    raw.save(str(outp), overwrite=True)

    # 9) Log (guard empty parents)
    rec = {
        'input': input_path,
        'output': str(outp),
        'n_channels': len(raw.ch_names),
        'n_events': len(events),
        'bad_channels': ';'.join(raw.info.get('bads', [])) if raw.info.get('bads') else '',
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
