# -*- coding: utf-8 -*-
"""
Simple, configurable HEP extraction (library-style; optional multiprocessing)
=============================================================================

This module provides:
  • `extract_hep_for_file(...)` – process a single EDF/FIF and save HEP outputs.
  • `run_directory(...)` – batch-process a directory (optionally multiprocessing).
No CLI entry-point is exposed; import and call these functions from your scripts.

Outputs per input file:
  • <save_stem>_<base>_epo.fif        – all-beats epochs
  • <save_stem>_<base>_ave.fif        – all-beats evoked
  • <save_stem>_<base>_minrr_epo.fif  – minRR-filtered epochs
  • <save_stem>_<base>_minrr_ave.fif  – minRR-filtered evoked
  • <save_stem>_<base>_proc.json      – provenance (parameters & preprocessing info)

Method references
-----------------
MNE-Python: Gramfort A, Luessi M, Larson E, et al. NeuroImage (2014).
PREP: Bigdely‑Shamlo N, et al. Front Neuroinform (2015). (via PyPREP)
PyPREP: Appelhoff S, et al. Zenodo, 2025
ASR: Mullen T, et al. IEEE TBME (2015). (asrpy implementation)
ICA (extended-Infomax): Bell & Sejnowski (1995); Lee, Girolami, Sejnowski (1999).
ICLabel: Pion‑Tonachini L, Kreutz‑Delgado K, Makeig S. NeuroImage (2019).
NeuroKit2 (ECG): Makowski D, et al. Behav Res Methods (2021).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union
import mne
import neurokit2 as nk
import numpy as np


# ------------------------------- configuration -------------------------------

@dataclass
class HEPConfig:
    """Parameters controlling HEP extraction.

    Attributes
    ----------
    montage_name : str | Path | None
        Standard montage name (e.g., "standard_1020", "biosemi128"), a custom
        montage filepath, or None to skip setting montage.
    rename_to_1020 : bool
        Best-effort channel name normalisation toward 10-20 labels.
    tmin, tmax : float
        Epoch window in seconds relative to R-peak.
    baseline : tuple[float, float] | None
        Baseline correction window in seconds (or None to disable).
    amp_rej_uv : float
        Peak-to-peak amplitude threshold in microvolts for epoch rejection.
    amp_window_s : tuple[float, float]
        Time window (s) used for peak-to-peak rejection.
    min_rr_s : float
        Present RR threshold (keeps beats whose FOLLOWING RR ≥ min_rr_s).
    ecg_channel : str | None
        ECG channel name if known. If None, first ECG channel is used.
    use_pyprep : bool
        Run PyPREP robust referencing / bad-channel handling.
    use_asr : float | None
        ASR cutoff in standard deviations (None or <=0 to disable).
    use_ica : bool
        Run ICA + ICLabel and remove components not labelled 'brain'/'other'.
    target_sfreq : float | None
        Resample to this sampling rate; None to keep native.
    random_seed : int
        Random seed for reproducible ICA/PyPREP behaviour.
    save_stem : str
        Stem used for output filenames.
    """
    montage_name: Optional[Union[str, Path]] = "standard_1020"
    rename_to_1020: bool = True
    tmin: float = -0.2
    tmax: float = 0.8
    baseline: Optional[Tuple[float, float]] = (-0.2, -0.05)
    amp_rej_uv: float = 150.0
    amp_window_s: Tuple[float, float] = (0.1, 0.5)
    min_rr_s: float = 0.7
    ecg_channel: Optional[str] = None
    use_pyprep: bool = True
    use_asr: Optional[float] = 20.0
    use_ica: bool = True
    target_sfreq: Optional[float] = 256.0
    random_seed: int = 42
    save_stem: str = "hep"


# ------------------------------- utilities -------------------------------

def _read_raw_any(path: Union[str, Path]) -> mne.io.BaseRaw:
    """Read EDF or FIF by extension.

    Notes
    -----
    Uses MNE readers:
      • .edf  → mne.io.read_raw_edf
      • .fif  → mne.io.read_raw_fif
    """
    path = str(path)
    if path.lower().endswith(".edf"):
        return mne.io.read_raw_edf(path, preload=True, verbose=False)
    if path.lower().endswith(".fif"):
        return mne.io.read_raw_fif(path, preload=True, verbose=False)
    # Fall back to MNE's generic if available; otherwise raise.
    try:
        return mne.io.read_raw(path, preload=True, verbose=False)  # type: ignore[attr-defined]
    except Exception as e:
        raise ValueError(f"Unsupported file type for {path}") from e


def _maybe_set_montage(raw: mne.io.BaseRaw,
                       montage_name: str | Path | None,
                       rename_to_1020: bool = True) -> mne.io.BaseRaw:
    """Optionally normalise names and set montage (standard or custom)."""
    if rename_to_1020:
        mapping = {}
        for ch in raw.ch_names:
            nm = ch.replace('EEG ', '').split('-')[0]
            nm = nm.upper().replace('Z', 'z')
            mapping[ch] = nm
        raw.rename_channels(mapping)
        for ch in raw.ch_names:
            if 'ECG' in ch.upper() or 'EKG' in ch.upper():
                raw.set_channel_types({ch: 'ecg'})

    if not montage_name:
        return raw

    if isinstance(montage_name, (str, Path)) and Path(str(montage_name)).exists():
        mont = mne.channels.read_custom_montage(str(montage_name))
    else:
        mont = mne.channels.make_standard_montage(str(montage_name))

    eeg_names = [ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t == 'eeg']
    keep_for_mont = [ch for ch in eeg_names if ch in mont.ch_names]
    if keep_for_mont:
        raw.pick(keep_for_mont + [ch for ch, t in zip(raw.ch_names, raw.get_channel_types())
                                  if t in ('ecg', 'stim') and ch not in keep_for_mont])
    raw.set_montage(mont, match_case=False)
    return raw


def _detect_rpeaks(raw: mne.io.BaseRaw, ecg_channel: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect R-peaks using NeuroKit2; returns (rpeaks, present_rr_sec, ecg_clean)."""
    ecg_picks = [c for c, t in zip(raw.ch_names, raw.get_channel_types()) if t == 'ecg']
    if ecg_channel is None:
        if not ecg_picks:
            return np.array([]), np.array([]), np.array([])
        ecg_channel = ecg_picks[0]

    sig = raw.get_data(picks=ecg_channel)[0]
    sf  = raw.info['sfreq']
    ecg_clean = nk.ecg_clean(sig, sampling_rate=sf)
    _, info    = nk.ecg_peaks(ecg_clean, sampling_rate=sf)
    rpeaks     = info.get("ECG_R_Peaks", np.array([], dtype=int))
    if rpeaks is None or len(rpeaks) < 2:
        return np.array([]), np.array([]), ecg_clean
    rr_sec = np.diff(rpeaks) / sf
    return rpeaks, rr_sec, ecg_clean


def _amp_reject_indices(epochs: mne.Epochs, amp_window_s: tuple[float, float], thresh_uv: float) -> list[int]:
    """Return indices of epochs exceeding peak-to-peak amplitude within a window (EEG channels only)."""
    sf = epochs.info['sfreq']
    picks = mne.pick_types(epochs.info, eeg=True)
    w0 = int((amp_window_s[0] - epochs.tmin) * sf)
    w1 = int((amp_window_s[1] - epochs.tmin) * sf)
    data = epochs.get_data()[:, picks, w0:w1]
    ptp  = data.max(axis=2) - data.min(axis=2)
    bad  = np.where(ptp > thresh_uv)[0]
    return bad.tolist()


# ------------------------------- preprocessing -------------------------------

def preprocess_raw(raw: mne.io.BaseRaw, *,
                   use_pyprep=True, use_asr=20, use_ica=True,
                   target_sfreq=256, random_seed=42) -> tuple[mne.io.BaseRaw, dict]:
    """Light preprocessing: optional PREP → optional ASR → optional ICA → optional resample.

    Returns a tuple of (cleaned_raw, provenance_dict).
    """
    prov = {
        "sfreq_in": float(raw.info['sfreq']),
        "pyprep": bool(use_pyprep),
        "asr_cutoff": float(use_asr) if use_asr else None,
        "ica": bool(use_ica),
        "target_sfreq": float(target_sfreq) if target_sfreq else None,
        "bad_channels_in": list(raw.info.get('bads', [])),
    }

    if use_pyprep:
        from pyprep import PrepPipeline
        prep = PrepPipeline(raw=raw, montage=raw.get_montage(), prep_params={
            "ref_chs": "eeg", "reref_chs": "eeg",
            "l_freq": 1.0, "h_freq": 100.0,  # 100 Hz helps ICLabel features
        }, random_state=random_seed, ransac=False, channel_wise=True)
        prep.fit()
        raw = prep.raw_eeg
        if getattr(prep, 'raw_non_eeg', None) is not None:
            raw.add_channels([prep.raw_non_eeg], force_update_info=True)
        prov.update({
            "pyprep_l_freq": 1.0,
            "pyprep_h_freq": 100.0,
            "still_noisy": list(getattr(prep, "still_noisy_channels", [])),
        })

    raw, _ = mne.set_eeg_reference(raw, ref_channels='average')
    prov["referenced"] = "average"

    if use_asr:
        import asrpy
        asr = asrpy.ASR(sfreq=raw.info['sfreq'], cutoff=use_asr)
        _tmp = raw.copy().pick('eeg').resample(100)
        asr.fit(_tmp)
        raw._data[mne.pick_types(raw.info, eeg=True)] = asr.transform(raw.copy().pick('eeg')).get_data()
        prov["asr_fitted_on_downsampled"] = True

    if use_ica:
        from mne.preprocessing import ICA
        from mne_icalabel import label_components
        eeg = raw.copy().pick('eeg')
        n_comp = min(len(eeg.ch_names) - len(eeg.info['bads']) - 1, 48)
        ica = ICA(n_components=n_comp, method='infomax', random_state=random_seed, fit_params={"extended": True})
        ica.fit(eeg, decim=3)
        labels = label_components(eeg, ica, method='iclabel')['labels']
        bads = [i for i, lab in enumerate(labels) if lab not in ('brain', 'other')]
        ica.exclude = bads
        eeg_clean = ica.apply(eeg)
        raw._data[mne.pick_types(raw.info, eeg=True)] = eeg_clean.get_data()
        prov.update({
            "ica_method": "infomax_extended",
            "ica_n_components": int(n_comp),
            "ica_labels_counts": {lab: int(sum(l == lab for l in labels)) for lab in set(labels)},
            "ica_excluded_idx": list(map(int, bads)),
        })

    raw.interpolate_bads(reset_bads=False)
    if target_sfreq and raw.info['sfreq'] != target_sfreq:
        raw.resample(target_sfreq, npad='auto')
    prov.update({
        "sfreq_out": float(raw.info['sfreq']),
        "bad_channels_out": list(raw.info.get('bads', [])),
    })
    return raw, prov


# ------------------------------- core routine -------------------------------

def extract_hep_for_file(
    infile: str | Path,
    outdir: str | Path,
    *,
    montage_name: str | Path | None,
    rename_to_1020: bool,
    save_stem: str,
    tmin: float,
    tmax: float,
    baseline: tuple[float, float] | None,
    amp_rej_uv: float,
    amp_window_s: tuple[float, float],
    min_rr_s: float,
    ecg_channel: str | None,
    use_pyprep: bool,
    use_asr: float | None,
    use_ica: bool,
    target_sfreq: float | None,
    random_seed: int = 42,
    verbose: bool = True,
) -> dict:
    """End-to-end HEP extraction for a single file. See module docstring for outputs."""
    infile = Path(infile)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    base = infile.stem

    # Load & montage
    raw = _read_raw_any(infile)
    montage_str = str(montage_name) if montage_name is not None else None
    _maybe_set_montage(raw, montage_name, rename_to_1020)

    # Preprocess
    raw, prov = preprocess_raw(
        raw,
        use_pyprep=use_pyprep, use_asr=use_asr, use_ica=use_ica,
        target_sfreq=target_sfreq, random_seed=random_seed
    )
    prov["montage"] = montage_str
    prov["rename_to_1020"] = bool(rename_to_1020)

    # R-peaks
    rpeaks, rr_sec, _ = _detect_rpeaks(raw, ecg_channel=ecg_channel)
    prov["n_rpeaks"] = int(len(rpeaks))
    if len(rpeaks) < 2:
        if verbose:
            print(f"[{base}] No (or too few) R-peaks; skipping.")
        with open(outdir / f"{save_stem}_{base}_proc.json", "w") as f:
            json.dump({"file": base, **prov, "n_all": 0, "n_minrr": 0}, f, indent=2)
        return {"file": base, "n_all": 0, "n_minrr": 0}

    # All-beats epochs
    events_all = np.c_[rpeaks, np.zeros_like(rpeaks), np.ones_like(rpeaks)]
    epochs = mne.Epochs(raw, events_all, tmin=tmin, tmax=tmax,
                        baseline=baseline, preload=True, verbose=False)
    bad_idx = _amp_reject_indices(epochs, amp_window_s, amp_rej_uv)
    if bad_idx:
        epochs.drop(bad_idx, reason=f"EEG>{amp_rej_uv}uV")
    n_all = len(epochs)
    if n_all:
        epochs.set_meas_date(1).save(outdir / f"{save_stem}_{base}_epo.fif", overwrite=True)
        epochs.average().set_meas_date(1).save(outdir / f"{save_stem}_{base}_ave.fif", overwrite=True)

    # MinRR epochs (present RR)
    keep = rr_sec >= min_rr_s
    if keep.any():
        valid_peaks = np.asarray(rpeaks)[:-1][keep]
        ev_minrr = np.c_[valid_peaks, np.zeros_like(valid_peaks), np.ones_like(valid_peaks)]
        hep = mne.Epochs(raw, ev_minrr, tmin=tmin, tmax=tmax,
                         baseline=baseline, preload=True, verbose=False)
        bad_idx = _amp_reject_indices(hep, amp_window_s, amp_rej_uv)
        if bad_idx:
            hep.drop(bad_idx, reason=f"EEG>{amp_rej_uv}uV")
        n_minrr = len(hep)
        if n_minrr:
            hep.set_meas_date(1).save(outdir / f"{save_stem}_{base}_minrr_epo.fif", overwrite=True)
            hep.average().set_meas_date(1).save(outdir / f"{save_stem}_{base}_minrr_ave.fif", overwrite=True)
    else:
        n_minrr = 0

    # Provenance JSON
    summary = {"file": base, "n_all": int(n_all), "n_minrr": int(n_minrr), **prov}
    with open(outdir / f"{save_stem}_{base}_proc.json", "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"[{base}] all={n_all}, minRR={n_minrr}")
    return summary


# ------------------------------- batch runner -------------------------------

def _process_one(path: str,
                 outdir: str,
                 cfg: HEPConfig,
                 verbose: bool = True) -> dict:
    """Thin wrapper around `extract_hep_for_file` for use in pools."""
    try:
        return extract_hep_for_file(
            path, outdir,
            montage_name=cfg.montage_name,
            rename_to_1020=cfg.rename_to_1020,
            save_stem=cfg.save_stem,
            tmin=cfg.tmin, tmax=cfg.tmax, baseline=cfg.baseline,
            amp_rej_uv=cfg.amp_rej_uv, amp_window_s=cfg.amp_window_s,
            min_rr_s=cfg.min_rr_s, ecg_channel=cfg.ecg_channel,
            use_pyprep=cfg.use_pyprep, use_asr=cfg.use_asr, use_ica=cfg.use_ica,
            target_sfreq=cfg.target_sfreq, random_seed=cfg.random_seed,
            verbose=verbose,
        )
    except Exception as e:
        if verbose:
            print(f"[{Path(path).stem}] ERROR: {e}")
        return {"file": Path(path).stem, "error": str(e), "n_all": 0, "n_minrr": 0}


def run_directory(input_dir: str | Path,
                  output_dir: str | Path,
                  cfg: HEPConfig,
                  pattern: Iterable[str] = ("*.edf", "*.fif"),
                  n_jobs: int = 1,
                  write_summary_csv: bool = True,
                  verbose: bool = True) -> List[Dict]:
    """Process all EEG files in a directory, optionally with multiprocessing.

    Parameters
    ----------
    input_dir : str | Path
        Root directory containing EEG files; search is recursive.
    output_dir : str | Path
        Destination directory for FIF+JSON outputs.
    cfg : HEPConfig
        Processing parameters.
    pattern : Iterable[str]
        Filename patterns to include.
    n_jobs : int
        Number of worker processes; 1 ⇒ serial.
    write_summary_csv : bool
        Whether to write a <save_stem>_summary.csv in the output directory.
    verbose : bool
        Print progress information.

    Returns
    -------
    list of dict
        One provenance/summary row per processed file.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files: List[str] = []
    for pat in pattern:
        files.extend([str(p) for p in input_dir.rglob(pat)])
    files = sorted(set(files))

    if verbose:
        print(f"Found {len(files)} files under {input_dir} matching {list(pattern)}")

    if n_jobs == 1:
        rows = [_process_one(f, str(output_dir), cfg, verbose=verbose) for f in files]
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        rows = []
        with ProcessPoolExecutor(max_workers=int(n_jobs)) as ex:
            futs = {ex.submit(_process_one, f, str(output_dir), cfg, verbose): f for f in files}
            for fut in as_completed(futs):
                rows.append(fut.result())

    if write_summary_csv:
        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            df.to_csv(output_dir / f"{cfg.save_stem}_summary.csv", index=False)
        except Exception as e:
            if verbose:
                print(f"[WARN] Could not write summary CSV: {e}")

    return rows
