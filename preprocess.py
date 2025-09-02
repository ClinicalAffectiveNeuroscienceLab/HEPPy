# heppy/preprocess.py
from __future__ import annotations
from typing import Tuple, Optional
import mne
from config import HEPConfig

# Reuse your existing robust utilities (keeps behaviour consistent)
# (from preprocessing.py in your codebase)
from preprocessing import standardise_and_montage, run_pyprep, run_asr_ica

def apply_montage(raw: mne.io.BaseRaw, cfg: HEPConfig) -> mne.io.BaseRaw:
    raw = standardise_and_montage(raw, cfg.montage_name, cfg.rename_to_1020)
    return raw

def clean_raw(raw: mne.io.BaseRaw, cfg: HEPConfig) -> Tuple[mne.io.BaseRaw, dict]:
    prov = {
        "sfreq_in": float(raw.info["sfreq"]),
        "use_pyprep": bool(cfg.use_pyprep),
        "use_asr": float(cfg.use_asr) if cfg.use_asr else None,
        "use_ica": bool(cfg.use_ica),
        "target_sfreq": float(cfg.target_sfreq) if cfg.target_sfreq else None,
    }

    eeg = raw.copy()
    if cfg.use_pyprep:
        eeg = run_pyprep(eeg, random_seed=cfg.random_seed)
    
    if cfg.use_asr or cfg.use_ica:
        eeg = run_asr_ica(eeg, asr_thresh=cfg.use_asr, use_ica=cfg.use_ica, random_seed=cfg.random_seed)
    else:
        # Just apply average reference if no ASR/ICA
        eeg, _ = mne.set_eeg_reference(eeg, ref_channels='average')

    # resample if requested
    if cfg.target_sfreq and eeg.info["sfreq"] != cfg.target_sfreq:
        eeg.resample(cfg.target_sfreq, npad="auto")

    prov["sfreq_out"] = float(eeg.info["sfreq"])
    return eeg, prov
