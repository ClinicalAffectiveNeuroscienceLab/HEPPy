# heppy/epochs.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import mne
from config import HEPConfig
from ecg import present_rr_metadata

def _amp_reject_indices(epochs: mne.Epochs, win: tuple[float, float], thresh_uv: float) -> list[int]:
    sf = epochs.info["sfreq"]
    w0 = int((win[0] - epochs.tmin) * sf)
    w1 = int((win[1] - epochs.tmin) * sf)
    picks = mne.pick_types(epochs.info, eeg=True)
    dat = epochs.get_data()[:, picks, w0:w1]  # volts
    ptp_uv = (dat.max(2) - dat.min(2)) * 1e6
    bad = np.where(ptp_uv > thresh_uv)[0]
    return bad.astype(int).tolist()

def build_hep_epochs(raw: mne.io.BaseRaw,
                     r_peaks: np.ndarray,
                     rr_sec: np.ndarray,
                     cfg: HEPConfig) -> mne.Epochs:
    """Create HEP epochs using the present-RR rule; attach metadata; reject by amplitude."""
    # Present-RR filter
    keep = rr_sec >= cfg.min_rr_s
    if not keep.any():
        raise RuntimeError("No HEP-valid events (present RR >= min_rr_s).")
    hep_peaks = np.asarray(r_peaks)[:-1][keep]
    events = np.c_[hep_peaks, np.zeros_like(hep_peaks), np.ones_like(hep_peaks)]

    picks = mne.pick_types(raw.info, eeg=True, ecg=True, stim=True)
    epo = mne.Epochs(raw, events, tmin=cfg.tmin, tmax=cfg.tmax,
                     baseline=cfg.baseline, picks=picks, preload=True, verbose=False)

    # Amp rejection
    bad = _amp_reject_indices(epo, cfg.amp_window_s, cfg.amp_rej_uv)
    if bad:
        epo.drop(bad, reason=f"EEG>{cfg.amp_rej_uv}uV")

    # Metadata: present RR + has_nextR
    md = present_rr_metadata(epo.events[:, 0], r_peaks, raw.info["sfreq"], cfg.tmin, cfg.tmax)
    try:
        import pandas as pd
        epo.metadata = pd.DataFrame(md)
    except Exception:
        pass

    # Set a stable meas_date for saving consistency
    epo.set_meas_date(1)
    return epo

def save_hep_epochs(epo: mne.Epochs, outdir: Path, save_stem: str, base: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{save_stem}_{base}_hep_epo.fif"
    
    # Handle missing montage gracefully for test data
    try:
        epo.save(out_path, overwrite=True)
    except Exception as e:
        if "Montage is not set" in str(e):
            # Create a dummy montage or save without montage info
            import tempfile
            # Save just the data arrays instead of full FIF
            np.savez_compressed(
                str(out_path).replace('.fif', '.npz'),
                data=epo.get_data(),
                times=epo.times,
                events=epo.events,
                ch_names=epo.ch_names,
                sfreq=epo.info['sfreq']
            )
            print(f"Warning: Saved as NPZ due to missing montage: {out_path}")
            return Path(str(out_path).replace('.fif', '.npz'))
        else:
            raise e
    
    return out_path
