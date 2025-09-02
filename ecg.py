# heppy/ecg.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import mne
import neurokit2 as nk
from matplotlib import pyplot as plt
from config import HEPConfig

# --- Pulled & adapted from your CFA pipeline (export QRS plot, RPEAK stim, present-RR) :contentReference[oaicite:6]{index=6}

def save_ecg_qc_plot(ecg_signal: np.ndarray, sf: float, save_path: Path, 
                     r_peaks: Optional[np.ndarray] = None, 
                     title_suffix: str = "") -> None:
    """Save NeuroKit2 ECG QC plot (R-peaks etc.).
    
    Parameters
    ----------
    ecg_signal : np.ndarray
        Clean ECG signal
    sf : float
        Sampling frequency
    save_path : Path
        Output path for the plot
    r_peaks : np.ndarray, optional
        Custom R-peak locations (in samples). If None, auto-detects.
    title_suffix : str
        Additional text to add to plot title
    """
    try:
        if r_peaks is not None:
            # For custom R-peaks, create a simple matplotlib plot
            t = np.arange(len(ecg_signal)) / sf
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(t, ecg_signal, 'b-', linewidth=0.8, label='ECG')
            
            # Mark R-peaks
            valid_peaks = r_peaks[(r_peaks >= 0) & (r_peaks < len(ecg_signal))]
            if len(valid_peaks) > 0:
                ax.plot(valid_peaks / sf, ecg_signal[valid_peaks], 'ro', markersize=4, label='R-peaks')
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'ECG with R-peaks {title_suffix}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            # Auto-detect R-peaks using NeuroKit2
            signals_df, info = nk.ecg_process(ecg_signal, sampling_rate=int(sf))
            nk.ecg_plot(signals_df, info)
            fig = plt.gcf()
            fig.set_size_inches(10, 12, forward=True)
            
            # Add title suffix if provided
            if title_suffix:
                fig.suptitle(f"ECG Analysis {title_suffix}", fontsize=14)
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path), dpi=150)
            plt.close(fig)
    except Exception as err:
        print(f"[heppy] ECG QC plot failed: {err}")

def _first_ecg_name(raw: mne.io.BaseRaw, preferred: Optional[str]) -> Optional[str]:
    if preferred and preferred in raw.ch_names:
        return preferred
    for ch, tp in zip(raw.ch_names, raw.get_channel_types()):
        if tp == "ecg" or "ecg" in ch.lower() or "ekg" in ch.lower():
            return ch
    return None

def detect_rpeaks(raw: mne.io.BaseRaw, ecg_name: Optional[str]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (r_peaks_samples, rr_sec, ecg_clean)."""
    name = _first_ecg_name(raw, ecg_name)
    if not name:
        return np.array([]), np.array([]), np.array([])
    sig = raw.get_data(picks=name)[0]
    sf  = float(raw.info["sfreq"])
    sig_clean = nk.ecg_clean(sig, sampling_rate=sf)
    _, info = nk.ecg_peaks(sig_clean, sampling_rate=sf, method="promac")
    r = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    rr = np.diff(r) / sf if r.size >= 2 else np.array([], dtype=float)
    return r, rr, sig_clean

def refine_with_rr_hint(r_peaks: np.ndarray, sf: float, rr_hint_s: float, factor: float = 1.8) -> np.ndarray:
    """
    If large gaps (> factor * rr_hint_s) exist, insert evenly spaced synthetic peaks.
    Helps when NK merges beats (underestimates BPM).
    """
    if r_peaks.size < 2 or not np.isfinite(rr_hint_s) or rr_hint_s <= 0:
        return r_peaks
    r = np.unique(r_peaks.astype(int))
    rr_s = np.diff(r) / sf
    out = list(r)
    max_gap = factor * rr_hint_s
    for i, gap in enumerate(rr_s):
        if gap > max_gap:
            n_missing = int(np.round(gap / rr_hint_s)) - 1
            for j in range(1, n_missing + 1):
                out.append(r[i] + int(j * rr_hint_s * sf))
    return np.unique(out).astype(int)

def add_rpeak_stim(raw: mne.io.BaseRaw, r_peaks: np.ndarray, name: str = "RPEAK"):
    stim = np.zeros((1, raw.n_times), dtype=np.int16)
    r_idx = np.asarray(r_peaks, dtype=int)
    r_idx = r_idx[(0 <= r_idx) & (r_idx < raw.n_times)]
    stim[0, r_idx] = 1
    info = mne.create_info([name], raw.info["sfreq"], ch_types="stim")
    raw.add_channels([mne.io.RawArray(stim, info, verbose=False)], force_update_info=True)

def present_rr_metadata(event_samples: np.ndarray,
                        r_peaks_samples: np.ndarray,
                        sf: float,
                        tmin: float, tmax: float):
    """
    Attach present RR and whether next R is within the epoch window.
    Returns dict of arrays ready to become a pandas DataFrame.
    """
    r_sorted = np.sort(np.asarray(r_peaks_samples, int))
    pres_rr, has_nextR = [], []
    win_len = int(round((tmax - tmin) * sf))
    for s0 in event_samples:
        i = np.searchsorted(r_sorted, s0, side="right")
        if i < r_sorted.size:
            s_next = r_sorted[i]
            rr_s = (s_next - s0) / sf
            pres_rr.append(rr_s)
            has_nextR.append(bool(s_next - s0 <= win_len))
        else:
            pres_rr.append(np.nan)
            has_nextR.append(False)
    return {"present_rr_s": pres_rr, "has_next_r_in_window": has_nextR}
