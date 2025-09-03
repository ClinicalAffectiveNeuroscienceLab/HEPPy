# Basic preprocessing functions for HEPPy
from __future__ import annotations
import mne
from pathlib import Path
from typing import Optional, Union

def standardise_and_montage(raw: mne.io.BaseRaw, 
                           montage_name: Optional[Union[str, Path]] = "standard_1020",
                           rename_to_1020: bool = True) -> mne.io.BaseRaw:
    """Standardize channel names and apply montage."""
    if rename_to_1020:
        # Basic channel name normalization
        mapping = {}
        for ch in raw.ch_names:
            nm = ch.replace('EEG ', '').split('-')[0]
            nm = nm.upper().replace('Z', 'z')
            mapping[ch] = nm
        raw.rename_channels(mapping)
        
        # Set ECG channel types
        for ch in raw.ch_names:
            if 'ECG' in ch.upper() or 'EKG' in ch.upper():
                raw.set_channel_types({ch: 'ecg'})
    
    if montage_name:
        try:
            if isinstance(montage_name, (str, Path)) and Path(str(montage_name)).exists():
                mont = mne.channels.read_custom_montage(str(montage_name))
            else:
                mont = mne.channels.make_standard_montage(str(montage_name))
            
            # Filter to keep only channels in montage
            eeg_names = [ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t == 'eeg']
            keep_for_mont = [ch for ch in eeg_names if ch in mont.ch_names]
            if keep_for_mont:
                other_chs = [ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) 
                            if t in ('ecg', 'stim') and ch not in keep_for_mont]
                raw.pick(keep_for_mont + other_chs)
                raw.set_montage(mont, match_case=False)
            else:
                # No matching channels, skip montage but keep all channels
                print(f"Warning: No channels match montage {montage_name}, proceeding without montage")
        except Exception as e:
            print(f"Warning: Failed to set montage {montage_name}: {e}")
            # Continue without montage
    
    return raw

def run_pyprep(raw: mne.io.BaseRaw, prep_params: dict, random_seed: int = 42) -> mne.io.BaseRaw:
    """Run PyPREP pipeline."""
    from pyprep import PrepPipeline
    
    prep = PrepPipeline(
        raw=raw, 
        montage=raw.get_montage(),
        prep_params=prep_params,
        random_state=random_seed, 
        ransac=False, 
        channel_wise=True
    )
    prep.fit()
    
    # Combine EEG and non-EEG channels
    cleaned_raw = prep.raw_eeg
    if hasattr(prep, 'raw_non_eeg') and prep.raw_non_eeg is not None:
        cleaned_raw.add_channels([prep.raw_non_eeg], force_update_info=True)
    
    return cleaned_raw

def run_asr_ica(raw: mne.io.BaseRaw, 
                asr_thresh: Optional[float] = 20.0, 
                use_ica: bool = True,
                random_seed: int = 42) -> mne.io.BaseRaw:
    """Run ASR and ICA cleaning."""
    # Average reference
    raw, _ = mne.set_eeg_reference(raw, ref_channels='average')
    
    # ASR
    if asr_thresh:
        if type(asr_thresh) == float:
            pass
        elif type(asr_thresh) == int:
            asr_thresh = float(asr_thresh)
        elif type(asr_thresh) == bool and asr_thresh:
            asr_thresh = 20.0
        else:
            raise ValueError("asr_thresh must be numerical or True/False")
        print("Running ASR with threshold:", asr_thresh)
        import asrpy
        asr = asrpy.ASR(sfreq=raw.info['sfreq'], cutoff=asr_thresh)
        # Fit on downsampled data for efficiency
        tmp = raw.copy().pick('eeg').resample(100)
        asr.fit(tmp)
        # Apply to full resolution EEG data
        eeg_picks = mne.pick_types(raw.info, eeg=True)
        raw._data[eeg_picks] = asr.transform(raw.copy().pick('eeg')).get_data()
    
    # ICA with ICLabel
    if use_ica:
        print("Running ICA with ICLabel")
        from mne.preprocessing import ICA
        try:
            from mne_icalabel import label_components
            
            eeg = raw.copy().pick('eeg')
            n_comp = min(len(eeg.ch_names) - len(eeg.info['bads']) - 1, 40)
            
            ica = ICA(
                n_components=n_comp, 
                method='infomax', 
                random_state=random_seed, 
                fit_params={"extended": True}
            )
            ica.fit(eeg, decim=3)
            
            # Label components
            labels = label_components(eeg, ica, method='iclabel')['labels']
            bads = [i for i, lab in enumerate(labels) if lab not in ('brain', 'other')]
            ica.exclude = bads
            
            # Apply ICA cleaning
            eeg_clean = ica.apply(eeg)
            eeg_picks = mne.pick_types(raw.info, eeg=True)
            raw._data[eeg_picks] = eeg_clean.get_data()
            
        except ImportError:
            print("Warning: mne_icalabel not available, skipping ICA cleaning")
    
    # Interpolate bad channels
    raw.interpolate_bads(reset_bads=False)
    
    return raw
