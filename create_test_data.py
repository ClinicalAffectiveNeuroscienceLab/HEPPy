#!/usr/bin/env python3
"""
Test script to validate HEPPy workflow improvements
"""
import numpy as np
import mne
from pathlib import Path
import sys

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def create_test_edf(output_path: Path, duration_s: float = 60, sfreq: float = 256):
    """Create a synthetic EDF file with EEG and ECG for testing."""
    n_samples = int(duration_s * sfreq)
    
    # Create synthetic EEG data (10 channels)
    eeg_data = np.random.randn(10, n_samples) * 1e-5  # Scale to microvolts
    
    # Create synthetic ECG with realistic R-peaks
    t = np.arange(n_samples) / sfreq
    heart_rate = 70  # BPM
    rr_interval = 60 / heart_rate  # seconds
    
    # Generate R-peaks with some variability
    r_times = np.arange(0, duration_s, rr_interval)
    r_times += np.random.normal(0, 0.05, len(r_times))  # HRV
    r_samples = (r_times * sfreq).astype(int)
    r_samples = r_samples[r_samples < n_samples]
    
    # Create ECG signal with R-peaks
    ecg_data = np.zeros(n_samples)
    for r_samp in r_samples:
        if r_samp < n_samples - 10:
            # Simple R-wave spike
            ecg_data[r_samp:r_samp+5] = np.array([0, 0.5, 1, 0.5, 0]) * 1e-3
    
    # Add some noise
    ecg_data += np.random.randn(n_samples) * 1e-5
    
    # Combine data
    data = np.vstack([eeg_data, ecg_data[np.newaxis, :]])
    
    # Create channel names
    ch_names = [f'EEG {i+1}' for i in range(10)] + ['ECG']
    ch_types = ['eeg'] * 10 + ['ecg']
    
    # Create MNE info and raw
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    
    # Save as EDF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw.export(str(output_path), fmt='edf', overwrite=True)
    
    print(f"Created test EDF: {output_path}")
    print(f"Duration: {duration_s}s, R-peaks: {len(r_samples)}, HR: ~{len(r_samples)/duration_s*60:.1f} BPM")
    return output_path

if __name__ == "__main__":
    # Create test directory and data
    test_dir = Path("/tmp/test_data")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a few test files
    for i in range(2):
        create_test_edf(test_dir / f"test_subject_{i+1}.edf", duration_s=30)
    
    print(f"\nTest data created in: {test_dir}")
    print("Files created:")
    for f in test_dir.glob("*.edf"):
        print(f"  {f}")