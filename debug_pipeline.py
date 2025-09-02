#!/usr/bin/env python3
"""
Debug script to test individual stages of the pipeline
"""
import sys
sys.path.append('.')

from config import load_from_config_hep
from review import _iter_inputs
from preprocess import apply_montage, clean_raw
from ecg import detect_rpeaks, refine_with_rr_hint, save_ecg_qc_plot
from epochs import build_hep_epochs, save_hep_epochs
import pandas as pd
import mne
import numpy as np
from pathlib import Path

def debug_pipeline():
    """Debug the pipeline step by step."""
    cfg = load_from_config_hep('test_config')
    df = pd.read_csv(cfg.output_root / cfg.qc_csv_name)
    
    # Test first file only
    row = df.iloc[0]
    f = str(row["file"])
    base = str(row["base"])
    status = str(row.get("qc_status", "")).lower().strip()
    
    print(f"Processing: {base}")
    print(f"Status: {status}")
    
    # Step 1: Load and preprocess
    print("Step 1: Loading raw data...")
    raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
    print(f"  Loaded: {len(raw.ch_names)} channels, {raw.n_times} samples")
    
    print("Step 2: Applying montage...")
    raw = apply_montage(raw, cfg)
    print(f"  After montage: {len(raw.ch_names)} channels")
    print(f"  Has montage: {raw.get_montage() is not None}")
    
    print("Step 3: Cleaning raw data...")
    raw, prov = clean_raw(raw, cfg)
    print(f"  After cleaning: {len(raw.ch_names)} channels")
    
    print("Step 4: Detecting R-peaks...")
    r, rr, ecg_clean = detect_rpeaks(raw, ecg_name=cfg.ecg_channel)
    print(f"  R-peaks detected: {len(r)}")
    print(f"  RR intervals: {len(rr)}")
    
    print("Step 5: Building HEP epochs...")
    hep = build_hep_epochs(raw, r, rr, cfg)
    print(f"  HEP epochs created: {len(hep)}")
    print(f"  Epochs has montage: {hep.get_montage() is not None}")
    
    print("Step 6: Saving epochs...")
    outdir = cfg.output_root
    try:
        path = save_hep_epochs(hep, outdir, cfg.save_stem, base)
        print(f"  Saved successfully: {path}")
    except Exception as e:
        print(f"  Save failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_pipeline()