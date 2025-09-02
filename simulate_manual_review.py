#!/usr/bin/env python3
"""
Simulate manual QC review by modifying the CSV file
"""
import pandas as pd
from pathlib import Path

def simulate_manual_review():
    """Simulate manual review by editing the QC CSV."""
    csv_path = Path("/tmp/hep_output/qc_review.csv")
    
    if not csv_path.exists():
        print("QC CSV not found. Run QC stage first.")
        return
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Simulate manual review decisions
    # Mark first file as "ok" (no changes needed)
    df.loc[0, 'qc_status'] = 'ok'
    df.loc[0, 'notes'] = 'Good quality ECG'
    
    # Mark second file as "bad" but provide manual RR correction
    df.loc[1, 'qc_status'] = 'bad'
    df.loc[1, 'manual_est_rr_s'] = 0.75  # Faster heart rate estimate
    df.loc[1, 'notes'] = 'R-peaks missed, corrected with manual RR estimate'
    
    # Save back
    df.to_csv(csv_path, index=False)
    print(f"Updated QC CSV: {csv_path}")
    print("\nSimulated manual review:")
    print(df[['base', 'qc_status', 'manual_est_rr_s', 'notes']].to_string(index=False))

if __name__ == "__main__":
    simulate_manual_review()