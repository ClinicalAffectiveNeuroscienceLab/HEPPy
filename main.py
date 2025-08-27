# heppy/main.py
from __future__ import annotations
from pathlib import Path
import argparse
from .config import load_from_config_hep
from .review import make_qc_checklist, apply_qc_and_extract

def main():
    p = argparse.ArgumentParser(description="HEPPy: Reliable HEP extraction with manual ECG QC")
    p.add_argument("--config-module", default="config_hep", help="Python module to import (e.g., config_hep)")
    p.add_argument("--stage", choices=["qc", "extract"], default="qc",
                   help="qc: create QC pngs + CSV; extract: read QC CSV and save HEP epochs")
    args = p.parse_args()

    cfg = load_from_config_hep(args.config_module)
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    if args.stage == "qc":
        df = make_qc_checklist(cfg)
        print(f"[heppy] Wrote QC CSV with {len(df)} rows → {cfg.output_root / cfg.qc_csv_name}")
        print(f"[heppy] ECG PNGs in → {cfg.output_root / cfg.qc_dirname}")
    else:
        df = apply_qc_and_extract(cfg)
        print(f"[heppy] Processed {len(df)} files. Summary → {cfg.output_root / f'{cfg.save_stem}_summary.csv'}")

if __name__ == "__main__":
    main()
