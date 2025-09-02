# heppy/main.py
from __future__ import annotations
from pathlib import Path
from config import load_from_config_hep
from review import make_qc_checklist, apply_qc_and_extract


def QC():
    # first do QC checklist
    cfg = load_from_config_hep("test_config")
    print(cfg)
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    make_qc_checklist(cfg)

def HEP():
    # then apply QC and extract HEPs
    cfg = load_from_config_hep("test_config")
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    apply_qc_and_extract(cfg)


if __name__ == "__main__":
    HEP()
