# HEPPy package initialization
from .config import HEPConfig, load_from_config_hep
from .ecg import detect_rpeaks, refine_with_rr_hint, save_ecg_qc_plot
from .epochs import build_hep_epochs, save_hep_epochs
from .review import make_qc_checklist, apply_qc_and_extract

__version__ = "0.1.0"
__all__ = [
    "HEPConfig", "load_from_config_hep",
    "detect_rpeaks", "refine_with_rr_hint", "save_ecg_qc_plot",
    "build_hep_epochs", "save_hep_epochs", 
    "make_qc_checklist", "apply_qc_and_extract"
]