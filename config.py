# heppy/config.py
from __future__ import annotations
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

@dataclass
class HEPConfig:
    """Configuration for HEP extraction pipeline."""
    input_glob: Union[str, list]
    output_root: Path
    save_stem: str = "hep"
    qc_csv_name: str = "qc_review.csv"
    qc_dirname: str = "qc_plots"
    
    # Preprocessing
    use_pyprep: bool = True
    use_asr: Optional[float] = 20.0
    use_ica: bool = True
    random_seed: int = 42
    target_sfreq: Optional[float] = 256.0
    
    # Montage
    montage_name: Optional[str] = "standard_1020"
    rename_to_1020: bool = True
    
    # Epoching
    tmin: float = -0.2
    tmax: float = 0.8
    baseline: Optional[Tuple[float, float]] = (-0.2, -0.05)
    amp_rej_uv: float = 150.0
    amp_window_s: Tuple[float, float] = (0.1, 0.5)
    min_rr_s: float = 0.7
    ecg_channel: Optional[str] = None
    stim_name: str = "STI 014"
    
    # Logging
    verbose: bool = True

def load_from_config_hep(module_name: str) -> HEPConfig:
    """Load configuration from a Python module."""
    try:
        mod = importlib.import_module(module_name)
        
        # Extract attributes from the config module, excluding built-ins and imports
        config_dict = {}
        for attr in dir(mod):
            if not attr.startswith('_') and attr != 'Path':  # Exclude Path import
                value = getattr(mod, attr)
                # Skip imported modules/functions
                if not callable(value) or attr in {
                    'input_glob', 'output_root', 'save_stem', 'qc_csv_name', 'qc_dirname',
                    'use_pyprep', 'use_asr', 'use_ica', 'random_seed', 'target_sfreq',
                    'montage_name', 'rename_to_1020', 'tmin', 'tmax', 'baseline',
                    'amp_rej_uv', 'amp_window_s', 'min_rr_s', 'ecg_channel', 'stim_name', 'verbose'
                }:
                    config_dict[attr] = value
        
        # Convert paths
        if 'output_root' in config_dict:
            config_dict['output_root'] = Path(config_dict['output_root'])
            
        return HEPConfig(**config_dict)
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {module_name}: {e}")