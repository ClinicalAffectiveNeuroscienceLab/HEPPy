# heppy/config.py
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

    # Pyprep parameters
    line_freqs: Tuple[float, float] = (50.0, 100.0)
    high_pass: float = 1.0
    low_pass: float = 100.0
    ref_chs: Union[str, list] = "eeg"
    reref_chs: Union[str, list] = "eeg"
    prep_ransac: bool = True

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

    # HRV
    do_hrv: bool = False

    # Logging
    verbose: bool = True


def load_from_config_hep(module_name: str) -> HEPConfig:
    """Load configuration from a Python module."""
    try:
        mod = importlib.import_module(module_name)
        config_dict = {}
        for attr in dir(mod):
            if not attr.startswith('_') and attr != 'Path':
                value = getattr(mod, attr)
                # Avoid importing callables from the module as config values
                if not callable(value):
                    config_dict[attr] = value

        if 'output_root' in config_dict:
            config_dict['output_root'] = Path(config_dict['output_root'])
        return HEPConfig(**config_dict)
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {module_name}: {e}")

if __name__ == "__main__":
    # Example usage
    config = load_from_config_hep("config_hep")
    print(config)