# _configuration_handler.py
"""
Centralized configuration loader.
Loads configuration values from modules in order and returns a typed HEPConfig.
Order (later modules override earlier ones):
 - config_base (optional)
 - example_config (optional)
 - config_local (optional, gitignored by convention)

Provides:
 - HEPConfig dataclass (same schema as previous `config.py`)
 - load_config(module_names=None) -> HEPConfig
 - load_from_config_hep(module_name) -> HEPConfig  # backward compatible shim
"""
import importlib
# warnings intentionally not used; keep minimal imports
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Any


@dataclass
class HEPConfig:
    # I/O
    input_glob: Union[str, list] = "*.edf"
    output_root: Path = Path("./output")
    save_stem: str = "hep"
    qc_csv_name: str = "qc_review.csv"
    qc_dirname: str = "qc_plots"
    input_root: Optional[Path] = None

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


def _collect_module_values(module_name: str) -> Dict[str, Any]:
    """Import a module by name and return dict of public, non-callable attributes."""
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return {}

    vals = {
        k: getattr(mod, k)
        for k in dir(mod)
        if not k.startswith("_")
        and k != "Path"
        and not callable(getattr(mod, k))
    }
    return vals


def load_config(module_names: Optional[List[str]] = None) -> HEPConfig:
    """Load and merge configuration modules into a HEPConfig.

    module_names: list of module names to search in order (later override earlier).
    If None, defaults to ['config_base', 'example_config', 'config_local'].
    """
    if module_names is None:
        module_names = ["config_base", "example_config", "config_local"]

    merged: Dict[str, Any] = {}
    for name in module_names:
        vals = _collect_module_values(name)
        if vals:
            merged.update(vals)

    # Keep only allowed fields
    allowed = {f.name for f in fields(HEPConfig)}
    cfg = {k: v for k, v in merged.items() if k in allowed}

    # Cast paths
    if "output_root" in cfg:
        cfg["output_root"] = Path(cfg["output_root"])
    if "input_root" in cfg and cfg["input_root"] is not None:
        cfg["input_root"] = Path(cfg["input_root"])

    return HEPConfig(**cfg)


# Backwards-compatible helper with the old name/signature
def load_from_config_hep(module_name: str) -> HEPConfig:
    """Compatibility wrapper: load config from a single module (old API).
    This will behave like the previous implementation: it loads only from the
    provided module and maps values to HEPConfig fields.
    """
    vals = _collect_module_values(module_name)
    allowed = {f.name for f in fields(HEPConfig)}
    cfg = {k: v for k, v in vals.items() if k in allowed}

    # Cast paths
    if "output_root" in cfg:
        cfg["output_root"] = Path(cfg["output_root"])
    if "input_root" in cfg and cfg["input_root"] is not None:
        cfg["input_root"] = Path(cfg["input_root"])

    if not cfg:
        raise RuntimeError(f"Failed to load config: module {module_name} not found or empty")

    return HEPConfig(**cfg)
