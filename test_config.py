# Test configuration for HEPPy
from pathlib import Path

# I/O - using test directory
input_glob = "/tmp/test_data/*.edf"
output_root = Path("/tmp/hep_output")
save_stem = "hep"
qc_csv_name = "qc_review.csv"
qc_dirname = "qc_plots"

# Preproc toggles
use_pyprep = False  # Disable for faster testing
use_asr = False     # Disable for faster testing
use_ica = False     # Disable for faster testing
random_seed = 42
target_sfreq = 256

# Montage / channels
montage_name = "standard_1020"
rename_to_1020 = True

# Epoching / rejection
tmin, tmax = -0.2, 0.8
baseline = (-0.2, -0.05)
amp_rej_uv = 150.0
amp_window_s = (0.1, 0.5)
min_rr_s = 0.7
ecg_channel = None
stim_name = "STI 014"

# Logging
verbose = True