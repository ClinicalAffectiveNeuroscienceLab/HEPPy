# Test configuration for HEPPy
from pathlib import Path

# I/O - using test directory
input_glob = "test_files/*.fif"
output_root = Path("test_files/hep_output")
save_stem = "hep"
qc_csv_name = "qc_review.csv"
qc_dirname = "qc_plots"

# Preproc toggles
use_pyprep = True  # Disable for faster testing
use_asr = True     # Disable for faster testing
use_ica = True     # Disable for testing (requires montage)
random_seed = 42
target_sfreq = 256

# Pyprep parameters
line_freqs = (50, 100)  # Not used in test
high_pass = 1.0         # Not used in test
low_pass = 100.0        # Not used in test
ref_chs = "eeg"
reref_chs = "eeg"
prep_ransac = False

# Montage / channels
montage_name = None  # Disable montage for test data
rename_to_1020 = True

# Epoching / rejection
tmin, tmax = -0.2, 0.8
baseline = (-0.2, -0.05)
amp_rej_uv = 150.0
amp_window_s = (0.1, 0.5)
min_rr_s = 0.5
ecg_channel = None
stim_name = "STI 014"

# Logging
verbose = True
