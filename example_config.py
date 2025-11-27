# example_config.py
# Example HEP extraction config (previously `config_hep.py`).
# This is an example config - you will need to consider the contents carefully.

# I/O
save_stem       = "hep"                    # controls file names
input_glob = "*.edf"  # Adjust as needed
output_root    = "./output"              # will be created if needed
qc_csv_name    = "qc_review.csv"         # summary CSV of QC metrics
qc_dirname     = "qc_plots"              # directory for QC figures

# Preproc toggles
use_pyprep      = True
use_asr         = 15        # ASR cutoff in SD; set False/None to disable
use_ica         = True
random_seed     = 42
target_sfreq    = 100       # resample after filtering; set None to keep native

# Pyprep parameters
line_freqs = (50, 100)  # Not used in test
high_pass = 1.0         # Not used in test
low_pass = 100.0        # Not used in test
ref_chs = "eeg"
reref_chs = "eeg"
prep_ransac = True

# HRV
do_hrv          = True  # set False to disable

# Montage / channels
montage_name    = "standard_1020"  # or 'biosemi128', etc. - or a custom montage file path
rename_to_1020  = True             # light renaming (best‑effort)
# Reference choice: None ⇒ use average/reference from preprocessing; else list of electrode names
# e.g., linked ears: ["A1", "A2"],
# Note - the EEG will shift to average reference before applying these references, so the original reference will not be preserved.
reference_electrodes = None

# Epoching / rejection
tmin, tmax      = -0.2, 0.8
baseline        = (-0.2, -0.05)
amp_rej_uv      = 150.0           # global amplitude rejection (µV)
amp_window_s    = (0.1, 0.5)      # window to measure peak‑to‑peak
min_rr_s        = 0.7             # present RR threshold (beats with RR>=min_rr_s)
ecg_channel     = None            # None ⇒ first ECG found; else a name
stim_name       = "STI 014"       # name for synthetic stim, if you ever add one

# Logging
verbose         = True
