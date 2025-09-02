# HEP extraction config (simple)

# This is an example config - you will need to consider the contents carefully

from pathlib import Path

# I/O
input_glob      = r"D:/RootFolder/**/*.edf"   # or a list of paths
output_root     = Path(r"D:/RootFolder/HEP_output")
save_stem       = "hep"                    # controls file names

# Preproc toggles
use_pyprep      = True
use_asr         = 20        # ASR cutoff in SD; set False/None to disable
use_ica         = True
random_seed     = 42
target_sfreq    = 256       # resample after filtering; set None to keep native

# Pyprep parameters
line_freqs = (50, 100)  # Not used in test
high_pass = 1.0         # Not used in test
low_pass = 100.0        # Not used in test
ref_chs = "eeg"
reref_chs = "eeg"
prep_ransac = True

# Montage / channels
montage_name    = "standard_1020"  # or 'biosemi128', etc. - or a csutom montage file path
rename_to_1020  = True             # light renaming (best‑effort)

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
