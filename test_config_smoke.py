file = r"F:\Chalfont_Preictal Samia 2025\pp\raw_fif\Chalfont1_Preictal_2_pp_raw.fif"

import mne

raw = mne.io.read_raw_fif(file, preload=False)

print(raw.ch_names)