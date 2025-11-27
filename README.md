# HEPPy — Heartbeat-Evoked Potential (HEP) extraction for EEG
## Updated 27-11-2025 - uses streamlit for GUI

HEPPy is a simple, user friendly pipeline for extracting heartbeat-evoked potentials (HEPs) from EEG recordings that include an ECG channel. It provides a **thin GUI** to:

1. run **ecg analysis** to detect R-peaks and generate per-file QRS plots;
2. **review** those plots to make sure the R peaks are fitted properly.
3. run **HEP** extraction using your QC decisions.
4. Optional **HRV** metrics

Outputs are standard MNE objects on disk plus a summary CSV. Optionally, HEPPy can compute **HRV metrics** via NeuroKit2.

---

## Features

* **Simple GUI** wrapper over a robust MNE-Python pipeline.
* **Automatic R-peak detection** (+ manual RR hint when needed).
* **QRS QC plots** for rapid screening.
* **Epoching and amplitude rejection** around R-peaks (configurable).
* **Error logs per file** with full Python tracebacks.
* **Optional HRV metrics** (time/frequency/non-linear) via NeuroKit2.
* Works with **EDF** or **FIF** inputs; handles typical montages.

## How to use the GUI (step-by-step)
The GUI requires an input directory, and supplies defaults for other values.
A config file can be set up to import new settings - recommended for reproducibilty/convenience.

## Note on AI use for development of GUI
While the underlying methods were written by myself, development of the streamlit-based app was largely AI driven.
If you are using the GUI, be aware that there may be some glitches as a result. Let me know if there are any errors that
pop up - which should display in the window.

## Installation
HEPPy requires Python 3.8+
requirements.txt lists the required packages.
To install, clone this repository and run:
```bash
pip install -r requirements.txt
```

The easiest way to run this is to load your python environment (with the installed requirements) and then run:
```bash
streamlit run heppy_gui.py
```

## Usage
1. Prepare your EEG data files in a directory (EDF or FIF format) with an ECG channel.
2. Launch the GUI using the command above.
3. Select your input directory and configure parameters as needed.
4. Run the ECG analysis to detect R-peaks and generate QRS plots. (optionally run HRV metrics)
5. Review the QRS plots and make QC decisions.
6. Run the HEP extraction using your QC decisions.
7. Find the outputs in the specified output directory.

## Todo:
- [ ] Add statistical analysis options
- [ ] Add some sort of file grouping/handling (?regex etc.)


## Licence

MIT. See `LICENSE`.

---

## Citation

If you use HEPPy in a paper or product, please cite:






