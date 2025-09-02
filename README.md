# HEPPy

# UNDER CONSTRUCTION

This WILL BE a simple package for extracting the HEP from EEG data (with an ECG lead).

It will run on a directory of EEG files in the form of .edf

## Installation
Can be installed by using git clone etc.

```bash
git clone

cd HEPPy
pip install -e .
```

## Usage
Currently this is written poorly, and you run one function, manually check things, then add another.

In main:
run "QC"
then run "HEP"

You will need to set your config file up appropriately.
You can see an example config file in the repo.


## Requirements
- Python 3.8+
- numpy
- mne
- asrpy
- mne-icalabel
- torch
- pyprep

## To do
- [ ] Make it have a simple GUI
- [ ] Make it more modular
- [ ] Add more documentation
- [ ] Add plotting functions
- [ ] Add more robust error handling and logging

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citing
If you use this code, please cite this repository. Please review this repository at time of using the
code because there might be a paper to cite!

