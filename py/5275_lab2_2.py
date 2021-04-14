# !pip install -U mne

from matplotlib import pyplot as plt
import numpy as np
import time, mne

BASE_DIR = ""

raw = mne.io.read_raw_eeglab(BASE_DIR + "sXD_5678.set")
# print(raw.info)
print(raw.ch_names)
mne.rename_channels(raw.info, mapping = 
	{"FP1": "Fp1", "FP2": "Fp2", 
	"PZ": "Pz", "FZ": "Fz", "CZ": "Cz", 
	"FCZ": "FCz", "CPZ": "CPz", "OZ": "Oz"})
print(raw.ch_names)

# setting montage and add "vehicle positio" pt
montage_1020 = mne.channels.make_standard_montage("standard_1020")
montage = mne.channels.make_dig_montage(ch_pos = {"vehicle positio": [0, -50, 2]}, coord_frame = "unknown")
print(montage_1020)
montage_1020 = montage_1020.__add__(montage)
print(montage_1020)

## Problem 6
eeg = raw.copy()
# plot 2D channel location map
eeg.set_montage(montage_1020)
eeg.plot_sensors(ch_type = "eeg", show_names = True, sphere = [0,0,0,0.1])
# re-reference data by (A1+A2)/2
refchan = ["A1", "A2"]
mne.set_eeg_reference(eeg, ref_channels = refchan)
# Down-sampling to 250Hz
eeg = eeg.resample(250)
# drop A1,A2
eeg = eeg.drop_channels(refchan)
# Run ICA
ica = mne.preprocessing.ICA(n_components = len(eeg.ch_names))
ica.fit(eeg)
# plot component map
ica.plot_components()

# denoising by ICA
reconstruct = eeg.copy()
ica.exclude = [0, 2, 3, 7, 8, 11]
ica.apply(reconstruct)

# plot the eeg signal
print("=> original EEG signal")
fig1 = eeg.plot(duration = 10.0, n_channels = 33)
print("=> EEG signal after ICA")
fig2 = reconstruct.plot(duration = 10.0, n_channels = 33)

## Problem 7
eeg2 = raw.copy()
eeg2.drop_channels(["vehicle positio"])
# plot 2D channel location map
montage_1020 = mne.channels.make_standard_montage("standard_1020")
eeg2.set_montage(montage_1020)
eeg2.plot_sensors(ch_type = "eeg", show_names = True)
# re-reference data by (A1+A2)/2
refchan = ["A1", "A2"]
mne.set_eeg_reference(eeg2, ref_channels = refchan)
# Down-sampling to 250Hz
eeg2 = eeg2.resample(250)
# Bandpass filtering [1,50] Hz
eeg2 = eeg2.filter(8.0, 13.0, filter_length = 826)
# drop A1,A2
eeg2 = eeg2.drop_channels(refchan)
# Run ICA
ica = mne.preprocessing.ICA(n_components = len(eeg2.ch_names))
ica.fit(eeg2)
# plot component map
ica.plot_components()

# denoising by ICA
reconstruct2 = eeg2.copy()
ica.exclude = [0, 1, 10, 23, 24]
ica.apply(reconstruct2)

# plot the eeg signal
print("=> original EEG signal")
fig1 = eeg2.plot(duration = 10.0, n_channels = 33)
print("=> EEG signal after ICA")
fig2 = reconstruct2.plot(duration = 10.0, n_channels = 33)