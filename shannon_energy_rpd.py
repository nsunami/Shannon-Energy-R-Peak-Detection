# =================================================================
# Shannon Energy Envelope R-Peak Detection
# =================================================================
# This script implements the R-peak detection algorithm described in:
# Manikandan, M. S., & Soman, K. P. (2012).
# A novel method for detecting R-peaks in electrocardiogram (ECG) signal.
# Biomedical Signal Processing and Control, 7(2), 118–128.
# https://doi.org/10.1016/j.bspc.2011.03.004


# =================================================================
# Required packages
# =================================================================
# - scipy.signal
# - numpy
# - bokeh
# - wfbd python (for reading test data)
# - urlib

# Test data is taken from:
# https://physionet.org/physiobank/database/mitdb/
# 205
# https://physionet.org/physiobank/database/mitdb/205.dat

from scipy.signal import hilbert, cheby1, filtfilt
import numpy as np
from bokeh.plotting import figure, output_file, show
import wfdb

# =================================================================
# Parameters
# =================================================================
# Get the sample data
f205 = wfdb.rdsamp('mitdb/205', pb_dir='mitdb')
# EKG array
EKG = f205[0][:,0]
# Define the sampling rate
sampling_rate = f205[1]['fs']

# Time
time_sec = np.arange(0, len(EKG)*sampling_rate)


# =================================================================
# Define Filter Functions
# There are two filters in this proceudre
# 1. Chebyshev Type I filter
# 2. Running mean filter
# =================================================================

# Forward and backward filtering using filtfilt.
def cheby1_bandpass_filter(data, lowcut, highcut, fs, order=5, rp=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = cheby1(order, rp=rp, Wn=[low, high], btype='bandpass')
    y = filtfilt(b, a, data)
    return y


# Running mean filter function from stackoverflow
# https://stackoverflow.com/a/27681394/6205282
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# =================================================================
# Actual processing of Signal
# =================================================================

# Apply Chebyshev Type I Bandpass filter
# Low cut frequency = 6 Hz
# High cut frequency = 18
EKG_f = cheby1_bandpass_filter(EKG, lowcut=6, highcut=18, fs=sampling_rate, order=4)

# First-order differencing difference (Eq. 1)
dn = (np.append(EKG_f[1:], 0) - EKG_f)

# Equations presented in the paper:
# Eq. 2
dtn = dn/(np.max(abs(dn)))
# Eq. 3
an = abs(dtn)
# Eq. 4
en = an**2
# Eq. 5
sen = -abs(dtn) * np.log10(abs(dtn))
# Eq. 6
sn = -(dtn**2) * np.log10(dtn**2)

# Zero-Phase Filtering, Running Mean
# https://dsp.stackexchange.com/a/27317
# 360 samples/s : 55 samples (length)
# 500 samples/s : 76 samples
# Normal QRS duration is .12 sec. 500*.12 = 60
window_len = 79

# Moving Average of the Filtered Signal
sn_f = np.insert(running_mean(sn, window_len), 0, [0] * (window_len - 1))

# Hilbert Transformation
zn = np.imag(hilbert(sn_f))

# Moving Average of the Hilbert Transformed Signal
# 2.5 sec from Manikanda (900 samples)
# 2.5 sec in 500 Hz == 1250 samples
ma_len = 1250
zn_ma = np.insert(running_mean(zn, ma_len), 0, [0] * (ma_len - 1))

# Get the difference between the Hilbert signal and the MA filtered signal
zn_ma_s = zn - zn_ma


# =================================================================
# R-Peak Detection
# =================================================================
# Look for points crossing zero
# https://stackoverflow.com/a/28766902/6205282
# Original paper: +/- 25 frames from the ref point (360Hz)
# if 500Hz---35 frames. With some trials, it seems that ~60 is good.

# Find points crossing zero upwards (negative to positive)
idx = np.argwhere(np.diff(np.sign(zn_ma_s)) > 0).flatten().tolist()
# Prepare a container for windows
idx_search = []
id_maxes = np.empty(0, dtype=int)
search_window_half = round(sampling_rate * .12)  # <------------ Parameter
for i in idx:
    lows = np.arange(i-search_window_half, i)
    highs = np.arange(i+1, i+search_window_half+1)
    if highs[-1] > len(EKG):
        highs = np.delete(highs, np.arange(np.where(highs == len(EKG))[0], len(highs)+1))
    ekg_window = np.concatenate((lows, [i], highs))
    idx_search.append(ekg_window)
    ekg_window_wave = EKG[ekg_window]
    id_maxes = np.append(id_maxes, ekg_window[np.where(ekg_window_wave == np.max(ekg_window_wave))[0]])

# =================================================================
# Output Plots
output_file("shannon_energy.html")
# create a new plot with a title and axis labels
p_width = 1500  # <------------ Parameter
p_height = 500  # <------------ Parameter
p = figure(title="Shannon Energy Envelope R-Peak Detection", x_axis_label='Samples', y_axis_label='EKG',
           width=p_width, height=p_height)
end = slice(0, len(EKG))
p.line(time_sec[end], EKG[end], legend="Original", color="grey", line_width=2)
p.circle(time_sec[id_maxes][end], EKG[id_maxes][end])
show(p)
