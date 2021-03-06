# Shannon-Energy-R-Peak-Detection
Python Implementation of Shannon Energy R Peak Detection Method

This code is an implementation of the Shannon energy R-peak detection method described by Manikandan and Soman (2012).

Reference: Manikandan, M. S., & Soman, K. P. (2012). A novel method for detecting R-peaks in electrocardiogram (ECG) signal. Biomedical Signal Processing and Control, 7(2), 118–128. https://doi.org/10.1016/j.bspc.2011.03.004

## Required Packages
* `scipy`
* `numpy`
* `wfdb`
* `bokeh`

The code pulls the EKG data from WFDB website (https://physionet.org/physiobank/database/mitdb/), processes the data, and plots the outcome.
The output does not match with the publication. I'm working on it to match it with the pub. If anyone can point out what's wrong, I will appreciate it.
