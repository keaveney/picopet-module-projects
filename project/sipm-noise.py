
# Importing the simulation module

import SiPM_MPPC.sipm as sipm
import matplotlib.pylab as plt
import numpy as np

# Creating a single sipm pulse
# Input parameters

Rt = 2e-9 # Rising time in seconds 
Ft = 50e-9   # Falling time in seconds
A = 1 # Pulse amplitude (pe) photo-electron
R = 0.5 # Time step in ns

pulse = sipm.Pulse(Rt, Ft, A, R, plot=True)
# Output
# pulse, sipm pulse shape with time step R



# Simulating a sipm signal during a recording window
# Input parameters

DCR = 159.6e3  # Dark count rate in Hz/mm2
p_size = 36.0 # SiPM size mm2
CT = 0.31 # Crosstalk normalized to 1
AP = 0.01 # Afterpulse normalized to 1
T_rec = 55e-9 # Recovery time in ns
T_AP = 14.8e-9 # Trapp releasing time in ns
sigma = 0.1 # Amplitude variance in pe
W = 1000  # Recording window in ns
Np = 1 # Number of SiPM

signal, time = sipm.MPPC(pulse, Np, DCR, p_size, CT, AP, T_rec, T_AP, W, R, sigma)
# Output
# signal, sipm signal amplitude in pe
# time, sipm signal time in ns



# Generating the peak spectrum and inter-time distributions

A, I, X, Y = sipm.Amplitude_Intertime(signal, Np, W, R, plot=True)
# Output
# A, amplitude vector in pe
# I, time difference between consecutive pulses in s
# X, peak spectrum x-axis
# Y, peak spectrum y-axis



# Generating the DCR vs. threshold curve
# Input parameters

Lt = 0.1 # Lower threshold in pe
Ut = 8 # Upper threshold in pe
Pt = 200 # Threshold evaluation points

Th, Noise = sipm.DCR_threshold(signal, W, R, Lt, Ut, Pt, plot=True)
# Output
# Th, threshold vector in pe
# Noise, noise frequency in Hz
