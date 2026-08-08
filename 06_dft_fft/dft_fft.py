import numpy as np
import matplotlib.pyplot as plt

# Sampling parameters
fs = 100
N = 100

# Time vector
t = np.arange(N) / fs

# Generate two sinusoidal components
f1 = 5
f2 = 12

signal = (
    1.5 * np.sin(2 * np.pi * f1 * t)
    + 0.8 * np.sin(2 * np.pi * f2 * t)
)

# -------------------------------
# DFT
# -------------------------------
dft = np.fft.fft(signal)
frequencies = np.fft.fftfreq(N, 1 / fs)

# Magnitude spectrum
magnitude = np.abs(dft) / N

# Only positive frequencies
positive = frequencies >= 0

# -------------------------------
# FFT
# -------------------------------
fft_result = np.fft.fft(signal)

print("Signal contains frequencies:", f1, "Hz and", f2, "Hz")

# -------------------------------
# Plot
# -------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 9))

# Time-domain signal
axes[0].plot(
    t,
    signal,
    linewidth=2.5,
    label="Composite Signal"
)

axes[0].set_title(
    "Time-Domain Signal",
    fontsize=17,
    fontweight="bold"
)

axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("Amplitude")

axes[0].grid(
    True,
    linestyle="--",
    alpha=0.5
)

axes[0].legend()

# Frequency-domain spectrum
axes[1].plot(
    frequencies[positive],
    magnitude[positive],
    linewidth=2.5,
    marker="o",
    markersize=5,
    label="FFT Magnitude Spectrum"
)

axes[1].set_title(
    "Frequency-Domain Spectrum using FFT",
    fontsize=17,
    fontweight="bold"
)

axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Magnitude")

axes[1].set_xlim(0, 30)

axes[1].grid(
    True,
    linestyle="--",
    alpha=0.5
)

axes[1].legend()

plt.suptitle(
    "Digital Signal Processing: DFT and FFT Analysis",
    fontsize=20,
    fontweight="bold"
)

plt.tight_layout()

plt.show()
