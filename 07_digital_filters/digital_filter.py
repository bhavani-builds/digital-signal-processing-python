import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ==========================================
# 1. Sampling Parameters
# ==========================================

fs = 200                 # Sampling frequency
duration = 2             # Signal duration

t = np.arange(0, duration, 1 / fs)

# ==========================================
# 2. Generate Clean Signal
# ==========================================

# Low-frequency signal
f_signal = 5

clean_signal = np.sin(2 * np.pi * f_signal * t)

# ==========================================
# 3. Generate High-Frequency Noise
# ==========================================

f_noise = 50

noise = 0.5 * np.sin(2 * np.pi * f_noise * t)

# Random noise
random_noise = 0.25 * np.random.randn(len(t))

# Noisy signal
noisy_signal = clean_signal + noise + random_noise

# ==========================================
# 4. Design Butterworth Low-Pass Filter
# ==========================================

cutoff_frequency = 10
filter_order = 4

nyquist_frequency = fs / 2

normalized_cutoff = cutoff_frequency / nyquist_frequency

b, a = butter(
    filter_order,
    normalized_cutoff,
    btype="low"
)

# ==========================================
# 5. Apply Filter
# ==========================================

filtered_signal = filtfilt(
    b,
    a,
    noisy_signal
)

# ==========================================
# 6. Plot Results
# ==========================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(14, 11)
)

# ------------------------------------------
# Clean Signal
# ------------------------------------------

axes[0].plot(
    t,
    clean_signal,
    color="royalblue",
    linewidth=2.5,
    label="Clean 5 Hz Signal"
)

axes[0].set_title(
    "Original Clean Signal",
    fontsize=18,
    fontweight="bold"
)

axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("Amplitude")

axes[0].grid(
    True,
    linestyle="--",
    alpha=0.4
)

axes[0].legend()

# ------------------------------------------
# Noisy Signal
# ------------------------------------------

axes[1].plot(
    t,
    noisy_signal,
    color="crimson",
    linewidth=1.5,
    label="Noisy Signal"
)

axes[1].plot(
    t,
    clean_signal,
    color="royalblue",
    linewidth=2,
    alpha=0.7,
    label="Original Signal"
)

axes[1].set_title(
    "Signal with High-Frequency Noise",
    fontsize=18,
    fontweight="bold"
)

axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Amplitude")

axes[1].grid(
    True,
    linestyle="--",
    alpha=0.4
)

axes[1].legend()

# ------------------------------------------
# Filtered Signal
# ------------------------------------------

axes[2].plot(
    t,
    noisy_signal,
    color="lightcoral",
    linewidth=1,
    alpha=0.5,
    label="Noisy Signal"
)

axes[2].plot(
    t,
    filtered_signal,
    color="seagreen",
    linewidth=2.8,
    label="Filtered Signal"
)

axes[2].set_title(
    "Butterworth Low-Pass Filter Output",
    fontsize=18,
    fontweight="bold"
)

axes[2].set_xlabel("Time (seconds)")
axes[2].set_ylabel("Amplitude")

axes[2].grid(
    True,
    linestyle="--",
    alpha=0.4
)

axes[2].legend()

# ------------------------------------------
# Overall Title
# ------------------------------------------

fig.suptitle(
    "Digital Signal Processing: Noise Removal using Low-Pass Filter",
    fontsize=21,
    fontweight="bold"
)

plt.tight_layout(
    rect=[0, 0, 1, 0.96]
)

plt.show()

# ==========================================
# Filter Information
# ==========================================

print("Sampling Frequency:", fs, "Hz")
print("Signal Frequency:", f_signal, "Hz")
print("Noise Frequency:", f_noise, "Hz")
print("Cutoff Frequency:", cutoff_frequency, "Hz")
print("Filter Order:", filter_order)
