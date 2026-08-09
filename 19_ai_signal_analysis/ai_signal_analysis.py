import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)

# Sampling settings
fs = 200
duration = 2

t = np.arange(
    0,
    duration,
    1 / fs
)


# ------------------------------------------
# Create a test signal
# ------------------------------------------

signal1 = 1.2 * np.sin(
    2 * np.pi * 5 * t
)

signal2 = 0.6 * np.sin(
    2 * np.pi * 12 * t
)

noise = 0.3 * np.random.randn(
    len(t)
)

signal = signal1 + signal2 + noise


# ------------------------------------------
# Basic signal measurements
# ------------------------------------------

mean_value = np.mean(signal)

std_value = np.std(signal)

rms_value = np.sqrt(
    np.mean(signal ** 2)
)

energy = np.sum(
    signal ** 2
)

peak = np.max(
    np.abs(signal)
)


# ------------------------------------------
# FFT analysis
# ------------------------------------------

fft_result = np.fft.rfft(signal)

frequencies = np.fft.rfftfreq(
    len(signal),
    1 / fs
)

magnitude = np.abs(
    fft_result
)

magnitude[0] = 0

dominant_index = np.argmax(
    magnitude
)

dominant_frequency = frequencies[
    dominant_index
]


# ------------------------------------------
# Estimate noise level
# ------------------------------------------

signal_power = np.mean(
    signal ** 2
)

noise_power = np.var(
    signal - np.mean(signal)
)

snr = 10 * np.log10(
    signal_power / noise_power
)


# ------------------------------------------
# Simple AI-style classification
# ------------------------------------------

if rms_value > 1.0 and dominant_frequency < 8:

    signal_type = "Strong Low-Frequency Signal"

elif dominant_frequency < 8:

    signal_type = "Low-Frequency Signal"

elif dominant_frequency < 20:

    signal_type = "Medium-Frequency Signal"

else:

    signal_type = "High-Frequency Signal"


# ------------------------------------------
# Print analysis
# ------------------------------------------

print("=" * 50)
print("AI SIGNAL ANALYSIS")
print("=" * 50)

print(f"Mean               : {mean_value:.3f}")
print(f"Standard Deviation : {std_value:.3f}")
print(f"RMS                : {rms_value:.3f}")
print(f"Energy             : {energy:.3f}")
print(f"Peak Amplitude     : {peak:.3f}")

print(
    f"Dominant Frequency : "
    f"{dominant_frequency:.2f} Hz"
)

print(
    f"Estimated SNR      : "
    f"{snr:.2f} dB"
)

print(
    f"Signal Analysis    : "
    f"{signal_type}"
)


# ------------------------------------------
# Visualization
# ------------------------------------------

fig, axes = plt.subplots(
    2,
    1,
    figsize=(14, 9)
)


# Time-domain signal

axes[0].plot(
    t,
    signal,
    color="royalblue",
    linewidth=1.8,
    label="Analyzed Signal"
)

axes[0].set_title(
    "Signal in Time Domain",
    fontsize=17,
    fontweight="bold"
)

axes[0].set_xlabel(
    "Time (seconds)"
)

axes[0].set_ylabel(
    "Amplitude"
)

axes[0].grid(
    True,
    linestyle="--",
    alpha=0.4
)

axes[0].legend()


# Frequency-domain signal

axes[1].plot(
    frequencies,
    magnitude,
    color="crimson",
    linewidth=2,
    label="FFT Spectrum"
)

axes[1].scatter(
    dominant_frequency,
    magnitude[dominant_index],
    color="darkorange",
    s=120,
    zorder=5,
    label=(
        f"Dominant Frequency: "
        f"{dominant_frequency:.2f} Hz"
    )
)

axes[1].set_xlim(
    0,
    30
)

axes[1].set_title(
    "Frequency Spectrum",
    fontsize=17,
    fontweight="bold"
)

axes[1].set_xlabel(
    "Frequency (Hz)"
)

axes[1].set_ylabel(
    "Magnitude"
)

axes[1].grid(
    True,
    linestyle="--",
    alpha=0.4
)

axes[1].legend()


fig.suptitle(
    "AI-Based Signal Analysis",
    fontsize=21,
    fontweight="bold"
)

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)

plt.savefig(
    "Figure_19.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
