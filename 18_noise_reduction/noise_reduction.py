import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt


np.random.seed(42)

# Signal settings
fs = 200
duration = 2

t = np.arange(
    0,
    duration,
    1 / fs
)


# ------------------------------------------
# Create clean signal
# ------------------------------------------

signal1 = np.sin(
    2 * np.pi * 5 * t
)

signal2 = 0.5 * np.sin(
    2 * np.pi * 12 * t
)

clean_signal = signal1 + signal2


# ------------------------------------------
# Add noise
# ------------------------------------------

noise = (
    0.5 * np.sin(2 * np.pi * 50 * t)
    + 0.3 * np.random.randn(len(t))
)

noisy_signal = (
    clean_signal + noise
)


# ------------------------------------------
# Design low-pass filter
# ------------------------------------------

cutoff = 20
order = 4

nyquist = fs / 2

normal_cutoff = cutoff / nyquist

b, a = butter(
    order,
    normal_cutoff,
    btype="low"
)


# ------------------------------------------
# Filter the noisy signal
# ------------------------------------------

filtered_signal = filtfilt(
    b,
    a,
    noisy_signal
)


# ------------------------------------------
# Calculate noise reduction
# ------------------------------------------

noise_before = np.mean(
    (noisy_signal - clean_signal) ** 2
)

noise_after = np.mean(
    (filtered_signal - clean_signal) ** 2
)

reduction = (
    1 - noise_after / noise_before
) * 100


print("Noise Reduction Analysis")
print("-----------------------------")

print(
    f"Noise before filtering : "
    f"{noise_before:.4f}"
)

print(
    f"Noise after filtering  : "
    f"{noise_after:.4f}"
)

print(
    f"Noise reduction        : "
    f"{reduction:.2f}%"
)


# ------------------------------------------
# Plot signals
# ------------------------------------------

fig, axes = plt.subplots(
    3,
    1,
    figsize=(14, 11)
)


# Clean signal

axes[0].plot(
    t,
    clean_signal,
    color="royalblue",
    linewidth=2.5,
    label="Clean Signal"
)

axes[0].set_title(
    "Original Clean Signal",
    fontsize=17,
    fontweight="bold"
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


# Noisy signal

axes[1].plot(
    t,
    noisy_signal,
    color="crimson",
    linewidth=1.4,
    label="Noisy Signal"
)

axes[1].set_title(
    "Signal with Noise",
    fontsize=17,
    fontweight="bold"
)

axes[1].set_ylabel(
    "Amplitude"
)

axes[1].grid(
    True,
    linestyle="--",
    alpha=0.4
)

axes[1].legend()


# Filtered signal

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
    color="mediumseagreen",
    linewidth=2.5,
    label="Filtered Signal"
)

axes[2].set_title(
    "Signal After Noise Reduction",
    fontsize=17,
    fontweight="bold"
)

axes[2].set_xlabel(
    "Time (seconds)"
)

axes[2].set_ylabel(
    "Amplitude"
)

axes[2].grid(
    True,
    linestyle="--",
    alpha=0.4
)

axes[2].legend()


fig.suptitle(
    "Noise Reduction using Digital Low-Pass Filtering",
    fontsize=21,
    fontweight="bold"
)

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)

plt.savefig(
    "Figure_18.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
