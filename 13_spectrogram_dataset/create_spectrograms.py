import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square, sawtooth, spectrogram

np.random.seed(42)

fs = 200
duration = 1

t = np.arange(0, duration, 1 / fs)

# Generate one example from each class

frequency = 5

sine = np.sin(2 * np.pi * frequency * t)

square_wave = square(
    2 * np.pi * frequency * t
)

saw = sawtooth(
    2 * np.pi * frequency * t
)

noise = 0.4 * np.random.randn(len(t))

noisy_sine = sine + noise


signals = [
    sine,
    square_wave,
    saw,
    noisy_sine
]

names = [
    "Sine Wave",
    "Square Wave",
    "Sawtooth Wave",
    "Noisy Sine Wave"
]

colors = [
    "royalblue",
    "darkorange",
    "mediumseagreen",
    "crimson"
]


# Create spectrograms

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)

axes = axes.flatten()

for i in range(4):

    frequencies, times, Sxx = spectrogram(
        signals[i],
        fs=fs
    )

    axes[i].pcolormesh(
        times,
        frequencies,
        10 * np.log10(Sxx + 1e-10),
        shading="gouraud",
        cmap="turbo"
    )

    axes[i].set_title(
        names[i],
        fontsize=16,
        fontweight="bold"
    )

    axes[i].set_xlabel(
        "Time (seconds)"
    )

    axes[i].set_ylabel(
        "Frequency (Hz)"
    )

    axes[i].set_ylim(0, 50)


fig.suptitle(
    "Spectrogram Representation of Different Signals",
    fontsize=21,
    fontweight="bold"
)

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)

plt.show()
