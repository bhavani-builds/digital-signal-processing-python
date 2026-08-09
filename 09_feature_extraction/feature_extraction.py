import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square, sawtooth


# ==========================================
# 1. Dataset Parameters
# ==========================================

np.random.seed(42)

num_samples = 100
signal_length = 200

fs = 200

t = np.arange(signal_length) / fs

X = []
y = []


# ==========================================
# 2. Generate Signals
# ==========================================

for i in range(num_samples):

    # Sine Wave
    frequency = np.random.uniform(3, 8)

    sine_signal = np.sin(
        2 * np.pi * frequency * t
    )

    X.append(sine_signal)
    y.append(0)

    # Square Wave
    frequency = np.random.uniform(3, 8)

    square_signal = square(
        2 * np.pi * frequency * t
    )

    X.append(square_signal)
    y.append(1)

    # Sawtooth Wave
    frequency = np.random.uniform(3, 8)

    sawtooth_signal = sawtooth(
        2 * np.pi * frequency * t
    )

    X.append(sawtooth_signal)
    y.append(2)

    # Noisy Sine Wave
    frequency = np.random.uniform(3, 8)

    clean_signal = np.sin(
        2 * np.pi * frequency * t
    )

    noise = 0.4 * np.random.randn(
        signal_length
    )

    noisy_signal = clean_signal + noise

    X.append(noisy_signal)
    y.append(3)


X = np.array(X)
y = np.array(y)


# ==========================================
# 3. Feature Extraction Function
# ==========================================

def extract_features(signal):

    # Mean
    mean_value = np.mean(signal)

    # Standard deviation
    std_value = np.std(signal)

    # RMS
    rms_value = np.sqrt(
        np.mean(signal ** 2)
    )

    # Energy
    energy_value = np.sum(
        signal ** 2
    )

    # Peak amplitude
    peak_value = np.max(
        np.abs(signal)
    )

    # Zero Crossing Rate
    zero_crossings = np.sum(
        np.diff(np.sign(signal)) != 0
    )

    zero_crossing_rate = (
        zero_crossings / len(signal)
    )

    # FFT
    fft_values = np.fft.rfft(signal)

    frequencies = np.fft.rfftfreq(
        len(signal),
        1 / fs
    )

    magnitude = np.abs(fft_values)

    # Ignore DC component
    magnitude[0] = 0

    dominant_frequency = frequencies[
        np.argmax(magnitude)
    ]

    return [
        mean_value,
        std_value,
        rms_value,
        energy_value,
        peak_value,
        zero_crossing_rate,
        dominant_frequency
    ]


# ==========================================
# 4. Extract Features for All Signals
# ==========================================

features = []

for signal in X:

    signal_features = extract_features(
        signal
    )

    features.append(signal_features)


features = np.array(features)


# ==========================================
# 5. Display Dataset
# ==========================================

feature_names = [
    "Mean",
    "Standard Deviation",
    "RMS",
    "Energy",
    "Peak",
    "Zero Crossing Rate",
    "Dominant Frequency"
]

print("=" * 60)
print("FEATURE EXTRACTION")
print("=" * 60)

print("Original signal dataset :", X.shape)
print("Feature dataset         :", features.shape)

print("\nFeature Names:")

for name in feature_names:
    print("-", name)


# ==========================================
# 6. Display First Five Samples
# ==========================================

print("\nFirst 5 Feature Vectors:\n")

for i in range(5):

    print(
        f"Sample {i + 1}:",
        np.round(features[i], 3)
    )


# ==========================================
# 7. Visualize Feature Distribution
# ==========================================

plt.figure(figsize=(13, 8))

colors = [
    "royalblue",
    "darkorange",
    "mediumseagreen",
    "crimson"
]

labels = [
    "Sine",
    "Square",
    "Sawtooth",
    "Noisy Sine"
]

for class_id in range(4):

    class_features = features[
        y == class_id
    ]

    plt.scatter(
        class_features[:, 6],
        class_features[:, 2],
        color=colors[class_id],
        label=labels[class_id],
        s=60,
        alpha=0.75
    )


plt.title(
    "Feature Distribution of Signal Classes",
    fontsize=20,
    fontweight="bold"
)

plt.xlabel(
    "Dominant Frequency (Hz)",
    fontsize=13
)

plt.ylabel(
    "RMS Amplitude",
    fontsize=13
)

plt.grid(
    True,
    linestyle="--",
    alpha=0.4
)

plt.legend(
    fontsize=11
)

plt.tight_layout()

plt.show()
