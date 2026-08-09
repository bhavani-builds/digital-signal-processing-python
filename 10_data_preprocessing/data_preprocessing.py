import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import square, sawtooth
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
# 2. Generate Signal Dataset
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
# 3. Feature Extraction
# ==========================================

def extract_features(signal):

    mean_value = np.mean(signal)

    std_value = np.std(signal)

    rms_value = np.sqrt(
        np.mean(signal ** 2)
    )

    energy_value = np.sum(
        signal ** 2
    )

    peak_value = np.max(
        np.abs(signal)
    )

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


features = np.array([
    extract_features(signal)
    for signal in X
])


# ==========================================
# 4. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    features,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 5. Feature Scaling
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ==========================================
# 6. Display Dataset Information
# ==========================================

print("=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

print("\nOriginal Feature Dataset:")
print("Shape:", features.shape)

print("\nTraining Data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting Data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nAfter Scaling:")
print("X_train_scaled:", X_train_scaled.shape)
print("X_test_scaled :", X_test_scaled.shape)


# ==========================================
# 7. Visualize Before and After Scaling
# ==========================================

feature_names = [
    "Mean",
    "Std",
    "RMS",
    "Energy",
    "Peak",
    "Zero Crossing",
    "Dominant Frequency"
]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(15, 6)
)


# ------------------------------------------
# Before Scaling
# ------------------------------------------

axes[0].boxplot(
    X_train,
    patch_artist=True
)

axes[0].set_title(
    "Features Before Scaling",
    fontsize=17,
    fontweight="bold"
)

axes[0].set_xticks(
    range(1, len(feature_names) + 1)
)

axes[0].set_xticklabels(
    feature_names,
    rotation=45,
    ha="right"
)

axes[0].set_ylabel(
    "Feature Value"
)

axes[0].grid(
    True,
    linestyle="--",
    alpha=0.4
)


# ------------------------------------------
# After Scaling
# ------------------------------------------

axes[1].boxplot(
    X_train_scaled,
    patch_artist=True
)

axes[1].set_title(
    "Features After Standard Scaling",
    fontsize=17,
    fontweight="bold"
)

axes[1].set_xticks(
    range(1, len(feature_names) + 1)
)

axes[1].set_xticklabels(
    feature_names,
    rotation=45,
    ha="right"
)

axes[1].set_ylabel(
    "Standardized Value"
)

axes[1].grid(
    True,
    linestyle="--",
    alpha=0.4
)


# ==========================================
# Overall Title
# ==========================================

fig.suptitle(
    "Machine Learning Data Preprocessing",
    fontsize=21,
    fontweight="bold"
)

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)

plt.show()
