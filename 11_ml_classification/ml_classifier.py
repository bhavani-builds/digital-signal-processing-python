import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import square, sawtooth

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


np.random.seed(42)

num_samples = 100
signal_length = 200
fs = 200

t = np.arange(signal_length) / fs

X_signals = []
y = []


# ==========================================
# 2. Generate Signal Dataset
# ==========================================

for i in range(num_samples):

    # --------------------------------------
    # Class 0: Sine Wave
    # --------------------------------------

    frequency = np.random.uniform(3, 8)

    sine_signal = np.sin(
        2 * np.pi * frequency * t
    )

    X_signals.append(sine_signal)
    y.append(0)

    # --------------------------------------
    # Class 1: Square Wave
    # --------------------------------------

    frequency = np.random.uniform(3, 8)

    square_signal = square(
        2 * np.pi * frequency * t
    )

    X_signals.append(square_signal)
    y.append(1)

    # --------------------------------------
    # Class 2: Sawtooth Wave
    # --------------------------------------

    frequency = np.random.uniform(3, 8)

    sawtooth_signal = sawtooth(
        2 * np.pi * frequency * t
    )

    X_signals.append(sawtooth_signal)
    y.append(2)

    # --------------------------------------
    # Class 3: Noisy Sine Wave
    # --------------------------------------

    frequency = np.random.uniform(3, 8)

    clean_signal = np.sin(
        2 * np.pi * frequency * t
    )

    noise = 0.4 * np.random.randn(
        signal_length
    )

    noisy_signal = clean_signal + noise

    X_signals.append(noisy_signal)
    y.append(3)


X_signals = np.array(X_signals)
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
    for signal in X_signals
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
# 6. Create ML Model
# ==========================================

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)


# ==========================================
# 7. Train Model
# ==========================================

model.fit(
    X_train_scaled,
    y_train
)


# ==========================================
# 8. Make Predictions
# ==========================================

y_pred = model.predict(
    X_test_scaled
)


# ==========================================
# 9. Calculate Accuracy
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("=" * 60)
print("MACHINE LEARNING SIGNAL CLASSIFICATION")
print("=" * 60)

print("\nTraining samples :", len(X_train))
print("Testing samples  :", len(X_test))

print(
    f"\nModel Accuracy: {accuracy * 100:.2f}%"
)


# ==========================================
# 10. Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

class_names = [
    "Sine",
    "Square",
    "Sawtooth",
    "Noisy Sine"
]


# ==========================================
# 11. Visualize Results
# ==========================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(15, 6)
)


# ------------------------------------------
# Confusion Matrix
# ------------------------------------------

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=axes[0],
    cmap="Blues",
    colorbar=False
)

axes[0].set_title(
    "Signal Classification\nConfusion Matrix",
    fontsize=17,
    fontweight="bold"
)

axes[0].set_xlabel(
    "Predicted Class"
)

axes[0].set_ylabel(
    "Actual Class"
)


# ------------------------------------------
# Model Accuracy
# ------------------------------------------

axes[1].bar(
    ["Accuracy"],
    [accuracy * 100],
    color="mediumseagreen",
    width=0.5
)

axes[1].set_ylim(
    0,
    100
)

axes[1].set_ylabel(
    "Accuracy (%)"
)

axes[1].set_title(
    "Random Forest Classification Accuracy",
    fontsize=17,
    fontweight="bold"
)

axes[1].text(
    0,
    accuracy * 100 + 2,
    f"{accuracy * 100:.2f}%",
    ha="center",
    fontsize=16,
    fontweight="bold"
)

axes[1].grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)


# ==========================================
# Overall Title
# ==========================================

fig.suptitle(
    "Machine Learning Based Signal Classification",
    fontsize=21,
    fontweight="bold"
)

plt.tight_layout(
    rect=[0, 0, 1, 0.94]
)

plt.show()
