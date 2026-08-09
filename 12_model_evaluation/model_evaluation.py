import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import square, sawtooth
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Generate signals
np.random.seed(42)

fs = 200
samples = 100
signal_length = 200

t = np.arange(signal_length) / fs

signals = []
labels = []

for i in range(samples):

    # Sine wave
    f = np.random.uniform(3, 8)
    sine = np.sin(2 * np.pi * f * t)

    signals.append(sine)
    labels.append(0)

    # Square wave
    f = np.random.uniform(3, 8)
    square_wave = square(2 * np.pi * f * t)

    signals.append(square_wave)
    labels.append(1)

    # Sawtooth wave
    f = np.random.uniform(3, 8)
    saw = sawtooth(2 * np.pi * f * t)

    signals.append(saw)
    labels.append(2)

    # Noisy sine wave
    f = np.random.uniform(3, 8)
    clean = np.sin(2 * np.pi * f * t)
    noise = 0.4 * np.random.randn(signal_length)

    noisy = clean + noise

    signals.append(noisy)
    labels.append(3)

signals = np.array(signals)
labels = np.array(labels)


# Extract features from each signal
def get_features(signal):

    mean = np.mean(signal)
    std = np.std(signal)
    rms = np.sqrt(np.mean(signal ** 2))
    energy = np.sum(signal ** 2)
    peak = np.max(np.abs(signal))

    zero_crossings = np.sum(
        np.diff(np.sign(signal)) != 0
    )

    zero_crossing_rate = zero_crossings / len(signal)

    # Find dominant frequency
    fft = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(len(signal), 1 / fs)

    magnitude = np.abs(fft)
    magnitude[0] = 0

    dominant_frequency = freq[np.argmax(magnitude)]

    return [
        mean,
        std,
        rms,
        energy,
        peak,
        zero_crossing_rate,
        dominant_frequency
    ]


# Create feature dataset
X = []

for signal in signals:
    X.append(get_features(signal))

X = np.array(X)
y = labels


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Scale the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Create and train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# Make predictions
predictions = model.predict(X_test)


# Calculate evaluation scores
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test, predictions, average="weighted"
)
recall = recall_score(
    y_test, predictions, average="weighted"
)
f1 = f1_score(
    y_test, predictions, average="weighted"
)

print("Model Evaluation")
print("------------------------")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")


# Confusion matrix
cm = confusion_matrix(y_test, predictions)

class_names = [
    "Sine",
    "Square",
    "Sawtooth",
    "Noisy Sine"
]


# Plot results
fig, axes = plt.subplots(1, 3, figsize=(17, 6))

# Confusion matrix
axes[0].imshow(cm, cmap="Blues")

axes[0].set_title(
    "Confusion Matrix",
    fontsize=15,
    fontweight="bold"
)

axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

axes[0].set_xticks(range(4))
axes[0].set_yticks(range(4))

axes[0].set_xticklabels(class_names, rotation=45)
axes[0].set_yticklabels(class_names)

for i in range(4):
    for j in range(4):
        axes[0].text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=12
        )


# Model scores
scores = [
    accuracy * 100,
    precision * 100,
    recall * 100,
    f1 * 100
]

score_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

axes[1].bar(
    score_names,
    scores,
    color=[
        "royalblue",
        "orange",
        "mediumseagreen",
        "crimson"
    ]
)

axes[1].set_ylim(0, 105)
axes[1].set_ylabel("Score (%)")

axes[1].set_title(
    "Model Performance",
    fontsize=15,
    fontweight="bold"
)

axes[1].grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

for i, score in enumerate(scores):
    axes[1].text(
        i,
        score + 2,
        f"{score:.1f}%",
        ha="center",
        fontweight="bold"
    )


# Feature importance
feature_names = [
    "Mean",
    "Std",
    "RMS",
    "Energy",
    "Peak",
    "Zero Crossing",
    "Dominant Frequency"
]

importance = model.feature_importances_

axes[2].barh(
    feature_names,
    importance,
    color="mediumseagreen"
)

axes[2].set_title(
    "Feature Importance",
    fontsize=15,
    fontweight="bold"
)

axes[2].set_xlabel("Importance")

axes[2].grid(
    axis="x",
    linestyle="--",
    alpha=0.4
)


plt.suptitle(
    "Random Forest - Signal Classification",
    fontsize=19,
    fontweight="bold"
)

plt.tight_layout()
plt.show()
