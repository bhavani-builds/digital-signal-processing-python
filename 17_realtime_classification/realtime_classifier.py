import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square, sawtooth

from sklearn.ensemble import RandomForestClassifier


np.random.seed(42)

# Signal settings
fs = 200
window_size = 200

t = np.arange(window_size) / fs


# ------------------------------------------
# Feature extraction
# ------------------------------------------

def get_features(signal):

    mean = np.mean(signal)

    std = np.std(signal)

    rms = np.sqrt(
        np.mean(signal ** 2)
    )

    energy = np.sum(
        signal ** 2
    )

    peak = np.max(
        np.abs(signal)
    )

    zero_crossings = np.sum(
        np.diff(np.sign(signal)) != 0
    )

    return [
        mean,
        std,
        rms,
        energy,
        peak,
        zero_crossings
    ]


# ------------------------------------------
# Create training data
# ------------------------------------------

train_features = []
train_labels = []

for i in range(300):

    frequency = np.random.uniform(3, 8)

    # Sine
    signal = np.sin(
        2 * np.pi * frequency * t
    )

    train_features.append(
        get_features(signal)
    )

    train_labels.append(0)

    # Square
    signal = square(
        2 * np.pi * frequency * t
    )

    train_features.append(
        get_features(signal)
    )

    train_labels.append(1)

    # Sawtooth
    signal = sawtooth(
        2 * np.pi * frequency * t
    )

    train_features.append(
        get_features(signal)
    )

    train_labels.append(2)

    # Noisy sine
    clean = np.sin(
        2 * np.pi * frequency * t
    )

    noise = 0.4 * np.random.randn(
        window_size
    )

    signal = clean + noise

    train_features.append(
        get_features(signal)
    )

    train_labels.append(3)


train_features = np.array(
    train_features
)

train_labels = np.array(
    train_labels
)


# ------------------------------------------
# Train model
# ------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    train_features,
    train_labels
)


print("Model trained successfully.")
print("Starting real-time classification...\n")


# ------------------------------------------
# Signal names
# ------------------------------------------

signal_names = [
    "Sine Wave",
    "Square Wave",
    "Sawtooth Wave",
    "Noisy Sine Wave"
]


# ------------------------------------------
# Create plot
# ------------------------------------------

plt.ion()

fig, ax = plt.subplots(
    figsize=(11, 6)
)


# ------------------------------------------
# Real-time simulation
# ------------------------------------------

for i in range(30):

    # Randomly select a signal
    signal_type = np.random.randint(0, 4)

    frequency = np.random.uniform(
        3,
        8
    )

    if signal_type == 0:

        signal = np.sin(
            2 * np.pi * frequency * t
        )

    elif signal_type == 1:

        signal = square(
            2 * np.pi * frequency * t
        )

    elif signal_type == 2:

        signal = sawtooth(
            2 * np.pi * frequency * t
        )

    else:

        clean = np.sin(
            2 * np.pi * frequency * t
        )

        noise = 0.4 * np.random.randn(
            window_size
        )

        signal = clean + noise


    # Extract features

    features = get_features(
        signal
    )

    features = np.array(
        features
    ).reshape(1, -1)


    # Predict signal

    prediction = model.predict(
        features
    )[0]

    probability = model.predict_proba(
        features
    )[0]

    confidence = (
        probability[prediction] * 100
    )


    # --------------------------------------
    # Update graph
    # --------------------------------------

    ax.clear()

    ax.plot(
        t,
        signal,
        linewidth=2.2
    )

    ax.set_title(
        "Real-Time Signal Classification",
        fontsize=19,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Time (seconds)"
    )

    ax.set_ylabel(
        "Amplitude"
    )

    ax.set_ylim(
        -2,
        2
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.4
    )


    # Prediction information

    ax.text(
        0.02,
        0.92,
        f"Detected: {signal_names[prediction]}",
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold"
    )

    ax.text(
        0.02,
        0.86,
        f"Confidence: {confidence:.1f}%",
        transform=ax.transAxes,
        fontsize=13
    )

    ax.text(
        0.02,
        0.80,
        f"Frequency: {frequency:.2f} Hz",
        transform=ax.transAxes,
        fontsize=13
    )


    plt.pause(0.5)


plt.ioff()

plt.show()
