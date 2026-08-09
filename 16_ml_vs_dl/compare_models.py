import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import square, sawtooth, spectrogram

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


np.random.seed(42)

fs = 200
signal_length = 200
samples = 100

t = np.arange(signal_length) / fs

signals = []
labels = []


# ------------------------------------------
# Generate signals
# ------------------------------------------

for i in range(samples):

    f = np.random.uniform(3, 8)

    sine = np.sin(2 * np.pi * f * t)

    square_signal = square(
        2 * np.pi * f * t
    )

    saw_signal = sawtooth(
        2 * np.pi * f * t
    )

    noise = 0.4 * np.random.randn(
        signal_length
    )

    noisy_sine = sine + noise

    signals.extend([
        sine,
        square_signal,
        saw_signal,
        noisy_sine
    ])

    labels.extend([
        0,
        1,
        2,
        3
    ])


signals = np.array(signals)
labels = np.array(labels)


# ------------------------------------------
# ML: Extract simple features
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


X_features = np.array([
    get_features(signal)
    for signal in signals
])


# ------------------------------------------
# Train Random Forest
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_prediction = rf_model.predict(
    X_test
)

rf_accuracy = accuracy_score(
    y_test,
    rf_prediction
)

print(
    f"Random Forest Accuracy: "
    f"{rf_accuracy * 100:.2f}%"
)


# ------------------------------------------
# Create spectrogram images for CNN
# ------------------------------------------

images = []

for signal in signals:

    freq, time, power = spectrogram(
        signal,
        fs=fs,
        nperseg=64,
        noverlap=48
    )

    power = 10 * np.log10(
        power + 1e-10
    )

    # Normalize
    power = (
        power - power.min()
    ) / (
        power.max() - power.min() + 1e-10
    )

    images.append(power)


X_images = np.array(images)

X_images = X_images[..., np.newaxis]


# ------------------------------------------
# Train-test split for CNN
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_images,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

y_train = to_categorical(
    y_train,
    4
)

y_test = to_categorical(
    y_test,
    4
)


# ------------------------------------------
# CNN model
# ------------------------------------------

cnn_model = Sequential([

    Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same",
        input_shape=X_train.shape[1:]
    ),

    MaxPooling2D(
        (2, 2)
    ),

    Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    MaxPooling2D(
        (2, 2)
    ),

    Flatten(),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        4,
        activation="softmax"
    )
])


cnn_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ------------------------------------------
# Train CNN
# ------------------------------------------

cnn_model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


# ------------------------------------------
# Test CNN
# ------------------------------------------

cnn_loss, cnn_accuracy = cnn_model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(
    f"CNN Accuracy: "
    f"{cnn_accuracy * 100:.2f}%"
)


# ------------------------------------------
# Compare the models
# ------------------------------------------

models = [
    "Random Forest",
    "CNN"
]

accuracies = [
    rf_accuracy * 100,
    cnn_accuracy * 100
]


plt.figure(
    figsize=(10, 6)
)

bars = plt.bar(
    models,
    accuracies,
    color=[
        "royalblue",
        "crimson"
    ],
    width=0.55
)

plt.ylabel(
    "Accuracy (%)",
    fontsize=13
)

plt.xlabel(
    "Model",
    fontsize=13
)

plt.title(
    "Machine Learning vs Deep Learning",
    fontsize=19,
    fontweight="bold"
)

plt.ylim(
    0,
    105
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)


# Add accuracy values above bars

for bar, value in zip(
    bars,
    accuracies
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 2,
        f"{value:.2f}%",
        ha="center",
        fontsize=13,
        fontweight="bold"
    )


plt.tight_layout()

plt.savefig(
    "Figure_16.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------
# Final comparison
# ------------------------------------------

print("\n-----------------------------")
print("FINAL MODEL COMPARISON")
print("-----------------------------")

print(
    f"Random Forest : {rf_accuracy * 100:.2f}%"
)

print(
    f"CNN           : {cnn_accuracy * 100:.2f}%"
)
