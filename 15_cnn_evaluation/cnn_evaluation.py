import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import square, sawtooth, spectrogram

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


np.random.seed(42)

fs = 200
signal_length = 200
samples = 100

t = np.arange(signal_length) / fs

images = []
labels = []


# Generate signals

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

    signals = [
        sine,
        square_signal,
        saw_signal,
        noisy_sine
    ]

    for label, signal in enumerate(signals):

        freq, time, power = spectrogram(
            signal,
            fs=fs,
            nperseg=64,
            noverlap=48
        )

        power = 10 * np.log10(
            power + 1e-10
        )

        power = (
            power - power.min()
        ) / (
            power.max() - power.min() + 1e-10
        )

        images.append(power)
        labels.append(label)


# Convert data

X = np.array(images)
y = np.array(labels)

X = X[..., np.newaxis]

print("Dataset:", X.shape)


# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

y_train = to_categorical(
    y_train,
    4
)

y_test = to_categorical(
    y_test,
    4
)


# Build CNN

model = Sequential([

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


# Compile

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# Train

model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


# Predict test data

predicted = model.predict(
    X_test,
    verbose=0
)

predicted_classes = np.argmax(
    predicted,
    axis=1
)

actual_classes = np.argmax(
    y_test,
    axis=1
)


# Confusion matrix

cm = confusion_matrix(
    actual_classes,
    predicted_classes
)

class_names = [
    "Sine",
    "Square",
    "Sawtooth",
    "Noisy Sine"
]


# Display confusion matrix

plt.figure(
    figsize=(9, 7)
)

plt.imshow(
    cm,
    cmap="Blues"
)

plt.title(
    "CNN Signal Classification",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.xticks(
    range(4),
    class_names,
    rotation=30
)

plt.yticks(
    range(4),
    class_names
)


# Show numbers inside matrix

for i in range(4):

    for j in range(4):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold"
        )


plt.colorbar(
    label="Number of Samples"
)

plt.tight_layout()

plt.savefig(
    "Figure_15.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Show some predictions

print("\nPrediction Results")
print("-------------------------")

for i in range(10):

    print(
        f"Sample {i + 1}: "
        f"Actual = {class_names[actual_classes[i]]}, "
        f"Predicted = {class_names[predicted_classes[i]]}"
    )
