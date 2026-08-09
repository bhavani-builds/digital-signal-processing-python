import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import square, sawtooth, spectrogram

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.utils import to_categorical


np.random.seed(42)

fs = 200
signal_length = 200
number_of_signals = 100

t = np.arange(signal_length) / fs

images = []
labels = []


# Create spectrogram dataset

for i in range(number_of_signals):

    frequency = np.random.uniform(3, 8)

    sine = np.sin(
        2 * np.pi * frequency * t
    )

    square_signal = square(
        2 * np.pi * frequency * t
    )

    saw_signal = sawtooth(
        2 * np.pi * frequency * t
    )

    noise = 0.4 * np.random.randn(
        signal_length
    )

    noisy_signal = sine + noise

    signals = [
        sine,
        square_signal,
        saw_signal,
        noisy_signal
    ]

    for class_number, signal in enumerate(signals):

        freq, time, power = spectrogram(
            signal,
            fs=fs,
            nperseg=64,
            noverlap=48
        )

        power = 10 * np.log10(
            power + 1e-10
        )

        # Normalize the spectrogram
        power = (
            power - power.min()
        ) / (
            power.max() - power.min() + 1e-10
        )

        images.append(power)
        labels.append(class_number)


# Convert to NumPy arrays

X = np.array(images)
y = np.array(labels)

print("Original image shape:", X.shape)


# Add channel dimension

X = X[..., np.newaxis]

print("CNN input shape:", X.shape)


# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


y_train = to_categorical(
    y_train,
    num_classes=4
)

y_test = to_categorical(
    y_test,
    num_classes=4
)


# Get actual image dimensions

image_height = X.shape[1]
image_width = X.shape[2]


# CNN model

model = Sequential([

    Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same",
        input_shape=(
            image_height,
            image_width,
            1
        )
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

    # Pool only after checking the dimensions
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


# Compile model

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


print("\nTraining CNN...\n")


# Train model

history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


# Test model

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)


print("\n------------------------------")
print("CNN MODEL RESULTS")
print("------------------------------")

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)


# Plot training results

fig, axes = plt.subplots(
    1,
    2,
    figsize=(15, 6)
)


# Accuracy

axes[0].plot(
    history.history["accuracy"],
    color="royalblue",
    linewidth=2.5,
    marker="o",
    label="Training Accuracy"
)

axes[0].plot(
    history.history["val_accuracy"],
    color="darkorange",
    linewidth=2.5,
    marker="o",
    label="Validation Accuracy"
)

axes[0].set_title(
    "CNN Training Accuracy",
    fontsize=17,
    fontweight="bold"
)

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")

axes[0].grid(
    True,
    linestyle="--",
    alpha=0.4
)

axes[0].legend()


# Loss

axes[1].plot(
    history.history["loss"],
    color="crimson",
    linewidth=2.5,
    marker="o",
    label="Training Loss"
)

axes[1].plot(
    history.history["val_loss"],
    color="mediumseagreen",
    linewidth=2.5,
    marker="o",
    label="Validation Loss"
)

axes[1].set_title(
    "CNN Training Loss",
    fontsize=17,
    fontweight="bold"
)

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")

axes[1].grid(
    True,
    linestyle="--",
    alpha=0.4
)

axes[1].legend()


fig.suptitle(
    "Deep Learning for Signal Classification using CNN",
    fontsize=20,
    fontweight="bold"
)

plt.tight_layout(
    rect=[0, 0, 1, 0.94]
)

plt.savefig(
    "Figure_14.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
