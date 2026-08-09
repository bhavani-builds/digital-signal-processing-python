import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square, sawtooth

# ==========================================
# 1. Dataset Parameters
# ==========================================

np.random.seed(42)

num_samples = 100
signal_length = 200

t = np.linspace(0, 1, signal_length)

X = []
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

    X.append(sine_signal)
    y.append(0)

    # --------------------------------------
    # Class 1: Square Wave
    # --------------------------------------

    frequency = np.random.uniform(3, 8)

    square_signal = square(
        2 * np.pi * frequency * t
    )

    X.append(square_signal)
    y.append(1)

    # --------------------------------------
    # Class 2: Sawtooth Wave
    # --------------------------------------

    frequency = np.random.uniform(3, 8)

    sawtooth_signal = sawtooth(
        2 * np.pi * frequency * t
    )

    X.append(sawtooth_signal)
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

    noisy_sine = clean_signal + noise

    X.append(noisy_sine)
    y.append(3)


# ==========================================
# 3. Convert to NumPy Arrays
# ==========================================

X = np.array(X)
y = np.array(y)

# ==========================================
# 4. Display Dataset Information
# ==========================================

print("=" * 50)
print("SIGNAL DATASET INFORMATION")
print("=" * 50)

print("Dataset shape :", X.shape)
print("Labels shape  :", y.shape)

print("\nNumber of signals in each class:")

print("Class 0 - Sine Wave       :", np.sum(y == 0))
print("Class 1 - Square Wave     :", np.sum(y == 1))
print("Class 2 - Sawtooth Wave   :", np.sum(y == 2))
print("Class 3 - Noisy Sine Wave :", np.sum(y == 3))

# ==========================================
# 5. Visualize Example Signals
# ==========================================

fig, axes = plt.subplots(
    4,
    1,
    figsize=(14, 11)
)

# ------------------------------------------
# Class 0 - Sine
# ------------------------------------------

axes[0].plot(
    t,
    X[y == 0][0],
    color="royalblue",
    linewidth=2.5,
    label="Sine Wave"
)

axes[0].set_title(
    "Class 0 — Sine Wave",
    fontsize=17,
    fontweight="bold"
)

axes[0].set_ylabel("Amplitude")
axes[0].grid(
    True,
    linestyle="--",
    alpha=0.4
)

axes[0].legend()

# ------------------------------------------
# Class 1 - Square
# ------------------------------------------

axes[1].plot(
    t,
    X[y == 1][0],
    color="darkorange",
    linewidth=2.5,
    label="Square Wave"
)

axes[1].set_title(
    "Class 1 — Square Wave",
    fontsize=17,
    fontweight="bold"
)

axes[1].set_ylabel("Amplitude")
axes[1].grid(
    True,
    linestyle="--",
    alpha=0.4
)

axes[1].legend()

# ------------------------------------------
# Class 2 - Sawtooth
# ------------------------------------------

axes[2].plot(
    t,
    X[y == 2][0],
    color="mediumseagreen",
    linewidth=2.5,
    label="Sawtooth Wave"
)

axes[2].set_title(
    "Class 2 — Sawtooth Wave",
    fontsize=17,
    fontweight="bold"
)

axes[2].set_ylabel("Amplitude")
axes[2].grid(
    True,
    linestyle="--",
    alpha=0.4
)

axes[2].legend()

# ------------------------------------------
# Class 3 - Noisy Sine
# ------------------------------------------

axes[3].plot(
    t,
    X[y == 3][0],
    color="crimson",
    linewidth=1.5,
    label="Noisy Sine Wave"
)

axes[3].set_title(
    "Class 3 — Noisy Sine Wave",
    fontsize=17,
    fontweight="bold"
)

axes[3].set_xlabel("Time (seconds)")
axes[3].set_ylabel("Amplitude")

axes[3].grid(
    True,
    linestyle="--",
    alpha=0.4
)

axes[3].legend()

# ==========================================
# Overall Figure
# ==========================================

fig.suptitle(
    "Signal Dataset Generation for Machine Learning",
    fontsize=21,
    fontweight="bold"
)

plt.tight_layout(
    rect=[0, 0, 1, 0.96]
)

plt.show()
