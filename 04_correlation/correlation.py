import numpy as np
import matplotlib.pyplot as plt

# Two discrete-time signals
x = np.array([1, 2, 3, 2, 1])
y = np.array([1, 1, 2, 1, 0])

# Cross-correlation
correlation = np.correlate(x, y, mode='full')

# Lag values
lags = np.arange(-(len(y) - 1), len(x))

# Display values
print("Signal x[n] =", x)
print("Signal y[n] =", y)
print("Cross-correlation =", correlation)

# Create figure
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Signal x[n]
axes[0].stem(
    np.arange(len(x)),
    x,
    linefmt='b-',
    markerfmt='bo',
    basefmt='k-'
)

axes[0].set_title("Signal x[n]", fontsize=16, fontweight='bold')
axes[0].set_xlabel("Sample Index (n)")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True, linestyle='--', alpha=0.6)

# Signal y[n]
axes[1].stem(
    np.arange(len(y)),
    y,
    linefmt='g-',
    markerfmt='go',
    basefmt='k-'
)

axes[1].set_title("Signal y[n]", fontsize=16, fontweight='bold')
axes[1].set_xlabel("Sample Index (n)")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True, linestyle='--', alpha=0.6)

# Cross-correlation
axes[2].stem(
    lags,
    correlation,
    linefmt='r-',
    markerfmt='ro',
    basefmt='k-'
)

axes[2].set_title(
    "Cross-Correlation of x[n] and y[n]",
    fontsize=16,
    fontweight='bold'
)

axes[2].set_xlabel("Lag")
axes[2].set_ylabel("Correlation")
axes[2].grid(True, linestyle='--', alpha=0.6)

plt.subplots_adjust(hspace=0.45)
plt.show()
