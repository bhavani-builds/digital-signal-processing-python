import numpy as np
import matplotlib.pyplot as plt

# Signal frequency
f = 5

# Continuous-time signal
t = np.linspace(0, 1, 1000)
x = np.sin(2 * np.pi * f * t)

# Sampling frequencies
fs_good = 20
fs_low = 7

# Sampled signals
t_good = np.arange(0, 1, 1 / fs_good)
x_good = np.sin(2 * np.pi * f * t_good)

t_low = np.arange(0, 1, 1 / fs_low)
x_low = np.sin(2 * np.pi * f * t_low)

# Plot
plt.figure(figsize=(12, 10))

# Original continuous signal
plt.subplot(3, 1, 1)
plt.plot(t, x, linewidth=2)
plt.title("Original Continuous-Time Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True, linestyle="--", alpha=0.6)

# Properly sampled signal
plt.subplot(3, 1, 2)
plt.plot(t, x, linewidth=2)
plt.stem(t_good, x_good)
plt.title("Sampling at 20 Hz")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True, linestyle="--", alpha=0.6)

# Undersampled signal
plt.subplot(3, 1, 3)
plt.plot(t, x, linewidth=2)
plt.stem(t_low, x_low)
plt.title("Undersampling at 7 Hz")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
