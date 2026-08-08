import numpy as np
import matplotlib.pyplot as plt

# Discrete-time index
n = np.arange(0, 20)

# Original signal
x = np.sin(2 * np.pi * 0.1 * n)

# Time shifting
x_shifted = np.sin(2 * np.pi * 0.1 * (n - 3))

# Time reversal
x_reversed = np.sin(2 * np.pi * 0.1 * (-n))

# Amplitude scaling
x_scaled = 2 * x

# Second signal
x2 = np.cos(2 * np.pi * 0.1 * n)

# Signal addition
x_added = x + x2

# Plot signals
plt.figure(figsize=(12, 10))

plt.subplot(5, 1, 1)
plt.stem(n, x)
plt.title("Original Signal")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(5, 1, 2)
plt.stem(n, x_shifted)
plt.title("Time Shifted Signal")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(5, 1, 3)
plt.stem(n, x_reversed)
plt.title("Time Reversed Signal")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(5, 1, 4)
plt.stem(n, x_scaled)
plt.title("Amplitude Scaled Signal")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(5, 1, 5)
plt.stem(n, x_added)
plt.title("Added Signal")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()
