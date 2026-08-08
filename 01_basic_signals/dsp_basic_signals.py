
import numpy as np
import matplotlib.pyplot as plt

# Discrete-time index
n = np.arange(-10, 11)

# Unit Impulse Signal
impulse = np.where(n == 0, 1, 0)

# Unit Step Signal
step = np.where(n >= 0, 1, 0)

# Ramp Signal
ramp = np.where(n >= 0, n, 0)

# Exponential Signal
exponential = np.where(n >= 0, 0.8 ** n, 0)

# Sine Signal
sine = np.sin(2 * np.pi * 0.1 * n)

# Cosine Signal
cosine = np.cos(2 * np.pi * 0.1 * n)

# Plot all signals
plt.figure(figsize=(12, 12))

plt.subplot(3, 2, 1)
plt.stem(n, impulse)
plt.title("Unit Impulse Signal")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(3, 2, 2)
plt.stem(n, step)
plt.title("Unit Step Signal")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(3, 2, 3)
plt.stem(n, ramp)
plt.title("Ramp Signal")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(3, 2, 4)
plt.stem(n, exponential)
plt.title("Exponential Signal")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(3, 2, 5)
plt.stem(n, sine)
plt.title("Sine Signal")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(3, 2, 6)
plt.stem(n, cosine)
plt.title("Cosine Signal")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()
