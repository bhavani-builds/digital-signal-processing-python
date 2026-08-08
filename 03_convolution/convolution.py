import numpy as np
import matplotlib.pyplot as plt

# Input signal
x = np.array([1, 2, 1])

# Impulse response
h = np.array([1, 1])

# Linear convolution
y = np.convolve(x, h)

# Sample indices
n_x = np.arange(len(x))
n_h = np.arange(len(h))
n_y = np.arange(len(y))

# Display result
print("Input signal x[n]:", x)
print("Impulse response h[n]:", h)
print("Convolution output y[n]:", y)

# Plot signals
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.stem(n_x, x)
plt.title("Input Signal x[n]")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(3, 1, 2)
plt.stem(n_h, h)
plt.title("Impulse Response h[n]")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(3, 1, 3)
plt.stem(n_y, y)
plt.title("Linear Convolution y[n]")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()
