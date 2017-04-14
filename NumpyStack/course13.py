import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 100, 10000)
y = np.sin(x) + np.sin(3*x) + np.sin(5*x)

plt.plot(y)
plt.show()

Y = np.fft.fft(y)
plt.plot(np.abs(Y))
plt.show()

2*np.pi*16/100
2*np.pi*48/100
2*np.pi*80/100
