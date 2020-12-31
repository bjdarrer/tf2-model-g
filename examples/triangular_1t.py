from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-25, 25, 500)
triangle = 10 * signal.sawtooth(40 * np.pi * 1/800 * x + 15, 0.5) - 10
plt.plot(x, triangle)
plt.show()



"""
def triangle2(length, amplitude):
    section = length // 4
    x = np.linspace(0, amplitude, section+1)
    mx = -x
    return np.r_[x, x[-2::-1], mx[1:], mx[-2:0:-1]]

plt.plot(triangle2(2,3))
plt.show()
"""

#plt.plot(x, triangle2(3, 3))

