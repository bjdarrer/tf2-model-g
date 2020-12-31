from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

def triangle2(length, amplitude):
    section = length // 4
    x = np.linspace(0, amplitude, section+1)
    mx = -x
    return np.r_[x, x[-2::-1], mx[1:], mx[-2:0:-1]]

plt.plot(triangle2(2,3))
plt.show()

"""
x = np.linspace(-25, 25, 500)
triangle = signal.sawtooth(40 * np.pi * 1/800 * x + 2, 0.5) - 1
plt.plot(x, triangle)
plt.show()
"""

 
#plt.plot(x, triangle2(3, 3))

