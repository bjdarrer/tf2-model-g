from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

#x = np.linspace(-40, 40, 500)
t = np.linspace(0, 6, 501)
#triangle = 10 * signal.sawtooth(2 * np.pi * 1/70 * x + 0.001, 0.5) + 10
trapzoid_signal = slope*width*signal.sawtooth(2*np.pi*t/width, width=0.5)/4.
plt.plot(x, trapzoid_signal)
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

