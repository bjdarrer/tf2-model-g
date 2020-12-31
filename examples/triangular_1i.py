from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-5, 5, 500)
triangle = signal.sawtooth(2 * np.pi * 1/6 * x + 2, 0.9) - 1
plt.plot(x, triangle)
plt.show()

def triangle2(length, amplitude):
    section = length // 4
    x = np.linspace(0, amplitude, section+1)
    mx = -x
    return np.r_[x, x[-2::-1], mx[1:], mx[-2:0:-1]]
 
#plt.plot(x, triangle2(3, 3))

