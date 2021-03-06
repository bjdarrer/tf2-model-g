from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


def trapzoid_signal(x, width=2., slope=1., amp=1., offs=0):
    a = slope*width*signal.sawtooth(2 * np.pi * x/width, width=0.5)/4.
    a[a>amp/2.] = amp/2.
    a[a<-amp/2.] = -amp/2.
    return a + amp/2. + offs

x = np.linspace(0, 6, 501)
#plt.plot(t,trapzoid_signal(x, width=2, slope=2, amp=1.), label="width=2, slope=2, amp=1")
plt.plot(x,trapzoid_signal(x, width=4, slope=1, amp=0.6), label="width=4, slope=1, amp=0.6")

plt.legend( loc=(0.25,1.015))
plt.show()
