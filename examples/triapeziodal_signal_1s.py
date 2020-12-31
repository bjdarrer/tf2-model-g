from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


#def trapzoid_signal(x, width=2., slope=1., amp=1., offs=0):
def trapzoid_signal(x, width=2., slope=1., amp=10., offs=1):
    #a = 10 * slope*width*signal.sawtooth(2 * np.pi * 1/10* x/width, width=0.5)/4.
    #a = slope * width * signal.sawtooth(2 * np.pi * 1/10 * x/width, width=0.5)/4.
    a = slope * width * signal.sawtooth(2 * np.pi * 1/10 * x/width, width=0.5)/4.
    a[a>amp/2.] = amp/2.
    a[a<-amp/2.] = -amp/2.
    return a + amp/2. + offs

x = np.linspace(100, -100, 1000)
#plt.plot(t,trapzoid_signal(x, width=2, slope=2, amp=1.), label="width=2, slope=2, amp=1")
#plt.plot(x,trapzoid_signal(x, width=4, slope=1, amp=0.6), label="width=4, slope=1, amp=0.6")
plt.plot(x,trapzoid_signal(x, width=10, slope=6, amp=20), label="width=10, slope=5, amp=20")

plt.legend( loc=(0.25,1.015))
plt.show()
