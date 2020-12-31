import numpy
#from numpy import * # BJD added 20.11.2020
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Set up grid and test data
#nx, ny = 256, 1024
nx, ny = 240, 426
#(426,240)
x = range(nx)
y = range(ny)

#data = numpy.random.random((nx, ny))
data = numpy.loadtxt("/home/brendan/software/tf2-model-g/arrays/array10/Y.txt")

hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')

X, Y = numpy.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
#ha.plot_surface(X, Y, data)
ha.plot_surface(X, Y, data, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
#ha.plot_surface(X.T, Y.T, data)

plt.show()