import numpy as np
#from numpy import * # BJD added 20.11.2020
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

c1 = 0
c1 = c1 + 1
# Set up grid and test data
#nx, ny = 256, 1024
nx, ny = 240, 426
#(426,240)
x = range(nx)
y = range(ny)

#data = numpy.random.random((nx, ny))
data = np.loadtxt("/home/brendan/software/tf2-model-g/arrays/array10/Y.txt")

hf = plt.figure(figsize=(10,10))
ha = hf.add_subplot(111, projection='3d')

X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
#ha.plot_surface(X, Y, data)
#ha.plot_surface(X, Y, data, rstride=1, cstride=1, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
#ha.plot_surface(X.T, Y.T, data)

surf = ha.plot_surface(X, Y, data, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ha.set_zlim(-4, 4)
hf.colorbar(surf, shrink=0.5, aspect=10)

plt.title('X, Y, G potential vs 2D space - time = ' + str(c1))
plt.xlabel("x spacial units")
plt.ylabel("y spacial units")
#plt.zlabel("X, Y, G pot. - concentration per unit vol")
#fig.savefig('test2.png')   # save the figure to file
#plt.legend(["X", "Y", "G"]) # BJD legend added 21.11.2020

#plt.figure(figsize=(15,10))
plt.show()

hf.savefig('/home/brendan/software/tf2-model-g/plots/3D_video18/3D_video_XYG_' + str(c1) + '.png')
plt.close(hf)    # close the figure window