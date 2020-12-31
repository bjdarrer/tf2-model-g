import numpy as np; import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D

# Generate data
x = np.linspace(-2,2, 500)
y = np.linspace(-2,2, 40)
X,Y = np.meshgrid(x,y)
Z = np.sin(X**2+Y**2)
# Generate waterfall plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
waterfall_plot(fig,ax,X,Y,Z) 
ax.set_xlabel('X') ; ax.set_xlim3d(-2,2)
ax.set_ylabel('Y') ; ax.set_ylim3d(-2,2)
ax.set_zlabel('Z') ; ax.set_zlim3d(-1,1)
