import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate

# prepare some coordinates
dimsize = 8
x, y, z = np.indices((dimsize, dimsize, dimsize))

# draw cuboids in the top left and bottom right corners, and a link between them
cube1 = (x < 3) & (y < 3) & (z < 3)
cube2 = (x >= 5) & (y >= 5) & (z >= 5)
link = abs(x - y) + abs(y - z) + abs(z - x) <= 2

# combine the objects into a single boolean array
voxels = cube1 | cube2 | link

# set the colors of each object
colors = np.empty(voxels.shape, dtype=object)
colors[link] = 'red'
colors[cube1] = 'blue'
colors[cube2] = 'green'

# and plot everything
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(voxels, facecolors=colors, edgecolor='k')

plt.show()

import cv2

xxx = np.reshape(x, [dimsize*dimsize*dimsize, 1]).astype('float32')
yyy = np.reshape(y, [dimsize*dimsize*dimsize, 1]).astype('float32')
zzz = np.reshape(z, [dimsize*dimsize*dimsize, 1]).astype('float32')
vals = np.reshape(voxels.astype('float32'), [dimsize*dimsize*dimsize, 1])

x_i, y_i = np.indices((dimsize, dimsize))
xx = np.reshape(x_i, [dimsize*dimsize, 1]).astype('float32')
yy = np.reshape(y_i, [dimsize*dimsize, 1]).astype('float32')
imagePoints = np.concatenate((xx, yy), 1)
objectPoints = np.concatenate((xxx, yyy, zzz), 1)

rvec = np.array([0,0,0], np.float) # rotation vector
tvec = np.array([0,0,0], np.float) # translation vector
fx = fy = 1
cx = cy = 0
cameraMatrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float)
#result = cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, None, imagePoints)
result = cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, None)
result = np.squeeze(result[0])


f = interpolate.interp2d(result[:,0], result[:,1], np.squeeze(vals[:,0]), kind='cubic')

newres = f(np.squeeze(xx),np.squeeze(yy))
plt.figure()
plt.imshow(newres)
#voxels_g = voxels.astype('float32')
#
#voxel_z = np.sum(voxels_g,axis=1)
#
#plt.figure()
#plt.imshow(voxel_z[::-1,:])
