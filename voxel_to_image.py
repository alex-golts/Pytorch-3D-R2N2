import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from scipy import ndimage
# prepare some coordinates
dimsize = 32
x, y, z = np.indices((dimsize, dimsize, dimsize))

cube1 = (x>5) & (x<15) & (y>5) & (y<15) & (z>5) & (z<15)
cube2 = (x>20) & (x<25) & (y>20) & (y<25) & (z>20) & (z<25)

# combine the objects into a single boolean array
voxels = cube1 | cube2 
colors = np.zeros(voxels.shape + (3,))
colors[voxels, :] = 1
# and plot everything
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(voxels, facecolors=colors, edgecolor='k')
plt.show()

voxels_float = voxels.astype('float')

rot_angles = range(0,180,30)
fig = plt.figure()
cnt=1
for ang in rot_angles:
    voxels_rot = ndimage.interpolation.rotate(voxels_float, ang, axes=(1, 0), reshape=False, output=np.float32, order=0, mode='constant', cval=0.0, prefilter=False)
    voxels_rot = voxels_rot>0
    colors_rot = np.zeros(voxels_rot.shape + (3,))
    colors_rot[voxels_rot,:] = 1
    ax = fig.add_subplot(2, 3, cnt, projection='3d')
    ax.set_title(str(ang))
    ax.voxels(voxels_rot, facecolors=colors_rot, edgecolor='k')
    cnt+=1
plt.show()

# now cast the rotations in 2d: