'''
Generate training cubes from random walk and power flier paths
'''
import numpy as np
from motion_cube import motion_cube, crop_tracks, gauss_kernel
from scipy import ndimage as ndi
from skimage.morphology import disk
from skimage.filters import gaussian
import os

save_dir = '/media/jkimmel/HDD0/myctophid/bin_disk25/'
rw_dir = os.path.join(save_dir, 'rw_cubes')
pf_dir = os.path.join(save_dir, 'pf_cubes')
fbm_dir = os.path.join(save_dir, 'fbm_cubes')
fold_compression = 2

print('Loading and compressing paths...')
# Load paths
rw_x = np.loadtxt('sim_data/rw_X_100k.csv', delimiter=',')
rw_y = np.loadtxt('sim_data/rw_Y_100k.csv', delimiter=',')
pf_x = np.loadtxt('sim_data/pf_X_100k_mu5.csv', delimiter=',')
pf_y = np.loadtxt('sim_data/pf_Y_100k_mu5.csv', delimiter=',')
fbm_x = np.loadtxt('sim_data/fbm_X_100k_mu5.csv', delimiter=',')
fbm_y = np.loadtxt('sim_data/fbm_Y_100k_mu5.csv', delimiter=',')

# Compress paths
rw_x = rw_x / fold_compression
rw_y = rw_y / fold_compression
pf_x = pf_x / fold_compression
pf_y = pf_y / fold_compression
fbm_x = fbm_x / fold_compression
fbm_y = fbm_y / fold_compression

# Crop tracks
print('Cropping tracks...')
rw_x, rw_y = crop_tracks(rw_x, rw_y, lbound=64, hbound=78)
pf_x, pf_y = crop_tracks(pf_x, pf_y, lbound=64, hbound=78)
fbm_x, fbm_y = crop_tracks(fbm_x, fbm_y, lbound=64, hbound=78)

# Class balance
print('Class balancing...')
min_class = min(rw_x.shape[0], pf_x.shape[0], fbm_x.shape[0])
min_class = 3500
rw_x, rw_y = rw_x[:min_class, :], rw_y[:min_class,:]
pf_x, pf_y = pf_x[:min_class, :], pf_y[:min_class,:]
fbm_x, fbm_y = fbm_x[:min_class, :], fbm_y[:min_class,:]
print('Min. class size : ', min_class)


# Set cube size
x_max, y_max = 156, 156
width = 25
sigma = 20
staticK = np.zeros([x_max*2, y_max*2])
staticK[x_max,y_max] = 1
se = disk(width)
staticK = ndi.binary_dilation(staticK, structure=se)
#gKern = gauss_kernel(5*sigma, sigma)
#staticK = np.pad(gKern, ((2*x_max-5*sigma)//2, (2*y_max-5*sigma)//2), mode='constant')
#staticK = staticK / staticK.max()

# Generate RW cubes and save
print('Generating RW cubes...')
rw_cubes = motion_cube(rw_x, rw_y, x_max, y_max, staticK=staticK, binary=True, save=rw_dir)
#np.savez('sim_data/rw_cubes', rw_cubes)

print('Generating PF cubes...')
# Generate PF cubes and save
pf_cubes = motion_cube(pf_x, pf_y, x_max, y_max, staticK=staticK, binary=True, save=pf_dir)
#np.savez('sim_data/pf_cubes', pf_cubes)

print('Generating fBm cubes...')
# Generate fBm cubes and save
fbm_cubes = motion_cube(fbm_x, fbm_y, x_max, y_max, staticK=staticK, binary=True, save=fbm_dir)
