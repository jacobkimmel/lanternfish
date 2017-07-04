'''
Generate training cubes from real cell motility data
'''
import numpy as np
from motion_cube import motion_cube, crop_tracks, gauss_kernel, center_tracks
from scipy import ndimage as ndi
from skimage.morphology import disk
from skimage.filters import gaussian
import os

save_dir = '/media/jkimmel/HDD0/myctophid/cell_data'
#musc_dir = os.path.join(save_dir, 'musc_cubes')
myo_dir = os.path.join(save_dir, 'myo_cubes')
musc_dir = os.path.join(save_dir, 'musc_cubes')

fold_compression = 6

print('Loading and compressing paths...')
# Load paths
musc_x = np.loadtxt(os.path.join(musc_dir, 'all_tracksX.csv'), delimiter=',')
musc_y = np.loadtxt(os.path.join(musc_dir, 'all_tracksY.csv'), delimiter=',')
myo_x = np.loadtxt(os.path.join(myo_dir, 'all_tracksX.csv'), delimiter=',')
myo_y = np.loadtxt(os.path.join(myo_dir, 'all_tracksY.csv'), delimiter=',')

# Center paths around an origin
musc_x, musc_y = center_tracks(musc_x, musc_y)
myo_x, myo_y = center_tracks(myo_x, myo_y)

# Compress paths
musc_x = musc_x / fold_compression
musc_y = musc_y / fold_compression
myo_x = myo_x / fold_compression
myo_y = myo_y / fold_compression

# Set equivalent track length
T = np.min([musc_x.shape[1], myo_x.shape[1]])
musc_x = musc_x[:,:T]
musc_y = musc_y[:,:T]
myo_x = myo_x[:,:T]
myo_y = myo_y[:,:T]

# Crop tracks
print('Cropping tracks...')
musc_x, musc_y = crop_tracks(musc_x, musc_y, lbound=64, hbound=108)
myo_x, myo_y = crop_tracks(myo_x, myo_y, lbound=64, hbound=108)

# Class balance
print('Class balancing...')
min_class = min(musc_x.shape[0], myo_x.shape[0])
musc_x, musc_y = musc_x[:min_class, :], musc_y[:min_class,:]
myo_x, myo_y = myo_x[:min_class, :], myo_y[:min_class,:]
print('Min. class size : ', min_class)


# Set cube size
x_max, y_max = 216, 216
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
print('Generating MuSC cubes...')
musc_cubes = motion_cube(musc_x, musc_y, x_max, y_max, staticK=staticK, binary=True, save=musc_dir)
#np.savez('sim_data/musc_cubes', musc_cubes)

print('Generating Myoblast cubes...')
# Generate PF cubes and save
myo_cubes = motion_cube(myo_x, myo_y, x_max, y_max, staticK=staticK, binary=True, save=myo_dir)
#np.savez('sim_data/myo_cubes', myo_cubes)
