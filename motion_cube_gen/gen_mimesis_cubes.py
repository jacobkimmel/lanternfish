'''
Generate training cubes from random walk and power flier paths
'''
import numpy as np
from motion_cube import motion_cube, crop_tracks, gauss_kernel
from scipy import ndimage as ndi
from skimage.morphology import disk
from skimage.filters import gaussian
import os

load_dir = 'sim_data/'
musc_mimesis = os.path.join(load_dir, 'musc_mimesis')
musc_nb_class = 4
myo_mimesis = os.path.join(load_dir, 'myo_mimesis')
myo_nb_class = 2

musc_dir = '/media/jkimmel/HDD0/myctophid/cell_mimesis/musc_cubes'
myo_dir = '/media/jkimmel/HDD0/myctophid/cell_mimesis/myo_cubes'

fold_compression = 6

print('Loading and compressing paths...')
# Load paths and compress paths


def load_compress_crop(path, nb_class, fold_compression):
    xl = []
    yl = []
    class_count = np.zeros(nb_class)
    for i in range(nb_class):
        m_x = np.loadtxt(os.path.join(path, 'mimesisX_class'+str(i+1)+'.csv'))
        m_y = np.loadtxt(os.path.join(path, 'mimesisY_class'+str(i+1)+'.csv'))
        # Compress paths
        print('Compressing paths, class ', str(i))
        m_x /= fold_compression
        m_y /= fold_compression
        # Crop tracks
        print('Cropping tracks, class ', str(i))
        m_x, m_y = crop_tracks(m_x, m_y, lbound=64, hbound=108)
        class_count[i] = m_x.shape[0]

        xl.append(m_x)
        yl.append(m_y)

    x = np.concatenate(xl)
    y = np.concatenate(yl)
    return x, y, class_count

musc_x, musc_y, musc_class_count = load_compress_crop(musc_mimesis, musc_nb_class, fold_compression)
myo_x, myo_y, myo_class_count = load_compress_crop(myo_mimesis, myo_nb_class, fold_compression)

# Class balancing

min_num = np.min([musc_x.shape[0], myo_x.shape[0]])
if min_num > 15000:
    min_num = 15000
print('Samples per cell state : ', min_num)

idx = np.arange(musc_x.shape[0])
musc_rand_idx = np.random.choice(idx, size=min_num, replace=False)
musc_x = musc_x[musc_rand_idx, :]
musc_y = musc_y[musc_rand_idx, :]

idx = np.arange(myo_x.shape[0])
myo_rand_idx = np.random.choice(idx, size=min_num, replace=False)
myo_x = myo_x[myo_rand_idx, :]
myo_y = myo_y[myo_rand_idx, :]

# Set cube size
x_max, y_max = 216, 216
width = 25
sigma = 20
staticK = np.zeros([x_max*2, y_max*2])
staticK[x_max,y_max] = 1
se = disk(width)
staticK = ndi.binary_dilation(staticK, structure=se)

# Generate RW cubes and save
print('Generating MuSC mimetic cubes...')
musc_cubes = motion_cube(musc_x, musc_y, x_max, y_max, staticK=staticK, binary=True, save=musc_dir)
#np.savez('sim_data/rw_cubes', rw_cubes)

print('Generating Myoblast mimetic cubes...')
# Generate PF cubes and save
myo_cubes = motion_cube(myo_x, myo_y, x_max, y_max, staticK=staticK, binary=True, save=myo_dir)
#np.savez('sim_data/pf_cubes', pf_cubes)
