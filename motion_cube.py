'''
Motion Cube
Contains functions for generating spatial representations of biological motion

todo: implement GPU convolutions on minibatches
todo: implement pre-computed gaussian kernel
'''
import numpy as np
import scipy.ndimage as ndi
from skimage.filters import gaussian
from skimage.morphology import disk
import multiprocessing
from functools import partial
import os
import keras.backend as K

def gauss_kernel(kernel_length, sigma):
    '''
    Generates a Gaussian kernel

    Parameters
    ----------
    kernel_length : integer.
        side length of the square kernel.
    sigma : float.
        sigma of the Gaussian distribution to be generated.

    Returns
    -------
    kernel : ndarray.
        2D ndarray of floats, shape (kernel, kernel_length)
        centered around a Gaussian peak
    '''
    inp = np.zeros((kernel_length, kernel_length))
    inp[kernel_length//2, kernel_length//2] = 1
    return gaussian(inp, sigma)

def center_tracks(x, y):
    '''
    Centers provided tracks so that the starting loction is the origin

    Parameters
    ----------
    x : ndarray.
        size N x T ndarray containins sequential x coordinates.
    y : ndarray.
        size N x T ndarray containins sequential y coordinates.

    Returns
    -------
    x_c : ndarray.
        size N x T ndarray containins sequential x coordinates, centered.
    y_c : ndarray.
        size N x T ndarray containins sequential y coordinates, centered.
    '''

    N, T = x.shape
    x_origins = np.reshape(x[:,0], (N, 1))
    x_origin_mat = np.tile(x_origins, (1, T))
    y_origins = np.reshape(y[:,0], (N, 1))
    y_origin_mat = np.tile(y_origins, (1, T))

    x_c = x - x_origin_mat
    y_c = y - y_origin_mat

    return x_c, y_c

def crop_tracks(x, y, lbound = 128, hbound = 256):
    '''
    Removes tracks that conclude >2 SDs away from the mean distance-to-origin
    This is performed to avoid overcompression of motility traces due to a few
    outlier tracks.

    Parameters
    ----------
    x : ndarray.
        size N x T ndarray containins sequential x coordinates.
    y : ndarray.
        size N x T ndarray containins sequential y coordinates.
    lbound : integer, optional.
        minimum cropping size.
    hbound : integer, optional.
        maximum cropping size.

    Returns
    -------
    x_cr : ndarray.
        size N x T ndarray containins sequential x coordinates, cropped.
    y_cr : ndarray.
        size N x T ndarray containins sequential y coordinates, cropped.
    '''
    x_mu = np.mean(x)
    x_std = np.std(x)
    x_high_bound = np.min([np.max([x_mu+3*x_std, lbound]), hbound])
    x_low_bound = np.max([np.min([x_mu-3*x_std, -lbound]), -hbound])
    x_outlier = np.logical_or(np.max(x, axis = 1) > x_high_bound, np.min(x, axis = 1) < x_low_bound)

    y_mu = np.mean(y)
    y_std = np.std(y)
    y_high_bound = np.min([np.max([y_mu+3*y_std, 128]), hbound])
    y_low_bound = np.max([np.min([y_mu-3*y_std, -128]), -hbound])
    y_outlier = np.logical_or(np.max(y, axis = 1) > y_high_bound, np.min(y, axis = 1) < y_low_bound)

    idx = np.logical_and( np.logical_not(x_outlier), np.logical_not(y_outlier) )

    return x[idx,:], y[idx,:]


def depthwiseConvGPU(cube, k):
    '''
    Convolves a kernel depthwise (each z-plane independently)
    by using K.conv3d with an (x,y,1) 3d kernel

    Parameters
    ----------
    cube : ndarray.
        3d ndarray, (x, y, z).
    k : ndarray.
        2d ndarray (x, y) to be convolved depthwise.

    Returns
    -------
    cubeC : ndarray.
        3d ndarray (x, y, z) convolved with kernel depthwise.
    '''
    kT = K.variable(k.reshape(1,1,k.shape[0],k.shape[1], 1))
    xT = K.variable(cube.reshape(1,1,cube.shape[0], cube.shape[1], cube.shape[2]))
    out = K.conv3d(xT, kernel=kT, border_mode='same')
    cubeC = out.eval()
    return np.squeeze(cubeC)

def static_kernel_sampling(cube, staticK):
    '''
    Subsamples a static kernel to avoid computationally expensive repeat convolutions
    '''

    X, Y, T = cube.shape
    kx, ky = staticK.shape
    icx = X//2
    icy = Y//2
    kc = kx//2
    r, c, t = np.where(cube == 1)

    for i in range(T):
        sx = icx - r[i] # shift x and y
        sy = icy - c[i]
        cube[:,:,i] = staticK[int(kc-X//2+sx):int(kc+X//2+sx), int(kc-Y//2+sy):int(kc+Y//2+sy)]

    return cube

def process_cube(n, x, y, x_max, y_max, width=3, se=None, intensity=None, gsig=None, gpu=False, staticK=None, binary=False, nb_channels = 1, save=None):
    '''
    Processes a single motion cube from x and y
    Use with multiprocess.Pool.map() and funtools.partial()
    to parallelize processing

    Parameters
    ----------
    n : integer.
        sample to be processed from arrays x, y.
    x : ndarray.
        size N x T ndarray containins sequential x coordinates.
    y : ndarray.
        size N x T ndarray containins sequential y coordinates.
    x_max : integer.
        maximum extent of X dim (i.e. X field size).
    y_max : integer.
        maximum extent of Y dim (i.e. Y field size).
    width : integer or ndarray, optional.
        if integer : the size of the square used to represent the objects
            location in each XY plane for each timepoint t
        if ndarray : size N x T array of integer values representing the size to
            represent an object in the XY plane at each timpoint t
    se : ndarray, optional.
        Structuring element to use for path dilation.
        Default dilates only in XY for each timepoint, not in T.
    intensity : ndarray, optional.
        N x T array of scalars representing the intensity value used to represent
        an object for each timepoint.
    gsig : float.
        sigma for gaussian filtering of images.
    gpu : ndarray.
        perform convolutions of the specified 2D ndarray
        using Keras backend on the GPU.
    staticK : ndarray, optional.
        static kernel to be sampled spatially at each plane.
        staticK.shape >= (2*x_max, 2*y_max)
        i.e. a Gaussian filtered disk struct. element centered at a different
        point for each *t* in the stack.
    binary : boolean.
        convert resulting cubes to boolean arrays.
    nb_channels : integer.
        number of channels for returned motion cubes
        Default = 1. No current plans for >1 channel generation.
    save : string.
        directory to save arrays to (as .npy files)

    Returns
    -------
    cubeR : ndarray.
        (1, nb_channels, x_max, y_max, T) array containing a dilated motion path

    '''
    N, T = x.shape
    cube = np.zeros([x_max, y_max, T]) # make empty cube for each object
    t_idx = np.arange(T) # generate time index
    cube[np.int32(x[n,:]), np.int32(y[n,:]), t_idx] = 1 # set all XYT positions to True

    # Dilate pixels to specified size
    if not np.any(staticK):
        cubeD = ndi.binary_dilation(cube, structure=se)

    if np.any(staticK):
        cubeD = static_kernel_sampling(cube, staticK)
    elif gsig and np.any(gpu):
        cubeD = depthwiseConvGPU(cubeD, k=gpu)
    elif gsig and not np.any(gpu):
        cubeD = gaussian(cubeD, gsig, multichannel=True)
    else:
        pass
    if intensity:
        # Multiply each timepoint in T by the specified intensity
        # tile a 3D matrix from the 1, T vector of intensities
        intens = np.tile(intensity[n,:], (x_max, y_max, 1))
        cubeD = cudeD * intens # elementwise multiply1111
    if intensity or gsig or np.any(staticK) or binary:
        cubeR = np.reshape(cubeD, (1, nb_channels, x_max, y_max, T)).astype('float16')
    else:
        cubeR = np.reshape(cubeD, (1, nb_channels, x_max, y_max, T)).astype('bool')

    if save:
        np.savez(os.path.join(save, 'motcube_' + str(n).zfill(5) + '.npz'), cubeR)
    else:
        return cubeR

def motion_cube(x, y, x_max, y_max, width=3, se=None, intensity=None, gsig=None, gpu=False, staticK=None, binary=False, nb_channels=1, save=None):
    '''
    Generates an (x,y,t) motion cube from a set of sequential XY coordinates

    Parameters
    ----------
    x : ndarray.
        size N x T ndarray containins sequential x coordinates.
    y : ndarray.
        size N x T ndarray containins sequential y coordinates.
    x_max : integer.
        maximum extent of X dim (i.e. X field size).
    y_max : integer.
        maximum extent of Y dim (i.e. Y field size).
    width : integer or ndarray, optional.
        if integer : the size of the disk used to represent the objects
            location in each XY plane for each timepoint t
        if ndarray : size N x T array of integer values representing the size to
            represent an object in the XY plane at each timpoint t
    se : ndarray, optional.
        Structuring element to use for path dilation.
        Default dilates only in XY for each timepoint, not in T.
    intensity : ndarray, optional.
        N x T array of scalars representing the intensity value used to represent
        an object for each timepoint.
    gsig : float.
        sigma for gaussian filtering of images.
    gpu : ndarray.
        perform convolutions of the specified 2D ndarray
        using Keras backend on the GPU.
    staticK : ndarray, optional.
        static kernel to be sampled spatially at each plane.
        staticK.shape >= (2*x_max, 2*y_max)
        i.e. a Gaussian filtered disk struct. element centered at a different
        point for each *t* in the stack.
    binary : boolean, optional.
        converts cubes to boolean arrays.
    nb_channels : integer.
        number of channels for returned motion cubes
        Default = 1. No current plans for >1 channel generation.
    save : string.
        directory to save arrays to (as .npy files)

    Returns
    -------
    cubes : ndarray.
        4D (N, X, Y, T) array of motion cubes
    '''

    N, T = x.shape
    x = x + np.floor(x_max*0.5) # place the origin (0,0) at the image center
    y = y + np.floor(y_max*0.5)

    if se == None and type(width) == int:
        # generate a disk in XY, no dilation in T
        d = disk(width)
        se = np.zeros([d.shape[0], d.shape[1], 3])
        se[:,:,1] = d

    # use funtools.partial() to preload all other arguments
    # allowing for use of Pool.map()
    partial_process_cube = partial(process_cube, x=x, y=y, x_max=x_max, y_max=y_max, width=width, se=se, intensity=intensity, gsig=gsig, gpu=gpu, staticK=staticK, binary=binary, nb_channels=nb_channels, save=save)
    with multiprocessing.Pool() as p:
        if save:
            p.map(partial_process_cube, range(N))
        else:
            cubes_list = list(p.map(partial_process_cube,range(N)))

    if save==None:
        cubes = np.concatenate(cubes_list)
        return cubes
