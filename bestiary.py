'''
Bestiary (i.e. Model Zoo)

Contains models for analysis of spatial representations of motion
'''

from keras.models import Model, Sequential
from keras.layers import Convolution3D, MaxPooling3D, UpSampling3D, Dense, Input, BatchNormalization, Activation, Flatten, Reshape, Dropout, ZeroPadding3D
import keras.backend as K

import numpy as np

def convpool(nb_classes=2, nb_channels=1, image_x=2048, image_y=2048, image_z=60):
    '''
    Generates a vanilla ConvPooling classification net for 3D Tensors

    Parameters
    ----------
    nb_classes : integer, optional.
        number of ground truth classes to be learned.
    nb_channels : integer, optional.
        number of channels in the image data.
    image_x, image_y, image_z : integer, optional.
        dimension of the input images along x, y, and z

    Returns
    -------
    model : keras model object.
    '''

    x = Input(shape=(nb_channels, image_x, image_y, image_z))
    conv_1 = Convolution3D(4, 3, 3, 3, activation='relu', border_mode = 'same')(x)
    conv_2 = Convolution3D(4, 3, 3, 3, activation='relu', border_mode = 'same')(conv_1)
    pool_1 = MaxPooling3D(pool_size=(2,2,2))(conv_2) # 1024, 1024, 30

    conv_3 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode = 'same')(pool_1)
    conv_4 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode = 'same')(conv_3)
    pool_2 = MaxPooling3D(pool_size=(2,2,1))(conv_4) # 512, 512, 30

    conv_5 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode = 'same')(pool_2)
    conv_6 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode = 'same')(conv_5)
    conv_7 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode = 'same')(conv_6)
    pool_3 = MaxPooling3D(pool_size=(2,2,2))(conv_7) # 256, 256, 15

    conv_8 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode = 'same')(pool_3)
    conv_9 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode = 'same')(conv_8)
    conv_10 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode = 'same')(conv_9)
    pool_3 = MaxPooling3D(pool_size=(2,2,2))(conv_10) # 128, 128, 7

    flat = Flatten()(pool_3)

    fc_1 = Dense(1024, activation='relu')(flat)
    fc_2 = Dense(nb_classes, activation='relu')(fc_1)
    sm = Activation('softmax')(fc_2)

    model = Model(x, sm)

    return model

def convpool_small(nb_classes=2, nb_channels=1, image_x=312, image_y=312, image_z=101):
    '''
    Generates a vanilla ConvPooling classification net for 3D Tensors

    Parameters
    ----------
    nb_classes : integer, optional.
        number of ground truth classes to be learned.
    nb_channels : integer, optional.
        number of channels in the image data.
    image_x, image_y, image_z : integer, optional.
        dimension of the input images along x, y, and z

    Returns
    -------
    model : keras model object.
    '''

    x = Input(shape=(nb_channels, image_x, image_y, image_z))
    conv_1 = Convolution3D(2, 3, 3, 3, activation='relu', border_mode = 'same')(x)
    conv_2 = Convolution3D(2, 3, 3, 3, activation='relu', border_mode = 'same')(conv_1)
    pool_1 = MaxPooling3D(pool_size=(2,2,2))(conv_2) # 128, 128, 50

    conv_3 = Convolution3D(3, 3, 3, 3, activation='relu', border_mode = 'same')(pool_1)
    conv_4 = Convolution3D(3, 3, 3, 3, activation='relu', border_mode = 'same')(conv_3)
    pool_2 = MaxPooling3D(pool_size=(2,2,2))(conv_4) # 64, 64, 25

    conv_5 = Convolution3D(24, 3, 3, 3, activation='relu', border_mode = 'same')(pool_2)
    conv_6 = Convolution3D(24, 3, 3, 3, activation='relu', border_mode = 'same')(conv_5)
    conv_7 = Convolution3D(24, 3, 3, 3, activation='relu', border_mode = 'same')(conv_6)
    pool_3 = MaxPooling3D(pool_size=(2,2,2))(conv_7) # 32, 32, 12

    conv_8 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode = 'same')(pool_3)
    conv_9 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode = 'same')(conv_8)
    conv_10 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode = 'same')(conv_9)
    pool_3 = MaxPooling3D(pool_size=(2,2,2))(conv_10) # 16, 16, 6

    flat = Flatten()(pool_3)

    fc_1 = Dense(1024, activation='relu')(flat)
    fc_2 = Dense(nb_classes, activation='relu')(fc_1)
    sm = Activation('softmax')(fc_2)

    model = Model(x, sm)

    return model

def test_tb(batch_size=16, nb_classes=2, nb_channels=1, image_x=64, image_y=64, image_z=32):
    '''
    Generates a vanilla ConvPooling classification net for 3D Tensors

    Parameters
    ----------
    nb_classes : integer, optional.
        number of ground truth classes to be learned.
    nb_channels : integer, optional.
        number of channels in the image data.
    image_x, image_y, image_z : integer, optional.
        dimension of the input images along x, y, and z

    Returns
    -------
    model : keras model object.
    '''

    x = Input(batch_shape=(batch_size, nb_channels, image_x, image_y, image_z))
    conv_1 = Convolution3D(16, 7, 7, 7, activation='relu', border_mode = 'same')(x)
    conv_2 = Convolution3D(16, 5, 5, 5, activation='relu', border_mode = 'same')(conv_1)
    conv_3 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode = 'same')(conv_2)
    pool_1 = MaxPooling3D(pool_size=(2,2,2))(conv_3) # 32, 32, 16

    conv_4 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode = 'same')(pool_1)
    conv_5 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode = 'same')(conv_4)
    pool_2 = MaxPooling3D(pool_size=(2,2,2))(conv_5) # 16, 16, 8

    conv_6 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode = 'same')(pool_2)
    conv_7 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode = 'same')(conv_6)
    pool_3 = MaxPooling3D(pool_size=(2,2,2))(conv_7) # 8, 8, 4

    flat = Flatten()(pool_3)

    fc_1 = Dense(1024, activation='relu')(flat)
    fc_2 = Dense(256, activation='relu')(fc_1)
    fc_3 = Dense(1, activation='sigmoid')(fc_2)

    model = Model(x, fc_3)

    return model

def large_context(batch_size=3, nb_channels=1, image_x=64, image_y=64, image_z=32):
    '''
    Generates a binary classification net for 3D Tensors that utilizes stacked
    convolutions to generate a large receptive field

    Parameters
    ----------
    nb_channels : integer, optional.
        number of channels in the image data.
    image_x, image_y, image_z : integer, optional.
        dimension of the input images along x, y, and z

    Returns
    -------
    model : keras model object.
    '''

    x = Input(batch_shape=(batch_size, nb_channels, image_x, image_y, image_z))
    conv_1 = Convolution3D(2, 3, 3, 3, activation='relu', border_mode = 'same')(x)
    conv_2 = Convolution3D(2, 3, 3, 3, activation='relu', border_mode = 'same')(conv_1)
    pool_1 = MaxPooling3D(pool_size=(2,2,2))(conv_2) # 156, 156, 50

    conv_3 = Convolution3D(4, 3, 3, 3, activation='relu', border_mode = 'same')(pool_1)
    conv_4 = Convolution3D(4, 3, 3, 3, activation='relu', border_mode = 'same')(conv_3)
    pool_2 = MaxPooling3D(pool_size=(2,2,2))(conv_4) # 78, 78, 25

    conv_5 = Convolution3D(12, 3, 3, 3, activation='relu', border_mode = 'same')(pool_2)
    conv_6 = Convolution3D(12, 3, 3, 3, activation='relu', border_mode = 'same')(conv_5)
    pool_3 = MaxPooling3D(pool_size=(2,2,2))(conv_6) # 39, 39, 12

    conv_7 = Convolution3D(12, 3, 3, 3, activation='relu', border_mode = 'same')(pool_3)
    conv_8 = Convolution3D(12, 3, 3, 3, activation='relu', border_mode = 'same')(conv_7)
    pool_4 = MaxPooling3D(pool_size=(2,2,2))(conv_8) # 19, 19, 6

    flat = Flatten()(pool_4)

    fc_1 = Dense(1024, activation='relu')(flat)
    fc_2 = Dense(256, activation='relu')(fc_1)
    fc_3 = Dense(1, activation='sigmoid')(fc_2)

    model = Model(x, fc_3)

    return model

def large_context_dropout(batch_size=3, nb_channels=1, image_x=64, image_y=64, image_z=32):
    '''
    Generates a binary classification net for 3D Tensors that utilizes stacked
    convolutions to generate a large receptive field

    Parameters
    ----------
    nb_channels : integer, optional.
        number of channels in the image data.
    image_x, image_y, image_z : integer, optional.
        dimension of the input images along x, y, and z

    Returns
    -------
    model : keras model object.
    '''

    x = Input(batch_shape=(batch_size, nb_channels, image_x, image_y, image_z))
    conv_1 = Convolution3D(2, 3, 3, 3, activation='relu', border_mode = 'same')(x)
    conv_2 = Convolution3D(2, 3, 3, 3, activation='relu', border_mode = 'same')(conv_1)
    pool_1 = MaxPooling3D(pool_size=(2,2,2))(conv_2) # 156, 156, 50

    conv_3 = Convolution3D(4, 3, 3, 3, activation='relu', border_mode = 'same')(pool_1)
    conv_4 = Convolution3D(4, 3, 3, 3, activation='relu', border_mode = 'same')(conv_3)
    pool_2 = MaxPooling3D(pool_size=(2,2,2))(conv_4) # 78, 78, 25

    conv_5 = Convolution3D(12, 3, 3, 3, activation='relu', border_mode = 'same')(pool_2)
    conv_6 = Convolution3D(12, 3, 3, 3, activation='relu', border_mode = 'same')(conv_5)
    pool_3 = MaxPooling3D(pool_size=(2,2,2))(conv_6) # 39, 39, 12

    conv_7 = Convolution3D(12, 3, 3, 3, activation='relu', border_mode = 'same')(pool_3)
    conv_8 = Convolution3D(12, 3, 3, 3, activation='relu', border_mode = 'same')(conv_7)
    pool_4 = MaxPooling3D(pool_size=(2,2,2))(conv_8) # 19, 19, 6

    flat = Flatten()(pool_4)

    fc_1 = Dense(1024, activation='relu')(flat)
    do_1 = Dropout(0.3)(fc_1)
    fc_2 = Dense(256, activation='relu')(do_1)
    do_2 = Dropout(0.3)(fc_2)
    fc_3 = Dense(1, activation='sigmoid')(do_2)

    model = Model(x, fc_3)

    return model

def multi_context(batch_size=3, nb_classes=3, nb_channels=1, image_x=64, image_y=64, image_z=32):
    '''
    Generates a binary classification net for 3D Tensors that utilizes stacked
    convolutions to generate a large receptive field

    Parameters
    ----------
    nb_channels : integer, optional.
        number of channels in the image data.
    image_x, image_y, image_z : integer, optional.
        dimension of the input images along x, y, and z

    Returns
    -------
    model : keras model object.
    '''

    x = Input(batch_shape=(batch_size, nb_channels, image_x, image_y, image_z))
    conv_1 = Convolution3D(2, 3, 3, 3, activation='relu', border_mode = 'same')(x)
    conv_2 = Convolution3D(2, 3, 3, 3, activation='relu', border_mode = 'same')(conv_1)
    pool_1 = MaxPooling3D(pool_size=(2,2,2))(conv_2) # 156, 156, 50

    conv_3 = Convolution3D(4, 3, 3, 3, activation='relu', border_mode = 'same')(pool_1)
    conv_4 = Convolution3D(4, 3, 3, 3, activation='relu', border_mode = 'same')(conv_3)
    pool_2 = MaxPooling3D(pool_size=(2,2,2))(conv_4) # 78, 78, 25

    conv_5 = Convolution3D(12, 3, 3, 3, activation='relu', border_mode = 'same')(pool_2)
    conv_6 = Convolution3D(12, 3, 3, 3, activation='relu', border_mode = 'same')(conv_5)
    pool_3 = MaxPooling3D(pool_size=(2,2,2))(conv_6) # 39, 39, 12

    conv_7 = Convolution3D(12, 3, 3, 3, activation='relu', border_mode = 'same')(pool_3)
    conv_8 = Convolution3D(12, 3, 3, 3, activation='relu', border_mode = 'same')(conv_7)
    pool_4 = MaxPooling3D(pool_size=(2,2,2))(conv_8) # 19, 19, 6

    flat = Flatten()(pool_4)

    fc_1 = Dense(1024, activation='relu')(flat)
    do_1 = Dropout(0.3)(fc_1)
    fc_2 = Dense(256, activation='relu')(do_1)
    do_2 = Dropout(0.3)(fc_2)
    fc_3 = Dense(nb_classes, activation='softmax')(do_2)

    model = Model(x, fc_3)

    return model

def multi_contextL(batch_size=3, nb_classes=3, nb_channels=1, image_x=64, image_y=64, image_z=32):
    '''
    Generates a binary classification net for 3D Tensors that utilizes stacked
    convolutions to generate a large receptive field

    Parameters
    ----------
    nb_channels : integer, optional.
        number of channels in the image data.
    image_x, image_y, image_z : integer, optional.
        dimension of the input images along x, y, and z

    Returns
    -------
    model : keras model object.
    '''

    x = Input(batch_shape=(batch_size, nb_channels, image_x, image_y, image_z))
    conv_1 = Convolution3D(2, 3, 3, 3, activation='relu', border_mode = 'same')(x)
    conv_2 = Convolution3D(2, 3, 3, 3, activation='relu', border_mode = 'same')(conv_1)
    pool_1 = MaxPooling3D(pool_size=(2,2,2))(conv_2) # 78, 78, 50

    conv_3 = Convolution3D(8, 3, 3, 3, activation='relu', border_mode = 'same')(pool_1)
    conv_4 = Convolution3D(8, 3, 3, 3, activation='relu', border_mode = 'same')(conv_3)
    pool_2 = MaxPooling3D(pool_size=(2,2,2))(conv_4) # 39, 39, 25

    conv_5 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode = 'same')(pool_2)
    conv_6 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode = 'same')(conv_5)
    pool_3 = MaxPooling3D(pool_size=(2,2,2))(conv_6) # 19, 19, 12

    conv_7 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode = 'same')(pool_3)
    conv_8 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode = 'same')(conv_7)
    pool_4 = MaxPooling3D(pool_size=(2,2,2))(conv_8) # 19, 19, 6

    flat = Flatten()(pool_4)

    fc_1 = Dense(1024, activation='relu')(flat)
    do_1 = Dropout(0.3)(fc_1)
    fc_2 = Dense(256, activation='relu')(do_1)
    do_2 = Dropout(0.3)(fc_2)
    fc_3 = Dense(nb_classes, activation='softmax')(do_2)

    model = Model(x, fc_3)

    return model

def motcube_ae(batch_size=3, nb_channels=1, image_x=156, image_y=156, image_z=100):
    '''
    Generates a binary classification net for 3D Tensors that utilizes stacked
    convolutions to generate a large receptive field

    Parameters
    ----------
    nb_channels : integer, optional.
        number of channels in the image data.
    image_x, image_y, image_z : integer, optional.
        dimension of the input images along x, y, and z

    Returns
    -------
    model : keras model object.
    '''

    x = Input(batch_shape=(batch_size, nb_channels, image_x, image_y, image_z))
    conv_1 = Convolution3D(2, 3, 3, 3, activation='relu', border_mode = 'same')(x)
    conv_2 = Convolution3D(2, 3, 3, 3, activation='relu', border_mode = 'same')(conv_1)
    pool_1 = MaxPooling3D(pool_size=(2,2,2))(conv_2) # 78, 78, 50

    conv_3 = Convolution3D(8, 3, 3, 3, activation='relu', border_mode = 'same')(pool_1)
    conv_4 = Convolution3D(8, 3, 3, 3, activation='relu', border_mode = 'same')(conv_3)
    pool_2 = MaxPooling3D(pool_size=(2,2,2))(conv_4) # 39, 39, 25

    conv_5 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode = 'same')(pool_2)
    conv_6 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode = 'same')(conv_5)
    pool_3 = MaxPooling3D(pool_size=(2,2,2))(conv_6) # 19, 19, 12

    conv_7 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode = 'same')(pool_3)
    conv_8 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode = 'same')(conv_7)
    pool_4 = MaxPooling3D(pool_size=(2,2,2))(conv_8) # 9, 9, 6

    conv_9 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode = 'same')(pool_4)
    conv_10 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode = 'same')(conv_9)
    up_1 = UpSampling3D((2,2,2))(conv_10) # 18, 18, 12

    conv_11 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode = 'same')(up_1)
    conv_12 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode = 'same')(conv_11)
    up_2 = UpSampling3D((2,2,2))(conv_12) # 36, 36, 24
    zpad_1 = ZeroPadding3D((1,1,0))(up_2) # 38, 38, 24

    conv_13 = Convolution3D(8, 3, 3, 3, activation='relu', border_mode = 'same')(zpad_1)
    conv_14 = Convolution3D(8, 3, 3, 3, activation='relu', border_mode = 'same')(conv_13)
    up_3 = UpSampling3D((2,2,2))(conv_14) # 76, 76, 48
    zpad_2 = ZeroPadding3D((1,1,1))(up_3) # 78, 78, 50

    conv_15 = Convolution3D(2, 3, 3, 3, activation='relu', border_mode = 'same')(zpad_2)
    conv_16 = Convolution3D(2, 3, 3, 3, activation='relu', border_mode = 'same')(conv_15)
    up_4 = UpSampling3D((2,2,2))(conv_16) # 156, 156, 100

    conv_fin = Convolution3D(1, 3, 3, 3, activation='relu', border_mode = 'same')(up_4)


    model = Model(x, conv_fin)

    return model

def bottleneck_ae(batch_size=3, nb_channels=1, image_x=156, image_y=156, image_z=100):
    '''
    Generates a binary classification net for 3D Tensors that utilizes stacked
    convolutions to generate a large receptive field

    Parameters
    ----------
    nb_channels : integer, optional.
        number of channels in the image data.
    image_x, image_y, image_z : integer, optional.
        dimension of the input images along x, y, and z

    Returns
    -------
    model : keras model object.
    '''

    x = Input(batch_shape=(batch_size, nb_channels, image_x, image_y, image_z))
    conv_1 = Convolution3D(2, 3, 3, 3, activation='relu', border_mode = 'same')(x)
    conv_2 = Convolution3D(2, 3, 3, 3, activation='relu', border_mode = 'same')(conv_1)
    pool_1 = MaxPooling3D(pool_size=(2,2,2))(conv_2) # 78, 78, 50

    conv_3 = Convolution3D(8, 3, 3, 3, activation='relu', border_mode = 'same')(pool_1)
    conv_4 = Convolution3D(8, 3, 3, 3, activation='relu', border_mode = 'same')(conv_3)
    pool_2 = MaxPooling3D(pool_size=(2,2,2))(conv_4) # 39, 39, 25

    conv_5 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode = 'same')(pool_2)
    conv_6 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode = 'same')(conv_5)
    pool_3 = MaxPooling3D(pool_size=(2,2,2))(conv_6) # 19, 19, 12

    conv_7 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode = 'same')(pool_3)
    conv_8 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode = 'same')(conv_7)
    pool_4 = MaxPooling3D(pool_size=(2,2,2))(conv_8) # 9, 9, 6

    conv_9 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode = 'same')(pool_4)
    conv_10 = Convolution3D(1, 3, 3, 3, activation='relu', border_mode = 'same')(conv_9)

    flat_1 = Flatten()(conv_10)
    fc_1 = Dense(256, activation='relu')(flat_1)
    do_1 = Dropout(0.3)(fc_1)
    fc_2 = Dense(256, activation='relu')(do_1)
    do_2 = Dropout(0.3)(fc_2)
    fc_3 = Dense(486, activation='relu')(do_2)
    reshape_1 = Reshape((1,9,9,6), input_shape=(486,))(fc_3)

    conv_11 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode = 'same')(reshape_1)
    conv_12 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode = 'same')(conv_11)
    up_1 = UpSampling3D((2,2,2))(conv_12) # 18, 18, 12

    conv_13 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode = 'same')(up_1)
    conv_14 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode = 'same')(conv_13)
    up_2 = UpSampling3D((2,2,2))(conv_14) # 36, 36, 24
    zpad_1 = ZeroPadding3D((1,1,0))(up_2) # 38, 38, 24

    conv_15 = Convolution3D(8, 3, 3, 3, activation='relu', border_mode = 'same')(zpad_1)
    conv_16 = Convolution3D(8, 3, 3, 3, activation='relu', border_mode = 'same')(conv_15)
    up_3 = UpSampling3D((2,2,2))(conv_16) # 76, 76, 48
    zpad_2 = ZeroPadding3D((1,1,1))(up_3) # 78, 78, 50

    conv_17 = Convolution3D(2, 3, 3, 3, activation='relu', border_mode = 'same')(zpad_2)
    conv_18 = Convolution3D(2, 3, 3, 3, activation='relu', border_mode = 'same')(conv_17)
    up_4 = UpSampling3D((2,2,2))(conv_18) # 156, 156, 100

    conv_fin = Convolution3D(1, 3, 3, 3, activation='relu', border_mode = 'same')(up_4)


    model = Model(x, conv_fin)

    return model
