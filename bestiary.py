'''
Bestiary (i.e. Model Zoo)

Contains models for analysis of spatial representations of motion
rewritten (slowly!) using the Keras v2 API
'''

from keras.models import Model, Sequential
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Dense, Input, BatchNormalization, Activation, Flatten, Reshape, Dropout, ZeroPadding3D
import keras.backend as K
from keras.layers import Conv1D, MaxPooling1D, LSTM, UpSampling1D, Input, Reshape

import numpy as np

def multi_contextL(batch_size=3, nb_classes=3, nb_channels=1,
                    image_x=64, image_y=64, image_z=32, dim_ordering='tf'):
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

    if dim_ordering in ['tf', 'channels_last']:
        x = Input(shape=(image_x, image_y, image_z, nb_channels))
    else:
        x = Input(shape=(nb_channels, image_x, image_y, image_z))
    conv_1 = Conv3D(2, (3, 3, 3), activation='relu', padding = 'same')(x)
    conv_2 = Conv3D(2, (3, 3, 3), activation='relu', padding = 'same')(conv_1)
    pool_1 = MaxPooling3D(pool_size=(2,2,2))(conv_2) # 78, 78, 50

    conv_3 = Conv3D(8, (3, 3, 3), activation='relu', padding = 'same')(pool_1)
    conv_4 = Conv3D(8, (3, 3, 3), activation='relu', padding = 'same')(conv_3)
    pool_2 = MaxPooling3D(pool_size=(2,2,2))(conv_4) # 39, 39, 25

    conv_5 = Conv3D(32, (3, 3, 3), activation='relu', padding = 'same')(pool_2)
    conv_6 = Conv3D(32, (3, 3, 3), activation='relu', padding = 'same')(conv_5)
    pool_3 = MaxPooling3D(pool_size=(2,2,2))(conv_6) # 19, 19, 12

    conv_7 = Conv3D(64, (3, 3, 3), activation='relu', padding = 'same')(pool_3)
    conv_8 = Conv3D(64, (3, 3, 3), activation='relu', padding = 'same')(conv_7)
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
    conv_1 = Conv3D(2, (3, 3, 3), activation='relu', padding = 'same')(x)
    conv_2 = Conv3D(2, (3, 3, 3), activation='relu', padding = 'same')(conv_1)
    pool_1 = MaxPooling3D(pool_size=(2,2,2))(conv_2) # 78, 78, 50

    conv_3 = Conv3D(8, (3, 3, 3), activation='relu', padding = 'same')(pool_1)
    conv_4 = Conv3D(8, (3, 3, 3), activation='relu', padding = 'same')(conv_3)
    pool_2 = MaxPooling3D(pool_size=(2,2,2))(conv_4) # 39, 39, 25

    conv_5 = Conv3D(32, (3, 3, 3), activation='relu', padding = 'same')(pool_2)
    conv_6 = Conv3D(32, (3, 3, 3), activation='relu', padding = 'same')(conv_5)
    pool_3 = MaxPooling3D(pool_size=(2,2,2))(conv_6) # 19, 19, 12

    conv_7 = Conv3D(64, (3, 3, 3), activation='relu', padding = 'same')(pool_3)
    conv_8 = Conv3D(64, (3, 3, 3), activation='relu', padding = 'same')(conv_7)
    pool_4 = MaxPooling3D(pool_size=(2,2,2))(conv_8) # 9, 9, 6

    conv_9 = Conv3D(64, (3, 3, 3), activation='relu', padding = 'same')(pool_4)
    conv_10 = Conv3D(64, (3, 3, 3), activation='relu', padding = 'same')(conv_9)
    up_1 = UpSampling3D((2,2,2))(conv_10) # 18, 18, 12

    conv_11 = Conv3D(32, (3, 3, 3), activation='relu', padding = 'same')(up_1)
    conv_12 = Conv3D(32, (3, 3, 3), activation='relu', padding = 'same')(conv_11)
    up_2 = UpSampling3D((2,2,2))(conv_12) # 36, 36, 24
    zpad_1 = ZeroPadding3D((1,1,0))(up_2) # 38, 38, 24

    conv_13 = Conv3D(8, (3, 3, 3), activation='relu', padding = 'same')(zpad_1)
    conv_14 = Conv3D(8, (3, 3, 3), activation='relu', padding = 'same')(conv_13)
    up_3 = UpSampling3D((2,2,2))(conv_14) # 76, 76, 48
    zpad_2 = ZeroPadding3D((1,1,1))(up_3) # 78, 78, 50

    conv_15 = Conv3D(2, (3, 3, 3), activation='relu', padding = 'same')(zpad_2)
    conv_16 = Conv3D(2, (3, 3, 3), activation='relu', padding = 'same')(conv_15)
    up_4 = UpSampling3D((2,2,2))(conv_16) # 156, 156, 100

    conv_fin = Conv3D(1, (3, 3, 3), activation='relu', padding = 'same')(up_4)


    model = Model(x, conv_fin)

    return model

def bottleneck_ae(batch_size=3, nb_channels=1,
                    image_x=156, image_y=156, image_z=100,
                    dim_ordering='channels_last'):
    '''
    Generates a binary classification net for 3D Tensors that utilizes stacked
    convolutions to generate a large receptive field

    Parameters
    ----------
    nb_channels : integer, optional.
        number of channels in the image data.
    image_x, image_y, image_z : integer, optional.
        dimension of the input images along x, y, and z
    dim_ordering : string, optional.
        specify dimension ordering in:
            ['channels_last', 'channels_first', 'tf', 'th']

    Returns
    -------
    model : keras model object.
    '''

    if dim_ordering in ['channels_last', 'tf', 'tensorflow']:
        x = Input(batch_shape=(batch_size, image_x, image_y, image_z, nb_channels))
    elif dim_ordering in ['channels_first', 'th', 'theano']:
        x = Input(batch_shape=(batch_size, nb_channels, image_x, image_y, image_z))
    else:
        raise ValueError('dim_ordering must be in ["channels_first, "channels_last", "tf"]')

    conv_1 = Conv3D(2, (3, 3, 3), activation='relu', padding = 'same')(x)
    conv_2 = Conv3D(2, (3, 3, 3), activation='relu', padding = 'same')(conv_1)
    pool_1 = MaxPooling3D(pool_size=(2,2,2))(conv_2) # 78, 78, 50

    conv_3 = Conv3D(8, (3, 3, 3), activation='relu', padding = 'same')(pool_1)
    conv_4 = Conv3D(8, (3, 3, 3), activation='relu', padding = 'same')(conv_3)
    pool_2 = MaxPooling3D(pool_size=(2,2,2))(conv_4) # 39, 39, 25

    conv_5 = Conv3D(32, (3, 3, 3), activation='relu', padding = 'same')(pool_2)
    conv_6 = Conv3D(32, (3, 3, 3), activation='relu', padding = 'same')(conv_5)
    pool_3 = MaxPooling3D(pool_size=(2,2,2))(conv_6) # 19, 19, 12

    conv_7 = Conv3D(64, (3, 3, 3), activation='relu', padding = 'same')(pool_3)
    conv_8 = Conv3D(64, (3, 3, 3), activation='relu', padding = 'same')(conv_7)
    pool_4 = MaxPooling3D(pool_size=(2,2,2))(conv_8) # 9, 9, 6

    conv_9 = Conv3D(64, (3, 3, 3), activation='relu', padding = 'same')(pool_4)
    conv_10 = Conv3D(1, (3, 3, 3), activation='relu', padding = 'same')(conv_9)

    flat_1 = Flatten()(conv_10)
    fc_1 = Dense(256, activation='relu')(flat_1)
    do_1 = Dropout(0.3)(fc_1)
    fc_2 = Dense(256, activation='relu')(do_1)
    do_2 = Dropout(0.3)(fc_2)
    fc_3 = Dense(486, activation='relu')(do_2)

    if dim_ordering in ['channels_last', 'tf', 'tensorflow']:
        reshape_1 = Reshape((9,9,6,1), input_shape=(486,))(fc_3)
    elif dim_ordering in ['channels_first', 'th', 'theano']:
        reshape_1 = Reshape((1,9,9,6), input_shape=(486,))(fc_3)
    else:
        raise ValueError('dim_ordering must be in ["channels_first, "channels_last", "tf"]')

    conv_11 = Conv3D(64, (3, 3, 3), activation='relu', padding = 'same')(reshape_1)
    conv_12 = Conv3D(64, (3, 3, 3), activation='relu', padding = 'same')(conv_11)
    up_1 = UpSampling3D((2,2,2))(conv_12) # 18, 18, 12

    conv_13 = Conv3D(32, (3, 3, 3), activation='relu', padding = 'same')(up_1)
    conv_14 = Conv3D(32, (3, 3, 3), activation='relu', padding = 'same')(conv_13)
    up_2 = UpSampling3D((2,2,2))(conv_14) # 36, 36, 24
    zpad_1 = ZeroPadding3D((1,1,0))(up_2) # 38, 38, 24

    conv_15 = Conv3D(8, (3, 3, 3), activation='relu', padding = 'same')(zpad_1)
    conv_16 = Conv3D(8, (3, 3, 3), activation='relu', padding = 'same')(conv_15)
    up_3 = UpSampling3D((2,2,2))(conv_16) # 76, 76, 48
    zpad_2 = ZeroPadding3D((1,1,1))(up_3) # 78, 78, 50

    conv_17 = Conv3D(2, (3, 3, 3), activation='relu', padding = 'same')(zpad_2)
    conv_18 = Conv3D(2, (3, 3, 3), activation='relu', padding = 'same')(conv_17)
    up_4 = UpSampling3D((2,2,2))(conv_18) # 156, 156, 100

    conv_fin = Conv3D(1, (3, 3, 3), activation='relu', padding = 'same')(up_4)


    model = Model(x, conv_fin)

    return model

def xy_ts_rnn_model(t, n_channels=2, n_classes=2, do_rate=0.3, reg=None, min_dim=128):
    '''
    Accepts X, Y time series as input, performs 1D convs with an LSTM layer and
    classifies

    Parameters
    ----------
    t : integer.
        length of time series.
    n_channels : integer.
        number of channels in data.
    n_classes : integer.
        number of output classes.
    reg : callable or None.
        regularizer.

    Returns
    -------
    model : keras model object.
    '''

    x = Input(shape=(t, n_channels))

    cv1 = Conv1D(16, 3, padding='same', activation='relu', activity_regularizer=reg)(x)
    cv2 = Conv1D(16, 3, padding='same', activation='relu', activity_regularizer=reg)(cv1)
    cv3 = Conv1D(32, 3, padding='same', activation='relu', activity_regularizer=reg)(cv2)
    cv4 = Conv1D(32, 3, padding='same', activation='relu', activity_regularizer=reg)(cv3)
    p1 = MaxPooling1D(pool_size=2)(cv4)

    lstm = LSTM(256, dropout=0.2, recurrent_dropout=0.2)(p1)

    d1 = Dense(256, activation='relu')(lstm)
    do1 = Dropout(rate=do_rate)(d1)
    d2 = Dense(128, activation='relu')(do1)
    do2 = Dropout(rate=do_rate)(d2)

    classif = Dense(n_classes, activation='softmax')(do2)

    model = Model(x, classif)

    return model

def xy_ts_rnn_ae(t, n_channels=2, n_classes=2, do_rate=0.3, reg=None, min_dim=128):
    '''
    Accepts X, Y time series as input, performs 1D convs with an LSTM layer
    and upsamples as an autoencoder.

    Parameters
    ----------
    t : integer.
        length of time series.
    n_channels : integer.
        number of channels in data.
    n_classes : integer.
        number of output classes.
    reg : callable or None.
        regularizer.

    Returns
    -------
    model : keras model object.
    '''

    x = Input(shape=(t, n_channels))

    cv1 = Conv1D(16, 3, padding='same', activation='relu', activity_regularizer=reg)(x)
    cv2 = Conv1D(16, 3, padding='same', activation='relu', activity_regularizer=reg)(cv1)
    cv3 = Conv1D(32, 3, padding='same', activation='relu', activity_regularizer=reg)(cv2)
    cv4 = Conv1D(16, 3, padding='same', activation='relu', activity_regularizer=reg)(cv3)
    p1 = MaxPooling1D(pool_size=2)(cv4)
    # cv5 = Conv1D(32, 3, padding='same', activation='relu', activity_regularizer=reg)(p1)
    # cv6 = Conv1D(8, 3, padding='same', activation='relu', activity_regularizer=reg)(cv5)
    # p2 = MaxPooling1D(pool_size=2)(cv6)

    lstm = LSTM(t//2*16, dropout=0.2, recurrent_dropout=0.2)(p1)

    d1 = Dense(min_dim, activation='relu')(lstm)
    do1 = Dropout(rate=do_rate)(d1)
    d2 = Dense(t//2*16, activation='relu')(do1)
    do2 = Dropout(rate=do_rate)(d2)

    rs = Reshape(target_shape=(t//2, 16))(do2)

    # us1 = UpSampling1D(size=2)(rs)
    # uc1 = Conv1D(8, 3, padding='same', activation='relu', activity_regularizer=reg)(us1)
    # uc2 = Conv1D(32, 3, padding='same', activation='relu', activity_regularizer=reg)(uc1)

    us2 = UpSampling1D(size=2)(rs)
    uc3 = Conv1D(32, 3, padding='same', activation='relu', activity_regularizer=reg)(us2)
    uc4 = Conv1D(32, 3, padding='same', activation='relu', activity_regularizer=reg)(uc3)
    uc5 = Conv1D(16, 3, padding='same', activation='relu', activity_regularizer=reg)(uc4)
    uc6 = Conv1D(n_channels, 3, padding='same', activity_regularizer=reg)(uc5)

    model = Model(x, uc6)

    return model

def rnn_motion_pred(t_in, t_out, n_channels=2, do_rate=0.3, reg=None, k=32):
    '''
    Accepts X, Y time series as input, performs 1D convs, pushes to LSTM,
    and convolves the output.

    Parameters
    ----------
    t_in : integer.
        length of input time series.
    t_out : integer.
        length of output prediction.
    n_channels : integer.
        number of channels in data.
    n_classes : integer.
        number of output classes.
    reg : callable or None.
        regularizer.
    k : integer.
        base number of Conv1D filter kernels.

    Returns
    -------
    model : keras model object.
    '''

    x = Input(shape=(t_in, n_channels))

    cv1 = Conv1D(k*2, 3, padding='same', activation='relu', activity_regularizer=reg)(x)
    cv2 = Conv1D(k*2, 3, padding='same', activation='relu', activity_regularizer=reg)(cv1)
    cv3 = Conv1D(k*2, 3, padding='same', activation='relu', activity_regularizer=reg)(cv2)
    cv4 = Conv1D(k, 3, padding='same', activation='relu', activity_regularizer=reg)(cv3)

    lstm = LSTM(t_out*n_channels, dropout=0.2, recurrent_dropout=0.2)(cv4)

    rshp = Reshape(target_shape=(t_out, n_channels))(lstm)

    uc3 = Conv1D(k, 3, padding='same', activation='relu', activity_regularizer=reg)(rshp)
    uc4 = Conv1D(k*2, 3, padding='same', activation='relu', activity_regularizer=reg)(uc3)
    uc5 = Conv1D(k*2, 3, padding='same', activation='relu', activity_regularizer=reg)(uc4)
    uc6 = Conv1D(n_channels, 3, padding='same', activity_regularizer=reg)(uc5)

    model = Model(x, uc6)

    return model
