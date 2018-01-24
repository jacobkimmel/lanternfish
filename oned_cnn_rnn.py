'''
1D CNN / RNN models for motility analysis]
'''

from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers import Input, LSTM, UpSampling1D
from keras.layers.merge import Concatenate
from keras.models import Model


def xy_ts_model(t, n_channels=2, n_classes=2, do_rate=0.3, reg=None):
    '''
    Accepts X, Y time series as input, performs 1D convs and classifies

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

    cv5 = Conv1D(64, 3, padding='same', activation='relu', activity_regularizer=reg)(p1)
    cv6 = Conv1D(64, 3, padding='same', activation='relu', activity_regularizer=reg)(cv5)
    cv7 = Conv1D(64, 3, padding='same', activation='relu', activity_regularizer=reg)(cv6)
    cv8 = Conv1D(64, 3, padding='same', activation='relu', activity_regularizer=reg)(cv7)
    p2 = MaxPooling1D(pool_size=2)(cv8)

    cv9 = Conv1D(128, 3, padding='same', activation='relu', activity_regularizer=reg)(p2)
    cv10 = Conv1D(128, 3, padding='same', activation='relu', activity_regularizer=reg)(cv9)
    cv11 = Conv1D(128, 3, padding='same', activation='relu', activity_regularizer=reg)(cv10)
    cv12 = Conv1D(128, 3, padding='same', activation='relu', activity_regularizer=reg)(cv11)
    p3 = MaxPooling1D(pool_size=2)(cv12)

    flat = Flatten()(p3)

    d1 = Dense(256, activation='relu')(flat)
    do1 = Dropout(rate=do_rate)(d1)
    d2 = Dense(256, activation='relu')(do1)
    do2 = Dropout(rate=do_rate)(d2)

    classif = Dense(n_classes, activation='softmax')(do2)

    model = Model(x, classif)

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
