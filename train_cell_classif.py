'''
Train a cell type classifier
'''

from motcube_preprocessing import *
from bestiary import multi_contextL as model_fcn
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, CSVLogger
from keras.optimizers import SGD
from keras.models import load_model
import numpy as np

def lr_schedule(rate=0.01, decay=0.8):
    '''
    Generates a schedule function with exp decay

    alpha_new = alpha_init * decay_coeff^epoch

    Parameters
    ----------
    rate : float.
        learning rate.
    decay : float.
        decay coefficient
    '''
    def sched(epoch):
        return (rate * (decay**np.int(epoch)))
    return sched

sched = lr_schedule(rate = 0.005, decay = 0.8)
sgd = SGD(momentum = 0.5) #momentum = 0.5

trained_model_path = 'nets/20170315_multiclass_bin_disk25.h5'
train_dir = '/media/jkimmel/HDD0/myctophid/cell_data/train'
val_dir = '/media/jkimmel/HDD0/myctophid/cell_data/val'
nb_classes = 2
batch_size = 6
cube_size = (216,216,63)
file_name_save = '20170503_musc_v_myo_dynamic_noise_ms4.h5'
callbacks = [ModelCheckpoint(file_name_save, monitor = 'val_loss', verbose = 0, save_best_only = True, mode = 'auto'),
    LearningRateScheduler(sched),
    EarlyStopping(monitor='val_loss', patience=5),
    CSVLogger(filename=file_name_save[:-3]+'_train_log.csv', separator=',')]

mcgen = MotcubeDataGenerator(horizontal_flip=True, vertical_flip=True, dynamic_noise=(0.5,4))
mc_generator = mcgen.flow_from_directory(train_dir, class_mode = 'categorical', color_mode='grayscale', target_size = cube_size, batch_size = batch_size)
valgen = MotcubeDataGenerator(horizontal_flip=True, vertical_flip=True)
val_generator = valgen.flow_from_directory(val_dir, class_mode = 'categorical', color_mode='grayscale', target_size = cube_size, batch_size = batch_size)

model = model_fcn(batch_size = batch_size, nb_classes = nb_classes, nb_channels=1, image_x=cube_size[0], image_y=cube_size[1], image_z=cube_size[2])
model.compile(optimizer=sgd, metrics=['accuracy'], loss='categorical_crossentropy')

# Transfer model weights
pretrain_model = load_model(trained_model_path)
w0 = pretrain_model.get_weights()
w1 = model.get_weights()
w = w0
w[16:] = w1[16:] # Dense layers are different sizes, use different weights
model.set_weights(w)


hist = model.fit_generator(mc_generator, samples_per_epoch=mc_generator.nb_sample, nb_epoch=30, callbacks=callbacks, validation_data=val_generator, nb_val_samples=val_generator.nb_sample)
