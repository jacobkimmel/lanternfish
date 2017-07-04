'''
Train RW vs Power Flier Classifier
'''

from motcube_preprocessing import *
from bestiary import multi_contextL as model_fcn
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, CSVLogger
from keras.optimizers import SGD
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

train_dir = '/media/jkimmel/HDD0/myctophid/sim_data/train'
val_dir = '/media/jkimmel/HDD0/myctophid/sim_data/val'
batch_size = 12
cube_size = (156,156,101)
file_name_save = 'tmp.h5'
callbacks = [ModelCheckpoint(file_name_save, monitor = 'val_loss', verbose = 0, save_best_only = True, mode = 'auto'),
    LearningRateScheduler(sched),
    EarlyStopping(monitor='val_loss', patience=3),
    CSVLogger(filename=file_name_save[:-3]+'_train_log.csv', separator=',')]

mcgen = MotcubeDataGenerator()
mc_generator = mcgen.flow_from_directory(train_dir, class_mode = 'categorical', color_mode='grayscale', target_size = cube_size, batch_size = batch_size)
valgen = MotcubeDataGenerator()
val_generator = valgen.flow_from_directory(val_dir, class_mode = 'categorical', color_mode='grayscale', target_size = cube_size, batch_size = batch_size)

model = model_fcn(batch_size = batch_size, nb_classes = 3, nb_channels=1, image_x=cube_size[0], image_y=cube_size[1], image_z=cube_size[2])
model.compile(optimizer=sgd, metrics=['accuracy'], loss='categorical_crossentropy')

hist = model.fit_generator(mc_generator, samples_per_epoch=mc_generator.nb_sample, nb_epoch=30, callbacks=callbacks, validation_data=val_generator, nb_val_samples=val_generator.nb_sample)
