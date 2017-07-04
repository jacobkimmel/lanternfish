'''
Evaluate motion cube autoencoders
'''
import keras.backend as K
from keras.models import load_model
from motcube_preprocessing import *
from scipy.io import savemat
import os

model_path = '/home/jkimmel/src/myctophid/nets/20170315_ae_bin_disk25.h5'
input_dir = '/media/jkimmel/HDD0/myctophid/ae_data/val/'
save_dir = '/home/jkimmel/src/myctophid/figs/mats/'
exp_name = '20170315_ae_bin_disk25'
batch_size = 3
cube_size = (156,156,100)


print('Loading model...')
model = load_model(model_path)

print('Building data generator...')

def auto_gt_generator(generator):
    '''
    Wraps a generator to provide the same output twice in a tuple
    Useful for training with input as ground truth
    '''
    for batch in generator:
        yield (batch, batch)

valgen = MotcubeDataGenerator()
val_generator = valgen.flow_from_directory(input_dir, class_mode = None, color_mode='grayscale', target_size=cube_size, batch_size = batch_size)
ae_gen = auto_gt_generator(val_generator)

x = next(val_generator)

print('Evaluating model...')
evaluate_model = K.function(
    [model.layers[0].input, K.learning_phase()],
    [model.layers[-1].output]
    )

y = evaluate_model([x, 0])[0]

print('Saving .mat files...')
savemat(os.path.join(save_dir, exp_name + '_ae_io.mat'), mdict = {'x' : x, 'y' : y})
