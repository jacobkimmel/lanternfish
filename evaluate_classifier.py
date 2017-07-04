'''
Evaluate classifier
'''
from keras.models import load_model
from motcube_preprocessing import *
import numpy as np
import argparse

model_path = 'nets/20170405_musc_v_mef_n150.h5'
model = load_model(model_path)

eval_dir = '/media/jkimmel/HDD0/myctophid/cell_data/eval'
batch_size = 6
cube_size = (216,216,63)

evalgen = MotcubeDataGenerator()
eval_generator = evalgen.flow_from_directory(eval_dir, class_mode = 'categorical', color_mode='grayscale', target_size = cube_size, batch_size = batch_size)

e = model.evaluate_generator(eval_generator, val_samples=eval_generator.nb_sample)
