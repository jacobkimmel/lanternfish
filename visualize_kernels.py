'''
Visualize Lanternfish network learning
'''

from keras.models import load_model
from keras import backend as K
import numpy as np
from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_activation, get_num_filters


# Load model
model_path = 'nets/20170502_musc_v_myo.h5'
model = load_model(model_path)

layer_name = 'conv1'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

# Visualize all filters in this layer.
filters = np.arange(get_num_filters(model.layers[layer_idx]))

# Generate input image for each filter. Here `text` field is used to overlay `filter_value` on top of the image.
vis_images = []
for idx in filters:
    img = visualize_activation(model, layer_idx, filter_indices=idx)
    img = utils.draw_text(img, str(idx))
    vis_images.append(img)
