
import numpy as np
import scipy.ndimage as nd
import tensorflow as tf
from matplotlib import pyplot as plt
#from lucid.modelzoo import vision_models as models
#import lucid.modelzoo.vision_models as models
from lucid.misc.io import show
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform


from lucid.modelzoo.vision_base import Model, Layer

from lucid.modelzoo.caffe_models import *
from lucid.modelzoo.slim_models import *
from lucid.modelzoo.other_models import *


__all__ = [_name for _name, _obj in list(globals().items())
           if isinstance(_obj, type) and issubclass(_obj, Model)]

model = InceptionV3_slim()
print(model.layers)
model.load_graphdef()

LEARNING_RATE = 0.05
DECORRELATE = True
ROBUSTNESS  = True
optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

L1   = -0.05
TV   = -0.25
BLUR = -1.0

obj  = objectives.channel("mixed4b_pre_relu", 452)
obj += L1 * objectives.L1(constant=.5)
obj += TV * objectives.total_variation()
obj += BLUR * objectives.blur_input_each_step()


JITTER = 1
ROTATE = 5
SCALE  = 1.1



# `fft` parameter controls spatial decorrelation
# `decorrelate` parameter controls channel decorrelation
param_f = lambda: param.image(227, fft=DECORRELATE, decorrelate=DECORRELATE)
print(param_f)

if ROBUSTNESS:
    transforms = transform.standard_transforms
else:
    transforms = []



imgs = render.render_vis(model, "InceptionV3/InceptionV3/Mixed_6d/concat:1",
                         optimizer=optimizer,
                         transforms=transforms,
                         param_f=param_f, 
                         thresholds=(100, 256), verbose=False)


# Note that we're doubling the image scale to make artifacts more obvious
for im in imgs: 
    plt.imshow(im[0])
    plt.show()