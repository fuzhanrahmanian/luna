import argparse
from luna.pretrained_models import models
from luna.featurevis import featurevis, images, image_reader
import tensorflow as tf
import numpy as np
import os

# tf.compat.v1.disable_eager_execution()


parser = argparse.ArgumentParser()

parser.add_argument("-a", "--architecture", type=str,
                    default="inceptionv3", help="The model architecture")
parser.add_argument("-l", "--layerName", type=str, default="mixed6",
                    help="The chosen layer of the model architecture")
parser.add_argument("-c", "--channelNum", type=int, default=7,
                    help="The chosen channel index of the defined model architecture's layer ")
args = parser.parse_args()

arch = args.architecture
layer_name = args.layerName
channel_num = args.channelNum


model = models.model_inceptionv3()
#model = models.model_resnet50v2()
#model = models.model_inceptionv1()
#model = models.model_inceptionv1_slim()
#model = models.model_vgg16()
print(model.get_layer(layer_name).output)
# model.summary()
# model = models.model_alexnet()
#image = images.initialize_image(224, 224)


def image(): return images.initialize_image_luna(227, fft=True, decorrelate=True)


#image_old = images.initialize_image(224, 224)
print(image)
#image = tf.transpose(image(), [0,3, 1,2])


# 'inception_4b/output'
# 'mixed6'
#loss, image
loss3, t_image_2 = featurevis.visualize_filter(image(), model, layer_name,
                                               channel_num, 2600, 0.05, transforms=None)
name = "feature_vis_{}_{}_{}".format(
    arch, layer_name.replace("/", "-"), channel_num)
print(loss3)

images.save_image(t_image_2, name=name)
image_reader.save_npy_as_png("{}.npy".format(name), "luna/outputs")
