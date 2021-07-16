import argparse
from luna.pretrained_models import models
from luna.featurevis import featurevis, images, image_reader
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os.path as path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

#tf.compat.v1.disable_eager_execution()
# resnet 50 v2
#block5_conv4 vgg19
parser = argparse.ArgumentParser()

parser.add_argument("-a", "--architecture", type=str,
                    default="inceptionV3", help="The model architecture")
parser.add_argument("-l", "--layerName", type=str, default="mixed5", #inception_4a/output- conv2d_12
                    help="The chosen layer of the model architecture") #mixed4
parser.add_argument("-c", "--channelNum", type=int, default=751,
                    help="The chosen channel index of the defined model architecture's layer ")
args = parser.parse_args()

arch = args.architecture
#layer_name = args.layerName
#channel_num = args.channelNum
#output_folder = "luna/outputs/cifar10"

#model = models.model_cifar10()
model = models.model_inceptionv3()
#model = models.model_resnet50v2()
#model = models.model_inceptionv1()
#model = models.model_inceptionv1_slim()
#model = models.model_vgg16()
#model = models.model_vgg19()
#model = models.model_alexnet()
# model = tf.keras.applications.efficientnet.EfficientNetB0(
#     include_top=True, weights='imagenet', input_tensor=None,
#     input_shape=None, pooling=None, classes=1000,
#     classifier_activation='softmax'
# )
#model = models.model_mobilenetV2()
#model = models.model_efficientnet()



model.summary()    #in the case that we dont know the name of the layers.


#print(model.get_layer('conv2d_12').output.shape[3])

# def image(): return images.initialize_image_luna(227, fft=True, decorrelate=True)

# image_old = images.initialize_image(224, 224)

# image = tf.transpose(image(), [0,3, 1,2])


#'inception_4b/output'
#'mixed6's
#for i in range(10):

#i = 1
#for a in range(110,140):

output_folder = r"C:\Users\lucaz\Documents\Fuzhi\GitHub\luna\luna\outputs\inceptionv3"
#def image(): return images.initialize_image_luna(299, fft=True, decorrelate=True)
#image = images.initialize_image_ref(32,32, decorrelate=True)
layer_name_comp_inception_v3 = ["mixed5"]
layer_name_cifar10 = ["conv2d_3", "conv2d_6", "conv2d_8", "conv2d_12"]
layer_name_mobilenet = ["block_7_expand", "block_10_expand", "block_14_expand", "block_16_expand"]
layer_name_efficientnet = ["block7b_se_expand", "block6b_se_expand", "block5c_se_expand", "block5b_se_expand"]
channel_num_comp = [30]
model_name = "efficientnet"
for layer_name, channel_num in zip(layer_name_comp_inception_v3, channel_num_comp):
    image = images.initialize_image(224, 224)
    print(image)
    opt_param = featurevis.OptimizationParameters(3500, 0.65)
    aug_param = featurevis.AugmentationParameters(blur= True, scale= True, pad_crop=False, flip=False,
                                                    rotation=False, noise=False, color_aug=False)
    #print(image)
    loss, image, loss_record, activation = featurevis.visualize_filter(image, model, layer_name, channel_num, opt_param, aug_param)
    #name = "feature_vis_{}_{}_{}".format(
    #    arch, layer_name.replace("/", "-"), channel_num)
    name = "feature_vis_{}_{}_{}".format(arch, layer_name, channel_num)
    print(loss)
    #print(loss_record)
    print(activation)
    #tf.Tensor(7.1122437, shape=(), dtype=float32)
    plt.plot(loss_record)
    #plt.ylim((-15, 0))
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.savefig("{}/loss_{}_{}_{}.png".format(output_folder, model_name, layer_name, channel_num))
    plt.close()
    images.save_image(image, name=name)
    image_reader.save_npy_as_png("{}.npy".format(name), output_folder)

# layer_name_cifar = ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'conv2d_5', 'conv2d_6',
#                     'conv2d_7', 'conv2d_8', 'conv2d_9', 'conv2d_10', 'conv2d_11', 'conv2d_12']

# channel_num_cifar = [model.get_layer(i).output.shape[3] for i in layer_name_cifar]

# for k in tqdm([12]): #range(12)
#     output_folder = "C:/Users/lucaz/OneDrive/Fuzhi/Uni Ulm/luna/cifar10/{}".format(layer_name_cifar[k])
#     loss_dict = dict()

#     if path.exists(output_folder):
#         for t in range(channel_num_cifar[k]):
#             #def image(): return images.initialize_image_luna(299, fft=True, decorrelate=True)
#             #image = images.initialize_image_ref(32,32, decorrelate=True)
#             image = images.initialize_image(32,32)
#             print(image)
#             opt_param = featurevis.OptimizationParameters(3500, 0.7)
#             aug_param = featurevis.AugmentationParameters(blur= True, scale= True, pad_crop=False, flip=False,
#                                                             rotation=False, noise=False, color_aug=False)
#             #print(image)
#             loss, image = featurevis.visualize_filter(image, model, layer_name_cifar[k], t, opt_param, aug_param)
#             #name = "feature_vis_{}_{}_{}".format(
#             #    arch, layer_name.replace("/", "-"), channel_num)
#             name = "feature_vis_{}_{}_{}".format(arch, layer_name_cifar[k], t)
#             print(loss)
#             loss_dict.update({"{}_{}".format(layer_name_cifar[k], t): str(loss)})
#             print(loss_dict)


#             images.save_image(image, name=name)
#             image_reader.save_npy_as_png("{}.npy".format(name), output_folder)
#         with open('C:/Users/lucaz/OneDrive/Fuzhi/Uni Ulm/luna/cifar10/{}/loss.json'.format(layer_name_cifar[k]), 'w') as f:
#             json.dump(loss_dict, f)


