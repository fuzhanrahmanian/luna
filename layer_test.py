from __future__ import absolute_import, division, print_function, unicode_literals
import timeit
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output

import tensorflow as tf
print(tf.__version__)


# load the model architecture and the pretrained parameters
base_model = tf.keras.applications.InceptionV3(weights='imagenet')
# get the labels file
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', 'https://storage.'
                                      'googleapis.com/download.tensorflow.org/'
                                      'data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())


def VisualizeImageGrayscale(image_3d, percentile=99):
    """Returns a 3D tensor as a grayscale 2D tensor.
    This method sums a 3D tensor across the absolute value of axis=2, and then
    clips values at a given percentile.

    Args:
      image_3d: The image that has to be visualized as greyscale.
      percentile: The percentile that is used to clip the visualized values.
    Returns:
      img: The converted image.
    """
    image_2d = np.sum(np.abs(image_3d), axis=2)

    vmax = np.percentile(image_2d, percentile)
    vmin = np.min(image_2d)

    return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)


def download(url):
    """Downloads an image from a given URL and reads it into a numpy array.

    Args:
    url: Location to fetch the image from.
    Returns:
    img: The image that has been fetched.
    """
    name = url.split('/')[-1]
    image_path = tf.keras.utils.get_file(name, origin=url)
    img = tf.keras.preprocessing.image.load_img(image_path)
    return img


def image_to_tensor(image):
    """Gets a tensor from a numpy image.

    Args:
    image: The image to obtain a tensor from.
    Returns:
    img: The tensor obtained from the image.
    """
    img = tf.keras.preprocessing.image.img_to_array(image)
    img = tf.Variable(tf.keras.applications.inception_v3.preprocess_input(img))
    return img


def show_classification_result(image, prediction, ax):
    """Show the classification result alongside the input image.

    Args:
    image: The image that has been fed into the model.
    prediction: The model prediction for that image.
    ax: The axis position for the image in the plot.
    """
    if ax is None:
        plt.figure()
    plt.axis('off')
    plt.imshow(original_img)
    predicted_class_name = imagenet_labels[prediction+1]
    _ = plt.title("Prediction: " + "Whatever this is...")


def resize_and_extend_heatmap(grad_cam, image_tensor):
    """Resize the heatmap to be the same size as the input and make it
    three-dimensional.

    Args:
    grad_cam: The result of the gradient calculation.
    image_tensor: The image tensor the gradient was calculated for.
    Returns:
    heatmap: The resized heatmap for display.
    """
    heatmap = grad_cam / \
        np.max(grad_cam)  # values need to be [0,1] to be resized
    heatmap = np.squeeze(tf.image.resize(
        np.expand_dims(np.expand_dims(heatmap, 0), 3),
        image_tensor.shape[:2]))
    # Make three-dimensional.
    heatmap = np.expand_dims(heatmap, axis=2)
    heatmap = np.tile(heatmap, [1, 1, 3])
    return heatmap


def show_saliency_result(saliency_map, ax):
    """Show the result of the saliency attribution.

    Args:
    saliency_map: The saliency map that has been obtained from the model.
    ax: The axis position for the image in the plot.
    """
    if ax is None:
        plt.figure()
    plt.axis('off')
    plt.imshow(VisualizeImageGrayscale(saliency_map), cmap=plt.cm.gray)


#url = 'https://upload.wikimedia.org/wikipedia/commons/d/d9/Motorboat_at_Kankaria_lake.JPG'
#url = 'https://cdn.shopify.com/s/files/1/1072/0212/articles/doberman_grande.jpg?v=1503669405'
#url = 'https://farm4.static.flickr.com/3645/3433417334_88c733cacc.jpg'
url = 'https://dogfoodsmart.com/wp-content/uploads/2019/06/Best_Dog_Food_For_German_Shepherds-810x492.jpg'
original_img = download(url).resize((299, 299))


def get_model_gradient(img):
    """Get the gradient from the model prediction with respect to the input image.

    Args:
      img: The input image to get the gradient for.
    Returns:
      gradient: The gradient for the prediction based on the input image.
      predicted_class: The index of the class the model predicted.
    """

    with tf.GradientTape() as tape:  # record the gradients and save them in a tape called 'tape'
        image = img[None, ...]
        ## BEGIN CODE HERE ###
        result = base_model(image)
        probability = np.max(result)
    # get the absolute value of the gradient for the image with respect
    # to the prediction probability
    gradient = tape.gradient(result, image)
    gradient_abs = tf.math.abs(gradient)
    prediction = tf.math.argmax(gradient)
    ### END CODE HERE ###
    predicted_class = prediction.numpy()
    return gradient_abs, predicted_class

def vanilla_gradients(image):
    """Display the vanilla gradient saliency for a given input.

    Args:
        image: The input to display the gradient for.
    """
    image_tensor = image_to_tensor(image)
    gradient, prediction = get_model_gradient(image_tensor)
    plt.figure(figsize=(1 * 15, 2 * 15))
    show_classification_result(image, prediction, ax=plt.subplot(1, 2, 1))
    show_saliency_result(gradient[0], ax=plt.subplot(1, 2, 2))

vanilla_gradients(original_img)  