"""
The main file for the feature vis process
"""
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import random
#tf.compat.v1.disable_eager_execution()
from tensorflow import keras
from luna.featurevis import images as imgs
from luna.featurevis import transform
#from luna.featurevis import relu_grad as rg
#tf.config.run_functions_eagerly(True)
import tensorflow_addons as tfa
from luna.featurevis import relu_grad as rg


def compose(transforms):
    def inner(x):
        for transform in transforms:
            x = transform(x)
        return x

    return inner



def _rand_select(xs, seed=None):
    xs_list = list(xs)
    rand_n = tf.random.uniform((), 0, len(xs_list), "int32", seed=seed)
    return tf.constant(xs_list)[rand_n]


def _angle2rads(angle, units):
    angle = tf.cast(angle, "float32")
    if units.lower() == "degrees":
        angle = 3.14 * angle / 180.
    elif units.lower() in ["radians", "rads", "rad"]:
        angle = angle
    return angle


def crop_or_pad_to(height, width):
    """Ensures the specified spatial shape by either padding or cropping.
    Meant to be used as a last transform for architectures insisting on a specific
    spatial shape of their inputs.
    """
    def inner(t_image):
        return tf.image.resize_with_crop_or_pad(t_image, height, width)
    return inner

def add_noise(img, noise, pctg):
    """Adds noise to the image to be manipulated.

    Args:
        img (list): the image data to which noise should be added
        noise (boolean): whether noise should be added at all
        pctg (number): how much noise should be added to the image
        channels_first (bool, optional): whether the image is encoded channels
        first. Defaults to False.

    Returns:
        list: the modified image
    """
    if noise:
        if tf.compat.v1.keras.backend.image_data_format() == "channels_first":
            img_noise = tf.random.uniform((img.shape),
                                          dtype=tf.dtypes.float32)
        else:
            img_noise = tf.random.uniform((img.shape),
                                          dtype=tf.dtypes.float32)
        img_noise = (img_noise - 0.5) * 0.25 * ((100 - pctg) / 100)
        img = img + img_noise
        img = tf.clip_by_value(img, -1, 1)
    return img

def jitter(image, d, seed=None):

    image = tf.convert_to_tensor(value=image, dtype_hint=tf.float32)
    t_shp = tf.shape(input=image)
    crop_shape = tf.concat([t_shp[:-3], t_shp[-3:-1] - d, t_shp[-1:]], 0)
    crop = tf.image.random_crop(image, crop_shape, seed=seed)
    shp = image.get_shape().as_list()
    mid_shp_changed = [
        shp[-3] - d if shp[-3] is not None else None,
        shp[-2] - d if shp[-3] is not None else None,
    ]
    crop.set_shape(shp[:-3] + mid_shp_changed + shp[-1:])
    return crop

def random_scale(scales, seed=None):
    def inner(t):
        t = tf.convert_to_tensor(value=t, dtype_hint=tf.float32)
        scale = _rand_select(scales, seed=seed)
        shp = tf.shape(input=t)
        scale_shape = tf.cast(scale * tf.cast(shp[-3:-1], "float32"), "int32")
        return tf.image.resize(t, scale_shape, method=tf.image.ResizeMethod.BILINEAR)

    return inner


def random_rotate(image, angles, units="degrees", seed=None):
        t = tf.convert_to_tensor(value=t, dtype_hint=tf.float32)
        angle = _rand_select(angles, seed=seed)
        angle = _angle2rads(angle, units)

    #return tf.image.rotate(t, angle)


def pad(img, w, mode="REFLECT", constant_value=0.5):

    if constant_value == "uniform":
        constant_value_ = tf.random.uniform([], 0, 1)
    else:
        constant_value_ = constant_value
    return tf.pad(img, paddings=[(0, 0), (w, w), (w, w), (0, 0)], mode=mode, constant_values=constant_value_)


def blur_image(img, blur, pctg):
    """Gaussian blur the image to be modified.

    Args:
        img (list): the image to be blurred
        blur (boolean): whether to blur the image
        pctg (number): how much blur should be applied

    Returns:
        list: the blurred image
    """
    if blur:
        img = gaussian_blur(img, sigma=0.001 + ((100-pctg) / 100) * 1)
        img = tf.clip_by_value(img, -1, 1)
    return img


def rescale_image(img, scale, pctg):
    """
    Will rescale the current state of the image

    :param img: the current state of the feature vis image
    :param scale: true, if image should be randomly scaled
    :param pctg: the amount of scaling in percentage
    :return: the altered image
    """
    if scale:
        scale_factor = [1, 0.975, 1.025, 0.95, 1.05]
        factor = random.choice(scale_factor)
        #scale_factor = tf.random.normal([1], 1, pctg)
        #img *= scale_factor[0].numpy()  # working
        img = img * factor
    return img


# https://gist.github.com/blzq/c87d42f45a8c5a53f5b393e27b1f5319
def gaussian_blur(img, kernel_size=3, sigma=5):
    """
    helper function for blurring the image, will soon be replaced by
    tfa.image.gaussian_filter2d

    :param img: the current state of the image
    :param kernel_size: size of the convolution kernel used for blurring
    :param sigma: gaussian blurring constant
    :return: the altered image
    """
    def gauss_kernel(channels, kernel_size, sigma):
        """
        Calculates the gaussian convolution kernel for the blurring process

        :param channels: amount of feature channels
        :param kernel_size: size of the kernel
        :param sigma: gaussian blurring constant
        :return: the kernel for the given values
        """
        axis = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xvals, yvals = tf.meshgrid(axis, axis)
        kernel = tf.exp(-(xvals ** 2 + yvals ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel

    gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
    gaussian_kernel = gaussian_kernel[..., tf.newaxis]

    return tf.nn.depthwise_conv2d(img, gaussian_kernel, [1, 1, 1, 1],
                                  padding='SAME', data_format='NHWC')

def make_transform_f(transforms):
    if type(transforms) is not list:
        transforms = transform.standard_transforms
    transform_f = transform.compose(transforms)
    return transform_f


def visualize_filter(image, model, layer, filter_index, iterations,
                     learning_rate, noise, blur, scale):
    """Create a feature visualization for a filter in a layer of the model.

    Args:
        image (array): the image to be modified by the feature vis process
        model (object): the model to be used for the feature visualization
        layer (string): the name of the layer to be used in the visualization
        filter_index (number): the index of the filter to be visualized
        iterations (number): hoe many iterations to optimize for
        learning_rate (number): update amount after each iteration
        noise (number): how much noise to add to the image
        blur (number): how much blur to add to the image
        scale (number): how much to scale the image

    Returns:
        tuple: loss and result image for the process
    """
    image = tf.Variable(image)
    #transform_f = make_transform_f(transforms)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    feature_extractor = get_feature_extractor(model, layer)
    choice_num = [0, 1, 2, 3]
    augmentation = ['noise', 'blur', 'scale']
    print('Starting Feature Vis Process')
    for iteration in range(iterations):
        pctg = int(iteration / iterations * 100)
        image_aug = {'noise':add_noise(image, noise, pctg), 'blur': blur_image(image, blur, pctg), 'scale':rescale_image(image, scale, pctg)}
        num = random.choice(choice_num)
        
        image = jitter(image, random.choice([i for i in range(4)]))
        if num ==1:
            ind = random.sample(augmentation, 1)
            print(ind)
            image = image_aug[ind[0]]
        if num ==2:
            ind = random.sample(augmentation, 2)
            print(ind)
            image = image_aug[ind[0]]
            image = image_aug[ind[1]]
        else:
            image = add_noise(image, noise, pctg)
            image = blur_image(image, blur, pctg)
            image = rescale_image(image, scale, pctg)
        #print(image.shape)
        #print(image.shape[1])
        #pad_size = random.choice([i for i in range(16)])
        #image = tf.image.resize_with_pad(image, image.shape[1]+pad_size, image.shape[1]+pad_size)
        #crop_size = random.choice([i for i in range(16)])
        #image = tf.image.crop_and_resize(image, tf.random.uniform(shape=(5, 4)), tf.random.uniform(shape=(5,), minval=0, maxval=1, dtype=tf.int32), (crop_size, crop_size))
        #image = tfa.image.rotate(image, random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]))

        
        #image = add_noise(image, noise, pctg)
        #image = blur_image(image, blur, pctg)
        #image = rescale_image(image, scale, pctg)
        #img = transform_f(img)
        #print(type(image))
        #image = transform_f(image)
        
        #image = tf.Variable(image)
        #print(f"type of image after transformation is {type(image)}")
        loss, image = gradient_ascent_step(
            image, feature_extractor, filter_index, learning_rate, optimizer)

        print('>>', pctg, '%', end="\r", flush=True)
    print(f"loss is {loss} and image is {image}")
    print('>> 100 %')
    # Decode the resulting input image
    image = imgs.deprocess_image(image[0].numpy())
    return loss, image

#@tf.function
def compute_loss(input_image, model, filter_index):
    """Computes the loss for the feature visualization process.

    Args:
        input_image (array): the image that is used to compute the loss
        model (object): the model on which to compute the loss
        filter_index (number): for which filter to compute the loss
        channels_first (bool, optional): Whether the image is channels first.
        Defaults to False.

    Returns:
        number: the loss for the specified setting
    """
    with rg.gradient_override_map({'Relu': rg.redirected_relu_grad,'Relu6': rg.redirected_relu6_grad}):
        activation = model(input_image)
    #activation = model(input_image)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    if tf.compat.v1.keras.backend.image_data_format() == "channels_first":
        filter_activation = activation[:, filter_index, :, :]
    else:
        filter_activation = activation[:, :, :, filter_index]
    #return tf.reduce_mean(tf.square(filter_activation, input_image))
    return tf.reduce_mean(filter_activation)


#@tf.function
def gradient_ascent_step(img, model, filter_index, learning_rate, optimizer):
    """Performing one step of gradient ascend.

    Args:
        img (array): the image to be changed by the gradiend ascend
        model (object): the model with which to perform the image change
        filter_index (number): which filter to optimize for
        learning_rate (number): how much to change the image per iteration
        channels_first (bool, optional): Whether the image is channels first.
        Defaults to False.

    Returns:
        tuple: the loss and the modified image
    """

    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, model, filter_index)
    # Compute gradients.
    #optim = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    #loss = compute_loss(img, model, filter_index)
    #print(f"type of loss is {type(loss)}")
    #loss = lambda : compute_loss(img, model, filter_index)
    #loss = loss(img, model, filter_index)
    #optim.minimize(-loss, var_list=[img])

    #tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss, var_list=[img])
    #optimizer.minimize(loss, var_list=[img])
    #img.numpy()
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    #optimizer.apply_gradients(zip([grads], [img]))
    img += learning_rate * grads
    return loss, img


def get_feature_extractor(model, layer_name):
    """Builds a model that that returns the activation of the specified layer.

    Args:
        model (object): the model used as a basis for the feature extractor
        layer (string): the layer at which to cap the original model
    """
    layer = model.get_layer(name=layer_name)
    return keras.Model(inputs=model.inputs, outputs=layer.output)
