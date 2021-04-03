"""
The main file for the feature vis process
"""
from __future__ import absolute_import, division, print_function
import tensorflow as tf

from tensorflow import keras
from luna.featurevis import images as imgs
from luna.featurevis import objectives_luna as objectives
from luna.featurevis import transform_luna_v2 as transform

import tqdm
from tqdm import tqdm


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
            img_noise = tf.random.uniform((1, 3, len(img[2]), len(img[3])),
                                          dtype=tf.dtypes.float32)
        else:
            img_noise = tf.random.uniform((1, len(img[1]), len(img[2]), 3),
                                          dtype=tf.dtypes.float32)
        img_noise = (img_noise - 0.5) * 0.25 * ((100 - pctg) / 100)
        img = img + img_noise
        img = tf.clip_by_value(img, -1, 1)
    return img


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
        scale_factor = tf.random.normal([1], 1, pctg)
        img *= scale_factor[0]  # not working
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


def visualize_filter(param_f, model, layer, filter_index, iterations,
                     learning_rate, noise, blur, scale, transforms):
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

    feature_extractor = get_feature_extractor(model, layer)

    loss3, t_image_2 = make_vis_T(model, layer, filter_index,
                                  learning_rate, param_f, transforms, iterations)
    #loss, vis_op, t_image = T("loss"), T("vis_op"), T("input")

    print(t_image_2)
    print(t_image_2[0].numpy())
    t_image_2 = t_image_2[0].numpy()

    return loss3, t_image_2


def make_t_image(param_f):
    if param_f is None:
        t_image = imgs.initialize_image_luna(128)
    elif callable(param_f):
        t_image = param_f()
    elif isinstance(param_f, tf.Tensor):
        t_image = param_f
    else:
        raise TypeError("Incompatible type for param_f, " + str(type(param_f)))

    if not isinstance(t_image, tf.Tensor):
        raise TypeError("param_f should produce a Tensor, but instead created a "
                        + str(type(t_image)))
    else:
        return t_image


def make_vis_T(model, layer, filter_index, learning_rate, param_f=None, transforms=None, epc_num=500):
    """Even more flexible optimization-base feature vis.

  This function is the inner core of render_vis(), and can be used
  when render_vis() isn't flexible enough. Unfortunately, it's a bit more
  tedious to use:

  >  with tf.Graph().as_default() as graph, tf.Session() as sess:
  >
  >    T = make_vis_T(model, "mixed4a_pre_relu:0")
  >    tf.initialize_all_variables().run()
  >
  >    for i in range(10):
  >      T("vis_op").run()
  >      showarray(T("input").eval()[0])

  This approach allows more control over how the visualizaiton is displayed
  as it renders. It also allows a lot more flexibility in constructing
  objectives / params because the session is already in scope.


  Args:
    model: The model to be visualized, from Alex' modelzoo.
    objective_f: The objective our visualization maximizes.
      See the objectives module for more details.
    param_f: Paramaterization of the image we're optimizing.
      See the paramaterization module for more details.
      Defaults to a naively paramaterized [1, 128, 128, 3] image.
    optimizer: Optimizer to optimize with. Either tf.train.Optimizer instance,
      or a function from (graph, sess) to such an instance.
      Defaults to Adam with lr .05.
    transforms: A list of stochastic transformations that get composed,
      which our visualization should robustly activate the network against.
      See the transform module for more details.
      Defaults to [transform.jitter(8)].

  Returns:
    A function T, which allows access to:
      * T("vis_op") -- the operation for to optimize the visualization
      * T("input") -- the visualization itself
      * T("loss") -- the loss for the visualization
      * T(layer) -- any layer inside the network
  """

    t_image = make_t_image(param_f)

    t_image_2 = tf.Variable(make_t_image(param_f))
    objective_f = objectives.as_objective("{}:{}".format(layer, filter_index))
    transform_f = make_transform_f(transforms)
    #t_image = transform_f(t_image)
    optimizer = make_optimizer(
        tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate), [])
    optimizer_2 = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    global_step = tf.compat.v1.train.get_or_create_global_step()
    T = import_model(model, transform_f(t_image), t_image)

    feature_extractor = get_feature_extractor(model, layer)
    for i in tqdm(range(epc_num)):
        pctg = int(i/epc_num*100)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t_image)
            tape.watch(t_image_2)
            #T1 = model(t_image)
            #T2 = model(t_image_2)
            T3 = model(transform_f(t_image_2))
            #loss = -objective_f(T1)
            #loss_2 = -objective_f(T2)

            #activation1 = feature_extractor(t_image_2)
            #activation2 = feature_extractor(t_image_2)
            # We avoid border artifacts by only involving non-border pixels in the loss.
            # if tf.compat.v1.keras.backend.image_data_format() == "channels_first":
            #    filter_activation = activation1[:, filter_index, 2:-2, 2:-2]
            # else:
            #    filter_activation = activation1[:, 2:-2, 2:-2, filter_index]
            #loss3 = tf.reduce_mean(activation1)
            #loss4 = tf.reduce_mean(activation2)
            loss3 = -objective_f(T3)
        #gradients2 = tape.gradient(loss_2, [t_image_2])
        gradients3 = tape.gradient(loss3, [t_image_2])
        #gradients4 = tape.gradient(loss4, [t_image_2])

        #optimizer_2.apply_gradients(zip(gradients, [t_image]))
        #vis_opt_1 = optimizer_2.apply_gradients(zip(gradients2, [t_image_2]))
        #optimizer_2.apply_gradients(zip(gradients3, [t_image]))
        #vis_opt_2 = optimizer_2.apply_gradients(zip(gradients4, [t_image_2]))
        optimizer_2.apply_gradients(zip(gradients3, [t_image_2]))
        print('>>', pctg, '%', end="\r", flush=True)
    return loss3, t_image_2


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
    activation = model(input_image)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    if tf.compat.v1.keras.backend.image_data_format() == "channels_first":
        filter_activation = activation[:, filter_index, 2:-2, 2:-2]
    else:
        filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)


@tf.function
def gradient_ascent_step(img, model, filter_index, learning_rate):
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
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
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


def make_transform_f(transforms):
    if type(transforms) is not list:
        transforms = transform.standard_transforms
    transform_f = transform.compose(transforms)
    return transform_f


def import_model(model, t_image, t_image_raw):

    #model.import_graph(t_image, scope="import", forget_xy_shape=True)

    def T(layer):
        if layer == "input":
            return t_image_raw
        if layer == "labels":
            return model.labels
        return model.get_layer(layer).output
        # return t_image.graph.get_tensor_by_name("import/%s:0"%layer)

    return T


def make_optimizer(optimizer, args):
    if optimizer is None:
        return tf.keras.optimizers.Adam(0.05)
    elif callable(optimizer):
        return optimizer(*args)
    elif isinstance(optimizer, tf.compat.v1.train.Optimizer):
        return optimizer
    else:
        print("Could not convert optimizer argument to usable optimizer. "
              "Needs to be one of None, function from (graph, sess) to "
              "optimizer, or tf.train.Optimizer instance.")


def import_graph(self, t_input=None, scope='import', forget_xy_shape=True):
    """Import model GraphDef into the current graph."""
    if self.graph_def is None:
        raise Exception(
            "Model.import_graph(): Must load graph def before importing it.")
    graph = tf.get_default_graph()
    assert graph.unique_name(scope, False) == scope, (
        'Scope "%s" already exists. Provide explicit scope names when '
        'importing multiple instances of the model.') % scope
    t_input, t_prep_input = self.create_input(t_input, forget_xy_shape)
    tf.import_graph_def(
        self.graph_def, {self.input_name: t_prep_input}, name=scope)
    self.post_import(scope)
