"""
The utility functions for creating and processing the images
for the feature visualisation process
"""
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras




def initialize_image_luna(w, h=None, batch=None, sd=None, decorrelate=True, fft=True, alpha=False):
    h = h or w
    batch = batch or 1
    channels = 4 if alpha else 3
    shape = [batch, w, h, channels]
    param_f = fft_image if fft else naive
    t = param_f(shape, sd=sd)
    rgb = to_valid_rgb(t[..., :3], decorrelate=decorrelate, sigmoid=True)
    if alpha:
        a = tf.nn.sigmoid(t[..., 3:])
        return tf.concat([rgb, a], -1)
    return rgb


def naive(shape, sd=None):
  return lowres_tensor(shape, shape, sd=sd)


def fft_image(shape, sd=None, decay_power=1):
  b, h, w, ch = shape
  imgs = []
  for _ in range(b):
    freqs = _rfft2d_freqs(h, w)
    fh, fw = freqs.shape
    sd = sd or 0.01
    init_val = sd*np.random.randn(2, ch, fh, fw).astype("float32")
    spectrum_var = tf.Variable(init_val)
    spectrum = tf.complex(spectrum_var[0], spectrum_var[1])
    spertum_scale = 1.0 / np.maximum(freqs, 1.0/max(h, w))**decay_power
    # Scale the spectrum by the square-root of the number of pixels
    # to get a unitary transformation. This allows to use similar
    # leanring rates to pixel-wise optimisation.
    spertum_scale *= np.sqrt(w*h)
    scaled_spectrum = spectrum * spertum_scale
    img = tf.compat.v1.spectral.irfft2d(scaled_spectrum)
    # in case of odd input dimension we cut off the additional pixel
    # we get from irfft2d length computation
    img = img[:ch, :h, :w]
    img = tf.transpose(img, [1, 2, 0])
    imgs.append(img)
  return tf.stack(imgs)/4.


def lowres_tensor(shape, underlying_shape, offset=None, sd=None):
  """Produces a tensor paramaterized by a interpolated lower resolution tensor.

  This is like what is done in a laplacian pyramid, but a bit more general. It
  can be a powerful way to describe images.

  Args:
    shape: desired shape of resulting tensor
    underlying_shape: shape of the tensor being resized into final tensor
    offset: Describes how to offset the interpolated vector (like phase in a
      Fourier transform). If None, apply no offset. If a scalar, apply the same
      offset to each dimension; if a list use each entry for each dimension.
      If a int, offset by that much. If False, do not offset. If True, offset by
      half the ratio between shape and underlying shape (analagous to 90
      degrees).
    sd: Standard deviation of initial tensor variable.

  Returns:
    A tensor paramaterized by a lower resolution tensorflow variable.
  """
  sd = sd or 0.01
  init_val = sd*np.random.randn(*underlying_shape).astype("float32")
  underlying_t = tf.Variable(init_val)
  t = resize_bilinear_nd(underlying_t, shape)
  if offset is not None:
    # Deal with non-list offset
    if not isinstance(offset, list):
      offset = len(shape)*[offset]
    # Deal with the non-int offset entries
    for n in range(len(offset)):
      if offset[n] is True:
        offset[n] = shape[n]/underlying_shape[n]/2
      if offset[n] is False:
        offset[n] = 0
      offset[n] = int(offset[n])
    # Actually apply offset by padding and then croping off the excess.
    padding = [(pad, 0) for pad in offset]
    t = tf.pad(t, padding, "SYMMETRIC")
    begin = len(shape)*[0]
    t = tf.slice(t, begin, shape)
  return t

def _rfft2d_freqs(h, w):
  """Compute 2d spectrum frequences."""
  fy = np.fft.fftfreq(h)[:, None]
  # when we have an odd input dimension we need to keep one additional
  # frequency and later cut off 1 pixel
  if w % 2 == 1:
    fx = np.fft.fftfreq(w)[:w//2+2]
  else:
    fx = np.fft.fftfreq(w)[:w//2+1]
  return np.sqrt(fx*fx + fy*fy)


def resize_bilinear_nd(t, target_shape):
  """Bilinear resizes a tensor t to have shape target_shape.

  This function bilinearly resizes a n-dimensional tensor by iteratively
  applying tf.image.resize_bilinear (which can only resize 2 dimensions).
  For bilinear interpolation, the order in which it is applied does not matter.

  Args:
    t: tensor to be resized
    target_shape: the desired shape of the new tensor.

  Returns:
   The resized tensor
  """
  shape = t.get_shape().as_list()
  target_shape = list(target_shape)
  assert len(shape) == len(target_shape)

  # We progressively move through the shape, resizing dimensions...
  d = 0
  while d < len(shape):

    # If we don't need to deal with the next dimesnion, step over it
    if shape[d] == target_shape[d]:
      d += 1
      continue

    # Otherwise, we'll resize the next two dimensions...
    # If d+2 doesn't need to be resized, this will just be a null op for it
    new_shape = shape[:]
    new_shape[d : d+2] = target_shape[d : d+2]

    # The helper collapse_shape() makes our shapes 4-dimensional with
    # the two dimesnions we want to deal with in the middle.
    shape_ = collapse_shape(shape, d, d+2)
    new_shape_ = collapse_shape(new_shape, d, d+2)

    # We can then reshape and use the 2d tf.image.resize_bilinear() on the
    # inner two dimesions.
    t_ = tf.reshape(t, shape_)
    t_ = tf.image.resize_bilinear(t_, new_shape_[1:3])

    # And then reshape back to our uncollapsed version, having finished resizing
    # two more dimensions in our shape.
    t = tf.reshape(t_, new_shape)
    shape = new_shape
    d += 2

  return t


color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                         [0.27, 0.00, -0.05],
                                         [0.27, -0.09, 0.03]]).astype("float32")
max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))

color_mean = [0.48, 0.46, 0.41]       


def _linear_decorelate_color(t):
  """Multiply input by sqrt of emperical (ImageNet) color correlation matrix.
  
  If you interpret t's innermost dimension as describing colors in a
  decorrelated version of the color space (which is a very natural way to
  describe colors -- see discussion in Feature Visualization article) the way
  to map back to normal colors is multiply the square root of your color
  correlations. 
  """
  # check that inner dimension is 3?
  t_flat = tf.reshape(t, [-1, 3])
  color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
  t_flat = tf.matmul(t_flat, color_correlation_normalized.T)
  t = tf.reshape(t_flat, tf.shape(t))
  return t


def to_valid_rgb(t, decorrelate=False, sigmoid=True):
  """Transform inner dimension of t to valid rgb colors.
  
  In practice this consistes of two parts: 
  (1) If requested, transform the colors from a decorrelated color space to RGB.
  (2) Constrain the color channels to be in [0,1], either using a sigmoid
      function or clipping.
  
  Args:
    t: input tensor, innermost dimension will be interpreted as colors
      and transformed/constrained.
    decorrelate: should the input tensor's colors be interpreted as coming from
      a whitened space or not?
    sigmoid: should the colors be constrained using sigmoid (if True) or
      clipping (if False).
  
  Returns:
    t with the innermost dimension transformed.
  """
  if decorrelate:
    t = _linear_decorelate_color(t)
  if decorrelate and not sigmoid:
    t += color_mean
  if sigmoid:
    return tf.nn.sigmoid(t)
  else:
    return constrain_L_inf(2*t-1)/2 + 0.5














def initialize_image(width, height, val_range_top=1.0, val_range_bottom=-1.0, decorrelate=True, fft=True):
    """
    Creates an initial randomized image to start feature vis process.
    This could be subject to optimization in the future.

    :param width: The width of the image
    :param height: The height of the image

    :return: A randomly generated image
    """
    print('initializing image')
    # We start from a gray image with some random noise
    if tf.compat.v1.keras.backend.image_data_format() == 'channels_first':
        img = tf.random.uniform((1, 3, width, height), dtype=tf.dtypes.float32)
    else:
        img = tf.random.uniform((1, width, height, 3), dtype=tf.dtypes.float32)
    # rescale values to be in the middle quarter of possible values
    img = (img - 0.5) * 0.25 + 0.5
    val_range = val_range_top - val_range_bottom
    # rescale values to be in the given range of values
    img = val_range_bottom + img * val_range
    return img


def deprocess_image(img):
    """
    Takes the values of an image array and normalizes them to be in the
    standard 0-255 RGB range

    :param img: The generated image array

    :return: A rescaled version of the image
    """
    print('Deprocessing image')
    # Normalize array between 0 and 1
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    # Convert to RGB array
    img *= 255
    img = img.astype("uint8")
    return img


def save_image(img, name=None):
    """
    Saves a generated image array as a numpy array in a file
    :param img: The generated image
    :param name: A possible name, if none given it is auto generated
    """
    #if (tf.compat.v1.keras.backend.image_data_format() == "channels_first"):
    #    img = tf.transpose(img, [1, 2, 0])
    arr = keras.preprocessing.image.img_to_array(img)#, data_format="channels_last")
    if name is None:
        name = datetime.now().isoformat()
        name = name.replace("-", "")
        name = name.replace(":", "")
        name = name.replace("+", "")
        name = name.replace(".", "")
    np.save("{0}.npy".format(name), arr)
