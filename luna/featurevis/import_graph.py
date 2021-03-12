import tensorflow as tf


def import_graph(self, t_input=None, scope='import', forget_xy_shape=True):
    """Import model GraphDef into the current graph."""
    if self.graph_def is None:
      raise Exception("Model.import_graph(): Must load graph def before importing it.")
    graph = tf.get_default_graph()
    assert graph.unique_name(scope, False) == scope, (
        'Scope "%s" already exists. Provide explicit scope names when '
        'importing multiple instances of the model.') % scope
    t_input, t_prep_input = self.create_input(t_input, forget_xy_shape)
    tf.import_graph_def(
        self.graph_def, {self.input_name: t_prep_input}, name=scope)
    self.post_import(scope)

def create_input(self, t_input=None, forget_xy_shape=True):
    """Create input tensor."""
    if t_input is None:
      t_input = tf.placeholder(tf.float32, self.image_shape)
    t_prep_input = t_input
    if len(t_prep_input.shape) == 3:
      t_prep_input = tf.expand_dims(t_prep_input, 0)
    if forget_xy_shape:
      t_prep_input = forget_xy(t_prep_input)

    lo, hi = self.image_value_range
    t_prep_input = lo + t_prep_input * (hi - lo)
    return t_input, t_prep_input

def forget_xy(t):
  """Ignore sizes of dimensions (1, 2) of a 4d tensor in shape inference.

  This allows using smaller input sizes, which create an invalid graph at higher
  layers (for example because a spatial dimension becomes smaller than a conv
  filter) when we only use early parts of it.
  """
  shape = (t.shape[0], None, None, t.shape[3])
  return tf.placeholder_with_default(t, shape)