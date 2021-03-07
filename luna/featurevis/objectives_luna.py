#from __future__ import absolute_import, division, print_function

from decorator import decorator
import tensorflow as tf

class Objective(object):
    """"A wrapper to make objective functions easy to combine.

    For example, suppose you want to optimize 20% for mixed4a:20 and 80% for
    mixed4a:21. Then you could use:

        objetive = 0.2 * channel("mixed4a", 20) + 0.8 * channel("mixed4a", 21)

    Under the hood, we think of objectives as functions of the form:

        T => tensorflow scalar for loss

    where T is a function allowing you to index layers in the network -- that is,
    if there's a layer "mixed4a" then T("mixed4a") would give you its
    activations).

    This allows objectives to be declared outside the rendering function, but then
    actually constructed within its graph/session.
    """

    def __init__(self, objective_func, name="", description=""):
        self.objective_func = objective_func
        self.name = name
        self.description = description

    def __add__(self, other):
        if isinstance(other, (int, float)):
            objective_func = lambda T: other + self(T)
            name = self.name
            description = self.description
        else:
            objective_func = lambda T: self(T) + other(T)
            name = ", ".join([self.name, other.name])
            description = "Sum(" + " +\n".join([self.description, other.description]) + ")"
        return Objective(objective_func, name=name, description=description)

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        return self + (-1 * other)

    @staticmethod
    def sum(objs):
        objective_func = lambda T: sum([obj(T) for obj in objs])
        descriptions = [obj.description for obj in objs]
        description = "Sum(" + " +\n".join(descriptions) + ")"
        names = [obj.name for obj in objs]
        name = ", ".join(names)
        return Objective(objective_func, name=name, description=description)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            objective_func = lambda T: other * self(T)
        else:
            objective_func = lambda T: self(T) * other(T)
        return Objective(objective_func, name=self.name, description=self.description)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __call__(self, T):
        return self.objective_func(T)


def _make_arg_str(arg):
    arg = str(arg)
    too_big = len(arg) > 15 or "\n" in arg
    return "..." if too_big else arg


@decorator
def wrap_objective(f, *args, **kwds):
    """Decorator for creating Objective factories.

    Changes f from the closure: (args) => () => TF Tensor
    into an Obejective factory: (args) => Objective

    while perserving function name, arg info, docs... for interactive python.
    """
    objective_func = f(*args, **kwds)
    objective_name = f.__name__
    args_str = " [" + ", ".join([_make_arg_str(arg) for arg in args]) + "]"
    description = objective_name.title() + args_str
    return Objective(objective_func, objective_name, description)

@wrap_objective
def channel(layer, n_channel, batch=None):
    """Visualize a single channel"""
    if batch is None:
        return lambda T: tf.reduce_mean(T(layer)[..., n_channel])
    else:
        return lambda T: tf.reduce_mean(T(layer)[batch, ..., n_channel])


def as_objective(obj):
    """Convert obj into Objective class.

    Strings of the form "layer:n" become the Objective channel(layer, n).
    Objectives are returned unchanged.

    Args:
        obj: string or Objective.

    Returns:
        Objective
    """
    if isinstance(obj, Objective):
        return obj
    elif callable(obj):
        return obj
    elif isinstance(obj, str):
        layer, n = obj.split(":")
        layer, n = layer.strip(), int(n)
        return channel(layer, n)