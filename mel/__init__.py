import numpy as np


import tensorflow
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_util


try:
  import torch
except ModuleNotFoundError:
  torch = None


def is_lib_type(value, libname):
  name = type(value).__module__
  if name == libname:
    return True
  if '.' in name:
    return name.split('.')[0] == libname
  return False


def is_tf_type(value):
  return is_lib_type(value, 'tensorflow')
  # if tensorflow.is_tensor(value):
  #   return True
  # if isinstance(value, tensorflow.Variable):
  #   return True
  # if isinstance(value, tensorflow.TensorShape):
  #   return True
  # if isinstance(value, tensorflow.Dimension):
  #   return True
  # return False


def is_torch_type(value):
  return is_lib_type(value, 'torch')
  # if torch is None:
  #   return False
  # if torch.is_tensor(value):
  #   return True
  # return False


def is_np_type(value):
  return type(value).__module__ == 'numpy'
  # for k, v in np.typeDict.items():
  #   if isinstance(value, v):
  #     return True
  # return False


def pynum(value):
  return isinstance(value, float) or isinstance(value, int)


def pybool(value):
  return value is True or value is False


def pystr(value):
  return isinstance(value, str) or isinstance(value, bytes)


def pynil(value):
  return value is None


def pyatom(value):
  return pynum(value) or pybool(value) or pystr(value) or pynil(value) or is_np_type(value)


def pylist(value):
  return isinstance(value, list)


def pydict(value):
  return isinstance(value, dict)


def pyobj(value):
  return pylist(value) or pydict(value)


def is_np(value, strict=False, deep=True):
  if isinstance(value, np.ndarray):
    return True
  if not strict and pyatom(value):
    return True
  if deep and pylist(value):
    return all([is_np(x, strict=strict, deep=deep) for x in value])
  if deep and pydict(value):
    return all([is_np(v, strict=strict, deep=deep) for k, v in value.items()])
  return False


def is_tf(value, strict=False, deep=True):
  if tensorflow.is_tensor(value):
    return True
  if isinstance(value, tensorflow.Variable):
    return True
  if isinstance(value, tensorflow.TensorShape):
    return True
  if isinstance(value, tensorflow.Dimension):
    return True
  if not strict and pyatom(value):
    return True
  if deep and pylist(value):
    return all([is_tf(x, strict=strict, deep=deep) for x in value])
  if deep and pydict(value):
    return all([is_tf(v, strict=strict, deep=deep) for k, v in value.items()])
  return False


def is_tf_variable(value):
  if isinstance(value, tensorflow.Variable):
    return True
  return False


def is_tf_tensor(value):
  if tensorflow.is_tensor(value):
    return True
  return False


def is_torch_tensor(value):
  if torch is None:
    return False
  if torch.is_tensor(value):
    return True
  return False


def is_tf_constant(value):
  try:
    return constant_op.is_constant(value)
  except AttributeError:
    # TODO: is there a cleaner way to accomplish this?
    return False


def val(value, eager=False, session=None, deep=True):
  if isinstance(value, tensorflow.TensorShape):
    value = value.as_list()
  if isinstance(value, tensorflow.Dimension):
    value = value.value
  if torch:
    if isinstance(value, torch.Size):
      value = list(value)
  #if is_tf(value, strict=True) and not pyobj(value) and constant_op.is_constant(value):
  if is_tf_constant(value):
    value = tensor_util.constant_value(value)
  if deep and pylist(value):
    value = [val(x, eager=eager, session=session, deep=deep) for x in value]
  if deep and pydict(value):
    value = {k: val(v, eager=eager, session=session, deep=deep) for k, v in value.items()}
  if is_tf_variable(value) or is_tf_tensor(value):
    if not eager:
      raise ValueError("val called on tensorflow variable {} but eager is False".format(value))
    # TODO: batch multiple nested reads.
    result = value.eval(session=session)
    return val(result, eager=eager, session=session, deep=deep)
  if is_torch_tensor(value):
    if not eager:
      raise ValueError("val called on torch tensor {} but eager is False".format(value))
    result = value.numpy()
    return val(result, eager=eager, session=session, deep=deep)
  if not is_np(value):
    raise ValueError("val called with value of unknown type: {}".format(value))
  return value


import importlib


def as_lib(lib=None, hint=None):
  if isinstance(lib, str):
    if lib == 'tf':
      lib = 'tensorflow'
    if lib == 'np':
      lib = 'numpy'
    if isinstance(lib, str):
      if lib == 'tensorflow':
        lib = tensorflow
      elif lib == 'numpy':
        lib = np
      elif lib == 'torch':
        lib = torch
      else:
        lib = importlib.__import__(lib)
      return lib
  if lib is None:
    if is_tf_type(hint):
      return tensorflow
    if is_torch_type(hint):
      return torch
    return np
  assert lib is tensorflow or lib is np or lib is torch
  return lib


dtype_remap = {
    'bool': 'b',

    'f': 'f32',
    'float': 'f32',
    'float32': 'f32',

    'd': 'f64',
    'double': 'f64',
    'float64': 'f64',

    'c': 'f128',
    'complex': 'f128',
    'complex128': 'f128',

    'byte': 'u8',
    'uint8': 'u8',
    'ushort': 'u16',
    'uint16': 'u16',
    'uint': 'u32',
    'ulong': 'u64',
    'ulong64': 'u64',

    'u': 'u32',
    'unsigned': 'u32',
    'uint32': 'u32',

    'int8': 'i8',

    'short': 'i16',
    'int16': 'i16',

    'i': 'i32',
    'signed': 'i32',
    'int': 'i32',
    'int32': 'i32',

    'l': 'i64',
    'long': 'i64',
    'long64': 'i64',
    'resource': 'r',
    'variant': 'v',
}


for v in list(dtype_remap.values()):
  dtype_remap[v] = v


dtype_long = {
    'b': 'bool',
    'f32': 'float32',
    'f64': 'float64',
    'f128': 'complex',
    'u8': 'uint8',
    'u16': 'uint16',
    'u32': 'uint32',
    'u64': 'uint64',
    'i8': 'int8',
    'i16': 'int16',
    'i32': 'int32',
    'i64': 'int64',
    'r': 'resource',
    'v': 'variant',
}


for v in list(dtype_remap.values()):
  assert v in dtype_long


def as_dtype(value, lib=None, deep=True):
  lib = as_lib(lib, hint=value)
  if isinstance(value, str):
    if value not in dtype_remap:
      raise ValueError("Unknown dtype spec: {}".format(value))
    value = dtype_remap[value]
    assert value in dtype_long
    value = dtype_long[value]
    if not hasattr(lib, value):
      raise ValueError("Library {} does not support dtype {}".format(lib, value))
    value = getattr(lib, value)
  if hasattr(value, 'dtype'):
    value = value.dtype
  if deep and pylist(value):
    value = [as_dtype(x, lib=lib, deep=deep) for x in value]
  if deep and pydict(value):
    value = {k: as_dtype(v, lib=lib, deep=deep) for k, v in value.items()}
  return value


from . import tf


# from . import _version
# __version__ = _version.__version__


# Version info -- read without importing
import os
_locals = {}
__dir__ = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(__dir__, "_version.py")) as fp:
  exec(fp.read(), None, _locals)
__version__ = _locals["__version__"]
